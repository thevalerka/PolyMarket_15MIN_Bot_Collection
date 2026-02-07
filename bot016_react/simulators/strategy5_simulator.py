import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class Trade:
    """Represents a single trade"""
    timestamp: str
    option_type: str  # 'CALL' or 'PUT'
    action: str  # 'BUY' or 'SELL'
    price: float
    reason: str  # 'OPEN', 'DOUBLE_DOWN', 'TP', 'SL', 'EXPIRY'
    pnl: float = 0.0
    expiry_time: str = None
    open_price: float = None  # For SELL trades
    close_price: float = None  # For SELL trades
    seconds_to_expiry: float = None  # For SELL trades
    quantity: int = 1  # Number of contracts

@dataclass
class Position:
    """Represents an open position"""
    option_type: str
    entry_price: float
    entry_time: str
    expiry_time: str
    take_profit: float
    stop_loss: float
    strategy: str

class Strategy5Simulator:
    """Simulates Strategy 5 variations (5A, 5B, 5C)"""
    
    def __init__(self, call_file: str, put_file: str, base_output_dir: str = "/home/ubuntu/013_2025_polymarket/bot016_react/simulators"):
        self.call_file = Path(call_file)
        self.put_file = Path(put_file)
        self.base_output_dir = Path(base_output_dir)
        
        # Create separate output directories for each strategy
        self.output_dirs = {
            'strategy5A': self.base_output_dir / "strategy5A_results",
            'strategy5B': self.base_output_dir / "strategy5B_results",
            'strategy5C': self.base_output_dir / "strategy5C_results"
        }
        
        for output_dir in self.output_dirs.values():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track last modified times
        self.last_call_mtime = 0
        self.last_put_mtime = 0
        
        # Current prices
        self.current_call_ask = None
        self.current_call_bid = None
        self.current_put_ask = None
        self.current_put_bid = None
        
        # Last valid prices
        self.last_valid_call_ask = None
        self.last_valid_call_bid = None
        self.last_valid_put_ask = None
        self.last_valid_put_bid = None
        
        # Strategies configuration
        self.strategies = {
            'strategy5A': {
                'tp': 0.15,
                'sl': None,
                'max_buy_price': 0.99,
                'min_entry_price': 0.60,
                'description': 'Enter when one side >$0.60, TP=$0.15, No SL'
            },
            'strategy5B': {
                'tp': 0.15,
                'sl': None,
                'max_buy_price': 0.99,
                'min_entry_price': 0.60,
                'double_down_price': 0.20,
                'max_positions_per_side': 2,
                'description': 'Like 5A, double down when ≤$0.20'
            },
            'strategy5C': {
                'tp': 0.15,
                'sl': None,
                'max_buy_price': 0.99,
                'double_down_drop': 0.25,
                'max_positions_per_side': 2,
                'open_at_start': True,
                'description': 'Open both sides at start, double down at -$0.25 from last entry'
            }
        }
        
        # Positions: List of positions per side per strategy
        self.positions: Dict[str, Dict[str, List[Position]]] = {
            strategy: {'CALL': [], 'PUT': []} 
            for strategy in self.strategies.keys()
        }
        
        self.trades: Dict[str, List[Trade]] = {
            strategy: [] for strategy in self.strategies.keys()
        }
        
        # Period tracking
        self.current_period_start = None
        self.period_pnl: Dict[str, Dict[str, float]] = {
            strategy: {'CALL': 0.0, 'PUT': 0.0, 'TOTAL': 0.0}
            for strategy in self.strategies.keys()
        }
        
        # Overall PNL tracking
        self.overall_pnl: Dict[str, Dict[str, float]] = {
            strategy: {'CALL': 0.0, 'PUT': 0.0, 'TOTAL': 0.0}
            for strategy in self.strategies.keys()
        }
        
        # Track if we've opened initial positions this period (for 5C)
        self.opened_initial_5c = False
        
        # Track if we've done end-of-period cleanup
        self.end_cleanup_done = False
    
    def get_current_period_start(self) -> datetime:
        """Get the start time of the current 15-minute period"""
        now = datetime.now()
        minute = now.minute
        
        if minute < 15:
            period_minute = 0
        elif minute < 30:
            period_minute = 15
        elif minute < 45:
            period_minute = 30
        else:
            period_minute = 45
        
        return now.replace(minute=period_minute, second=0, microsecond=0)
    
    def get_next_expiry(self, from_time: datetime = None) -> datetime:
        """Calculate the next 15-minute expiry time"""
        if from_time is None:
            from_time = datetime.now()
        
        minute = from_time.minute
        if minute < 15:
            next_minute = 15
        elif minute < 30:
            next_minute = 30
        elif minute < 45:
            next_minute = 45
        else:
            next_minute = 0
        
        expiry = from_time.replace(second=0, microsecond=0)
        if next_minute == 0:
            expiry = expiry + timedelta(hours=1)
            expiry = expiry.replace(minute=0)
        else:
            expiry = expiry.replace(minute=next_minute)
        
        return expiry
    
    def seconds_to_expiry(self, expiry: datetime) -> float:
        """Calculate seconds remaining to expiry"""
        return (expiry - datetime.now()).total_seconds()
    
    def force_cleanup_all_positions(self):
        """Force close ALL positions for ALL strategies during end buffer"""
        call_bid = self.current_call_bid if self.current_call_bid is not None else self.last_valid_call_bid
        put_bid = self.current_put_bid if self.current_put_bid is not None else self.last_valid_put_bid
        
        if call_bid is None or put_bid is None:
            return
        
        for strategy_name in self.strategies.keys():
            # Close CALL positions
            if self.positions[strategy_name]['CALL']:
                call_settlement = 1.00 if call_bid >= 0.5 else 0.00
                print(f"[{strategy_name}] 🧹 END CLEANUP CALL: Bid ${call_bid:.4f} → ${call_settlement:.2f}")
                self.close_all_positions('CALL', call_settlement, strategy_name, 'EXPIRY')
            
            # Close PUT positions
            if self.positions[strategy_name]['PUT']:
                put_settlement = 1.00 if put_bid >= 0.5 else 0.00
                print(f"[{strategy_name}] 🧹 END CLEANUP PUT: Bid ${put_bid:.4f} → ${put_settlement:.2f}")
                self.close_all_positions('PUT', put_settlement, strategy_name, 'EXPIRY')
        
        self.end_cleanup_done = True
        print("✅ All positions closed for period end")
    
    def verify_all_positions_closed(self):
        """Verify that all positions are actually closed"""
        for strategy_name in self.strategies.keys():
            if self.positions[strategy_name]['CALL'] or self.positions[strategy_name]['PUT']:
                print(f"⚠️  WARNING: {strategy_name} still has open positions!")
                # Force clear them
                self.positions[strategy_name]['CALL'] = []
                self.positions[strategy_name]['PUT'] = []
    
    def force_close_all_positions(self, strategy_name: str):
        """Force close all positions during end buffer (last 10 seconds)"""
        # Get current bid prices
        call_bid = self.current_call_bid if self.current_call_bid is not None else self.last_valid_call_bid
        put_bid = self.current_put_bid if self.current_put_bid is not None else self.last_valid_put_bid
        
        if call_bid is None or put_bid is None:
            return
        
        # Determine expiry prices based on bid
        call_expiry_price = self.get_expiry_price(call_bid)
        put_expiry_price = self.get_expiry_price(put_bid)
        
        # Close CALL positions if any exist
        if self.positions[strategy_name]['CALL']:
            print(f"[{strategy_name}] 🛑 FORCE CLOSING CALL at end buffer: Bid ${call_bid:.4f} → ${call_expiry_price:.2f}")
            self.close_all_positions('CALL', call_expiry_price, strategy_name, 'EXPIRY')
        
        # Close PUT positions if any exist
        if self.positions[strategy_name]['PUT']:
            print(f"[{strategy_name}] 🛑 FORCE CLOSING PUT at end buffer: Bid ${put_bid:.4f} → ${put_expiry_price:.2f}")
            self.close_all_positions('PUT', put_expiry_price, strategy_name, 'EXPIRY')
    
    def is_in_end_buffer(self) -> bool:
        """Check if we're in the END buffer (last 10 seconds)"""
        next_expiry = self.get_next_expiry()
        seconds_to_expiry = self.seconds_to_expiry(next_expiry)
        return seconds_to_expiry <= 10
    
    def is_in_start_buffer(self) -> bool:
        """Check if we're in the START buffer (first 10 seconds)"""
        next_expiry = self.get_next_expiry()
        seconds_to_expiry = self.seconds_to_expiry(next_expiry)
        # Start buffer: 890-900 seconds remaining (first 10 seconds of 15-min period)
        return 890 <= seconds_to_expiry <= 900
    
    def is_in_buffer_period(self) -> bool:
        """Check if we're in the 10-second buffer at start or end of period"""
        return self.is_in_start_buffer() or self.is_in_end_buffer()
    
    def should_check_expiry(self, expiry: datetime) -> bool:
        """Check if we're at expiry time (5 seconds or less)"""
        now = datetime.now()
        seconds_remaining = (expiry - now).total_seconds()
        return 0 <= seconds_remaining <= 5
    
    def get_expiry_price(self, current_bid: float) -> float:
        """Determine expiry price based on bid"""
        return 1.00 if current_bid >= 0.5 else 0.00
    
    def calculate_avg_entry_price(self, positions: List[Position]) -> float:
        """Calculate weighted average entry price"""
        if not positions:
            return 0.0
        return sum(p.entry_price for p in positions) / len(positions)
    
    def get_last_entry_price(self, option_type: str, strategy_name: str) -> Optional[float]:
        """Get the last entry price for a given side"""
        positions = self.positions[strategy_name][option_type]
        if not positions:
            return None
        # Return the most recent position's entry price
        return positions[-1].entry_price
    
    def calculate_tp_sl(self, entry_price: float, strategy_name: str) -> tuple:
        """Calculate take profit and stop loss"""
        config = self.strategies[strategy_name]
        
        # All Strategy 5 variants use fixed TP
        tp = entry_price + config['tp']
        tp = min(1.00, tp)
        
        # All Strategy 5 variants have no SL
        sl = -1.00
        
        return tp, sl
    
    def can_enter_position_5a(self, option_type: str, ask_price: float) -> bool:
        """Check if we can enter for Strategy 5A"""
        if self.is_in_buffer_period():
            return False
        
        config = self.strategies['strategy5A']
        
        # Check max buy price
        if ask_price >= config.get('max_buy_price', 0.99):
            return False
        
        min_entry = config.get('min_entry_price', 0.60)
        
        # Get current prices for both sides
        call_ask = self.current_call_ask if self.current_call_ask is not None else self.last_valid_call_ask
        put_ask = self.current_put_ask if self.current_put_ask is not None else self.last_valid_put_ask
        
        if call_ask is None or put_ask is None:
            return False
        
        # Check if we have any positions yet
        has_positions = len(self.positions['strategy5A']['CALL']) > 0 or len(self.positions['strategy5A']['PUT']) > 0
        
        if not has_positions:
            # First entry: need one side above min_entry_price
            if max(call_ask, put_ask) < min_entry:
                return False
        
        return True
    
    def can_enter_position_5b(self, option_type: str, ask_price: float) -> bool:
        """Check if we can enter for Strategy 5B (same as 5A initially)"""
        if self.is_in_buffer_period():
            return False
        
        config = self.strategies['strategy5B']
        
        if ask_price >= config.get('max_buy_price', 0.99):
            return False
        
        min_entry = config.get('min_entry_price', 0.60)
        
        call_ask = self.current_call_ask if self.current_call_ask is not None else self.last_valid_call_ask
        put_ask = self.current_put_ask if self.current_put_ask is not None else self.last_valid_put_ask
        
        if call_ask is None or put_ask is None:
            return False
        
        has_positions = len(self.positions['strategy5B']['CALL']) > 0 or len(self.positions['strategy5B']['PUT']) > 0
        
        if not has_positions:
            if max(call_ask, put_ask) < min_entry:
                return False
        
        return True
    
    def can_enter_position_5c(self, option_type: str, ask_price: float) -> bool:
        """Check if we can enter for Strategy 5C"""
        # Don't enter in end buffer, but OK in start buffer for initial positions
        next_expiry = self.get_next_expiry()
        seconds_to_expiry = self.seconds_to_expiry(next_expiry)
        
        # End buffer only
        if seconds_to_expiry <= 10:
            return False
        
        config = self.strategies['strategy5C']
        
        if ask_price >= config.get('max_buy_price', 0.99):
            return False
        
        return True
    
    def should_double_down_5b(self, option_type: str, ask_price: float) -> bool:
        """Check if we should add another position for 5B"""
        config = self.strategies['strategy5B']
        double_down_price = config.get('double_down_price', 0.20)
        max_positions = config.get('max_positions_per_side', 2)
        
        current_positions = len(self.positions['strategy5B'][option_type])
        
        return current_positions < max_positions and ask_price <= double_down_price
    
    def should_double_down_5c(self, option_type: str, ask_price: float) -> bool:
        """Check if we should add another position for 5C (based on price drop)"""
        config = self.strategies['strategy5C']
        double_down_drop = config.get('double_down_drop', 0.25)
        max_positions = config.get('max_positions_per_side', 2)
        
        current_positions = len(self.positions['strategy5C'][option_type])
        
        if current_positions >= max_positions:
            return False
        
        # Get last entry price
        last_entry = self.get_last_entry_price(option_type, 'strategy5C')
        if last_entry is None:
            return False
        
        # Check if price has dropped by the required amount
        return ask_price <= (last_entry - double_down_drop)
    
    def open_position(self, option_type: str, ask_price: float, strategy_name: str, is_double_down: bool = False):
        """Open a new position"""
        timestamp = datetime.now()
        expiry = self.get_next_expiry(timestamp)
        
        tp, sl = self.calculate_tp_sl(ask_price, strategy_name)
        
        position = Position(
            option_type=option_type,
            entry_price=ask_price,
            entry_time=timestamp.isoformat(),
            expiry_time=expiry.isoformat(),
            take_profit=tp,
            stop_loss=sl,
            strategy=strategy_name
        )
        
        self.positions[strategy_name][option_type].append(position)
        
        trade = Trade(
            timestamp=timestamp.isoformat(),
            option_type=option_type,
            action='BUY',
            price=ask_price,
            reason='DOUBLE_DOWN' if is_double_down else 'OPEN',
            pnl=0.0,
            expiry_time=expiry.isoformat()
        )
        
        self.trades[strategy_name].append(trade)
        
        position_count = len(self.positions[strategy_name][option_type])
        time_to_expiry = self.seconds_to_expiry(expiry)
        
        action_str = "DOUBLED DOWN" if is_double_down else "Opened"
        print(f"[{strategy_name}] {action_str} {option_type} (#{position_count}) @ ${ask_price:.4f} | TP: ${tp:.4f} | Expiry: {expiry.strftime('%H:%M:%S')} ({time_to_expiry:.0f}s)")
    
    def close_all_positions(self, option_type: str, exit_price: float, strategy_name: str, reason: str):
        """Close all positions for a given side"""
        positions = self.positions[strategy_name][option_type]
        
        if not positions:
            return
        
        timestamp = datetime.now()
        quantity = len(positions)
        
        # Calculate average entry price
        avg_entry = self.calculate_avg_entry_price(positions)
        
        # Calculate PNL
        pnl = (exit_price - avg_entry) * quantity
        
        # Get expiry info from first position
        expiry = datetime.fromisoformat(positions[0].expiry_time)
        seconds_to_exp = self.seconds_to_expiry(expiry)
        
        trade = Trade(
            timestamp=timestamp.isoformat(),
            option_type=option_type,
            action='SELL',
            price=exit_price,
            reason=reason,
            pnl=pnl,
            expiry_time=positions[0].expiry_time,
            open_price=avg_entry,
            close_price=exit_price,
            seconds_to_expiry=seconds_to_exp,
            quantity=quantity
        )
        
        self.trades[strategy_name].append(trade)
        
        # Update PNL
        self.period_pnl[strategy_name][option_type] += pnl
        self.period_pnl[strategy_name]['TOTAL'] += pnl
        self.overall_pnl[strategy_name][option_type] += pnl
        self.overall_pnl[strategy_name]['TOTAL'] += pnl
        
        print(f"[{strategy_name}] Closed {quantity}x {option_type} @ ${exit_price:.4f} (avg ${avg_entry:.4f}) | {reason} | PNL: ${pnl:+.4f} | Period: ${self.period_pnl[strategy_name]['TOTAL']:+.4f} | Overall: ${self.overall_pnl[strategy_name]['TOTAL']:+.4f}")
        
        # Clear all positions
        self.positions[strategy_name][option_type] = []
    
    def check_expiry(self, option_type: str, current_bid: float, strategy_name: str):
        """Check if positions should be closed due to expiry - FORCE CLOSE at 10 seconds"""
        positions = self.positions[strategy_name][option_type]
        
        if not positions:
            return
        
        expiry = datetime.fromisoformat(positions[0].expiry_time)
        seconds_left = self.seconds_to_expiry(expiry)
        
        # FORCE CLOSE when 10 seconds or less remaining
        if seconds_left <= 10:
            # Determine settlement price based on current bid
            if current_bid >= 0.5:
                settlement_price = 1.00
                result_str = "WIN ($1.00)"
            else:
                settlement_price = 0.00
                result_str = "LOSS ($0.00)"
            
            print(f"[{strategy_name}] ⏰ FORCE SETTLEMENT ({seconds_left:.1f}s left) {option_type}: Bid ${current_bid:.4f} → {result_str}")
            self.close_all_positions(option_type, settlement_price, strategy_name, 'EXPIRY')
    
    def check_take_profit(self, option_type: str, bid_price: float, strategy_name: str):
        """Check if take profit hit"""
        positions = self.positions[strategy_name][option_type]
        
        if not positions:
            return False
        
        tp = positions[0].take_profit
        
        if bid_price >= tp:
            self.close_all_positions(option_type, bid_price, strategy_name, 'TP')
            return True
        
        return False
    
    def check_position_5a(self, option_type: str, ask_price: float, bid_price: float):
        """Check positions and manage trades for Strategy 5A"""
        strategy_name = 'strategy5A'
        positions = self.positions[strategy_name][option_type]
        
        # Check for expiry first
        self.check_expiry(option_type, bid_price, strategy_name)
        positions = self.positions[strategy_name][option_type]
        
        # If no positions after expiry, try to open
        if not positions:
            if self.can_enter_position_5a(option_type, ask_price):
                self.open_position(option_type, ask_price, strategy_name)
            return
        
        # Check TP
        if self.check_take_profit(option_type, bid_price, strategy_name):
            # Immediately try to reopen
            if self.can_enter_position_5a(option_type, ask_price):
                self.open_position(option_type, ask_price, strategy_name)
    
    def check_position_5b(self, option_type: str, ask_price: float, bid_price: float):
        """Check positions and manage trades for Strategy 5B"""
        strategy_name = 'strategy5B'
        positions = self.positions[strategy_name][option_type]
        
        # Check for expiry first
        self.check_expiry(option_type, bid_price, strategy_name)
        positions = self.positions[strategy_name][option_type]
        
        # If no positions after expiry, try to open
        if not positions:
            if self.can_enter_position_5b(option_type, ask_price):
                self.open_position(option_type, ask_price, strategy_name)
            return
        
        # Check TP
        if self.check_take_profit(option_type, bid_price, strategy_name):
            # Immediately try to reopen
            if self.can_enter_position_5b(option_type, ask_price):
                self.open_position(option_type, ask_price, strategy_name)
            return
        
        # Check if we should double down
        if self.should_double_down_5b(option_type, ask_price):
            self.open_position(option_type, ask_price, strategy_name, is_double_down=True)
    
    def check_position_5c(self, option_type: str, ask_price: float, bid_price: float):
        """Check positions and manage trades for Strategy 5C"""
        strategy_name = 'strategy5C'
        positions = self.positions[strategy_name][option_type]
        
        # Check for expiry first
        self.check_expiry(option_type, bid_price, strategy_name)
        positions = self.positions[strategy_name][option_type]
        
        # If no positions, open initial position (can happen at start of period)
        if not positions:
            if self.can_enter_position_5c(option_type, ask_price):
                self.open_position(option_type, ask_price, strategy_name)
            return
        
        # Check TP
        if self.check_take_profit(option_type, bid_price, strategy_name):
            # Immediately reopen
            if self.can_enter_position_5c(option_type, ask_price):
                self.open_position(option_type, ask_price, strategy_name)
            return
        
        # Check if we should double down (based on price drop)
        if self.should_double_down_5c(option_type, ask_price):
            self.open_position(option_type, ask_price, strategy_name, is_double_down=True)
    
    def read_price_file(self, filepath: Path) -> Optional[Dict[str, float]]:
        """Read price data from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            ask_price = data.get('best_ask')
            bid_price = data.get('best_bid')
            
            if ask_price is not None:
                ask_price = float(ask_price)
            if bid_price is not None:
                bid_price = float(bid_price)
            
            return {'ask': ask_price, 'bid': bid_price}
        except Exception as e:
            return None
    
    def update_prices(self):
        """Check for file updates and update prices"""
        updated = False
        
        if self.call_file.exists():
            mtime = self.call_file.stat().st_mtime
            if mtime > self.last_call_mtime:
                prices = self.read_price_file(self.call_file)
                if prices:
                    self.current_call_ask = prices['ask']
                    self.current_call_bid = prices['bid']
                    
                    if prices['ask'] is not None:
                        self.last_valid_call_ask = prices['ask']
                    if prices['bid'] is not None:
                        self.last_valid_call_bid = prices['bid']
                    
                    self.last_call_mtime = mtime
                    updated = True
        
        if self.put_file.exists():
            mtime = self.put_file.stat().st_mtime
            if mtime > self.last_put_mtime:
                prices = self.read_price_file(self.put_file)
                if prices:
                    self.current_put_ask = prices['ask']
                    self.current_put_bid = prices['bid']
                    
                    if prices['ask'] is not None:
                        self.last_valid_put_ask = prices['ask']
                    if prices['bid'] is not None:
                        self.last_valid_put_bid = prices['bid']
                    
                    self.last_put_mtime = mtime
                    updated = True
        
        return updated
    
    def check_period_change(self):
        """Check if we've moved to a new period"""
        current_period = self.get_current_period_start()
        
        if self.current_period_start is None:
            self.current_period_start = current_period
            self.opened_initial_5c = False
            self.end_cleanup_done = False
            return False
        
        if current_period > self.current_period_start:
            # Verify all positions are closed before period change
            self.verify_all_positions_closed()
            
            # Save results
            self.save_period_results()
            
            # Reset for new period
            self.current_period_start = current_period
            self.opened_initial_5c = False
            self.end_cleanup_done = False
            
            # Reset period PNL
            for strategy in self.strategies.keys():
                self.period_pnl[strategy] = {'CALL': 0.0, 'PUT': 0.0, 'TOTAL': 0.0}
            
            print(f"\n🔄 NEW PERIOD STARTED: {current_period.strftime('%H:%M')} - All positions cleared, starting fresh\n")
            return True
        
        return False
    
    def save_period_results(self):
        """Save results for completed period - separate file per strategy"""
        if self.current_period_start is None:
            return
        
        period_end = self.current_period_start + timedelta(minutes=15)
        timestamp_str = self.current_period_start.strftime('%Y%m%d_%H%M')
        
        for strategy_name in self.strategies.keys():
            filename = f"period_{timestamp_str}.json"
            output_file = self.output_dirs[strategy_name] / filename
            
            period_trades = [
                t for t in self.trades[strategy_name] 
                if self.current_period_start <= datetime.fromisoformat(t.timestamp) < period_end
            ]
            
            sell_trades = [t for t in period_trades if t.action == 'SELL']
            
            total_trades = len(sell_trades)
            tp_trades = len([t for t in sell_trades if t.reason == 'TP'])
            sl_trades = len([t for t in sell_trades if t.reason == 'SL'])
            expiry_trades = len([t for t in sell_trades if t.reason == 'EXPIRY'])
            
            winning_trades = len([t for t in sell_trades if t.pnl > 0])
            losing_trades = len([t for t in sell_trades if t.pnl < 0])
            
            avg_win = sum([t.pnl for t in sell_trades if t.pnl > 0]) / winning_trades if winning_trades > 0 else 0
            avg_loss = sum([t.pnl for t in sell_trades if t.pnl < 0]) / losing_trades if losing_trades > 0 else 0
            
            formatted_trades = []
            for trade in period_trades:
                if trade.action == 'SELL':
                    formatted_trades.append({
                        'timestamp': trade.timestamp,
                        'option_type': trade.option_type,
                        'action': trade.action,
                        'quantity': trade.quantity,
                        'open_price': trade.open_price,
                        'close_price': trade.close_price,
                        'pnl': trade.pnl,
                        'reason': trade.reason,
                        'seconds_to_expiry': trade.seconds_to_expiry
                    })
                else:
                    formatted_trades.append({
                        'timestamp': trade.timestamp,
                        'option_type': trade.option_type,
                        'action': trade.action,
                        'price': trade.price,
                        'reason': trade.reason
                    })
            
            result = {
                'strategy': strategy_name,
                'strategy_config': self.strategies[strategy_name],
                'period_start': self.current_period_start.isoformat(),
                'period_end': period_end.isoformat(),
                'saved_at': datetime.now().isoformat(),
                'period_pnl': self.period_pnl[strategy_name],
                'overall_pnl': self.overall_pnl[strategy_name],
                'statistics': {
                    'total_trades': total_trades,
                    'tp_trades': tp_trades,
                    'sl_trades': sl_trades,
                    'expiry_trades': expiry_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss
                },
                'trades': formatted_trades
            }
            
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
        
        print(f"\n{'='*100}")
        print(f"📊 PERIOD COMPLETED: {self.current_period_start.strftime('%Y-%m-%d %H:%M')}")
        for strategy_name in self.strategies.keys():
            pnl = self.period_pnl[strategy_name]
            overall = self.overall_pnl[strategy_name]
            print(f"   {strategy_name}: Period=${pnl['TOTAL']:+.4f} | Overall=${overall['TOTAL']:+.4f}")
        print(f"{'='*100}\n")
    
    def print_status(self):
        """Print current status"""
        print("\n" + "="*100)
        print(f"STATUS - {datetime.now().strftime('%H:%M:%S')}")
        print("="*100)
        
        call_ask_str = f"${self.current_call_ask:.4f}" if self.current_call_ask is not None else "None"
        call_bid_str = f"${self.current_call_bid:.4f}" if self.current_call_bid is not None else "None"
        put_ask_str = f"${self.current_put_ask:.4f}" if self.current_put_ask is not None else "None"
        put_bid_str = f"${self.current_put_bid:.4f}" if self.current_put_bid is not None else "None"
        
        print(f"CALL: Ask={call_ask_str} Bid={call_bid_str} | PUT: Ask={put_ask_str} Bid={put_bid_str}")
        
        next_expiry = self.get_next_expiry()
        seconds_to_expiry = self.seconds_to_expiry(next_expiry)
        in_buffer = self.is_in_buffer_period()
        
        buffer_status = " [BUFFER]" if in_buffer else ""
        print(f"Expiry: {next_expiry.strftime('%H:%M:%S')} (in {seconds_to_expiry:.0f}s){buffer_status}")
        print("-"*100)
        
        for strategy_name in self.strategies.keys():
            period_pnl = self.period_pnl[strategy_name]
            overall_pnl = self.overall_pnl[strategy_name]
            
            call_pos = len(self.positions[strategy_name]['CALL'])
            put_pos = len(self.positions[strategy_name]['PUT'])
            
            print(f"{strategy_name}: Period=${period_pnl['TOTAL']:+.4f} | Overall=${overall_pnl['TOTAL']:+.4f} | Pos: C={call_pos} P={put_pos}")
        
        print("="*100 + "\n")
    
    def run(self, check_interval: float = 0.5, status_interval: int = 30):
        """Main monitoring loop"""
        print("\n" + "="*100)
        print("🚀 STRATEGY 5 SIMULATOR (5A, 5B, 5C)")
        print("="*100)
        print(f"Monitoring: {self.call_file}")
        print(f"           {self.put_file}")
        print(f"\n📋 STRATEGIES:")
        for name, config in self.strategies.items():
            print(f"   {name}: {config['description']}")
            print(f"            Output: {self.output_dirs[name]}")
        print(f"\n⏰ Force close at 10s before expiry | No trading first 10s or last 12s")
        print("="*100 + "\n")
        
        last_status_time = time.time()
        
        try:
            while True:
                # Update prices first
                if self.update_prices():
                    call_ask = self.current_call_ask if self.current_call_ask is not None else self.last_valid_call_ask
                    call_bid = self.current_call_bid if self.current_call_bid is not None else self.last_valid_call_bid
                    put_ask = self.current_put_ask if self.current_put_ask is not None else self.last_valid_put_ask
                    put_bid = self.current_put_bid if self.current_put_bid is not None else self.last_valid_put_bid
                    
                    if all([call_ask, call_bid, put_ask, put_bid]):
                        # CRITICAL: Force close at exactly 10 seconds
                        next_expiry = self.get_next_expiry()
                        seconds_left = self.seconds_to_expiry(next_expiry)
                        
                        if 9.5 <= seconds_left <= 10.5 and not self.end_cleanup_done:
                            print(f"\n🛑 FORCE CLOSING ALL POSITIONS ({seconds_left:.1f}s to expiry)...")
                            self.force_cleanup_all_positions()
                            print("✅ All positions closed and saved\n")
                        
                        # Don't trade in end buffer (≤12s)
                        if seconds_left <= 12:
                            pass  # Just wait for period to end
                        
                        # Don't trade in start buffer (first 10s)
                        elif 890 <= seconds_left <= 900:
                            pass  # Just observe prices
                        
                        # Normal trading window
                        else:
                            # 5A
                            if not self.positions['strategy5A']['CALL']:
                                if self.can_enter_position_5a('CALL', call_ask):
                                    self.open_position('CALL', call_ask, 'strategy5A')
                            else:
                                if self.check_take_profit('CALL', call_bid, 'strategy5A'):
                                    if self.can_enter_position_5a('CALL', call_ask):
                                        self.open_position('CALL', call_ask, 'strategy5A')
                            
                            if not self.positions['strategy5A']['PUT']:
                                if self.can_enter_position_5a('PUT', put_ask):
                                    self.open_position('PUT', put_ask, 'strategy5A')
                            else:
                                if self.check_take_profit('PUT', put_bid, 'strategy5A'):
                                    if self.can_enter_position_5a('PUT', put_ask):
                                        self.open_position('PUT', put_ask, 'strategy5A')
                            
                            # 5B
                            if not self.positions['strategy5B']['CALL']:
                                if self.can_enter_position_5b('CALL', call_ask):
                                    self.open_position('CALL', call_ask, 'strategy5B')
                            else:
                                if self.check_take_profit('CALL', call_bid, 'strategy5B'):
                                    if self.can_enter_position_5b('CALL', call_ask):
                                        self.open_position('CALL', call_ask, 'strategy5B')
                                elif self.should_double_down_5b('CALL', call_ask):
                                    self.open_position('CALL', call_ask, 'strategy5B', is_double_down=True)
                            
                            if not self.positions['strategy5B']['PUT']:
                                if self.can_enter_position_5b('PUT', put_ask):
                                    self.open_position('PUT', put_ask, 'strategy5B')
                            else:
                                if self.check_take_profit('PUT', put_bid, 'strategy5B'):
                                    if self.can_enter_position_5b('PUT', put_ask):
                                        self.open_position('PUT', put_ask, 'strategy5B')
                                elif self.should_double_down_5b('PUT', put_ask):
                                    self.open_position('PUT', put_ask, 'strategy5B', is_double_down=True)
                            
                            # 5C
                            if not self.positions['strategy5C']['CALL']:
                                if self.can_enter_position_5c('CALL', call_ask):
                                    self.open_position('CALL', call_ask, 'strategy5C')
                            else:
                                if self.check_take_profit('CALL', call_bid, 'strategy5C'):
                                    if self.can_enter_position_5c('CALL', call_ask):
                                        self.open_position('CALL', call_ask, 'strategy5C')
                                elif self.should_double_down_5c('CALL', call_ask):
                                    self.open_position('CALL', call_ask, 'strategy5C', is_double_down=True)
                            
                            if not self.positions['strategy5C']['PUT']:
                                if self.can_enter_position_5c('PUT', put_ask):
                                    self.open_position('PUT', put_ask, 'strategy5C')
                            else:
                                if self.check_take_profit('PUT', put_bid, 'strategy5C'):
                                    if self.can_enter_position_5c('PUT', put_ask):
                                        self.open_position('PUT', put_ask, 'strategy5C')
                                elif self.should_double_down_5c('PUT', put_ask):
                                    self.open_position('PUT', put_ask, 'strategy5C', is_double_down=True)
                
                # Check for period change AFTER all trading is done
                self.check_period_change()
                
                if time.time() - last_status_time > status_interval:
                    if self.last_valid_call_ask:
                        self.print_status()
                    last_status_time = time.time()
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down...")
            self.save_period_results()
            self.print_status()
            print("✅ Goodbye!\n")


if __name__ == "__main__":
    simulator = Strategy5Simulator(
        call_file="/home/ubuntu/013_2025_polymarket/15M_BTC_CALL_rest.json",
        put_file="/home/ubuntu/013_2025_polymarket/15M_BTC_PUT_rest.json",
        base_output_dir="/home/ubuntu/013_2025_polymarket/bot016_react/simulators"
    )
    
    simulator.run(check_interval=0.5, status_interval=30)
