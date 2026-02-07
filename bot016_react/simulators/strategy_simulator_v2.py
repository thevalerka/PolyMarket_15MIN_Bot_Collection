import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

@dataclass
class Trade:
    """Represents a single trade"""
    timestamp: str
    option_type: str  # 'CALL' or 'PUT'
    action: str  # 'BUY' or 'SELL'
    price: float
    reason: str  # 'OPEN', 'TP', 'SL', 'EXPIRY'
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

class StrategySimulator:
    """Simulates trading strategies on binary options"""

    def __init__(self, call_file: str, put_file: str, output_dir: str = "/home/ubuntu/013_2025_polymarket/bot016_react/simulators/simulation_results_v2"):
        self.call_file = Path(call_file)
        self.put_file = Path(put_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
            'strategy1': {
                'tp': 0.10, 'sl': 0.08, 'tp_type': 'fixed', 'sl_type': 'fixed',
                'max_buy_price': 0.99, 'description': 'TP=$0.10, SL=$0.08'
            },
            'strategy2': {
                'tp': 0.10, 'sl': 0.10, 'tp_type': 'fixed', 'sl_type': 'fixed',
                'max_buy_price': 0.99, 'description': 'TP=$0.10, SL=$0.10'
            },
            'strategy3': {
                'tp': 0.10, 'sl': None, 'tp_type': 'fixed', 'sl_type': 'never',
                'max_buy_price': 0.99, 'description': 'TP=$0.10, No SL'
            },
            'strategy3B': {
                'tp': 0.15, 'sl': None, 'tp_type': 'fixed', 'sl_type': 'never',
                'max_buy_price': 0.99, 'description': 'TP=$0.15, No SL'
            },
            'strategy3C': {
                'tp': 0.03, 'sl': None, 'tp_type': 'fixed', 'sl_type': 'never',
                'max_buy_price': 0.99, 'description': 'TP=$0.03, No SL'
            },
            'strategy4': {
                'tp': None, 'sl': None, 'tp_type': 'one_third_distance', 'sl_type': 'one_third_distance',
                'min_distance': 0.05, 'max_buy_price': 0.95,
                'description': 'TP/SL at 1/3 distance (min $0.05), max buy $0.95'
            },
            'strategy5A': {
                'tp': 0.15, 'sl': None, 'tp_type': 'fixed', 'sl_type': 'never',
                'max_buy_price': 0.99, 'min_entry_price': 0.60,
                'description': 'Like 3B, enter when one side >$0.60'
            },
            'strategy5B': {
                'tp': 0.15, 'sl': None, 'tp_type': 'fixed', 'sl_type': 'never',
                'max_buy_price': 0.99, 'min_entry_price': 0.60,
                'double_down_price': 0.20, 'max_positions_per_side': 2,
                'description': 'Like 5A, double down when ≤$0.20'
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

    def is_in_buffer_period(self) -> bool:
        """Check if we're in the 10-second buffer at start or end of period"""
        next_expiry = self.get_next_expiry()
        seconds_to_expiry = self.seconds_to_expiry(next_expiry)

        # End buffer: 10 seconds before expiry
        if seconds_to_expiry <= 10:
            return True

        # Start buffer: first 10 seconds (890-900 seconds remaining)
        if 890 <= seconds_to_expiry <= 900:
            return True

        return False

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

    def calculate_tp_sl(self, entry_price: float, strategy_name: str) -> tuple:
        """Calculate take profit and stop loss"""
        config = self.strategies[strategy_name]

        # Calculate TP
        if config['tp_type'] == 'fixed':
            tp = entry_price + config['tp']
        elif config['tp_type'] == 'one_third_distance':
            distance = 1.00 - entry_price
            min_distance = config.get('min_distance', 0.0)
            if distance < min_distance:
                distance = min_distance
            tp = entry_price + (distance / 3)
        else:
            tp = entry_price + 0.10

        # Calculate SL
        if config['sl_type'] == 'fixed':
            sl = entry_price - config['sl']
        elif config['sl_type'] == 'never':
            sl = -1.00
        elif config['sl_type'] == 'one_third_distance':
            distance = 1.00 - entry_price
            min_distance = config.get('min_distance', 0.0)
            if distance < min_distance:
                distance = min_distance
            sl = entry_price - (distance / 3)
        else:
            sl = -1.00

        tp = min(1.00, tp)
        if sl <= 0.00:
            sl = -1.00
        else:
            sl = max(0.00, sl)

        return tp, sl

    def can_enter_position(self, option_type: str, ask_price: float, strategy_name: str) -> bool:
        """Check if we can enter a position based on strategy rules"""
        if self.is_in_buffer_period():
            return False

        config = self.strategies[strategy_name]

        # Check max buy price
        if ask_price >= config.get('max_buy_price', 0.99):
            return False

        # For 5A and 5B: check minimum entry price requirement
        if strategy_name in ['strategy5A', 'strategy5B']:
            min_entry = config.get('min_entry_price', 0.60)

            # Get current prices for both sides
            call_ask = self.current_call_ask if self.current_call_ask is not None else self.last_valid_call_ask
            put_ask = self.current_put_ask if self.current_put_ask is not None else self.last_valid_put_ask

            if call_ask is None or put_ask is None:
                return False

            # Check if we have any positions yet
            has_positions = len(self.positions[strategy_name]['CALL']) > 0 or len(self.positions[strategy_name]['PUT']) > 0

            if not has_positions:
                # First entry: need one side above min_entry_price
                if max(call_ask, put_ask) < min_entry:
                    return False

        return True

    def should_double_down(self, option_type: str, ask_price: float, strategy_name: str) -> bool:
        """Check if we should add another position (Strategy 5B only)"""
        if strategy_name != 'strategy5B':
            return False

        config = self.strategies[strategy_name]
        double_down_price = config.get('double_down_price', 0.20)
        max_positions = config.get('max_positions_per_side', 2)

        current_positions = len(self.positions[strategy_name][option_type])

        # Can only double down if we have less than max and price is low enough
        return current_positions < max_positions and ask_price <= double_down_price

    def open_position(self, option_type: str, ask_price: float, strategy_name: str):
        """Open a new position"""
        if not self.can_enter_position(option_type, ask_price, strategy_name):
            return

        timestamp = datetime.now()
        expiry = self.get_next_expiry(timestamp)

        # For 5B, check if this is a double-down
        is_double_down = self.should_double_down(option_type, ask_price, strategy_name)

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
        sl_str = f"${sl:.4f}" if sl >= 0 else "NEVER"

        action_str = "DOUBLED DOWN" if is_double_down else "Opened"
        print(f"[{strategy_name}] {action_str} {option_type} (#{position_count}) @ ${ask_price:.4f} | TP: ${tp:.4f} | SL: {sl_str} | Expiry: {expiry.strftime('%H:%M:%S')} ({time_to_expiry:.0f}s)")

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

        print(f"[{strategy_name}] Closed {quantity}x {option_type} @ ${exit_price:.4f} (avg entry ${avg_entry:.4f}) | Reason: {reason} | PNL: ${pnl:+.4f} | Period: ${self.period_pnl[strategy_name]['TOTAL']:+.4f} | Overall: ${self.overall_pnl[strategy_name]['TOTAL']:+.4f}")

        # Clear all positions
        self.positions[strategy_name][option_type] = []

    def check_expiry(self, option_type: str, current_bid: float, strategy_name: str):
        """Check if positions should be closed due to expiry"""
        positions = self.positions[strategy_name][option_type]

        if not positions:
            return

        expiry = datetime.fromisoformat(positions[0].expiry_time)

        if self.should_check_expiry(expiry):
            expiry_price = self.get_expiry_price(current_bid)
            seconds_left = self.seconds_to_expiry(expiry)
            print(f"[{strategy_name}] ⏰ EXPIRY ({seconds_left:.1f}s left) {option_type}: Bid ${current_bid:.4f} → ${expiry_price:.2f}")
            self.close_all_positions(option_type, expiry_price, strategy_name, 'EXPIRY')

    def check_take_profit(self, option_type: str, bid_price: float, strategy_name: str):
        """Check if take profit hit (based on highest TP of all positions)"""
        positions = self.positions[strategy_name][option_type]

        if not positions:
            return False

        # Use the TP of the first position (they should all be similar)
        tp = positions[0].take_profit

        if bid_price >= tp:
            self.close_all_positions(option_type, bid_price, strategy_name, 'TP')
            return True

        return False

    def check_stop_loss(self, option_type: str, bid_price: float, strategy_name: str):
        """Check if stop loss hit"""
        positions = self.positions[strategy_name][option_type]

        if not positions:
            return False

        sl = positions[0].stop_loss

        if sl >= 0.00 and bid_price <= sl:
            self.close_all_positions(option_type, bid_price, strategy_name, 'SL')
            return True

        return False

    def check_position(self, option_type: str, ask_price: Optional[float], bid_price: Optional[float], strategy_name: str):
        """Check positions and manage trades"""
        if ask_price is None or bid_price is None:
            return

        positions = self.positions[strategy_name][option_type]

        # Check for expiry first
        self.check_expiry(option_type, bid_price, strategy_name)

        # Refresh positions list after expiry check
        positions = self.positions[strategy_name][option_type]

        # If no positions after expiry, try to open
        if not positions:
            self.open_position(option_type, ask_price, strategy_name)
            return

        # Check TP
        if self.check_take_profit(option_type, bid_price, strategy_name):
            # Immediately try to reopen
            self.open_position(option_type, ask_price, strategy_name)
            return

        # Check SL
        if self.check_stop_loss(option_type, bid_price, strategy_name):
            # Immediately try to reopen
            self.open_position(option_type, ask_price, strategy_name)
            return

        # For 5B: check if we should double down
        if strategy_name == 'strategy5B':
            if self.should_double_down(option_type, ask_price, strategy_name):
                self.open_position(option_type, ask_price, strategy_name)

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
            print(f"Error reading {filepath}: {e}")
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
            return False

        if current_period > self.current_period_start:
            self.save_period_results()
            self.current_period_start = current_period
            for strategy in self.strategies.keys():
                self.period_pnl[strategy] = {'CALL': 0.0, 'PUT': 0.0, 'TOTAL': 0.0}
            return True

        return False

    def save_period_results(self):
        """Save results for completed period"""
        if self.current_period_start is None:
            return

        period_end = self.current_period_start + timedelta(minutes=15)
        filename = f"period_{self.current_period_start.strftime('%Y%m%d_%H%M')}.json"
        output_file = self.output_dir / filename

        all_strategies_data = {}

        for strategy_name in self.strategies.keys():
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
                        'seconds_to_expiry': trade.seconds_to_expiry,
                        'expiry_time': trade.expiry_time
                    })
                else:
                    formatted_trades.append({
                        'timestamp': trade.timestamp,
                        'option_type': trade.option_type,
                        'action': trade.action,
                        'price': trade.price,
                        'reason': trade.reason,
                        'expiry_time': trade.expiry_time
                    })

            all_strategies_data[strategy_name] = {
                'strategy_config': self.strategies[strategy_name],
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
                    'avg_loss': avg_loss,
                    'profit_factor': abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else 0
                },
                'trades': formatted_trades
            }

        result = {
            'period_start': self.current_period_start.isoformat(),
            'period_end': period_end.isoformat(),
            'saved_at': datetime.now().isoformat(),
            'strategies': all_strategies_data
        }

        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n{'='*100}")
        print(f"📊 PERIOD COMPLETED: {self.current_period_start.strftime('%Y-%m-%d %H:%M')} - {period_end.strftime('%H:%M')}")
        print(f"💾 Saved to: {output_file}")
        for strategy_name in self.strategies.keys():
            pnl = self.period_pnl[strategy_name]
            overall = self.overall_pnl[strategy_name]
            print(f"   {strategy_name}: Period=${pnl['TOTAL']:+.4f} | Overall=${overall['TOTAL']:+.4f}")
        print(f"{'='*100}\n")

    def print_status(self):
        """Print current status"""
        print("\n" + "="*100)
        print(f"STATUS UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*100)

        call_ask_str = f"${self.current_call_ask:.4f}" if self.current_call_ask is not None else "None"
        call_bid_str = f"${self.current_call_bid:.4f}" if self.current_call_bid is not None else "None"
        put_ask_str = f"${self.current_put_ask:.4f}" if self.current_put_ask is not None else "None"
        put_bid_str = f"${self.current_put_bid:.4f}" if self.current_put_bid is not None else "None"

        print(f"CALL: Ask={call_ask_str} Bid={call_bid_str}")
        print(f"PUT:  Ask={put_ask_str} Bid={put_bid_str}")

        next_expiry = self.get_next_expiry()
        seconds_to_expiry = self.seconds_to_expiry(next_expiry)
        in_buffer = self.is_in_buffer_period()

        if in_buffer:
            buffer_status = " [END BUFFER]" if seconds_to_expiry <= 10 else " [START BUFFER]"
        else:
            buffer_status = ""

        print(f"Next Expiry: {next_expiry.strftime('%H:%M:%S')} (in {seconds_to_expiry:.0f}s){buffer_status}")

        if self.current_period_start:
            period_end = self.current_period_start + timedelta(minutes=15)
            print(f"Current Period: {self.current_period_start.strftime('%H:%M')} - {period_end.strftime('%H:%M')}")

        print("-"*100)

        for strategy_name in self.strategies.keys():
            period_pnl = self.period_pnl[strategy_name]
            overall_pnl = self.overall_pnl[strategy_name]

            # Count positions
            call_positions = len(self.positions[strategy_name]['CALL'])
            put_positions = len(self.positions[strategy_name]['PUT'])

            # Get trade counts
            if self.current_period_start:
                period_end = self.current_period_start + timedelta(minutes=15)
                period_trades = [
                    t for t in self.trades[strategy_name]
                    if t.action == 'SELL' and self.current_period_start <= datetime.fromisoformat(t.timestamp) < period_end
                ]
                tp = len([t for t in period_trades if t.reason == 'TP'])
                sl = len([t for t in period_trades if t.reason == 'SL'])
                exp = len([t for t in period_trades if t.reason == 'EXPIRY'])
                trades = len(period_trades)
            else:
                tp = sl = exp = trades = 0

            print(f"{strategy_name.upper()}: Period=${period_pnl['TOTAL']:+.4f} | Overall=${overall_pnl['TOTAL']:+.4f} | Positions: CALL={call_positions} PUT={put_positions} | Trades={trades} (TP:{tp} SL:{sl} EXP:{exp})")

        print("="*100 + "\n")

    def run(self, check_interval: float = 0.5, status_interval: int = 30):
        """Main monitoring loop"""
        print("Starting Binary Options Strategy Simulator with 15-Minute Expiries...")
        print(f"Monitoring: {self.call_file}")
        print(f"Monitoring: {self.put_file}")
        print(f"Output directory: {self.output_dir}")
        print(f"\n📋 STRATEGIES:")
        for name, config in self.strategies.items():
            print(f"   {name}: {config['description']}")
        print(f"\n⏰ Options expire at :00, :15, :30, :45 of every hour")
        print(f"⏰ Expiry evaluation when ≤5 seconds remaining")
        print(f"⏰ Price >= $0.50 → Expires at $1.00, < $0.50 → $0.00")
        print(f"🛡️  10-second buffer at START and END: NO NEW ENTRIES")
        print(f"💾 Results saved per period\n")
        print("Waiting for prices...\n")

        last_status_time = time.time()

        try:
            while True:
                self.check_period_change()

                if self.update_prices():
                    call_ask = self.current_call_ask if self.current_call_ask is not None else self.last_valid_call_ask
                    call_bid = self.current_call_bid if self.current_call_bid is not None else self.last_valid_call_bid
                    put_ask = self.current_put_ask if self.current_put_ask is not None else self.last_valid_put_ask
                    put_bid = self.current_put_bid if self.current_put_bid is not None else self.last_valid_put_bid

                    if all([call_ask is not None, call_bid is not None, put_ask is not None, put_bid is not None]):
                        for strategy_name in self.strategies.keys():
                            self.check_position('CALL', call_ask, call_bid, strategy_name)
                            self.check_position('PUT', put_ask, put_bid, strategy_name)

                if time.time() - last_status_time > status_interval:
                    if self.last_valid_call_ask is not None:
                        self.print_status()
                    last_status_time = time.time()

                time.sleep(check_interval)

        except KeyboardInterrupt:
            print("\n\nShutting down...")
            self.save_period_results()
            self.print_status()
            print("Goodbye!")


if __name__ == "__main__":
    simulator = StrategySimulator(
        call_file="/home/ubuntu/013_2025_polymarket/15M_BTC_CALL_rest.json",
        put_file="/home/ubuntu/013_2025_polymarket/15M_BTC_PUT_rest.json",
        output_dir="/home/ubuntu/013_2025_polymarket/bot016_react/simulators/simulation_results"
    )

    simulator.run(check_interval=0.5, status_interval=30)
