import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import copy

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

    def __init__(self, call_file: str, put_file: str, output_dir: str = "/home/ubuntu/013_2025_polymarket/bot016_react/simulators/simulation_results"):
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

        # Last valid prices (fallback when best_bid/best_ask is None)
        self.last_valid_call_ask = None
        self.last_valid_call_bid = None
        self.last_valid_put_ask = None
        self.last_valid_put_bid = None

        # Strategies configuration
        self.strategies = {
            'strategy1': {
                'tp': 0.10,
                'sl': 0.08,
                'tp_type': 'fixed',
                'sl_type': 'fixed',
                'max_buy_price': 0.99,
                'description': 'TP=$0.10, SL=$0.08'
            },
            'strategy2': {
                'tp': 0.10,
                'sl': 0.10,
                'tp_type': 'fixed',
                'sl_type': 'fixed',
                'max_buy_price': 0.99,
                'description': 'TP=$0.10, SL=$0.10'
            },
            'strategy3': {
                'tp': 0.12, 
                'sl': None,
                'tp_type': 'fixed',
                'sl_type': 'never',
                'max_buy_price': 0.99,
                'description': 'TP=$0.10, No SL'
            },
            'strategy3B': {
                'tp': 0.15,
                'sl': None,
                'tp_type': 'fixed',
                'sl_type': 'never',
                'max_buy_price': 0.99,
                'description': 'TP=$0.15, No SL'
            },
            'strategy3C': {
                'tp': 0.03,
                'sl': None,
                'tp_type': 'fixed',
                'sl_type': 'never',
                'max_buy_price': 0.99,
                'description': 'TP=$0.03, No SL'
            },
            'strategy4': {
                'tp': None,
                'sl': None,
                'tp_type': 'one_third_distance',
                'sl_type': 'one_third_distance',
                'min_distance': 0.05,
                'max_buy_price': 0.95,
                'description': 'TP/SL at 1/3 distance (min $0.05), max buy $0.95'
            }
        }

        # Positions and trades for each strategy
        self.positions: Dict[str, Dict[str, Position]] = {
            strategy: {'CALL': None, 'PUT': None}
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

        period_start = now.replace(minute=period_minute, second=0, microsecond=0)
        return period_start

    def get_next_expiry(self, from_time: datetime = None) -> datetime:
        """Calculate the next 15-minute expiry time (00, 15, 30, 45)"""
        if from_time is None:
            from_time = datetime.now()

        # Get current minute
        minute = from_time.minute

        # Calculate next expiry minute
        if minute < 15:
            next_minute = 15
        elif minute < 30:
            next_minute = 30
        elif minute < 45:
            next_minute = 45
        else:
            next_minute = 0

        # Create expiry time
        expiry = from_time.replace(second=0, microsecond=0)

        if next_minute == 0:
            # Next hour
            expiry = expiry + timedelta(hours=1)
            expiry = expiry.replace(minute=0)
        else:
            expiry = expiry.replace(minute=next_minute)

        return expiry

    def get_expiry_check_time(self, expiry: datetime) -> datetime:
        """Get the time to check for expiry (5 seconds before expiry)"""
        return expiry - timedelta(seconds=5)

    def seconds_to_expiry(self, expiry: datetime) -> float:
        """Calculate seconds remaining to expiry"""
        return (expiry - datetime.now()).total_seconds()

    def should_check_expiry(self, expiry: datetime) -> bool:
        """Check if we're at the expiry check time (5 seconds before expiry)"""
        now = datetime.now()
        seconds_remaining = (expiry - now).total_seconds()
        # Settle when there are 5 seconds or less remaining
        return 0 <= seconds_remaining <= 5

    def get_expiry_price(self, current_bid: float) -> float:
        """Determine expiry price based on last bid price"""
        if current_bid >= 0.5:
            return 1.00
        else:
            return 0.00

    def is_in_buffer_period(self) -> bool:
        """Check if we're in the 10-second buffer at start or end of period"""
        next_expiry = self.get_next_expiry()
        seconds_to_expiry = self.seconds_to_expiry(next_expiry)

        # End buffer: 10 seconds before expiry
        if seconds_to_expiry <= 10:
            return True

        # Start buffer: first 10 seconds after expiry (when 14:50-15:00 remaining)
        # For 15-minute periods, this means when seconds_to_expiry is between 890-900
        if 890 <= seconds_to_expiry <= 900:
            return True

        return False

    def calculate_tp_sl(self, entry_price: float, strategy_name: str) -> tuple:
        """Calculate take profit and stop loss based on strategy"""
        config = self.strategies[strategy_name]

        # Calculate Take Profit
        if config['tp_type'] == 'fixed':
            tp = entry_price + config['tp']
        elif config['tp_type'] == 'one_third_distance':
            distance = 1.00 - entry_price
            # Apply minimum distance constraint
            min_distance = config.get('min_distance', 0.0)
            if distance < min_distance:
                # If distance is less than minimum, use the minimum
                distance = min_distance
            tp = entry_price + (distance / 3)
        else:
            tp = entry_price + 0.10  # Default

        # Calculate Stop Loss
        if config['sl_type'] == 'fixed':
            sl = entry_price - config['sl']
        elif config['sl_type'] == 'never':
            sl = -1.00  # Never triggers
        elif config['sl_type'] == 'one_third_distance':
            distance = 1.00 - entry_price
            # Apply minimum distance constraint
            min_distance = config.get('min_distance', 0.0)
            if distance < min_distance:
                # If distance is less than minimum, use the minimum
                distance = min_distance
            sl = entry_price - (distance / 3)
        else:
            sl = -1.00  # Default: never

        # Ensure TP is achievable
        tp = min(1.00, tp)

        # If SL would be <= 0, disable it
        if sl <= 0.00:
            sl = -1.00  # Impossible to hit, effectively disabling SL
        else:
            sl = max(0.00, sl)

        return tp, sl

    def should_skip_entry(self, ask_price: float, strategy_name: str) -> bool:
        """Check if we should skip entry at this price"""
        config = self.strategies[strategy_name]
        max_buy_price = config.get('max_buy_price', 0.99)

        # Never buy at or above max_buy_price
        if ask_price >= max_buy_price:
            return True
        return False

    def open_position(self, option_type: str, ask_price: float, strategy_name: str):
        """Open a new position"""
        # Don't open positions in buffer period
        if self.is_in_buffer_period():
            return

        # Skip if price is too high
        if self.should_skip_entry(ask_price, strategy_name):
            config = self.strategies[strategy_name]
            max_buy = config.get('max_buy_price', 0.99)
            print(f"[{strategy_name}] Skipping {option_type} entry @ ${ask_price:.4f} (>= ${max_buy:.2f})")
            return

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

        self.positions[strategy_name][option_type] = position

        trade = Trade(
            timestamp=timestamp.isoformat(),
            option_type=option_type,
            action='BUY',
            price=ask_price,
            reason='OPEN',
            pnl=0.0,
            expiry_time=expiry.isoformat()
        )

        self.trades[strategy_name].append(trade)

        time_to_expiry = self.seconds_to_expiry(expiry)
        sl_str = f"${sl:.4f}" if sl >= 0 else "NEVER"
        print(f"[{strategy_name}] Opened {option_type} @ ${ask_price:.4f} | TP: ${tp:.4f} | SL: {sl_str} | Expiry: {expiry.strftime('%H:%M:%S')} ({time_to_expiry:.0f}s)")

    def close_position(self, option_type: str, exit_price: float, strategy_name: str, reason: str):
        """Close an existing position"""
        position = self.positions[strategy_name][option_type]

        if position is None:
            return

        timestamp = datetime.now()
        pnl = exit_price - position.entry_price
        expiry = datetime.fromisoformat(position.expiry_time)
        seconds_to_exp = self.seconds_to_expiry(expiry)

        trade = Trade(
            timestamp=timestamp.isoformat(),
            option_type=option_type,
            action='SELL',
            price=exit_price,
            reason=reason,
            pnl=pnl,
            expiry_time=position.expiry_time,
            open_price=position.entry_price,
            close_price=exit_price,
            seconds_to_expiry=seconds_to_exp
        )

        self.trades[strategy_name].append(trade)

        # Update period PNL
        self.period_pnl[strategy_name][option_type] += pnl
        self.period_pnl[strategy_name]['TOTAL'] += pnl

        # Update overall PNL
        self.overall_pnl[strategy_name][option_type] += pnl
        self.overall_pnl[strategy_name]['TOTAL'] += pnl

        print(f"[{strategy_name}] Closed {option_type} @ ${exit_price:.4f} | Reason: {reason} | PNL: ${pnl:+.4f} | Period PNL: ${self.period_pnl[strategy_name]['TOTAL']:+.4f} | Overall PNL: ${self.overall_pnl[strategy_name]['TOTAL']:+.4f}")

        # Clear position
        self.positions[strategy_name][option_type] = None

    def check_expiry(self, option_type: str, current_bid: float, strategy_name: str):
        """Check if position should be closed due to expiry"""
        position = self.positions[strategy_name][option_type]

        if position is None:
            return

        expiry = datetime.fromisoformat(position.expiry_time)

        # Check if we're at the expiry check time (5 seconds or less remaining)
        if self.should_check_expiry(expiry):
            expiry_price = self.get_expiry_price(current_bid)
            seconds_left = self.seconds_to_expiry(expiry)
            print(f"[{strategy_name}] ⏰ EXPIRY ({seconds_left:.1f}s left) {option_type}: Bid ${current_bid:.4f} → ${expiry_price:.2f}")
            self.close_position(option_type, expiry_price, strategy_name, 'EXPIRY')

    def check_position(self, option_type: str, ask_price: Optional[float], bid_price: Optional[float], strategy_name: str):
        """Check if position needs to be closed and reopen if needed"""
        # Skip if prices are None
        if ask_price is None or bid_price is None:
            return

        position = self.positions[strategy_name][option_type]

        # If no position, try to open one (if price is acceptable)
        if position is None:
            self.open_position(option_type, ask_price, strategy_name)
            return

        # First check for expiry
        self.check_expiry(option_type, bid_price, strategy_name)

        # If position was closed due to expiry, try to reopen (if price is acceptable)
        if self.positions[strategy_name][option_type] is None:
            self.open_position(option_type, ask_price, strategy_name)
            return

        # Check Take Profit (sell at bid_price when bid >= TP)
        if bid_price >= position.take_profit:
            self.close_position(option_type, bid_price, strategy_name, 'TP')
            # Immediately try to reopen at ask (if price is acceptable)
            self.open_position(option_type, ask_price, strategy_name)
            return

        # Check Stop Loss (only if SL is valid/enabled - i.e., >= 0)
        if position.stop_loss >= 0.00 and bid_price <= position.stop_loss:
            self.close_position(option_type, bid_price, strategy_name, 'SL')
            # Immediately try to reopen at ask (if price is acceptable)
            self.open_position(option_type, ask_price, strategy_name)
            return

    def read_price_file(self, filepath: Path) -> Optional[Dict[str, float]]:
        """Read price data from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Extract ask and bid prices - handle None/null values
            ask_price = data.get('best_ask')
            bid_price = data.get('best_bid')

            # Convert to float if not None
            if ask_price is not None:
                ask_price = float(ask_price)
            if bid_price is not None:
                bid_price = float(bid_price)

            return {
                'ask': ask_price,
                'bid': bid_price,
                'timestamp': data.get('timestamp'),
                'timestamp_readable': data.get('timestamp_readable')
            }
        except (json.JSONDecodeError, FileNotFoundError, KeyError, ValueError) as e:
            print(f"Error reading {filepath}: {e}")
            return None

    def update_prices(self):
        """Check for file updates and update prices"""
        updated = False

        # Check CALL file
        if self.call_file.exists():
            mtime = self.call_file.stat().st_mtime
            if mtime > self.last_call_mtime:
                prices = self.read_price_file(self.call_file)
                if prices:
                    # Update current prices (can be None)
                    self.current_call_ask = prices['ask']
                    self.current_call_bid = prices['bid']

                    # Store last valid prices for fallback
                    if prices['ask'] is not None:
                        self.last_valid_call_ask = prices['ask']
                    if prices['bid'] is not None:
                        self.last_valid_call_bid = prices['bid']

                    self.last_call_mtime = mtime
                    updated = True

        # Check PUT file
        if self.put_file.exists():
            mtime = self.put_file.stat().st_mtime
            if mtime > self.last_put_mtime:
                prices = self.read_price_file(self.put_file)
                if prices:
                    # Update current prices (can be None)
                    self.current_put_ask = prices['ask']
                    self.current_put_bid = prices['bid']

                    # Store last valid prices for fallback
                    if prices['ask'] is not None:
                        self.last_valid_put_ask = prices['ask']
                    if prices['bid'] is not None:
                        self.last_valid_put_bid = prices['bid']

                    self.last_put_mtime = mtime
                    updated = True

        return updated

    def check_period_change(self):
        """Check if we've moved to a new period and save results if needed"""
        current_period = self.get_current_period_start()

        if self.current_period_start is None:
            self.current_period_start = current_period
            return False

        if current_period > self.current_period_start:
            # New period started - save results for previous period
            self.save_period_results()

            # Reset for new period
            self.current_period_start = current_period
            for strategy in self.strategies.keys():
                self.period_pnl[strategy] = {'CALL': 0.0, 'PUT': 0.0, 'TOTAL': 0.0}

            return True

        return False

    def save_period_results(self):
        """Save results for the completed period - ONE FILE for ALL strategies"""
        if self.current_period_start is None:
            return

        period_end = self.current_period_start + timedelta(minutes=15)

        # Create filename based on period
        filename = f"period_{self.current_period_start.strftime('%Y%m%d_%H%M')}.json"
        output_file = self.output_dir / filename

        # Compile all strategies data
        all_strategies_data = {}

        for strategy_name in self.strategies.keys():
            # Get trades for this period
            period_trades = [
                t for t in self.trades[strategy_name]
                if self.current_period_start <= datetime.fromisoformat(t.timestamp) < period_end
            ]

            # Filter only SELL trades for statistics
            sell_trades = [t for t in period_trades if t.action == 'SELL']

            # Calculate statistics
            total_trades = len(sell_trades)
            tp_trades = len([t for t in sell_trades if t.reason == 'TP'])
            sl_trades = len([t for t in sell_trades if t.reason == 'SL'])
            expiry_trades = len([t for t in sell_trades if t.reason == 'EXPIRY'])

            winning_trades = len([t for t in sell_trades if t.pnl > 0])
            losing_trades = len([t for t in sell_trades if t.pnl < 0])

            avg_win = sum([t.pnl for t in sell_trades if t.pnl > 0]) / winning_trades if winning_trades > 0 else 0
            avg_loss = sum([t.pnl for t in sell_trades if t.pnl < 0]) / losing_trades if losing_trades > 0 else 0

            # Format trades with complete information
            formatted_trades = []
            for trade in period_trades:
                trade_dict = asdict(trade)
                if trade.action == 'SELL':
                    # Include all relevant fields for closed trades
                    formatted_trades.append({
                        'timestamp': trade.timestamp,
                        'option_type': trade.option_type,
                        'action': trade.action,
                        'open_price': trade.open_price,
                        'close_price': trade.close_price,
                        'pnl': trade.pnl,
                        'reason': trade.reason,
                        'seconds_to_expiry': trade.seconds_to_expiry,
                        'expiry_time': trade.expiry_time
                    })
                else:
                    # For BUY trades, just basic info
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
                'period_pnl': {
                    'CALL': self.period_pnl[strategy_name]['CALL'],
                    'PUT': self.period_pnl[strategy_name]['PUT'],
                    'TOTAL': self.period_pnl[strategy_name]['TOTAL']
                },
                'overall_pnl': {
                    'CALL': self.overall_pnl[strategy_name]['CALL'],
                    'PUT': self.overall_pnl[strategy_name]['PUT'],
                    'TOTAL': self.overall_pnl[strategy_name]['TOTAL']
                },
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

        # Create final output with metadata
        result = {
            'period_start': self.current_period_start.isoformat(),
            'period_end': period_end.isoformat(),
            'saved_at': datetime.now().isoformat(),
            'strategies': all_strategies_data
        }

        # Save to file
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n{'='*100}")
        print(f"📊 PERIOD COMPLETED: {self.current_period_start.strftime('%Y-%m-%d %H:%M')} - {period_end.strftime('%H:%M')}")
        print(f"💾 Saved results to: {output_file}")
        for strategy_name in self.strategies.keys():
            pnl = self.period_pnl[strategy_name]
            overall = self.overall_pnl[strategy_name]
            print(f"   {strategy_name}: Period PNL=${pnl['TOTAL']:+.4f} | Overall PNL=${overall['TOTAL']:+.4f}")
        print(f"{'='*100}\n")

    def print_status(self):
        """Print current status of all strategies"""
        print("\n" + "="*100)
        print(f"STATUS UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*100)

        # Show current prices (handle None values)
        call_ask_str = f"${self.current_call_ask:.4f}" if self.current_call_ask is not None else "None"
        call_bid_str = f"${self.current_call_bid:.4f}" if self.current_call_bid is not None else "None"
        put_ask_str = f"${self.current_put_ask:.4f}" if self.current_put_ask is not None else "None"
        put_bid_str = f"${self.current_put_bid:.4f}" if self.current_put_bid is not None else "None"

        print(f"CALL: Ask={call_ask_str} Bid={call_bid_str}")
        print(f"PUT:  Ask={put_ask_str} Bid={put_bid_str}")

        # Show next expiry
        next_expiry = self.get_next_expiry()
        seconds_to_expiry = self.seconds_to_expiry(next_expiry)
        in_buffer = self.is_in_buffer_period()

        if in_buffer:
            if seconds_to_expiry <= 10:
                buffer_status = " [END BUFFER - NO NEW ENTRIES]"
            else:
                buffer_status = " [START BUFFER - NO NEW ENTRIES]"
        else:
            buffer_status = ""

        print(f"Next Expiry: {next_expiry.strftime('%H:%M:%S')} (in {seconds_to_expiry:.0f}s){buffer_status}")

        # Show current period
        if self.current_period_start:
            period_end = self.current_period_start + timedelta(minutes=15)
            print(f"Current Period: {self.current_period_start.strftime('%H:%M')} - {period_end.strftime('%H:%M')}")

        print("-"*100)

        for strategy_name in self.strategies.keys():
            period_pnl = self.period_pnl[strategy_name]
            overall_pnl = self.overall_pnl[strategy_name]

            # Get trade counts for current period
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

            print(f"{strategy_name.upper()}: Period=${period_pnl['TOTAL']:+.4f} | Overall=${overall_pnl['TOTAL']:+.4f} | Trades={trades} (TP:{tp} SL:{sl} EXP:{exp})")

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
        print(f"⏰ Expiry evaluation occurs when 5 seconds or less remaining")
        print(f"⏰ Price >= $0.50 → Expires at $1.00")
        print(f"⏰ Price < $0.50 → Expires at $0.00")
        print(f"🛡️  10-second buffer at START of each period: NO NEW ENTRIES")
        print(f"🛡️  10-second buffer at END of each period: NO NEW ENTRIES")
        print(f"💾 Results saved per period in ONE file containing ALL strategies\n")
        print("⚠️  Handling None/null prices gracefully\n")
        print("Waiting for initial prices...\n")

        last_status_time = time.time()

        try:
            while True:
                # Check for period change
                self.check_period_change()

                if self.update_prices():
                    # Use current prices (or last valid if current is None)
                    call_ask = self.current_call_ask if self.current_call_ask is not None else self.last_valid_call_ask
                    call_bid = self.current_call_bid if self.current_call_bid is not None else self.last_valid_call_bid
                    put_ask = self.current_put_ask if self.current_put_ask is not None else self.last_valid_put_ask
                    put_bid = self.current_put_bid if self.current_put_bid is not None else self.last_valid_put_bid

                    # Process all strategies if we have valid prices
                    if all([call_ask is not None, call_bid is not None, put_ask is not None, put_bid is not None]):
                        for strategy_name in self.strategies.keys():
                            self.check_position('CALL', call_ask, call_bid, strategy_name)
                            self.check_position('PUT', put_ask, put_bid, strategy_name)

                # Print status periodically
                if time.time() - last_status_time > status_interval:
                    if self.last_valid_call_ask is not None:
                        self.print_status()
                    last_status_time = time.time()

                time.sleep(check_interval)

        except KeyboardInterrupt:
            print("\n\nShutting down simulator...")
            self.save_period_results()
            self.print_status()
            print("Final results saved. Goodbye!")


if __name__ == "__main__":
    simulator = StrategySimulator(
        call_file="/home/ubuntu/013_2025_polymarket/15M_BTC_CALL_rest.json",
        put_file="/home/ubuntu/013_2025_polymarket/15M_BTC_PUT_rest.json",
        output_dir="/home/ubuntu/013_2025_polymarket/bot016_react/simulators/simulation_results"
    )

    # Run with 0.5 second check interval and status every 30 seconds
    simulator.run(check_interval=0.5, status_interval=30)
