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

class VerboseStrategy3C:
    """
    Strategy 3C: TP=$0.03, No SL
    Verbose logging of all operations
    """

    def __init__(self, call_file: str, put_file: str, output_dir: str = "/home/ubuntu/013_2025_polymarket/bot016_react/simulators/strategy3c_verbose"):
        self.call_file = Path(call_file)
        self.put_file = Path(put_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Strategy configuration
        self.TAKE_PROFIT = 0.07
        self.MAX_BUY_PRICE = 0.99

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

        # Positions
        self.call_position: Optional[Position] = None
        self.put_position: Optional[Position] = None

        # Trades and PNL
        self.trades: List[Trade] = []
        self.realized_pnl = {'CALL': 0.0, 'PUT': 0.0, 'TOTAL': 0.0}

        # Period tracking
        self.current_period_start = None
        self.period_pnl = {'CALL': 0.0, 'PUT': 0.0, 'TOTAL': 0.0}

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

    def get_expiry_check_time(self, expiry: datetime) -> datetime:
        """Get the time to check for expiry (5 seconds before)"""
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

    def calculate_unrealized_pnl(self, option_type: str, current_bid: float) -> Optional[float]:
        """Calculate unrealized PNL for a position"""
        position = self.call_position if option_type == 'CALL' else self.put_position
        if position is None:
            return None
        return current_bid - position.entry_price

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

    def open_position(self, option_type: str, ask_price: float):
        """Open a new position"""
        # Don't open positions in buffer period
        if self.is_in_buffer_period():
            next_expiry = self.get_next_expiry()
            seconds_left = self.seconds_to_expiry(next_expiry)
            if seconds_left <= 10:
                print(f"\n⏸️  END BUFFER: Not opening {option_type} ({seconds_left:.0f}s to expiry)")
            else:
                print(f"\n⏸️  START BUFFER: Not opening {option_type} (period just started)")
            return

        # Skip if price too high
        if ask_price >= self.MAX_BUY_PRICE:
            print(f"\n⚠️  SKIPPED {option_type} BUY @ ${ask_price:.4f} (>= ${self.MAX_BUY_PRICE:.2f})")
            return

        timestamp = datetime.now()
        expiry = self.get_next_expiry(timestamp)

        # Calculate TP (no SL for this strategy)
        tp = ask_price + self.TAKE_PROFIT
        tp = min(1.00, tp)  # Cap at $1.00

        position = Position(
            option_type=option_type,
            entry_price=ask_price,
            entry_time=timestamp.isoformat(),
            expiry_time=expiry.isoformat(),
            take_profit=tp,
            stop_loss=-1.0  # No stop loss
        )

        if option_type == 'CALL':
            self.call_position = position
        else:
            self.put_position = position

        trade = Trade(
            timestamp=timestamp.isoformat(),
            option_type=option_type,
            action='BUY',
            price=ask_price,
            reason='OPEN',
            pnl=0.0,
            expiry_time=expiry.isoformat()
        )

        self.trades.append(trade)

        time_to_expiry = self.seconds_to_expiry(expiry)
        print(f"\n{'='*80}")
        print(f"🟢 BOUGHT {option_type}")
        print(f"{'='*80}")
        print(f"   Time: {timestamp.strftime('%H:%M:%S')}")
        print(f"   Entry Price: ${ask_price:.4f}")
        print(f"   Take Profit: ${tp:.4f} (${self.TAKE_PROFIT:.2f} profit)")
        print(f"   Stop Loss: NONE")
        print(f"   Expiry: {expiry.strftime('%H:%M:%S')} ({time_to_expiry:.0f}s away)")
        print(f"{'='*80}")

    def close_position(self, option_type: str, exit_price: float, reason: str):
        """Close an existing position"""
        position = self.call_position if option_type == 'CALL' else self.put_position

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

        self.trades.append(trade)

        # Update PNL
        self.realized_pnl[option_type] += pnl
        self.realized_pnl['TOTAL'] += pnl
        self.period_pnl[option_type] += pnl
        self.period_pnl['TOTAL'] += pnl

        # Clear position
        if option_type == 'CALL':
            self.call_position = None
        else:
            self.put_position = None

        # Verbose output
        print(f"\n{'='*80}")
        print(f"🔴 SOLD {option_type} - {reason}")
        print(f"{'='*80}")
        print(f"   Time: {timestamp.strftime('%H:%M:%S')}")
        print(f"   Entry Price: ${position.entry_price:.4f}")
        print(f"   Exit Price: ${exit_price:.4f}")
        print(f"   PNL: ${pnl:+.4f}")
        print(f"   Time to Expiry: {seconds_to_exp:.0f}s")
        print(f"   ---")
        print(f"   Realized PNL (This Period):")
        print(f"      CALL: ${self.period_pnl['CALL']:+.4f}")
        print(f"      PUT:  ${self.period_pnl['PUT']:+.4f}")
        print(f"      TOTAL: ${self.period_pnl['TOTAL']:+.4f}")
        print(f"   ---")
        print(f"   Realized PNL (Overall):")
        print(f"      CALL: ${self.realized_pnl['CALL']:+.4f}")
        print(f"      PUT:  ${self.realized_pnl['PUT']:+.4f}")
        print(f"      TOTAL: ${self.realized_pnl['TOTAL']:+.4f}")
        print(f"{'='*80}")

    def check_expiry(self, option_type: str, current_bid: float):
        """Check if position should be closed due to expiry"""
        position = self.call_position if option_type == 'CALL' else self.put_position

        if position is None:
            return

        expiry = datetime.fromisoformat(position.expiry_time)

        if self.should_check_expiry(expiry):
            expiry_price = self.get_expiry_price(current_bid)
            seconds_left = self.seconds_to_expiry(expiry)
            print(f"\n⏰ EXPIRY SETTLEMENT ({seconds_left:.1f}s remaining)")
            print(f"   {option_type}: Bid ${current_bid:.4f} → {'$1.00 (WIN)' if expiry_price == 1.00 else '$0.00 (LOSS)'}")
            self.close_position(option_type, expiry_price, 'EXPIRY')

    def check_position(self, option_type: str, ask_price: float, bid_price: float):
        """Check position and manage trades"""
        position = self.call_position if option_type == 'CALL' else self.put_position

        # If no position, open one
        if position is None:
            self.open_position(option_type, ask_price)
            return

        # Check for expiry
        self.check_expiry(option_type, bid_price)

        # If position was closed, reopen
        position = self.call_position if option_type == 'CALL' else self.put_position
        if position is None:
            self.open_position(option_type, ask_price)
            return

        # Check Take Profit
        if bid_price >= position.take_profit:
            self.close_position(option_type, bid_price, 'TP')
            # Immediately reopen
            self.open_position(option_type, ask_price)
            return

    def print_status(self):
        """Print detailed status"""
        print(f"\n{'#'*80}")
        print(f"📊 STATUS UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*80}")

        # Current prices
        print(f"\n💰 CURRENT PRICES:")
        call_ask_str = f"${self.current_call_ask:.4f}" if self.current_call_ask is not None else "None"
        call_bid_str = f"${self.current_call_bid:.4f}" if self.current_call_bid is not None else "None"
        put_ask_str = f"${self.current_put_ask:.4f}" if self.current_put_ask is not None else "None"
        put_bid_str = f"${self.current_put_bid:.4f}" if self.current_put_bid is not None else "None"

        print(f"   CALL: Ask={call_ask_str} | Bid={call_bid_str}")
        print(f"   PUT:  Ask={put_ask_str} | Bid={put_bid_str}")

        # Expiry info
        next_expiry = self.get_next_expiry()
        seconds_to_expiry = self.seconds_to_expiry(next_expiry)
        in_buffer = self.is_in_buffer_period()

        print(f"\n⏰ EXPIRY:")
        print(f"   Next Expiry: {next_expiry.strftime('%H:%M:%S')} (in {seconds_to_expiry:.0f}s)")
        if in_buffer:
            if seconds_to_expiry <= 10:
                print(f"   🛡️  END BUFFER - No new positions, only closings!")
            else:
                print(f"   🛡️  START BUFFER - Waiting for stable prices!")
        else:
            print(f"   ✅ Trading active")

        # Period info
        if self.current_period_start:
            period_end = self.current_period_start + timedelta(minutes=15)
            print(f"   Current Period: {self.current_period_start.strftime('%H:%M')} - {period_end.strftime('%H:%M')}")

        # CALL Position
        print(f"\n📈 CALL POSITION:")
        if self.call_position:
            unrealized = self.calculate_unrealized_pnl('CALL', self.current_call_bid if self.current_call_bid else 0)
            expiry = datetime.fromisoformat(self.call_position.expiry_time)
            time_left = self.seconds_to_expiry(expiry)
            print(f"   ✅ OPEN")
            print(f"   Entry: ${self.call_position.entry_price:.4f}")
            print(f"   Current Bid: ${self.current_call_bid:.4f}" if self.current_call_bid else "   Current Bid: None")
            print(f"   Take Profit: ${self.call_position.take_profit:.4f}")
            print(f"   Unrealized PNL: ${unrealized:+.4f}" if unrealized is not None else "   Unrealized PNL: N/A")
            print(f"   Time to Expiry: {time_left:.0f}s")
        else:
            print(f"   ❌ NO POSITION")

        # PUT Position
        print(f"\n📉 PUT POSITION:")
        if self.put_position:
            unrealized = self.calculate_unrealized_pnl('PUT', self.current_put_bid if self.current_put_bid else 0)
            expiry = datetime.fromisoformat(self.put_position.expiry_time)
            time_left = self.seconds_to_expiry(expiry)
            print(f"   ✅ OPEN")
            print(f"   Entry: ${self.put_position.entry_price:.4f}")
            print(f"   Current Bid: ${self.current_put_bid:.4f}" if self.current_put_bid else "   Current Bid: None")
            print(f"   Take Profit: ${self.put_position.take_profit:.4f}")
            print(f"   Unrealized PNL: ${unrealized:+.4f}" if unrealized is not None else "   Unrealized PNL: N/A")
            print(f"   Time to Expiry: {time_left:.0f}s")
        else:
            print(f"   ❌ NO POSITION")

        # Total unrealized
        total_unrealized = 0.0
        if self.call_position and self.current_call_bid:
            total_unrealized += self.calculate_unrealized_pnl('CALL', self.current_call_bid)
        if self.put_position and self.current_put_bid:
            total_unrealized += self.calculate_unrealized_pnl('PUT', self.current_put_bid)

        # PNL Summary
        print(f"\n💵 PNL SUMMARY:")
        print(f"   Realized (This Period):")
        print(f"      CALL:  ${self.period_pnl['CALL']:+.4f}")
        print(f"      PUT:   ${self.period_pnl['PUT']:+.4f}")
        print(f"      TOTAL: ${self.period_pnl['TOTAL']:+.4f}")
        print(f"   Realized (Overall):")
        print(f"      CALL:  ${self.realized_pnl['CALL']:+.4f}")
        print(f"      PUT:   ${self.realized_pnl['PUT']:+.4f}")
        print(f"      TOTAL: ${self.realized_pnl['TOTAL']:+.4f}")
        print(f"   Unrealized: ${total_unrealized:+.4f}")
        print(f"   Combined (Realized + Unrealized): ${self.realized_pnl['TOTAL'] + total_unrealized:+.4f}")

        # Trade count
        period_trades = 0
        if self.current_period_start:
            period_end = self.current_period_start + timedelta(minutes=15)
            period_trades = len([t for t in self.trades if t.action == 'SELL' and
                               self.current_period_start <= datetime.fromisoformat(t.timestamp) < period_end])

        total_trades = len([t for t in self.trades if t.action == 'SELL'])
        print(f"\n📊 TRADES:")
        print(f"   This Period: {period_trades}")
        print(f"   Overall: {total_trades}")

        print(f"{'#'*80}\n")

    def read_price_file(self, filepath: Path) -> Optional[Dict[str, Any]]:
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

            return {
                'ask': ask_price,
                'bid': bid_price,
                'timestamp': data.get('timestamp'),
                'timestamp_readable': data.get('timestamp_readable')
            }
        except Exception as e:
            print(f"❌ Error reading {filepath}: {e}")
            return None

    def update_prices(self):
        """Update prices from files"""
        updated = False

        # Check CALL file
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

        # Check PUT file
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
            print(f"\n🔄 NEW PERIOD STARTED: {current_period.strftime('%H:%M')}")

            # Reset period PNL
            self.current_period_start = current_period
            self.period_pnl = {'CALL': 0.0, 'PUT': 0.0, 'TOTAL': 0.0}

            return True

        return False

    def run(self, check_interval: float = 0.5, status_interval: int = 30):
        """Main monitoring loop"""
        print("\n" + "="*80)
        print("🚀 STARTING STRATEGY 3C VERBOSE SIMULATOR")
        print("="*80)
        print(f"Strategy: TP=${self.TAKE_PROFIT:.2f}, No SL")
        print(f"Max Buy Price: ${self.MAX_BUY_PRICE:.2f}")
        print(f"🛡️  10-second buffer at START of each period: NO NEW ENTRIES")
        print(f"🛡️  10-second buffer at END of each period: NO NEW ENTRIES")
        print(f"Monitoring CALL: {self.call_file}")
        print(f"Monitoring PUT: {self.put_file}")
        print(f"Output: {self.output_dir}")
        print("="*80 + "\n")
        print("⏳ Waiting for initial prices...\n")

        last_status_time = time.time()

        try:
            while True:
                # Check for period change
                self.check_period_change()

                if self.update_prices():
                    # Use current or last valid prices
                    call_ask = self.current_call_ask if self.current_call_ask is not None else self.last_valid_call_ask
                    call_bid = self.current_call_bid if self.current_call_bid is not None else self.last_valid_call_bid
                    put_ask = self.current_put_ask if self.current_put_ask is not None else self.last_valid_put_ask
                    put_bid = self.current_put_bid if self.current_put_bid is not None else self.last_valid_put_bid

                    # Process positions if we have valid prices
                    if all([call_ask is not None, call_bid is not None, put_ask is not None, put_bid is not None]):
                        self.check_position('CALL', call_ask, call_bid)
                        self.check_position('PUT', put_ask, put_bid)

                # Print status periodically
                if time.time() - last_status_time > status_interval:
                    if self.last_valid_call_ask is not None:
                        self.print_status()
                    last_status_time = time.time()

                time.sleep(check_interval)

        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down simulator...")
            self.print_status()
            print("✅ Goodbye!\n")


if __name__ == "__main__":
    simulator = VerboseStrategy3C(
        call_file="/home/ubuntu/013_2025_polymarket/15M_BTC_CALL_rest.json",
        put_file="/home/ubuntu/013_2025_polymarket/15M_BTC_PUT_rest.json",
        output_dir="/home/ubuntu/013_2025_polymarket/bot016_react/simulators/strategy3c_verbose"
    )

    simulator.run(check_interval=0.5, status_interval=30)
