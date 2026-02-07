import json
import time
import csv
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

class SimplePriceRecorder:
    """Records PUT/CALL prices every second to CSV"""

    def __init__(self, call_file: str, put_file: str, output_dir: str = "/home/ubuntu/013_2025_polymarket/bot016_react/simulators/csv_data"):
        self.call_file = Path(call_file)
        self.put_file = Path(put_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track last modified times
        self.last_call_mtime = 0
        self.last_put_mtime = 0

        # Current period tracking
        self.current_period_start = None
        self.csv_writer = None
        self.csv_file = None

        # Cache for current prices
        self.current_call_bid = None
        self.current_call_ask = None
        self.current_put_bid = None
        self.current_put_ask = None

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
            print(f"Error reading {filepath}: {e}")
            return None

    def open_csv_for_period(self):
        """Open a new CSV file for the current period"""
        if self.csv_file:
            self.csv_file.close()

        # Create filename with period timestamp
        filename = f"prices_{self.current_period_start.strftime('%Y%m%d_%H%M')}.csv"
        filepath = self.output_dir / filename

        self.csv_file = open(filepath, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write header
        self.csv_writer.writerow([
            'timestamp',
            'timestamp_ms',
            'call_bid',
            'call_ask',
            'call_mid',
            'put_bid',
            'put_ask',
            'put_mid',
            'seconds_to_expiry'
        ])
        
        print(f"📝 Opened new CSV file: {filepath}")

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

    def record_prices(self):
        """Record current prices to CSV"""
        if None in [self.current_call_bid, self.current_call_ask,
                    self.current_put_bid, self.current_put_ask]:
            return

        if not self.csv_writer:
            return

        now = datetime.now()
        expiry = self.get_next_expiry(now)
        seconds_to_expiry = (expiry - now).total_seconds()

        call_mid = (self.current_call_bid + self.current_call_ask) / 2
        put_mid = (self.current_put_bid + self.current_put_ask) / 2

        # Write row to CSV
        self.csv_writer.writerow([
            now.isoformat(),
            int(now.timestamp() * 1000),
            f"{self.current_call_bid:.6f}",
            f"{self.current_call_ask:.6f}",
            f"{call_mid:.6f}",
            f"{self.current_put_bid:.6f}",
            f"{self.current_put_ask:.6f}",
            f"{put_mid:.6f}",
            f"{seconds_to_expiry:.1f}"
        ])
        
        # Flush to disk
        self.csv_file.flush()

    def check_period_change(self):
        """Check if we've moved to a new period"""
        current_period = self.get_current_period_start()

        if self.current_period_start is None:
            self.current_period_start = current_period
            self.open_csv_for_period()
            return False

        if current_period > self.current_period_start:
            # Close old CSV
            if self.csv_file:
                self.csv_file.close()
                print(f"✅ Closed CSV for period: {self.current_period_start.strftime('%Y-%m-%d %H:%M')}")

            # Open new CSV for new period
            self.current_period_start = current_period
            self.open_csv_for_period()

            return True

        return False

    def update_prices(self):
        """Update prices from files"""
        updated = False

        # Check CALL file
        if self.call_file.exists():
            mtime = self.call_file.stat().st_mtime
            if mtime > self.last_call_mtime:
                prices = self.read_price_file(self.call_file)
                if prices and prices['ask'] is not None and prices['bid'] is not None:
                    self.current_call_ask = prices['ask']
                    self.current_call_bid = prices['bid']
                    self.last_call_mtime = mtime
                    updated = True

        # Check PUT file
        if self.put_file.exists():
            mtime = self.put_file.stat().st_mtime
            if mtime > self.last_put_mtime:
                prices = self.read_price_file(self.put_file)
                if prices and prices['ask'] is not None and prices['bid'] is not None:
                    self.current_put_ask = prices['ask']
                    self.current_put_bid = prices['bid']
                    self.last_put_mtime = mtime
                    updated = True

        return updated

    def run(self, check_interval: float = 1.0):
        """Main recording loop"""
        print("🎯 Starting Simple CSV Price Recorder...")
        print(f"📊 Monitoring: {self.call_file}")
        print(f"📊 Monitoring: {self.put_file}")
        print(f"💾 Output directory: {self.output_dir}")
        print(f"⏱️  Recording interval: {check_interval} second(s)")
        print(f"📝 Format: One CSV per 15-minute period\n")

        try:
            while True:
                self.check_period_change()

                if self.update_prices():
                    self.record_prices()

                time.sleep(check_interval)

        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down recorder...")
            if self.csv_file:
                self.csv_file.close()
            print("✅ CSV file closed. Goodbye!")


if __name__ == "__main__":
    recorder = SimplePriceRecorder(
        call_file="/home/ubuntu/013_2025_polymarket/15M_BTC_CALL_rest.json",
        put_file="/home/ubuntu/013_2025_polymarket/15M_BTC_PUT_rest.json",
        output_dir="/home/ubuntu/013_2025_polymarket/bot016_react/simulators/csv_data"
    )

    recorder.run(check_interval=1.0)
