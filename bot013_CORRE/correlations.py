import json
import time
from datetime import datetime, timedelta
from collections import deque
import numpy as np
from pathlib import Path

class CorrelationMonitor:
    def __init__(self):
        self.data_dir = Path("/home/ubuntu/013_2025_polymarket")
        self.price_files = {
            'BTC': self.data_dir / "bybit_btc_price.json",
            'ETH': self.data_dir / "bybit_eth_price.json",
            'SOL': self.data_dir / "bybit_sol_price.json",
            'XRP': self.data_dir / "bybit_xrp_price.json"
        }
        self.output_file = self.data_dir / "bybit_correlations.json"

        # Store price history with timestamps (60 minutes = 3600 seconds)
        self.price_history = {coin: deque(maxlen=3600) for coin in self.price_files.keys()}
        self.window_seconds = 3600  # 60 minutes

    def read_latest_price(self, filepath):
        """Read the latest price from a price file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                # Extract price and timestamp from the JSON structure
                price = float(data['price'])
                timestamp_ms = int(data['timestamp'])
                # Convert millisecond timestamp to datetime
                timestamp = datetime.fromtimestamp(timestamp_ms / 1000.0)
                return price, timestamp
        except (FileNotFoundError, json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Error reading {filepath}: {e}")
            return None, None

    def collect_prices(self):
        """Collect current prices for all coins"""
        current_time = datetime.now()
        prices = {}

        for coin, filepath in self.price_files.items():
            price, file_timestamp = self.read_latest_price(filepath)
            if price is not None:
                prices[coin] = price
                # Use current time for consistency in correlation calculation
                self.price_history[coin].append({
                    'timestamp': current_time,
                    'price': price,
                    'file_timestamp': file_timestamp
                })

        return prices

    def clean_old_data(self):
        """Remove data older than 60 minutes"""
        cutoff_time = datetime.now() - timedelta(seconds=self.window_seconds)

        for coin in self.price_history.keys():
            # Remove old entries
            while self.price_history[coin] and self.price_history[coin][0]['timestamp'] < cutoff_time:
                self.price_history[coin].popleft()

    def calculate_correlations(self):
        """Calculate correlations between all pairs"""
        correlations = {}
        coins = list(self.price_files.keys())

        # Check if we have enough data
        min_data_points = min(len(self.price_history[coin]) for coin in coins)

        if min_data_points < 2:
            print(f"Not enough data points yet: {min_data_points}")
            return None

        # Extract price arrays for each coin (aligned by index)
        price_arrays = {}
        for coin in coins:
            if len(self.price_history[coin]) > 0:
                price_arrays[coin] = np.array([entry['price'] for entry in self.price_history[coin]])
            else:
                price_arrays[coin] = np.array([])

        # Calculate correlations for all pairs
        for i, coin1 in enumerate(coins):
            for coin2 in coins[i+1:]:
                pair_name = f"{coin1}_{coin2}"

                # Get the minimum length to align arrays
                min_len = min(len(price_arrays[coin1]), len(price_arrays[coin2]))

                if min_len < 2:
                    correlations[pair_name] = {
                        'correlation': None,
                        'data_points': min_len,
                        'error': 'Insufficient data'
                    }
                    continue

                # Use the last min_len points from each array
                prices1 = price_arrays[coin1][-min_len:]
                prices2 = price_arrays[coin2][-min_len:]

                # Calculate correlation
                try:
                    corr_matrix = np.corrcoef(prices1, prices2)
                    correlation = corr_matrix[0, 1]

                    correlations[pair_name] = {
                        'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
                        'data_points': min_len,
                        'coin1': coin1,
                        'coin2': coin2
                    }
                except Exception as e:
                    correlations[pair_name] = {
                        'correlation': None,
                        'data_points': min_len,
                        'error': str(e)
                    }

        return correlations

    def save_correlations(self, correlations):
        """Save correlations to JSON file"""
        if correlations is None:
            return

        output_data = {
            'timestamp': datetime.now().isoformat(),
            'timestamp_ms': int(datetime.now().timestamp() * 1000),
            'window_minutes': self.window_seconds / 60,
            'data_points': {coin: len(self.price_history[coin]) for coin in self.price_files.keys()},
            'correlations': correlations
        }

        try:
            with open(self.output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
        except Exception as e:
            print(f"Error saving correlations: {e}")

    def run(self):
        """Main loop - run every second"""
        print("Starting correlation monitor...")
        print(f"Monitoring: {', '.join(self.price_files.keys())}")
        print(f"Window: {self.window_seconds / 60} minutes")
        print(f"Output: {self.output_file}")
        print("-" * 50)

        iteration = 0
        while True:
            try:
                # Collect current prices
                prices = self.collect_prices()

                # Clean old data
                self.clean_old_data()

                # Calculate correlations
                correlations = self.calculate_correlations()

                # Save to file
                self.save_correlations(correlations)

                # Print summary every 10 seconds
                if iteration % 10 == 0 and correlations:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Correlations:")
                    for pair, data in correlations.items():
                        if data.get('correlation') is not None:
                            print(f"  {pair}: {data['correlation']:+.4f} ({data['data_points']} points)")
                        else:
                            print(f"  {pair}: {data.get('error', 'N/A')}")

                iteration += 1

                # Wait 1 second
                time.sleep(1)

            except KeyboardInterrupt:
                print("\nStopping correlation monitor...")
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(1)

if __name__ == "__main__":
    monitor = CorrelationMonitor()
    monitor.run()
