import json
import time
import csv
from datetime import datetime, timedelta
from pathlib import Path
from itertools import product
from collections import defaultdict
import threading
import sys

class BinaryOptionsPairTrader:
    def __init__(self):
        self.base_path = Path("/home/ubuntu/013_2025_polymarket/")
#C:\Users\theva\OneDrive\Documenti\2025_progetti\013_polymarket2025\bot014_Corre_Xtreme
        # Define all files
        self.files = {
            'BTC_CALL': self.base_path / '15M_CALL.json',
            'ETH_CALL': self.base_path / '15M_CALL_ETH.json',
            'SOL_CALL': self.base_path / '15M_CALL_SOL.json',
            'XRP_CALL': self.base_path / '15M_CALL_XRP.json',
            'BTC_PUT': self.base_path / '15M_PUT.json',
            'ETH_PUT': self.base_path / '15M_PUT_ETH.json',
            'SOL_PUT': self.base_path / '15M_PUT_SOL.json',
            'XRP_PUT': self.base_path / '15M_PUT_XRP.json',
        }

        # Generate all opposite direction pairs
        self.pairs = self._generate_pairs()

        # Storage for current period data
        self.current_period_data = defaultdict(list)
        self.current_period_start = None
        self.current_period_end = None

        # CSV output file
        self.output_file = self.base_path / 'bot014_Corre_Xtreme/data/pair_trading_results.csv'
        self._initialize_csv()

        self.running = True

    def _generate_pairs(self):
        """Generate all possible CALL+PUT pairs with different assets"""
        calls = ['BTC_CALL', 'ETH_CALL', 'SOL_CALL', 'XRP_CALL']
        puts = ['BTC_PUT', 'ETH_PUT', 'SOL_PUT', 'XRP_PUT']

        pairs = []
        for call in calls:
            for put in puts:
                # Create pairs with different underlying assets
                call_asset = call.split('_')[0]
                put_asset = put.split('_')[0]

                # Include all combinations (same and different assets)
                pair_name = f"{call}+{put}"
                pairs.append({
                    'name': pair_name,
                    'call': call,
                    'put': put,
                    'call_asset': call_asset,
                    'put_asset': put_asset
                })

        return pairs

    def _initialize_csv(self):
        """Initialize CSV file with headers"""
        headers = [
            'period_start', 'period_end', 'pair_name',
            'best_entry_time', 'best_entry_call_price', 'best_entry_put_price',
            'best_entry_combined_price', 'minutes_to_end_at_entry',
            'best_exit_time', 'best_exit_call_price', 'best_exit_put_price',
            'best_exit_combined_price', 'minutes_to_end_at_exit',
            'profit', 'profit_pct', 'holding_period_seconds'
        ]

        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def _get_next_period_times(self):
        """Calculate the current 15-minute period boundaries (00, 15, 30, 45)"""
        now = datetime.now()

        # Calculate minutes in current hour
        minutes = now.minute

        # Find the period start (00, 15, 30, 45)
        if minutes < 15:
            period_minute = 0
        elif minutes < 30:
            period_minute = 15
        elif minutes < 45:
            period_minute = 30
        else:
            period_minute = 45

        # Create period start time
        period_start = now.replace(minute=period_minute, second=0, microsecond=0)

        # Calculate period end (next boundary)
        if period_minute == 45:
            period_end = (period_start + timedelta(hours=1)).replace(minute=0)
        else:
            period_end = period_start + timedelta(minutes=15)

        return period_start, period_end

    def _read_json_file(self, filepath):
        """Safely read JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                return data
        except (FileNotFoundError, json.JSONDecodeError) as e:
            #print(f"Error reading {filepath}: {e}")
            return None

    def _get_current_prices(self):
        """Read all JSON files and extract current prices"""
        prices = {}

        for key, filepath in self.files.items():
            data = self._read_json_file(filepath)
            if data:
                # Extract ask price (for entry) and bid price (for exit)
                prices[key] = {
                    'ask': data.get('ask_price', None),
                    'bid': data.get('bid_price', None),
                    'timestamp': datetime.now()
                }
            else:
                prices[key] = None

        return prices

    def _calculate_pair_prices(self, prices):
        """Calculate combined entry and exit prices for all pairs"""
        pair_prices = {}

        for pair in self.pairs:
            call_data = prices.get(pair['call'])
            put_data = prices.get(pair['put'])

            if call_data and put_data and call_data['ask'] and put_data['ask']:
                # Entry price: sum of ask prices
                entry_price = call_data['ask'] + put_data['ask']

                # Exit price: sum of bid prices
                exit_price = call_data['bid'] + put_data['bid'] if call_data['bid'] and put_data['bid'] else None

                pair_prices[pair['name']] = {
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'call_ask': call_data['ask'],
                    'call_bid': call_data['bid'],
                    'put_ask': put_data['ask'],
                    'put_bid': put_data['bid'],
                    'timestamp': datetime.now()
                }

        return pair_prices

    def _record_snapshot(self, pair_prices):
        """Record current prices for all pairs"""
        now = datetime.now()
        minutes_to_end = (self.current_period_end - now).total_seconds() / 60

        for pair_name, data in pair_prices.items():
            self.current_period_data[pair_name].append({
                'timestamp': data['timestamp'],
                'entry_price': data['entry_price'],
                'exit_price': data['exit_price'],
                'call_ask': data['call_ask'],
                'call_bid': data['call_bid'],
                'put_ask': data['put_ask'],
                'put_bid': data['put_bid'],
                'minutes_to_end': minutes_to_end
            })

    def _find_best_entry_exit(self, snapshots):
        """Find the most profitable entry and exit combination for a pair"""
        if not snapshots:
            return None

        best_profit = float('-inf')
        best_entry_idx = None
        best_exit_idx = None

        # Try all entry/exit combinations where exit comes after entry
        for i in range(len(snapshots)):
            entry = snapshots[i]

            for j in range(i + 1, len(snapshots)):
                exit_data = snapshots[j]

                # Calculate profit
                if exit_data['exit_price'] is not None:
                    profit = exit_data['exit_price'] - entry['entry_price']

                    if profit > best_profit:
                        best_profit = profit
                        best_entry_idx = i
                        best_exit_idx = j

        if best_entry_idx is not None and best_exit_idx is not None:
            entry = snapshots[best_entry_idx]
            exit_data = snapshots[best_exit_idx]

            holding_seconds = (exit_data['timestamp'] - entry['timestamp']).total_seconds()
            profit_pct = (best_profit / entry['entry_price']) * 100 if entry['entry_price'] > 0 else 0

            return {
                'entry': entry,
                'exit': exit_data,
                'profit': best_profit,
                'profit_pct': profit_pct,
                'holding_seconds': holding_seconds
            }

        return None

    def _save_period_results(self):
        """Analyze and save results for the completed period"""
        print(f"\n{'='*80}")
        print(f"Period completed: {self.current_period_start} to {self.current_period_end}")
        print(f"{'='*80}")

        results = []

        for pair_name, snapshots in self.current_period_data.items():
            print(f"\nAnalyzing {pair_name}: {len(snapshots)} snapshots")

            best_trade = self._find_best_entry_exit(snapshots)

            if best_trade:
                result = {
                    'period_start': self.current_period_start.strftime('%Y-%m-%d %H:%M:%S'),
                    'period_end': self.current_period_end.strftime('%Y-%m-%d %H:%M:%S'),
                    'pair_name': pair_name,
                    'best_entry_time': best_trade['entry']['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'best_entry_call_price': best_trade['entry']['call_ask'],
                    'best_entry_put_price': best_trade['entry']['put_ask'],
                    'best_entry_combined_price': best_trade['entry']['entry_price'],
                    'minutes_to_end_at_entry': round(best_trade['entry']['minutes_to_end'], 2),
                    'best_exit_time': best_trade['exit']['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'best_exit_call_price': best_trade['exit']['call_bid'],
                    'best_exit_put_price': best_trade['exit']['put_bid'],
                    'best_exit_combined_price': best_trade['exit']['exit_price'],
                    'minutes_to_end_at_exit': round(best_trade['exit']['minutes_to_end'], 2),
                    'profit': round(best_trade['profit'], 6),
                    'profit_pct': round(best_trade['profit_pct'], 2),
                    'holding_period_seconds': round(best_trade['holding_seconds'])
                }

                results.append(result)

                print(f"  Best profit: {result['profit']:.6f} ({result['profit_pct']:.2f}%)")
                print(f"  Entry: {result['best_entry_time']} @ {result['best_entry_combined_price']:.6f}")
                print(f"  Exit:  {result['best_exit_time']} @ {result['best_exit_combined_price']:.6f}")

        # Save to CSV
        if results:
            with open(self.output_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writerows(results)

            print(f"\n✓ Saved {len(results)} pair results to {self.output_file}")

        print(f"{'='*80}\n")

    def _wait_for_period_start(self):
        """Wait until the next period starts"""
        period_start, period_end = self._get_next_period_times()
        now = datetime.now()

        if now < period_start:
            wait_seconds = (period_start - now).total_seconds()
            print(f"Waiting {wait_seconds:.0f} seconds for period to start at {period_start}")
            time.sleep(wait_seconds)

        return period_start, period_end

    def run(self):
        """Main monitoring loop"""
        print("Binary Options Pair Trading Monitor Started")
        print(f"Monitoring {len(self.pairs)} pairs")
        print(f"Output file: {self.output_file}")

        try:
            while self.running:
                # Wait for period start if needed
                self.current_period_start, self.current_period_end = self._wait_for_period_start()

                print(f"\n{'='*80}")
                print(f"NEW PERIOD: {self.current_period_start} to {self.current_period_end}")
                print(f"{'='*80}\n")

                # Clear data for new period
                self.current_period_data = defaultdict(list)

                # Monitor throughout the period
                while datetime.now() < self.current_period_end:
                    # Read current prices
                    prices = self._get_current_prices()

                    # Calculate pair prices
                    pair_prices = self._calculate_pair_prices(prices)

                    # Record snapshot
                    self._record_snapshot(pair_prices)

                    # Show progress
                    now = datetime.now()
                    seconds_remaining = (self.current_period_end - now).total_seconds()
                    print(f"\r[{now.strftime('%H:%M:%S')}] Monitoring... {seconds_remaining:.0f}s remaining   ", end='', flush=True)

                    # Wait 1 second before next reading
                    time.sleep(1)

                # Period ended - analyze and save results
                self._save_period_results()

        except KeyboardInterrupt:
            print("\n\nShutting down gracefully...")
            self.running = False
        except Exception as e:
            print(f"\nError in main loop: {e}")
            import traceback
            traceback.print_exc()

def main():
    trader = BinaryOptionsPairTrader()
    trader.run()

if __name__ == "__main__":
    main()
