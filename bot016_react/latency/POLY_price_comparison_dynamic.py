#!/usr/bin/env python3
"""
Dynamic price comparison with automatic token ID updates
Reads TOKEN_ID from WebSocket JSON file every 30 seconds to handle 15-minute contract rollovers
"""

import json
import time
import csv
from datetime import datetime
from py_clob_client.client import ClobClient
from collections import defaultdict
import statistics
import os

class DynamicPriceComparator:
    def __init__(self, ws_file_path, log_file=None):
        self.ws_file_path = ws_file_path
        self.client = ClobClient("https://clob.polymarket.com")
        self.log_file = log_file or f"price_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        # Current token tracking
        self.current_token_id = None
        self.last_token_check = 0
        self.token_check_interval = 30  # Check token every 30 seconds
        self.token_changes = 0

        # Statistics tracking (reset on token change)
        self.stats = {
            'total_comparisons': 0,
            'price_differences': 0,
            'ws_ahead': 0,
            'api_ahead': 0,
            'exact_match': 0,
            'bid_differences': [],
            'ask_differences': [],
            'spread_differences': [],
            'time_differences': [],
            'ws_fresher': [],
            'api_fresher': []
        }

        # Token-specific stats history
        self.token_history = []

        # Initialize CSV log
        self._init_csv_log()

    def _init_csv_log(self):
        """Initialize CSV log file with headers"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'token_id', 'asset_name',
                    'ws_bid', 'ws_ask', 'ws_spread', 'ws_timestamp',
                    'api_bid', 'api_ask', 'api_spread', 'api_timestamp',
                    'bid_diff', 'ask_diff', 'spread_diff', 'time_diff',
                    'price_differs', 'who_ahead'
                ])

    def read_token_from_ws(self):
        """Read token ID and asset info from WebSocket JSON file"""
        try:
            with open(self.ws_file_path, 'r') as f:
                data = json.load(f)

            token_id = data.get('asset_id')
            asset_name = data.get('asset_name')

            return token_id, asset_name, data
        except Exception as e:
            print(f"Error reading token from WS file: {e}")
            return None, None, None

    def check_and_update_token(self):
        """Check if we need to update the token ID"""
        current_time = time.time()

        # Check every 30 seconds
        if current_time - self.last_token_check < self.token_check_interval:
            return False

        self.last_token_check = current_time

        # Read token from file
        new_token_id, asset_name, _ = self.read_token_from_ws()

        if new_token_id and new_token_id != self.current_token_id:
            # Token changed - save current stats and reset
            if self.current_token_id:
                print(f"\n{'='*80}")
                print(f"🔄 TOKEN CHANGE DETECTED")
                print(f"{'='*80}")
                print(f"Old Token: {self.current_token_id}")
                print(f"New Token: {new_token_id}")
                print(f"Asset:     {asset_name}")
                print(f"{'='*80}\n")

                # Save stats for old token
                self.token_history.append({
                    'token_id': self.current_token_id,
                    'stats': self.stats.copy(),
                    'ended_at': datetime.now()
                })

                # Print summary for old token
                self._print_token_summary()

                # Reset stats
                self._reset_stats()
                self.token_changes += 1
            else:
                print(f"\n{'='*80}")
                print(f"🎯 INITIAL TOKEN DETECTED")
                print(f"{'='*80}")
                print(f"Token ID:  {new_token_id}")
                print(f"Asset:     {asset_name}")
                print(f"{'='*80}\n")

            self.current_token_id = new_token_id
            return True

        return False

    def _reset_stats(self):
        """Reset statistics for new token"""
        self.stats = {
            'total_comparisons': 0,
            'price_differences': 0,
            'ws_ahead': 0,
            'api_ahead': 0,
            'exact_match': 0,
            'bid_differences': [],
            'ask_differences': [],
            'spread_differences': [],
            'time_differences': [],
            'ws_fresher': [],
            'api_fresher': []
        }

    def _print_token_summary(self):
        """Print summary for completed token"""
        print(f"\n{'─'*80}")
        print(f"📊 SUMMARY FOR COMPLETED TOKEN")
        print(f"{'─'*80}")
        self.print_statistics(brief=True)

    def read_ws_data(self):
        """Read the latest WebSocket data from JSON file"""
        try:
            with open(self.ws_file_path, 'r') as f:
                data = json.load(f)
            return data
        except FileNotFoundError:
            print(f"Warning: WS file not found: {self.ws_file_path}")
            return None
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in WS file: {e}")
            return None
        except Exception as e:
            print(f"Error reading WS file: {e}")
            return None

    def get_api_data(self):
        """Fetch current prices from REST API"""
        if not self.current_token_id:
            return None

        try:
            start_time = time.time()

            # Get both bid and ask prices - returns {price: string}
            bid_response = self.client.get_price(self.current_token_id, side="BUY")
            ask_response = self.client.get_price(self.current_token_id, side="SELL")

            # Extract price from response dict
            bid_price = float(bid_response['price'])
            ask_price = float(ask_response['price'])

            api_latency = (time.time() - start_time) * 1000  # ms

            return {
                'bid': bid_price,
                'ask': ask_price,
                'spread': ask_price - bid_price,
                'timestamp': time.time() * 1000,
                'latency': api_latency
            }
        except Exception as e:
            print(f"Error fetching API data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def compare_prices(self, ws_data, api_data):
        """Compare WebSocket and API prices with detailed analysis"""
        if not ws_data or not api_data or not self.current_token_id:
            return None

        self.stats['total_comparisons'] += 1

        # Extract prices
        ws_bid = ws_data['best_bid']['price']
        ws_ask = ws_data['best_ask']['price']
        ws_spread = ws_data['spread']
        ws_timestamp = float(ws_data['timestamp'])
        asset_name = ws_data.get('asset_name', 'Unknown')

        api_bid = api_data['bid']
        api_ask = api_data['ask']
        api_spread = api_data['spread']
        api_timestamp = api_data['timestamp']
        api_latency = api_data['latency']

        # Calculate differences
        bid_diff = ws_bid - api_bid
        ask_diff = ws_ask - api_ask
        spread_diff = ws_spread - api_spread
        time_diff = api_timestamp - ws_timestamp

        # Determine if prices differ
        price_differs = abs(bid_diff) > 0.001 or abs(ask_diff) > 0.001

        # Determine who is ahead
        if price_differs:
            self.stats['price_differences'] += 1
            self.stats['bid_differences'].append(abs(bid_diff))
            self.stats['ask_differences'].append(abs(ask_diff))
            self.stats['spread_differences'].append(abs(spread_diff))
            self.stats['time_differences'].append(time_diff)

            if ws_timestamp > api_timestamp:
                who_ahead = "WebSocket"
                self.stats['ws_ahead'] += 1
                self.stats['ws_fresher'].append(time_diff)
            elif api_timestamp > ws_timestamp:
                who_ahead = "REST_API"
                self.stats['api_ahead'] += 1
                self.stats['api_fresher'].append(time_diff)
            else:
                who_ahead = "Equal"
        else:
            who_ahead = "Match"
            self.stats['exact_match'] += 1

        # Log to CSV
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                self.current_token_id,
                asset_name,
                ws_bid, ws_ask, ws_spread, ws_timestamp,
                api_bid, api_ask, api_spread, api_timestamp,
                bid_diff, ask_diff, spread_diff, time_diff,
                price_differs, who_ahead
            ])

        # Print real-time comparison if prices differ
        if price_differs:
            print(f"\n{'='*80}")
            print(f"⚠️  PRICE DIFFERENCE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
            print(f"Asset: {asset_name} | Token: {self.current_token_id[:20]}...")
            print(f"{'='*80}")
            print(f"WebSocket:  Bid={ws_bid:.4f}  Ask={ws_ask:.4f}  Spread={ws_spread:.4f}")
            print(f"REST API:   Bid={api_bid:.4f}  Ask={api_ask:.4f}  Spread={api_spread:.4f}  (latency: {api_latency:.1f}ms)")
            print(f"Δ Absolute: Bid={abs(bid_diff):.4f}  Ask={abs(ask_diff):.4f}  Spread={abs(spread_diff):.4f}")
            print(f"Δ Signed:   Bid={bid_diff:+.4f}  Ask={ask_diff:+.4f}  Spread={spread_diff:+.4f}")
            print(f"Time Diff:  {time_diff:+.0f}ms")
            print(f"Data Ahead: {who_ahead}")

            if abs(bid_diff) > 0.01 or abs(ask_diff) > 0.01:
                print(f"🔴 SIGNIFICANT DIFFERENCE (>1 cent)")

        return {
            'differs': price_differs,
            'who_ahead': who_ahead,
            'bid_diff': bid_diff,
            'ask_diff': ask_diff,
            'time_diff': time_diff
        }

    def print_statistics(self, brief=False):
        """Print comprehensive statistics"""
        if self.stats['total_comparisons'] == 0:
            print("\nNo comparisons performed yet.")
            return

        if not brief:
            print(f"\n{'='*80}")
            print(f"📊 CURRENT TOKEN STATISTICS")
            print(f"{'='*80}")

        print(f"Token ID:               {self.current_token_id[:30]}..." if self.current_token_id else "None")
        print(f"Total Comparisons:      {self.stats['total_comparisons']}")
        print(f"Exact Matches:          {self.stats['exact_match']} ({self.stats['exact_match']/self.stats['total_comparisons']*100:.2f}%)")
        print(f"Price Differences:      {self.stats['price_differences']} ({self.stats['price_differences']/self.stats['total_comparisons']*100:.2f}%)")

        if self.stats['price_differences'] > 0:
            print(f"\n📍 Data Freshness:")
            print(f"  WebSocket Ahead:      {self.stats['ws_ahead']} ({self.stats['ws_ahead']/self.stats['price_differences']*100:.1f}%)")
            print(f"  REST API Ahead:       {self.stats['api_ahead']} ({self.stats['api_ahead']/self.stats['price_differences']*100:.1f}%)")

        if self.stats['bid_differences'] and not brief:
            print(f"\n💰 Bid Price Differences:")
            print(f"  Min:     ${min(self.stats['bid_differences']):.4f}")
            print(f"  Max:     ${max(self.stats['bid_differences']):.4f}")
            print(f"  Avg:     ${statistics.mean(self.stats['bid_differences']):.4f}")
            print(f"  Median:  ${statistics.median(self.stats['bid_differences']):.4f}")
            if len(self.stats['bid_differences']) > 1:
                print(f"  StdDev:  ${statistics.stdev(self.stats['bid_differences']):.4f}")

        if self.stats['ask_differences'] and not brief:
            print(f"\n💰 Ask Price Differences:")
            print(f"  Min:     ${min(self.stats['ask_differences']):.4f}")
            print(f"  Max:     ${max(self.stats['ask_differences']):.4f}")
            print(f"  Avg:     ${statistics.mean(self.stats['ask_differences']):.4f}")
            print(f"  Median:  ${statistics.median(self.stats['ask_differences']):.4f}")
            if len(self.stats['ask_differences']) > 1:
                print(f"  StdDev:  ${statistics.stdev(self.stats['ask_differences']):.4f}")

        if self.stats['time_differences'] and not brief:
            print(f"\n⏱️  Timestamp Differences (API - WS):")
            print(f"  Min:     {min(self.stats['time_differences']):+.0f}ms")
            print(f"  Max:     {max(self.stats['time_differences']):+.0f}ms")
            print(f"  Avg:     {statistics.mean(self.stats['time_differences']):+.0f}ms")
            print(f"  Median:  {statistics.median(self.stats['time_differences']):+.0f}ms")
            if len(self.stats['time_differences']) > 1:
                print(f"  StdDev:  {statistics.stdev(self.stats['time_differences']):.0f}ms")

        if not brief:
            print(f"\n📝 Log file: {self.log_file}")
            print(f"{'='*80}\n")

    def print_overall_summary(self):
        """Print summary across all tokens"""
        print(f"\n{'='*80}")
        print(f"🏁 OVERALL SUMMARY")
        print(f"{'='*80}")
        print(f"Total Token Changes:    {self.token_changes}")
        print(f"Tokens Tracked:         {len(self.token_history) + 1}")  # +1 for current

        if self.token_history:
            print(f"\n📜 Token History:")
            for i, hist in enumerate(self.token_history, 1):
                stats = hist['stats']
                print(f"\n  Token #{i}:")
                print(f"    ID:          {hist['token_id'][:30]}...")
                print(f"    Ended:       {hist['ended_at'].strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"    Comparisons: {stats['total_comparisons']}")
                print(f"    Differences: {stats['price_differences']} ({stats['price_differences']/stats['total_comparisons']*100:.1f}%)" if stats['total_comparisons'] > 0 else "    Differences: 0")

        print(f"\n📊 Current Token:")
        self.print_statistics(brief=True)

    def run(self, interval=1.0, max_iterations=None):
        """Main loop to continuously compare prices"""
        print(f"\n{'='*80}")
        print(f"🚀 STARTING DYNAMIC PRICE COMPARISON")
        print(f"{'='*80}")
        print(f"WebSocket file:  {self.ws_file_path}")
        print(f"Check interval:  {interval}s")
        print(f"Token check:     Every {self.token_check_interval}s")
        print(f"Log file:        {self.log_file}")
        if max_iterations:
            print(f"Max iterations:  {max_iterations}")
        print(f"Press Ctrl+C to stop\n")

        comparison_count = 0
        last_status_time = time.time()

        try:
            while True:
                if max_iterations and comparison_count >= max_iterations:
                    print(f"\nReached maximum iterations: {max_iterations}")
                    break

                # Check for token updates every 30 seconds
                self.check_and_update_token()

                if not self.current_token_id:
                    print("Waiting for token ID...")
                    time.sleep(interval)
                    continue

                comparison_count += 1

                # Read WebSocket data
                ws_data = self.read_ws_data()

                # Get API data
                api_data = self.get_api_data()

                # Compare
                result = self.compare_prices(ws_data, api_data)

                # Print periodic status (every 60 seconds)
                if time.time() - last_status_time >= 60:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status: {comparison_count} checks, "
                          f"{self.stats['price_differences']} differences "
                          f"({self.stats['price_differences']/self.stats['total_comparisons']*100:.1f}% for current token)" if self.stats['total_comparisons'] > 0 else "")
                    last_status_time = time.time()

                # Wait before next check
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping comparison...")
            self.print_overall_summary()
        except Exception as e:
            print(f"\n❌ Error in main loop: {e}")
            import traceback
            traceback.print_exc()
            self.print_overall_summary()


if __name__ == "__main__":
    # Configuration
    WS_FILE = "/home/ubuntu/013_2025_polymarket/15M_CALL.json"
    CHECK_INTERVAL = 0.1  # seconds
    LOG_FILE = "price_comparison_dynamic.csv"

    # Create and run comparator
    comparator = DynamicPriceComparator(WS_FILE, log_file=LOG_FILE)
    comparator.run(interval=CHECK_INTERVAL)
