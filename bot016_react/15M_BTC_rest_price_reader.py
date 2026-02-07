#!/usr/bin/env python3
"""
15-Minute BTC Binary Options REST Price Reader
Fetches bid/ask prices via REST API and writes atomically to JSON files
Updates asset IDs every 15 minutes (00, 15, 30, 45)
"""

import json
import time
import os
import sys
import tempfile
import requests
from datetime import datetime, timezone
from typing import Optional, Dict

# Add polymarket_trading_core path
sys.path.insert(0, '/home/ubuntu')
from polymarket_trading_core_debug import PolymarketTrader, load_credentials_from_env

# Output file paths
CALL_OUTPUT = "/home/ubuntu/013_2025_polymarket/15M_BTC_CALL_rest.json"
PUT_OUTPUT = "/home/ubuntu/013_2025_polymarket/15M_BTC_PUT_rest.json"

# Refresh interval (seconds)
REFRESH_INTERVAL = 0.05  # 500ms

def atomic_write_json(filepath, data):
    """
    Write JSON atomically to prevent race conditions.
    Reader will never see partial/empty data.
    """
    target_dir = os.path.dirname(os.path.abspath(filepath))
    if not target_dir:
        target_dir = '.'

    # Create temp file in same directory (required for atomic rename)
    temp_fd, temp_path = tempfile.mkstemp(
        dir=target_dir,
        prefix='.tmp_',
        suffix='.json'
    )

    try:
        # Write data to temp file
        with os.fdopen(temp_fd, 'w') as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())  # Ensure data is written to disk

        # Atomic rename - reader sees old or new, never partial
        os.replace(temp_path, filepath)

    except Exception as e:
        # Clean up temp file on error
        try:
            os.unlink(temp_path)
        except:
            pass
        raise e


class RestPriceReader:
    def __init__(self, verbose=True):
        self.verbose = verbose

        # Initialize trader for REST API calls
        print("🔧 Initializing Polymarket trader...")
        #credentials = load_credentials_from_env()
        env_path = '/home/ubuntu/013_2025_polymarket/keys/keys_ovh38.env'
        credentials = load_credentials_from_env(env_path)
        self.trader = PolymarketTrader(
            clob_api_url=credentials['clob_api_url'],
            private_key=credentials['private_key'],
            api_key=credentials['api_key'],
            api_secret=credentials['api_secret'],
            api_passphrase=credentials['api_passphrase']
        )

        # Current token IDs
        self.call_token_id = None
        self.put_token_id = None
        self.current_market_slug = None

        # Track last quarter hour update
        utc_time = self.get_utc_time()
        self.last_update_quarter = self.get_current_quarter_hour(utc_time)

        # Initial market data fetch
        self.update_market_data()

    def get_utc_time(self):
        """Get current UTC time"""
        return datetime.now(timezone.utc)

    def get_current_quarter_hour(self, dt):
        """Get the current quarter hour (0, 15, 30, 45) from a datetime"""
        minute = dt.minute
        if minute < 15:
            return 0
        elif minute < 30:
            return 15
        elif minute < 45:
            return 30
        else:
            return 45

    def generate_market_slug(self) -> str:
        """Generate the market slug based on current UTC time rounded to last quarter hour"""
        utc_time = self.get_utc_time()

        # Round down to the last quarter hour (00, 15, 30, 45)
        current_minute = utc_time.minute
        if current_minute < 15:
            target_minute = 0
        elif current_minute < 30:
            target_minute = 15
        elif current_minute < 45:
            target_minute = 30
        else:
            target_minute = 45

        # Create datetime with rounded minute
        rounded_time = utc_time.replace(minute=target_minute, second=0, microsecond=0)

        # Convert to Unix timestamp
        unix_timestamp = int(rounded_time.timestamp())

        slug = f"btc-updown-15m-{unix_timestamp}"

        if self.verbose:
            print(f"🕐 Current UTC: {utc_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🕐 Rounded UTC: {rounded_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🔗 Generated slug: {slug}")

        return slug

    def generate_previous_quarter_slug(self) -> str:
        """Generate slug for the previous quarter hour"""
        utc_time = self.get_utc_time()

        current_quarter = self.get_current_quarter_hour(utc_time)

        # Calculate previous quarter
        if current_quarter == 0:
            # Go to previous hour, 45 minutes
            prev_time = utc_time.replace(hour=utc_time.hour-1, minute=45, second=0, microsecond=0)
            if utc_time.hour == 0:
                # Handle day rollover
                prev_time = utc_time.replace(hour=23, minute=45, second=0, microsecond=0)
        else:
            # Just go to previous quarter in same hour
            prev_quarter = current_quarter - 15
            prev_time = utc_time.replace(minute=prev_quarter, second=0, microsecond=0)

        unix_timestamp = int(prev_time.timestamp())
        slug = f"btc-updown-15m-{unix_timestamp}"

        if self.verbose:
            print(f"🔄 Previous quarter UTC: {prev_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🔗 Previous quarter slug: {slug}")

        return slug

    def fetch_market_data(self, slug: str) -> Optional[Dict]:
        """Fetch market data from Polymarket API"""
        try:
            api_url = f"https://gamma-api.polymarket.com/markets?slug={slug}"
            if self.verbose:
                print(f"🌐 Fetching market data from: {api_url}")

            response = requests.get(api_url, timeout=10)
            response.raise_for_status()

            data = response.json()
            if data and len(data) > 0:
                return data[0]  # Take first market result
            else:
                if self.verbose:
                    print(f"⚠️ No market data found for slug: {slug}")
                return None

        except Exception as e:
            print(f"❌ Error fetching market data: {e}")
            return None

    def update_market_data(self):
        """Update asset IDs from current market"""
        slug = self.generate_market_slug()
        market_data = self.fetch_market_data(slug)

        # If current slug fails, try previous quarter hour
        if not market_data:
            if self.verbose:
                print(f"⚠️ Current market not found, trying previous quarter hour...")

            prev_slug = self.generate_previous_quarter_slug()
            market_data = self.fetch_market_data(prev_slug)
            slug = prev_slug

        if market_data:
            clob_token_ids = market_data.get("clobTokenIds")
            if clob_token_ids:
                try:
                    # Parse the JSON string
                    token_ids = json.loads(clob_token_ids)

                    if len(token_ids) >= 2:
                        # First is CALL, second is PUT
                        self.call_token_id = token_ids[0]
                        self.put_token_id = token_ids[1]
                        self.current_market_slug = slug

                        print(f"✅ Market data updated:")
                        print(f"   CALL: ...{self.call_token_id[-12:]}")
                        print(f"   PUT:  ...{self.put_token_id[-12:]}")
                        print(f"   Slug: {slug}")
                    else:
                        print(f"❌ Expected 2 token IDs, got {len(token_ids)}")

                except Exception as e:
                    print(f"❌ Error parsing token IDs: {e}")
            else:
                print(f"❌ No clobTokenIds in market data")
        else:
            print(f"❌ Failed to fetch market data")

    def check_quarter_hour_change(self):
        """Check if we've crossed into a new quarter hour"""
        utc_time = self.get_utc_time()
        current_quarter = self.get_current_quarter_hour(utc_time)

        if current_quarter != self.last_update_quarter:
            print(f"\n{'='*60}")
            print(f"⏰ QUARTER HOUR CHANGE: {self.last_update_quarter} → {current_quarter}")
            print(f"🔄 Updating market data...")
            print(f"{'='*60}\n")

            self.update_market_data()
            self.last_update_quarter = current_quarter

    def get_token_price(self, token_id: str, token_type: str) -> Optional[Dict]:
        """Get bid/ask prices for a token via REST API"""
        try:
            start_time = time.time()

            # Get bid and ask prices
            bid_response = self.trader.get_price(token_id, side="BUY")
            ask_response = self.trader.get_price(token_id, side="SELL")

            bid_price = float(bid_response['price'])
            ask_price = float(ask_response['price'])

            delay_ms = int((time.time() - start_time) * 1000)

            return {
                'best_bid': bid_price,
                'best_ask': ask_price,
                'spread': ask_price - bid_price,
                'delay_ms': delay_ms
            }

        except Exception as e:
            print(f"❌ Error fetching {token_type} price: {e}")
            return None

    def write_price_file(self, token_id: str, token_type: str, output_path: str):
        """Fetch prices and write to file"""
        price_data = self.get_token_price(token_id, token_type)

        if price_data:
            now = datetime.now(timezone.utc)
            timestamp_ms = int(now.timestamp() * 1000)

            output = {
                'asset_name': f"15M_{token_type}",
                'asset_id': token_id,
                'market': self.current_market_slug,
                'best_bid': price_data['best_bid'],
                'best_ask': price_data['best_ask'],
                'spread': price_data['spread'],
                'timestamp': timestamp_ms,
                'timestamp_readable': now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                'delay_ms': price_data['delay_ms']
            }

            atomic_write_json(output_path, output)

            if self.verbose:
                print(f"{token_type}: bid={price_data['best_bid']:.4f} ask={price_data['best_ask']:.4f} delay={price_data['delay_ms']}ms")

    def run(self):
        """Main loop - fetch and write prices continuously"""
        print("\n" + "="*60)
        print("🚀 REST Price Reader Started")
        print("="*60)
        print(f"📊 CALL output: {CALL_OUTPUT}")
        print(f"📊 PUT output: {PUT_OUTPUT}")
        print(f"⏱️  Refresh interval: {REFRESH_INTERVAL}s")
        print("="*60 + "\n")

        while True:
            try:
                # Check for quarter hour change
                self.check_quarter_hour_change()

                # Skip if we don't have token IDs yet
                if not self.call_token_id or not self.put_token_id:
                    print("⚠️ Waiting for token IDs...")
                    time.sleep(5)
                    continue

                # Fetch and write CALL prices
                self.write_price_file(self.call_token_id, "CALL", CALL_OUTPUT)

                # Fetch and write PUT prices
                self.write_price_file(self.put_token_id, "PUT", PUT_OUTPUT)

                time.sleep(REFRESH_INTERVAL)

            except KeyboardInterrupt:
                print("\n\n⏸️  Stopped by user")
                break

            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)


if __name__ == "__main__":
    reader = RestPriceReader(verbose=False)
    reader.run()
