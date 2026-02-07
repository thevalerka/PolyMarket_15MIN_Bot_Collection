from websocket import WebSocketApp
import json
import time
import threading
from datetime import datetime, timezone, timedelta
import os
import requests
from typing import Dict, List, Optional

#  pm2 start 15M_BTC_polymarket_dynamic_monitor_NEXT.py --interpreter python3
#  Monitors the NEXT 15-minute period (not the current one)
#  Example: If current time is 14:04, monitors the 14:15-14:30 option

MARKET_CHANNEL = "market"
USER_CHANNEL = "user"

class DynamicPolymarketMonitorNext:
    def __init__(self, url, auth, verbose=True):
        self.url = url
        self.auth = auth
        self.verbose = verbose
        self.ws = None

        # Asset mapping for filenames - will be updated dynamically
        self.asset_names = {}
        self.asset_ids = []

        # Store last known book data for each asset
        self.last_book_data = {}

        # Initialize with next market data
        self.update_market_data()

        # Set initial quarter hour to current quarter hour to prevent immediate update
        utc_time = self.get_utc_time()
        self.last_update_quarter = self.get_current_quarter_hour(utc_time)

    def get_utc_time(self):
        """Get current UTC time"""
        utc_time = datetime.now(timezone.utc)
        if self.verbose:
            print(f"🌍 Current UTC: {utc_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        return utc_time

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

    def get_next_quarter_hour(self, dt):
        """Get the NEXT quarter hour (0, 15, 30, 45) from a datetime"""
        minute = dt.minute
        if minute < 15:
            return 15
        elif minute < 30:
            return 30
        elif minute < 45:
            return 45
        else:
            return 0  # Next hour

    def generate_market_slug(self) -> str:
        """Generate the market slug based on NEXT quarter hour (not current)"""
        utc_time = self.get_utc_time()

        # Calculate the NEXT quarter hour start time
        current_minute = utc_time.minute
        if current_minute < 15:
            target_minute = 15
            target_hour = utc_time.hour
            target_day = utc_time.day
        elif current_minute < 30:
            target_minute = 30
            target_hour = utc_time.hour
            target_day = utc_time.day
        elif current_minute < 45:
            target_minute = 45
            target_hour = utc_time.hour
            target_day = utc_time.day
        else:
            # Next hour
            target_minute = 0
            if utc_time.hour == 23:
                # Handle day rollover
                next_day = utc_time + timedelta(days=1)
                target_hour = 0
                target_day = next_day.day
            else:
                target_hour = utc_time.hour + 1
                target_day = utc_time.day

        # Create datetime for the NEXT quarter hour
        try:
            next_quarter_time = utc_time.replace(
                day=target_day,
                hour=target_hour, 
                minute=target_minute, 
                second=0, 
                microsecond=0
            )
        except ValueError:
            # Handle month rollover edge case
            next_quarter_time = utc_time + timedelta(hours=1)
            next_quarter_time = next_quarter_time.replace(minute=0, second=0, microsecond=0)

        # Convert to Unix timestamp
        unix_timestamp = int(next_quarter_time.timestamp())

        slug = f"btc-updown-15m-{unix_timestamp}"

        if self.verbose:
            print(f"🕐 Current UTC: {utc_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🕐 NEXT quarter UTC: {next_quarter_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🔗 Generated slug (NEXT period): {slug}")

        return slug

    def generate_current_quarter_slug(self) -> str:
        """Generate slug for the current quarter hour (fallback)"""
        utc_time = self.get_utc_time()

        # Round down to the current quarter hour
        current_minute = utc_time.minute
        if current_minute < 15:
            target_minute = 0
        elif current_minute < 30:
            target_minute = 15
        elif current_minute < 45:
            target_minute = 30
        else:
            target_minute = 45

        current_time = utc_time.replace(minute=target_minute, second=0, microsecond=0)

        unix_timestamp = int(current_time.timestamp())
        slug = f"btc-updown-15m-{unix_timestamp}"

        if self.verbose:
            print(f"🔄 Current quarter UTC: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🔗 Current quarter slug: {slug}")

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
        """Update asset IDs and names from NEXT market period"""
        slug = self.generate_market_slug()
        market_data = self.fetch_market_data(slug)

        # If next quarter slug fails, the market may not exist yet
        # This is expected if we're too early in the current period
        if not market_data:
            if self.verbose:
                print(f"⚠️ NEXT market not found yet (may not be created yet)")
                print(f"⚠️ Will retry on next check...")
            return False

        if market_data:
            clob_token_ids = market_data.get("clobTokenIds")
            if clob_token_ids:
                try:
                    # Parse the JSON string
                    token_ids = json.loads(clob_token_ids)

                    if len(token_ids) >= 2:
                        # First is CALL, second is PUT
                        call_id = token_ids[0]
                        put_id = token_ids[1]

                        # Update asset mapping - using NEXT_ prefix to distinguish
                        self.asset_names = {
                            call_id: "NEXT_15M_CALL",
                            put_id: "NEXT_15M_PUT"
                        }

                        self.asset_ids = [call_id, put_id]

                        # Initialize empty book data for each asset
                        self.last_book_data = {
                            call_id: {"bids": [], "asks": []},
                            put_id: {"bids": [], "asks": []}
                        }

                        if self.verbose:
                            print(f"✅ Updated NEXT market data")
                            print(f"   CALL ID: {call_id}")
                            print(f"   PUT ID: {put_id}")

                        return True
                    else:
                        print(f"⚠️ Expected 2 token IDs, got {len(token_ids)}")

                except json.JSONDecodeError as e:
                    print(f"❌ Error parsing clobTokenIds: {e}")
            else:
                print(f"⚠️ No clobTokenIds found in market data")

        print(f"❌ Failed to find NEXT Bitcoin Up/Down market")
        return False

    def update_market_data_and_reconnect(self):
        """Update market data and reconnect WebSocket"""
        if self.verbose:
            print("\n🔄 Quarter-hourly update: Fetching NEXT market data...")

        success = self.update_market_data()
        if success and self.ws:
            if self.verbose:
                print("🔌 Reconnecting WebSocket with new asset IDs...")

            # Close current connection
            self.ws.close()
            time.sleep(2)

            # Start new connection
            self.connect_websocket()

    def calculate_timestamp_delay(self, message_timestamp):
        """Calculate delay between message timestamp and current time"""
        current_time_ms = int(time.time() * 1000)

        # Handle None or invalid timestamp
        if message_timestamp is None:
            return 0

        try:
            delay_ms = current_time_ms - int(message_timestamp)
            return delay_ms
        except (ValueError, TypeError):
            return 0

    def get_best_bid_ask(self, bids, asks):
        """Extract best bid and ask from the order book"""
        best_bid = None
        best_ask = None

        if bids and len(bids) > 0:
            try:
                # Bids are sorted highest to lowest, so last is best
                best_bid = {"price": float(bids[-1]["price"]), "size": float(bids[-1]["size"])}
            except (KeyError, ValueError, TypeError, IndexError):
                pass

        if asks and len(asks) > 0:
            try:
                # Asks are sorted lowest to highest, so last is best (lowest price)
                best_ask = {"price": float(asks[-1]["price"]), "size": float(asks[-1]["size"])}
            except (KeyError, ValueError, TypeError, IndexError):
                pass

        return best_bid, best_ask

    def export_book_by_asset(self, asset_id, book_data):
        """Export book data to asset-specific file - EXACT SAME STRUCTURE"""
        asset_name = self.asset_names.get(asset_id, "UNKNOWN")
        filename = f"{asset_name}.json"

        try:
            # Write/overwrite the file for this asset
            with open(filename, 'w') as f:
                json.dump(book_data, f, indent=2)

            if self.verbose:
                print(f"   💾 Exported to {filename}")

        except Exception as e:
            print(f"❌ Failed to export {filename}: {e}")

    def create_book_export_from_current_data(self, asset_id, message_data=None):
        """Create the EXACT same book export structure using current book data"""
        asset_name = self.asset_names.get(asset_id, asset_id[:8] + "..." if asset_id else "UNKNOWN")

        # Handle None message_data or missing keys
        if message_data is None:
            message_data = {}

        timestamp = message_data.get("timestamp")
        if timestamp is None:
            timestamp = int(time.time() * 1000)  # Use current time if no timestamp

        delay_ms = self.calculate_timestamp_delay(timestamp)

        # Get current book data (stored from last book update)
        current_book = self.last_book_data.get(asset_id, {"bids": [], "asks": []})
        bids = current_book.get("bids", [])
        asks = current_book.get("asks", [])

        best_bid, best_ask = self.get_best_bid_ask(bids, asks)

        # Calculate spread safely
        spread = None
        if best_bid and best_ask:
            try:
                spread = best_ask["price"] - best_bid["price"]
            except (KeyError, TypeError):
                pass

        # Create EXACT same structure as original book_export
        book_export = {
            "asset_name": asset_name,
            "asset_id": asset_id,
            "market": message_data.get("market"),
            "timestamp": timestamp,
            "timestamp_readable": datetime.fromtimestamp(int(timestamp)/1000).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            "delay_ms": delay_ms,
            "hash": message_data.get("hash"),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "total_bids": len(bids),
            "total_asks": len(asks),
            "complete_book": {
                "bids": bids,
                "asks": asks
            },
            "updated_at": int(time.time() * 1000)
        }

        return book_export

    def update_all_assets_with_timestamp(self, message_data=None, event_type_name="UNKNOWN"):
        """Update all asset files with current book data and new timestamp"""
        for asset_id in self.asset_ids:
            try:
                book_export = self.create_book_export_from_current_data(asset_id, message_data)
                self.export_book_by_asset(asset_id, book_export)

                if self.verbose:
                    asset_name = self.asset_names.get(asset_id, asset_id[:8] + "...")
                    timestamp = book_export.get("timestamp", "N/A")
                    print(f"🔄 Updated {asset_name} from {event_type_name} event (ts: {timestamp})")

            except Exception as e:
                if self.verbose:
                    asset_name = self.asset_names.get(asset_id, asset_id[:8] + "...")
                    print(f"❌ Failed to update {asset_name}: {e}")

    def process_book_message(self, message_data):
        """Process order book messages and export by asset"""
        if message_data is None:
            if self.verbose:
                print("⚠️ Received None book message, updating with current data")
            self.update_all_assets_with_timestamp(None, "NULL_BOOK")
            return

        asset_id = message_data.get("asset_id")
        if asset_id is None or asset_id not in self.asset_ids:
            if self.verbose:
                print(f"⚠️ Book message missing asset_id or unknown asset, updating all assets")
            self.update_all_assets_with_timestamp(message_data, "BOOK_NO_ASSET")
            return

        asset_name = self.asset_names.get(asset_id, asset_id[:8] + "...")

        timestamp = message_data.get("timestamp")
        if timestamp is None:
            timestamp = int(time.time() * 1000)

        delay_ms = self.calculate_timestamp_delay(timestamp)

        bids = message_data.get("bids", [])
        asks = message_data.get("asks", [])

        # Update stored book data
        self.last_book_data[asset_id] = {"bids": bids, "asks": asks}

        best_bid, best_ask = self.get_best_bid_ask(bids, asks)

        # Calculate spread safely
        spread = None
        if best_bid and best_ask:
            try:
                spread = best_ask["price"] - best_bid["price"]
            except (KeyError, TypeError):
                pass

        # Create complete book export data - EXACT SAME STRUCTURE
        book_export = {
            "asset_name": asset_name,
            "asset_id": asset_id,
            "market": message_data.get("market"),
            "timestamp": timestamp,
            "timestamp_readable": datetime.fromtimestamp(int(timestamp)/1000).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            "delay_ms": delay_ms,
            "hash": message_data.get("hash"),
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "total_bids": len(bids),
            "total_asks": len(asks),
            "complete_book": {
                "bids": bids,
                "asks": asks
            },
            "updated_at": int(time.time() * 1000)
        }

        # Export to asset-specific file
        self.export_book_by_asset(asset_id, book_export)

        if self.verbose:
            print(f"\n📊 ORDER BOOK UPDATE - {asset_name}")
            print(f"   Timestamp: {datetime.fromtimestamp(int(timestamp)/1000).strftime('%H:%M:%S.%f')[:-3]}")
            print(f"   Delay: {delay_ms}ms")
            if best_bid:
                print(f"   🟢 Best Bid: ${best_bid['price']:.3f} (Size: {best_bid['size']:.2f})")
            if best_ask:
                print(f"   🔴 Best Ask: ${best_ask['price']:.3f} (Size: {best_ask['size']:.2f})")
            if spread is not None:
                print(f"   📏 Spread: ${spread:.3f}")
            print(f"   📚 Book: {len(bids)} bids, {len(asks)} asks")
            if bids:
                try:
                    print(f"   📈 Bid range: ${float(bids[0]['price']):.3f} → ${float(bids[-1]['price']):.3f}")
                except (KeyError, ValueError, TypeError, IndexError):
                    pass
            if asks:
                try:
                    print(f"   📉 Ask range: ${float(asks[0]['price']):.3f} → ${float(asks[-1]['price']):.3f}")
                except (KeyError, ValueError, TypeError, IndexError):
                    pass

    def process_trade_message(self, message_data):
        """Process trade messages and update files with same structure"""
        if message_data is None:
            if self.verbose:
                print("⚠️ Received None trade message, updating with current data")
            self.update_all_assets_with_timestamp(None, "NULL_TRADE")
            return

        asset_id = message_data.get("asset_id")
        if asset_id is None or asset_id not in self.asset_ids:
            if self.verbose:
                print(f"⚠️ Trade message missing asset_id or unknown asset, updating all assets")
            self.update_all_assets_with_timestamp(message_data, "TRADE_NO_ASSET")
            return

        asset_name = self.asset_names.get(asset_id, asset_id[:8] + "...")

        timestamp = message_data.get("timestamp")
        if timestamp is None:
            timestamp = int(time.time() * 1000)

        delay_ms = self.calculate_timestamp_delay(timestamp)

        # Create book export with current book data - SAME STRUCTURE
        book_export = self.create_book_export_from_current_data(asset_id, message_data)

        # Export to asset-specific file
        self.export_book_by_asset(asset_id, book_export)

        if self.verbose:
            print(f"\n💰 TRADE - {asset_name}")
            try:
                price = float(message_data.get('price', 0))
                size = float(message_data.get('size', 0))
                side = message_data.get('side', 'UNKNOWN')
                print(f"   Price: ${price:.3f}")
                print(f"   Size: {size:.2f}")
                print(f"   Side: {side}")
            except (ValueError, TypeError):
                print(f"   Raw data: {message_data}")
            print(f"   Delay: {delay_ms}ms")

    def process_price_change_message(self, message_data):
        """Process price change messages and update files with same structure"""
        if message_data is None:
            if self.verbose:
                print("⚠️ Received None price change message, updating with current data")
            self.update_all_assets_with_timestamp(None, "NULL_PRICE_CHANGE")
            return

        asset_id = message_data.get("asset_id")
        if asset_id is None or asset_id not in self.asset_ids:
            if self.verbose:
                print(f"⚠️ Price change message missing asset_id or unknown asset, updating all assets")
            self.update_all_assets_with_timestamp(message_data, "PRICE_CHANGE_NO_ASSET")
            return

        asset_name = self.asset_names.get(asset_id, asset_id[:8] + "...")

        timestamp = message_data.get("timestamp")
        if timestamp is None:
            timestamp = int(time.time() * 1000)

        delay_ms = self.calculate_timestamp_delay(timestamp)

        changes = message_data.get("changes", [])

        # Create book export with current book data - SAME STRUCTURE
        book_export = self.create_book_export_from_current_data(asset_id, message_data)

        # Export to asset-specific file
        self.export_book_by_asset(asset_id, book_export)

        if self.verbose:
            print(f"\n📈 PRICE CHANGE - {asset_name}")
            print(f"   Delay: {delay_ms}ms")
            for change in changes:
                try:
                    side = change.get('side', 'UNKNOWN')
                    price = float(change.get('price', 0))
                    size = float(change.get('size', 0))
                    print(f"   {side}: ${price:.3f} (Size: {size:.2f})")
                except (ValueError, TypeError, KeyError):
                    print(f"   Raw change: {change}")

    def on_message(self, ws, message):
        if message.strip() == "PONG":
            if self.verbose:
                print("🏓 PONG")
            return

        try:
            # Parse JSON message
            data = json.loads(message)

            # Handle both single messages and arrays
            if isinstance(data, list):
                messages = data
            else:
                messages = [data]

            for msg in messages:
                try:
                    # Handle None messages
                    if msg is None:
                        if self.verbose:
                            print("⚠️ Received None message, updating with current data")
                        self.update_all_assets_with_timestamp(None, "NULL_MESSAGE")
                        continue

                    event_type = msg.get("event_type")

                    if event_type == "book":
                        self.process_book_message(msg)
                    elif event_type == "last_trade_price":
                        self.process_trade_message(msg)
                    elif event_type == "price_change":
                        self.process_price_change_message(msg)
                    elif event_type is None:
                        # Handle None event types - still update files with current data
                        if self.verbose:
                            print("⚠️ Received message with None event_type, updating with current data")
                        self.update_all_assets_with_timestamp(msg, "NULL_EVENT_TYPE")
                    else:
                        # Handle unknown event types - still update files with current data
                        if self.verbose:
                            print(f"❓ Unknown event type: {event_type}, updating with current data")
                        self.update_all_assets_with_timestamp(msg, f"UNKNOWN_{event_type}")

                except Exception as e:
                    if self.verbose:
                        print(f"❌ Error processing individual message: {e}")
                    # Still try to update files even on error
                    try:
                        self.update_all_assets_with_timestamp(msg if 'msg' in locals() else None, "ERROR_MESSAGE")
                    except:
                        pass

        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"❌ Failed to parse JSON: {message[:100]}...")
                print(f"   JSON Error: {e}")
            # Update files even on JSON decode error
            self.update_all_assets_with_timestamp(None, "JSON_DECODE_ERROR")
        except Exception as e:
            print(f"❌ Error processing message: {e}")
            # Update files even on general error
            self.update_all_assets_with_timestamp(None, "GENERAL_ERROR")

    def on_error(self, ws, error):
        print(f"❌ WebSocket Error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        print("🔌 WebSocket connection closed")

    def on_open(self, ws):
        print("✅ WebSocket connection opened")
        print("📄 Will create files: NEXT_15M_CALL.json and NEXT_15M_PUT.json")
        print("💾 Files will be updated on ALL event types with SAME structure")
        print("🛡️ Robust handling for None/missing data - files always updated")

        subscription_msg = {"assets_ids": self.asset_ids, "type": MARKET_CHANNEL}
        ws.send(json.dumps(subscription_msg))
        print(f"📡 Subscribed to market data for {len(self.asset_ids)} asset(s)")

        # Debug: Show what we're subscribing to
        print(f"🔍 Assets subscribed (NEXT period):")
        for asset_id in self.asset_ids:
            asset_name = self.asset_names.get(asset_id, asset_id[:8] + "...")
            print(f"   {asset_name}: {asset_id}")

        # Start ping thread
        ping_thread = threading.Thread(target=self.ping_loop, args=(ws,))
        ping_thread.daemon = True
        ping_thread.start()

    def ping_loop(self, ws):
        """Send periodic PING messages"""
        while True:
            try:
                ws.send("PING")
                time.sleep(10)
            except Exception as e:
                print(f"❌ Ping failed: {e}")
                break

    def connect_websocket(self):
        """Create and connect WebSocket"""
        if not self.asset_ids:
            print("❌ No asset IDs available. Cannot connect.")
            print("⏳ NEXT period market may not exist yet. Waiting for next update cycle...")
            return

        furl = self.url + "/ws/" + MARKET_CHANNEL
        self.ws = WebSocketApp(
            furl,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open,
        )

        self.ws.run_forever()

    def check_quarter_hourly_update(self):
        """Check if we need to update market data (at each quarter hour: 00, 15, 30, 45)"""
        utc_time = self.get_utc_time()
        current_quarter = self.get_current_quarter_hour(utc_time)
        current_minute = utc_time.minute

        # Update when we reach a new quarter hour (00, 15, 30, 45) and haven't updated for this quarter yet
        if current_minute == current_quarter and current_quarter != self.last_update_quarter:
            if self.verbose:
                print(f"\n🔄 Quarter hour changed to {utc_time.hour}:{current_quarter:02d} UTC - Updating NEXT market data...")

            self.last_update_quarter = current_quarter
            self.update_market_data_and_reconnect()

    def quarter_hourly_monitor(self):
        """Monitor for quarter-hourly updates"""
        while True:
            try:
                self.check_quarter_hourly_update()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                print(f"❌ Error in quarter-hourly monitor: {e}")
                time.sleep(60)  # Wait longer on error

    def run(self):
        """Start the WebSocket monitor with dynamic updates"""
        print("🚀 Starting Dynamic Polymarket Monitor for NEXT Period...")
        print("📊 Monitoring NEXT 15-minute CALL and PUT tokens")
        print("   (If now is minute 4, monitoring the 15-30 option)")
        print("🔄 Quarter-hourly updates at (00, 15, 30, 45) UTC")
        print("🔍 Files update on ALL events but keep EXACT same structure")
        print("🛡️ Robust error handling - files updated even on None/error events")

        # Start quarter-hourly monitor thread
        monitor_thread = threading.Thread(target=self.quarter_hourly_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()

        # If we don't have asset IDs yet, wait and retry
        retry_count = 0
        max_retries = 10
        while not self.asset_ids and retry_count < max_retries:
            print(f"⏳ NEXT period market not available yet. Retry {retry_count + 1}/{max_retries}...")
            time.sleep(30)
            self.update_market_data()
            retry_count += 1

        if not self.asset_ids:
            print("❌ Failed to initialize NEXT market data after retries.")
            print("⚠️ The NEXT period market may not exist yet on Polymarket.")
            return

        # Start WebSocket connection
        self.connect_websocket()


if __name__ == "__main__":
    # Configuration
    url = "wss://ws-subscriptions-clob.polymarket.com"

    # API credentials
    api_key = "b4ade5ec-0d70-fd17-fcde-ec71e36c3ce4"
    api_secret = "VCt8m1QtPUhkFQkWTgzVOt1CwSuSI7kskPUFzE5cy9w="
    api_passphrase = "cf18b43b4e7da0483fb1e0289d01f68accde03d0896f084f749e367d6cbf670b"

    auth = {
        "apiKey": api_key,
        "secret": api_secret,
        "passphrase": api_passphrase
    }

    # Create and run the monitor
    try:
        monitor = DynamicPolymarketMonitorNext(url, auth, verbose=True)
        monitor.run()
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
