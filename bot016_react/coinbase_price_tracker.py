import websocket
import json
import time
from datetime import datetime

#pm2 start coinbase_price_tracker.py --cron-restart="00 * * * *" --interpreter python3

class CoinbaseBTCPriceTracker:
    def __init__(self):
        self.ws = None
        self.url = "wss://ws-feed.exchange.coinbase.com"
        self.output_file = "coinbase_btc_price.json"
        self.last_price = None

    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)

            # Handle subscription confirmation
            if data.get('type') == 'subscriptions':
                print(f"✅ Subscribed to: {data.get('channels', [])}")
                return

            # Handle ticker messages (real-time price updates)
            if data.get('type') == 'ticker':
                self._process_price_update(data)

            # Handle match messages (actual trade executions)
            elif data.get('type') == 'match':
                self._process_price_update(data)

        except json.JSONDecodeError as e:
            print(f"❌ Error decoding JSON: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

    def _process_price_update(self, data):
        """Process price update and save to file"""
        try:
            # Extract price
            price = float(data.get('price', 0))

            # Extract and convert timestamp
            time_str = data.get('time')  # ISO format: 2024-01-27T12:34:56.789012Z
            if time_str:
                # Parse ISO format and convert to Unix timestamp in milliseconds
                dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                timestamp_ms = int(dt.timestamp() * 1000)
            else:
                # Fallback to current time if no timestamp
                timestamp_ms = int(time.time() * 1000)

            # Create price data object
            price_data = {
                "price": price,
                "timestamp": timestamp_ms,
                "symbol": "BTCUSDT",
                "symbol-Coinbase": "BTC-USD"
            }

            # Save to JSON file
            with open(self.output_file, 'w') as f:
                json.dump(price_data, f, indent=2)

            # Store for reference
            self.last_price = price_data

            # Print update
            message_type = data.get('type', 'unknown')
            readable_time = datetime.fromtimestamp(timestamp_ms / 1000).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            print(f"💰 {message_type.upper()}: ${price:,.2f} | {readable_time} | Saved to {self.output_file}")

        except Exception as e:
            print(f"❌ Error processing price update: {e}")

    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        print(f"❌ WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        print(f"🔴 WebSocket connection closed (code: {close_status_code}, msg: {close_msg})")

    def on_open(self, ws):
        """Handle WebSocket open and send subscription"""
        print("🟢 WebSocket connection opened")
        print("📡 Subscribing to Coinbase BTC-USD...")

        # Subscribe to ticker and matches channels
        subscribe_message = {
            "type": "subscribe",
            "product_ids": ["BTC-USD"],
            "channels": [
                "ticker",      # Real-time price updates
                "matches"      # Actual trade executions
            ]
        }

        ws.send(json.dumps(subscribe_message))
        print("✅ Subscription message sent")

    def get_last_price(self):
        """Get the most recent price data"""
        return self.last_price

    def start(self, debug=False):
        """Start the WebSocket connection"""
        print("="*60)
        print("Coinbase BTC Price Tracker")
        print("="*60)
        print(f"Output file: {self.output_file}")
        print("Press Ctrl+C to stop\n")

        if debug:
            websocket.enableTrace(True)

        self.ws = websocket.WebSocketApp(
            self.url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )

        # Run forever (blocking)
        self.ws.run_forever()

    def stop(self):
        """Stop the WebSocket connection"""
        if self.ws:
            self.ws.close()
            print("\n🛑 Stopping WebSocket connection...")


# Simple function-based alternative
def simple_coinbase_tracker():
    """Simple function-based implementation"""
    output_file = "coinbase_btc_price.json"

    def on_message(ws, message):
        try:
            data = json.loads(message)

            if data.get('type') in ['ticker', 'match']:
                price = float(data.get('price', 0))
                time_str = data.get('time')

                if time_str:
                    dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                    timestamp_ms = int(dt.timestamp() * 1000)
                else:
                    timestamp_ms = int(time.time() * 1000)

                price_data = {
                    "price": price,
                    "timestamp": timestamp_ms,
                    "symbol": "BTCUSDT",
                    "symbol-Coinbase": "BTC-USD"
                }

                with open(output_file, 'w') as f:
                    json.dump(price_data, f, indent=2)

                print(f"💰 ${price:,.2f} | Saved to {output_file}")

        except Exception as e:
            print(f"❌ Error: {e}")

    def on_error(ws, error):
        print(f"❌ Error: {error}")

    def on_close(ws, close_status_code, close_msg):
        print(f"🔴 Connection closed")

    def on_open(ws):
        print("🟢 Connected to Coinbase")
        subscribe_msg = {
            "type": "subscribe",
            "product_ids": ["BTC-USD"],
            "channels": ["ticker", "matches"]
        }
        ws.send(json.dumps(subscribe_msg))
        print("✅ Subscribed to BTC-USD")

    ws = websocket.WebSocketApp(
        "wss://ws-feed.exchange.coinbase.com",
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    print("="*60)
    print("Coinbase BTC Price Tracker (Simple)")
    print("="*60)
    print(f"Output file: {output_file}")
    print("Press Ctrl+C to stop\n")

    ws.run_forever()


# Usage
if __name__ == "__main__":
    # Option 1: Using the class (recommended)
    tracker = CoinbaseBTCPriceTracker()

    try:
        # Set debug=True to see detailed WebSocket trace
        tracker.start(debug=False)
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user")

        # Print the last received price before closing
        last_price_data = tracker.get_last_price()
        if last_price_data:
            print(f"\nFinal price data:")
            print(json.dumps(last_price_data, indent=2))

        tracker.stop()

    # Option 2: Using the simple function
    # simple_coinbase_tracker()
