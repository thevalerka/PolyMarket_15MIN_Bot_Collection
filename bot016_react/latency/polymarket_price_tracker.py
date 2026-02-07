import websocket
import json
import time
from datetime import datetime

class PolymarketBTCPriceTracker:
    def __init__(self):
        self.ws_crypto = None
        self.ws_chainlink = None
        self.url = "wss://ws-live-data.polymarket.com"
        self.crypto_output_file = "polymarket_crypto_btc_price.json"
        self.chainlink_output_file = "polymarket_chainlink_btc_price.json"
        
    # ============================================================
    # CRYPTO_PRICES STREAM
    # ============================================================
    
    def on_message_crypto(self, ws, message):
        """Handle incoming messages from crypto_prices stream"""
        try:
            data = json.loads(message)
            
            # Handle crypto_prices updates
            if data.get('topic') == 'crypto_prices' and data.get('type') == 'update':
                payload = data.get('payload', {})
                
                # Extract data
                price = payload.get('value', 0)
                timestamp = payload.get('timestamp', int(time.time() * 1000))
                symbol = payload.get('symbol', 'btcusdt')
                
                # Create price data object
                price_data = {
                    "price": float(price),
                    "timestamp": timestamp,
                    "symbol": "BTCUSDT",
                    "symbol-Polymarket": symbol
                }
                
                # Save to JSON file
                with open(self.crypto_output_file, 'w') as f:
                    json.dump(price_data, f, indent=2)
                
                # Print update
                readable_time = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                print(f"💰 CRYPTO_PRICES: ${price:,.2f} | {readable_time} | Saved to {self.crypto_output_file}")
                
        except Exception as e:
            print(f"❌ Error (crypto_prices): {e}")
    
    def on_error_crypto(self, ws, error):
        print(f"❌ WebSocket error (crypto_prices): {error}")
    
    def on_close_crypto(self, ws, close_status_code, close_msg):
        print(f"🔴 crypto_prices connection closed")
    
    def on_open_crypto(self, ws):
        print("🟢 crypto_prices connection opened")
        subscribe_message = {
            "action": "subscribe",
            "subscriptions": [
                {
                    "topic": "crypto_prices",
                    "type": "update",
                    "filters": "btcusdt"
                }
            ]
        }
        ws.send(json.dumps(subscribe_message))
        print("✅ Subscribed to crypto_prices (btcusdt)")
    
    # ============================================================
    # CRYPTO_PRICES_CHAINLINK STREAM
    # ============================================================
    
    def on_message_chainlink(self, ws, message):
        """Handle incoming messages from chainlink stream"""
        try:
            data = json.loads(message)
            
            # Handle chainlink updates
            if data.get('topic') == 'crypto_prices_chainlink' and data.get('type') == 'update':
                payload = data.get('payload', {})
                
                # Extract data
                price = payload.get('value', 0)
                timestamp = payload.get('timestamp', int(time.time() * 1000))
                symbol = payload.get('symbol', 'btc/usd')
                
                # Create price data object
                price_data = {
                    "price": float(price),
                    "timestamp": timestamp,
                    "symbol": "BTCUSDT",
                    "symbol-Polymarket-Chainlink": symbol
                }
                
                # Save to JSON file
                with open(self.chainlink_output_file, 'w') as f:
                    json.dump(price_data, f, indent=2)
                
                # Print update
                readable_time = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                print(f"🔗 CHAINLINK: ${price:,.2f} | {readable_time} | Saved to {self.chainlink_output_file}")
                
        except Exception as e:
            print(f"❌ Error (chainlink): {e}")
    
    def on_error_chainlink(self, ws, error):
        print(f"❌ WebSocket error (chainlink): {error}")
    
    def on_close_chainlink(self, ws, close_status_code, close_msg):
        print(f"🔴 chainlink connection closed")
    
    def on_open_chainlink(self, ws):
        print("🟢 chainlink connection opened")
        subscribe_message = {
            "action": "subscribe",
            "subscriptions": [
                {
                    "topic": "crypto_prices_chainlink",
                    "type": "*",
                    "filters": "{\"symbol\":\"btc/usd\"}"
                }
            ]
        }
        ws.send(json.dumps(subscribe_message))
        print("✅ Subscribed to crypto_prices_chainlink (btc/usd)")
    
    # ============================================================
    # MAIN CONTROL
    # ============================================================
    
    def start(self, debug=False):
        """Start both WebSocket connections"""
        print("="*70)
        print("Polymarket BTC Price Tracker - Dual Stream")
        print("="*70)
        print(f"Stream 1: crypto_prices → {self.crypto_output_file}")
        print(f"Stream 2: crypto_prices_chainlink → {self.chainlink_output_file}")
        print("Press Ctrl+C to stop\n")
        
        if debug:
            websocket.enableTrace(True)
        
        # Create both WebSocket connections
        self.ws_crypto = websocket.WebSocketApp(
            self.url,
            on_open=self.on_open_crypto,
            on_message=self.on_message_crypto,
            on_error=self.on_error_crypto,
            on_close=self.on_close_crypto
        )
        
        self.ws_chainlink = websocket.WebSocketApp(
            self.url,
            on_open=self.on_open_chainlink,
            on_message=self.on_message_chainlink,
            on_error=self.on_error_chainlink,
            on_close=self.on_close_chainlink
        )
        
        # Run both connections in threads
        import threading
        
        crypto_thread = threading.Thread(target=self.ws_crypto.run_forever)
        chainlink_thread = threading.Thread(target=self.ws_chainlink.run_forever)
        
        crypto_thread.daemon = True
        chainlink_thread.daemon = True
        
        crypto_thread.start()
        chainlink_thread.start()
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping both connections...")
            self.stop()
    
    def stop(self):
        """Stop both WebSocket connections"""
        if self.ws_crypto:
            self.ws_crypto.close()
        if self.ws_chainlink:
            self.ws_chainlink.close()


# Simple combined tracker alternative
def simple_dual_tracker():
    """Simple function-based implementation for both streams"""
    crypto_output = "polymarket_crypto_btc_price.json"
    chainlink_output = "polymarket_chainlink_btc_price.json"
    
    def save_crypto_price(data):
        try:
            if data.get('topic') == 'crypto_prices' and data.get('type') == 'update':
                payload = data.get('payload', {})
                price_data = {
                    "price": float(payload.get('value', 0)),
                    "timestamp": payload.get('timestamp', int(time.time() * 1000)),
                    "symbol": "BTCUSDT",
                    "symbol-Polymarket": payload.get('symbol', 'btcusdt')
                }
                with open(crypto_output, 'w') as f:
                    json.dump(price_data, f, indent=2)
                print(f"💰 CRYPTO: ${price_data['price']:,.2f} → {crypto_output}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def save_chainlink_price(data):
        try:
            if data.get('topic') == 'crypto_prices_chainlink' and data.get('type') == 'update':
                payload = data.get('payload', {})
                price_data = {
                    "price": float(payload.get('value', 0)),
                    "timestamp": payload.get('timestamp', int(time.time() * 1000)),
                    "symbol": "BTCUSDT",
                    "symbol-Polymarket-Chainlink": payload.get('symbol', 'btc/usd')
                }
                with open(chainlink_output, 'w') as f:
                    json.dump(price_data, f, indent=2)
                print(f"🔗 CHAINLINK: ${price_data['price']:,.2f} → {chainlink_output}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def on_message_crypto(ws, message):
        data = json.loads(message)
        save_crypto_price(data)
    
    def on_message_chainlink(ws, message):
        data = json.loads(message)
        save_chainlink_price(data)
    
    def on_open_crypto(ws):
        print("🟢 Connected to crypto_prices")
        ws.send(json.dumps({
            "action": "subscribe",
            "subscriptions": [{
                "topic": "crypto_prices",
                "type": "update",
                "filters": "btcusdt"
            }]
        }))
    
    def on_open_chainlink(ws):
        print("🟢 Connected to chainlink")
        ws.send(json.dumps({
            "action": "subscribe",
            "subscriptions": [{
                "topic": "crypto_prices_chainlink",
                "type": "*",
                "filters": "{\"symbol\":\"btc/usd\"}"
            }]
        }))
    
    # Create connections
    ws_crypto = websocket.WebSocketApp(
        "wss://ws-live-data.polymarket.com",
        on_open=on_open_crypto,
        on_message=on_message_crypto
    )
    
    ws_chainlink = websocket.WebSocketApp(
        "wss://ws-live-data.polymarket.com",
        on_open=on_open_chainlink,
        on_message=on_message_chainlink
    )
    
    # Run in threads
    import threading
    threading.Thread(target=ws_crypto.run_forever, daemon=True).start()
    threading.Thread(target=ws_chainlink.run_forever, daemon=True).start()
    
    print("="*70)
    print("Polymarket Dual Tracker (Simple)")
    print("="*70)
    print(f"Output 1: {crypto_output}")
    print(f"Output 2: {chainlink_output}")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")


# Usage
if __name__ == "__main__":
    # Option 1: Using the class (recommended)
    tracker = PolymarketBTCPriceTracker()
    tracker.start(debug=False)
    
    # Option 2: Using the simple function
    # simple_dual_tracker()
