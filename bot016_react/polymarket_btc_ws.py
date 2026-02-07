#!/usr/bin/env python3
"""
Polymarket BTC/USD WebSocket Client
Connects to Polymarket's live data feed and atomically writes BTC price updates
Tracks 15-minute strike prices at :00, :15, :30, :45
"""

import json
import asyncio
import websockets
import os
from datetime import datetime

# Configuration
WS_URL = "wss://ws-live-data.polymarket.com"
OUTPUT_FILE = "/home/ubuntu/013_2025_polymarket/chainlink_btc_price.json"

# Strike tracking
strike_intervals = [0, 15, 30, 45]  # Minutes for strike prices
strike_window = 5  # Max seconds tolerance to find strike
current_strike = None
current_strike_timestamp = None

async def connect_and_subscribe():
    """Connect to Polymarket WebSocket and handle price updates"""
    
    global current_strike, current_strike_timestamp
    
    def is_strike_time(timestamp_ms):
        """Check if timestamp is at a strike interval (00, 15, 30, 45 minutes, 0 seconds)"""
        dt = datetime.fromtimestamp(timestamp_ms / 1000.0)
        minute = dt.minute
        second = dt.second
        
        # Check if minute is at a strike interval
        if minute in strike_intervals:
            # Check if seconds are within tolerance window (0 +/- 5 seconds)
            if second <= strike_window:
                return True
        return False
    
    def update_strike(price, timestamp):
        """Update strike price if this is a new strike interval"""
        global current_strike, current_strike_timestamp
        
        dt = datetime.fromtimestamp(timestamp / 1000.0)
        minute = dt.minute
        
        # Check if we're at a strike minute
        if minute in strike_intervals:
            # Create a canonical timestamp for this strike (set seconds to 0)
            strike_dt = dt.replace(second=0, microsecond=0)
            strike_ts = int(strike_dt.timestamp() * 1000)
            
            # Only update if this is a new strike or we don't have one yet
            if current_strike_timestamp != strike_ts:
                current_strike = price
                current_strike_timestamp = strike_ts
                return True
        return False
    
    print(f"[{datetime.now()}] Connecting to {WS_URL}...")
    
    async with websockets.connect(WS_URL) as websocket:
        print(f"[{datetime.now()}] Connected successfully!")
        
        # Subscribe to BTC/USD prices
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
        
        await websocket.send(json.dumps(subscribe_message))
        print(f"[{datetime.now()}] Subscription message sent")
        
        # Listen for messages
        message_count = 0
        async for message in websocket:
            try:
                data = json.loads(message)
                
                # Check if this is a price update
                if data.get("topic") == "crypto_prices_chainlink" and data.get("type") == "update":
                    payload = data.get("payload", {})
                    
                    # Extract the price data
                    price = payload.get("value")
                    timestamp = payload.get("timestamp")  # Use payload timestamp, not message timestamp
                    symbol = payload.get("symbol")
                    
                    if price is not None and timestamp is not None:
                        # Check and update strike price
                        is_new_strike = update_strike(price, timestamp)
                        
                        # Prepare output data
                        output_data = {
                            "price": price,
                            "timestamp": timestamp,
                            "symbol": "BTCUSDT"
                        }
                        
                        # Add strike data if available
                        if current_strike is not None:
                            output_data["strike"] = current_strike
                            output_data["strike_timestamp"] = current_strike_timestamp
                        
                        # Atomic write
                        temp_file = OUTPUT_FILE + ".tmp"
                        with open(temp_file, 'w') as f:
                            json.dump(output_data, f, indent=2)
                        os.replace(temp_file, OUTPUT_FILE)
                        
                        message_count += 1
                        strike_indicator = " [NEW STRIKE]" if is_new_strike else ""
                        print(f"[{datetime.now()}] Update #{message_count}: ${price:,.2f} @ {timestamp}{strike_indicator}")
                        
                        if is_new_strike:
                            dt = datetime.fromtimestamp(timestamp / 1000.0)
                            print(f"                   Strike updated: ${current_strike:,.2f} at {dt.strftime('%H:%M:%S')}")
                    else:
                        print(f"[{datetime.now()}] Received incomplete data: {data}")
                else:
                    # Log other message types
                    print(f"[{datetime.now()}] Received: {data}")
                    
            except json.JSONDecodeError as e:
                print(f"[{datetime.now()}] JSON decode error: {e}")
            except Exception as e:
                print(f"[{datetime.now()}] Error processing message: {e}")

async def main():
    """Main loop with auto-reconnect"""
    reconnect_delay = 5
    
    while True:
        try:
            await connect_and_subscribe()
        except websockets.exceptions.WebSocketException as e:
            print(f"[{datetime.now()}] WebSocket error: {e}")
            print(f"[{datetime.now()}] Reconnecting in {reconnect_delay} seconds...")
            await asyncio.sleep(reconnect_delay)
        except Exception as e:
            print(f"[{datetime.now()}] Unexpected error: {e}")
            print(f"[{datetime.now()}] Reconnecting in {reconnect_delay} seconds...")
            await asyncio.sleep(reconnect_delay)

if __name__ == "__main__":
    print("=== Polymarket BTC/USD WebSocket Client ===")
    print(f"Output file: {OUTPUT_FILE}")
    print("Press Ctrl+C to stop\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n[{datetime.now()}] Shutting down gracefully...")
