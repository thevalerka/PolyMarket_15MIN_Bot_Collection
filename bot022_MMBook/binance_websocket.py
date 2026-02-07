import websocket
import json
import threading
import os
from datetime import datetime
from collections import deque
import time

class BinanceWebSocket:
    def __init__(self):
        self.ws = None
        self.url = "wss://stream.binance.com:9443/stream"
        self.last_price = None  # Store the latest price data
        
        # Store trades with timestamps for rolling window analysis
        # Each entry: {'timestamp': ms, 'is_buy': bool, 'size': float, 'price': float}
        self.trades_buffer = deque(maxlen=10000)  # Keep last 10k trades
        self.buffer_lock = threading.Lock()

    def _clean_old_trades(self, current_time_ms, max_age_seconds=35):
        """Remove trades older than max_age_seconds"""
        cutoff = current_time_ms - (max_age_seconds * 1000)
        with self.buffer_lock:
            while self.trades_buffer and self.trades_buffer[0]['timestamp'] < cutoff:
                self.trades_buffer.popleft()

    def _calculate_stats(self, current_time_ms):
        """Calculate buy/sell stats for different time windows"""
        stats = {
            '1s': {'buy_count': 0, 'sell_count': 0, 'buy_volume': 0.0, 'sell_volume': 0.0},
            '5s': {'buy_count': 0, 'sell_count': 0, 'buy_volume': 0.0, 'sell_volume': 0.0},
            '15s': {'buy_count': 0, 'sell_count': 0, 'buy_volume': 0.0, 'sell_volume': 0.0},
            '30s': {'buy_count': 0, 'sell_count': 0, 'buy_volume': 0.0, 'sell_volume': 0.0},
        }
        
        cutoff_1s = current_time_ms - 1000
        cutoff_5s = current_time_ms - 5000
        cutoff_15s = current_time_ms - 15000
        cutoff_30s = current_time_ms - 30000
        
        with self.buffer_lock:
            for trade in self.trades_buffer:
                ts = trade['timestamp']
                is_buy = trade['is_buy']
                size = trade['size']
                
                # 30s window (includes all)
                if ts >= cutoff_30s:
                    if is_buy:
                        stats['30s']['buy_count'] += 1
                        stats['30s']['buy_volume'] += size
                    else:
                        stats['30s']['sell_count'] += 1
                        stats['30s']['sell_volume'] += size
                    
                    # 15s window
                    if ts >= cutoff_15s:
                        if is_buy:
                            stats['15s']['buy_count'] += 1
                            stats['15s']['buy_volume'] += size
                        else:
                            stats['15s']['sell_count'] += 1
                            stats['15s']['sell_volume'] += size
                        
                        # 5s window
                        if ts >= cutoff_5s:
                            if is_buy:
                                stats['5s']['buy_count'] += 1
                                stats['5s']['buy_volume'] += size
                            else:
                                stats['5s']['sell_count'] += 1
                                stats['5s']['sell_volume'] += size
                            
                            # 1s window
                            if ts >= cutoff_1s:
                                if is_buy:
                                    stats['1s']['buy_count'] += 1
                                    stats['1s']['buy_volume'] += size
                                else:
                                    stats['1s']['sell_count'] += 1
                                    stats['1s']['sell_volume'] += size
        
        return stats

    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)

            # Handle subscription confirmation
            if 'id' in data and data.get('id') == 1:
                if 'result' in data:
                    print(f"✅ Subscription successful: {data}")
                elif 'error' in data:
                    print(f"❌ Subscription error: {data['error']}")
                return

            # Handle stream data
            if 'stream' in data and 'data' in data:
                stream_name = data['stream']
                stream_data = data['data']

                if 'aggtrade' in stream_name.lower():
                    # Extract trade data
                    symbol = stream_data['s']
                    price = float(stream_data['p'])
                    quantity = float(stream_data['q'])
                    timestamp = stream_data['T']  # Unix timestamp in milliseconds
                    is_market_maker = stream_data['m']  # True = SELL, False = BUY
                    
                    is_buy = not is_market_maker  # Invert: m=False means BUY
                    trade_side = "BUY" if is_buy else "SELL"
                    
                    # Add trade to buffer
                    trade_entry = {
                        'timestamp': timestamp,
                        'is_buy': is_buy,
                        'size': quantity,
                        'price': price
                    }
                    with self.buffer_lock:
                        self.trades_buffer.append(trade_entry)
                    
                    # Clean old trades
                    self._clean_old_trades(timestamp)
                    
                    # Calculate stats
                    stats = self._calculate_stats(timestamp)
                    
                    # Calculate volume comparisons
                    volume_1s = stats['1s']['buy_volume'] + stats['1s']['sell_volume']
                    volume_15s = stats['15s']['buy_volume'] + stats['15s']['sell_volume']
                    volume_30s = stats['30s']['buy_volume'] + stats['30s']['sell_volume']
                    
                    # Average volume per second over 15s and 30s
                    avg_volume_per_second_15s = volume_15s / 15.0 if volume_15s > 0 else 0
                    avg_volume_per_second_30s = volume_30s / 30.0 if volume_30s > 0 else 0
                    
                    # Ratio of 1s volume to average 15s and 30s volume
                    volume_ratio_1s_vs_15s_avg = (volume_1s / avg_volume_per_second_15s) if avg_volume_per_second_15s > 0 else 0
                    volume_ratio_1s_vs_30s_avg = (volume_1s / avg_volume_per_second_30s) if avg_volume_per_second_30s > 0 else 0
                    
                    # Print summary
                    print(f"\n{'='*60}")
                    print(f"🟢 {symbol} | {trade_side} | Price: {price:.2f} | Qty: {quantity:.5f}")
                    print(f"📊 Last 1s:  BUY: {stats['1s']['buy_count']} ({stats['1s']['buy_volume']:.5f}) | SELL: {stats['1s']['sell_count']} ({stats['1s']['sell_volume']:.5f})")
                    print(f"📊 Last 5s:  BUY: {stats['5s']['buy_count']} ({stats['5s']['buy_volume']:.5f}) | SELL: {stats['5s']['sell_count']} ({stats['5s']['sell_volume']:.5f})")
                    print(f"📊 Last 15s: BUY: {stats['15s']['buy_count']} ({stats['15s']['buy_volume']:.5f}) | SELL: {stats['15s']['sell_count']} ({stats['15s']['sell_volume']:.5f})")
                    print(f"📊 Last 30s: BUY: {stats['30s']['buy_count']} ({stats['30s']['buy_volume']:.5f}) | SELL: {stats['30s']['sell_count']} ({stats['30s']['sell_volume']:.5f})")
                    print(f"📈 1s Vol: {volume_1s:.5f} | 15s Avg/s: {avg_volume_per_second_15s:.5f} | 30s Avg/s: {avg_volume_per_second_30s:.5f}")
                    print(f"📈 Ratio 1s vs 15s: {volume_ratio_1s_vs_15s_avg:.2f}x | Ratio 1s vs 30s: {volume_ratio_1s_vs_30s_avg:.2f}x")

                    # Build comprehensive price data
                    price_data = {
                        'price': price,
                        'timestamp': timestamp,
                        'datetime': datetime.fromtimestamp(timestamp / 1000).isoformat(),
                        'symbol': symbol,
                        'last_trade': {
                            'side': trade_side,
                            'quantity': quantity,
                            'price': price,
                            'is_buy': is_buy
                        },
                        'stats_1s': {
                            'buy_count': stats['1s']['buy_count'],
                            'sell_count': stats['1s']['sell_count'],
                            'buy_volume': round(stats['1s']['buy_volume'], 8),
                            'sell_volume': round(stats['1s']['sell_volume'], 8),
                            'total_volume': round(volume_1s, 8),
                            'net_volume': round(stats['1s']['buy_volume'] - stats['1s']['sell_volume'], 8),
                            'buy_sell_count_ratio': round(stats['1s']['buy_count'] / stats['1s']['sell_count'], 4) if stats['1s']['sell_count'] > 0 else None,
                            'buy_sell_volume_ratio': round(stats['1s']['buy_volume'] / stats['1s']['sell_volume'], 4) if stats['1s']['sell_volume'] > 0 else None
                        },
                        'stats_5s': {
                            'buy_count': stats['5s']['buy_count'],
                            'sell_count': stats['5s']['sell_count'],
                            'buy_volume': round(stats['5s']['buy_volume'], 8),
                            'sell_volume': round(stats['5s']['sell_volume'], 8),
                            'total_volume': round(stats['5s']['buy_volume'] + stats['5s']['sell_volume'], 8),
                            'net_volume': round(stats['5s']['buy_volume'] - stats['5s']['sell_volume'], 8),
                            'buy_sell_count_ratio': round(stats['5s']['buy_count'] / stats['5s']['sell_count'], 4) if stats['5s']['sell_count'] > 0 else None,
                            'buy_sell_volume_ratio': round(stats['5s']['buy_volume'] / stats['5s']['sell_volume'], 4) if stats['5s']['sell_volume'] > 0 else None
                        },
                        'stats_15s': {
                            'buy_count': stats['15s']['buy_count'],
                            'sell_count': stats['15s']['sell_count'],
                            'buy_volume': round(stats['15s']['buy_volume'], 8),
                            'sell_volume': round(stats['15s']['sell_volume'], 8),
                            'total_volume': round(volume_15s, 8),
                            'net_volume': round(stats['15s']['buy_volume'] - stats['15s']['sell_volume'], 8),
                            'buy_sell_count_ratio': round(stats['15s']['buy_count'] / stats['15s']['sell_count'], 4) if stats['15s']['sell_count'] > 0 else None,
                            'buy_sell_volume_ratio': round(stats['15s']['buy_volume'] / stats['15s']['sell_volume'], 4) if stats['15s']['sell_volume'] > 0 else None
                        },
                        'stats_30s': {
                            'buy_count': stats['30s']['buy_count'],
                            'sell_count': stats['30s']['sell_count'],
                            'buy_volume': round(stats['30s']['buy_volume'], 8),
                            'sell_volume': round(stats['30s']['sell_volume'], 8),
                            'total_volume': round(volume_30s, 8),
                            'net_volume': round(stats['30s']['buy_volume'] - stats['30s']['sell_volume'], 8),
                            'buy_sell_count_ratio': round(stats['30s']['buy_count'] / stats['30s']['sell_count'], 4) if stats['30s']['sell_count'] > 0 else None,
                            'buy_sell_volume_ratio': round(stats['30s']['buy_volume'] / stats['30s']['sell_volume'], 4) if stats['30s']['sell_volume'] > 0 else None
                        },
                        'volume_analysis': {
                            'volume_1s': round(volume_1s, 8),
                            'volume_15s': round(volume_15s, 8),
                            'volume_30s': round(volume_30s, 8),
                            'avg_volume_per_second_15s': round(avg_volume_per_second_15s, 8),
                            'avg_volume_per_second_30s': round(avg_volume_per_second_30s, 8),
                            'ratio_1s_vs_15s_avg': round(volume_ratio_1s_vs_15s_avg, 4),
                            'ratio_1s_vs_30s_avg': round(volume_ratio_1s_vs_30s_avg, 4),
                            'is_volume_spike': volume_ratio_1s_vs_30s_avg > 2.0  # Flag if 1s volume is 2x the 30s average
                        }
                    }

                    # Save to binance_BTC_details.json
                    try:
                        with open('binance_BTC_details.json', 'w') as f:
                            json.dump(price_data, f, indent=2)
                    except Exception as e:
                        print(f"❌ Error saving to file: {e}")

                    # Store the last price
                    self.last_price = price_data

                elif 'depth' in stream_name.lower():
                    # Silently ignore depth updates or uncomment to see them
                    # print(f"📊 Depth Update - Symbol: {stream_data['s']}, Bids: {len(stream_data['b'])}, Asks: {len(stream_data['a'])}")
                    pass

            else:
                # Print any other messages for debugging
                print(f"📝 Other message type: {data}")

        except json.JSONDecodeError:
            print(f"❌ Error decoding JSON: {message}")
        except KeyError as e:
            print(f"❌ Missing key in data: {e}")
            print(f"Raw message: {message}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            print(f"Raw message: {message}")

    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        print(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        print("WebSocket connection closed")

    def on_open(self, ws):
        """Handle WebSocket open and send subscription"""
        print("WebSocket connection opened")

        # Send subscription message - only aggTrade needed for this analysis
        subscribe_message = {
            "method": "SUBSCRIBE",
            "params": [
                "btcusdt@aggTrade"
            ],
            "id": 1
        }

        ws.send(json.dumps(subscribe_message))
        print("Subscription message sent")

    def get_last_price(self):
        """Get the most recent price data"""
        return self.last_price

    def start(self, debug=False):
        """Start the WebSocket connection"""
        if debug:
            websocket.enableTrace(True)
        self.ws = websocket.WebSocketApp(
            self.url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )

        # Run forever
        self.ws.run_forever()

    def stop(self):
        """Stop the WebSocket connection"""
        if self.ws:
            self.ws.close()


# Usage
if __name__ == "__main__":
    print("Starting Binance WebSocket connection with BUY/SELL tracking...")
    binance_ws = BinanceWebSocket()

    try:
        binance_ws.start(debug=False)
    except KeyboardInterrupt:
        print("\nStopping WebSocket...")

        last_price_data = binance_ws.get_last_price()
        if last_price_data:
            print(f"Final price data saved to binance_BTC_details.json")

        binance_ws.stop()
