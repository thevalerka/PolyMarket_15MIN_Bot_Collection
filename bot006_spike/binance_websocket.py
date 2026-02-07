# binance_websocket.py - Enhanced WebSocket with Spike Trading Integration
"""
Enhanced Binance WebSocket for BTC Spike Trading

Features:
1. Real-time BTC price monitoring via WebSocket
2. Integration with spike detection system
3. Automatic spike trigger callbacks
4. Price history management for candlestick analysis
5. Connection resilience and error handling
6. Trading bot integration hooks

Integration with Spike Trading:
- Feeds real-time price data to SpikeDetector
- Triggers spike buy logic when thresholds exceeded
- Maintains price history for candlestick analysis
- Provides connection status monitoring
"""

import websocket
import json
import threading
import time
import os
from datetime import datetime
from typing import Optional, Callable, Dict, List
from collections import deque

class EnhancedBinanceWebSocket:
    """Enhanced WebSocket client with spike trading integration."""
    
    def __init__(self, spike_callback: Optional[Callable] = None):
        self.ws = None
        self.url = "wss://stream.binance.com:9443/stream"
        
        # Price data storage
        self.last_price = None
        self.price_history = deque(maxlen=1000)  # Store last 1000 price points
        
        # Connection management
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 5  # seconds
        
        # Spike integration
        self.spike_callback = spike_callback
        self.last_spike_check = 0
        self.spike_check_interval = 1.0  # Check for spikes every second
        
        # Statistics
        self.message_count = 0
        self.error_count = 0
        self.last_message_time = 0
        
        print("🔗 Enhanced Binance WebSocket initialized")
        
    def set_spike_callback(self, callback: Callable):
        """Set the callback function for spike detection."""
        self.spike_callback = callback
        print("✅ Spike callback registered")
        
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages with spike integration."""
        try:
            self.message_count += 1
            self.last_message_time = time.time()
            
            data = json.loads(message)
            
            # Handle subscription confirmation
            if 'id' in data and data.get('id') == 1:
                if 'result' in data:
                    print(f"✅ Subscription successful: {data}")
                    self.connected = True
                    self.reconnect_attempts = 0
                elif 'error' in data:
                    print(f"❌ Subscription error: {data['error']}")
                    self.error_count += 1
                return
            
            # Handle stream data
            if 'stream' in data and 'data' in data:
                stream_name = data['stream']
                stream_data = data['data']
                
                if 'aggtrade' in stream_name.lower():
                    self._handle_trade_data(stream_data)
                elif 'depth' in stream_name.lower():
                    self._handle_depth_data(stream_data)
                else:
                    print(f"❓ Unknown stream type: {stream_name}")
                    
        except json.JSONDecodeError:
            print(f"❌ Error decoding JSON: {message}")
            self.error_count += 1
        except Exception as e:
            print(f"❌ Unexpected error in on_message: {e}")
            self.error_count += 1
    
    def _handle_trade_data(self, stream_data):
        """Handle aggregate trade data and trigger spike detection."""
        try:
            symbol = stream_data['s']
            price = float(stream_data['p'])
            quantity = stream_data['q']
            timestamp = stream_data['T']  # Unix timestamp in milliseconds
            
            # Create price data object
            price_data = {
                'price': price,
                'timestamp': timestamp,
                'datetime': datetime.fromtimestamp(timestamp / 1000).isoformat(),
                'symbol': symbol,
                'quantity': float(quantity)
            }
            
            # Update internal state
            self.last_price = price_data
            self.price_history.append((timestamp, price))
            
            # Save to file for other components
            self._save_price_to_file(price_data)
            
            # Check for spikes and trigger callback
            current_time = time.time()
            if (self.spike_callback and 
                current_time - self.last_spike_check > self.spike_check_interval):
                
                try:
                    self.spike_callback(price_data)
                    self.last_spike_check = current_time
                except Exception as e:
                    print(f"❌ Error in spike callback: {e}")
            
            # Display price update (less verbose than original)
            if self.message_count % 10 == 0:  # Show every 10th message
                print(f"💰 BTC: ${price:,.2f} | Messages: {self.message_count}")
                
        except Exception as e:
            print(f"❌ Error handling trade data: {e}")
            self.error_count += 1
    
    def _handle_depth_data(self, stream_data):
        """Handle order book depth data."""
        try:
            symbol = stream_data['s']
            bids_count = len(stream_data.get('b', []))
            asks_count = len(stream_data.get('a', []))
            
            # Less verbose depth logging
            if self.message_count % 50 == 0:  # Show every 50th depth update
                print(f"📊 {symbol} Depth: {bids_count} bids, {asks_count} asks")
                
        except Exception as e:
            print(f"❌ Error handling depth data: {e}")
    
    def _save_price_to_file(self, price_data):
        """Save price data to JSON file for other components."""
        try:
            # Save to the same file the trading bot expects
            with open('btc_price.json', 'w') as f:
                json.dump(price_data, f, indent=2)
                
        except Exception as e:
            print(f"❌ Error saving price to file: {e}")
    
    def on_error(self, ws, error):
        """Handle WebSocket errors with reconnection logic."""
        print(f"❌ WebSocket error: {error}")
        self.error_count += 1
        self.connected = False
        
        # Attempt reconnection
        if self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            print(f"🔄 Attempting reconnection {self.reconnect_attempts}/{self.max_reconnect_attempts} in {self.reconnect_delay}s...")
            time.sleep(self.reconnect_delay)
            self._reconnect()
        else:
            print(f"💔 Max reconnection attempts reached. Manual restart required.")
    
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close."""
        print(f"🔌 WebSocket connection closed: {close_status_code} - {close_msg}")
        self.connected = False
        
        # Attempt reconnection if not intentionally closed
        if close_status_code != 1000:  # 1000 = normal closure
            self.on_error(ws, f"Connection closed unexpectedly: {close_status_code}")
    
    def on_open(self, ws):
        """Handle WebSocket open and send subscription."""
        print("🟢 WebSocket connection opened")
        
        # Send subscription message for BTC data
        subscribe_message = {
            "method": "SUBSCRIBE",
            "params": [
                "btcusdt@aggTrade",  # Aggregate trades for price updates
                "btcusdt@depth5@100ms"  # Top 5 levels, 100ms updates
            ],
            "id": 1
        }
        
        try:
            ws.send(json.dumps(subscribe_message))
            print("📡 Subscription message sent")
        except Exception as e:
            print(f"❌ Error sending subscription: {e}")
    
    def _reconnect(self):
        """Attempt to reconnect to WebSocket."""
        try:
            print("🔄 Reconnecting to Binance WebSocket...")
            self.ws = None
            self.start(debug=False)
        except Exception as e:
            print(f"❌ Reconnection failed: {e}")
    
    def get_last_price(self) -> Optional[Dict]:
        """Get the most recent price data."""
        return self.last_price
    
    def get_price_history(self, count: int = 100) -> List[tuple]:
        """Get recent price history."""
        return list(self.price_history)[-count:] if self.price_history else []
    
    def get_connection_stats(self) -> Dict:
        """Get connection statistics."""
        return {
            'connected': self.connected,
            'message_count': self.message_count,
            'error_count': self.error_count,
            'reconnect_attempts': self.reconnect_attempts,
            'last_message_ago': time.time() - self.last_message_time if self.last_message_time > 0 else None,
            'price_history_size': len(self.price_history)
        }
    
    def start(self, debug: bool = False):
        """Start the WebSocket connection."""
        if debug:
            websocket.enableTrace(True)
            
        try:
            self.ws = websocket.WebSocketApp(
                self.url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            print("🚀 Starting WebSocket connection...")
            self.ws.run_forever(
                ping_interval=30,  # Send ping every 30 seconds
                ping_timeout=10,   # Wait 10 seconds for pong
                ping_payload="ping"
            )
            
        except Exception as e:
            print(f"❌ Failed to start WebSocket: {e}")
            raise
    
    def stop(self):
        """Stop the WebSocket connection."""
        if self.ws:
            print("🛑 Stopping WebSocket connection...")
            self.connected = False
            self.ws.close()
            print("✅ WebSocket stopped")

# Maintain backward compatibility
class BinanceWebSocket(EnhancedBinanceWebSocket):
    """Backward compatible alias for EnhancedBinanceWebSocket."""
    pass

# Spike-aware WebSocket wrapper for easy integration
class SpikeAwareWebSocket:
    """WebSocket wrapper specifically for spike trading integration."""
    
    def __init__(self, spike_detector=None, trading_callback=None):
        self.spike_detector = spike_detector
        self.trading_callback = trading_callback
        self.ws = EnhancedBinanceWebSocket(self._spike_callback)
        
    def _spike_callback(self, price_data):
        """Internal callback that integrates with spike detector."""
        if self.spike_detector:
            # Add price to spike detector
            self.spike_detector.add_price_point(
                price_data['price'], 
                price_data['timestamp']
            )
            
            # Check for spike
            spike_detected, spike_info = self.spike_detector.detect_spike()
            
            if spike_detected and self.trading_callback:
                # Trigger trading callback
                try:
                    self.trading_callback(spike_info, price_data)
                except Exception as e:
                    print(f"❌ Error in trading callback: {e}")
    
    def start(self):
        """Start the spike-aware WebSocket."""
        self.ws.start()
    
    def stop(self):
        """Stop the spike-aware WebSocket."""
        self.ws.stop()
    
    def get_stats(self):
        """Get comprehensive statistics."""
        ws_stats = self.ws.get_connection_stats()
        spike_stats = self.spike_detector.get_analysis_summary() if self.spike_detector else {}
        
        return {
            'websocket': ws_stats,
            'spike_detector': spike_stats,
            'integration_active': self.spike_detector is not None and self.trading_callback is not None
        }

# Simple test function
def test_websocket():
    """Test function for WebSocket connectivity."""
    def test_spike_callback(price_data):
        print(f"🧪 Test spike callback: ${price_data['price']:,.2f}")
    
    print("🧪 Testing WebSocket connection...")
    ws = EnhancedBinanceWebSocket(test_spike_callback)
    
    try:
        ws.start()
    except KeyboardInterrupt:
        print("\n🛑 Test stopped by user")
        ws.stop()

# Usage examples
if __name__ == "__main__":
    # Basic WebSocket test
    test_websocket()
