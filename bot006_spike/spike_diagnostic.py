#!/usr/bin/env python3
# spike_diagnostic.py - Real-time spike detection diagnostics
"""
Spike Detection Diagnostic Tool

Shows real-time information about why spikes are/aren't being detected.
"""

import time
import json
from datetime import datetime
from spike_strategy import SpikeDetector
from binance_websocket import EnhancedBinanceWebSocket
from spike_config import SPIKE_DETECTION

class SpikeDiagnostic:
    """Diagnostic tool for spike detection."""
    
    def __init__(self):
        # Create spike detector with current settings
        self.spike_detector = SpikeDetector(
            lookback_minutes=SPIKE_DETECTION.get('lookback_minutes', 5),
            min_spike_threshold=SPIKE_DETECTION.get('min_spike_threshold', 0.003)
        )
        
        # WebSocket for live data
        self.ws = EnhancedBinanceWebSocket(self._price_callback)
        
        # Tracking
        self.last_prices = []
        self.iteration = 0
        
        print("🔍 Spike Detection Diagnostic Tool")
        print("=" * 60)
        print(f"⚙️ Settings:")
        print(f"   Lookback: {SPIKE_DETECTION.get('lookback_minutes', 5)} minutes")
        print(f"   Min Threshold: {SPIKE_DETECTION.get('min_spike_threshold', 0.003):.4f} ({SPIKE_DETECTION.get('min_spike_threshold', 0.003)*100:.2f}%)")
        print(f"   Cooldown: {SPIKE_DETECTION.get('spike_cooldown_seconds', 120)} seconds")
        print("=" * 60)
    
    def _price_callback(self, price_data):
        """Handle new price data."""
        try:
            price = price_data['price']
            timestamp = price_data['timestamp']
            
            # Add to spike detector
            self.spike_detector.add_price_point(price, timestamp)
            
            # Track recent prices for manual calculation
            self.last_prices.append((time.time(), price))
            if len(self.last_prices) > 100:
                self.last_prices = self.last_prices[-100:]
            
            self.iteration += 1
            
            # Display diagnostics every 10 iterations
            if self.iteration % 10 == 0:
                self._display_diagnostics(price)
                
        except Exception as e:
            print(f"❌ Error in price callback: {e}")
    
    def _display_diagnostics(self, current_price):
        """Display current diagnostic information."""
        print(f"\n🔍 DIAGNOSTIC {self.iteration} - {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 50)
        
        # Current price
        print(f"💰 Current BTC: ${current_price:,.2f}")
        
        # Get spike detector analysis
        analysis = self.spike_detector.get_analysis_summary()
        
        # Data points
        data_points = analysis['data_points']
        print(f"📊 Data Points: {data_points['price_points']} prices, {data_points['candlesticks']} candlesticks")
        
        # Current metrics
        metrics = analysis['metrics']
        current_threshold = metrics['current_threshold']
        avg_body_size = metrics['avg_body_size']
        
        print(f"⚡ Current Threshold: {current_threshold:.4f} ({current_threshold*100:.2f}%)")
        print(f"📏 Avg Body Size: {avg_body_size:.4f}")
        
        # Manual recent change calculation
        if len(self.last_prices) >= 2:
            recent_change = self._calculate_recent_change()
            print(f"📈 Recent Change: {recent_change:.4f} ({recent_change*100:.2f}%)")
            
            # Compare with threshold
            if recent_change > current_threshold:
                print("🚨 WOULD TRIGGER SPIKE! (Recent change > threshold)")
            else:
                needed = current_threshold - recent_change
                print(f"⚪ No spike (need {needed:.4f} more = {needed*100:.2f}%)")
        
        # Check for actual spike
        spike_detected, spike_info = self.spike_detector.detect_spike()
        if spike_detected:
            print("🎉 SPIKE DETECTED!")
            print(f"   Strength: {spike_info.get('strength', 0):.2f}x")
            print(f"   Change: {spike_info.get('price_change', 0):.4f}")
        
        # Recent candlesticks
        recent_candles = analysis.get('recent_candlesticks', [])
        if recent_candles:
            print(f"🕯️ Recent Candlesticks:")
            for candle in recent_candles[-3:]:  # Last 3
                direction = candle['direction']
                body = candle['body']
                print(f"   {direction} Body: {body:.2f}")
    
    def _calculate_recent_change(self):
        """Calculate recent price change manually."""
        if len(self.last_prices) < 10:
            return 0
        
        # Get prices from last 30 seconds
        current_time = time.time()
        recent_prices = [p for t, p in self.last_prices if current_time - t < 30]
        
        if len(recent_prices) < 2:
            return 0
        
        # Calculate change
        start_price = recent_prices[0]
        end_price = recent_prices[-1]
        
        if start_price > 0:
            return abs(end_price - start_price) / start_price
        
        return 0
    
    def start(self):
        """Start diagnostics."""
        print("🚀 Starting diagnostic mode...")
        print("📡 Connecting to Binance WebSocket...")
        
        try:
            import threading
            ws_thread = threading.Thread(target=self.ws.start, daemon=True)
            ws_thread.start()
            
            # Wait for connection
            time.sleep(3)
            
            if self.ws.connected:
                print("✅ Connected! Monitoring spike detection...")
                print("⏳ Waiting for data to accumulate...")
                print("📋 Press Ctrl+C to stop")
                
                # Keep running
                while True:
                    time.sleep(1)
            else:
                print("❌ WebSocket connection failed")
                
        except KeyboardInterrupt:
            print("\n👋 Diagnostics stopped")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.ws.stop()

def quick_settings_test():
    """Test different sensitivity settings."""
    print("🧪 SENSITIVITY SETTINGS TEST")
    print("=" * 40)
    
    settings_to_test = [
        {"threshold": 0.005, "name": "Conservative (0.5%)"},
        {"threshold": 0.003, "name": "Default (0.3%)"},  
        {"threshold": 0.001, "name": "Sensitive (0.1%)"},
        {"threshold": 0.0005, "name": "Very Sensitive (0.05%)"},
    ]
    
    print("With recent BTC volatility, you should use:")
    for setting in settings_to_test:
        threshold = setting["threshold"]
        name = setting["name"]
        
        # Estimate signals per day
        if threshold >= 0.005:
            signals = "1-3 per day"
        elif threshold >= 0.003:
            signals = "3-8 per day"
        elif threshold >= 0.001:
            signals = "10-20 per day"
        else:
            signals = "20+ per day"
        
        print(f"   {name}: ~{signals}")
    
    print("\n💡 Recommendation: Start with 0.001 (0.1%) for more signals")

def main():
    """Main diagnostic function."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'settings':
        quick_settings_test()
        return
    
    diagnostic = SpikeDiagnostic()
    diagnostic.start()

if __name__ == "__main__":
    main()
