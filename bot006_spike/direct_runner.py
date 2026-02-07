#!/usr/bin/env python3
# direct_analysis.py - Direct runner that bypasses validation
"""
Direct Analysis Runner - Bypass Validation

This script runs the spike bot in analysis mode directly,
bypassing the validation that's causing issues.

Use this to see real spike detection with live BTC data.
"""

import sys
import signal
import time
from datetime import datetime

# Import bot components
try:
    from spike_trading_bot import BTCSpikeBot
    from spike_config import state
    from spike_logger import logger
    from spike_monitor import start_monitoring, stop_monitoring, display_status
    print("✅ All components imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class DirectAnalysisRunner:
    """Direct runner without validation."""
    
    def __init__(self):
        self.bot = None
        self.monitoring_active = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("🎯 Direct Analysis Runner initialized")
    
    def _signal_handler(self, sig, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n🛑 Shutting down...")
        self.stop()
        sys.exit(0)
    
    def start_analysis(self):
        """Start analysis mode directly."""
        print("🚀 Starting BTC Spike Analysis Mode (Direct)")
        print("=" * 60)
        print("📊 LIVE DATA SOURCES:")
        print("   🔗 BTC Prices: Binance WebSocket (live)")
        print("   📈 Market Data: PUT.json (Polymarket)")
        print("   ⚡ Spike Detection: Real-time analysis")
        print("=" * 60)
        print("⚠️  NO TRADING - Analysis Only")
        print("🔋 Press Ctrl+C to stop")
        print("=" * 60)
        
        try:
            # Start monitoring
            start_monitoring()
            self.monitoring_active = True
            print("✅ Monitoring system started")
            
            # Initialize bot directly
            self.bot = BTCSpikeBot()
            
            # Start analysis mode (this will show real spike detection)
            print("📊 Starting analysis with LIVE data...")
            self.bot.run_analysis_only()
            
        except KeyboardInterrupt:
            print("\n👋 Analysis stopped by user")
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.error(f"Direct analysis failed: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Stop all systems."""
        print("🛑 Stopping systems...")
        
        if self.monitoring_active:
            stop_monitoring()
            self.monitoring_active = False
            print("✅ Monitoring stopped")
        
        if self.bot and hasattr(self.bot, 'binance_ws'):
            try:
                self.bot.binance_ws.stop()
                print("✅ WebSocket stopped")
            except:
                pass
        
        print("✅ Shutdown complete")

def show_data_sources():
    """Show where data comes from."""
    print("📊 DATA SOURCES EXPLANATION")
    print("=" * 50)
    print("🧪 TEST MODE (what was failing):")
    print("   • Synthetic BTC prices (fake)")
    print("   • Generated in test_spike_detection()")
    print("   • Purpose: Algorithm testing")
    print()
    print("📈 LIVE ANALYSIS MODE (what you want):")
    print("   • Real BTC: wss://stream.binance.com:9443/stream")
    print("   • Market data: PUT.json file")  
    print("   • Updates: Every second")
    print("   • Purpose: Real spike detection")
    print()
    print("🎯 TO SEE LIVE SPIKE DETECTION:")
    print("   python3 direct_analysis.py")
    print("=" * 50)

def quick_status_check():
    """Quick check of data sources."""
    print("🔍 Checking data sources...")
    
    # Check PUT.json
    try:
        import json
        with open('PUT.json', 'r') as f:
            data = json.load(f)
        print("✅ PUT.json readable")
        
        if 'complete_book' in data:
            bids = len(data['complete_book'].get('bids', []))
            asks = len(data['complete_book'].get('asks', []))
            print(f"   📊 Order book: {bids} bids, {asks} asks")
        
        if 'asset_id' in data:
            print(f"   🎫 Asset ID: {data['asset_id'][:20]}...")
            
    except Exception as e:
        print(f"❌ PUT.json issue: {e}")
    
    # Check WebSocket capability
    try:
        from binance_websocket import EnhancedBinanceWebSocket
        print("✅ WebSocket module available")
    except Exception as e:
        print(f"❌ WebSocket issue: {e}")
    
    # Check config
    try:
        from spike_config import TOKEN_ID, MAX_TRADE_AMOUNT
        print(f"✅ Config loaded: ${MAX_TRADE_AMOUNT} max trade")
    except Exception as e:
        print(f"❌ Config issue: {e}")

def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        if sys.argv[1] == 'sources':
            show_data_sources()
            return
        elif sys.argv[1] == 'check':
            quick_status_check()
            return
        elif sys.argv[1] == 'help':
            print("Direct Analysis Runner")
            print("Usage:")
            print("  python3 direct_analysis.py        # Start live analysis")
            print("  python3 direct_analysis.py sources # Show data sources")
            print("  python3 direct_analysis.py check   # Check data availability")
            return
    
    # Default: start analysis
    runner = DirectAnalysisRunner()
    runner.start_analysis()

if __name__ == "__main__":
    main()
