#!/usr/bin/env python3
# spike_trading_bot.py - BTC Spike-Driven Trading Bot
"""
BTC Spike Trading Bot - Reactive Market Spike Trading

Strategy:
1. Monitor BTC price via WebSocket for market spikes
2. Spike Detection: BTC price change > average absolute body of last 5min candles
3. Buy Action: Place limit order close to market ASK when spike detected
4. Sell Protection: Place protected sell order for 1 minute (min 1000 shares)
5. Spread-Based Sells: Use market making logic for regular sell operations

Dependencies:
- python-dotenv: pip3 install python-dotenv
- websocket-client: pip3 install websocket-client
- numpy (optional): pip3 install numpy
"""

import time
import json
import sys
import threading
from datetime import datetime, timedelta

# Import all modules
from collar_config import (
    BOOK_DATA_PATH, BTC_PRICE_PATH, TOKEN_ID, MAX_TRADE_AMOUNT, MAX_TOKEN_VALUE_USD,
    MIN_QUANTITY_FILTER, MANDATORY_SELL_PRICE, CLOB_API_URL,
    PRIVATE_KEY, API_KEY, API_SECRET, API_PASSPHRASE, MAX_DELAY_MS,
    MIN_SPREAD, MAX_SPREAD, SPREAD_BUFFER, state, reload_token_id
)
from spike_strategy import SpikeDetector, MarketAnalyzer, display_analysis
from spike_trading import TradingExecutor, SpikeTradingLogic, display_trading_status
from binance_websocket import BinanceWebSocket

class BTCSpikeBot:
    """BTC Spike-driven trading bot with protected sells."""

    def __init__(self):
        print("🚀 Initializing BTC Spike Trading Bot...")

        # Initialize components
        self.spike_detector = SpikeDetector()
        self.market_analyzer = MarketAnalyzer(BOOK_DATA_PATH, BTC_PRICE_PATH)
        self.trading_executor = TradingExecutor()
        self.trading_logic = SpikeTradingLogic(self.trading_executor, self.spike_detector)
        
        # WebSocket for real-time BTC data
        self.binance_ws = BinanceWebSocket()
        
        # Trading state
        self.spike_buy_active = False
        self.protected_sell_active = False
        self.protected_sell_end_time = 0
        
        # Threading for WebSocket
        self.ws_thread = None
        self.running = False

        print("✅ All components initialized")

    def validate_setup(self) -> bool:
        """Validate bot setup and configuration."""
        print("🔍 Validating setup...")

        # Check file access
        try:
            with open(BOOK_DATA_PATH, 'r') as f:
                json.load(f)
            print(f"✅ Book data accessible: {BOOK_DATA_PATH}")
        except Exception as e:
            print(f"❌ Cannot access book data: {e}")
            return False

        # Check API credentials
        if not all([CLOB_API_URL, PRIVATE_KEY, API_KEY, API_SECRET, API_PASSPHRASE]):
            print("❌ Missing API credentials")
            return False
        print("✅ API credentials loaded")

        # Test balance retrieval
        try:
            balance_raw, balance_readable = self.trading_executor.get_balance()
            print(f"✅ Account balance: {balance_readable:.3f} tokens")
        except Exception as e:
            print(f"❌ Cannot retrieve balance: {e}")
            return False

        return True

    def start_btc_websocket(self):
        """Start BTC WebSocket in separate thread."""
        def ws_runner():
            print("🔗 Starting BTC WebSocket connection...")
            try:
                # Override the on_message to include spike detection
                original_on_message = self.binance_ws.on_message
                
                def enhanced_on_message(ws, message):
                    # Call original message handler
                    original_on_message(ws, message)
                    
                    # Check for spike after price update
                    if self.binance_ws.last_price:
                        self.check_for_spike_trigger()
                
                self.binance_ws.on_message = enhanced_on_message
                self.binance_ws.start(debug=False)
                
            except Exception as e:
                print(f"❌ WebSocket error: {e}")
                
        self.ws_thread = threading.Thread(target=ws_runner, daemon=True)
        self.ws_thread.start()
        
        # Wait for initial connection
        time.sleep(3)
        print("✅ BTC WebSocket started")

    def check_for_spike_trigger(self):
        """Check if current BTC price constitutes a spike and trigger trading."""
        if not self.running:
            return
            
        # Get latest BTC price data
        btc_data = self.binance_ws.get_last_price()
        if not btc_data:
            return
            
        # Update spike detector with latest price
        self.spike_detector.add_price_point(btc_data['price'], btc_data['timestamp'])
        
        # Check for spike
        spike_detected, spike_info = self.spike_detector.detect_spike()
        
        if spike_detected and not self.spike_buy_active:
            print(f"\n🚨 BTC SPIKE DETECTED!")
            print(f"💥 Spike Info: {spike_info}")
            
            # Trigger spike buy logic
            self.execute_spike_buy()

    def execute_spike_buy(self):
        """Execute buy order near ASK when spike detected."""
        try:
            # Get current market analysis
            analysis = self.market_analyzer.analyze_market()
            if not analysis:
                print("❌ Cannot execute spike buy - no market data")
                return
                
            # Execute spike buy logic
            self.trading_logic.execute_spike_buy(analysis)
            self.spike_buy_active = True
            
        except Exception as e:
            print(f"❌ Spike buy execution error: {e}")

    def display_startup_info(self):
        """Display startup information."""
        print("=" * 90)
        print("🎯 BTC SPIKE TRADING BOT - REACTIVE MARKET TRADING")
        print("=" * 90)
        print(f"📁 Book Data: {BOOK_DATA_PATH}")
        print(f"📁 BTC Data: {BTC_PRICE_PATH}")
        print(f"🎫 Token ID: {TOKEN_ID}")
        print(f"💰 Max Trade: ${MAX_TRADE_AMOUNT}")
        print(f"💳 Max Token Value: ${MAX_TOKEN_VALUE_USD} (total limit)")
        print(f"📊 Min Quantity: {MIN_QUANTITY_FILTER}")
        print(f"⏰ Max Delay: {MAX_DELAY_MS}ms")
        print(f"🚨 Mandatory Sell: @ ${MANDATORY_SELL_PRICE:.2f} (market over)")
        print("=" * 90)
        print("SPIKE TRADING STRATEGY:")
        print("⚡ BTC SPIKE DETECTION: Monitors real-time BTC price via WebSocket")
        print("   • Spike = price change > avg absolute body of last 5min candles")
        print("   • Uses 5-minute rolling window for dynamic threshold calculation")
        print("📈 SPIKE BUY ACTION: Places limit order close to market ASK")
        print("   • Targets best available ASK price")
        print("   • Executes immediately when spike detected")
        print("🛡️ PROTECTED SELL: 1-minute protected sell order after spike buy")
        print("   • Requires minimum 1000 shares on ASK level")
        print("   • Automatic timeout after 1 minute")
        print("📊 SPREAD-BASED SELLS: Market making logic for regular conditions")
        print("   • Uses spread analysis for non-spike periods")
        print("   • Maintains liquidity provision")
        print("🔗 REAL-TIME MONITORING: WebSocket connection to Binance")
        print("   • Live BTC price feed")
        print("   • Instant spike detection")
        print("=" * 90)
        print("⚠️  WARNING: This bot will place REAL orders with REAL money!")
        print("⚡ REAL-TIME: Bot reacts instantly to BTC market spikes!")
        print("=" * 90)

    def run_analysis_only(self):
        """Run bot in analysis-only mode (no trading)."""
        print("📊 Running in ANALYSIS-ONLY mode (no trading)")
        print("🔋 Press Ctrl+C to stop")
        print()

        try:
            iteration = 0
            while True:
                iteration += 1
                print(f"\n🔄 Analysis {iteration} - {datetime.now().strftime('%H:%M:%S')}")

                # Get latest BTC data
                btc_data = self.binance_ws.get_last_price()
                if btc_data:
                    self.spike_detector.add_price_point(btc_data['price'], btc_data['timestamp'])
                    
                # Check for spike
                spike_detected, spike_info = self.spike_detector.detect_spike()
                
                # Perform market analysis
                analysis = self.market_analyzer.analyze_market()

                if analysis:
                    # Display analysis
                    display_analysis(analysis, spike_info)

                    # Show what trading decisions would be made
                    print("\n🤔 TRADING DECISION ANALYSIS (SIMULATION)")
                    print("-" * 50)

                    if spike_detected:
                        print("🚨 SPIKE BUY WOULD TRIGGER!")
                        print(f"📊 Spike strength: {spike_info.get('strength', 0):.2%}")
                        print(f"⚡ BTC change: {spike_info.get('price_change', 0):.2%}")
                        
                        book_data = analysis['book_data']
                        best_ask = book_data.get('best_filtered_ask', {})
                        if best_ask:
                            print(f"🎯 Would buy near ASK: ${best_ask['price']:.3f}")
                    else:
                        print("⚪ No spike detected - monitoring...")
                        
                    # Check for sell conditions
                    balance_raw, _ = self.trading_executor.get_balance()
                    if balance_raw >= 100000:  # Has tokens to sell
                        print("📊 SELL ANALYSIS: Using spread-based logic")
                        
                else:
                    print("❌ Analysis failed")

                print(f"\n💤 Waiting 5 seconds...")
                time.sleep(5)

        except KeyboardInterrupt:
            print("\n👋 Analysis stopped by user")

    def run_live_trading(self):
        """Run bot in live trading mode."""
        print("⚡ STARTING LIVE SPIKE TRADING MODE")
        self.running = True
        
        # Start BTC WebSocket
        self.start_btc_websocket()
        
        print("🚀 Bot active - monitoring for BTC spikes...")

        try:
            iteration = 0
            while True:
                iteration += 1
                print(f"\n🔄 Trading Iteration {iteration} - {datetime.now().strftime('%H:%M:%S')}")

                # Check if it's a new hour and reload TOKEN_ID if needed
                if state.is_new_hour():
                    print("🆕 NEW HOUR DETECTED - Binary option reset!")
                    if reload_token_id():
                        print("✅ TOKEN_ID reloaded for new hour")
                        self.trading_executor = TradingExecutor()
                        self.trading_logic = SpikeTradingLogic(self.trading_executor, self.spike_detector)
                        print("✅ Trading client reinitialized")

                # Get latest spike detection status
                spike_detected, spike_info = self.spike_detector.get_latest_spike_status()
                
                # Perform market analysis
                analysis = self.market_analyzer.analyze_market()

                if analysis:
                    # Display analysis with spike info
                    display_analysis(analysis, spike_info)

                    # Display trading status
                    display_trading_status(analysis, self.trading_executor, spike_info)

                    # Execute trading logic
                    print("\n⚙️ EXECUTING SPIKE TRADING LOGIC...")
                    self.trading_logic.execute_trading_logic(analysis, spike_detected, spike_info)

                else:
                    print("❌ Analysis failed - skipping trading")

                # Update protected sell status
                self.update_protected_sell_status()

                print(f"\n💤 Waiting 10 seconds...")
                time.sleep(10)

        except KeyboardInterrupt:
            print("\n\n🛑 STOPPING BOT...")
            self.running = False
            print("🚫 Cancelling all orders...")
            self.trading_executor.cancel_all_orders("bot shutdown")
            print("📱 Stopping WebSocket...")
            if self.binance_ws:
                self.binance_ws.stop()
            print("👋 Bot stopped by user")
        except Exception as e:
            print(f"\n❌ Bot error: {e}")
            self.running = False
            print("🛑 Emergency order cancellation...")
            try:
                self.trading_executor.cancel_all_orders("bot error")
                print("✅ Orders cancelled")
            except:
                print("⚠️ Could not cancel orders - please check manually!")
            raise

    def update_protected_sell_status(self):
        """Update protected sell status and timeout."""
        if self.protected_sell_active and time.time() > self.protected_sell_end_time:
            print("⏰ Protected sell period expired")
            self.protected_sell_active = False
            self.spike_buy_active = False  # Allow new spike buys

    def run(self):
        """Main run method - starts live trading directly."""
        self.display_startup_info()

        if not self.validate_setup():
            print("❌ Setup validation failed - cannot continue")
            return

        # Start live trading directly
        self.run_live_trading()

def main():
    """Main entry point."""
    print("🎯 BTC SPIKE TRADING BOT")
    print("=" * 50)

    try:
        bot = BTCSpikeBot()
        bot.run()
    except KeyboardInterrupt:
        print("\n👋 Bot interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()