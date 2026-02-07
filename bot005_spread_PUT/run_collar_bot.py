#!/usr/bin/env python3
# run_collar_bot.py - Dynamic Main Runner for Collar Strategy Bot
"""
Dynamic Collar Strategy Bot - Main Runner

Dependencies:
- python-dotenv: pip3 install python-dotenv
- numpy: pip3 install numpy
- scikit-learn (optional): pip3 install scikit-learn
  Note: Bot works without sklearn using fallback calculations

Files Structure:
├── collar_config.py      - Enhanced configuration with spread parameters
├── collar_strategy.py    - Dynamic spread calculation with ML
├── collar_trading.py     - Trading execution with manipulation detection
└── run_collar_bot.py     - This main runner file

Usage:
    python3 run_collar_bot.py

Features (Advanced):
- Dynamic spread prediction using polynomial regression
- 6-factor model: BTC volatility, token volatility, short momentum (10min),
  very short momentum (1min), latest BTC change, time decay
- Real-time model validation and auto-recalibration
- Market manipulation detection
- Persistent formula storage per hour
- Fallback calculations when ML unavailable
"""

import time
import json
import sys
from datetime import datetime

# Import all modules
from collar_config import (
    BOOK_DATA_PATH, BTC_PRICE_PATH, TOKEN_ID, MAX_TRADE_AMOUNT, MAX_TOKEN_VALUE_USD,
    MIN_QUANTITY_FILTER, MANDATORY_SELL_PRICE, CLOB_API_URL,
    PRIVATE_KEY, API_KEY, API_SECRET, API_PASSPHRASE, MAX_DELAY_MS,
    MIN_SPREAD, MAX_SPREAD, SPREAD_BUFFER, SPREAD_FORMULA_PATH, state,
    reload_token_id
)
from collar_strategy import MarketAnalyzer, display_analysis
from collar_trading import TradingExecutor, TradingLogic, display_trading_status

class DynamicCollarBot:
    """Dynamic collar strategy bot with polynomial regression."""

    def __init__(self):
        print("🚀 Initializing Dynamic Collar Strategy Bot...")

        # Initialize components
        self.market_analyzer = MarketAnalyzer(BOOK_DATA_PATH, BTC_PRICE_PATH)
        self.trading_executor = TradingExecutor()
        self.trading_logic = TradingLogic(self.trading_executor)

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

        try:
            with open(BTC_PRICE_PATH, 'r') as f:
                json.load(f)
            print(f"✅ BTC data accessible: {BTC_PRICE_PATH}")
        except Exception as e:
            print(f"❌ Cannot access BTC data: {e}")
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

    def check_data_delays(self, analysis: dict) -> bool:
        """Check if data delays are acceptable."""
        book_delay = analysis['book_data'].get('delay_ms', 9999)

        if book_delay > MAX_DELAY_MS:
            print(f"⚠️ High book delay: {book_delay}ms > {MAX_DELAY_MS}ms")
            return False

        return True

    def display_startup_info(self):
        """Display startup information."""
        print("=" * 90)
        print("🎯 DYNAMIC COLLAR STRATEGY BOT - LIVE TRADING")
        print("=" * 90)
        print(f"📁 Book Data: {BOOK_DATA_PATH}")
        print(f"📁 BTC Data: {BTC_PRICE_PATH}")
        print(f"📁 Spread Formula: {SPREAD_FORMULA_PATH}")
        print(f"🎫 Token ID: {TOKEN_ID}")
        print(f"💰 Max Trade: ${MAX_TRADE_AMOUNT}")
        print(f"💳 Max Token Value: ${MAX_TOKEN_VALUE_USD} (total limit)")
        print(f"📊 Min Quantity: {MIN_QUANTITY_FILTER}")
        print(f"⏰ Max Delay: {MAX_DELAY_MS}ms")
        print(f"📏 Spread Range: ${MIN_SPREAD:.2f} - ${MAX_SPREAD:.2f} + ${SPREAD_BUFFER:.2f} buffer")
        print(f"🚨 Mandatory Sell: @ ${MANDATORY_SELL_PRICE:.2f} (market over)")
        print("=" * 90)
        print("DYNAMIC STRATEGY:")
        print("🧠 POLYNOMIAL REGRESSION: Predicts optimal spreads using:")
        print("   • BTC price volatility (30min window)")
        print("   • Token price volatility (15min window)")
        print("   • Short-term momentum (10min)")
        print("   • Very short-term momentum (1min)")
        print("   • Latest BTC price change (tick-to-tick)")
        print("   • Time decay (approaching expiry)")
        print("🎯 ADAPTIVE TARGETS: Bid/Ask positioned around mid ± (predicted_spread + buffer)")
        print("🔍 VALIDATION: Constantly compares predictions vs market reality")
        print("📊 LEARNING: Auto-recalibrates formula when accuracy drops < 70%")
        print("🚨 MANIPULATION DETECTION: Detects unusual market conditions")
        print("💾 PERSISTENCE: Saves/loads formulas per hour (binary option resets)")
        print("=" * 90)
        print("⚠️  WARNING: This bot will place REAL orders with REAL money!")
        print("🧠 MACHINE LEARNING: Bot learns and adapts to market conditions!")
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

                # Perform market analysis
                analysis = self.market_analyzer.analyze_market()

                if analysis:
                    # Check data quality
                    if not self.check_data_delays(analysis):
                        print("⚠️ Skipping iteration due to stale data")
                        time.sleep(2)
                        continue

                    # Display analysis
                    display_analysis(analysis)

                    # Show what trading decisions would be made
                    print("\n🤔 TRADING DECISION ANALYSIS (SIMULATION)")
                    print("-" * 50)

                    balance_raw, balance_readable = self.trading_executor.get_balance()

                    # Check mandatory sell first
                    mandatory_sell = self.trading_logic.check_mandatory_sell_conditions(analysis, balance_raw)

                    if mandatory_sell:
                        print("🚨 MANDATORY SELL WOULD TRIGGER!")
                        print(f"📊 Market at ${MANDATORY_SELL_PRICE:.2f} - would execute immediate sell")
                        best_bid = analysis['book_data'].get('best_filtered_bid', {}).get('price', 0)
                        sell_price = max(MANDATORY_SELL_PRICE, best_bid) if best_bid > 0 else MANDATORY_SELL_PRICE
                        balance_tokens = balance_raw / 10**6
                        print(f"💰 Would sell {balance_tokens:.3f} tokens @ ${sell_price:.2f}")
                    else:
                        would_buy = self.trading_logic.should_place_buy_order(analysis)
                        would_sell = self.trading_logic.should_place_sell_order(analysis, balance_raw)

                        target_bid = target_prices.get('bid') if target_prices else None
                        target_ask = target_prices.get('ask') if target_prices else None

                        # Calculate current token value for limit check
                        book_data = analysis.get('book_data', {})
                        market_mid = book_data.get('market_mid', 0) or 0
                        balance_tokens = balance_raw / 10**6
                        current_token_value = balance_tokens * market_mid if market_mid > 0 else 0

                        # Check token value limit
                        would_exceed_limit = (current_token_value + MAX_TRADE_AMOUNT) > MAX_TOKEN_VALUE_USD

                        if would_buy and target_bid:
                            if would_exceed_limit:
                                print(f"🟡 WOULD BUY at ${target_bid:.3f} - BUT BLOCKED BY VALUE LIMIT")
                                print(f"   Current token value: ${current_token_value:.2f}")
                                print(f"   After buy: ${current_token_value + MAX_TRADE_AMOUNT:.2f}")
                                print(f"   Limit: ${MAX_TOKEN_VALUE_USD:.2f}")
                            else:
                                print(f"🟢 WOULD BUY at ${target_bid:.3f}")
                                trade_amount = MAX_TRADE_AMOUNT / target_bid
                                print(f"   Amount: {trade_amount:.0f} tokens = ${MAX_TRADE_AMOUNT}")
                                remaining_capacity = MAX_TOKEN_VALUE_USD - current_token_value - MAX_TRADE_AMOUNT
                                print(f"   Remaining capacity: ${remaining_capacity:.2f}")

                            # Show spread calculation details
                            spread_info = analysis.get('spread_analysis', {})
                            if spread_info.get('predicted_spread'):
                                print(f"   Predicted spread: ${spread_info['predicted_spread']:.3f}")
                                print(f"   Formula performance: {spread_info.get('formula_performance', 0):.1%}")

                            # Show if using base level pricing
                            if spread_info.get('manipulation_detected', False):
                                print(f"   📊 Using base level due to manipulation")
                        else:
                            print("⚪ No buy signal")
                            if not target_bid:
                                print("   (No target bid available)")
                            elif would_exceed_limit:
                                print("   (Blocked by token value limit)")

                        if would_sell and target_ask:
                            print(f"🔴 WOULD SELL at ${target_ask:.3f}")
                            balance_tokens = balance_raw / 10**6
                            print(f"   Amount: {balance_tokens:.3f} tokens (all balance)")
                        else:
                            print("⚪ No sell signal")
                            if balance_raw < 100000:  # MIN_BALANCE_SELL
                                print("   (Insufficient balance for selling)")
                            elif not target_ask:
                                print("   (No target ask available)")

                else:
                    print("❌ Analysis failed")

                print(f"\n💤 Waiting 5 seconds...")
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n👋 Analysis stopped by user")

    def run_live_trading(self):
        """Run bot in live trading mode."""
        print("⚡ STARTING LIVE TRADING MODE")
        print("🚀 Bot starting immediately...")

        try:
            iteration = 0
            while True:
                iteration += 1
                print(f"\n🔄 Trading Iteration {iteration} - {datetime.now().strftime('%H:%M:%S')}")

                # Check if it's a new hour and reload TOKEN_ID if needed
                if state.is_new_hour():
                    print("🆕 NEW HOUR DETECTED - Binary option reset!")

                    # Reload TOKEN_ID for new hour
                    if reload_token_id():
                        print("✅ TOKEN_ID reloaded for new hour")

                        # Reinitialize trading executor with new TOKEN_ID
                        print("🔄 Reinitializing trading client with new TOKEN_ID...")
                        self.trading_executor = TradingExecutor()
                        self.trading_logic = TradingLogic(self.trading_executor)
                        print("✅ Trading client reinitialized")
                    else:
                        print("📋 TOKEN_ID unchanged")

                # Perform market analysis
                analysis = self.market_analyzer.analyze_market()

                if analysis:
                    # Check data quality
                    if not self.check_data_delays(analysis):
                        print("⚠️ Skipping trading due to stale data")
                        time.sleep(2)
                        continue

                    # Display analysis
                    display_analysis(analysis)

                    # Display trading status
                    display_trading_status(analysis, self.trading_executor)

                    # Execute trading logic
                    print("\n⚙️ EXECUTING TRADING LOGIC...")
                    self.trading_logic.execute_trading_logic(analysis)

                else:
                    print("❌ Analysis failed - skipping trading")

                print(f"\n💤 Waiting 5 seconds...")
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n\n🛑 STOPPING BOT...")
            print("🚫 Cancelling all orders...")
            self.trading_executor.cancel_all_orders("bot shutdown")
            print("👋 Bot stopped by user")
        except Exception as e:
            print(f"\n❌ Bot error: {e}")
            print("🛑 Emergency order cancellation...")
            try:
                self.trading_executor.cancel_all_orders("bot error")
                print("✅ Orders cancelled")
            except:
                print("⚠️ Could not cancel orders - please check manually!")
            raise

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
    print("🎯 DYNAMIC COLLAR STRATEGY BOT")
    print("=" * 50)

    try:
        bot = DynamicCollarBot()
        bot.run()
    except KeyboardInterrupt:
        print("\n👋 Bot interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
