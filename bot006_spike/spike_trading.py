# spike_trading.py - BTC Spike Trading Execution
"""
BTC Spike Trading Logic

Trading Strategy:
1. SPIKE BUY: When BTC spike detected, place limit order near market ASK
   - Target: Best ASK price minus small buffer for better fill
   - Amount: Based on MAX_TRADE_AMOUNT
   - Immediate execution when spike confirmed

2. PROTECTED SELL: After spike buy, place protected sell order for 1 minute
   - Target: ASK level with minimum 1000 shares
   - Duration: 1 minute timeout
   - Protection: Only at levels with sufficient liquidity

3. SPREAD-BASED SELL: Regular market making for non-spike periods
   - Target: Above market mid with spread buffer
   - Logic: Traditional spread-based positioning
   - Condition: When no spike activity

4. MANDATORY SELL: Emergency exit at $0.99 (market over)
   - Immediate market sell when token reaches $0.99
   - Override all other logic
"""

import time
import math
from typing import Tuple, Dict, Optional

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, BalanceAllowanceParams, AssetType, OrderArgs
from py_clob_client.order_builder.constants import BUY, SELL

from spike_config import (
    state, CLOB_API_URL, PRIVATE_KEY, API_KEY, API_SECRET, API_PASSPHRASE,
    CHAIN_ID, TOKEN_ID, MAX_TRADE_AMOUNT, MAX_TOKEN_VALUE_USD, MIN_BALANCE_SELL,
    MANDATORY_SELL_PRICE
)

class TradingExecutor:
    """Handle spike-driven order execution."""

    def __init__(self):
        self.client = self._setup_client()
        
        # Order tracking
        self.spike_buy_active = False
        self.protected_sell_active = False
        self.spread_sell_active = False
        
        # Price tracking
        self.spike_buy_price = 0.0
        self.protected_sell_price = 0.0
        self.spread_sell_price = 0.0
        
        # Timing
        self.protected_sell_start_time = 0
        self.protected_sell_duration = 60  # 1 minute

    def _setup_client(self):
        """Initialize the trading client."""
        creds = ApiCreds(
            api_key=API_KEY,
            api_secret=API_SECRET,
            api_passphrase=API_PASSPHRASE,
        )
        return ClobClient(CLOB_API_URL, key=PRIVATE_KEY, chain_id=CHAIN_ID, creds=creds)

    def get_balance(self) -> Tuple[int, float]:
        """Get current token balance."""
        try:
            balance_response = self.client.get_balance_allowance(
                params=BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL, token_id=TOKEN_ID)
            )
            state.api_calls_count += 1
            balance_raw = int(balance_response['balance'])
            balance_readable = balance_raw / 10**6
            return balance_raw, balance_readable
        except Exception as e:
            print(f"❌ Error getting balance: {e}")
            return 0, 0.0

    def cancel_all_orders(self, reason: str = "") -> bool:
        """Cancel all active orders."""
        try:
            resp = self.client.cancel_market_orders(asset_id=TOKEN_ID)
            print(f"🚫 Cancelled all orders ({reason}): {resp}")
            
            # Reset all order states
            self.spike_buy_active = False
            self.protected_sell_active = False
            self.spread_sell_active = False
            self.spike_buy_price = 0.0
            self.protected_sell_price = 0.0
            self.spread_sell_price = 0.0
            
            state.api_calls_count += 1
            return True

        except Exception as e:
            print(f"❌ Cancel error: {e}")
            return False

    def cancel_sell_orders_only(self, reason: str = "") -> bool:
        """Cancel only sell orders, keep buy orders active."""
        try:
            # This is a simplified approach - cancel all then re-place buy if needed
            # In a real implementation, you'd want more granular order management
            if self.protected_sell_active or self.spread_sell_active:
                self.cancel_all_orders(f"sell orders only - {reason}")
                return True
            return False
        except Exception as e:
            print(f"❌ Error cancelling sell orders: {e}")
            return False

    def execute_spike_buy_order(self, price: float) -> bool:
        """Execute spike buy order near ASK."""
        try:
            # Calculate trade amount
            trade_amount = MAX_TRADE_AMOUNT / price
            trade_amount = round(trade_amount, 0)  # Round to whole tokens

            if trade_amount < 1:
                print(f"🚫 Spike buy blocked - trade amount {trade_amount:.1f} < 1 token")
                return False

            print(f"⚡ SPIKE BUY ORDER: {trade_amount:.0f} tokens @ ${price:.3f} = ${trade_amount * price:.2f}")

            order_args = OrderArgs(price=price, size=trade_amount, side=BUY, token_id=TOKEN_ID)
            signed_order = self.client.create_order(order_args)
            resp = self.client.post_order(signed_order)

            print(f"✅ Spike buy response: {resp}")

            self.spike_buy_active = True
            self.spike_buy_price = price
            state.api_calls_count += 1

            return True

        except Exception as e:
            print(f"❌ Spike buy error: {e}")
            return False

    def execute_protected_sell_order(self, price: float, balance_raw: int) -> bool:
        """Execute protected sell order at ASK level with 1000+ shares."""
        try:
            # Calculate sellable amount
            balance_tokens = balance_raw / 10**6
            sell_amount = math.floor(balance_tokens * 1000) / 1000  # Round down to 3 decimals

            if sell_amount < 0.001:
                print(f"🚫 Protected sell blocked - amount {sell_amount:.3f} too small")
                return False

            print(f"🛡️ PROTECTED SELL ORDER: {sell_amount:.3f} tokens @ ${price:.3f} = ${sell_amount * price:.2f}")
            print(f"   Duration: {self.protected_sell_duration} seconds")

            order_args = OrderArgs(price=price, size=sell_amount, side=SELL, token_id=TOKEN_ID)
            signed_order = self.client.create_order(order_args)
            resp = self.client.post_order(signed_order)

            print(f"✅ Protected sell response: {resp}")

            self.protected_sell_active = True
            self.protected_sell_price = price
            self.protected_sell_start_time = time.time()
            state.api_calls_count += 1

            return True

        except Exception as e:
            print(f"❌ Protected sell error: {e}")
            return False

    def execute_spread_sell_order(self, price: float, balance_raw: int) -> bool:
        """Execute spread-based sell order."""
        try:
            # Calculate sellable amount
            balance_tokens = balance_raw / 10**6
            sell_amount = math.floor(balance_tokens * 1000) / 1000  # Round down to 3 decimals

            if sell_amount < 0.001:
                print(f"🚫 Spread sell blocked - amount {sell_amount:.3f} too small")
                return False

            print(f"📊 SPREAD SELL ORDER: {sell_amount:.3f} tokens @ ${price:.3f} = ${sell_amount * price:.2f}")

            order_args = OrderArgs(price=price, size=sell_amount, side=SELL, token_id=TOKEN_ID)
            signed_order = self.client.create_order(order_args)
            resp = self.client.post_order(signed_order)

            print(f"✅ Spread sell response: {resp}")

            self.spread_sell_active = True
            self.spread_sell_price = price
            state.api_calls_count += 1

            return True

        except Exception as e:
            print(f"❌ Spread sell error: {e}")
            return False

    def execute_mandatory_sell_order(self, price: float, balance_raw: int) -> bool:
        """Execute mandatory market sell at $0.99."""
        try:
            balance_tokens = balance_raw / 10**6
            sell_amount = math.floor(balance_tokens * 1000) / 1000

            if sell_amount < 0.001:
                print(f"🚫 Mandatory sell blocked - amount {sell_amount:.3f} too small")
                return False

            print(f"🚨 MANDATORY SELL ORDER: {sell_amount:.3f} tokens @ ${price:.3f} = ${sell_amount * price:.2f}")

            order_args = OrderArgs(price=price, size=sell_amount, side=SELL, token_id=TOKEN_ID)
            signed_order = self.client.create_order(order_args)
            resp = self.client.post_order(signed_order)

            print(f"✅ Mandatory sell response: {resp}")
            state.api_calls_count += 1

            return True

        except Exception as e:
            print(f"❌ Mandatory sell error: {e}")
            return False

    def is_protected_sell_expired(self) -> bool:
        """Check if protected sell period has expired."""
        if not self.protected_sell_active:
            return False
        
        return time.time() - self.protected_sell_start_time > self.protected_sell_duration

    def get_protected_sell_time_remaining(self) -> float:
        """Get remaining time for protected sell period."""
        if not self.protected_sell_active:
            return 0
        
        elapsed = time.time() - self.protected_sell_start_time
        return max(0, self.protected_sell_duration - elapsed)

    def get_trading_status(self) -> Dict:
        """Get current trading status."""
        return {
            'spike_buy_active': self.spike_buy_active,
            'protected_sell_active': self.protected_sell_active,
            'spread_sell_active': self.spread_sell_active,
            'spike_buy_price': self.spike_buy_price,
            'protected_sell_price': self.protected_sell_price,
            'spread_sell_price': self.spread_sell_price,
            'protected_sell_time_remaining': self.get_protected_sell_time_remaining(),
            'api_calls_count': state.api_calls_count
        }

class SpikeTradingLogic:
    """Implement BTC spike-driven trading logic."""

    def __init__(self, trading_executor: TradingExecutor, spike_detector):
        self.executor = trading_executor
        self.spike_detector = spike_detector
        self.last_spike_buy_time = 0
        self.spike_buy_cooldown = 120  # 2 minutes between spike buys

    def check_mandatory_sell_conditions(self, analysis: Dict, balance_raw: int) -> bool:
        """Check if mandatory sell conditions are met (market at $0.99)."""
        if balance_raw < MIN_BALANCE_SELL:
            return False

        book_data = analysis.get('book_data', {})
        if not book_data:
            return False

        # Check various price indicators for $0.99
        best_bid_data = book_data.get('best_filtered_bid') or {}
        best_ask_data = book_data.get('best_filtered_ask') or {}
        best_bid = best_bid_data.get('price', 0) if best_bid_data else 0
        best_ask = best_ask_data.get('price', 0) if best_ask_data else 0
        market_mid = book_data.get('market_mid', 0) or 0

        # Mandatory sell triggers
        mandatory_conditions = [
            best_bid >= MANDATORY_SELL_PRICE,
            market_mid >= MANDATORY_SELL_PRICE,
            (best_ask >= MANDATORY_SELL_PRICE and best_bid >= 0.97)
        ]

        return any(mandatory_conditions)

    def execute_mandatory_sell(self, analysis: Dict, balance_raw: int) -> bool:
        """Execute mandatory market sell at $0.99."""
        book_data = analysis.get('book_data', {})
        if not book_data:
            print("❌ Cannot execute mandatory sell - no book data")
            return False

        best_bid_data = book_data.get('best_filtered_bid') or {}
        best_bid = best_bid_data.get('price', 0) if best_bid_data else 0

        # Use market bid price or $0.99, whichever is higher
        sell_price = max(MANDATORY_SELL_PRICE, best_bid) if best_bid > 0 else MANDATORY_SELL_PRICE

        print(f"🚨 MANDATORY MARKET SELL TRIGGERED!")
        print(f"📊 Market conditions: Bid ${best_bid:.3f}")

        return self.executor.execute_mandatory_sell_order(sell_price, balance_raw)

    def should_execute_spike_buy(self, analysis: Dict, spike_detected: bool) -> bool:
        """Determine if we should execute a spike buy."""
        if not spike_detected or self.executor.spike_buy_active:
            return False
        
        # Check cooldown
        current_time = time.time()
        if current_time - self.last_spike_buy_time < self.spike_buy_cooldown:
            return False
        
        # Check token value limits
        balance_raw, balance_tokens = self.executor.get_balance()
        book_data = analysis.get('book_data', {})
        market_mid = book_data.get('market_mid', 0) or 0
        current_token_value_usd = balance_tokens * market_mid if market_mid > 0 else 0
        
        total_value_after_buy = current_token_value_usd + MAX_TRADE_AMOUNT
        if total_value_after_buy > MAX_TOKEN_VALUE_USD:
            print(f"🚫 Spike buy blocked - Token value limit exceeded")
            return False
        
        # Check if we have a valid spike buy target
        spike_targets = analysis.get('spike_targets', {})
        buy_target = spike_targets.get('buy_near_ask')
        
        return buy_target is not None and buy_target > 0.01

    def should_execute_protected_sell(self, analysis: Dict, balance_raw: int) -> bool:
        """Determine if we should execute a protected sell."""
        if (balance_raw < MIN_BALANCE_SELL or 
            self.executor.protected_sell_active or 
            not self.executor.spike_buy_active):
            return False
        
        # Find suitable protected ASK level
        spike_targets = analysis.get('spike_targets', {})
        protected_levels = spike_targets.get('protected_ask_levels', [])
        
        return len(protected_levels) > 0

    def should_execute_spread_sell(self, analysis: Dict, balance_raw: int) -> bool:
        """Determine if we should execute a spread-based sell."""
        if (balance_raw < MIN_BALANCE_SELL or 
            self.executor.spread_sell_active or 
            self.executor.protected_sell_active):
            return False
        
        # Only during non-spike periods
        if self.executor.spike_buy_active:
            return False
        
        # Check if we have a valid spread sell target
        spread_targets = analysis.get('spread_targets', {})
        sell_target = spread_targets.get('sell_above_mid')
        
        return sell_target is not None and sell_target > 0.01

    def execute_spike_buy(self, analysis: Dict) -> bool:
        """Execute spike buy near ASK."""
        spike_targets = analysis.get('spike_targets', {})
        buy_target = spike_targets.get('buy_near_ask')
        
        if not buy_target:
            print("❌ No spike buy target available")
            return False
        
        print(f"⚡ EXECUTING SPIKE BUY at ${buy_target:.3f}")
        
        if self.executor.execute_spike_buy_order(buy_target):
            self.last_spike_buy_time = time.time()
            return True
        
        return False

    def execute_protected_sell(self, analysis: Dict, balance_raw: int) -> bool:
        """Execute protected sell at ASK level with 1000+ shares."""
        spike_targets = analysis.get('spike_targets', {})
        protected_levels = spike_targets.get('protected_ask_levels', [])
        
        if not protected_levels:
            print("❌ No protected ASK levels available")
            return False
        
        # Use the best (lowest) protected level
        best_protected = protected_levels[0]
        sell_price = best_protected['price']
        
        print(f"🛡️ EXECUTING PROTECTED SELL at ${sell_price:.3f} ({best_protected['size']:.0f} shares available)")
        
        return self.executor.execute_protected_sell_order(sell_price, balance_raw)

    def execute_spread_sell(self, analysis: Dict, balance_raw: int) -> bool:
        """Execute spread-based sell."""
        spread_targets = analysis.get('spread_targets', {})
        sell_target = spread_targets.get('sell_above_mid')
        
        if not sell_target:
            print("❌ No spread sell target available")
            return False
        
        print(f"📊 EXECUTING SPREAD SELL at ${sell_target:.3f}")
        
        return self.executor.execute_spread_sell_order(sell_target, balance_raw)

    def handle_protected_sell_timeout(self):
        """Handle protected sell timeout after 1 minute."""
        if self.executor.is_protected_sell_expired():
            print("⏰ Protected sell period expired - cancelling protected orders")
            self.executor.cancel_sell_orders_only("protected sell timeout")
            
            # Reset spike buy status to allow new spikes
            self.executor.spike_buy_active = False

    def execute_trading_logic(self, analysis: Dict, spike_detected: bool, spike_info: Dict) -> None:
        """Execute complete spike trading logic."""
        if not analysis:
            print("⚠️ Skipping trading - no analysis data")
            return

        # Get balance
        balance_raw, balance_readable = self.executor.get_balance()

        # PRIORITY 1: Handle protected sell timeout
        self.handle_protected_sell_timeout()

        # PRIORITY 2: Mandatory sell at $0.99 (market over)
        if self.check_mandatory_sell_conditions(analysis, balance_raw):
            print("🚨 MANDATORY SELL CONDITIONS DETECTED!")
            
            # Cancel all orders first
            self.executor.cancel_all_orders("mandatory sell triggered")
            time.sleep(0.5)
            
            # Execute mandatory sell
            if self.execute_mandatory_sell(analysis, balance_raw):
                print("✅ Mandatory sell executed - market is over!")
                return
            else:
                print("❌ Mandatory sell failed - continuing with normal logic")

        # PRIORITY 3: Spike buy execution
        if self.should_execute_spike_buy(analysis, spike_detected):
            print("⚡ SPIKE BUY CONDITIONS MET!")
            
            # Cancel any existing orders first
            if (self.executor.protected_sell_active or 
                self.executor.spread_sell_active):
                self.executor.cancel_all_orders("preparing for spike buy")
                time.sleep(0.5)
            
            # Execute spike buy
            if self.execute_spike_buy(analysis):
                print("✅ Spike buy executed!")
                return

        # PRIORITY 4: Protected sell after spike buy
        if self.should_execute_protected_sell(analysis, balance_raw):
            print("🛡️ PROTECTED SELL CONDITIONS MET!")
            
            # Execute protected sell
            if self.execute_protected_sell(analysis, balance_raw):
                print("✅ Protected sell executed!")
                return

        # PRIORITY 5: Regular spread-based sell
        if self.should_execute_spread_sell(analysis, balance_raw):
            print("📊 SPREAD SELL CONDITIONS MET!")
            
            # Execute spread sell
            if self.execute_spread_sell(analysis, balance_raw):
                print("✅ Spread sell executed!")
                return

        # PRIORITY 6: Order price adjustments if conditions changed
        price_change_tolerance = 0.005  # 0.5 cents
        
        # Check spike buy price adjustment
        if self.executor.spike_buy_active:
            spike_targets = analysis.get('spike_targets', {})
            new_buy_target = spike_targets.get('buy_near_ask')
            
            if (new_buy_target and 
                abs(self.executor.spike_buy_price - new_buy_target) > price_change_tolerance):
                print(f"🔄 Spike buy price changed: {self.executor.spike_buy_price:.3f} → {new_buy_target:.3f}")
                # For spike buys, we usually don't adjust - let them execute as is
        
        # Check spread sell price adjustment
        if self.executor.spread_sell_active:
            spread_targets = analysis.get('spread_targets', {})
            new_sell_target = spread_targets.get('sell_above_mid')
            
            if (new_sell_target and 
                abs(self.executor.spread_sell_price - new_sell_target) > price_change_tolerance):
                print(f"🔄 Spread sell price changed: {self.executor.spread_sell_price:.3f} → {new_sell_target:.3f}")
                self.executor.cancel_sell_orders_only("price adjustment")
                time.sleep(0.5)

def display_trading_status(analysis: Dict, trading_executor: TradingExecutor, spike_info: Dict = None):
    """Display current trading status for spike trading."""
    if not analysis:
        return

    balance_raw, balance_readable = trading_executor.get_balance()
    trading_status = trading_executor.get_trading_status()
    book = analysis.get('book_data', {})

    if not book:
        print("⚠️ No book data available - skipping trading status display")
        return

    print("=" * 90)
    print("📊 SPIKE TRADING STATUS")
    print("=" * 90)

    # Account Status
    balance_tokens = balance_raw / 10**6
    market_mid = book.get('market_mid', 0) or 0
    token_value_usd = balance_tokens * market_mid if market_mid > 0 else 0

    print(f"💰 Balance: {balance_readable:.3f} tokens (${token_value_usd:.2f} value)")
    print(f"📞 API Calls: {trading_status['api_calls_count']}")
    
    # Token value limit status
    remaining_capacity = MAX_TOKEN_VALUE_USD - token_value_usd
    capacity_percentage = (token_value_usd / MAX_TOKEN_VALUE_USD * 100) if MAX_TOKEN_VALUE_USD > 0 else 0
    
    print(f"💳 Token Value Limit: ${token_value_usd:.2f} / ${MAX_TOKEN_VALUE_USD:.2f} ({capacity_percentage:.1f}%)")
    
    if remaining_capacity <= 0:
        print(f"🚫 BUY CAPACITY: FULL - No more spike buys allowed")
    elif remaining_capacity < MAX_TRADE_AMOUNT:
        print(f"⚠️ BUY CAPACITY: LOW - Only ${remaining_capacity:.2f} remaining")
    else:
        print(f"✅ BUY CAPACITY: OK - ${remaining_capacity:.2f} remaining")

    # Spike Status
    if spike_info and spike_info.get('detected'):
        print(f"🚨 SPIKE ACTIVE! Strength: {spike_info.get('strength', 0):.2f}x")
        if 'age_seconds' in spike_info:
            print(f"   Age: {spike_info['age_seconds']:.1f}s")
    else:
        print("👁️ Monitoring for BTC spikes...")

    # Current Orders Status
    if trading_status['spike_buy_active']:
        print(f"⚡ SPIKE BUY ACTIVE @ ${trading_status['spike_buy_price']:.3f}")
    else:
        print("⚪ No active spike buy order")

    if trading_status['protected_sell_active']:
        time_remaining = trading_status['protected_sell_time_remaining']
        print(f"🛡️ PROTECTED SELL ACTIVE @ ${trading_status['protected_sell_price']:.3f} ({time_remaining:.1f}s remaining)")
    elif trading_status['spread_sell_active']:
        print(f"📊 SPREAD SELL ACTIVE @ ${trading_status['spread_sell_price']:.3f}")
    else:
        print("⚪ No active sell orders")

    # Target Prices
    spike_targets = analysis.get('spike_targets', {})
    spread_targets = analysis.get('spread_targets', {})
    
    print(f"\n🎯 CURRENT TARGETS:")
    
    buy_target = spike_targets.get('buy_near_ask')
    if buy_target:
        print(f"⚡ Spike Buy: ${buy_target:.3f}")
    else:
        print("❌ No spike buy target")
    
    protected_levels = spike_targets.get('protected_ask_levels', [])
    if protected_levels:
        best_protected = protected_levels[0]
        print(f"🛡️ Protected Sell: ${best_protected['price']:.3f} ({best_protected['size']:.0f} shares)")
    else:
        print("❌ No protected sell levels (need 1000+ shares)")
    
    spread_sell = spread_targets.get('sell_above_mid')
    if spread_sell:
        print(f"📊 Spread Sell: ${spread_sell:.3f}")
    else:
        print("❌ No spread sell target")

    # Mandatory sell check
    best_bid_data = book.get('best_filtered_bid') or {}
    best_ask_data = book.get('best_filtered_ask') or {}
    best_bid = best_bid_data.get('price', 0) if best_bid_data else 0
    market_mid_check = market_mid or 0

    mandatory_sell_triggered = any([
        best_bid >= MANDATORY_SELL_PRICE,
        market_mid_check >= MANDATORY_SELL_PRICE,
    ])

    if mandatory_sell_triggered and balance_raw >= MIN_BALANCE_SELL:
        print("🚨 MANDATORY SELL CONDITIONS ACTIVE!")
        print("⚡ Will execute mandatory sell on next iteration")

    print("=" * 90)