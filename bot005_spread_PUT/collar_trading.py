# collar_trading.py - Simplified Trading Execution
import time
import math
from typing import Tuple, Dict

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, BalanceAllowanceParams, AssetType, OrderArgs
from py_clob_client.order_builder.constants import BUY, SELL

from collar_config import (
    state, CLOB_API_URL, PRIVATE_KEY, API_KEY, API_SECRET, API_PASSPHRASE,
    CHAIN_ID, TOKEN_ID, MAX_TRADE_AMOUNT, MAX_TOKEN_VALUE_USD, MIN_BALANCE_SELL,
    MANDATORY_SELL_PRICE
)

class TradingExecutor:
    """Handle simple order execution."""

    def __init__(self):
        self.client = self._setup_client()
        self.buy_order_active = False
        self.sell_order_active = False
        self.current_buy_price = 0.0
        self.current_sell_price = 0.0

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
            self.buy_order_active = False
            self.sell_order_active = False
            self.current_buy_price = 0.0
            self.current_sell_price = 0.0
            state.api_calls_count += 1
            return True

        except Exception as e:
            print(f"❌ Cancel error: {e}")
            return False

    def execute_buy_order(self, price: float) -> bool:
        """Execute buy order at specified price."""
        try:
            # Calculate trade amount based on USD limit
            trade_amount = MAX_TRADE_AMOUNT / price
            trade_amount = round(trade_amount, 0)  # Round to whole tokens

            if trade_amount < 1:
                print(f"🚫 Buy blocked - trade amount {trade_amount:.1f} < 1 token")
                return False

            print(f"💰 BUY ORDER: {trade_amount:.0f} tokens @ ${price:.2f} = ${trade_amount * price:.2f}")

            order_args = OrderArgs(price=price, size=trade_amount, side=BUY, token_id=TOKEN_ID)
            signed_order = self.client.create_order(order_args)
            resp = self.client.post_order(signed_order)

            print(f"✅ Buy order response: {resp}")

            self.buy_order_active = True
            self.current_buy_price = price
            state.api_calls_count += 1

            return True

        except Exception as e:
            print(f"❌ Buy order error: {e}")
            return False

    def execute_sell_order(self, price: float, balance_raw: int) -> bool:
        """Execute sell order for entire balance."""
        try:
            # Calculate sellable amount (sell everything)
            balance_tokens = balance_raw / 10**6
            sell_amount = math.floor(balance_tokens * 1000) / 1000  # Round down to 3 decimals

            if sell_amount < 0.001:
                print(f"🚫 Sell blocked - amount {sell_amount:.3f} too small")
                return False

            print(f"💰 SELL ORDER: {sell_amount:.3f} tokens @ ${price:.2f} = ${sell_amount * price:.2f}")

            order_args = OrderArgs(price=price, size=sell_amount, side=SELL, token_id=TOKEN_ID)
            signed_order = self.client.create_order(order_args)
            resp = self.client.post_order(signed_order)

            print(f"✅ Sell order response: {resp}")

            self.sell_order_active = True
            self.current_sell_price = price
            state.api_calls_count += 1

            return True

        except Exception as e:
            print(f"❌ Sell order error: {e}")
            return False

    def get_trading_status(self) -> Dict:
        """Get current trading status."""
        return {
            'buy_order_active': self.buy_order_active,
            'sell_order_active': self.sell_order_active,
            'current_buy_price': self.current_buy_price,
            'current_sell_price': self.current_sell_price,
            'api_calls_count': state.api_calls_count
        }

class TradingLogic:
    """Implement simple trading logic."""

    def __init__(self, trading_executor: TradingExecutor):
        self.executor = trading_executor
        self.last_spike_check = 0
        self.spike_detection_enabled = True

    def detect_btc_spike_and_cancel(self, analysis: Dict) -> bool:
        """Detect sudden BTC spikes and cancel orders immediately."""
        current_time = time.time()
        
        # Only check every 10 seconds to avoid excessive checking
        if current_time - self.last_spike_check < 10:
            return False
        
        self.last_spike_check = current_time
        
        if not self.spike_detection_enabled:
            return False
        
        # Get BTC price history
        btc_history = state.btc_price_history
        if len(btc_history) < 10:
            return False  # Need sufficient history
        
        # Get last 5 minutes of BTC data
        five_minutes_ago = current_time - 300  # 5 minutes
        recent_btc = [(t, p) for t, p in btc_history if t >= five_minutes_ago]
        
        if len(recent_btc) < 5:
            return False  # Need at least 5 data points
        
        # Calculate recent price movements (absolute changes between consecutive points)
        price_moves = []
        for i in range(1, len(recent_btc)):
            prev_price = recent_btc[i-1][1]
            curr_price = recent_btc[i][1]
            if prev_price > 0:
                move = abs((curr_price - prev_price) / prev_price)
                price_moves.append(move)
        
        if len(price_moves) < 3:
            return False
        
        # Calculate average movement over the 5-minute period
        avg_movement = sum(price_moves) / len(price_moves)
        
        # Get the most recent movement (last minute)
        if len(recent_btc) >= 2:
            latest_price = recent_btc[-1][1]
            prev_price = recent_btc[-2][1]
            if prev_price > 0:
                latest_movement = abs((latest_price - prev_price) / prev_price)
            else:
                return False
        else:
            return False
        
        # Define spike threshold: movement is 3x larger than recent average AND > 0.5%
        spike_threshold_multiplier = 3.0
        min_spike_threshold = 0.005  # 0.5% minimum
        
        spike_detected = (
            latest_movement > (avg_movement * spike_threshold_multiplier) and
            latest_movement > min_spike_threshold
        )
        
        if spike_detected:
            print(f"🚨 BTC SPIKE DETECTED!")
            print(f"   Latest movement: {latest_movement:.3%}")
            print(f"   5-min average: {avg_movement:.3%}")
            print(f"   Spike ratio: {latest_movement/avg_movement:.1f}x")
            print(f"   Current BTC: ${recent_btc[-1][1]:,.2f}")
            
            # Cancel all orders immediately
            if self.executor.buy_order_active or self.executor.sell_order_active:
                success = self.executor.cancel_all_orders("BTC spike detected")
                if success:
                    print("✅ Orders cancelled due to BTC spike")
                    # Brief pause to let cancellation process
                    time.sleep(2)
                    return True
                else:
                    print("❌ Failed to cancel orders during spike")
        
        return spike_detected
        """Check if mandatory sell conditions are met (market at $0.99)."""
        if balance_raw < MIN_BALANCE_SELL:
            return False

        book_data = analysis.get('book_data', {})
        if not book_data:
            return False

        # Check various price indicators for $0.99 - handle None values safely
        best_bid_data = book_data.get('best_filtered_bid') or {}
        best_ask_data = book_data.get('best_filtered_ask') or {}
        best_bid = best_bid_data.get('price', 0) if best_bid_data else 0
        best_ask = best_ask_data.get('price', 0) if best_ask_data else 0
        market_mid = book_data.get('market_mid', 0) or 0

        # Mandatory sell triggers
        mandatory_conditions = [
            best_bid >= MANDATORY_SELL_PRICE,  # Best bid at or above $0.99
            market_mid >= MANDATORY_SELL_PRICE,  # Market mid at or above $0.99
            (best_ask >= MANDATORY_SELL_PRICE and best_bid >= 0.97)  # Ask at $0.99 with bid close
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
        print(f"📊 Market conditions: Bid ${best_bid:.2f}")

        return self.executor.execute_sell_order(sell_price, balance_raw)

    def should_place_buy_order(self, analysis: Dict) -> bool:
        """Determine if we should place a buy order."""
        if not analysis:
            return False
            
        target_prices = analysis.get('target_prices', {})
        if not target_prices:
            return False
            
        target_bid = target_prices.get('bid')
        if not target_bid:
            return False

        # Get current balance to check total token value limit
        balance_raw, balance_tokens = self.executor.get_balance()
        
        # Calculate current token value in USD
        book_data = analysis.get('book_data', {})
        market_mid = book_data.get('market_mid', 0) or 0
        current_token_value_usd = balance_tokens * market_mid if market_mid > 0 else 0
        
        # Calculate potential new order value
        potential_order_value = MAX_TRADE_AMOUNT
        total_value_after_buy = current_token_value_usd + potential_order_value
        
        # Check total token value limit
        if total_value_after_buy > MAX_TOKEN_VALUE_USD:
            print(f"🚫 Buy blocked - Total token value limit exceeded")
            print(f"   Current token value: ${current_token_value_usd:.2f}")
            print(f"   Potential order: ${potential_order_value:.2f}")
            print(f"   Total after buy: ${total_value_after_buy:.2f}")
            print(f"   Limit: ${MAX_TOKEN_VALUE_USD:.2f}")
            return False

        # Basic conditions
        basic_conditions = (
            not self.executor.buy_order_active and
            target_bid > 0.05 and  # Minimum price check
            target_bid < 0.95  # Maximum price check
        )

        return basic_conditions

    def should_place_sell_order(self, analysis: Dict, balance_raw: int) -> bool:
        """Determine if we should place a sell order."""
        if not analysis or balance_raw < MIN_BALANCE_SELL:
            return False
            
        target_prices = analysis.get('target_prices', {})
        if not target_prices:
            return False
            
        target_ask = target_prices.get('ask')
        if not target_ask:
            return False

        # Basic conditions
        basic_conditions = (
            not self.executor.sell_order_active and
            target_ask > 0.05 and  # Minimum price check
            target_ask < 0.95  # Maximum price check
        )

        return basic_conditions

    def execute_trading_logic(self, analysis: Dict) -> None:
        """Execute simple trading logic based on analysis."""
        if not analysis:
            print("⚠️ Skipping trading - no analysis data")
            return

        # Get balance
        balance_raw, balance_readable = self.executor.get_balance()

        # PRIORITY CHECK: BTC Spike Detection - Cancel orders immediately
        spike_detected = self.detect_btc_spike_and_cancel(analysis)
        if spike_detected:
            print("⚡ Skipping trading iteration due to BTC spike - letting market settle")
            return

        # PRIORITY CHECK: Mandatory sell at $0.99 (market over)
        if self.check_mandatory_sell_conditions(analysis, balance_raw):
            print("🚨 MANDATORY SELL CONDITIONS DETECTED - Market at $0.99!")

            # Cancel any existing orders first
            if self.executor.buy_order_active or self.executor.sell_order_active:
                self.executor.cancel_all_orders("mandatory sell triggered")
                time.sleep(0.5)  # Brief pause

            # Execute mandatory sell
            if self.execute_mandatory_sell(analysis, balance_raw):
                print("✅ Mandatory sell executed - market is over!")
                return
            else:
                print("❌ Mandatory sell failed - continuing with normal logic")

        # Get target prices safely
        target_prices = analysis.get('target_prices', {})
        target_bid = target_prices.get('bid')
        target_ask = target_prices.get('ask')

        # Check if we have valid prices
        if target_bid is None or target_ask is None:
            print("⚠️ Skipping trading - no valid target prices")
            if not target_bid:
                print("   📊 No target bid available")
            if not target_ask:
                print("   📊 No target ask available")
            return

        # Cancel orders if prices changed significantly
        price_change_tolerance = 0.005  # 0.5 cents
        
        if (self.executor.buy_order_active and
            abs(self.executor.current_buy_price - target_bid) > price_change_tolerance):
            print(f"🔄 Buy price changed: {self.executor.current_buy_price:.2f} → {target_bid:.2f}")
            self.executor.cancel_all_orders("price change")
            time.sleep(0.5)
            
        if (self.executor.sell_order_active and
            abs(self.executor.current_sell_price - target_ask) > price_change_tolerance):
            print(f"🔄 Sell price changed: {self.executor.current_sell_price:.2f} → {target_ask:.2f}")
            self.executor.cancel_all_orders("price change")
            time.sleep(0.5)

        # Buy order logic
        if self.should_place_buy_order(analysis):
            print(f"🟢 BUY CONDITIONS MET - Placing order at ${target_bid:.2f}")
            self.executor.execute_buy_order(target_bid)

        # Sell order logic
        if self.should_place_sell_order(analysis, balance_raw):
            print(f"🔴 SELL CONDITIONS MET - Placing order at ${target_ask:.2f}")
            self.executor.execute_sell_order(target_ask, balance_raw)

def display_trading_status(analysis: Dict, trading_executor: TradingExecutor):
    """Display current trading status with dynamic spread info."""
    if not analysis:
        return

    balance_raw, balance_readable = trading_executor.get_balance()
    trading_status = trading_executor.get_trading_status()
    targets = analysis.get('target_prices', {})
    spread_info = analysis.get('spread_analysis', {})
    book = analysis.get('book_data', {})

    # Safely handle None book data
    if not book:
        print("⚠️ No book data available - skipping trading status display")
        return

    print("=" * 90)
    print("📊 DYNAMIC TRADING STATUS")
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
        print(f"🚫 BUY CAPACITY: FULL - No more buys allowed")
    elif remaining_capacity < MAX_TRADE_AMOUNT:
        print(f"⚠️ BUY CAPACITY: LOW - Only ${remaining_capacity:.2f} remaining")
    else:
        print(f"✅ BUY CAPACITY: OK - ${remaining_capacity:.2f} remaining")

    # Spike Detection Status
    btc_history = state.btc_price_history
    if len(btc_history) >= 2:
        latest_btc_price = btc_history[-1][1]
        prev_btc_price = btc_history[-2][1]
        if prev_btc_price > 0:
            latest_btc_move = abs((latest_btc_price - prev_btc_price) / prev_btc_price)
            print(f"⚡ BTC Spike Monitor: Latest move {latest_btc_move:.3%}")
            
            # Calculate 5-min average for context
            current_time = time.time()
            five_minutes_ago = current_time - 300
            recent_btc = [(t, p) for t, p in btc_history if t >= five_minutes_ago]
            
            if len(recent_btc) >= 3:
                price_moves = []
                for i in range(1, len(recent_btc)):
                    prev_price = recent_btc[i-1][1]
                    curr_price = recent_btc[i][1]
                    if prev_price > 0:
                        move = abs((curr_price - prev_price) / prev_price)
                        price_moves.append(move)
                
                if price_moves:
                    avg_movement = sum(price_moves) / len(price_moves)
                    spike_ratio = latest_btc_move / avg_movement if avg_movement > 0 else 0
                    print(f"   5-min avg: {avg_movement:.3%}, Ratio: {spike_ratio:.1f}x (threshold: 3.0x)")
                    
                    if spike_ratio > 3.0 and latest_btc_move > 0.005:
                        print(f"   🚨 SPIKE THRESHOLD EXCEEDED!")
    else:
        print("⚡ BTC Spike Monitor: Insufficient data")

    # Spread Performance
    print(f"🎯 Formula Performance: {spread_info.get('formula_performance', 0):.1%}")
    if spread_info.get('manipulation_detected', False):
        print("🔧 MARKET MANIPULATION DETECTED - Using base level pricing")
        
        # Show base level info
        if book.get('base_bid_level'):
            base_bid = book['base_bid_level']
            print(f"   📊 Base bid: ${base_bid['price']:.3f} ({base_bid['size']:.0f} shares)")
    else:
        print("✅ Normal market conditions - Using mid-price targeting")

    # Training Data Status
    if hasattr(state, 'btc_price_history'):
        print(f"📊 Training Data: {len(state.btc_price_history)} BTC prices, {len(state.token_price_history)} token prices")

    # Check mandatory sell conditions - with safe None handling
    best_bid_data = book.get('best_filtered_bid') or {}
    best_ask_data = book.get('best_filtered_ask') or {}
    best_bid = best_bid_data.get('price', 0) if best_bid_data else 0
    best_ask = best_ask_data.get('price', 0) if best_ask_data else 0
    market_mid_check = market_mid or 0

    mandatory_sell_triggered = any([
        best_bid >= MANDATORY_SELL_PRICE,
        market_mid_check >= MANDATORY_SELL_PRICE,
        (best_ask >= MANDATORY_SELL_PRICE and best_bid >= 0.97)
    ])

    if mandatory_sell_triggered and balance_raw >= MIN_BALANCE_SELL:
        print("🚨 MANDATORY SELL CONDITIONS ACTIVE!")
        print(f"📊 Market prices: Bid ${best_bid:.2f}, Mid ${market_mid_check:.2f}")
        print("⚡ Will execute mandatory sell on next iteration")

    # Current Orders
    if trading_status['buy_order_active']:
        print(f"🟢 BUY ORDER ACTIVE @ ${trading_status['current_buy_price']:.3f}")
    else:
        print("⚪ No active buy order")

    if trading_status['sell_order_active']:
        print(f"🔴 SELL ORDER ACTIVE @ ${trading_status['current_sell_price']:.3f}")
    else:
        print("⚪ No active sell order")

    # Dynamic Target Prices - with safe None handling
    target_bid = targets.get('bid') if targets else None
    target_ask = targets.get('ask') if targets else None
    
    if target_bid and target_ask:
        print(f"🎯 Dynamic Targets: ${target_bid:.3f} x ${target_ask:.3f}")
        
        target_spread = target_ask - target_bid
        print(f"🔸 Target Spread: ${target_spread:.3f}")
        
        if spread_info.get('predicted_spread'):
            print(f"🔸 Predicted Spread: ${spread_info['predicted_spread']:.3f}")
            
        # Show positioning vs market - with safe None handling
        if best_bid_data and best_ask_data and best_bid > 0 and best_ask > 0:
            market_spread = best_ask - best_bid
            spread_comparison = target_spread - market_spread
            print(f"🔸 Spread vs Market: {spread_comparison:+.3f}")
        else:
            print("🔸 Market spread: N/A (empty book)")
    else:
        print("🎯 No dynamic target prices available")

    # Book status warning
    if not best_bid_data:
        print("⚠️ WARNING: No bids in order book!")
    if not best_ask_data:
        print("⚠️ WARNING: No asks in order book!")

    # Recent formula accuracy
    if hasattr(state, 'formula_accuracy_history') and state.formula_accuracy_history:
        recent_predictions = state.formula_accuracy_history[-5:]
        if recent_predictions:
            recent_avg = sum(acc for _, _, _, acc in recent_predictions) / len(recent_predictions)
            print(f"📈 Last 5 predictions: {recent_avg:.1%} accuracy")

    print("=" * 90)