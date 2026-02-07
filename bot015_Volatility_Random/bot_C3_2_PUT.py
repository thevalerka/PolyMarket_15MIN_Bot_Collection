#!/usr/bin/env python3
"""
Strategy C3_2 - ATR-Based Binary Options BTC_PUT Trader

Strategy C3_2 Characteristics:
- ATR-based dynamic TP/SL calculation
- TP:SL ratio of 1:3.5 (even more aggressive TP, much wider SL)
- 6 periods of 15 seconds each with exponential weighting
- Adaptive to market volatility

Entry Rules:
- Buy when:
  1. We hold 0 assets
  2. ATR data is available (need 90 seconds of price history)
  3. Ask price <= $0.96
  4. Calculated TP < $0.99
  5. Sufficient USDC balance

Exit Rules:
- Take Profit: Dynamic based on ATR (capped at $0.99)
- Stop Loss: Dynamic based on ATR (1:3 ratio with TP)
- Emergency Sell: ALWAYS sell when bid_price >= $0.99
- Adaptive: TP/SL adjust every second based on current ATR

Position Sizing:
- Fixed 5 shares per trade
- Use ask_price for buying
- Use bid_price for selling

ATR Calculation:
- 6 periods of 15 seconds (90 seconds total)
- Exponentially weighted (recent periods have more weight)
- True Range considers: high-low, high-prev_close, low-prev_close

pm2 start bot_C3_2_PUT.py --cron-restart="00 * * * *" --interpreter python3
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
from dotenv import load_dotenv

# Import Polymarket trading core
sys.path.insert(0, '/home/claude')
from polymarket_trading_core_debug import PolymarketTrader, load_credentials_from_env


class ATRCalculator:
    """Calculate Average True Range with exponential weighting"""

    @staticmethod
    def calculate_period_ranges(prices: List[float], timestamps: List[datetime],
                                period_seconds: int = 15) -> List[Tuple[float, float, datetime]]:
        """
        Calculate max-min range for each period
        Returns list of (max_price, min_price, period_end_time)
        """
        if not prices or not timestamps:
            return []

        periods = []
        current_period_prices = []
        current_period_start = timestamps[0]

        for price, ts in zip(prices, timestamps):
            time_diff = (ts - current_period_start).total_seconds()

            if time_diff >= period_seconds:
                if current_period_prices:
                    max_price = max(current_period_prices)
                    min_price = min(current_period_prices)
                    periods.append((max_price, min_price, ts))

                current_period_prices = [price]
                current_period_start = ts
            else:
                current_period_prices.append(price)

        if current_period_prices:
            max_price = max(current_period_prices)
            min_price = min(current_period_prices)
            periods.append((max_price, min_price, timestamps[-1]))

        return periods

    @staticmethod
    def calculate_atr(prices: List[float], timestamps: List[datetime],
                     period_seconds: int = 15, num_periods: int = 6) -> Optional[float]:
        """
        Calculate ATR using exponential weighting
        """
        periods = ATRCalculator.calculate_period_ranges(prices, timestamps, period_seconds)

        if len(periods) < 2:
            return None

        recent_periods = periods[-num_periods:] if len(periods) >= num_periods else periods

        if len(recent_periods) < 2:
            return None

        true_ranges = []

        for i in range(len(recent_periods)):
            max_price, min_price, _ = recent_periods[i]

            if i == 0:
                tr = max_price - min_price
            else:
                prev_max, prev_min, _ = recent_periods[i-1]
                prev_close = (prev_max + prev_min) / 2

                high_low = max_price - min_price
                high_prev_close = abs(max_price - prev_close)
                low_prev_close = abs(min_price - prev_close)

                tr = max(high_low, high_prev_close, low_prev_close)

            true_ranges.append(tr)

        if not true_ranges:
            return None

        # Apply exponential weighting
        n = len(true_ranges)
        alpha = 2.0 / (n + 1)

        weighted_sum = 0.0
        weight_sum = 0.0

        for i, tr in enumerate(true_ranges):
            weight = (1 - alpha) ** (n - 1 - i)
            weighted_sum += tr * weight
            weight_sum += weight

        atr = weighted_sum / weight_sum if weight_sum > 0 else 0.0

        return atr

    @staticmethod
    def estimate_movement_probability(atr: float, current_price: float) -> Tuple[float, float]:
        """
        Estimate probability and expected magnitude based on ATR
        """
        if current_price == 0:
            return 0.5, 0.05

        atr_pct = (atr / current_price) * 100

        if atr_pct > 15:
            prob_movement = min(0.95, 0.65 + (atr_pct - 15) / 50)
            expected_magnitude = min(0.15, atr * 1.5)
        elif atr_pct > 5:
            prob_movement = 0.45 + (atr_pct - 5) / 50
            expected_magnitude = atr * 1.2
        else:
            prob_movement = 0.35 + atr_pct / 25
            expected_magnitude = max(0.02, atr * 0.8)

        return prob_movement, expected_magnitude


class StrategyC3_2Trader:
    """BTC_PUT trader with Strategy C3_2 (ATR-based, 1:3 ratio)"""

    DATA_FILE = '/home/ubuntu/013_2025_polymarket/15M_PUT.json'

    POSITION_SIZE = 5
    MAX_BUY_PRICE = 0.96
    EMERGENCY_SELL_PRICE = 0.99
    MAX_TP_PRICE = 0.99  # Cap TP at 0.99

    # ATR parameters
    ATR_PERIOD_SECONDS = 15
    ATR_NUM_PERIODS = 6
    PRICE_HISTORY_SECONDS = 100  # Store 100 seconds for 6x15s periods

    def __init__(self, credentials: Dict[str, str]):
        """Initialize Strategy C2 trader"""
        self.trader = PolymarketTrader(
            clob_api_url=credentials['clob_api_url'],
            private_key=credentials['private_key'],
            api_key=credentials['api_key'],
            api_secret=credentials['api_secret'],
            api_passphrase=credentials['api_passphrase']
        )

        # Position tracking
        self.current_position = {
            'size': 0.0,
            'avg_buy_price': 0.0,
            'token_id': None,
            'take_profit': 0.0,
            'stop_loss': 0.0,
            'atr_at_entry': 0.0,
            'entry_time': None,
            'tp_sl_adjustments': 0
        }

        # Price history with timestamps
        self.price_history: Deque[float] = deque(maxlen=self.PRICE_HISTORY_SECONDS)
        self.timestamp_history: Deque[datetime] = deque(maxlen=self.PRICE_HISTORY_SECONDS)

        self.atr_calc = ATRCalculator()

        print("=" * 80)
        print("🎯 STRATEGY C3_2 - ATR-BASED TRADER (1:3 RATIO)")
        print("=" * 80)
        print(f"📊 Asset: BTC_PUT only")
        print(f"📦 Position Size: {self.POSITION_SIZE} shares")
        print(f"🧮 Strategy: ATR-based dynamic TP/SL")
        print(f"📐 TP:SL Ratio: 1:3 (very aggressive TP, very wide SL)")
        print(f"📏 ATR Calculation: 6 periods × 15 seconds (exponentially weighted)")
        print(f"🚫 Max Buy Price: ${self.MAX_BUY_PRICE}")
        print(f"🚨 Emergency Sell: ${self.EMERGENCY_SELL_PRICE}")
        print(f"⚡ TP/SL Updates: Every second based on current ATR")
        print(f"💰 Buy Price: ask_price")
        print(f"💰 Sell Price: bid_price")
        print("=" * 80)

    def read_market_data(self) -> Optional[Dict]:
        """Read BTC PUT market data from JSON file"""
        try:
            if not os.path.exists(self.DATA_FILE):
                return None

            with open(self.DATA_FILE, 'r') as f:
                data = json.load(f)

            return {
                'asset_id': data.get('asset_id'),
                'asset_name': data.get('asset_name'),
                'best_bid': data.get('best_bid', {}),
                'best_ask': data.get('best_ask', {}),
                'timestamp': data.get('timestamp'),
                'timestamp_readable': data.get('timestamp_readable')
            }

        except Exception as e:
            print(f"❌ Error reading market data: {e}")
            return None

    def minutes_to_expiry(self) -> float:
        """Calculate minutes remaining until current period expires"""
        now = datetime.now()
        minutes_into_hour = now.minute % 15
        seconds_into_period = minutes_into_hour * 60 + now.second
        seconds_remaining = 900 - seconds_into_period
        return seconds_remaining / 60

    def update_price_history(self, bid_price: float):
        """Add new price to history"""
        self.price_history.append(bid_price)
        self.timestamp_history.append(datetime.now())

    def calculate_atr(self) -> Optional[float]:
        """Calculate current ATR"""
        if len(self.price_history) < 20:  # Need some minimum data
            return None

        return self.atr_calc.calculate_atr(
            list(self.price_history),
            list(self.timestamp_history),
            self.ATR_PERIOD_SECONDS,
            self.ATR_NUM_PERIODS
        )

    def calculate_dynamic_tpsl(self, atr: float, entry_price: float) -> Tuple[float, float]:
        """
        Calculate dynamic TP and SL for Strategy C3_2
        Maintains 1:3.5 ratio (TP:SL)
        """
        prob_movement, expected_magnitude = self.atr_calc.estimate_movement_probability(
            atr, entry_price
        )

        # Base TP/SL with 1:3 ratio
        base_tp = 0.015
        base_sl = 0.0525  # 3× of TP

        scaling_factor = expected_magnitude * prob_movement

        tp_distance = base_tp + (scaling_factor * 1.0)
        sl_distance = base_sl + (scaling_factor * 3.0)

        # Clamp and maintain ratio
        tp_distance = max(0.015, min(0.20, tp_distance))
        sl_distance = tp_distance * 3.5 # Maintain 1:3.5 ratio

        take_profit = min(self.MAX_TP_PRICE, entry_price + tp_distance)
        stop_loss = entry_price - sl_distance

        return take_profit, stop_loss

    def get_usdc_balance(self) -> float:
        """Get USDC balance"""
        try:
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

            response = self.trader.client.get_balance_allowance(
                params=BalanceAllowanceParams(
                    asset_type=AssetType.COLLATERAL
                )
            )

            balance_raw = int(response.get('balance', 0))
            balance_usdc = balance_raw / 10**6

            return balance_usdc

        except Exception as e:
            print(f"❌ Error getting USDC balance: {e}")
            return 0.0

    def check_token_balance(self, token_id: str) -> float:
        """Check balance of a specific token"""
        try:
            balance_raw, balance = self.trader.get_token_balance(token_id)
            return balance
        except Exception as e:
            print(f"❌ Error checking balance for {token_id[:12]}...: {e}")
            return 0.0

    def execute_buy(self, token_id: str, ask_price: float, take_profit: float,
                    stop_loss: float, atr: float) -> bool:
        """
        Execute buy order for BTC_PUT with multi-phase verification
        Uses ORIGINAL WORKING LOGIC from trader_btc_PUT_oscillation.py
        """
        try:
            print(f"\n{'='*70}")
            print(f"🛒 EXECUTING BUY ORDER - STRATEGY C3_2")
            print(f"{'='*70}")
            print(f"📊 Asset: BTC_PUT")
            print(f"📦 Size: {self.POSITION_SIZE} shares")
            print(f"💰 Ask Price: ${ask_price:.4f}")
            print(f"🎯 Take Profit: ${take_profit:.4f} (+${take_profit - ask_price:.4f})")
            print(f"🛡️  Stop Loss: ${stop_loss:.4f} (-${ask_price - stop_loss:.4f})")
            print(f"📐 TP:SL Ratio: 1:3")
            print(f"📏 ATR: ${atr:.4f} ({(atr/ask_price)*100:.2f}% of price)")

            required = ask_price * self.POSITION_SIZE
            print(f"💵 Expected Cost: ${required:.2f}")

            # Check initial balance
            initial_usdc = self.get_usdc_balance()
            print(f"💰 Initial USDC: ${initial_usdc:.2f}")

            if initial_usdc < required:
                print(f"❌ Insufficient balance: need ${required:.2f}, have ${initial_usdc:.2f}")
                return False

            # ============================================================
            # PHASE 1: PLACE ORDER
            # ============================================================

            print(f"\n🚀 PHASE 1: Placing buy order...")

            order_id = self.trader.place_buy_order(
                token_id=token_id,
                price=ask_price,
                quantity=self.POSITION_SIZE
            )

            if not order_id:
                print(f"❌ Failed to place buy order")
                return False

            print(f"✅ Buy order placed: {order_id[:16] if len(order_id) > 16 else order_id}...")

            # ============================================================
            # PHASE 2: DELAYED VERIFICATION - Check at 1s, 5s, 10s, 20s
            # ============================================================

            print("\n🔍 PHASE 2: Verifying position (laggy API handling)...")

            verification_times = [1, 5, 10, 20]
            position_confirmed = False
            usdc_spent = 0

            for wait_time in verification_times:
                print(f"\n⏳ Waiting {wait_time} seconds for API to update...")
                time.sleep(wait_time)

                print(f"\n📊 Verification at {wait_time}s mark:")

                # Check USDC balance (faster hint)
                current_usdc = self.get_usdc_balance()
                usdc_spent = initial_usdc - current_usdc
                print(f"   💰 USDC: ${initial_usdc:.2f} → ${current_usdc:.2f} (spent: ${usdc_spent:.2f})")

                # Check token balance (ERC-1155)
                balance = self.check_token_balance(token_id)
                print(f"   🪙 Token balance: {balance:.2f} shares")

                # Determine status based on balances
                if balance >= self.POSITION_SIZE * 0.95:
                    position_confirmed = True
                    print(f"   ✅ Position confirmed by token balance")
                    break
                elif usdc_spent > (required * 0.8):
                    print(f"   💡 USDC spent suggests position may be open (API lag)")
                    position_confirmed = True  # Trust USDC balance
                    break

            # ============================================================
            # PHASE 3: DECISION LOGIC
            # ============================================================

            print("\n🎲 PHASE 3: Final status check...")
            print(f"   Position confirmed: {position_confirmed}")
            print(f"   USDC spent: ${usdc_spent:.2f} (expected: ${required:.2f})")

            # Get final balance
            final_balance = self.check_token_balance(token_id)
            print(f"   Final token balance: {final_balance:.2f}")

            # Case 1: Position confirmed
            if position_confirmed and final_balance >= 0.5:
                print("\n✅ SUCCESS: Position opened")

                # Update position tracking WITH STRATEGY C2 DATA
                self.current_position['size'] = final_balance
                self.current_position['avg_buy_price'] = ask_price
                self.current_position['token_id'] = token_id
                self.current_position['take_profit'] = take_profit
                self.current_position['stop_loss'] = stop_loss
                self.current_position['atr_at_entry'] = atr
                self.current_position['entry_time'] = datetime.now()
                self.current_position['tp_sl_adjustments'] = 0

                print(f"\n📊 POSITION OPENED")
                print(f"   Size: {final_balance:.2f} shares")
                print(f"   Avg Buy Price: ${ask_price:.4f}")
                print(f"   Take Profit Target: ${take_profit:.4f}")
                print(f"   Stop Loss Target: ${stop_loss:.4f}")
                print(f"   Emergency Sell: ${self.EMERGENCY_SELL_PRICE:.4f}")
                print(f"   ATR at Entry: ${atr:.4f}")

                return True

            # Case 2: Position not confirmed and wallet unchanged
            elif not position_confirmed and usdc_spent < 0.5:
                print("\n⚠️  Position not confirmed and wallet unchanged")
                print("   🔄 Cancelling all orders...")
                self.trader.cancel_all_orders()
                time.sleep(2)

                # Check if conditions still valid
                data = self.read_market_data()
                if data:
                    print("   Conditions may still be valid - will retry in next cycle")
                else:
                    print("   ❌ Unable to read market data")

                return False

            # Case 3: Some USDC spent but no tokens (API lag or partial fill)
            else:
                print("\n⚠️  USDC spent but position not fully confirmed")
                print("   Checking final balance...")

                # One more check after 5 seconds
                time.sleep(5)
                final_balance = self.check_token_balance(token_id)

                if final_balance >= 0.5:
                    print(f"   ✅ Position found: {final_balance:.2f} shares")

                    # Update position tracking
                    self.current_position['size'] = final_balance
                    self.current_position['avg_buy_price'] = ask_price
                    self.current_position['token_id'] = token_id
                    self.current_position['take_profit'] = take_profit
                    self.current_position['stop_loss'] = stop_loss
                    self.current_position['atr_at_entry'] = atr
                    self.current_position['entry_time'] = datetime.now()
                    self.current_position['tp_sl_adjustments'] = 0

                    return True
                else:
                    print(f"   ❌ No position found - cancelling orders")
                    self.trader.cancel_all_orders()
                    return False

        except Exception as e:
            print(f"❌ Error executing buy: {e}")
            import traceback
            traceback.print_exc()
            return False

    def execute_sell(self, token_id: str, bid_price: float, reason: str) -> bool:
        """
        Execute sell order for BTC_PUT
        Uses ORIGINAL WORKING LOGIC from trader_btc_PUT_oscillation.py
        """
        try:
            # SAFETY CHECK: Verify we actually have tokens to sell
            actual_balance = self.check_token_balance(token_id)

            if actual_balance < 0.5:
                print(f"\n⚠️  SELL ABORTED: No tokens in wallet")
                print(f"   Tracked size: {self.current_position['size']:.2f}")
                print(f"   Actual balance: {actual_balance:.2f}")

                # Reset position
                self.current_position = {
                    'size': 0.0,
                    'avg_buy_price': 0.0,
                    'token_id': None,
                    'take_profit': 0.0,
                    'stop_loss': 0.0,
                    'atr_at_entry': 0.0,
                    'entry_time': None,
                    'tp_sl_adjustments': 0
                }
                return False

            # Use actual balance for selling
            size = actual_balance

            print(f"\n{'='*60}")
            print(f"💰 EXECUTING SELL ORDER - {reason}")
            print(f"{'='*60}")
            print(f"📊 Asset: BTC_PUT")
            print(f"📦 Size: {size:.2f} shares (verified in wallet)")
            print(f"💰 Current Bid: ${bid_price:.4f}")
            print(f"📊 Buy Price: ${self.current_position['avg_buy_price']:.4f}")

            # Calculate P&L
            pnl = (bid_price - self.current_position['avg_buy_price']) * size
            pnl_percent = ((bid_price / self.current_position['avg_buy_price']) - 1) * 100

            print(f"📈 P&L: ${pnl:.2f} ({pnl_percent:+.2f}%)")
            print(f"🔄 TP/SL Adjustments: {self.current_position['tp_sl_adjustments']}")

            # Set sell price (slightly below bid for quick execution)
            if bid_price <= 0:
                print(f"⚠️  No bid price available, using $0.01")
                sell_price = 0.01
            else:
                sell_price = max(0.01, bid_price - 0.001)

            print(f"💰 Sell Price: ${sell_price:.4f}")
            print(f"💵 Total Value: ${sell_price * size:.2f}")

            # Place sell order using correct method
            order_id = self.trader.place_sell_order(
                token_id=token_id,
                price=sell_price,
                quantity=size
            )

            if not order_id:
                print(f"❌ Failed to place sell order")
                return False

            print(f"✅ Sell order placed: {order_id[:16] if len(order_id) > 16 else order_id}...")

            # Wait and verify
            time.sleep(3)

            # Check if position closed
            balance = self.check_token_balance(token_id)

            if balance < 0.5:  # Position closed
                print(f"✅ Sell order FILLED - Position CLOSED")
                print(f"💵 Final P&L: ${pnl:.2f} ({pnl_percent:+.2f}%)")

                # Reset position
                self.current_position = {
                    'size': 0.0,
                    'avg_buy_price': 0.0,
                    'token_id': None,
                    'take_profit': 0.0,
                    'stop_loss': 0.0,
                    'atr_at_entry': 0.0,
                    'entry_time': None,
                    'tp_sl_adjustments': 0
                }

                return True
            else:
                print(f"⚠️  Sell order may not be filled - Remaining balance: {balance:.2f}")

                # Update position size
                self.current_position['size'] = balance

                # Try to cancel unfilled portion
                try:
                    self.trader.cancel_all_orders()
                    print(f"🔄 Cancelled unfilled orders")
                except:
                    pass

                return balance < size * 0.1  # Consider success if >90% filled

        except Exception as e:
            print(f"❌ Error executing sell: {e}")
            import traceback
            traceback.print_exc()
            return False


    def execute_sell_TP(self, token_id: str, ask_price: float, reason: str) -> bool:
        """
        Execute sell order for BTC_PUT TAKE PROFIT ONLY we issue a limit order
        Uses ORIGINAL WORKING LOGIC from trader_btc_PUT_oscillation.py
        """
        try:
            # SAFETY CHECK: Verify we actually have tokens to sell
            actual_balance = self.check_token_balance(token_id)

            if actual_balance < 0.5:
                print(f"\n⚠️  SELL ABORTED: No tokens in wallet")
                print(f"   Tracked size: {self.current_position['size']:.2f}")
                print(f"   Actual balance: {actual_balance:.2f}")

                # Reset position
                self.current_position = {
                    'size': 0.0,
                    'avg_buy_price': 0.0,
                    'token_id': None,
                    'take_profit': 0.0,
                    'stop_loss': 0.0,
                    'atr_at_entry': 0.0,
                    'entry_time': None,
                    'tp_sl_adjustments': 0
                }
                return False

            # Use actual balance for selling
            size = actual_balance

            print(f"\n{'='*60}")
            print(f"💰 EXECUTING SELL ORDER - {reason}")
            print(f"{'='*60}")
            print(f"📊 Asset: BTC_PUT")
            print(f"📦 Size: {size:.2f} shares (verified in wallet)")
            print(f"💰 Current ask: ${ask_price:.4f}")
            print(f"📊 Buy Price: ${self.current_position['avg_buy_price']:.4f}")

            # Calculate P&L
            pnl = (ask_price - self.current_position['avg_buy_price']) * size
            pnl_percent = ((ask_price / self.current_position['avg_buy_price']) - 1) * 100

            print(f"📈 P&L: ${pnl:.2f} ({pnl_percent:+.2f}%)")
            print(f"🔄 TP/SL Adjustments: {self.current_position['tp_sl_adjustments']}")

            # Set sell price (slightly below ask for quick execution)
            if ask_price <= 0:
                print(f"⚠️  No ask price available, using $0.01")
                sell_price = 0.01
            else:
                sell_price = ask_price

            if ask_price >= 0.99:
                sell_price = 0.99


            print(f"💰 Sell Price: ${sell_price:.4f}")
            print(f"💵 Total Value: ${sell_price * size:.2f}")

            # Place sell order using correct method
            order_id = self.trader.place_sell_order(
                token_id=token_id,
                price=sell_price,
                quantity=size
            )

            if not order_id:
                print(f"❌ Failed to place sell order")
                return False

            print(f"✅ Sell order placed: {order_id[:16] if len(order_id) > 16 else order_id}...")

            # Wait and verify
            time.sleep(3)

            # Check if position closed
            balance = self.check_token_balance(token_id)

            if balance < 0.5:  # Position closed
                print(f"✅ Sell order FILLED - Position CLOSED")
                print(f"💵 Final P&L: ${pnl:.2f} ({pnl_percent:+.2f}%)")

                # Reset position
                self.current_position = {
                    'size': 0.0,
                    'avg_buy_price': 0.0,
                    'token_id': None,
                    'take_profit': 0.0,
                    'stop_loss': 0.0,
                    'atr_at_entry': 0.0,
                    'entry_time': None,
                    'tp_sl_adjustments': 0
                }

                return True
            else:
                print(f"⚠️  Sell order may not be filled - Remaining balance: {balance:.2f}")

                # Update position size
                self.current_position['size'] = balance

                # Try to cancel unfilled portion
                try:
                    self.trader.cancel_all_orders()
                    print(f"🔄 Cancelled unfilled orders")
                except:
                    pass

                return balance < size * 0.1  # Consider success if >90% filled

        except Exception as e:
            print(f"❌ Error executing sell: {e}")
            import traceback
            traceback.print_exc()
            return False

    def check_exit_conditions(self, bid_price: float, current_atr: Optional[float]) -> Optional[str]:
        """
        Check if we should exit position (STRATEGY C2 LOGIC)

        Returns exit reason or None
        """
        if self.current_position['size'] < 0.5:
            return None

        # Emergency sell at 0.99
        if bid_price >= self.EMERGENCY_SELL_PRICE:
            return "EMERGENCY_0.99"

        # Update TP/SL based on current ATR if available
        if current_atr is not None and current_atr != self.current_position['atr_at_entry']:
            entry_price = self.current_position['avg_buy_price']
            new_tp, new_sl = self.calculate_dynamic_tpsl(current_atr, entry_price)

            # Check if significant change (> $0.01)
            if abs(new_tp - self.current_position['take_profit']) > 0.01 or \
               abs(new_sl - self.current_position['stop_loss']) > 0.01:

                old_tp = self.current_position['take_profit']
                old_sl = self.current_position['stop_loss']

                self.current_position['take_profit'] = new_tp
                self.current_position['stop_loss'] = new_sl
                self.current_position['atr_at_entry'] = current_atr
                self.current_position['tp_sl_adjustments'] += 1

                print(f"\n🔄 TP/SL ADJUSTED (#{self.current_position['tp_sl_adjustments']}) - ATR changed")
                print(f"   ATR: ${current_atr:.4f} ({(current_atr/entry_price)*100:.2f}%)")
                print(f"   Old TP: ${old_tp:.4f} → New TP: ${new_tp:.4f} (Δ ${new_tp-old_tp:+.4f})")
                print(f"   Old SL: ${old_sl:.4f} → New SL: ${new_sl:.4f} (Δ ${new_sl-old_sl:+.4f})")

        # Check TP
        if bid_price >= self.current_position['take_profit']:
            return "TAKE_PROFIT"

        # Check SL
        if bid_price <= self.current_position['stop_loss']:
            return "STOP_LOSS"

        return None

    def monitor_position(self, data: Dict, current_atr: Optional[float]):
        """Monitor open position and check exit conditions"""
        if self.current_position['size'] < 0.5:
            return

        # CHECK EXPIRY: Stop trying to sell when period is expired
        minutes_left = self.minutes_to_expiry()
        if minutes_left <= 0:
            print(f"\n⏰ MARKET EXPIRED: Period has ended")
            print(f"   Stopping sell attempts - market is closed")
            print(f"   Position will be settled at expiry")
            print(f"   Resetting position tracking for next period")
            self.current_position = {
                'size': 0.0,
                'avg_buy_price': 0.0,
                'token_id': None,
                'take_profit': 0.0,
                'stop_loss': 0.0,
                'atr_at_entry': 0.0,
                'entry_time': None,
                'tp_sl_adjustments': 0
            }
            return

        # SAFETY CHECK: Verify we actually have the tokens
        token_id = self.current_position['token_id']
        actual_balance = self.check_token_balance(token_id)

        if actual_balance < 0.5:
            print(f"\n⚠️  WARNING: Position tracking shows {self.current_position['size']:.2f} but wallet shows {actual_balance:.2f}")
            print(f"   Resetting position tracking")
            self.current_position = {
                'size': 0.0,
                'avg_buy_price': 0.0,
                'token_id': None,
                'take_profit': 0.0,
                'stop_loss': 0.0,
                'atr_at_entry': 0.0,
                'entry_time': None,
                'tp_sl_adjustments': 0
            }
            return

        # Update position size if there's a discrepancy
        if abs(actual_balance - self.current_position['size']) > 0.5:
            print(f"\n⚠️  Position size mismatch: tracked {self.current_position['size']:.2f}, actual {actual_balance:.2f}")
            print(f"   Updating to actual balance")
            self.current_position['size'] = actual_balance

        bid_price = float(data['best_bid'].get('price', 0))
        ask_price = float(data['best_ask'].get('price', 0))

        if bid_price == 0:
            print("⚠️  Invalid bid price (0) - skipping exit check")
            return

        # Check exit conditions with current ATR
        exit_reason = self.check_exit_conditions(bid_price, current_atr)

        if exit_reason == "STOP_LOSS":
            print(f"\n🚨 EXIT CONDITION MET: {exit_reason}")
            self.execute_sell(token_id, bid_price, exit_reason)
        if exit_reason == "TAKE_PROFIT":
            print(f"\n🚨 EXIT CONDITION MET: {exit_reason}")
            self.execute_sell_TP(token_id, ask_price, exit_reason)

    def check_entry_conditions(self, data: Dict, current_atr: Optional[float]) -> bool:
        """
        Check if we should enter a position (STRATEGY C2 LOGIC)

        Entry conditions:
        1. No current position
        2. ATR is available
        3. Ask price <= MAX_BUY_PRICE
        4. Calculated TP < MAX_TP_PRICE
        5. Sufficient balance
        """
        # Must have 0 position
        if self.current_position['size'] >= 0.5:
            return False

        # Need ATR data
        if current_atr is None:
            return False

        ask_price = float(data['best_ask'].get('price', 0))

        if ask_price == 0 or ask_price > self.MAX_BUY_PRICE:
            return False

        # Calculate TP/SL
        tp, sl = self.calculate_dynamic_tpsl(current_atr, ask_price)

        # Check if TP is reasonable (below 0.99)
        if tp >= self.MAX_TP_PRICE:
            return False

        # Check balance
        required = ask_price * self.POSITION_SIZE
        balance = self.get_usdc_balance()

        if balance < required:
            return False

        # All conditions met - execute buy
        token_id = data['asset_id']
        return self.execute_buy(token_id, ask_price, tp, sl, current_atr)

    def run_trading_cycle(self):
        """Main trading loop"""
        print(f"\n{'='*70}")
        print(f"🚀 STARTING STRATEGY C3_2 TRADING CYCLE")
        print(f"{'='*70}\n")

        last_status_time = time.time()

        try:
            while True:
                # Read market data
                data = self.read_market_data()

                if not data:
                    time.sleep(1)
                    continue

                bid_price = float(data['best_bid'].get('price', 0))
                ask_price = float(data['best_ask'].get('price', 0))

                # Update price history
                if bid_price > 0:
                    self.update_price_history(bid_price)

                # Calculate current ATR
                current_atr = self.calculate_atr()

                # Monitor position if we have one
                if self.current_position['size'] > 0:
                    self.monitor_position(data, current_atr)

                # Check entry conditions if we don't have a position
                if self.current_position['size'] == 0:
                    self.check_entry_conditions(data, current_atr)

                # Status update every 10 seconds
                if time.time() - last_status_time >= 10:
                    print(f"\n--- Status Update: {datetime.now().strftime('%H:%M:%S')} ---")
                    print(f"⏰ Minutes to Expiry: {self.minutes_to_expiry():.1f}")

                    if current_atr is not None:
                        atr_pct = (current_atr / bid_price * 100) if bid_price > 0 else 0
                        print(f"📏 Current ATR: ${current_atr:.4f} ({atr_pct:.2f}%)")
                    else:
                        print(f"⏳ Building ATR... ({len(self.price_history)}/{self.PRICE_HISTORY_SECONDS} seconds)")

                    print(f"💰 Bid: ${bid_price:.4f} | Ask: ${ask_price:.4f}")

                    if self.current_position['size'] > 0:
                        entry_price = self.current_position['avg_buy_price']
                        pnl = (bid_price - entry_price) * self.current_position['size'] if bid_price > 0 else 0
                        pnl_pct = ((bid_price - entry_price) / entry_price * 100) if entry_price > 0 and bid_price > 0 else 0

                        duration = (datetime.now() - self.current_position['entry_time']).total_seconds() if self.current_position['entry_time'] else 0

                        print(f"📊 Position: {self.current_position['size']:.2f} shares @ ${entry_price:.4f}")
                        print(f"📈 PNL: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
                        print(f"🎯 TP: ${self.current_position['take_profit']:.4f} | SL: ${self.current_position['stop_loss']:.4f}")
                        print(f"🔄 Adjustments: {self.current_position['tp_sl_adjustments']}")
                        print(f"⏱️  Duration: {duration:.0f}s")
                    else:
                        print(f"📊 No open position")

                    print()
                    last_status_time = time.time()

                time.sleep(1)

        except KeyboardInterrupt:
            print(f"\n\n{'='*70}")
            print(f"🛑 TRADING STOPPED BY USER")
            print(f"{'='*70}\n")

            if self.current_position['size'] > 0:
                print(f"⚠️  Open position detected - consider manual close")
                print(f"   Token ID: {self.current_position['token_id']}")
                print(f"   Size: {self.current_position['size']:.2f}")
                print(f"   Entry: ${self.current_position['avg_buy_price']:.4f}")
                print(f"   TP: ${self.current_position['take_profit']:.4f}")
                print(f"   SL: ${self.current_position['stop_loss']:.4f}")


def main():
    """Main entry point"""
    # # Load credentials
    # load_dotenv()
    # credentials = load_credentials_from_env()

    # Load credentials
    try:
        env_path = '/home/ubuntu/013_2025_polymarket/keys/keys_ovh38.env'
        credentials = load_credentials_from_env(env_path)
        print(f"✅ Credentials loaded from {env_path}")
    except Exception as e:
        print(f"❌ Error loading credentials: {e}")
        return

    if not credentials:
        print("❌ Failed to load credentials from environment")
        sys.exit(1)

    # Create and run trader
    trader = StrategyC3_2Trader(credentials)
    trader.run_trading_cycle()


if __name__ == "__main__":
    main()
