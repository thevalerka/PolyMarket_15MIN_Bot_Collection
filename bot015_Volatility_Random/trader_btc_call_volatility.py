#!/usr/bin/env python3
"""
15-Minute Binary Options BTC_CALL Trader - Volatility-Based Strategy

Strategy:
- Trade ONLY BTC_CALL
- Track second-by-second price data for entire 15-minute period
- Calculate real-time price volatility (standard deviation)
- Compare to historical median volatility
- ONLY TRADE when current volatility is HIGH compared to median

Entry Rules:
- Buy 5 shares when:
  1. We hold 0 assets
  2. Current volatility > median volatility
  3. Sufficient USDC balance

Exit Rules:
- Sell ALL shares when:
  1. 5% profit reached (sell at bid_price >= 1.05 * avg_buy_price)
  2. 10% loss reached (sell at bid_price <= 0.90 * avg_buy_price)

Position Sizing:
- Fixed 5 shares per trade
- Use ask_price for buying
- Use bid_price for selling

Timeframe:
- Trade throughout entire 15-minute period (no time constraints)
- Continuous second-by-second monitoring
"""

import os
import sys
import json
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Deque
from collections import deque
from dotenv import load_dotenv

# Import Polymarket trading core
sys.path.insert(0, '/home/claude')
from polymarket_trading_core_debug import PolymarketTrader, load_credentials_from_env


class BTCCallVolatilityTrader:
    """BTC_CALL trader with volatility-based entry filtering"""

    # File path for BTC CALL market data
    DATA_FILE = '/home/ubuntu/013_2025_polymarket/15M_CALL.json'

    # Volatility history file (to track median)
    VOLATILITY_HISTORY_FILE = '/home/ubuntu/013_2025_polymarket/btc_call_volatility_history.json'

    # Trading parameters
    POSITION_SIZE = 5  # Fixed 5 shares per trade
    TAKE_PROFIT_PERCENT = 0.05  # 5% profit
    STOP_LOSS_PERCENT = 0.10  # 10% loss

    # Volatility tracking
    MAX_PRICE_HISTORY = 900  # Store up to 15 minutes of second-by-second data
    MIN_VOLATILITY_THRESHOLD = 0.032  # Minimum absolute volatility required to trade

    def __init__(self, credentials: Dict[str, str]):
        """Initialize the BTC CALL volatility trader"""
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
            'token_id': None
        }

        # Price history for volatility calculation (deque for efficient append/pop)
        self.price_history: Deque[float] = deque(maxlen=self.MAX_PRICE_HISTORY)

        # Volatility tracking
        self.current_volatility = 0.0
        self.period_volatilities = []  # Track volatility values throughout current period
        self.period_average_volatility = 0.0  # Average volatility for current period
        self.historical_volatility = []  # Historical period average volatilities

        # Load historical volatility data
        self._load_volatility_history()

        print("=" * 80)
        print("🎯 BTC_CALL VOLATILITY-BASED TRADER")
        print("=" * 80)
        print(f"📊 Asset: BTC_CALL only")
        print(f"📦 Position Size: {self.POSITION_SIZE} shares")
        print(f"📈 Take Profit: {self.TAKE_PROFIT_PERCENT*100}% (sell when bid >= {1+self.TAKE_PROFIT_PERCENT:.3f}x avg_buy)")
        print(f"📉 Stop Loss: {self.STOP_LOSS_PERCENT*100}% (sell when bid <= {1-self.STOP_LOSS_PERCENT:.3f}x avg_buy)")
        print(f"🌊 Volatility Filter: Must be > period average AND >= {self.MIN_VOLATILITY_THRESHOLD}")
        print(f"📊 Period Average: Calculated from beginning of each 15-min period")
        print(f"⏰ Trading Window: Entire 15-minute period")
        print(f"⚡ Monitoring: Every second")
        print(f"💰 Buy Price: ask_price")
        print(f"💰 Sell Price: bid_price")
        print("=" * 80)

    def _load_volatility_history(self):
        """Load historical volatility data (not currently used for trading decisions)"""
        try:
            if os.path.exists(self.VOLATILITY_HISTORY_FILE):
                with open(self.VOLATILITY_HISTORY_FILE, 'r') as f:
                    data = json.load(f)
                    self.historical_volatility = data.get('history', [])

                if len(self.historical_volatility) > 0:
                    print(f"✅ Loaded {len(self.historical_volatility)} historical period averages")
                else:
                    print("⚠️  No historical volatility data found")
            else:
                print("⚠️  Volatility history file not found - will create on first period")

        except Exception as e:
            print(f"⚠️  Error loading volatility history: {e}")

    def _save_volatility_history(self):
        """Save volatility history to file"""
        try:
            # Keep only last 100 periods (to avoid file growing indefinitely)
            if len(self.historical_volatility) > 100:
                self.historical_volatility = self.historical_volatility[-100:]

            data = {
                'history': self.historical_volatility,
                'last_updated': datetime.now().isoformat()
            }

            with open(self.VOLATILITY_HISTORY_FILE, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"⚠️  Error saving volatility history: {e}")

    def read_market_data(self) -> Optional[Dict]:
        """Read BTC CALL market data from JSON file"""
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

    def calculate_volatility(self) -> float:
        """
        Calculate volatility as standard deviation of price history

        Returns:
            Standard deviation of prices, or 0 if insufficient data
        """
        if len(self.price_history) < 10:  # Need at least 10 data points
            return 0.0

        try:
            # Calculate standard deviation
            volatility = statistics.stdev(self.price_history)
            return volatility
        except Exception as e:
            print(f"⚠️  Error calculating volatility: {e}")
            return 0.0

    def update_price_history(self, mid_price: float):
        """Add new price to history and update volatility"""
        self.price_history.append(mid_price)
        self.current_volatility = self.calculate_volatility()

        # Track volatility throughout the period (if we have valid volatility)
        if self.current_volatility > 0:
            self.period_volatilities.append(self.current_volatility)
            # Calculate running average for the period
            self.period_average_volatility = statistics.mean(self.period_volatilities)

    def get_usdc_balance(self) -> float:
        """Get USDC balance (ERC-20 token)"""
        try:
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

            response = self.trader.client.get_balance_allowance(
                params=BalanceAllowanceParams(
                    asset_type=AssetType.COLLATERAL
                )
            )

            balance_raw = int(response.get('balance', 0))
            balance_usdc = balance_raw / 10**6  # USDC has 6 decimals

            return balance_usdc

        except Exception as e:
            print(f"❌ Error getting USDC balance: {e}")
            return 0.0

    def check_token_balance(self, token_id: str) -> float:
        """
        Check balance of a specific token

        Args:
            token_id: Token ID to check

        Returns:
            Balance as float
        """
        try:
            balance_raw, balance = self.trader.get_token_balance(token_id)
            return balance
        except Exception as e:
            print(f"❌ Error checking balance for {token_id[:12]}...: {e}")
            return 0.0

    def execute_buy(self, token_id: str, ask_price: float) -> bool:
        """
        Execute buy order for BTC_CALL with multi-phase verification

        Phase 1: Place order
        Phase 2: Verify at 1s, 5s, 10s, 20s intervals
        Phase 3: Cancel and retry if nothing happened

        Args:
            token_id: Token ID to buy
            ask_price: Current ask price

        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"\n{'='*70}")
            print(f"🛒 EXECUTING BUY ORDER")
            print(f"{'='*70}")
            print(f"📊 Asset: BTC_CALL")
            print(f"📦 Size: {self.POSITION_SIZE} shares")
            print(f"💰 Ask Price: ${ask_price:.4f}")

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

                # Update position tracking
                self.current_position['size'] = final_balance
                self.current_position['avg_buy_price'] = ask_price
                self.current_position['token_id'] = token_id

                print(f"\n📊 POSITION OPENED")
                print(f"   Size: {final_balance:.2f} shares")
                print(f"   Avg Buy Price: ${ask_price:.4f}")
                print(f"   Take Profit Target: ${ask_price * (1 + self.TAKE_PROFIT_PERCENT):.4f} ({self.TAKE_PROFIT_PERCENT*100}% profit)")
                print(f"   Stop Loss Target: ${ask_price * (1 - self.STOP_LOSS_PERCENT):.4f} ({self.STOP_LOSS_PERCENT*100}% loss)")

                return True

            # Case 2: Position not confirmed and wallet unchanged
            elif not position_confirmed and usdc_spent < 0.5:
                print("\n⚠️  Position not confirmed and wallet unchanged")
                print("   🔄 Cancelling all orders...")
                self.trader.cancel_all_orders()
                time.sleep(2)

                # Check if conditions still valid
                data = self.read_market_data()
                if data and self.check_entry_conditions(data):
                    print("   ♻️  Conditions still valid - retrying in next cycle")
                else:
                    print("   ❌ Conditions no longer valid - ABORTING")

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
        Execute sell order for BTC_CALL

        Args:
            token_id: Token ID to sell
            bid_price: Current bid price
            reason: Reason for selling (take-profit or stop-loss)

        Returns:
            True if successful, False otherwise
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
                    'token_id': None
                }
                return False

            # Use actual balance for selling
            size = actual_balance

            print(f"\n{'='*60}")
            print(f"💰 EXECUTING SELL ORDER - {reason}")
            print(f"{'='*60}")
            print(f"📊 Asset: BTC_CALL")
            print(f"📦 Size: {size:.2f} shares (verified in wallet)")
            print(f"💰 Current Bid: ${bid_price:.4f}")
            print(f"📊 Buy Price: ${self.current_position['avg_buy_price']:.4f}")

            # Calculate P&L
            pnl = (bid_price - self.current_position['avg_buy_price']) * size
            pnl_percent = ((bid_price / self.current_position['avg_buy_price']) - 1) * 100

            print(f"📈 P&L: ${pnl:.2f} ({pnl_percent:+.2f}%)")

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
                    'token_id': None
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

    def check_exit_conditions(self, bid_price: float) -> Optional[str]:
        """
        Check if exit conditions are met

        Args:
            bid_price: Current bid price

        Returns:
            Reason for exit ('take-profit' or 'stop-loss'), or None if no exit
        """
        if self.current_position['size'] < 0.5:
            return None

        avg_buy = self.current_position['avg_buy_price']

        # Take profit: 5% profit
        take_profit_price = avg_buy * (1 + self.TAKE_PROFIT_PERCENT)
        if bid_price >= take_profit_price:
            return 'TAKE-PROFIT (5% profit)'

        # Stop loss: 10% loss
        stop_loss_price = avg_buy * (1 - self.STOP_LOSS_PERCENT)
        if bid_price <= stop_loss_price:
            return 'STOP-LOSS (10% loss)'

        return None

    def monitor_position(self, data: Dict):
        """Monitor open position and check exit conditions"""
        if self.current_position['size'] < 0.5:
            return

        # SAFETY CHECK: Verify we actually have the tokens before attempting to sell
        token_id = self.current_position['token_id']
        actual_balance = self.check_token_balance(token_id)

        if actual_balance < 0.5:
            print(f"\n⚠️  WARNING: Position tracking shows {self.current_position['size']:.2f} but wallet shows {actual_balance:.2f}")
            print(f"   Resetting position tracking")
            self.current_position = {
                'size': 0.0,
                'avg_buy_price': 0.0,
                'token_id': None
            }
            return

        # Update position size if there's a discrepancy
        if abs(actual_balance - self.current_position['size']) > 0.5:
            print(f"\n⚠️  Position size mismatch: tracked {self.current_position['size']:.2f}, actual {actual_balance:.2f}")
            print(f"   Updating to actual balance")
            self.current_position['size'] = actual_balance

        bid_price = float(data['best_bid'].get('price', 0))

        if bid_price == 0:
            print("⚠️  Invalid bid price (0) - skipping exit check")
            return

        # Check exit conditions
        exit_reason = self.check_exit_conditions(bid_price)

        if exit_reason:
            print(f"\n🚨 EXIT CONDITION MET: {exit_reason}")
            self.execute_sell(token_id, bid_price, exit_reason)

    def check_entry_conditions(self, data: Dict) -> bool:
        """
        Check if entry conditions are met

        Returns:
            True if we should enter a trade
        """
        # Must have 0 position
        if self.current_position['size'] >= 0.5:
            return False

        # Check volatility - must be both above period average AND above minimum threshold
        # Need at least some data points to calculate meaningful average
        if len(self.period_volatilities) < 10:
            return False

        if self.current_volatility <= self.period_average_volatility:
            return False

        if self.current_volatility < self.MIN_VOLATILITY_THRESHOLD:
            return False

        # Check balance
        ask_price = float(data['best_ask'].get('price', 0))
        if ask_price == 0:
            return False

        required = ask_price * self.POSITION_SIZE
        usdc_balance = self.get_usdc_balance()

        if usdc_balance < required:
            return False

        return True

    def run_trading_cycle(self):
        """Execute one trading cycle"""
        try:
            # Read market data
            data = self.read_market_data()

            if not data:
                print("⚠️  Could not read market data")
                return

            # Extract prices
            ask_price = float(data['best_ask'].get('price', 0))
            bid_price = float(data['best_bid'].get('price', 0))

            if ask_price == 0 or bid_price == 0:
                print("⚠️  Invalid prices - skipping cycle")
                return

            # Calculate mid price and update volatility
            mid_price = (ask_price + bid_price) / 2
            self.update_price_history(mid_price)

            # Calculate time to expiry
            time_left = self.minutes_to_expiry()

            # Display status
            print(f"\n{'='*80}")
            print(f"⏰ Time to Expiry: {time_left:.2f} minutes")
            print(f"💰 Ask: ${ask_price:.4f} | Bid: ${bid_price:.4f} | Mid: ${mid_price:.4f}")
            print(f"🌊 VOLATILITY: {self.current_volatility:.6f} | Period Avg: {self.period_average_volatility:.6f} | Min: {self.MIN_VOLATILITY_THRESHOLD}", end="")

            # Highlight volatility status
            above_avg = self.current_volatility > self.period_average_volatility
            above_min = self.current_volatility >= self.MIN_VOLATILITY_THRESHOLD

            if above_avg and above_min:
                ratio = self.current_volatility / self.period_average_volatility if self.period_average_volatility > 0 else 0
                print(f" | ✅ HIGH ({ratio:.2f}x avg, above min)")
            elif above_min and not above_avg:
                print(f" | ⚠️  Above min but below period average")
            elif above_avg and not above_min:
                print(f" | ⚠️  Above period average but below min threshold")
            else:
                print(f" | ⚠️  LOW (below both)")

            print(f"📊 Price History: {len(self.price_history)} data points | Period Vol Samples: {len(self.period_volatilities)}")

            # Display position status
            if self.current_position['size'] >= 0.5:
                avg_buy = self.current_position['avg_buy_price']
                current_value = bid_price * self.current_position['size']
                cost = avg_buy * self.current_position['size']
                pnl = current_value - cost
                pnl_percent = ((bid_price / avg_buy) - 1) * 100

                take_profit_price = avg_buy * (1 + self.TAKE_PROFIT_PERCENT)
                stop_loss_price = avg_buy * (1 - self.STOP_LOSS_PERCENT)

                print(f"\n📊 OPEN POSITION:")
                print(f"   Size: {self.current_position['size']:.2f} shares")
                print(f"   Avg Buy: ${avg_buy:.4f}")
                print(f"   Current Bid: ${bid_price:.4f}")
                print(f"   P&L: ${pnl:.2f} ({pnl_percent:+.2f}%)")
                print(f"   Take Profit: ${take_profit_price:.4f} (need {((take_profit_price/bid_price)-1)*100:+.2f}% move)")
                print(f"   Stop Loss: ${stop_loss_price:.4f} (need {((stop_loss_price/bid_price)-1)*100:+.2f}% move)")

                # Monitor for exit
                self.monitor_position(data)

            else:
                print(f"\n📊 No open position")
                print(f"💰 USDC Balance: ${self.get_usdc_balance():.2f}")

                # Check entry conditions
                if self.check_entry_conditions(data):
                    print(f"\n✅ ENTRY CONDITIONS MET!")
                    print(f"   Volatility: {self.current_volatility:.6f} > Period Avg: {self.period_average_volatility:.6f}")
                    print(f"   Volatility: {self.current_volatility:.6f} >= Min: {self.MIN_VOLATILITY_THRESHOLD}")
                    self.execute_buy(data['asset_id'], ask_price)
                else:
                    if len(self.period_volatilities) < 10:
                        print(f"\n⏸️  Waiting for more data (have {len(self.period_volatilities)} volatility samples, need 10)")
                    elif self.current_volatility < self.MIN_VOLATILITY_THRESHOLD:
                        print(f"\n⏸️  Waiting for higher volatility (current: {self.current_volatility:.6f} < min: {self.MIN_VOLATILITY_THRESHOLD})")
                    elif self.current_volatility <= self.period_average_volatility:
                        print(f"\n⏸️  Waiting for higher volatility (current: {self.current_volatility:.6f} <= period avg: {self.period_average_volatility:.6f})")

            # Check for new period (reset volatility tracking)
            if time_left > 14.5:
                print("\n🔄 NEW PERIOD DETECTED")

                # Save average volatility of completed period if we have data
                if len(self.period_volatilities) > 100:  # Only save if we have substantial data
                    period_avg = self.period_average_volatility
                    self.historical_volatility.append(period_avg)
                    print(f"   Saved period average volatility: {period_avg:.6f}")
                    print(f"   Total historical periods: {len(self.historical_volatility)}")

                    # Save to file
                    self._save_volatility_history()

                # Clear tracking for new period
                self.price_history.clear()
                self.period_volatilities.clear()
                self.current_volatility = 0.0
                self.period_average_volatility = 0.0
                print(f"   Price history and period volatilities cleared for new period")

        except Exception as e:
            print(f"❌ Error in trading cycle: {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        """Main trading loop - check every second"""
        print("\n🚀 Starting BTC_CALL volatility-based trader...")
        print(f"📊 Asset: BTC_CALL only")
        print(f"🌊 Strategy: Volatility must be > period average AND >= {self.MIN_VOLATILITY_THRESHOLD}")
        print(f"📊 Period average calculated from beginning of each 15-min period")
        print(f"📦 Position: {self.POSITION_SIZE} shares")
        print(f"📈 Take Profit: {self.TAKE_PROFIT_PERCENT*100}% profit")
        print(f"📉 Stop Loss: {self.STOP_LOSS_PERCENT*100}% loss")
        print(f"⚡ Monitoring: Every second")
        print("\n" + "=" * 80 + "\n")

        cycle_count = 0

        try:
            while True:
                cycle_count += 1

                print(f"\n{'='*80}")
                print(f"🔄 CYCLE #{cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*80}")

                self.run_trading_cycle()

                print(f"\n⚡ Next check in 1 second...")
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n\n🛑 Bot stopped by user")
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point"""
    print("\n" + "="*80)
    print("🎯 BTC_CALL VOLATILITY-BASED TRADER")
    print("="*80)

    # Load credentials
    try:
        env_path = '/home/ubuntu/013_2025_polymarket/keys/keys_ovh39.env'
        creds = load_credentials_from_env(env_path)
        print(f"✅ Credentials loaded from {env_path}")
    except Exception as e:
        print(f"❌ Error loading credentials: {e}")
        return

    # Initialize and run bot
    try:
        bot = BTCCallVolatilityTrader(creds)
        bot.run()
    except Exception as e:
        print(f"❌ Error initializing bot: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
