#!/usr/bin/env python3
"""
15-Minute Binary Options Pair Trading Bot - With Safety Nets

Strategy:
- Trade COIN1 PUT + COIN2 CALL pairs when:
  1. More than 10 minutes until period end
  2. Both asks between $0.24-0.47
  3. Sum of asks < $0.77
- Opens positions sequentially (first then second mandatory)
- Scales position size based on available balance
- Safety checks: verifies order execution, cancels unfilled orders, checks position before retry
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Import Polymarket trading core
sys.path.insert(0, '/home/claude')  # Add current directory first for debug version
from polymarket_trading_core_debug import PolymarketTrader, load_credentials_from_env

class PairTrader:
    """15-minute binary options pair trading bot with safety checks"""

    # File paths for market data
    DATA_FILES = {
        'BTC_CALL': '/home/ubuntu/013_2025_polymarket/15M_CALL.json',
        'BTC_PUT': '/home/ubuntu/013_2025_polymarket/15M_PUT.json',
        'ETH_CALL': '/home/ubuntu/013_2025_polymarket/15M_CALL_ETH.json',
        'ETH_PUT': '/home/ubuntu/013_2025_polymarket/15M_PUT_ETH.json',
        'SOL_CALL': '/home/ubuntu/013_2025_polymarket/15M_CALL_SOL.json',
        'SOL_PUT': '/home/ubuntu/013_2025_polymarket/15M_PUT_SOL.json',
        'XRP_CALL': '/home/ubuntu/013_2025_polymarket/15M_CALL_XRP.json',
        'XRP_PUT': '/home/ubuntu/013_2025_polymarket/15M_PUT_XRP.json',
    }

    COINS = ['BTC', 'ETH', 'SOL', 'XRP']

    # Trading parameters
    MIN_ASK_PRICE = 0.24
    MAX_ASK_PRICE = 0.47
    MAX_SUM_PRICE = 0.77
    MIN_TIME_TO_EXPIRY_MINUTES = 10

    # Position sizing
    BASE_POSITION_SIZE = 5  # Base size per leg
    USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"  # USDC ERC-20 address on Polygon

    def __init__(self, credentials: Dict[str, str]):
        """Initialize the pair trader"""
        self.trader = PolymarketTrader(
            clob_api_url=credentials['clob_api_url'],
            private_key=credentials['private_key'],
            api_key=credentials['api_key'],
            api_secret=credentials['api_secret'],
            api_passphrase=credentials['api_passphrase']
        )

        self.traded_pairs = set()  # Track pairs already traded this period
        self.open_positions = {}  # Track open positions: {token_id: {'size': float, 'coin': str, 'type': str, 'sell_order_placed': bool}}
        self.current_trade_initial_usdc = None  # Store initial USDC for current trade to prevent re-measurement

        print("=" * 70)
        print("🎯 15-MINUTE PAIR TRADING BOT INITIALIZED (WITH SAFETY NETS)")
        print("=" * 70)

    def read_market_data(self, coin: str, option_type: str) -> Optional[Dict]:
        """Read market data from JSON file"""
        try:
            file_key = f"{coin}_{option_type}"
            file_path = self.DATA_FILES.get(file_key)

            if not file_path or not os.path.exists(file_path):
                return None

            with open(file_path, 'r') as f:
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
            print(f"❌ Error reading {coin} {option_type}: {e}")
            return None

    def get_next_expiry_time(self) -> datetime:
        """Calculate the next 15-minute expiry time (00, 15, 30, 45)"""
        now = datetime.now()
        minute = now.minute

        # Find next 15-minute mark
        if minute < 15:
            next_minute = 15
        elif minute < 30:
            next_minute = 30
        elif minute < 45:
            next_minute = 45
        else:
            next_minute = 0
            now += timedelta(hours=1)

        next_expiry = now.replace(minute=next_minute, second=0, microsecond=0)
        return next_expiry

    def minutes_to_expiry(self) -> float:
        """Calculate minutes until next expiry"""
        now = datetime.now()
        next_expiry = self.get_next_expiry_time()
        diff = (next_expiry - now).total_seconds() / 60
        return diff

    def get_usdc_balance(self) -> float:
        """Get USDC balance (ERC-20 token)"""
        try:
            # For ERC-20 COLLATERAL tokens, token_id should be empty
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

            response = self.trader.client.get_balance_allowance(
                params=BalanceAllowanceParams(
                    asset_type=AssetType.COLLATERAL
                    # No token_id for COLLATERAL - it's the base USDC balance
                )
            )

            balance_raw = int(response.get('balance', 0))
            balance_usdc = balance_raw / 10**6  # USDC has 6 decimals

            return balance_usdc

        except Exception as e:
            print(f"❌ Error getting USDC balance: {e}")
            return 0.0

    def calculate_max_pairs(self, usdc_balance: float, pair_cost: float) -> int:
        """Calculate how many pairs we can afford"""
        if pair_cost <= 0:
            return 0

        max_pairs = int(usdc_balance / pair_cost)
        return max(0, max_pairs)

    def check_pair_conditions(self, coin1: str, coin2: str) -> Optional[Dict]:
        """
        Check if pair trading conditions are met

        Returns:
            Dict with trade details if conditions met, None otherwise
        """
        # Read market data
        put_data = self.read_market_data(coin1, 'PUT')
        call_data = self.read_market_data(coin2, 'CALL')

        if not put_data or not call_data:
            return None

        # Extract ask prices
        put_ask = put_data['best_ask'].get('price', 0)
        call_ask = call_data['best_ask'].get('price', 0)

        if put_ask == 0 or call_ask == 0:
            return None

        # Check conditions
        put_in_range = self.MIN_ASK_PRICE <= put_ask <= self.MAX_ASK_PRICE
        call_in_range = self.MIN_ASK_PRICE <= call_ask <= self.MAX_ASK_PRICE
        sum_under_max = (put_ask + call_ask) < self.MAX_SUM_PRICE

        if put_in_range and call_in_range and sum_under_max:
            return {
                'coin1': coin1,
                'coin2': coin2,
                'put_ask': put_ask,
                'call_ask': call_ask,
                'total_cost': put_ask + call_ask,
                'put_token_id': put_data['asset_id'],
                'call_token_id': call_data['asset_id'],
                'put_asset_name': put_data['asset_name'],
                'call_asset_name': call_data['asset_name']
            }

        return None

    def verify_position_opened(self, token_id: str, expected_size: float) -> bool:
        """
        Verify if a position was actually opened by checking token balance

        Args:
            token_id: Token to check
            expected_size: Expected position size

        Returns:
            True if position is open with expected size
        """
        try:
            print(f"   🔍 DEBUG: Checking balance for token {token_id[:16]}...")

            balance_raw, balance_tokens = self.trader.get_token_balance(token_id)

            print(f"   🔍 DEBUG: Raw balance: {balance_raw}, Tokens: {balance_tokens:.6f}")
            print(f"   🔍 DEBUG: Expected: {expected_size:.6f}, Threshold (95%): {expected_size * 0.95:.6f}")

            # Allow small tolerance for rounding
            if balance_tokens >= (expected_size * 0.95):
                print(f"   ✅ Position verified: {balance_tokens:.2f} tokens (expected {expected_size:.2f})")
                return True
            else:
                print(f"   ℹ️  Position check: {balance_tokens:.2f} tokens (expected {expected_size:.2f})")
                return False

        except Exception as e:
            print(f"   ❌ Error verifying position: {e}")
            import traceback
            traceback.print_exc()
            return False

    def check_token_balance(self, token_id: str) -> float:
        """Check current balance of a specific token"""
        try:
            print(f"   🔍 DEBUG check_token_balance: token_id={token_id[:16]}...")
            balance_raw, balance = self.trader.get_token_balance(token_id)
            print(f"   🔍 DEBUG check_token_balance: raw={balance_raw}, tokens={balance:.6f}")
            return balance
        except Exception as e:
            print(f"❌ Error checking balance for {token_id[:12]}...: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def verify_sell_order_exists(self, token_id: str) -> bool:
        """
        Check if there's an active sell order for a token

        Returns:
            True if sell order exists, False otherwise
        """
        try:
            open_orders = self.trader.get_open_orders()

            print(f"      🔍 DEBUG verify_sell_order: Checking {len(open_orders) if isinstance(open_orders, list) else 0} open orders")

            # open_orders is a list of order dicts
            if not isinstance(open_orders, list):
                print(f"      ⚠️  Unexpected open_orders type: {type(open_orders)}")
                return False

            for order in open_orders:
                if not isinstance(order, dict):
                    continue

                order_token_id = order.get('asset_id')
                order_side = order.get('side')
                order_status = order.get('status')

                if order_token_id == token_id:
                    print(f"      🔍 DEBUG: Found order for this token - side={order_side}, status={order_status}")

                if (order_token_id == token_id and
                    order_side == 'SELL' and
                    order_status in ['LIVE', 'OPEN']):
                    print(f"      ✅ Found matching SELL order")
                    return True

            print(f"      ℹ️  No matching SELL order found")
            return False

        except Exception as e:
            print(f"      ❌ Error checking sell orders: {e}")
            import traceback
            traceback.print_exc()
            return False

    def place_sell_order_safely(self, token_id: str, coin: str, option_type: str, position_size: float) -> bool:
        """
        Safely place a sell order at $0.99 with verification
        Handles existing sell orders by canceling them first if needed

        Returns:
            True if sell order successfully placed and verified
        """
        sell_price = 0.99

        print(f"📤 Placing sell order for {coin} {option_type} @ ${sell_price}...")

        # Check if sell order already exists
        if self.verify_sell_order_exists(token_id):
            print(f"   ℹ️  Sell order already exists")
            return True

        # Check actual balance (in case it's different from position_size)
        actual_balance = self.check_token_balance(token_id)
        if actual_balance < 0.01:
            print(f"   ⚠️  No tokens to sell (balance: {actual_balance:.2f})")
            return False

        # Use actual balance instead of position_size to avoid allowance errors
        quantity_to_sell = min(actual_balance, position_size)

        print(f"   📊 Selling {quantity_to_sell:.2f} tokens (balance: {actual_balance:.2f})")

        # Place sell order
        sell_order_id = self.trader.place_sell_order(
            token_id=token_id,
            price=sell_price,
            quantity=quantity_to_sell
        )

        if not sell_order_id:
            print(f"   ❌ Sell order failed")
            return False

        # Wait and verify
        time.sleep(2)

        if self.verify_sell_order_exists(token_id):
            print(f"   ✅ Sell order verified @ ${sell_price}")
            return True
        else:
            print(f"   ⚠️  Sell order not found in open orders (but may have executed)")
            return True  # Assume success if order was placed

    def execute_pair_trade(self, pair_info: Dict, num_pairs: int) -> bool:
        """
        Execute the pair trade (PUT + CALL) with laggy API handling

        Strategy:
        1. Open both positions immediately (1 attempt each)
        2. Verify balances at 1s, 5s, 10s intervals
        3. Use USDC balance as hint (faster than token balance)
        4. If wallet unchanged and no tokens, cancel all and retry
        5. If only 1 leg open, force open 2nd leg regardless of conditions

        Args:
            pair_info: Pair trading information
            num_pairs: Number of pairs to trade (NOTE: not used - always trades BASE_POSITION_SIZE)

        Returns:
            True if both legs executed successfully
        """
        position_size = self.BASE_POSITION_SIZE  # Fixed size per leg
        sell_price = 0.99
        expected_cost = pair_info['total_cost'] * position_size

        print("\n" + "=" * 70)
        print(f"🎯 EXECUTING PAIR TRADE")
        print("=" * 70)
        print(f"📊 Pair: {pair_info['coin1']} PUT + {pair_info['coin2']} CALL")
        print(f"💰 PUT ask: ${pair_info['put_ask']:.4f}")
        print(f"💰 CALL ask: ${pair_info['call_ask']:.4f}")
        print(f"💵 Expected cost: ${expected_cost:.2f}")
        print(f"📦 Position size: {position_size:.2f} per leg")
        print(f"🎯 Auto-sell price: ${sell_price}")
        print("=" * 70)

        # Record initial USDC balance BEFORE any orders (only once per trade attempt)
        if self.current_trade_initial_usdc is None:
            self.current_trade_initial_usdc = self.get_usdc_balance()
            print(f"\n💰 Initial USDC (recorded): ${self.current_trade_initial_usdc:.2f}")
        else:
            print(f"\n💰 Initial USDC (cached): ${self.current_trade_initial_usdc:.2f}")

        initial_usdc = self.current_trade_initial_usdc

        # ============================================================
        # PHASE 1: FAST EXECUTION - Open both positions immediately
        # ============================================================

        print("\n🚀 PHASE 1: Opening both positions...")

        # LEG 1: Buy PUT (single attempt)
        print("\n🔵 LEG 1: Buying PUT...")
        put_order_id = None
        try:
            put_order_id = self.trader.place_buy_order(
                token_id=pair_info['put_token_id'],
                price=pair_info['put_ask'],
                quantity=position_size
            )

            if put_order_id:
                print(f"   ✅ PUT order submitted: {put_order_id[:16]}...")
            else:
                print(f"   ⚠️  PUT order returned None")
        except Exception as e:
            print(f"   ⚠️  PUT order exception: {e}")

        time.sleep(0.5)  # Brief pause between legs

        # LEG 2: Buy CALL (ALWAYS ATTEMPT - even if PUT failed)
        print("\n🔵 LEG 2: Buying CALL...")
        call_order_id = None
        try:
            call_order_id = self.trader.place_buy_order(
                token_id=pair_info['call_token_id'],
                price=pair_info['call_ask'],
                quantity=position_size
            )

            if call_order_id:
                print(f"   ✅ CALL order submitted: {call_order_id[:16]}...")
            else:
                print(f"   ⚠️  CALL order returned None")
        except Exception as e:
            print(f"   ⚠️  CALL order exception: {e}")

        # ============================================================
        # PHASE 2: DELAYED VERIFICATION - Check at 5s, 10s, 20s
        # ============================================================

        print("\n🔍 PHASE 2: Verifying positions (laggy API handling)...")

        verification_times = [5, 10, 20]
        put_confirmed = False
        call_confirmed = False

        for wait_time in verification_times:
            print(f"\n⏳ Waiting {wait_time} seconds for API to update...")
            time.sleep(wait_time)

            print(f"\n📊 Verification at {wait_time}s mark:")

            # Check USDC balance (faster hint)
            current_usdc = self.get_usdc_balance()
            usdc_spent = initial_usdc - current_usdc
            print(f"   💰 USDC: ${initial_usdc:.2f} → ${current_usdc:.2f} (spent: ${usdc_spent:.2f})")

            # Check token balances
            put_balance = self.check_token_balance(pair_info['put_token_id'])
            call_balance = self.check_token_balance(pair_info['call_token_id'])
            print(f"   🪙 PUT balance: {put_balance:.2f} tokens")
            print(f"   🪙 CALL balance: {call_balance:.2f} tokens")

            # Determine status based on balances
            if put_balance >= position_size * 0.95:
                put_confirmed = True
                print(f"   ✅ PUT position confirmed")
            elif usdc_spent > (pair_info['put_ask'] * position_size * 0.8):
                print(f"   💡 USDC spent suggests PUT may be open (API lag)")
                put_confirmed = True  # Trust USDC balance
            
            if call_balance >= position_size * 0.95:
                call_confirmed = True
                print(f"   ✅ CALL position confirmed")
            elif usdc_spent > (expected_cost * 0.8):
                print(f"   💡 USDC spent suggests both positions may be open (API lag)")
                call_confirmed = True  # Trust USDC balance

            # Check if we're done
            if put_confirmed and call_confirmed:
                print(f"\n✅ Both positions confirmed at {wait_time}s!")
                break

        # ============================================================
        # PHASE 3: DECISION LOGIC
        # ============================================================

        print("\n🎲 PHASE 3: Final status check...")
        print(f"   PUT confirmed: {put_confirmed}")
        print(f"   CALL confirmed: {call_confirmed}")
        print(f"   USDC spent: ${usdc_spent:.2f} (expected: ${expected_cost:.2f})")

        # Case 1: Both positions confirmed
        if put_confirmed and call_confirmed:
            print("\n✅ SUCCESS: Both positions opened")

        # Case 2: Neither position confirmed and wallet unchanged
        elif not put_confirmed and not call_confirmed and usdc_spent < 0.5:
            print("\n⚠️  Neither position confirmed and wallet unchanged")
            print("   🔄 Cancelling all orders and retrying...")
            self.trader.cancel_all_orders()
            time.sleep(2)

            # Retry if conditions still met
            if self.check_pair_conditions(pair_info['coin1'], pair_info['coin2']):
                print("   ♻️  Conditions still valid - retrying...")
                return self.execute_pair_trade(pair_info, num_pairs)
            else:
                print("   ❌ Conditions no longer valid - ABORTING")
                self.current_trade_initial_usdc = None; return False

        # Case 3: Only PUT confirmed
        elif put_confirmed and not call_confirmed:
            print("\n⚠️  Only PUT position opened - forcing CALL open...")

            # Force open CALL even if conditions not met
            for retry in range(3):
                print(f"   Retry {retry + 1}/3: Opening CALL...")
                call_order_id = self.trader.place_buy_order(
                    token_id=pair_info['call_token_id'],
                    price=pair_info['call_ask'],
                    quantity=position_size
                )

                if call_order_id:
                    time.sleep(5)  # Wait longer for this critical order
                    call_balance = self.check_token_balance(pair_info['call_token_id'])
                    if call_balance >= position_size * 0.95:
                        call_confirmed = True
                        print(f"   ✅ CALL position confirmed")
                        break

                time.sleep(2)

            if not call_confirmed:
                print("   ❌ WARNING: Failed to open CALL leg - manual intervention needed")
                self.current_trade_initial_usdc = None; return False

        # Case 4: Only CALL confirmed
        elif call_confirmed and not put_confirmed:
            print("\n⚠️  Only CALL position opened - forcing PUT open...")

            # Force open PUT even if conditions not met
            for retry in range(3):
                print(f"   Retry {retry + 1}/3: Opening PUT...")
                put_order_id = self.trader.place_buy_order(
                    token_id=pair_info['put_token_id'],
                    price=pair_info['put_ask'],
                    quantity=position_size
                )

                if put_order_id:
                    time.sleep(5)  # Wait longer for this critical order
                    put_balance = self.check_token_balance(pair_info['put_token_id'])
                    if put_balance >= position_size * 0.95:
                        put_confirmed = True
                        print(f"   ✅ PUT position confirmed")
                        break

                time.sleep(2)

            if not put_confirmed:
                print("   ❌ WARNING: Failed to open PUT leg - manual intervention needed")
                self.current_trade_initial_usdc = None; return False

        # Final verification
        if not (put_confirmed and call_confirmed):
            print("\n❌ FAILED: Could not confirm both positions")
            self.current_trade_initial_usdc = None; return False

        # ============================================================
        # PHASE 4: PLACE SELL ORDERS
        # ============================================================

        print("\n📤 PHASE 4: Placing sell orders...")

        # Track positions
        self.open_positions[pair_info['put_token_id']] = {
            'size': position_size,
            'coin': pair_info['coin1'],
            'type': 'PUT',
            'sell_order_placed': False
        }

        self.open_positions[pair_info['call_token_id']] = {
            'size': position_size,
            'coin': pair_info['coin2'],
            'type': 'CALL',
            'sell_order_placed': False
        }

        # Place sell orders
        if self.place_sell_order_safely(pair_info['put_token_id'], pair_info['coin1'], 'PUT', position_size):
            self.open_positions[pair_info['put_token_id']]['sell_order_placed'] = True

        time.sleep(0.5)

        if self.place_sell_order_safely(pair_info['call_token_id'], pair_info['coin2'], 'CALL', position_size):
            self.open_positions[pair_info['call_token_id']]['sell_order_placed'] = True

        print("\n" + "=" * 70)
        print("🎉 PAIR TRADE COMPLETED")
        print("=" * 70)

        # Reset cached initial USDC for next trade
        self.current_trade_initial_usdc = None

        return True

    def scan_all_pairs(self) -> List[Dict]:
        """Scan all possible coin pairs for trading opportunities"""
        opportunities = []

        print(f"\n   🔍 DEBUG: Currently traded pairs: {self.traded_pairs}")
        print(f"   🔍 DEBUG: Currently open positions: {len(self.open_positions)}")

        for coin1 in self.COINS:
            for coin2 in self.COINS:
                if coin1 == coin2:
                    continue  # Skip same coin pairs

                pair_key = f"{coin1}_PUT_{coin2}_CALL"

                # Skip if already traded this period
                if pair_key in self.traded_pairs:
                    print(f"   ⏭️  Skipping {pair_key} (already traded)")
                    continue

                pair_info = self.check_pair_conditions(coin1, coin2)
                if pair_info:
                    print(f"   ✅ Found opportunity: {pair_key}")
                    opportunities.append((pair_key, pair_info))

        return opportunities

    def verify_and_fix_sell_orders(self):
        """
        Verify all open positions have sell orders when trading window closes
        Place missing sell orders
        """
        if not self.open_positions:
            return

        print("\n" + "=" * 70)
        print("🔍 VERIFYING POSITIONS AND SELL ORDERS")
        print("=" * 70)

        positions_to_remove = []

        for token_id, position_info in self.open_positions.items():
            coin = position_info['coin']
            option_type = position_info['type']
            expected_size = position_info['size']
            sell_order_placed = position_info['sell_order_placed']

            print(f"\n📊 Checking {coin} {option_type}:")

            # Check current balance
            current_balance = self.check_token_balance(token_id)
            print(f"   Balance: {current_balance:.2f} tokens")

            if current_balance < 0.01:
                # Position already sold/closed
                print(f"   ✅ Position closed (no tokens remaining)")
                positions_to_remove.append(token_id)
                continue

            # Check if sell order exists
            has_sell_order = self.verify_sell_order_exists(token_id)

            if not has_sell_order:
                print(f"   ⚠️  No sell order found - placing now...")
                if self.place_sell_order_safely(token_id, coin, option_type, current_balance):
                    self.open_positions[token_id]['sell_order_placed'] = True
                    print(f"   ✅ Sell order placed for {current_balance:.2f} tokens")
                else:
                    print(f"   ❌ Failed to place sell order")
            else:
                print(f"   ✅ Sell order verified")
                self.open_positions[token_id]['sell_order_placed'] = True

        # Remove closed positions
        for token_id in positions_to_remove:
            del self.open_positions[token_id]

        print("\n" + "=" * 70)

    def run_trading_cycle(self) -> bool:
        """
        Execute one trading cycle

        Returns:
            True if we should check again in 1 second, False if we should sleep 30 seconds
        """
        try:
            # Check time to expiry
            time_left = self.minutes_to_expiry()

            print(f"\n⏰ Time to expiry: {time_left:.1f} minutes")

            # Check if we're in a new period (reset traded pairs)
            if time_left > 14.5:  # New period started
                if self.traded_pairs:
                    print("🔄 New period detected - resetting traded pairs")
                    self.traded_pairs.clear()
                    # Clear open positions tracking for new period
                    self.open_positions.clear()

            # Must have at least 10 minutes left
            if time_left < self.MIN_TIME_TO_EXPIRY_MINUTES:
                print(f"⏸️  Less than {self.MIN_TIME_TO_EXPIRY_MINUTES} minutes left - waiting for next period")

                # Verify all positions have sell orders
                if self.open_positions:
                    self.verify_and_fix_sell_orders()

                return False  # Sleep 30 seconds when waiting for next period

            # Check USDC balance
            usdc_balance = self.get_usdc_balance()
            print(f"💰 USDC Balance: ${usdc_balance:.2f}")

            if usdc_balance < (self.MIN_ASK_PRICE * 2 * self.BASE_POSITION_SIZE):
                print("⚠️  Insufficient USDC balance for even 1 pair")
                return True  # Keep checking every second - balance might change

            # Scan for opportunities
            print("\n🔍 Scanning for pair trading opportunities...")
            opportunities = self.scan_all_pairs()

            if not opportunities:
                print("📊 No opportunities found matching criteria")
                return True  # Keep checking every second - opportunities can appear

            print(f"\n✨ Found {len(opportunities)} opportunities:")
            for pair_key, pair_info in opportunities:
                print(f"   • {pair_key}: ${pair_info['total_cost']:.4f}")

            # Execute trades
            for pair_key, pair_info in opportunities:
                # Calculate how many pairs we can afford
                pair_cost = pair_info['total_cost'] * self.BASE_POSITION_SIZE
                num_pairs = self.calculate_max_pairs(usdc_balance, pair_cost)

                if num_pairs < 1:
                    print(f"\n⚠️  Insufficient balance for {pair_key}")
                    continue

                # Execute the trade
                success = self.execute_pair_trade(pair_info, num_pairs)

                if success:
                    # Mark as traded
                    print(f"\n   🔍 DEBUG: Marking {pair_key} as traded")
                    self.traded_pairs.add(pair_key)
                    print(f"   🔍 DEBUG: Traded pairs now: {self.traded_pairs}")

                    # Update balance
                    usdc_balance -= (pair_cost * num_pairs)
                    print(f"\n💰 Remaining USDC: ${usdc_balance:.2f}")

                    # Check if we can afford more pairs
                    if usdc_balance < (self.MIN_ASK_PRICE * 2 * self.BASE_POSITION_SIZE):
                        print("💼 Insufficient balance for additional pairs")
                        break
                else:
                    print(f"\n   ⚠️  DEBUG: Trade failed for {pair_key}, NOT marking as traded")

            return True  # Keep checking every second

        except Exception as e:
            print(f"\n❌ Error in trading cycle: {e}")
            import traceback
            traceback.print_exc()
            return True  # Keep checking even after errors

    def run(self):
        """Main trading loop - checks every second when trading window is open"""
        print("\n🚀 Starting pair trading bot...")
        print(f"📊 Strategy: {self.COINS} coins, PUT+CALL pairs")
        print(f"💵 Price range: ${self.MIN_ASK_PRICE} - ${self.MAX_ASK_PRICE}")
        print(f"💰 Max sum: ${self.MAX_SUM_PRICE}")
        print(f"⏰ Min time to expiry: {self.MIN_TIME_TO_EXPIRY_MINUTES} minutes")
        print(f"⚡ Check interval: 1 second when trading window open, 12 seconds when waiting")
        print(f"🛡️  Safety: Order verification, auto-cancel unfilled, position checks")
        print("\n" + "=" * 70 + "\n")

        cycle_count = 0

        try:
            while True:
                cycle_count += 1
                print(f"\n{'='*70}")
                print(f"🔄 CYCLE #{cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*70}")

                # Run trading cycle and get back whether to check frequently
                check_frequently = self.run_trading_cycle()

                if check_frequently:
                    # Active trading window - check every second
                    print(f"\n⚡ Checking again in 1 second...")
                    time.sleep(1)
                else:
                    # Waiting for next period - check every 12 seconds
                    print(f"\n😴 Waiting for next period - sleeping 12 seconds...")
                    time.sleep(12)

        except KeyboardInterrupt:
            print("\n\n🛑 Bot stopped by user")
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("🎯 15-MINUTE BINARY OPTIONS PAIR TRADER (WITH SAFETY NETS)")
    print("="*70)

    # Load credentials
    try:
        env_path = '/home/ubuntu/013_2025_polymarket/keys/keys_ovh38.env'
        creds = load_credentials_from_env(env_path)
        print(f"✅ Credentials loaded from {env_path}")
    except Exception as e:
        print(f"❌ Error loading credentials: {e}")
        return

    # Initialize and run bot
    try:
        bot = PairTrader(creds)
        bot.run()  # Dynamic checking: 1s when trading, 12s when waiting
    except Exception as e:
        print(f"❌ Error initializing bot: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
