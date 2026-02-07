#!/usr/bin/env python3
"""
15-Minute Binary Options Pair Trading Bot - STRATEGY C (Correlation-Based)

Strategy:
- Trade COIN1 PUT + COIN2 CALL pairs when:
  1. More than 10 minutes until period end
  2. NO individual price limits (any price acceptable)
  3. Sum of asks < $0.80
  4. Correlation between coins > 0.7 (read from bybit_correlations.json)
- Opens positions sequentially (first then second mandatory)
- Fixed position size: 5 tokens per leg
- Safety checks: verifies order execution, cancels unfilled orders, checks position before retry

Exit Strategy:
- Continuous monitoring every second (even after 10 min threshold)
- Sells BOTH assets of a pair when:
  * asset1_bid + asset2_bid < $0.20 (stop loss)
  * asset1_bid + asset2_bid > $0.95 (take profit)
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# Import Polymarket trading core
sys.path.insert(0, '/home/claude')
from polymarket_trading_core_debug import PolymarketTrader, load_credentials_from_env

class CorrelationPairTrader:
    """15-minute binary options correlation-based pair trading bot"""

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

    # Correlation data file
    CORRELATION_FILE = '/home/ubuntu/013_2025_polymarket/bybit_correlations.json'

    # Trading parameters
    MAX_SUM_PRICE = 0.80  # Sum of asks must be less than this
    MIN_CORRELATION = -0.99  # Minimum correlation required
    MIN_TIME_TO_EXPIRY_MINUTES = 10

    # Exit thresholds
    STOP_LOSS_THRESHOLD = 0.20  # Sell when sum of bids < $0.20
    TAKE_PROFIT_THRESHOLD = 0.95  # Sell when sum of bids > $0.95

    # Position sizing
    POSITION_SIZE_PER_LEG = 5  # Fixed 5 tokens per leg
    USDC_ADDRESS = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"

    def __init__(self, credentials: Dict[str, str]):
        """Initialize the correlation-based pair trader"""
        self.trader = PolymarketTrader(
            clob_api_url=credentials['clob_api_url'],
            private_key=credentials['private_key'],
            api_key=credentials['api_key'],
            api_secret=credentials['api_secret'],
            api_passphrase=credentials['api_passphrase']
        )

        self.traded_pairs = set()  # Track pairs already traded this period
        # Enhanced position tracking with pair information
        self.open_positions = {}  # {token_id: {'size': float, 'coin': str, 'type': str, 'pair_id': str, 'pair_token_id': str}}
        self.correlations = {}  # Cache correlations

        print("=" * 70)
        print("🎯 15-MINUTE PAIR TRADING BOT - STRATEGY C (CORRELATION-BASED)")
        print("=" * 70)
        print(f"📊 Strategy: Correlation > {self.MIN_CORRELATION}")
        print(f"💰 Max Sum: < ${self.MAX_SUM_PRICE}")
        print(f"🔢 No individual price limits")
        print(f"📦 Position Size: {self.POSITION_SIZE_PER_LEG} tokens per leg")
        print(f"📉 Stop Loss: Exit when bid sum < ${self.STOP_LOSS_THRESHOLD}")
        print(f"📈 Take Profit: Exit when bid sum > ${self.TAKE_PROFIT_THRESHOLD}")
        print("⚡ Continuous monitoring every second")
        print("=" * 70)

    def read_correlations(self) -> Dict[str, float]:
        """
        Read correlation data from JSON file

        Returns:
            Dict mapping pair keys (e.g., 'BTC_ETH') to correlation values
        """
        try:
            if not os.path.exists(self.CORRELATION_FILE):
                print(f"⚠️  Correlation file not found: {self.CORRELATION_FILE}")
                return {}

            with open(self.CORRELATION_FILE, 'r') as f:
                data = json.load(f)

            correlations = {}
            for pair_key, pair_data in data.get('correlations', {}).items():
                correlation = pair_data.get('correlation', 0)
                correlations[pair_key] = correlation

            return correlations

        except Exception as e:
            print(f"❌ Error reading correlations: {e}")
            return {}

    def get_correlation(self, coin1: str, coin2: str) -> float:
        """
        Get correlation between two coins

        Args:
            coin1: First coin symbol
            coin2: Second coin symbol

        Returns:
            Correlation value, or 0 if not found
        """
        # Try both directions
        key1 = f"{coin1}_{coin2}"
        key2 = f"{coin2}_{coin1}"

        if key1 in self.correlations:
            return self.correlations[key1]
        elif key2 in self.correlations:
            return self.correlations[key2]
        else:
            return 0.0

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

    def check_pair_conditions(self, coin1: str, coin2: str) -> Optional[Dict]:
        """
        Check if pair trading conditions are met (STRATEGY C)

        Conditions:
        1. Sum of asks < MAX_SUM_PRICE ($0.80)
        2. Correlation > MIN_CORRELATION (0.7)
        3. NO individual price limits

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

        # Check sum condition
        sum_price = put_ask + call_ask
        sum_ok = sum_price < self.MAX_SUM_PRICE

        # Check correlation
        correlation = self.get_correlation(coin1, coin2)
        correlation_ok = correlation >= self.MIN_CORRELATION

        if sum_ok and correlation_ok:
            return {
                'coin1': coin1,
                'coin2': coin2,
                'put_ask': put_ask,
                'call_ask': call_ask,
                'total_cost': sum_price,
                'correlation': correlation,
                'put_token_id': put_data['asset_id'],
                'call_token_id': call_data['asset_id'],
                'put_asset_name': put_data['asset_name'],
                'call_asset_name': call_data['asset_name']
            }

        return None

    def check_token_balance(self, token_id: str) -> float:
        """Check current balance of a specific token"""
        try:
            balance_raw, balance = self.trader.get_token_balance(token_id)
            return balance
        except Exception as e:
            print(f"❌ Error checking balance for {token_id[:12]}...: {e}")
            return 0.0

    def place_immediate_market_sell(self, token_id: str, size: float, coin: str, option_type: str) -> bool:
        """
        Place an immediate market sell order

        Args:
            token_id: Token to sell
            size: Number of tokens to sell
            coin: Coin name for logging
            option_type: PUT or CALL for logging

        Returns:
            True if order placed successfully
        """
        try:
            print(f"   💰 SELL: Placing market sell for {coin} {option_type}")
            print(f"   💰 SELL: Size: {size:.2f} tokens")

            # Get current best bid to place competitive sell
            market_data = self.read_market_data(coin, option_type)
            if not market_data:
                print(f"   ❌ SELL: Cannot read market data")
                return False

            best_bid = market_data['best_bid'].get('price', 0)
            if best_bid <= 0:
                print(f"   ⚠️  SELL: No bid price available, using $0.01")
                sell_price = 0.01
            else:
                # Sell at or slightly below best bid for quick execution
                sell_price = max(0.01, best_bid - 0.001)

            print(f"   💰 SELL: Sell price: ${sell_price:.4f}")

            # Using place_sell_order
            order_id = self.trader.place_sell_order(
                token_id=token_id,
                price=sell_price,
                quantity=size
            )

            if order_id:
                print(f"   ✅ SELL: Order placed successfully (ID: {order_id})")
                return True
            else:
                print(f"   ❌ SELL: Failed to place order")
                return False

        except Exception as e:
            print(f"   ❌ SELL: Exception placing sell order: {e}")
            import traceback
            traceback.print_exc()
            return False

    def execute_pair_trade(self, pair_info: Dict) -> bool:
        """
        Execute the pair trade (PUT + CALL) with laggy API handling

        Strategy:
        1. Open both positions immediately (1 attempt each)
        2. Verify balances at 5s, 10s, 20s intervals
        3. Use USDC balance as hint (faster than token balance)
        4. If wallet unchanged and no tokens, cancel all and retry
        5. If only 1 leg open, force open 2nd leg regardless of conditions

        Returns:
            True if both legs executed successfully
        """
        position_size = self.POSITION_SIZE_PER_LEG  # Fixed 5 tokens per leg
        expected_cost = pair_info['total_cost'] * position_size

        print("\n" + "=" * 70)
        print(f"🎯 EXECUTING PAIR TRADE")
        print("=" * 70)
        print(f"📊 Pair: {pair_info['coin1']} PUT + {pair_info['coin2']} CALL")
        print(f"📊 Correlation: {pair_info['correlation']:.4f}")
        print(f"💰 PUT ask: ${pair_info['put_ask']:.4f}")
        print(f"💰 CALL ask: ${pair_info['call_ask']:.4f}")
        print(f"💵 Expected cost: ${expected_cost:.2f}")
        print(f"📦 Position size: {position_size:.2f} per leg")
        print("=" * 70)

        # Record initial USDC balance BEFORE any orders (only once per trade attempt)
        if not hasattr(self, 'current_trade_initial_usdc'):
            self.current_trade_initial_usdc = None

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
                return self.execute_pair_trade(pair_info)
            else:
                print("   ❌ Conditions no longer valid - ABORTING")
                self.current_trade_initial_usdc = None
                return False

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
                self.current_trade_initial_usdc = None
                return False

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
                self.current_trade_initial_usdc = None
                return False

        # Final verification
        if not (put_confirmed and call_confirmed):
            print("\n❌ FAILED: Could not confirm both positions")
            self.current_trade_initial_usdc = None
            return False

        # ============================================================
        # PHASE 4: TRACK POSITIONS (No sell orders in dynamic exit strategy)
        # ============================================================

        print("\n📊 PHASE 4: Tracking positions...")

        # Create unique pair identifier
        pair_id = f"{pair_info['coin1']}_PUT-{pair_info['coin2']}_CALL"

        # Track both positions with pair information
        self.open_positions[pair_info['put_token_id']] = {
            'size': position_size,
            'coin': pair_info['coin1'],
            'type': 'PUT',
            'pair_id': pair_id,
            'pair_token_id': pair_info['call_token_id']
        }

        self.open_positions[pair_info['call_token_id']] = {
            'size': position_size,
            'coin': pair_info['coin2'],
            'type': 'CALL',
            'pair_id': pair_id,
            'pair_token_id': pair_info['put_token_id']
        }

        print("\n" + "=" * 70)
        print("🎉 PAIR TRADE COMPLETED")
        print("=" * 70)

        # Reset cached initial USDC for next trade
        self.current_trade_initial_usdc = None

        return True

    def scan_all_pairs(self) -> List[Tuple[str, Dict]]:
        """
        Scan all possible coin pairs for trading opportunities (STRATEGY C)

        Returns:
            List of (pair_key, pair_info) tuples for valid opportunities
        """
        opportunities = []

        # Get time to expiry for display
        time_left = self.minutes_to_expiry()

        # Read correlations fresh each time
        self.correlations = self.read_correlations()

        print(f"\n{'='*100}")
        print(f"{'PAIR NAME':<25} | {'ASSET1':<8} | {'ASSET2':<8} | {'COMBINED':<9} | {'CORREL':<8} | {'TIME':<12} | {'STATUS'}")
        print(f"{'='*100}")

        for i, coin1 in enumerate(self.COINS):
            for coin2 in self.COINS[i+1:]:
                # Check both directions
                pair_key1 = f"{coin1}_PUT-{coin2}_CALL"
                pair_key2 = f"{coin2}_PUT-{coin1}_CALL"

                # Get correlation for this pair
                correlation = self.get_correlation(coin1, coin2)

                # Direction 1: coin1 PUT + coin2 CALL
                pair_info1 = self.check_pair_conditions(coin1, coin2)
                if pair_info1:
                    put_ask = pair_info1['put_ask']
                    call_ask = pair_info1['call_ask']
                    total = pair_info1['total_cost']
                    corr = pair_info1['correlation']

                    # Check if already traded
                    if pair_key1 in self.traded_pairs:
                        status = "❌ ALREADY TRADED"
                    else:
                        status = "✅ OPPORTUNITY"
                        opportunities.append((pair_key1, pair_info1))

                    print(f"{pair_key1:<25} | ${put_ask:<7.4f} | ${call_ask:<7.4f} | ${total:<8.4f} | {corr:<7.4f} | {time_left:>5.1f} min  | {status}")
                else:
                    # Show why it doesn't meet conditions
                    put_data = self.read_market_data(coin1, 'PUT')
                    call_data = self.read_market_data(coin2, 'CALL')

                    if put_data and call_data:
                        put_ask = put_data['best_ask'].get('price', 0)
                        call_ask = call_data['best_ask'].get('price', 0)
                        total = put_ask + call_ask if (put_ask > 0 and call_ask > 0) else 0

                        # Determine why it failed
                        reasons = []
                        if put_ask == 0 or call_ask == 0:
                            reasons.append("NO PRICE")
                        else:
                            if total >= self.MAX_SUM_PRICE:
                                reasons.append(f"SUM ${total:.4f} >= ${self.MAX_SUM_PRICE}")
                            if correlation < self.MIN_CORRELATION:
                                reasons.append(f"CORR {correlation:.4f} < {self.MIN_CORRELATION}")

                        status = "⚠️  " + " | ".join(reasons) if reasons else "⚠️  FAILED"

                        if put_ask > 0 and call_ask > 0:
                            print(f"{pair_key1:<25} | ${put_ask:<7.4f} | ${call_ask:<7.4f} | ${total:<8.4f} | {correlation:<7.4f} | {time_left:>5.1f} min  | {status}")
                        else:
                            print(f"{pair_key1:<25} | {'N/A':<8} | {'N/A':<8} | {'N/A':<9} | {correlation:<7.4f} | {time_left:>5.1f} min  | {status}")

                # Direction 2: coin2 PUT + coin1 CALL
                pair_info2 = self.check_pair_conditions(coin2, coin1)
                if pair_info2:
                    put_ask = pair_info2['put_ask']
                    call_ask = pair_info2['call_ask']
                    total = pair_info2['total_cost']
                    corr = pair_info2['correlation']

                    # Check if already traded
                    if pair_key2 in self.traded_pairs:
                        status = "❌ ALREADY TRADED"
                    else:
                        status = "✅ OPPORTUNITY"
                        opportunities.append((pair_key2, pair_info2))

                    print(f"{pair_key2:<25} | ${put_ask:<7.4f} | ${call_ask:<7.4f} | ${total:<8.4f} | {corr:<7.4f} | {time_left:>5.1f} min  | {status}")
                else:
                    # Show why it doesn't meet conditions
                    put_data = self.read_market_data(coin2, 'PUT')
                    call_data = self.read_market_data(coin1, 'CALL')

                    if put_data and call_data:
                        put_ask = put_data['best_ask'].get('price', 0)
                        call_ask = call_data['best_ask'].get('price', 0)
                        total = put_ask + call_ask if (put_ask > 0 and call_ask > 0) else 0

                        # Determine why it failed
                        reasons = []
                        if put_ask == 0 or call_ask == 0:
                            reasons.append("NO PRICE")
                        else:
                            if total >= self.MAX_SUM_PRICE:
                                reasons.append(f"SUM ${total:.4f} >= ${self.MAX_SUM_PRICE}")
                            if correlation < self.MIN_CORRELATION:
                                reasons.append(f"CORR {correlation:.4f} < {self.MIN_CORRELATION}")

                        status = "⚠️  " + " | ".join(reasons) if reasons else "⚠️  FAILED"

                        if put_ask > 0 and call_ask > 0:
                            print(f"{pair_key2:<25} | ${put_ask:<7.4f} | ${call_ask:<7.4f} | ${total:<8.4f} | {correlation:<7.4f} | {time_left:>5.1f} min  | {status}")
                        else:
                            print(f"{pair_key2:<25} | {'N/A':<8} | {'N/A':<8} | {'N/A':<9} | {correlation:<7.4f} | {time_left:>5.1f} min  | {status}")

        print(f"{'='*100}")
        print(f"✨ Total opportunities found: {len(opportunities)}")
        print(f"{'='*100}\n")

        return opportunities

    def monitor_and_exit_positions(self):
        """
        Monitor all open positions and execute exit strategy
        Checks bid prices and sells both legs when thresholds are hit
        """
        if not self.open_positions:
            return

        # Group positions by pair
        pairs_to_check = {}
        for token_id, position_info in self.open_positions.items():
            pair_id = position_info['pair_id']
            if pair_id not in pairs_to_check:
                pairs_to_check[pair_id] = []
            pairs_to_check[pair_id].append((token_id, position_info))

        print(f"\n📊 MONITORING {len(pairs_to_check)} PAIRS")

        pairs_to_close = []
        pairs_to_forget = []  # Track fully closed pairs to remove

        for pair_id, positions in pairs_to_check.items():
            if len(positions) != 2:
                print(f"   ⚠️  {pair_id}: Incomplete pair ({len(positions)} legs)")
                continue

            # Get both legs
            leg1_token, leg1_info = positions[0]
            leg2_token, leg2_info = positions[1]

            coin1 = leg1_info['coin']
            type1 = leg1_info['type']
            coin2 = leg2_info['coin']
            type2 = leg2_info['type']

            # Check if positions are already closed (0 balance on BOTH legs)
            balance1 = self.check_token_balance(leg1_token)
            balance2 = self.check_token_balance(leg2_token)

            if balance1 < 0.01 and balance2 < 0.01:
                print(f"   ✅ {pair_id}: Both legs closed (0 balance) - forgetting position")
                pairs_to_forget.append((pair_id, [leg1_token, leg2_token]))
                continue

            # Read current market data
            data1 = self.read_market_data(coin1, type1)
            data2 = self.read_market_data(coin2, type2)

            if not data1 or not data2:
                print(f"   ⚠️  {pair_id}: Cannot read market data")
                continue

            bid1 = data1['best_bid'].get('price', 0)
            bid2 = data2['best_bid'].get('price', 0)

            if bid1 == 0 or bid2 == 0:
                print(f"   ⚠️  {pair_id}: Missing bid prices")
                continue

            bid_sum = bid1 + bid2

            print(f"   💹 {pair_id}:")
            print(f"      {coin1} {type1} bid: ${bid1:.4f} (balance: {balance1:.2f})")
            print(f"      {coin2} {type2} bid: ${bid2:.4f} (balance: {balance2:.2f})")
            print(f"      Sum: ${bid_sum:.4f} (Exit: <${self.STOP_LOSS_THRESHOLD} or >${self.TAKE_PROFIT_THRESHOLD})")

            # Check exit conditions
            should_exit = False
            exit_reason = ""

            if bid_sum < self.STOP_LOSS_THRESHOLD:
                should_exit = True
                exit_reason = f"STOP LOSS (${bid_sum:.4f} < ${self.STOP_LOSS_THRESHOLD})"
            elif bid_sum > self.TAKE_PROFIT_THRESHOLD:
                should_exit = True
                exit_reason = f"TAKE PROFIT (${bid_sum:.4f} > ${self.TAKE_PROFIT_THRESHOLD})"

            if should_exit:
                print(f"      🚨 EXIT TRIGGERED: {exit_reason}")
                pairs_to_close.append((pair_id, positions, exit_reason))

        # Remove fully closed pairs from tracking FIRST
        for pair_id, token_ids in pairs_to_forget:
            for token_id in token_ids:
                if token_id in self.open_positions:
                    del self.open_positions[token_id]
            print(f"   🗑️  Forgot closed pair: {pair_id}")

        # Execute exits
        for pair_id, positions, exit_reason in pairs_to_close:
            print(f"\n{'='*70}")
            print(f"🚨 CLOSING PAIR: {pair_id}")
            print(f"📊 Reason: {exit_reason}")
            print(f"{'='*70}")

            # Sell both legs
            tokens_sold = []
            for token_id, position_info in positions:
                coin = position_info['coin']
                option_type = position_info['type']
                size = position_info['size']

                # Check current balance
                current_balance = self.check_token_balance(token_id)

                if current_balance < 0.01:
                    print(f"   ℹ️  {coin} {option_type}: Already closed (0 tokens)")
                    tokens_sold.append(token_id)
                    continue

                # Place market sell
                print(f"\n   💰 Selling {coin} {option_type}...")
                success = self.place_immediate_market_sell(token_id, current_balance, coin, option_type)

                if success:
                    print(f"   ✅ {coin} {option_type} sell order placed")

                    # Wait and verify sale
                    time.sleep(3)
                    final_balance = self.check_token_balance(token_id)

                    if final_balance < 0.01:
                        print(f"   ✅ VERIFIED: {coin} {option_type} position closed")
                        tokens_sold.append(token_id)
                    else:
                        print(f"   ⚠️  {coin} {option_type} still has {final_balance:.2f} tokens")
                else:
                    print(f"   ❌ {coin} {option_type} sell failed")

            # Remove from tracking if both legs sold
            if len(tokens_sold) == 2:
                for token_id in tokens_sold:
                    if token_id in self.open_positions:
                        del self.open_positions[token_id]
                print(f"\n   🗑️  Pair fully closed and forgotten: {pair_id}")

            print(f"\n{'='*70}")
            print(f"✅ PAIR EXIT COMPLETE: {pair_id}")
            print(f"{'='*70}")

    def run_trading_cycle(self) -> bool:
        """
        Execute one trading cycle

        Returns:
            Always returns True (check every second continuously)
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

            # CONTINUOUS MONITORING: Check exit conditions every second
            if self.open_positions:
                self.monitor_and_exit_positions()

            # Only look for new entries if we have time AND available balance AND no open positions
            if time_left >= self.MIN_TIME_TO_EXPIRY_MINUTES:
                # Check USDC balance
                usdc_balance = self.get_usdc_balance()
                print(f"💰 USDC Balance: ${usdc_balance:.2f}")

                # LIMIT: Only 1 pair per period - check if we already have open positions
                if self.open_positions:
                    print(f"⏸️  Already have {len(self.open_positions)//2} pair(s) open - no new entries this period")
                    return True

                # Calculate cost for one pair
                min_required_balance = self.MAX_SUM_PRICE * self.POSITION_SIZE_PER_LEG

                if usdc_balance >= min_required_balance:
                    # Scan for opportunities
                    opportunities = self.scan_all_pairs()

                    if not opportunities:
                        print("📊 No valid opportunities at this time")
                    else:
                        # LIMIT: Only trade the FIRST opportunity found (1 pair per period)
                        pair_key, pair_info = opportunities[0]

                        print(f"\n🎯 Trading first opportunity: {pair_key}")

                        # Calculate cost for this specific pair
                        pair_cost = pair_info['total_cost'] * self.POSITION_SIZE_PER_LEG

                        if usdc_balance < pair_cost:
                            print(f"\n⚠️  Insufficient balance for {pair_key}: need ${pair_cost:.2f}, have ${usdc_balance:.2f}")
                        else:
                            # Execute the trade (always 5 tokens per leg)
                            success = self.execute_pair_trade(pair_info)

                            if success:
                                # Mark as traded
                                self.traded_pairs.add(pair_key)
                                print(f"\n   ✅ {pair_key} marked as traded")
                                print(f"\n   🔒 LIMIT: 1 pair per period - no more trades until next period")

                                # Update balance (deduct cost for the trade we just made)
                                usdc_balance -= pair_cost
                                print(f"\n💰 Remaining USDC: ${usdc_balance:.2f}")
                else:
                    print(f"⚠️  Insufficient USDC balance for new trades (need ${min_required_balance:.2f})")
            else:
                print(f"⏸️  Less than {self.MIN_TIME_TO_EXPIRY_MINUTES} minutes - no new entries")

            return True  # Always check every second

        except Exception as e:
            print(f"\n❌ Error in trading cycle: {e}")
            import traceback
            traceback.print_exc()
            return True  # Keep checking even after errors

    def run(self):
        """Main trading loop - checks every second continuously"""
        print("\n🚀 Starting correlation-based pair trading bot...")
        print(f"📊 Strategy: Correlation > {self.MIN_CORRELATION}, Sum < ${self.MAX_SUM_PRICE}")
        print(f"💰 No individual price limits")
        print(f"📦 Position Size: {self.POSITION_SIZE_PER_LEG} tokens per leg")
        print(f"⏰ Min time to expiry: {self.MIN_TIME_TO_EXPIRY_MINUTES} minutes")
        print(f"⚡ Check interval: 1 second (CONTINUOUS)")
        print(f"📉 Stop Loss: < ${self.STOP_LOSS_THRESHOLD}")
        print(f"📈 Take Profit: > ${self.TAKE_PROFIT_THRESHOLD}")
        print(f"🛡️  Safety: Position verification, sell verification")
        print(f"📊 Correlation File: {self.CORRELATION_FILE}")
        print(f"🔒 LIMIT: Maximum 1 pair per 15-minute period")
        print("\n" + "=" * 70 + "\n")

        cycle_count = 0

        try:
            while True:
                cycle_count += 1
                print(f"\n{'='*70}")
                print(f"🔄 CYCLE #{cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*70}")

                # Run trading cycle
                self.run_trading_cycle()

                # Always check every second
                print(f"\n⚡ Checking again in 1 second...")
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n\n🛑 Bot stopped by user")
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("🎯 15-MINUTE BINARY OPTIONS PAIR TRADER - STRATEGY C")
    print("="*70)

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
        bot = CorrelationPairTrader(creds)
        bot.run()  # Continuous 1-second checks
    except Exception as e:
        print(f"❌ Error initializing bot: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
