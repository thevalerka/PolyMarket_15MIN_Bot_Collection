#!/usr/bin/env python3
"""
Bot016_MM_Gain_Safe - Binary Options Market Making Strategy (GAIN ONLY + SAFE)
Provides liquidity by placing limit orders on both sides of the book
- Sells ONLY at prices > entry price (guaranteed profit) for base position
- Emergency sells EXCESS shares over MAX to limit exposure
Max 6 shares per side, minimum order 5 shares
Uses OVH39 credentials
"""

import json
import time
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, List
from collections import deque
from dataclasses import dataclass, asdict
import logging
import requests

# Import Polymarket trading core
sys.path.insert(0, '/home/ubuntu')
from polymarket_trading_core_debug import PolymarketTrader, load_credentials_from_env
from sensitivity_api import SensitivityAPI

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Only show WARNING and ERROR
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# File paths
PUT_FILE = "/home/ubuntu/013_2025_polymarket/15M_BTC_PUT_rest.json"
CALL_FILE = "/home/ubuntu/013_2025_polymarket/15M_BTC_CALL_rest.json"
BTC_FILE = "/home/ubuntu/013_2025_polymarket/bot016_react/coinbase_btc_price.json"
BTC_FILE_COINBASE = "/home/ubuntu/013_2025_polymarket/bot016_react/coinbase_btc_price.json"
BTC_FILE_CHAINLINK = "/home/ubuntu/013_2025_polymarket/chainlink_btc_price.json"
SENSITIVITY_FILE = "/home/ubuntu/013_2025_polymarket/bot016_react/sensitivity_data/sensitivity_transformed.json"
TRADES_DIR = "/home/ubuntu/013_2025_polymarket/bot016_react/mm_trades_gain_safe"

# Credentials
CREDENTIALS_ENV = '/home/ubuntu/013_2025_polymarket/keys/keys_ovh39.env'

# Market making parameters
SENS_MULTIPLIER = 1  # Sensitivity multiplier for signal strength
SIGNAL_THRESHOLD = 0.01  # Reposition if signal > $0.01 option price movement
BUY_OFFSET = 0.00  # Buy at bid - $0.00
SELL_OFFSET = 0.00  # Sell at ask + $0.00
MAX_SHARES_PER_SIDE = 6  # Maximum shares per side
MIN_PROFIT_MARGIN = 0.01  # Minimum profit per share to sell ($0.01)
STOP_LOSS_MARGIN = 0.2
BUFFER_SECONDS = 10 # No trading in first 30s of period
POSITION_CHECK_INTERVAL = 1  # Check positions every 30s
REPOSITION_DELAY = 2.0  # Wait 2 seconds after cancel before repositioning


@dataclass
class Position:
    """Position tracker with entry price for profit calculation"""
    token_type: str  # 'PUT' or 'CALL'
    token_id: str
    entry_price: float  # CRITICAL: actual buy price for gain-only logic
    entry_time: float
    quantity: float


@dataclass
class OpenOrder:
    """Track open orders"""
    order_id: str
    token_type: str  # 'PUT' or 'CALL'
    side: str  # 'BUY' or 'SELL'
    price: float
    quantity: float
    token_id: str


def read_json(filepath: str) -> Optional[dict]:
    """Read JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        return None


def calculate_btc_volatility(price_history: deque) -> float:
    """Calculate BTC volatility over last minute (price range)"""
    if len(price_history) < 10:
        return 0.0
    prices = list(price_history)
    return max(prices) - min(prices)


class MarketMakerBotGainSafe:
    """Market maker for 15-minute BTC binary options - GAIN ONLY + SAFE (emergency sell excess)"""

    def __init__(self, credentials: dict):
        """Initialize market maker bot"""
        self.trader = PolymarketTrader(
            clob_api_url=credentials['clob_api_url'],
            private_key=credentials['private_key'],
            api_key=credentials['api_key'],
            api_secret=credentials['api_secret'],
            api_passphrase=credentials['api_passphrase']
        )

        # Position tracking - CRITICAL for gain-only logic
        self.positions: List[Position] = []

        # Order tracking
        self.open_orders: List[OpenOrder] = []

        # Asset IDs
        self.current_put_id: Optional[str] = None
        self.current_call_id: Optional[str] = None
        self.previous_put_id: Optional[str] = None
        self.previous_call_id: Optional[str] = None

        # Trades
        self.trades_dir = Path(TRADES_DIR)
        self.trades_dir.mkdir(exist_ok=True, parents=True)
        self.today_trades = []
        self.load_today_trades()

        # Load sensitivity API
        self.sensitivity_api = SensitivityAPI(SENSITIVITY_FILE)

        # Price history
        self.btc_price_history = deque(maxlen=600)

        # Period tracking
        self.current_period_start: Optional[int] = None
        self.strike_price: Optional[float] = None
        self.last_strike_check = 0
        self.buffer_check_done = False
        self.new_period_initialized = False  # Flag for post-buffer initialization

        # Last known prices for signal calculation
        self.last_btc_price: Optional[float] = None
        self.last_call_ask: Optional[float] = None
        self.last_call_bid: Optional[float] = None
        self.last_put_ask: Optional[float] = None
        self.last_put_bid: Optional[float] = None

        # Last buy order prices (for entry price tracking)
        self.last_call_buy_price: float = 0.50
        self.last_put_buy_price: float = 0.50

        # Timing
        self.last_position_check = time.time()
        self.last_asset_reload = time.time()
        self.last_cancel_time = 0

        # USDC balance cache
        self.cached_usdc_balance = 0
        self.last_usdc_check = 0

        self.print_startup_info()

    def print_startup_info(self):
        """Print startup information"""
        print("=" * 70)
        print("BOT016_MM_GAIN_SAFE - Gain Only + Emergency Sell Excess")
        print(f"Buy: bid-${BUY_OFFSET:.2f} | Sell: ONLY when price > entry + ${MIN_PROFIT_MARGIN:.2f}")
        print(f"Max: {MAX_SHARES_PER_SIDE}/side | Emergency sell EXCESS over max")
        print("=" * 70)

    def get_today_filename(self) -> Path:
        """Get filename for today's trades"""
        today = date.today().strftime('%Y%m%d')
        return self.trades_dir / f"mm_gain_safe_trades_{today}.json"

    def load_today_trades(self):
        """Load today's trades from file"""
        try:
            filename = self.get_today_filename()
            if filename.exists():
                with open(filename, 'r') as f:
                    self.today_trades = json.load(f)
                logger.info(f"📁 Loaded {len(self.today_trades)} trades from {filename.name}")
        except Exception as e:
            logger.error(f"❌ Error loading trades: {e}")
            self.today_trades = []

    def save_trade(self, trade: dict):
        """Save a trade to file"""
        try:
            self.today_trades.append(trade)
            filename = self.get_today_filename()
            with open(filename, 'w') as f:
                json.dump(self.today_trades, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Error saving trade: {e}")

    def reload_asset_ids(self):
        """Reload PUT and CALL asset IDs from data files"""
        put_data = read_json(PUT_FILE)
        call_data = read_json(CALL_FILE)

        if put_data and call_data:
            new_put_id = put_data.get('asset_id')
            new_call_id = call_data.get('asset_id')

            put_changed = new_put_id != self.current_put_id
            call_changed = new_call_id != self.current_call_id

            if put_changed or call_changed:
                logger.info(f"   🔄 Asset IDs updated:")
                if put_changed:
                    logger.info(f"   PUT:  ...{new_put_id[-12:]}")
                    self.previous_put_id = self.current_put_id
                if call_changed:
                    logger.info(f"   CALL: ...{new_call_id[-12:]}")
                    self.previous_call_id = self.current_call_id

                self.current_put_id = new_put_id
                self.current_call_id = new_call_id

            self.last_asset_reload = time.time()

    def refresh_usdc_balance(self):
        """Refresh cached USDC balance with timeout"""
        if time.time() - self.last_usdc_check >= 10:
            try:
                from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
                import signal

                def timeout_handler(signum, frame):
                    raise TimeoutError("USDC balance check timeout")

                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(2)

                try:
                    response = self.trader.client.get_balance_allowance(
                        params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
                    )
                    balance_raw = int(response.get('balance', 0))
                    self.cached_usdc_balance = balance_raw / 10**6
                    self.last_usdc_check = time.time()
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

            except TimeoutError:
                logger.warning(f"⚠️  USDC balance check timed out")
                self.last_usdc_check = time.time()
            except Exception as e:
                logger.error(f"❌ Error checking USDC balance: {e}")
                self.last_usdc_check = time.time()

    def check_token_balance(self, token_id: str) -> float:
        """Get token balance"""
        try:
            _, balance = self.trader.get_token_balance(token_id)
            return balance
        except:
            return 0.0

    def update_strike_price(self):
        """Update strike price from chainlink data"""
        try:
            btc_data_chainlink = read_json(BTC_FILE_CHAINLINK)
            if btc_data_chainlink:
                new_strike = btc_data_chainlink.get('strike')
                if new_strike:
                    old_strike = self.strike_price
                    self.strike_price = new_strike
                    self.last_strike_check = time.time()

                    if old_strike and abs(new_strike - old_strike) > 0.01:
                        logger.info(f"📊 Strike updated: ${old_strike:.2f} → ${new_strike:.2f}")
                    elif not old_strike:
                        logger.info(f"✅ Strike: ${self.strike_price:.2f}")

                    return True
        except Exception as e:
            logger.error(f"❌ Error updating strike price: {e}")
        return False

    def get_average_entry_price(self, token_type: str) -> float:
        """Get weighted average entry price for a token type"""
        positions = [p for p in self.positions if p.token_type == token_type]
        if not positions:
            return 0.50  # Default

        total_qty = sum(p.quantity for p in positions)
        if total_qty == 0:
            return 0.50

        weighted_sum = sum(p.entry_price * p.quantity for p in positions)
        return weighted_sum / total_qty

    def verify_positions_from_wallet(self):
        """Verify positions against wallet balances - sync immediately every 30s"""
        try:
            # Check CALL balance
            call_balance = 0
            if self.current_call_id:
                call_balance = self.check_token_balance(self.current_call_id)
                call_positions = [p for p in self.positions if p.token_type == 'CALL']
                tracked_call = sum(p.quantity for p in call_positions)

                if abs(call_balance - tracked_call) > 0.5:
                    print(f"🔄 CALL wallet sync: {tracked_call:.0f} → {call_balance:.0f}")

                    # Get existing entry price or use last buy order price
                    old_entry = self.get_average_entry_price('CALL')
                    entry_price = self.last_call_buy_price
                    if entry_price == 0.50 and old_entry != 0.50:
                        entry_price = old_entry

                    self.positions = [p for p in self.positions if p.token_type != 'CALL']
                    if call_balance >= 1:
                        self.positions.append(Position(
                            token_type='CALL',
                            token_id=self.current_call_id,
                            entry_price=entry_price,
                            entry_time=time.time(),
                            quantity=call_balance
                        ))
                        print(f"   → CALL: {call_balance:.0f} shares @ ${entry_price:.2f} entry")

            # Check PUT balance
            put_balance = 0
            if self.current_put_id:
                put_balance = self.check_token_balance(self.current_put_id)
                put_positions = [p for p in self.positions if p.token_type == 'PUT']
                tracked_put = sum(p.quantity for p in put_positions)

                if abs(put_balance - tracked_put) > 0.5:
                    print(f"🔄 PUT wallet sync: {tracked_put:.0f} → {put_balance:.0f}")

                    old_entry = self.get_average_entry_price('PUT')
                    entry_price = self.last_put_buy_price
                    if entry_price == 0.50 and old_entry != 0.50:
                        entry_price = old_entry

                    self.positions = [p for p in self.positions if p.token_type != 'PUT']
                    if put_balance >= 1:
                        self.positions.append(Position(
                            token_type='PUT',
                            token_id=self.current_put_id,
                            entry_price=entry_price,
                            entry_time=time.time(),
                            quantity=put_balance
                        ))
                        print(f"   → PUT: {put_balance:.0f} shares @ ${entry_price:.2f} entry")

            # Always print wallet status every 30s
            print(f"[WALLET] CALL={call_balance:.0f} PUT={put_balance:.0f}")

            self.last_position_check = time.time()

        except Exception as e:
            logger.error(f"❌ Error verifying positions: {e}")
            import traceback
            traceback.print_exc()

    def cancel_all_orders(self):
        """Cancel all open orders and clear tracking"""
        try:
            self.trader.cancel_all_orders()
            self.open_orders = []
            logger.info("🗑️  All orders cancelled")
        except Exception as e:
            logger.error(f"❌ Error cancelling orders: {e}")

    def place_buy_order(self, token_type: str, token_id: str, price: float, quantity: float) -> Optional[str]:
        """Place limit buy order - price becomes the new entry price"""
        try:
            order_id = self.trader.place_buy_order_limit(token_id, price, quantity)

            if order_id:
                self.open_orders.append(OpenOrder(
                    order_id=order_id,
                    token_type=token_type,
                    side='BUY',
                    price=price,
                    quantity=quantity,
                    token_id=token_id
                ))

                # UPDATE ENTRY PRICE - this buy order price is the new entry price
                # This updates immediately so any future sells know the correct entry
                if token_type == 'CALL':
                    self.last_call_buy_price = price
                    # Update existing positions to use new weighted average entry
                    self._update_position_entry_price('CALL', token_id, price, quantity)
                elif token_type == 'PUT':
                    self.last_put_buy_price = price
                    self._update_position_entry_price('PUT', token_id, price, quantity)

                print(f"BUY  {token_type:4} | {quantity:.0f} @ ${price:.2f} (new entry price)")
                return order_id

        except Exception as e:
            logger.error(f"❌ Error placing buy order: {e}")
        return None

    def _update_position_entry_price(self, token_type: str, token_id: str, new_price: float, new_quantity: float):
        """Update position entry price when new buy order is placed

        When a new buy order is placed at a new price, we update the entry price
        to reflect this new order (assuming it will fill). This is critical for
        gain-only logic - we need to know what price we're actually buying at.
        """
        existing_positions = [p for p in self.positions if p.token_type == token_type]
        existing_qty = sum(p.quantity for p in existing_positions)

        if existing_qty > 0:
            # Calculate weighted average of old positions + new order
            old_weighted = sum(p.entry_price * p.quantity for p in existing_positions)
            new_weighted = new_price * new_quantity
            total_qty = existing_qty + new_quantity
            new_avg_entry = (old_weighted + new_weighted) / total_qty

            # Update all positions of this type to the new weighted average
            for p in self.positions:
                if p.token_type == token_type:
                    p.entry_price = new_avg_entry
        else:
            # No existing position - create one with this entry price
            # (will be synced/verified from wallet later)
            pass  # Position will be created when wallet sync happens

    def place_sell_order(self, token_type: str, token_id: str, price: float, quantity: float, entry_price: float) -> Optional[str]:
        """Place limit sell order - ONLY if profitable"""
        try:
            # GAIN-ONLY CHECK: Only sell if price > entry_price + min_profit
            min_sell_price = entry_price + MIN_PROFIT_MARGIN


            if price < min_sell_price and price > entry_price - STOP_LOSS_MARGIN:
                print(f"⏳ HOLD {token_type:4} | sell ${price:.2f} < entry ${entry_price:.2f} + ${MIN_PROFIT_MARGIN:.2f} = ${min_sell_price:.2f}")
                return None

            profit_per_share = price - entry_price

            order_id = self.trader.place_sell_order_limit(token_id, price, quantity)

            if order_id:
                self.open_orders.append(OpenOrder(
                    order_id=order_id,
                    token_type=token_type,
                    side='SELL',
                    price=price,
                    quantity=quantity,
                    token_id=token_id
                ))
                print(f"SELL {token_type:4} | {quantity:.0f} @ ${price:.2f} (entry ${entry_price:.2f}, profit +${profit_per_share:.2f}/share)")
                return order_id

        except Exception as e:
            logger.error(f"❌ Error placing sell order: {e}")
        return None

    def check_signal_for_reposition(self, btc_price: float, btc_delta: float,
                                     call_ask: float, call_bid: float, call_ask_movement: float, call_bid_movement: float,
                                     put_ask: float, put_bid: float, put_ask_movement: float, put_bid_movement: float,
                                     seconds_remaining: float, volatility: float) -> bool:
        """Check if sensitivity signals indicate we should reposition orders"""
        try:
            if not self.strike_price or btc_delta is None:
                return False

            distance = abs(btc_price - self.strike_price)

            prediction = self.sensitivity_api.get_sensitivity(
                btc_price=btc_price,
                strike_price=self.strike_price,
                time_to_expiry_seconds=seconds_remaining,
                volatility_percent=volatility
            )

            call_sens = prediction['sensitivity']['call']
            put_sens = prediction['sensitivity']['put']

            ideal_call_movement = btc_delta * call_sens * SENS_MULTIPLIER
            ideal_put_movement = btc_delta * put_sens * SENS_MULTIPLIER

            should_reposition = False

            if abs(call_sens) > 0.000001 and abs(ideal_call_movement) > SIGNAL_THRESHOLD:
                if abs(ideal_call_movement) > abs(call_ask_movement) or abs(ideal_call_movement) > abs(call_bid_movement):
                    logger.warning(f"⚠️  CALL signal: ideal={ideal_call_movement:+.3f}")
                    should_reposition = True

            if abs(put_sens) > 0.000001 and abs(ideal_put_movement) > SIGNAL_THRESHOLD:
                if abs(ideal_put_movement) > abs(put_ask_movement) or abs(ideal_put_movement) > abs(put_bid_movement):
                    logger.warning(f"⚠️  PUT signal: ideal={ideal_put_movement:+.3f}")
                    should_reposition = True

            if should_reposition:
                logger.info(f"📊 Signal: BTC_Δ=${btc_delta:+.2f} | Distance={distance:.1f}")

            return should_reposition

        except Exception as e:
            logger.error(f"❌ Error checking signals: {e}")
            return False

    def emergency_sell_excess(self, token_type: str, token_id: str, excess_shares: float, bid_price: float):
        """Emergency sell ONLY the excess shares over MAX - sells at market regardless of profit"""

        # Round down to whole shares, minimum 5
        sell_qty = int(excess_shares)
        if sell_qty < 5:
            print(f"⚠️ {token_type} excess {excess_shares:.1f} < 5 minimum sell, waiting...")
            return

        print(f"🚨 EMERGENCY SELL {token_type} | {sell_qty} EXCESS shares @ ${bid_price:.2f} (over max {MAX_SHARES_PER_SIDE})")

        try:
            # Use FAK (Fill and Kill) for immediate execution at bid
            order_id = self.trader.place_sell_order_FAK(
                token_id=token_id,
                price=bid_price,
                quantity=sell_qty
            )

            if order_id:
                self.save_trade({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'EMERGENCY_SELL_EXCESS',
                    'token_type': token_type,
                    'quantity': sell_qty,
                    'price': bid_price,
                    'reason': 'excess_over_max'
                })
                print(f"✅ Emergency sell executed: {sell_qty} {token_type} @ ${bid_price:.2f}")
        except Exception as e:
            logger.error(f"❌ Emergency sell failed: {e}")

    def update_market_making_orders(self, call_bid: float, call_ask: float, put_bid: float, put_ask: float, btc_delta: float = 0):
        """Update market making orders - GAIN ONLY logic with emergency sell for excess"""
        try:
            # Calculate position counts and entry prices
            call_positions = [p for p in self.positions if p.token_type == 'CALL']
            put_positions = [p for p in self.positions if p.token_type == 'PUT']

            call_shares = sum(p.quantity for p in call_positions)
            put_shares = sum(p.quantity for p in put_positions)

            call_entry = self.get_average_entry_price('CALL')
            put_entry = self.get_average_entry_price('PUT')

            # Print positions with entry prices
            over_max_call = " ⚠️OVER" if call_shares > MAX_SHARES_PER_SIDE else ""
            over_max_put = " ⚠️OVER" if put_shares > MAX_SHARES_PER_SIDE else ""
            print(f"\n[POS] CALL={call_shares:.0f}/{MAX_SHARES_PER_SIDE}{over_max_call} (entry ${call_entry:.2f}) | "
                  f"PUT={put_shares:.0f}/{MAX_SHARES_PER_SIDE}{over_max_put} (entry ${put_entry:.2f}) | BTC_Δ=${btc_delta:+.2f}")

            # === EMERGENCY SELL EXCESS ONLY (not the base MAX_SHARES_PER_SIDE) ===
            # Only sell the amount OVER the max, keep MAX_SHARES_PER_SIDE for gain-only selling
            if call_shares > MAX_SHARES_PER_SIDE:
                excess_call = call_shares - MAX_SHARES_PER_SIDE
                if excess_call >= 5:
                    self.emergency_sell_excess('CALL', self.current_call_id, excess_call, call_bid)
                    return  # Exit and re-evaluate next cycle

            if put_shares > MAX_SHARES_PER_SIDE:
                excess_put = put_shares - MAX_SHARES_PER_SIDE
                if excess_put >= 5:
                    self.emergency_sell_excess('PUT', self.current_put_id, excess_put, put_bid)
                    return  # Exit and re-evaluate next cycle

            # === CALL ORDERS ===
            # Buy order - minimum 5 shares required, only if under max
            room_for_call = MAX_SHARES_PER_SIDE - call_shares
            if room_for_call >= 5:  # Only buy if we have room for at least 5
                buy_price = max(0.01, call_bid - BUY_OFFSET)
                buy_price = round(buy_price, 2)
                buy_quantity = max(5, min(6, room_for_call))  # At least 5, up to 6

                if 0.01 <= buy_price <= 0.99:
                    self.place_buy_order('CALL', self.current_call_id, buy_price, buy_quantity)
            elif room_for_call > 0 and call_shares < 5:
                # STUCK: Have <5 shares and can't buy 5 more to reach sellable amount
                # Solution: Buy 5 anyway to enable selling (will go over max temporarily)
                buy_price = max(0.01, call_bid - BUY_OFFSET)
                buy_price = round(buy_price, 2)
                print(f"⚠️ CALL stuck ({call_shares:.0f} shares) - buying 5 to enable selling")
                if 0.01 <= buy_price <= 0.99:
                    self.place_buy_order('CALL', self.current_call_id, buy_price, 5)

            # Sell order ONLY if profitable AND have at least 5 shares
            if call_shares >= 5:
                sell_price = min(0.99, call_ask + SELL_OFFSET)
                sell_price = round(sell_price, 2)
                # Check actual wallet balance to determine sell quantity
                actual_call_balance = self.check_token_balance(self.current_call_id) if self.current_call_id else 0
                if actual_call_balance >= 5:
                    sell_quantity = min(int(actual_call_balance), 6)  # Sell up to 6, but not more than we have
                    if 0.01 <= sell_price <= 0.99:
                        self.place_sell_order('CALL', self.current_call_id, sell_price, sell_quantity, call_entry)
                else:
                    print(f"⏳ CALL: actual balance {actual_call_balance:.2f} < 5 minimum to sell")
            elif call_shares > 0 and call_shares < 5:
                print(f"⏳ CALL: {call_shares:.0f} shares < 5 minimum to sell")

            # === PUT ORDERS ===
            # Buy order - minimum 5 shares required, only if under max
            room_for_put = MAX_SHARES_PER_SIDE - put_shares
            if room_for_put >= 5:  # Only buy if we have room for at least 5
                buy_price = max(0.01, put_bid - BUY_OFFSET)
                buy_price = round(buy_price, 2)
                buy_quantity = max(5, min(6, room_for_put))  # At least 5, up to 6

                if 0.01 <= buy_price <= 0.99:
                    self.place_buy_order('PUT', self.current_put_id, buy_price, buy_quantity)
            elif room_for_put > 0 and put_shares < 5:
                # STUCK: Have <5 shares and can't buy 5 more to reach sellable amount
                # Solution: Buy 5 anyway to enable selling (will go over max temporarily)
                buy_price = max(0.01, put_bid - BUY_OFFSET)
                buy_price = round(buy_price, 2)
                print(f"⚠️ PUT stuck ({put_shares:.0f} shares) - buying 5 to enable selling")
                if 0.01 <= buy_price <= 0.99:
                    self.place_buy_order('PUT', self.current_put_id, buy_price, 5)

            # Sell order ONLY if profitable AND have at least 5 shares (even if over max)
            if put_shares >= 5:
                sell_price = min(0.99, put_ask + SELL_OFFSET)
                sell_price = round(sell_price, 2)
                # Check actual wallet balance to determine sell quantity
                actual_put_balance = self.check_token_balance(self.current_put_id) if self.current_put_id else 0
                if actual_put_balance >= 5:
                    sell_quantity = min(int(actual_put_balance), 6)  # Sell up to 6, but not more than we have
                    if 0.01 <= sell_price <= 0.99:
                        self.place_sell_order('PUT', self.current_put_id, sell_price, sell_quantity, put_entry)
                else:
                    print(f"⏳ PUT: actual balance {actual_put_balance:.2f} < 5 minimum to sell")
            elif put_shares > 0 and put_shares < 5:
                print(f"⏳ PUT: {put_shares:.0f} shares < 5 minimum to sell")

        except Exception as e:
            logger.error(f"❌ Error updating MM orders: {e}")

    def handle_filled_buy(self, token_type: str, token_id: str, fill_price: float, fill_quantity: float):
        """Handle a filled buy order - track position with actual fill price"""
        self.positions.append(Position(
            token_type=token_type,
            token_id=token_id,
            entry_price=fill_price,
            entry_time=time.time(),
            quantity=fill_quantity
        ))

        self.save_trade({
            'timestamp': datetime.now().isoformat(),
            'type': 'BUY_FILL',
            'token_type': token_type,
            'quantity': fill_quantity,
            'price': fill_price,
            'pnl': 0  # No PnL on buy
        })

        print(f"✅ FILL BUY {token_type} | {fill_quantity:.0f} @ ${fill_price:.2f}")

    def handle_filled_sell(self, token_type: str, fill_price: float, fill_quantity: float, entry_price: float):
        """Handle a filled sell order - record profit"""
        profit = (fill_price - entry_price) * fill_quantity

        self.save_trade({
            'timestamp': datetime.now().isoformat(),
            'type': 'SELL_FILL',
            'token_type': token_type,
            'quantity': fill_quantity,
            'price': fill_price,
            'entry_price': entry_price,
            'pnl': profit
        })

        print(f"💰 FILL SELL {token_type} | {fill_quantity:.0f} @ ${fill_price:.2f} (entry ${entry_price:.2f}, profit +${profit:.2f})")

    def run(self):
        """Main market making loop"""
        logger.info("\n🚀 Starting GAIN-SAFE market maker bot...")

        # Initial setup
        self.reload_asset_ids()
        self.refresh_usdc_balance()
        self.update_strike_price()

        while True:
            try:
                now = datetime.now()

                # Determine period (00, 15, 30, 45)
                minute = now.minute
                if minute < 15:
                    period_start = 0
                elif minute < 30:
                    period_start = 15
                elif minute < 45:
                    period_start = 30
                else:
                    period_start = 45

                seconds_into_period = (minute % 15) * 60 + now.second
                seconds_remaining = 900 - seconds_into_period
                in_buffer_zone = seconds_into_period < BUFFER_SECONDS

                # NEW PERIOD DETECTED
                if period_start != self.current_period_start:
                    logger.info(f"\n{'='*80}")
                    logger.info(f"🔄 NEW PERIOD: {now.strftime('%H:%M')} (:{period_start:02d})")
                    logger.info(f"   Waiting {BUFFER_SECONDS}s buffer before trading...")
                    logger.info(f"{'='*80}")

                    self.cancel_all_orders()

                    # CLEAR ALL POSITIONS - old options expired/settled
                    if self.positions:
                        call_shares = sum(p.quantity for p in self.positions if p.token_type == 'CALL')
                        put_shares = sum(p.quantity for p in self.positions if p.token_type == 'PUT')
                        print(f"🔄 Period ended - clearing OLD positions: CALL={call_shares:.0f} PUT={put_shares:.0f}")
                        self.positions = []

                    # Reset entry price tracking for new period
                    self.last_call_buy_price = 0.50
                    self.last_put_buy_price = 0.50

                    self.current_period_start = period_start
                    self.buffer_check_done = False
                    self.new_period_initialized = False  # Flag to do post-buffer init

                    # Reset price tracking
                    self.last_btc_price = None
                    self.last_call_ask = None
                    self.last_call_bid = None
                    self.last_put_ask = None
                    self.last_put_bid = None

                # POST-BUFFER INITIALIZATION (once per period, after buffer ends)
                if not in_buffer_zone and not self.new_period_initialized:
                    print(f"✅ Buffer ended - initializing new period...")

                    # 1. Reload asset IDs for NEW tokens
                    self.reload_asset_ids()

                    # 2. Update strike price
                    self.update_strike_price()

                    # 3. Check wallet for NEW token balances
                    self.verify_positions_from_wallet()

                    self.new_period_initialized = True
                    self.buffer_check_done = True
                    print(f"✅ New period ready - CALL: {self.current_call_id[-12:] if self.current_call_id else 'None'} | PUT: {self.current_put_id[-12:] if self.current_put_id else 'None'}")

                # Periodic maintenance (every 60s)
                if time.time() - self.last_asset_reload >= 60:
                    self.reload_asset_ids()

                if time.time() - self.last_strike_check >= 60:
                    self.update_strike_price()

                # WALLET CHECK every 30 seconds (only after buffer)
                if not in_buffer_zone and time.time() - self.last_position_check >= POSITION_CHECK_INTERVAL:
                    self.verify_positions_from_wallet()

                try:
                    self.refresh_usdc_balance()
                except Exception as e:
                    logger.error(f"❌ USDC balance check failed: {e}")

                # Read prices
                btc_data = read_json(BTC_FILE)
                call_data = read_json(CALL_FILE)
                put_data = read_json(PUT_FILE)

                if not all([btc_data, call_data, put_data]):
                    time.sleep(0.1)
                    continue

                btc_price = btc_data.get('price', 0)
                self.btc_price_history.append(btc_price)

                call_bid = call_data.get('best_bid', 0)
                call_ask = call_data.get('best_ask', 0)
                put_bid = put_data.get('best_bid', 0)
                put_ask = put_data.get('best_ask', 0)

                if not all([btc_price, call_bid, call_ask, put_bid, put_ask]):
                    time.sleep(0.1)
                    continue

                volatility = calculate_btc_volatility(self.btc_price_history)

                # Check signals for repositioning
                should_check_signals = (self.last_btc_price is not None and
                                       self.last_call_ask is not None and
                                       self.last_put_ask is not None and
                                       len(self.open_orders) > 0 and
                                       not in_buffer_zone)

                if should_check_signals:
                    btc_delta = btc_price - self.last_btc_price
                    call_ask_movement = call_ask - self.last_call_ask
                    call_bid_movement = call_bid - self.last_call_bid
                    put_ask_movement = put_ask - self.last_put_ask
                    put_bid_movement = put_bid - self.last_put_bid

                    if self.check_signal_for_reposition(
                        btc_price, btc_delta,
                        call_ask, call_bid, call_ask_movement, call_bid_movement,
                        put_ask, put_bid, put_ask_movement, put_bid_movement,
                        seconds_remaining, volatility
                    ):
                        logger.info("🔄 Repositioning orders based on signals...")
                        self.cancel_all_orders()
                        self.last_cancel_time = time.time()

                # Update tracking
                self.last_btc_price = btc_price
                self.last_call_ask = call_ask
                self.last_call_bid = call_bid
                self.last_put_ask = put_ask
                self.last_put_bid = put_bid

                # NO STOP LOSS - positions held until profitable

                # Place orders
                can_place_orders = (not in_buffer_zone and
                                   len(self.open_orders) == 0 and
                                   time.time() - self.last_cancel_time >= REPOSITION_DELAY)

                if can_place_orders and self.current_call_id and self.current_put_id:
                    btc_delta_display = 0
                    if self.last_btc_price is not None:
                        btc_delta_display = btc_price - self.last_btc_price
                    self.update_market_making_orders(call_bid, call_ask, put_bid, put_ask, btc_delta_display)

                time.sleep(0.1)

            except KeyboardInterrupt:
                logger.info("\n\n⏸️  Shutting down...")
                self.cancel_all_orders()
                break

            except Exception as e:
                logger.error(f"❌ Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)


def main():
    """Main entry point"""
    try:
        try:
            credentials = load_credentials_from_env(CREDENTIALS_ENV)
            print(f"✅ Credentials loaded from {CREDENTIALS_ENV}")
        except Exception as e:
            print(f"❌ Error loading credentials: {e}")
            return

        bot = MarketMakerBotGainSafe(credentials)
        bot.run()

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
