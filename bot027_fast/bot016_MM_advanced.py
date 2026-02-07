#!/usr/bin/env python3
"""
Bot016_MM_Advanced - Binary Options Market Making with Advanced MM Engine Integration
=====================================================================================

Integrates with the advanced MM engine by reading quote recommendations from mm_quotes.json
and placing orders at those levels instead of simple bid/ask following.

Key Changes from bot016_MM_pure.py:
1. Reads quote recommendations from mm_quotes.json
2. Uses MM engine's calculated prices (with skewing, regime detection, etc.)
3. Falls back to simple bid/ask if mm_quotes.json unavailable
4. Uses only Layer 0 (first tier) quotes for now

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
# Configure logging
logging.basicConfig(
    level=logging.WARNING,
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
TRADES_DIR = "/home/ubuntu/013_2025_polymarket/bot016_react/mm_trades_advanced"

# Advanced MM Engine integration
MM_QUOTES_FILE = "/home/ubuntu/013_2025_polymarket/bot027_fast/mm_quotes.json"
MM_STATUS_FILE = "/home/ubuntu/013_2025_polymarket/bot027_fast/mm_status.json"

# Credentials
CREDENTIALS_ENV = '/home/ubuntu/013_2025_polymarket/keys/keys_ovh39.env'

# Market making parameters
MAX_SHARES_PER_SIDE = 6
MIN_PROFIT_MARGIN = 0.03  # Minimum profit per share to sell in first 60s
BUFFER_SECONDS = 10
POSITION_CHECK_INTERVAL = 1
REPOSITION_DELAY = 2.0

# MM Engine integration settings
MM_QUOTES_MAX_AGE_MS = 5000  # Consider quotes stale if older than 5 seconds
USE_MM_ENGINE = True  # Set to False to disable MM engine and use simple bid/ask
FALLBACK_TO_SIMPLE = True  # If MM quotes unavailable, fall back to simple strategy


@dataclass
class Position:
    """Position tracker with entry price for profit calculation"""
    token_type: str
    token_id: str
    entry_price: float
    entry_time: float
    quantity: float


@dataclass
class OpenOrder:
    """Track open orders"""
    order_id: str
    token_type: str
    side: str
    price: float
    quantity: float
    token_id: str


@dataclass
class MMQuote:
    """Quote recommendation from MM engine"""
    asset_type: str  # 'CALL' or 'PUT'
    side: str  # 'bid' or 'ask'
    price: float
    size: float
    layer: int


def read_json(filepath: str) -> Optional[dict]:
    """Read JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        return None


def read_mm_quotes() -> Optional[Dict]:
    """
    Read quote recommendations from MM engine.
    Returns dict with 'CALL' and 'PUT' quote lists, or None if unavailable/stale.
    """
    try:
        data = read_json(MM_QUOTES_FILE)
        if not data:
            return None

        # Check if quotes are fresh
        quote_timestamp = data.get('timestamp', 0)
        now_ms = int(time.time() * 1000)
        age_ms = now_ms - quote_timestamp

        if age_ms > MM_QUOTES_MAX_AGE_MS:
            logger.warning(f"⚠️ MM quotes stale: {age_ms}ms old")
            return None

        # Parse quotes
        result = {
            'CALL': [],
            'PUT': [],
            'phase': data.get('phase', 'unknown'),
            'regime': data.get('regime', 'unknown'),
            'time_to_expiry': data.get('time_to_expiry_seconds', 900),
        }

        quotes_data = data.get('quotes', {})

        for asset_type in ['CALL', 'PUT']:
            asset_quotes = quotes_data.get(asset_type, [])
            for q in asset_quotes:
                result[asset_type].append(MMQuote(
                    asset_type=asset_type,
                    side=q.get('side'),
                    price=q.get('price'),
                    size=q.get('size'),
                    layer=q.get('layer', 0),
                ))

        return result

    except Exception as e:
        logger.error(f"❌ Error reading MM quotes: {e}")
        return None


def get_mm_quote_for_side(quotes: Dict, asset_type: str, side: str, layer: int = 0) -> Optional[MMQuote]:
    """Get specific quote from MM engine output"""
    if not quotes or asset_type not in quotes:
        return None

    for q in quotes[asset_type]:
        if q.side == side and q.layer == layer:
            return q

    return None


class MarketMakerBotAdvanced:
    """Market maker with advanced MM engine integration"""

    def __init__(self, credentials: dict):
        """Initialize market maker bot"""
        self.trader = PolymarketTrader(
            clob_api_url=credentials['clob_api_url'],
            private_key=credentials['private_key'],
            api_key=credentials['api_key'],
            api_secret=credentials['api_secret'],
            api_passphrase=credentials['api_passphrase']
        )

        # Position tracking
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

        # Price history
        self.btc_price_history = deque(maxlen=600)

        # Period tracking
        self.current_period_start: Optional[int] = None
        self.strike_price: Optional[float] = None
        self.last_strike_check = 0
        self.buffer_check_done = False
        self.new_period_initialized = False

        # Last known prices
        self.last_btc_price: Optional[float] = None
        self.last_call_ask: Optional[float] = None
        self.last_call_bid: Optional[float] = None
        self.last_put_ask: Optional[float] = None
        self.last_put_bid: Optional[float] = None

        # Last buy order prices
        self.last_call_buy_price: float = 0.50
        self.last_put_buy_price: float = 0.50

        # Timing
        self.last_position_check = time.time()
        self.last_asset_reload = time.time()
        self.last_cancel_time = 0

        # USDC balance cache
        self.cached_usdc_balance = 0
        self.last_usdc_check = 0

        # MM Engine tracking
        self.mm_quotes_used = 0
        self.mm_quotes_fallback = 0
        self.last_mm_status = {}

        self.print_startup_info()

    def print_startup_info(self):
        """Print startup information"""
        print("=" * 70)
        print("BOT016_MM_ADVANCED - With MM Engine Integration")
        print(f"MM Engine: {'ENABLED' if USE_MM_ENGINE else 'DISABLED'}")
        print(f"MM Quotes File: {MM_QUOTES_FILE}")
        print(f"Max Shares: {MAX_SHARES_PER_SIDE}/side | Min Profit (60s): ${MIN_PROFIT_MARGIN:.2f}")
        print("=" * 70)

    def get_today_filename(self) -> Path:
        """Get filename for today's trades"""
        today = date.today().strftime('%Y%m%d')
        return self.trades_dir / f"mm_advanced_trades_{today}.json"

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
            return 0.50

        total_qty = sum(p.quantity for p in positions)
        if total_qty == 0:
            return 0.50

        weighted_sum = sum(p.entry_price * p.quantity for p in positions)
        return weighted_sum / total_qty

    def get_earliest_entry_time(self, token_type: str) -> float:
        """Get earliest entry time for a token type"""
        positions = [p for p in self.positions if p.token_type == token_type]
        if not positions:
            return 0.0
        return min(p.entry_time for p in positions)

    def verify_positions_from_wallet(self):
        """Verify positions against wallet balances"""
        try:
            call_balance = 0
            if self.current_call_id:
                call_balance = self.check_token_balance(self.current_call_id)
                call_positions = [p for p in self.positions if p.token_type == 'CALL']
                tracked_call = sum(p.quantity for p in call_positions)

                if abs(call_balance - tracked_call) > 0.5:
                    print(f"🔄 CALL wallet sync: {tracked_call:.0f} → {call_balance:.0f}")

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
        """Place limit buy order"""
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

                if token_type == 'CALL':
                    self.last_call_buy_price = price
                elif token_type == 'PUT':
                    self.last_put_buy_price = price

                print(f"BUY  {token_type:4} | {quantity:.0f} @ ${price:.2f}")
                return order_id

        except Exception as e:
            logger.error(f"❌ Error placing buy order: {e}")
        return None

    def place_sell_order(self, token_type: str, token_id: str, price: float, quantity: float, entry_price: float, entry_time: float,tte:float) -> Optional[str]:
        """Place limit sell order with gain protection"""
        try:
            seconds_since_buy = time.time() - entry_time

            if (seconds_since_buy <= 90 and tte > 600) or (seconds_since_buy <= 60 and tte > 360) or (seconds_since_buy <= 30 and tte > 180) or (seconds_since_buy <= 15 and tte > 90) :
                min_sell_price = entry_price + MIN_PROFIT_MARGIN
                if price < min_sell_price:
                    print(f"⏳ HOLD {token_type:4} | sell ${price:.2f} < entry ${entry_price:.2f} + ${MIN_PROFIT_MARGIN:.2f} = ${min_sell_price:.2f} ({seconds_since_buy:.0f}s)")
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

    def should_reposition_for_mm_quotes(self, mm_quotes: Dict) -> bool:
        """
        Check if we should reposition based on MM engine recommendations.
        Returns True if quote prices have changed significantly.
        """
        if not mm_quotes or not self.open_orders:
            return False

        for order in self.open_orders:
            # Get corresponding MM quote
            mm_quote = get_mm_quote_for_side(
                mm_quotes,
                order.token_type,
                order.side.lower(),  # Convert 'BUY' -> 'bid', 'SELL' -> 'ask'
                layer=0
            )

            if order.side == 'BUY':
                mm_quote = get_mm_quote_for_side(mm_quotes, order.token_type, 'bid', layer=0)
            else:
                mm_quote = get_mm_quote_for_side(mm_quotes, order.token_type, 'ask', layer=0)

            if mm_quote:
                price_diff = abs(order.price - mm_quote.price)
                if price_diff >= 0.02:  # Reposition if price differs by 2 cents or more
                    logger.info(f"🔄 MM reposition: {order.token_type} {order.side} ${order.price:.2f} -> ${mm_quote.price:.2f}")
                    return True

        return False

    def update_market_making_orders_with_mm(self, mm_quotes: Dict, call_bid: float, call_ask: float, put_bid: float, put_ask: float):
        """
        Update market making orders using MM engine recommendations.
        Uses Layer 0 (first tier) quotes only.
        """
        try:
            # Calculate position counts and entry prices
            call_positions = [p for p in self.positions if p.token_type == 'CALL']
            put_positions = [p for p in self.positions if p.token_type == 'PUT']

            call_shares = sum(p.quantity for p in call_positions)
            put_shares = sum(p.quantity for p in put_positions)

            call_entry = self.get_average_entry_price('CALL')
            call_entry_time = self.get_earliest_entry_time('CALL')
            put_entry = self.get_average_entry_price('PUT')
            put_entry_time = self.get_earliest_entry_time('PUT')

            # Get MM engine status
            phase = mm_quotes.get('phase', '?')
            regime = mm_quotes.get('regime', '?')
            tte = mm_quotes.get('time_to_expiry', 0)

            # Get all MM quotes for display
            call_bid_quote = get_mm_quote_for_side(mm_quotes, 'CALL', 'bid', layer=0)
            call_ask_quote = get_mm_quote_for_side(mm_quotes, 'CALL', 'ask', layer=0)
            put_bid_quote = get_mm_quote_for_side(mm_quotes, 'PUT', 'bid', layer=0)
            put_ask_quote = get_mm_quote_for_side(mm_quotes, 'PUT', 'ask', layer=0)

            # ========== QUOTE MONITOR DISPLAY ==========
            print(f"\n{'='*70}")
            print(f"  MM ENGINE QUOTES | Phase: {phase.upper()} | Regime: {regime.upper()} | TTX: {tte:.0f}s")
            print(f"{'='*70}")
            print(f"  {'ASSET':<6} | {'SIDE':<6} | {'MM PRICE':>10} | {'MARKET':>10} | {'DIFF':>8} | {'STATUS'}")
            print(f"  {'-'*64}")

            # CALL BID
            mm_call_bid = call_bid_quote.price if call_bid_quote else None
            if mm_call_bid:
                diff = mm_call_bid - call_bid
                status = "✓" if mm_call_bid < call_ask else "⚠️ >=ASK"
                print(f"  {'CALL':<6} | {'BID':<6} | ${mm_call_bid:>9.2f} | ${call_bid:>9.2f} | {diff:>+7.2f} | {status}")
            else:
                print(f"  {'CALL':<6} | {'BID':<6} | {'--':>10} | ${call_bid:>9.2f} | {'--':>8} | NO QUOTE")

            # CALL ASK
            mm_call_ask = call_ask_quote.price if call_ask_quote else None
            if mm_call_ask:
                diff = mm_call_ask - call_ask
                status = "✓" if mm_call_ask > call_bid else "⚠️ <=BID"
                print(f"  {'CALL':<6} | {'ASK':<6} | ${mm_call_ask:>9.2f} | ${call_ask:>9.2f} | {diff:>+7.2f} | {status}")
            else:
                print(f"  {'CALL':<6} | {'ASK':<6} | {'--':>10} | ${call_ask:>9.2f} | {'--':>8} | NO QUOTE")

            # PUT BID
            mm_put_bid = put_bid_quote.price if put_bid_quote else None
            if mm_put_bid:
                diff = mm_put_bid - put_bid
                status = "✓" if mm_put_bid < put_ask else "⚠️ >=ASK"
                print(f"  {'PUT':<6} | {'BID':<6} | ${mm_put_bid:>9.2f} | ${put_bid:>9.2f} | {diff:>+7.2f} | {status}")
            else:
                print(f"  {'PUT':<6} | {'BID':<6} | {'--':>10} | ${put_bid:>9.2f} | {'--':>8} | NO QUOTE")

            # PUT ASK
            mm_put_ask = put_ask_quote.price if put_ask_quote else None
            if mm_put_ask:
                diff = mm_put_ask - put_ask
                status = "✓" if mm_put_ask > put_bid else "⚠️ <=BID"
                print(f"  {'PUT':<6} | {'ASK':<6} | ${mm_put_ask:>9.2f} | ${put_ask:>9.2f} | {diff:>+7.2f} | {status}")
            else:
                print(f"  {'PUT':<6} | {'ASK':<6} | {'--':>10} | ${put_ask:>9.2f} | {'--':>8} | NO QUOTE")

            print(f"  {'-'*64}")
            print(f"  [POS] CALL={call_shares:.0f}/{MAX_SHARES_PER_SIDE} (entry ${call_entry:.2f}) | "
                  f"PUT={put_shares:.0f}/{MAX_SHARES_PER_SIDE} (entry ${put_entry:.2f})")
            print(f"{'='*70}")

            # Check if MM engine is recommending to pull quotes
            call_quotes = mm_quotes.get('CALL', [])
            put_quotes = mm_quotes.get('PUT', [])

            if not call_quotes and not put_quotes:
                print("⚠️ MM ENGINE: PULL ALL QUOTES - Not placing any orders")
                self.mm_quotes_used += 1
                return

            # === CALL ORDERS ===
            # Buy order
            room_for_call = MAX_SHARES_PER_SIDE - call_shares
            if room_for_call >= 5 and call_bid_quote:
                buy_price = call_bid_quote.price
                buy_price = max(0.01, min(0.99, round(buy_price, 2)))
                buy_quantity = max(5, min(6, room_for_call))

                # Sanity check: don't buy above current ask
                if buy_price < call_ask:
                    self.place_buy_order('CALL', self.current_call_id, buy_price, buy_quantity)
                else:
                    print(f"⚠️ CALL MM buy ${buy_price:.2f} >= ask ${call_ask:.2f}, skipping")
            elif room_for_call >= 5 and not call_bid_quote:
                # MM engine not providing bid quote - use market bid as fallback
                if FALLBACK_TO_SIMPLE:
                    buy_price = max(0.01, round(call_bid, 2))
                    buy_quantity = max(5, min(6, room_for_call))
                    print(f"[FALLBACK] CALL BUY @ ${buy_price:.2f}")
                    self.place_buy_order('CALL', self.current_call_id, buy_price, buy_quantity)
                    self.mm_quotes_fallback += 1

            # Sell order
            if call_shares >= 5 and call_ask_quote:
                sell_price = call_ask_quote.price
                sell_price = max(0.01, min(0.99, round(sell_price, 2)))

                # Sanity check: don't sell below current bid
                if sell_price > call_bid:
                    actual_call_balance = self.check_token_balance(self.current_call_id) if self.current_call_id else 0
                    if actual_call_balance >= 5:
                        sell_quantity = min(int(actual_call_balance), 6)
                        self.place_sell_order('CALL', self.current_call_id, sell_price, sell_quantity, call_entry, call_entry_time,tte)
                else:
                    print(f"⚠️ CALL MM sell ${sell_price:.2f} <= bid ${call_bid:.2f}, skipping")
            elif call_shares >= 5 and not call_ask_quote:
                # MM engine not providing ask quote - might be intentional (hold)
                print(f"[MM] CALL: No ask quote (holding position)")

            # === PUT ORDERS ===
            # Buy order
            room_for_put = MAX_SHARES_PER_SIDE - put_shares
            if room_for_put >= 5 and put_bid_quote:
                buy_price = put_bid_quote.price
                buy_price = max(0.01, min(0.99, round(buy_price, 2)))
                buy_quantity = max(5, min(6, room_for_put))

                if buy_price < put_ask:
                    self.place_buy_order('PUT', self.current_put_id, buy_price, buy_quantity)
                else:
                    print(f"⚠️ PUT MM buy ${buy_price:.2f} >= ask ${put_ask:.2f}, skipping")
            elif room_for_put >= 5 and not put_bid_quote:
                if FALLBACK_TO_SIMPLE:
                    buy_price = max(0.01, round(put_bid, 2))
                    buy_quantity = max(5, min(6, room_for_put))
                    print(f"[FALLBACK] PUT BUY @ ${buy_price:.2f}")
                    self.place_buy_order('PUT', self.current_put_id, buy_price, buy_quantity)
                    self.mm_quotes_fallback += 1

            # Sell order
            if put_shares >= 5 and put_ask_quote:
                sell_price = put_ask_quote.price
                sell_price = max(0.01, min(0.99, round(sell_price, 2)))

                if sell_price > put_bid:
                    actual_put_balance = self.check_token_balance(self.current_put_id) if self.current_put_id else 0
                    if actual_put_balance >= 5:
                        sell_quantity = min(int(actual_put_balance), 6)
                        print(f"[MM] PUT SELL @ ${sell_price:.2f} (market ask=${put_ask:.2f})")
                        self.place_sell_order('PUT', self.current_put_id, sell_price, sell_quantity, put_entry, put_entry_time,tte)
                else:
                    print(f"⚠️ PUT MM sell ${sell_price:.2f} <= bid ${put_bid:.2f}, skipping")
            elif put_shares >= 5 and not put_ask_quote:
                print(f"[MM] PUT: No ask quote (holding position)")

            self.mm_quotes_used += 1

        except Exception as e:
            logger.error(f"❌ Error updating MM orders: {e}")
            import traceback
            traceback.print_exc()

    def update_market_making_orders_simple(self, call_bid: float, call_ask: float, put_bid: float, put_ask: float):
        """
        Fallback: Simple market making at bid/ask (original logic).
        Used when MM engine quotes unavailable.
        """
        try:
            call_positions = [p for p in self.positions if p.token_type == 'CALL']
            put_positions = [p for p in self.positions if p.token_type == 'PUT']

            call_shares = sum(p.quantity for p in call_positions)
            put_shares = sum(p.quantity for p in put_positions)

            call_entry = self.get_average_entry_price('CALL')
            call_entry_time = self.get_earliest_entry_time('CALL')
            put_entry = self.get_average_entry_price('PUT')
            put_entry_time = self.get_earliest_entry_time('PUT')

            print(f"\n[SIMPLE] CALL={call_shares:.0f}/{MAX_SHARES_PER_SIDE} | PUT={put_shares:.0f}/{MAX_SHARES_PER_SIDE}")

            # === CALL ORDERS ===
            room_for_call = MAX_SHARES_PER_SIDE - call_shares
            if room_for_call >= 5:
                buy_price = max(0.01, round(call_bid, 2))
                buy_quantity = max(5, min(6, room_for_call))
                if 0.01 <= buy_price <= 0.99:
                    self.place_buy_order('CALL', self.current_call_id, buy_price, buy_quantity)

            if call_shares >= 5:
                sell_price = min(0.99, round(call_ask, 2))
                actual_call_balance = self.check_token_balance(self.current_call_id) if self.current_call_id else 0
                if actual_call_balance >= 5:
                    sell_quantity = min(int(actual_call_balance), 6)
                    if 0.01 <= sell_price <= 0.99:
                        self.place_sell_order('CALL', self.current_call_id, sell_price, sell_quantity, call_entry, call_entry_time)

            # === PUT ORDERS ===
            room_for_put = MAX_SHARES_PER_SIDE - put_shares
            if room_for_put >= 5:
                buy_price = max(0.01, round(put_bid, 2))
                buy_quantity = max(5, min(6, room_for_put))
                if 0.01 <= buy_price <= 0.99:
                    self.place_buy_order('PUT', self.current_put_id, buy_price, buy_quantity)

            if put_shares >= 5:
                sell_price = min(0.99, round(put_ask, 2))
                actual_put_balance = self.check_token_balance(self.current_put_id) if self.current_put_id else 0
                if actual_put_balance >= 5:
                    sell_quantity = min(int(actual_put_balance), 6)
                    if 0.01 <= sell_price <= 0.99:
                        self.place_sell_order('PUT', self.current_put_id, sell_price, sell_quantity, put_entry, put_entry_time)

            self.mm_quotes_fallback += 1

        except Exception as e:
            logger.error(f"❌ Error updating simple orders: {e}")

    def save_trade(self, trade: dict):
        """Save a trade to file"""
        try:
            self.today_trades.append(trade)
            filename = self.get_today_filename()
            with open(filename, 'w') as f:
                json.dump(self.today_trades, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Error saving trade: {e}")

    def run(self):
        """Main market making loop"""
        logger.info("\n🚀 Starting ADVANCED market maker bot...")

        self.reload_asset_ids()
        self.refresh_usdc_balance()
        self.update_strike_price()

        while True:
            try:
                now = datetime.now()

                # Determine period
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

                    if self.positions:
                        call_shares = sum(p.quantity for p in self.positions if p.token_type == 'CALL')
                        put_shares = sum(p.quantity for p in self.positions if p.token_type == 'PUT')
                        print(f"🔄 Period ended - clearing OLD positions: CALL={call_shares:.0f} PUT={put_shares:.0f}")
                        self.positions = []

                    self.last_call_buy_price = 0.50
                    self.last_put_buy_price = 0.50

                    self.current_period_start = period_start
                    self.buffer_check_done = False
                    self.new_period_initialized = False

                    self.last_btc_price = None
                    self.last_call_ask = None
                    self.last_call_bid = None
                    self.last_put_ask = None
                    self.last_put_bid = None

                # POST-BUFFER INITIALIZATION
                if not in_buffer_zone and not self.new_period_initialized:
                    print(f"✅ Buffer ended - initializing new period...")
                    self.reload_asset_ids()
                    self.update_strike_price()
                    self.verify_positions_from_wallet()
                    self.new_period_initialized = True
                    self.buffer_check_done = True
                    print(f"✅ New period ready - CALL: {self.current_call_id[-12:] if self.current_call_id else 'None'}")

                # Periodic maintenance
                if time.time() - self.last_asset_reload >= 60:
                    self.reload_asset_ids()

                if time.time() - self.last_strike_check >= 60:
                    self.update_strike_price()

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
                    time.sleep(0.01)
                    continue

                btc_price = btc_data.get('price', 0)
                self.btc_price_history.append(btc_price)

                call_bid = call_data.get('best_bid', 0)
                call_ask = call_data.get('best_ask', 0)
                put_bid = put_data.get('best_bid', 0)
                put_ask = put_data.get('best_ask', 0)

                if not all([btc_price, call_bid, call_ask, put_bid, put_ask]):
                    time.sleep(0.01)
                    continue

                # Read MM engine quotes
                mm_quotes = None
                if USE_MM_ENGINE:
                    mm_quotes = read_mm_quotes()

                # Check if we need to reposition based on MM quotes
                if mm_quotes and self.open_orders:
                    if self.should_reposition_for_mm_quotes(mm_quotes):
                        logger.info("🔄 Repositioning based on MM engine...")
                        self.cancel_all_orders()
                        self.last_cancel_time = time.time()

                # Update tracking
                self.last_btc_price = btc_price
                self.last_call_ask = call_ask
                self.last_call_bid = call_bid
                self.last_put_ask = put_ask
                self.last_put_bid = put_bid

                # Place orders
                can_place_orders = (not in_buffer_zone and
                                   len(self.open_orders) == 0 and
                                   time.time() - self.last_cancel_time >= REPOSITION_DELAY)

                if can_place_orders and self.current_call_id and self.current_put_id:
                    if mm_quotes:
                        # Use MM engine recommendations
                        self.update_market_making_orders_with_mm(mm_quotes, call_bid, call_ask, put_bid, put_ask)
                    elif FALLBACK_TO_SIMPLE:
                        # Fallback to simple bid/ask strategy
                        print("⚠️ MM quotes unavailable - NOT using SIMPLE strategy")
                        #self.update_market_making_orders_simple(call_bid, call_ask, put_bid, put_ask)

                time.sleep(0.01)

            except KeyboardInterrupt:
                logger.info("\n\n⏸️  Shutting down...")
                self.cancel_all_orders()
                print(f"\n📊 Session stats: MM quotes used={self.mm_quotes_used}, Fallbacks={self.mm_quotes_fallback}")
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

        bot = MarketMakerBotAdvanced(credentials)
        bot.run()

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
