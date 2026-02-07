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
SHARES_PER_POSITION = 6  # 6 shares per position
MAX_POSITIONS_PER_SIDE = 3  # Maximum 3 positions per side (CALL/PUT)
MAX_SHARES_PER_SIDE = SHARES_PER_POSITION * MAX_POSITIONS_PER_SIDE  # Total 18 shares
MIN_PROFIT_MARGIN = 0.01  # Minimum profit per share to sell ($0.01)
MIN_POSITION_DISTANCE = 0.2  # Minimum $0.20 between position entry prices
BUFFER_SECONDS = 10  # No trading in first 10s of period
POSITION_CHECK_INTERVAL = 1  # Check positions every 1s
REPOSITION_DELAY = 2.0  # Wait 2 seconds after cancel before repositioning


@dataclass
class Position:
    """Individual position tracker - each position is distinct with its own entry price"""
    position_id: int  # 1, 2, or 3 for each side
    token_type: str  # 'PUT' or 'CALL'
    token_id: str
    entry_price: float  # CRITICAL: actual buy price for gain-only logic
    entry_time: float
    quantity: float  # Should be SHARES_PER_POSITION (6)


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
        print("BOT016_MM_GAIN_SAFE - Gain Only + Multi-Position")
        print(f"Positions: {MAX_POSITIONS_PER_SIDE} x {SHARES_PER_POSITION} shares = {MAX_SHARES_PER_SIDE} max/side")
        print(f"Min distance between positions: ${MIN_POSITION_DISTANCE:.2f}")
        print(f"Sell: ONLY when price > entry + ${MIN_PROFIT_MARGIN:.2f}")
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
        """Verify positions against wallet balances - reconcile positions with actual tokens"""
        try:
            # Check CALL balance
            call_balance = 0
            if self.current_call_id:
                call_balance = self.check_token_balance(self.current_call_id)
                call_positions = [p for p in self.positions if p.token_type == 'CALL']
                tracked_call = sum(p.quantity for p in call_positions)
                num_positions = len(call_positions)

                # Calculate expected positions based on actual balance
                expected_positions = int(call_balance / SHARES_PER_POSITION) if call_balance >= 5 else 0

                # CASE 1: Have positions but NO tokens - clear all positions
                if num_positions > 0 and call_balance < 5:
                    print(f"⚠️ CALL: {num_positions} positions but only {call_balance:.1f} tokens - clearing all")
                    self.positions = [p for p in self.positions if p.token_type != 'CALL']

                # CASE 2: More positions than tokens support - remove furthest from current price
                elif num_positions > expected_positions and expected_positions >= 0:
                    print(f"🔄 CALL: {num_positions} positions but wallet supports {expected_positions} - adjusting")
                    # Get current market price to determine which positions to keep
                    current_price = self.last_call_bid if self.last_call_bid else 0.50

                    # Sort by distance from current price (keep closest)
                    sorted_by_distance = sorted(call_positions, key=lambda p: abs(p.entry_price - current_price))

                    # Keep only the positions closest to current price
                    positions_to_keep = sorted_by_distance[:expected_positions]
                    positions_to_remove = sorted_by_distance[expected_positions:]

                    for p in positions_to_remove:
                        self.positions.remove(p)
                        print(f"   → Removed CALL P{p.position_id} @ ${p.entry_price:.2f} (far from current ${current_price:.2f})")

                    # Adjust remaining position quantities to match wallet
                    remaining_balance = call_balance
                    for p in positions_to_keep:
                        p.quantity = min(SHARES_PER_POSITION, remaining_balance)
                        remaining_balance -= p.quantity

                # CASE 3: Fewer positions than tokens - create new positions at current price
                elif expected_positions > num_positions and call_balance >= 5:
                    print(f"🔄 CALL: {num_positions} positions but wallet has {call_balance:.0f} tokens ({expected_positions} expected)")
                    current_price = self.last_call_buy_price if self.last_call_buy_price != 0.50 else (self.last_call_bid if self.last_call_bid else 0.50)

                    positions_to_add = expected_positions - num_positions
                    remaining = call_balance - tracked_call

                    for i in range(positions_to_add):
                        if remaining >= 5:
                            next_id = self.get_next_position_id('CALL')
                            if next_id > 0:
                                qty = min(SHARES_PER_POSITION, remaining)
                                self.positions.append(Position(
                                    position_id=next_id,
                                    token_type='CALL',
                                    token_id=self.current_call_id,
                                    entry_price=current_price,
                                    entry_time=time.time(),
                                    quantity=qty
                                ))
                                print(f"   → Added CALL P{next_id}: {qty:.0f} shares @ ${current_price:.2f} (assumed)")
                                remaining -= qty

            # Check PUT balance
            put_balance = 0
            if self.current_put_id:
                put_balance = self.check_token_balance(self.current_put_id)
                put_positions = [p for p in self.positions if p.token_type == 'PUT']
                tracked_put = sum(p.quantity for p in put_positions)
                num_positions = len(put_positions)

                expected_positions = int(put_balance / SHARES_PER_POSITION) if put_balance >= 5 else 0

                # CASE 1: Have positions but NO tokens
                if num_positions > 0 and put_balance < 5:
                    print(f"⚠️ PUT: {num_positions} positions but only {put_balance:.1f} tokens - clearing all")
                    self.positions = [p for p in self.positions if p.token_type != 'PUT']

                # CASE 2: More positions than tokens support
                elif num_positions > expected_positions and expected_positions >= 0:
                    print(f"🔄 PUT: {num_positions} positions but wallet supports {expected_positions} - adjusting")
                    current_price = self.last_put_bid if self.last_put_bid else 0.50

                    sorted_by_distance = sorted(put_positions, key=lambda p: abs(p.entry_price - current_price))
                    positions_to_keep = sorted_by_distance[:expected_positions]
                    positions_to_remove = sorted_by_distance[expected_positions:]

                    for p in positions_to_remove:
                        self.positions.remove(p)
                        print(f"   → Removed PUT P{p.position_id} @ ${p.entry_price:.2f} (far from current ${current_price:.2f})")

                    remaining_balance = put_balance
                    for p in positions_to_keep:
                        p.quantity = min(SHARES_PER_POSITION, remaining_balance)
                        remaining_balance -= p.quantity

                # CASE 3: Fewer positions than tokens
                elif expected_positions > num_positions and put_balance >= 5:
                    print(f"🔄 PUT: {num_positions} positions but wallet has {put_balance:.0f} tokens ({expected_positions} expected)")
                    current_price = self.last_put_buy_price if self.last_put_buy_price != 0.50 else (self.last_put_bid if self.last_put_bid else 0.50)

                    positions_to_add = expected_positions - num_positions
                    remaining = put_balance - tracked_put

                    for i in range(positions_to_add):
                        if remaining >= 5:
                            next_id = self.get_next_position_id('PUT')
                            if next_id > 0:
                                qty = min(SHARES_PER_POSITION, remaining)
                                self.positions.append(Position(
                                    position_id=next_id,
                                    token_type='PUT',
                                    token_id=self.current_put_id,
                                    entry_price=current_price,
                                    entry_time=time.time(),
                                    quantity=qty
                                ))
                                print(f"   → Added PUT P{next_id}: {qty:.0f} shares @ ${current_price:.2f} (assumed)")
                                remaining -= qty

            # Print wallet status
            call_pos = len([p for p in self.positions if p.token_type == 'CALL'])
            put_pos = len([p for p in self.positions if p.token_type == 'PUT'])
            print(f"[WALLET] CALL={call_balance:.0f} ({call_pos} pos) | PUT={put_balance:.0f} ({put_pos} pos)")

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
        """Place limit buy order - creates a new distinct position"""
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

                # Track entry price for this new position
                if token_type == 'CALL':
                    self.last_call_buy_price = price
                elif token_type == 'PUT':
                    self.last_put_buy_price = price

                # Create a new distinct position with its own entry price
                next_id = self.get_next_position_id(token_type)
                if next_id > 0:
                    self.positions.append(Position(
                        position_id=next_id,
                        token_type=token_type,
                        token_id=token_id,
                        entry_price=price,
                        entry_time=time.time(),
                        quantity=quantity
                    ))

                print(f"BUY  {token_type:4} P{next_id} | {quantity:.0f} @ ${price:.2f}")
                return order_id

        except Exception as e:
            logger.error(f"❌ Error placing buy order: {e}")
        return None

    def place_sell_order(self, token_type: str, token_id: str, price: float, quantity: float, entry_price: float, position_id: int = 0) -> Optional[str]:
        """Place limit sell order - ONLY if profitable, removes position on success"""
        try:
            # GAIN-ONLY CHECK: Only sell if price > entry_price + min_profit
            min_sell_price = entry_price + MIN_PROFIT_MARGIN

            if price < min_sell_price:
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

                # Remove the position that was sold
                if position_id > 0:
                    self.positions = [p for p in self.positions
                                     if not (p.token_type == token_type and p.position_id == position_id)]

                # Save trade
                self.save_trade({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'SELL',
                    'token_type': token_type,
                    'position_id': position_id,
                    'quantity': quantity,
                    'price': price,
                    'entry_price': entry_price,
                    'profit_per_share': profit_per_share,
                    'total_profit': profit_per_share * quantity
                })

                print(f"SELL {token_type:4} P{position_id} | {quantity:.0f} @ ${price:.2f} (entry ${entry_price:.2f}, profit +${profit_per_share:.2f}/share)")
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

    def can_open_new_position(self, token_type: str, new_price: float) -> bool:
        """Check if we can open a new position at this price (distance check)"""
        positions = [p for p in self.positions if p.token_type == token_type]

        # Check if any existing position is within MIN_POSITION_DISTANCE
        for p in positions:
            distance = abs(new_price - p.entry_price)
            if distance < MIN_POSITION_DISTANCE:
                return False
        return True

    def get_next_position_id(self, token_type: str) -> int:
        """Get the next available position ID (1, 2, or 3)"""
        positions = [p for p in self.positions if p.token_type == token_type]
        used_ids = {p.position_id for p in positions}
        for i in range(1, MAX_POSITIONS_PER_SIDE + 1):
            if i not in used_ids:
                return i
        return 0  # No available slot

    def get_position_for_sell(self, token_type: str, sell_price: float) -> Optional[Position]:
        """Find a position that can be sold profitably at this price"""
        positions = [p for p in self.positions if p.token_type == token_type]

        # Find position with lowest entry price that can be sold profitably
        profitable_positions = [p for p in positions if sell_price >= p.entry_price + MIN_PROFIT_MARGIN]

        if profitable_positions:
            # Return the one with lowest entry price (highest profit)
            return min(profitable_positions, key=lambda p: p.entry_price)
        return None

    def update_market_making_orders(self, call_bid: float, call_ask: float, put_bid: float, put_ask: float, btc_delta: float = 0):
        """Update market making orders - Multi-position GAIN ONLY logic"""
        try:
            # Get positions for each side
            call_positions = [p for p in self.positions if p.token_type == 'CALL']
            put_positions = [p for p in self.positions if p.token_type == 'PUT']

            call_count = len(call_positions)
            put_count = len(put_positions)

            call_shares = sum(p.quantity for p in call_positions)
            put_shares = sum(p.quantity for p in put_positions)

            # Print positions with individual entry prices
            call_entries = ", ".join([f"P{p.position_id}=${p.entry_price:.2f}" for p in sorted(call_positions, key=lambda x: x.position_id)])
            put_entries = ", ".join([f"P{p.position_id}=${p.entry_price:.2f}" for p in sorted(put_positions, key=lambda x: x.position_id)])

            print(f"\n[POS] CALL: {call_count}/{MAX_POSITIONS_PER_SIDE} [{call_entries or 'none'}] | "
                  f"PUT: {put_count}/{MAX_POSITIONS_PER_SIDE} [{put_entries or 'none'}] | BTC_Δ=${btc_delta:+.2f}")

            # === EMERGENCY SELL EXCESS (over MAX_SHARES_PER_SIDE total) ===
            if call_shares > MAX_SHARES_PER_SIDE:
                excess_call = call_shares - MAX_SHARES_PER_SIDE
                if excess_call >= 5:
                    self.emergency_sell_excess('CALL', self.current_call_id, excess_call, call_bid)
                    return

            if put_shares > MAX_SHARES_PER_SIDE:
                excess_put = put_shares - MAX_SHARES_PER_SIDE
                if excess_put >= 5:
                    self.emergency_sell_excess('PUT', self.current_put_id, excess_put, put_bid)
                    return

            # === CALL ORDERS ===
            # BUY: Only open ONE new position if we have room AND price is far enough from existing
            if call_count < MAX_POSITIONS_PER_SIDE:
                buy_price = max(0.01, call_bid - BUY_OFFSET)
                buy_price = round(buy_price, 2)

                if self.can_open_new_position('CALL', buy_price):
                    next_id = self.get_next_position_id('CALL')
                    if next_id > 0 and 0.01 <= buy_price <= 0.99:
                        print(f"📈 CALL P{next_id}: Opening new position @ ${buy_price:.2f}")
                        self.place_buy_order('CALL', self.current_call_id, buy_price, SHARES_PER_POSITION)
                else:
                    # Show why we can't open
                    existing = [f"${p.entry_price:.2f}" for p in call_positions]
                    print(f"⏳ CALL: bid ${buy_price:.2f} too close to existing {existing} (need ${MIN_POSITION_DISTANCE:.2f} distance)")

            # SELL: Check each position individually for profitable exit
            for position in call_positions:
                sell_price = min(0.99, call_ask + SELL_OFFSET)
                sell_price = round(sell_price, 2)

                if sell_price >= position.entry_price + MIN_PROFIT_MARGIN:
                    # Check actual balance before selling
                    actual_balance = self.check_token_balance(self.current_call_id) if self.current_call_id else 0
                    if actual_balance >= 5:
                        sell_qty = min(int(position.quantity), int(actual_balance), 6)
                        if sell_qty >= 5 and 0.01 <= sell_price <= 0.99:
                            profit = sell_price - position.entry_price
                            print(f"💰 CALL P{position.position_id}: Selling @ ${sell_price:.2f} (entry ${position.entry_price:.2f}, +${profit:.2f})")
                            self.place_sell_order('CALL', self.current_call_id, sell_price, sell_qty, position.entry_price, position.position_id)
                            break  # Only place one sell order at a time
                else:
                    print(f"⏳ CALL P{position.position_id}: HOLD (sell ${sell_price:.2f} < entry ${position.entry_price:.2f} + ${MIN_PROFIT_MARGIN:.2f})")

            # === PUT ORDERS ===
            # BUY: Only open ONE new position if we have room AND price is far enough from existing
            if put_count < MAX_POSITIONS_PER_SIDE:
                buy_price = max(0.01, put_bid - BUY_OFFSET)
                buy_price = round(buy_price, 2)

                if self.can_open_new_position('PUT', buy_price):
                    next_id = self.get_next_position_id('PUT')
                    if next_id > 0 and 0.01 <= buy_price <= 0.99:
                        print(f"📉 PUT P{next_id}: Opening new position @ ${buy_price:.2f}")
                        self.place_buy_order('PUT', self.current_put_id, buy_price, SHARES_PER_POSITION)
                else:
                    existing = [f"${p.entry_price:.2f}" for p in put_positions]
                    print(f"⏳ PUT: bid ${buy_price:.2f} too close to existing {existing} (need ${MIN_POSITION_DISTANCE:.2f} distance)")

            # SELL: Check each position individually for profitable exit
            for position in put_positions:
                sell_price = min(0.99, put_ask + SELL_OFFSET)
                sell_price = round(sell_price, 2)

                if sell_price >= position.entry_price + MIN_PROFIT_MARGIN:
                    actual_balance = self.check_token_balance(self.current_put_id) if self.current_put_id else 0
                    if actual_balance >= 5:
                        sell_qty = min(int(position.quantity), int(actual_balance), 6)
                        if sell_qty >= 5 and 0.01 <= sell_price <= 0.99:
                            profit = sell_price - position.entry_price
                            print(f"💰 PUT P{position.position_id}: Selling @ ${sell_price:.2f} (entry ${position.entry_price:.2f}, +${profit:.2f})")
                            self.place_sell_order('PUT', self.current_put_id, sell_price, sell_qty, position.entry_price, position.position_id)
                            break  # Only place one sell order at a time
                else:
                    print(f"⏳ PUT P{position.position_id}: HOLD (sell ${sell_price:.2f} < entry ${position.entry_price:.2f} + ${MIN_PROFIT_MARGIN:.2f})")

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
