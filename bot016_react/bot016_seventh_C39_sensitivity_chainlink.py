#!/usr/bin/env python3
"""
Bot016_Seventh_C39 - Binary Options with Limit TP and Market SL
Uses Coinbase BTC price feed and 15-minute candles for strike price
Take Profit: LIMIT orders at open_price + $0.06 (automatically placed)
Stop Loss: MARKET orders at -$0.09/share loss (won't sell below $0.01)
Buys 6 shares, manages TP orders automatically
Max 12 shares per side (CALL/PUT), must stay balanced
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
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# File paths
PUT_FILE = "/home/ubuntu/013_2025_polymarket/15M_BTC_PUT_rest.json"
CALL_FILE = "/home/ubuntu/013_2025_polymarket/15M_BTC_CALL_rest.json"
BTC_FILE_COINBASE = "/home/ubuntu/013_2025_polymarket/bot016_react/coinbase_btc_price.json"
BTC_FILE_CHAINLINK = "/home/ubuntu/013_2025_polymarket/chainlink_btc_price.json"
SENSITIVITY_FILE = "/home/ubuntu/013_2025_polymarket/bot016_react/sensitivity_data/sensitivity_transformed.json"
TRADES_DIR = "/home/ubuntu/013_2025_polymarket/bot016_react/seventh_c39_trades"

# Credentials
CREDENTIALS_ENV = '/home/ubuntu/013_2025_polymarket/keys/keys_ovh39.env'

# Coinbase API
COINBASE_CANDLES_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"

# Trading parameters
SENS_MULTIPLIER = 1
ACTION_THRESHOLD = 0.035  # Single threshold
ACTION_THRESHOLD_imbalanced = 0.015  # Single threshold
TAKE_PROFIT = 0.99  # Take profit at +$0.06/share (LIMIT order)
STOP_LOSS = 0.99  # Stop loss at -$0.09/share (MARKET order)
MIN_SECONDS_BETWEEN_POSITIONS = 2
BUFFER_SECONDS = 30  # No trading in first 30s of period
POSITION_SIZE = 6  # Buy 6 shares
MAX_SPREAD = 0.03  # Suspend trading if spread > 0.03
MIN_BUY_PRICE = 0.10  # Never buy below this price
MAX_SHARES_PER_SIDE = 6  # Maximum 12 shares per side (CALL or PUT)


@dataclass
class Position:
    """Open position tracker"""
    token_type: str  # 'PUT' or 'CALL'
    token_id: str
    entry_price: float
    entry_time: float
    quantity: float
    entry_btc_price: float
    entry_bin: str
    edge: float
    tp_order_id: Optional[str] = None  # Take profit limit order ID


def read_json(filepath: str) -> Optional[dict]:
    """Read JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return None


def get_bin_key(distance: float, seconds_to_expiry: float, volatility: float) -> str:
    """Get bin key"""
    distance_bins = [
        (0, 1, "0-1"), (1, 5, "1-5"), (5, 10, "5-10"), (10, 20, "10-20"),
        (20, 40, "20-40"), (40, 80, "40-80"), (80, 160, "80-160"),
        (160, 320, "160-320"), (320, 640, "320-640"), (640, 1280, "640-1280"),
        (1280, float('inf'), "1280+")
    ]

    time_bins = [
        (13*60, 15*60, "15m-13m"), (11*60, 13*60, "13m-11m"), (10*60, 11*60, "11m-10m"),
        (9*60, 10*60, "10m-9m"), (8*60, 9*60, "9m-8m"), (7*60, 8*60, "8m-7m"),
        (6*60, 7*60, "7m-6m"), (5*60, 6*60, "6m-5m"), (4*60, 5*60, "5m-4m"),
        (3*60, 4*60, "4m-3m"), (2*60, 3*60, "3m-2m"), (90, 120, "120s-90s"),
        (60, 90, "90s-60s"), (40, 60, "60s-40s"), (30, 40, "40s-30s"),
        (20, 30, "30s-20s"), (10, 20, "20s-10s"), (5, 10, "10s-5s"),
        (2, 5, "5s-2s"), (0, 2, "last-2s")
    ]

    vol_bins = [
        (0, 10, "0-10"), (10, 20, "10-20"), (20, 30, "20-30"), (30, 40, "30-40"),
        (40, 60, "40-60"), (60, 90, "60-90"), (90, 120, "90-120"), (120, 240, "120-240"),
        (240, float('inf'), "240+")
    ]

    def get_bin_label(value, bins):
        for min_val, max_val, label in bins:
            if min_val <= value < max_val:
                return label
        return bins[-1][2]

    dist_label = get_bin_label(distance, distance_bins)
    time_label = get_bin_label(seconds_to_expiry, time_bins)
    vol_label = get_bin_label(volatility, vol_bins)

    return f"{dist_label}|{time_label}|{vol_label}"

"""
def get_strike_price() -> Optional[float]:
    #Get strike price from Coinbase 15-minute candles API
    try:
        from datetime import timezone
        import requests

        now = datetime.now(timezone.utc)
        current_minute = now.minute

        for start_min in [0, 15, 30, 45]:
            if current_minute >= start_min and current_minute < start_min + 15:
                period_start = now.replace(minute=start_min, second=0, microsecond=0)
                # Coinbase uses ISO 8601 format for start parameter
                start_iso = period_start.isoformat()

                params = {
                    'granularity': 900,  # 15 minutes in seconds
                    'start': start_iso
                }

                response = requests.get(COINBASE_CANDLES_URL, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    # Coinbase returns array of candles: [time, low, high, open, close, volume]
                    if data and len(data) > 0:
                        # Use the open price of the current 15-minute candle
                        return float(data[0][3])  # index 3 is 'open'
        return None
    except Exception as e:
        logger.error(f"Error getting Coinbase strike price: {e}")
        return None

"""

def get_seconds_to_expiry() -> float:
    """Get seconds to expiry"""
    now = datetime.now()
    current_minute = now.minute

    for start_min in [0, 15, 30, 45]:
        if current_minute >= start_min and current_minute < start_min + 15:
            seconds_into_period = (current_minute - start_min) * 60 + now.second
            return 900 - seconds_into_period
    return 0


def calculate_btc_volatility(price_history: deque) -> float:
    """Calculate BTC volatility over last minute (price range)"""
    if len(price_history) < 10:
        return 0.0
    prices = list(price_history)
    return max(prices) - min(prices)


class Bot016Third:
    """Production trading bot with Polymarket integration"""

    def __init__(self, credentials: dict):
        # Initialize Polymarket trader
        self.trader = PolymarketTrader(
            clob_api_url=credentials['clob_api_url'],
            private_key=credentials['private_key'],
            api_key=credentials['api_key'],
            api_secret=credentials['api_secret'],
            api_passphrase=credentials['api_passphrase']
        )

        # Position tracking - buy & hold strategy
        self.positions: List[Position] = []  # List of open positions
        self.last_position_close_time = 0

        # Take Profit order tracking (one per side)
        self.call_tp_order_id: Optional[str] = None
        self.put_tp_order_id: Optional[str] = None

        # Asset IDs - track previous to prevent buying expired tokens
        self.current_put_id: Optional[str] = None
        self.current_call_id: Optional[str] = None
        self.previous_put_id: Optional[str] = None
        self.previous_call_id: Optional[str] = None
        self.period_asset_ids_validated: bool = False  # Reset each period

        # Trades
        self.trades_dir = Path(TRADES_DIR)
        self.trades_dir.mkdir(exist_ok=True, parents=True)
        self.today_trades = []
        self.load_today_trades()

        # Load sensitivity API (ML predictor)
        self.sensitivity_api = SensitivityAPI(SENSITIVITY_FILE)

        # Price history
        self.btc_price_history = deque(maxlen=600)

        # Period tracking
        self.strike_price: Optional[float] = None
        self.last_strike_update_minute: Optional[int] = None
        self.current_period_start: Optional[int] = None

        # Buffer tracking
        self.start_buffer_reload_done = False  # Track if we've reloaded at end of start buffer

        # Last valid prices (for period-end closure)
        self.last_call_bid: Optional[float] = None
        self.last_put_bid: Optional[float] = None
        self.last_btc_price: Optional[float] = None

        # Tracking for TP order updates (fallback values)
        self.last_seconds_remaining: float = 900  # Default 15 minutes
        self.last_volatility: float = 1.0  # Default volatility
        self.btc_price_chainlink: float = 0  # Last known chainlink price

        # Last buy tracking for dynamic threshold logic

        # Position verification
        self.last_position_check = time.time()
        self.last_asset_reload = time.time()
        self.last_maintenance_cycle = time.time()  # For 30-second cleanup

        # Cached USDC balance (refresh every 10s)
        self.cached_usdc_balance = 0.0
        self.last_usdc_check = 0.0
        self.usdc_check_interval = 10.0  # 10 seconds

        # Previous spread tracking
        self.prev_call_bid = None
        self.prev_call_ask = None
        self.prev_put_bid = None
        self.prev_put_ask = None

        logger.info("="*80)
        logger.info("🤖 BOT016_SEVENTH_C39 - LIMIT TP & MARKET SL")
        logger.info("="*80)
        logger.info("📡 BTC Price: Coinbase WebSocket")
        logger.info("📡 CALL/PUT Prices: REST API (via JSON files)")
        logger.info("💰 Take Profit: LIMIT orders at entry + $0.06")
        logger.info("🛑 Stop Loss: MARKET orders (won't sell below $0.01)")
        logger.info(f"🔑 Credentials: OVH39")
        logger.info(f"Position Size: {POSITION_SIZE} shares")
        logger.info(f"Max Per Side: {MAX_SHARES_PER_SIDE} shares (balanced CALL/PUT)")
        logger.info(f"Max Spread: ${MAX_SPREAD:.2f} (suspend trading if exceeded)")
        logger.info(f"Sens Multiplier: {SENS_MULTIPLIER}x")
        logger.info(f"Action Threshold: {ACTION_THRESHOLD}")
        logger.info(f"Buffer Zone: {BUFFER_SECONDS}s (first 30s of each period)")
        logger.info(f"🔮 Sensitivity Predictor: {len(self.sensitivity_api.predictor.bins_data)} bins loaded")
        logger.info("="*80)

    def get_today_filename(self) -> Path:
        """Get filename for today's trades"""
        today = date.today().strftime('%Y%m%d')
        return self.trades_dir / f"seventh_trades_{today}.json"

    def load_today_trades(self):
        """Load today's trades if they exist"""
        filename = self.get_today_filename()
        if filename.exists():
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                    self.today_trades = data.get('trades', [])
                    logger.info(f"📂 Loaded {len(self.today_trades)} trades from {filename.name}")
            except:
                self.today_trades = []

    def save_trades(self):
        """Save today's trades to file"""
        filename = self.get_today_filename()

        daily_pnl = sum(t['pnl'] for t in self.today_trades)
        win_count = sum(1 for t in self.today_trades if t['pnl'] > 0)
        loss_count = sum(1 for t in self.today_trades if t['pnl'] < 0)

        data = {
            'date': date.today().isoformat(),
            'total_trades': len(self.today_trades),
            'wins': win_count,
            'losses': loss_count,
            'win_rate': win_count / len(self.today_trades) if self.today_trades else 0,
            'daily_pnl': daily_pnl,
            'trades': self.today_trades
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def get_usdc_balance(self) -> float:
        """Get USDC balance from Polymarket"""
        try:
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

            response = self.trader.client.get_balance_allowance(
                params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            )

            balance_raw = int(response.get('balance', 0))
            balance_usdc = balance_raw / 10**6

            return balance_usdc
        except Exception as e:
            logger.error(f"Error getting USDC balance: {e}")
            return 0.0

    def refresh_usdc_balance(self):
        """Refresh cached USDC balance if needed"""
        if time.time() - self.last_usdc_check >= self.usdc_check_interval:
            self.cached_usdc_balance = self.get_usdc_balance()
            self.last_usdc_check = time.time()
            logger.debug(f"💰 USDC balance refreshed: ${self.cached_usdc_balance:.2f}")

    def check_token_balance(self, token_id: str) -> float:
        """Check balance of specific token"""
        try:
            balance_raw, balance = self.trader.get_token_balance(token_id)
            return balance
        except Exception as e:
            logger.debug(f"Error checking balance for {token_id[:12]}...: {e}")
            return 0.0

    def reload_asset_ids(self):
        """Reload PUT and CALL asset IDs from data files and track previous"""
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
                    # Store previous before updating
                    self.previous_put_id = self.current_put_id
                if call_changed:
                    logger.info(f"   CALL: ...{new_call_id[-12:]}")
                    # Store previous before updating
                    self.previous_call_id = self.current_call_id

                self.current_put_id = new_put_id
                self.current_call_id = new_call_id

            self.last_asset_reload = time.time()

    def validate_new_period_asset_ids(self) -> bool:
        """
        Validate that asset IDs have changed from previous period.
        Returns True if validated (or first period), False if IDs are still from old period.
        """
        if not self.current_put_id or not self.current_call_id:
            logger.warning("⚠️  Cannot validate: Asset IDs not loaded yet")
            return False

        # First period - no previous IDs to compare
        if not self.previous_put_id or not self.previous_call_id:
            logger.info("✅ First period - asset IDs accepted")
            return True

        # Check if IDs have changed from previous period
        put_same = (self.current_put_id == self.previous_put_id)
        call_same = (self.current_call_id == self.previous_call_id)

        if put_same or call_same:
            if put_same:
                logger.error(f"❌ PUT asset ID UNCHANGED from previous period: ...{self.current_put_id[-12:]}")
            if call_same:
                logger.error(f"❌ CALL asset ID UNCHANGED from previous period: ...{self.current_call_id[-12:]}")
            logger.error("🚫 TRADING BLOCKED - Old period tokens detected!")
            return False

        logger.info("✅ Asset IDs validated: different from previous period")
        logger.info(f"   PUT:  ...{self.previous_put_id[-12:] if self.previous_put_id else 'N/A'} → ...{self.current_put_id[-12:]}")
        logger.info(f"   CALL: ...{self.previous_call_id[-12:] if self.previous_call_id else 'N/A'} → ...{self.current_call_id[-12:]}")
        return True

    def verify_position_from_wallet(self):
        """Verify positions match wallet (supports multiple positions) - 5% tolerance"""
        if not self.current_put_id or not self.current_call_id:
            return

        put_balance = self.check_token_balance(self.current_put_id)
        call_balance = self.check_token_balance(self.current_call_id)

        # Count existing positions
        put_positions = [p for p in self.positions if p.token_type == 'PUT']
        call_positions = [p for p in self.positions if p.token_type == 'CALL']

        tracked_put_tokens = sum(p.quantity for p in put_positions)
        tracked_call_tokens = sum(p.quantity for p in call_positions)

        # 5% tolerance for position verification
        put_tolerance = max(0.5, put_balance * 0.05)  # At least 0.5 or 5% of balance
        call_tolerance = max(0.5, call_balance * 0.05)

        # Sync PUT positions if mismatch beyond tolerance
        if abs(put_balance - tracked_put_tokens) > put_tolerance:
            logger.warning(f"⚠️  PUT mismatch beyond 5% tolerance: Wallet={put_balance:.2f}, Tracked={tracked_put_tokens:.2f}")

            # Clear all PUT positions and resync with actual wallet balance
            self.positions = [p for p in self.positions if p.token_type != 'PUT']

            if put_balance >= 0.5:
                # Calculate number of full positions to create
                num_positions = int(put_balance / POSITION_SIZE)
                remainder = put_balance % POSITION_SIZE

                # Resync with actual wallet balance using REST JSON
                put_data = read_json(PUT_FILE)
                if not put_data:
                    logger.warning("DEEP ITM/OTM PUT - cannot sync (data unavailable)")
                else:
                    price = (put_data['best_bid'] + put_data['best_ask']) / 2

                    # Create positions for each POSITION_SIZE chunk
                    for i in range(num_positions):
                        synced_position = Position(
                            token_type='PUT',
                            token_id=self.current_put_id,
                            entry_price=price,
                            entry_time=time.time(),
                            quantity=POSITION_SIZE,
                            entry_btc_price=0,
                            entry_bin="SYNCED",
                            edge=0,



                        )
                        self.positions.append(synced_position)

                    # Create position for remainder if significant
                    if remainder >= 0.5:
                        synced_position = Position(
                            token_type='PUT',
                            token_id=self.current_put_id,
                            entry_price=price,
                            entry_time=time.time(),
                            quantity=remainder,
                            entry_btc_price=0,
                            entry_bin="SYNCED",
                            edge=0,



                        )
                        self.positions.append(synced_position)

                    logger.info(f"✅ Synced PUT: {num_positions} full positions + {remainder:.2f} tokens = {put_balance:.2f} total")
            else:
                logger.info("✅ Cleared PUT positions from tracking")

        # Sync CALL positions if mismatch beyond tolerance
        if abs(call_balance - tracked_call_tokens) > call_tolerance:
            logger.warning(f"⚠️  CALL mismatch beyond 5% tolerance: Wallet={call_balance:.2f}, Tracked={tracked_call_tokens:.2f}")

            # Clear all CALL positions and resync with actual wallet balance
            self.positions = [p for p in self.positions if p.token_type != 'CALL']

            if call_balance >= 0.5:
                # Calculate number of full positions to create
                num_positions = int(call_balance / POSITION_SIZE)
                remainder = call_balance % POSITION_SIZE

                # Resync with actual wallet balance using REST JSON
                call_data = read_json(CALL_FILE)
                if not call_data:
                    logger.warning("DEEP ITM/OTM CALL - cannot sync (data unavailable)")
                else:
                    price = (call_data['best_bid'] + call_data['best_ask']) / 2

                    # Create positions for each POSITION_SIZE chunk
                    for i in range(num_positions):
                        synced_position = Position(
                            token_type='CALL',
                            token_id=self.current_call_id,
                            entry_price=price,
                            entry_time=time.time(),
                            quantity=POSITION_SIZE,
                            entry_btc_price=0,
                            entry_bin="SYNCED",
                            edge=0,



                        )
                        self.positions.append(synced_position)

                    # Create position for remainder if significant
                    if remainder >= 0.5:
                        synced_position = Position(
                            token_type='CALL',
                            token_id=self.current_call_id,
                            entry_price=price,
                            entry_time=time.time(),
                            quantity=remainder,
                            entry_btc_price=0,
                            entry_bin="SYNCED",
                            edge=0,



                        )
                        self.positions.append(synced_position)

                    logger.info(f"✅ Synced CALL: {num_positions} full positions + {remainder:.2f} tokens = {call_balance:.2f} total")
            else:
                logger.info("✅ Cleared CALL positions from tracking")

        self.last_position_check = time.time()

    def execute_buy(self, token_type: str, token_id: str, ask_price: float,
                    btc_price_chainlink: float, bin_key: str, edge: float, reason: str, bid_price: float = None,
                    seconds_remaining: float = None, volatility: float = None) -> bool: # MOD1
        """Execute buy order with stop loss and take profit monitoring"""
        try:
            logger.info(f"\n{'='*70}")
            logger.info(f"🛒 EXECUTING BUY ORDER")
            logger.info(f"{'='*70}")
            logger.info(f"📊 Token: {token_type}")
            logger.info(f"📦 Size: {POSITION_SIZE} shares")
            logger.info(f"💰 Ask Price: ${ask_price:.4f}")
            logger.info(f"📝 Reason: {reason}")

            required = ask_price * POSITION_SIZE
            logger.info(f"💵 Expected Cost: ${required:.2f}")

            # Use cached USDC balance (no API call here for speed)
            cached_balance = self.cached_usdc_balance
            logger.info(f"💰 USDC Balance (cached): ${cached_balance:.2f}")

            MIN_BALANCE = 4.90
            if cached_balance < MIN_BALANCE:
                logger.error(f"❌ INSUFFICIENT BALANCE: ${cached_balance:.2f} < ${MIN_BALANCE:.2f}")
                return False

            if cached_balance < required:
                logger.error(f"❌ Need: ${required:.2f}, Have: ${cached_balance:.2f}")
                return False

            # PHASE 1: Place order (FAST - no retry on error)
            logger.info(f"\n🚀 PHASE 1: Placing Fill-or-Kill buy order...")

            start_time = time.time()

            # Check position balance (cached count for speed - no API calls)
            put_count = sum(1 for p in self.positions if p.token_type == 'PUT')
            call_count = sum(1 for p in self.positions if p.token_type == 'CALL')
            is_imbalanced = abs(put_count - call_count) > 0

            # If imbalanced, add 0.01 to price for faster fill to rebalance
            if is_imbalanced:
                final_price = ask_price + 0.02
                logger.info(f"⚖️  Imbalanced positions (PUT:{put_count} CALL:{call_count}) - adding +$0.01 for faster fill")
            else:
                final_price = ask_price + 0.02
                logger.info(f"✅ Balanced positions (PUT:{put_count} CALL:{call_count}) - using ask price")

            try:
                order_id = self.trader.place_buy_order_FAK(
                    token_id=token_id,
                    price=final_price,
                    quantity=POSITION_SIZE
                )
            except Exception as order_error:
                logger.error(f"❌ Order error: {order_error}")
                return False

            if not order_id:
                logger.error(f"❌ Failed to place order")
                return False

            logger.info(f"✅ Order placed: {order_id[:16]}... ({time.time() - start_time:.3f}s)")

            ##  DOUBLE TAP

            # try:
            #     order_id = self.trader.place_buy_order_FAK(
            #         token_id=token_id,
            #         price=final_price,
            #         quantity=POSITION_SIZE
            #     )
            # except Exception as order_error:
            #     logger.error(f"❌ Order error: {order_error}")
            #     return False
            #
            # if not order_id:
            #     logger.error(f"❌ Failed to place order")
            #     return False

            logger.info(f"✅ Order placed: {order_id[:16]}... ({time.time() - start_time:.3f}s)")


            # Fill-or-Kill: Wait 1 second, then cancel if not filled
            time.sleep(1)

            # Quick check if filled
            balance_1s = self.check_token_balance(token_id)
            if balance_1s < POSITION_SIZE * 0.8:
                logger.warning(f"⚠️  Fill-or-Kill: Order not filled in 1s, canceling...")
                #self.trader.cancel_all_orders()
                #return False
            else:
                logger.info(f"✅ Fill-or-Kill: Order filled in 1s!")

            # PHASE 2: Verify (keep existing verification for confirmation)
            logger.info("\n🔍 PHASE 2: Verifying...")

            verification_times = [4, 9]  # Adjusted since we already waited 1s
            position_confirmed = True  # Already confirmed by Fill-or-Kill

            for wait_time in verification_times:
                time.sleep(wait_time)

                balance = self.check_token_balance(token_id)

                logger.info(f"   {wait_time + 1}s total: Balance {balance:.2f}")

                if balance >= POSITION_SIZE * 0.95:
                    position_confirmed = True
                    logger.info(f"   ✅ Position confirmed #########################")
                    break

            # PHASE 3: Final check
            final_balance = self.check_token_balance(token_id)

            if position_confirmed and final_balance >= 0.5:
                logger.info(f"\n✅ SUCCESS: Position opened ###############################")

                new_position = Position(
                    token_type=token_type,
                    token_id=token_id,
                    entry_price=ask_price,
                    entry_time=time.time(),
                    quantity=final_balance,
                    entry_btc_price=btc_price_chainlink,
                    entry_bin=bin_key,
                    edge=edge
                )

                self.positions.append(new_position)

                put_shares = sum(p.quantity for p in self.positions if p.token_type == 'PUT')
                call_shares = sum(p.quantity for p in self.positions if p.token_type == 'CALL')

                logger.info(f"📊 Position: {final_balance:.2f} @ ${ask_price:.4f}")
                logger.info(f"📦 Total shares: PUT:{put_shares:.0f}/{MAX_SHARES_PER_SIDE} CALL:{call_shares:.0f}/{MAX_SHARES_PER_SIDE}")

                # Update TP limit order for this side
                self.update_tp_order(token_type, current_bid=bid_price, btc_price_chainlink=btc_price_chainlink,
                                   seconds_remaining=seconds_remaining, volatility=volatility)

                return True
            else:
                logger.warning(f"\n⚠️  Position not confirmed")
                self.trader.cancel_all_orders()
                return False

        except Exception as e:
            logger.error(f"❌ Error executing buy: {e}")
            import traceback
            traceback.print_exc()
            return False

    def place_tp_limit_order(self, token_type: str, token_id: str, tp_price: float, quantity: float) -> Optional[str]:
        """Place take profit limit sell order with retry for crossing book"""
        try:
            logger.info(f"\n💰 Placing TP LIMIT order:")
            logger.info(f"   Type: {token_type}")
            logger.info(f"   Price: ${tp_price:.4f}")
            logger.info(f"   Quantity: {quantity:.2f}")
            logger.info(f"   Token ID: ...{token_id[-12:]}")

            order_id = self.trader.place_sell_order_limit(token_id, tp_price, quantity)

            if order_id:
                logger.info(f"   ✅ TP order placed: {order_id[:16]}...")
                return order_id
            else:
                logger.error("   ❌ Failed to place TP order")
                return None

        except Exception as e:
            error_str = str(e)

            # Check if error is "order crosses book" - price surpassed
            if "crosses book" in error_str or "invalid post-only" in error_str:
                logger.warning(f"⚠️  TP order crosses book - price already surpassed!")
                logger.info(f"   → Skipping TP order, position is already profitable")
                # Don't place order - position already hit TP, will be caught by checks
                return None
            else:
                logger.error(f"❌ Error placing TP order: {e}")
                import traceback
                traceback.print_exc()
                return None

    def update_tp_order(self, token_type: str, current_bid: float = None, btc_price_chainlink: float = None,
                        seconds_remaining: float = None, volatility: float = None):
        """Update TP order for a side (cancel old, place new)

        Args:
            token_type: 'CALL' or 'PUT'
            current_bid: Current bid price to check if TP would cross the book
            btc_price_chainlink: Chainlink BTC price for sensitivity
            seconds_remaining: Time remaining in period
            volatility: BTC volatility
        """
        try:
            # Get all positions for this side
            side_positions = [p for p in self.positions if p.token_type == token_type]

            if not side_positions:
                return

            # Calculate total quantity and average entry price
            total_quantity = sum(p.quantity for p in side_positions)
            total_cost = sum(p.entry_price * p.quantity for p in side_positions)
            avg_entry = total_cost / total_quantity if total_quantity > 0 else 0

            # Use provided values or fallback to instance variables
            if btc_price_chainlink is None:
                btc_price_chainlink = self.btc_price_chainlink if hasattr(self, 'btc_price_chainlink') else 0
            if seconds_remaining is None:
                seconds_remaining = self.last_seconds_remaining if hasattr(self, 'last_seconds_remaining') else 900
            if volatility is None:
                volatility = self.last_volatility if hasattr(self, 'last_volatility') else 1.0

            # Get sensitivity prediction from API
            prediction = self.sensitivity_api.get_sensitivity(
                btc_price=btc_price_chainlink,
                strike_price=self.strike_price,
                time_to_expiry_seconds=seconds_remaining,
                volatility_percent=volatility
            )

            call_sens = prediction['sensitivity']['call']
            put_sens = prediction['sensitivity']['put']

            # Calculate TP price and cap at 0.99 (binary option max)
            # Calculate TP price and cap at 0.99 (binary option max)
            adjusted_profit = 0.99
            adjusted_loss = 0.99
            # if adjusted_profit <= 0.07 :
            #     adjusted_profit = min(adjusted_profit, 0.07)
            #     #adjusted_loss = min(adjusted_loss, 0.35)

            tp_price = avg_entry + adjusted_profit

            tp_price = min(tp_price, 0.99)  # Cap at maximum
            tp_price = round(tp_price, 2)  # Round to tick size

            # Skip if entry is already too high for TP
            if avg_entry >= 0.99:
                logger.warning(f"⚠️  {token_type} entry too high (${avg_entry:.4f}) for TP order, skipping")
                return

            # Check if TP price would cross the book (price already surpassed)
            if current_bid and tp_price <= current_bid:
                # TP target already reached - don't place order, let main loop handle it
                logger.warning(f"⚠️  {token_type} TP target (${tp_price:.2f}) already surpassed (bid ${current_bid:.2f})")
                logger.info(f"   → Skipping TP order - position already profitable, will be closed by main loop")

                # Cancel old TP order if exists
                if token_type == 'CALL' and self.call_tp_order_id:
                    logger.info(f"   → Canceling old CALL TP order")
                    self.trader.cancel_order(self.call_tp_order_id)
                    self.call_tp_order_id = None
                elif token_type == 'PUT' and self.put_tp_order_id:
                    logger.info(f"   → Canceling old PUT TP order")
                    self.trader.cancel_order(self.put_tp_order_id)
                    self.put_tp_order_id = None
                return

            # Get token ID and current TP order
            token_id = self.current_call_id if token_type == 'CALL' else self.current_put_id
            current_tp_order = self.call_tp_order_id if token_type == 'CALL' else self.put_tp_order_id

            # Cancel existing TP order if any
            if current_tp_order:
                logger.info(f"🔄 Canceling old {token_type} TP order: {current_tp_order[:16]}...")
                self.trader.cancel_order(current_tp_order)

            # Place new TP order
            new_order_id = self.place_tp_limit_order(token_type, token_id, tp_price, total_quantity)

            if new_order_id:
                # Update tracking
                if token_type == 'CALL':
                    self.call_tp_order_id = new_order_id
                else:
                    self.put_tp_order_id = new_order_id

                logger.info(f"✅ {token_type} TP order updated: entry=${avg_entry:.4f} → TP=${tp_price:.4f} ({total_quantity:.0f} shares)")

        except Exception as e:
            logger.error(f"❌ Error updating TP order: {e}")

    def cancel_all_tp_orders(self):
        """Cancel all TP orders"""
        try:
            if self.call_tp_order_id:
                logger.info(f"❌ Canceling CALL TP order: {self.call_tp_order_id[:16]}...")
                self.trader.cancel_order(self.call_tp_order_id)
                self.call_tp_order_id = None

            if self.put_tp_order_id:
                logger.info(f"❌ Canceling PUT TP order: {self.put_tp_order_id[:16]}...")
                self.trader.cancel_order(self.put_tp_order_id)
                self.put_tp_order_id = None

        except Exception as e:
            logger.error(f"❌ Error canceling TP orders: {e}")



    def can_open_position(self, token_type: str) -> bool:
        """Check if we can open a new position - max 12 shares per side, must stay balanced"""
        if time.time() - self.last_position_close_time < MIN_SECONDS_BETWEEN_POSITIONS:
            return False

        # Count total shares per side
        put_shares = sum(p.quantity for p in self.positions if p.token_type == 'PUT')
        call_shares = sum(p.quantity for p in self.positions if p.token_type == 'CALL')

        # Check max shares per side
        if token_type == 'CALL':
            if call_shares + POSITION_SIZE > MAX_SHARES_PER_SIDE:
                return False
            # Must stay balanced - CALL can't exceed PUT
            return call_shares <= put_shares
        else:  # PUT
            if put_shares + POSITION_SIZE > MAX_SHARES_PER_SIDE:
                return False
            # Must stay balanced - PUT can't exceed CALL
            return put_shares <= call_shares

    def execute_sell(self, token_type: str, token_id: str, bid_price: float,
                     btc_price_chainlink: float, bin_key: str, reason: str) -> bool:
        """Execute sell order - MARKET for stop loss, FAK for others"""
        try:
            # Find all positions for this token type
            matching_positions = [p for p in self.positions if p.token_type == token_type and p.token_id == token_id]

            if not matching_positions:
                logger.warning(f"⚠️  No {token_type} positions to sell")
                return False

            # Get actual balance from wallet
            actual_balance = self.check_token_balance(token_id)

            if actual_balance < 0.5:
                logger.warning(f"⚠️  No {token_type} tokens in wallet")
                # Clear positions from tracking
                self.positions = [p for p in self.positions if not (p.token_type == token_type and p.token_id == token_id)]
                return False

            logger.info(f"\n{'='*70}")
            logger.info(f"💰 EXECUTING SELL - {reason}")
            logger.info(f"{'='*70}")
            logger.info(f"📊 Token: {token_type}")
            logger.info(f"📦 Size: {actual_balance:.2f} shares")
            logger.info(f"💰 Bid: ${bid_price:.4f}")

            # Calculate P&L
            oldest_position = matching_positions[0]
            total_pnl = sum((bid_price - p.entry_price) * p.quantity for p in matching_positions)
            logger.info(f"📈 Total P&L: ${total_pnl:+.2f}")

            # Determine order type and execute
            start_time = time.time()

            if "STOP_LOSS" in reason:
                # Use MARKET order for stop loss
                logger.info(f"🛑 Using MARKET order for stop loss")

                # Check if market price would be too low
                if bid_price < 0.01:
                    logger.warning(f"⚠️  Market price too low (${bid_price:.4f}), keeping position")
                    return False

                from py_clob_client.clob_types import MarketOrderArgs, OrderType

                market_args = MarketOrderArgs(
                    token_id=token_id,
                    amount=actual_balance,
                    price=bid_price,  # Use bid price as reference
                    side="SELL",
                    order_type=OrderType.FOK
                )

                # Create and post the market order
                signed_order = self.trader.client.create_market_order(market_args)
                order_response = self.trader.client.post_order(signed_order)

                # Extract order ID from response
                order_id = order_response.get('orderID') or order_response.get('orderId') if order_response else None

            else:
                # Use FAK for take profit (shouldn't happen - TP is via limit orders)
                logger.info(f"💰 Using FAK order")
                order_id = self.trader.place_sell_order_FAK(
                    token_id=token_id,
                    price=bid_price,
                    quantity=actual_balance
                )

            if not order_id:
                logger.error(f"❌ Failed to place sell order")
                return False

            logger.info(f"✅ Sell order placed: {order_id[:16]}... ({time.time() - start_time:.3f}s)")

            # Wait and verify
            time.sleep(3)
            balance_after = self.check_token_balance(token_id)

            if balance_after < actual_balance * 0.5:  # Sold at least half
                logger.info(f"✅ SELL FILLED - Position CLOSED")

                # Record trade
                trade = {
                    'type': oldest_position.token_type,
                    'open_time': datetime.fromtimestamp(oldest_position.entry_time).isoformat(),
                    'close_time': datetime.now().isoformat(),
                    'open_btc': oldest_position.entry_btc_price,
                    'close_btc': btc_price_chainlink,
                    'open_price': oldest_position.entry_price,
                    'close_price': bid_price,
                    'open_bin': oldest_position.entry_bin,
                    'close_bin': bin_key,
                    'edge': oldest_position.edge,
                    'quantity': actual_balance,
                    'pnl': total_pnl,
                    'close_reason': reason
                }

                self.today_trades.append(trade)
                self.save_trades()

                # Remove all positions for this token
                self.positions = [p for p in self.positions if not (p.token_type == token_type and p.token_id == token_id)]
                self.last_position_close_time = time.time()

                put_shares = sum(p.quantity for p in self.positions if p.token_type == 'PUT')
                call_shares = sum(p.quantity for p in self.positions if p.token_type == 'CALL')
                logger.info(f"📦 Remaining: PUT:{put_shares:.0f} CALL:{call_shares:.0f} shares")

                return True
            else:
                logger.warning(f"⚠️  Sell not filled, {balance_after:.2f} shares remain")
                return False

        except Exception as e:
            logger.error(f"❌ Error executing sell: {e}")
            import traceback
            traceback.print_exc()
            return False

    def execute_sell_limit(self, token_type: str, token_id: str, bid_price: float,
                     btc_price_chainlink: float, bin_key: str, reason: str) -> bool:
        """Execute sell order using FAK - sells entire balance for token type"""
        try:
            # Find all positions for this token type
            matching_positions = [p for p in self.positions if p.token_type == token_type and p.token_id == token_id]

            if not matching_positions:
                logger.warning(f"⚠️  No {token_type} positions to sell")
                return False

            # Get actual balance from wallet
            actual_balance = self.check_token_balance(token_id)

            if actual_balance < 0.5:
                logger.warning(f"⚠️  No {token_type} tokens in wallet")
                # Clear positions from tracking
                self.positions = [p for p in self.positions if not (p.token_type == token_type and p.token_id == token_id)]
                return False

            logger.info(f"\n{'='*70}")
            logger.info(f"💰 EXECUTING SELL - {reason}")
            logger.info(f"{'='*70}")
            logger.info(f"📊 Token: {token_type}")
            logger.info(f"📦 Size: {actual_balance:.2f} shares")
            logger.info(f"💰 Bid: ${bid_price:.4f}")

            # Determine sell price based on reason
            if "STOP_LOSS" in reason:
                # Aggressive pricing for stop loss
                sell_price = max(0.01, bid_price - 0.01)
                logger.info(f"🛑 Aggressive Stop Loss Price: ${sell_price:.4f} (bid - $0.01)")
            else:
                # Normal pricing for take profit
                sell_price = bid_price
                logger.info(f"💰 Take Profit Price: ${sell_price:.4f} (bid)")

            # Calculate P&L from oldest position (FIFO)
            oldest_position = matching_positions[0]
            total_pnl = sum((bid_price - p.entry_price) * p.quantity for p in matching_positions)
            logger.info(f"📈 Total P&L: ${total_pnl:+.2f}")

            # Place FAK sell order
            start_time = time.time()
            order_id = self.trader.place_sell_order_limit(
                token_id=token_id,
                price=sell_price,
                quantity=actual_balance
            )

            if not order_id:
                logger.error(f"❌ Failed to place sell order")
                return False

            logger.info(f"✅ FAK Sell order placed: {order_id[:16]}... ({time.time() - start_time:.3f}s)")

            # Wait and verify
            time.sleep(3)
            balance_after = self.check_token_balance(token_id)

            if balance_after < actual_balance * 0.5:  # Sold at least half
                logger.info(f"✅ SELL FILLED - Position CLOSED")

                # Record trade
                trade = {
                    'type': oldest_position.token_type,
                    'open_time': datetime.fromtimestamp(oldest_position.entry_time).isoformat(),
                    'close_time': datetime.now().isoformat(),
                    'open_btc': oldest_position.entry_btc_price,
                    'close_btc': btc_price_chainlink,
                    'open_price': oldest_position.entry_price,
                    'close_price': bid_price,
                    'open_bin': oldest_position.entry_bin,
                    'close_bin': bin_key,
                    'edge': oldest_position.edge,
                    'quantity': actual_balance,
                    'pnl': total_pnl,
                    'close_reason': reason
                }

                self.today_trades.append(trade)
                self.save_trades()

                # Remove all positions for this token
                self.positions = [p for p in self.positions if not (p.token_type == token_type and p.token_id == token_id)]
                self.last_position_close_time = time.time()

                put_shares = sum(p.quantity for p in self.positions if p.token_type == 'PUT')
                call_shares = sum(p.quantity for p in self.positions if p.token_type == 'CALL')
                logger.info(f"📦 Remaining: PUT:{put_shares:.0f} CALL:{call_shares:.0f} shares")

                return True
            else:
                logger.warning(f"⚠️  Sell not filled, {balance_after:.2f} shares remain")
                return False

        except Exception as e:
            logger.error(f"❌ Error executing sell: {e}")
            import traceback
            traceback.print_exc()
            return False


    def get_total_tokens(self) -> float:
        """Get total tokens held across all positions"""
        return sum(pos.quantity for pos in self.positions)

    def get_daily_pnl(self) -> float:
        """Get today's total PNL"""
        return sum(t['pnl'] for t in self.today_trades)

    def run(self):
        """Main trading loop"""
        logger.info("\n🚀 Starting Bot016_Third\n")

        # Track previous state
        prev_btc_price_coinbase = None
        prev_call_bid = None
        prev_call_ask = None
        prev_put_bid = None
        prev_put_ask = None

        try:
            while True:
                now = datetime.now()
                current_minute = now.minute
                current_second = now.second

                # Determine current period
                period_start = None
                for start_min in [0, 15, 30, 45]:
                    if current_minute >= start_min and current_minute < start_min + 15:
                        period_start = start_min
                        break

                # Calculate time remaining
                if period_start is not None:
                    seconds_into_period = (current_minute - period_start) * 60 + current_second
                    seconds_remaining = 900 - seconds_into_period

                    # Buffer zones: first 20s and last 20s of period
                    in_start_buffer = seconds_into_period <= BUFFER_SECONDS
                    in_end_buffer = seconds_remaining <= BUFFER_SECONDS
                    in_buffer_zone = in_start_buffer or in_end_buffer
                else:
                    seconds_into_period = 0
                    seconds_remaining = 0
                    in_start_buffer = True
                    in_end_buffer = True
                    in_buffer_zone = True

                # NEW PERIOD DETECTED
                if period_start != self.current_period_start:
                    # Cancel all TP orders at end of period
                    if self.current_period_start is not None:
                        logger.info("⏰ PERIOD ENDING - Canceling all TP orders")
                        self.cancel_all_tp_orders()

                    # Just update period - positions carry forward with stop loss/take profit
                    self.current_period_start = period_start

                    # Reset buffer flag and validation flag for new period
                    self.start_buffer_reload_done = False
                    self.period_asset_ids_validated = False

                    # Reset last valid prices
                    self.last_call_bid = None
                    self.last_put_bid = None
                    self.last_btc_price = None

                    # Initial asset ID reload at period start
                    logger.info(f"\n{'='*80}")
                    logger.info(f"🔄 NEW PERIOD: {now.strftime('%H:%M')} (:{period_start:02d})")
                    logger.info(f"{'='*80}")
                    logger.info(f"⏳ Start buffer active: no trading for first {BUFFER_SECONDS}s")
                    self.reload_asset_ids()

                    # Immediate validation attempt
                    logger.info("🔍 Validating asset IDs for new period...")
                    if self.validate_new_period_asset_ids():
                        self.period_asset_ids_validated = True
                    else:
                        logger.warning("⚠️  Asset IDs not yet updated - will retry during buffer period")

                # VALIDATION RETRY LOOP - During buffer period, keep checking until IDs are valid
                if period_start is not None and not self.period_asset_ids_validated and in_buffer_zone:
                    # Check every 3 seconds during buffer
                    if seconds_into_period % 3 == 0:
                        logger.info(f"🔍 Retry validation ({seconds_into_period}s into period)...")
                        self.reload_asset_ids()
                        if self.validate_new_period_asset_ids():
                            self.period_asset_ids_validated = True
                            logger.info("✅ Asset IDs NOW VALIDATED - Trading can proceed after buffer")

                # Reload asset IDs at END of start buffer (after 30 seconds)
                if period_start is not None and not self.start_buffer_reload_done:
                    if seconds_into_period > BUFFER_SECONDS and seconds_into_period <= BUFFER_SECONDS + 2:
                        logger.info(f"\n✅ START BUFFER ENDED - Final asset ID reload")
                        self.reload_asset_ids()

                        # Final validation check
                        if not self.period_asset_ids_validated:
                            logger.warning("⚠️  FINAL VALIDATION CHECK...")
                            if self.validate_new_period_asset_ids():
                                self.period_asset_ids_validated = True
                            else:
                                logger.error("🚫 CRITICAL: Asset IDs still not validated after buffer!")
                                logger.error("🚫 TRADING REMAINS BLOCKED until validation succeeds")

                        self.start_buffer_reload_done = True

                        if self.period_asset_ids_validated:
                            logger.info(f"🟢 Trading now active for remaining period\n")
                        else:
                            logger.error(f"🔴 Trading BLOCKED - waiting for valid asset IDs\n")

                # Update strike price
                is_period_start = current_minute in [0, 15, 30, 45]
                if is_period_start and current_second >= 5 and current_second < 10 and self.last_strike_update_minute != current_minute:

                    btc_data_chainlink = read_json(BTC_FILE_CHAINLINK)
                    if btc_data_chainlink:
                        new_strike = btc_data_chainlink['strike']
                    if new_strike:
                        self.strike_price = new_strike
                        self.last_strike_update_minute = current_minute
                        logger.info(f"✅ Strike: ${self.strike_price:.2f}")

                        put_count = sum(1 for p in self.positions if p.token_type == 'PUT')
                        call_count = sum(1 for p in self.positions if p.token_type == 'CALL')
                        put_shares = sum(p.quantity for p in self.positions if p.token_type == 'PUT')
                        call_shares = sum(p.quantity for p in self.positions if p.token_type == 'CALL')

                        logger.info(f"📦 Positions: {len(self.positions)} (PUT:{put_count}/{put_shares:.0f}sh CALL:{call_count}/{call_shares:.0f}sh)")
                        logger.info(f"📊 Daily: {len(self.today_trades)} trades | PNL: {self.get_daily_pnl():+.3f}\n")

                # Initialize strike on first run
                if self.strike_price is None:

                    btc_data_chainlink = read_json(BTC_FILE_CHAINLINK)
                    if btc_data_chainlink:
                        self.strike_price = btc_data_chainlink['strike']

                # Reload asset IDs every 60s
                if time.time() - self.last_asset_reload >= 60:
                    self.reload_asset_ids()

                # Verify position every 30s
                if time.time() - self.last_position_check >= 30:
                    self.verify_position_from_wallet()

                # Refresh cached USDC balance every 10s
                self.refresh_usdc_balance()

                # 30-SECOND MAINTENANCE CYCLE
                if time.time() - self.last_maintenance_cycle >= 30:
                    logger.info("🔧 MAINTENANCE: Canceling orders, verifying positions, reloading tokens...")
                    self.trader.cancel_all_orders()  # This will cancel TP orders too
                    self.verify_position_from_wallet()
                    self.reload_asset_ids()

                    # Recreate TP orders with correct quantities after verification
                    if any(p.token_type == 'CALL' for p in self.positions):
                        self.update_tp_order('CALL')
                    if any(p.token_type == 'PUT' for p in self.positions):
                        self.update_tp_order('PUT')

                    self.last_maintenance_cycle = time.time()

                # Read prices - all from JSON files
                btc_data_coinbase = read_json(BTC_FILE_COINBASE)
                btc_data_chainlink = read_json(BTC_FILE_CHAINLINK)
                call_data = read_json(CALL_FILE)  # REST JSON file
                put_data = read_json(PUT_FILE)    # REST JSON file

                # Validate data
                if not all([btc_data_coinbase,btc_data_chainlink, call_data, put_data]):
                    missing = []
                    if not btc_data_coinbase: missing.append("BTC price COINBASE")
                    if not btc_data_chainlink: missing.append("BTC price CHAINLINK")
                    if not call_data: missing.append("CALL price")
                    if not put_data: missing.append("PUT price")
                    if missing:
                        print(f"Missing data: {', '.join(missing)}")
                    time.sleep(0.1)
                    continue

                # Extract data
                btc_price_coinbase = btc_data_coinbase.get('price', 0) # MOD1 stays the same for volatility calculation
                btc_price_chainlink = btc_data_chainlink.get('price', 0) # MOD1 stays the same for volatility calculation
                self.btc_price_history.append(btc_price_coinbase) # MOD1 stays the same for volatility calculation

                # Extract prices from REST JSON files (best_bid, best_ask)
                call_bid_price = call_data.get('best_bid', 0)
                call_ask_price = call_data.get('best_ask', 0)

                put_bid_price = put_data.get('best_bid', 0)
                put_ask_price = put_data.get('best_ask', 0)

                # Spreads already calculated in REST JSON
                call_spread = call_data.get('spread', call_ask_price - call_bid_price)
                put_spread = put_data.get('spread', put_ask_price - put_bid_price)

                # Detect spread direction (upward/downward widening)
                call_spread_direction = ""
                put_spread_direction = ""

                if self.prev_call_bid is not None and self.prev_call_ask is not None:
                    bid_move = self.prev_call_bid - call_bid_price
                    ask_move = self.prev_call_ask - call_ask_price
                    if bid_move > ask_move:
                        call_spread_direction = "DOWN"
                    elif ask_move > bid_move:
                        call_spread_direction = "UP"

                if self.prev_put_bid is not None and self.prev_put_ask is not None:
                    bid_move = self.prev_put_bid - put_bid_price
                    ask_move = self.prev_put_ask - put_ask_price
                    if bid_move > ask_move:
                        put_spread_direction = "UP"
                    elif ask_move > bid_move:
                        put_spread_direction = "DOWN"

                # Update previous spread prices
                self.prev_call_bid = call_bid_price
                self.prev_call_ask = call_ask_price
                self.prev_put_bid = put_bid_price
                self.prev_put_ask = put_ask_price

                trading_suspended = False
                if call_spread > MAX_SPREAD or put_spread > MAX_SPREAD:
                    trading_suspended = True
                    if call_spread > MAX_SPREAD:
                        logger.warning(f"⚠️  TRADING SUSPENDED: CALL spread {call_spread:.3f} > {MAX_SPREAD} {call_spread_direction}")
                    if put_spread > MAX_SPREAD:
                        logger.warning(f"⚠️  TRADING SUSPENDED: PUT spread {put_spread:.3f} > {MAX_SPREAD} {put_spread_direction}")

                # Store last valid prices (before buffer)
                if not in_buffer_zone:
                    if call_bid_price > 0:
                        self.last_call_bid = call_bid_price
                    if put_bid_price > 0:
                        self.last_put_bid = put_bid_price
                    if btc_price_coinbase > 0:
                        self.last_btc_price = btc_price_coinbase

                # Calculate volatility and update tracking variables
                volatility = calculate_btc_volatility(self.btc_price_history)

                # Update tracking variables for TP order fallback
                self.last_seconds_remaining = seconds_remaining
                self.last_volatility = volatility
                self.btc_price_chainlink = btc_price_chainlink

                # Get sensitivity prediction from API
                prediction = self.sensitivity_api.get_sensitivity(
                    #btc_price=btc_price, # MOD1 old version
                    btc_price=btc_price_chainlink, # MOD1 new version

                    strike_price=self.strike_price,
                    time_to_expiry_seconds=seconds_remaining,
                    volatility_percent=volatility
                )

                call_sens = prediction['sensitivity']['call']
                put_sens = prediction['sensitivity']['put']

                # Calculate TP price and cap at 0.99 (binary option max)
                # Check position balance (cached count for speed - no API calls)
                put_count = sum(1 for p in self.positions if p.token_type == 'PUT')
                call_count = sum(1 for p in self.positions if p.token_type == 'CALL')
                is_imbalanced = abs(put_count - call_count) > 0

                if is_imbalanced :
                    adjusted_profit = 0.05
                    adjusted_loss = 0.99

                else :
                    adjusted_profit = 0.99
                    adjusted_loss = 0.99
                # if adjusted_profit <= 0.06 :
                #     adjusted_profit = min(adjusted_profit, 0.07)
                #     #adjusted_loss = min(adjusted_loss, 0.35)
                #tp_price = avg_entry + adjusted_profit


                # STOP LOSS & TAKE PROFIT CHECKS
                # TP is primarily handled by limit orders, but check here as fallback
                if len(self.positions) > 0 and not in_buffer_zone:
                    #distance = abs(btc_price - self.strike_price) if self.strike_price else 0 # MOD1 old version
                    distance = abs(btc_price_chainlink - self.strike_price) if self.strike_price else 0 # MOD1 NEW version

                    bin_key = get_bin_key(distance, seconds_remaining, volatility)

                    for position in self.positions[:]:  # Copy list to allow modification
                        current_price = call_bid_price if position.token_type == 'CALL' else put_bid_price

                        if current_price > 0.01:
                            # Calculate P&L per share
                            pnl_per_share = current_price - position.entry_price

                            # TAKE PROFIT: >= +$0.06/share (fallback if limit order not placed/filled)
                            if pnl_per_share >= adjusted_profit:
                                logger.info(f"💰 TAKE PROFIT TRIGGERED: {position.token_type} ${pnl_per_share:+.3f}/share (entry ${position.entry_price:.3f} → ${current_price:.3f})")
                                self.execute_sell(position.token_type, position.token_id, current_price,
                                                btc_price_chainlink, bin_key, f"TAKE_PROFIT_+${pnl_per_share:.3f}") # MOD1

                            # STOP LOSS: <= -$0.09/share
                            elif pnl_per_share <= -adjusted_loss:
                                            logger.info(f"🛑 STOP LOSS TRIGGERED: {position.token_type} ${pnl_per_share:+.3f}/share (entry ${position.entry_price:.3f} → ${current_price:.3f})")
                                            self.execute_sell(position.token_type, position.token_id, current_price,
                                                            btc_price_chainlink, bin_key, f"STOP_LOSS_{pnl_per_share:.3f}") # MOD1

                # TRADING LOGIC - Only when: outside buffer zone, spread acceptable, AND asset IDs validated
                if not in_buffer_zone and not trading_suspended and self.strike_price and prev_btc_price_coinbase is not None and self.period_asset_ids_validated:
                    #distance = abs(btc_price - self.strike_price) # MOD1 old version with coinbase price
                    distance = abs(btc_price_chainlink - self.strike_price) # MOD1 new version
                    bin_key = get_bin_key(distance, seconds_remaining, volatility)

                    # Get sensitivity prediction from API
                    prediction = self.sensitivity_api.get_sensitivity(
                        #btc_price=btc_price, # MOD1 old version
                        btc_price=btc_price_chainlink, # MOD1 new version

                        strike_price=self.strike_price,
                        time_to_expiry_seconds=seconds_remaining,
                        volatility_percent=volatility
                    )

                    call_sens = prediction['sensitivity']['call']
                    put_sens = prediction['sensitivity']['put']
                    confidence = prediction['confidence']


                    btc_delta = btc_price_coinbase - prev_btc_price_coinbase  #  MOD1   this btc_delta remains the same with coinbase prices it is very important
                    print("#### SITUA::", "Delta:", round(btc_delta,2), "SENS c-p: ", round(call_sens,4),"-", round(put_sens,4), "|| DIST:", round(distance,1), "|| VOL:",round(volatility,0), "Texp:",seconds_remaining )

                    # CALL signals
                    if abs(call_sens) > 0.000001 and call_ask_price > 0 and call_bid_price > 0:
                        ideal_call_movement = btc_delta * call_sens * SENS_MULTIPLIER
                        actual_call_ask_movement = call_ask_price - prev_call_ask

                        if ideal_call_movement > 0.01 :
                            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - CALL move: {ideal_call_movement}")



                        if is_imbalanced :
                            ACTION_THRESHOLD = 0.015

                        else :
                            ACTION_THRESHOLD = 0.035

                        if ideal_call_movement > ACTION_THRESHOLD:
                            # Price constraint: do not buy if price > 0.95 or < MIN_BUY_PRICE
                            if call_ask_price > 0.95:
                                logger.debug(f"❌ CALL BUY rejected: price {call_ask_price:.2f} too high (>0.95)")
                            elif call_ask_price < MIN_BUY_PRICE:
                                logger.warning(f"❌ CALL BUY BLOCKED: price {call_ask_price:.3f} < MIN_BUY_PRICE")
                            elif self.can_open_position('CALL'):
                                edge = ideal_call_movement
                                self.execute_buy('CALL', self.current_call_id, call_ask_price,
                                               btc_price_chainlink, bin_key, edge, f"Edge: {edge:.3f}",
                                               bid_price=call_bid_price, seconds_remaining=seconds_remaining,
                                               volatility=volatility) # MOD1
                                print("BUY CALL @",call_ask_price)


                    # PUT signals
                    if abs(put_sens) > 0.000001 and put_ask_price > 0 and put_bid_price > 0:
                        ideal_put_movement = btc_delta * put_sens * SENS_MULTIPLIER
                        actual_put_ask_movement = put_ask_price - prev_put_ask

                        if ideal_put_movement > 0.01 :
                            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - PUT move: {ideal_put_movement}")

                        if ideal_put_movement > ACTION_THRESHOLD:
                            # Price constraint: do not buy if price > 0.95 or < MIN_BUY_PRICE
                            if put_ask_price > 0.95:
                                logger.debug(f"❌ PUT BUY rejected: price {put_ask_price:.2f} too high (>0.95)")
                            elif put_ask_price < MIN_BUY_PRICE:
                                logger.warning(f"❌ PUT BUY BLOCKED: price {put_ask_price:.3f} < MIN_BUY_PRICE")
                            elif self.can_open_position('PUT'):
                                edge = ideal_put_movement
                                self.execute_buy('PUT', self.current_put_id, put_ask_price,
                                               btc_price_chainlink, bin_key, edge, f"Edge: {edge:.3f}",
                                               bid_price=put_bid_price, seconds_remaining=seconds_remaining,
                                               volatility=volatility)
                                print("BUY PUT @",put_ask_price)


                # Update previous state
                prev_btc_price_coinbase = btc_price_coinbase # MOD1
                prev_call_bid = call_bid_price
                prev_call_ask = call_ask_price
                prev_put_bid = put_bid_price
                prev_put_ask = put_ask_price

                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("\n\n⏸️  Stopped by user")


            # Final save
            self.save_trades()
            logger.info(f"\n💾 Saved {len(self.today_trades)} trades")
            logger.info(f"📊 Daily PNL: {self.get_daily_pnl():+.3f}")


def main():
    """Main entry point"""
    try:


        # Load credentials
        try:
            credentials = load_credentials_from_env(CREDENTIALS_ENV)
            print(f"✅ Credentials loaded from {CREDENTIALS_ENV}")
        except Exception as e:
            print(f"❌ Error loading credentials: {e}")
            return

        # Create and run bot
        bot = Bot016Third(credentials)
        bot.run()

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
