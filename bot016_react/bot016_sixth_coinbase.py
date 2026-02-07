#!/usr/bin/env python3
"""
Bot016_Fifth_Coinbase - Binary Options Buy & Hold Strategy (Coinbase Data)
Uses Coinbase BTC price feed and 15-minute candles for strike price
Buys CALL/PUT and holds until expiry
Balances PUT/CALL positions (max diff: 1, max total: 8)
No stop loss, no take profit - settles at binary expiry
90-second initial buffer per period
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

# Import sensitivity predictor
sys.path.insert(0, '/home/ubuntu/013_2025_polymarket/bot016_react')
from sensitivity_api import SensitivityAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# File paths
PUT_FILE = "/home/ubuntu/013_2025_polymarket/15M_PUT.json"
CALL_FILE = "/home/ubuntu/013_2025_polymarket/15M_CALL.json"
BTC_FILE = "/home/ubuntu/013_2025_polymarket/bot016_react/coinbase_btc_price.json"
SENSITIVITY_FILE = "/home/ubuntu/013_2025_polymarket/bot016_react/sensitivity_data/sensitivity_transformed.json"
TRADES_DIR = "/home/ubuntu/013_2025_polymarket/bot016_react/buyhold_trades"

# Coinbase API
COINBASE_CANDLES_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"

# Trading parameters
SENS_MULTIPLIER = 1.2
ACTION_THRESHOLD_BALANCED = 0.032  # Threshold when positions are balanced
ACTION_THRESHOLD_IMBALANCED = 0.016  # Lower threshold when imbalanced (easier to rebalance)
MIN_SECONDS_BETWEEN_POSITIONS = 2
BUFFER_SECONDS = 30  # No trading in first 90s of period
POSITION_SIZE = 5  # 6 shares
MAX_SPREAD = 0.03  # Suspend trading if spread > 0.03
MIN_BUY_PRICE = 0.10  # Never buy below this price
MAX_POSITIONS = 8  # Maximum 8 positions total (balanced PUT/CALL)


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
    strike_price: float  # Strike price when bought
    period_start: int  # Period start minute (0, 15, 30, 45)
    signal_reason: str  # What caused the buy


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


def get_strike_price() -> Optional[float]:
    """Get strike price from Coinbase 15-minute candles API"""
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

        # Asset IDs
        self.current_put_id: Optional[str] = None
        self.current_call_id: Optional[str] = None

        # Trades
        self.trades_dir = Path(TRADES_DIR)
        self.trades_dir.mkdir(exist_ok=True, parents=True)
        self.today_trades = []
        self.load_today_trades()

        # Initialize Sensitivity Predictor API
        logger.info("🔮 Initializing Sensitivity Predictor...")
        self.sensitivity_api = SensitivityAPI(SENSITIVITY_FILE)
        logger.info("✅ Sensitivity Predictor ready")

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
        logger.info("🤖 BOT016_FIFTH_COINBASE - BUY & HOLD STRATEGY")
        logger.info("="*80)
        logger.info("📡 Data Source: Coinbase BTC-USD")
        logger.info("⚠️  NO STOP LOSS / NO TAKE PROFIT")
        logger.info("📊 Holds until binary expiry (0 or 1)")
        logger.info(f"Position Size: {POSITION_SIZE} shares")
        logger.info(f"Max Positions: {MAX_POSITIONS} (balanced PUT/CALL)")
        logger.info(f"Max Spread: ${MAX_SPREAD:.2f} (suspend trading if exceeded)")
        logger.info(f"Sens Multiplier: {SENS_MULTIPLIER}x")
        logger.info(f"Action Threshold: Balanced={ACTION_THRESHOLD_BALANCED} | Imbalanced={ACTION_THRESHOLD_IMBALANCED}")
        logger.info(f"Buffer Zone: {BUFFER_SECONDS}s (first 90s of each period)")
        logger.info(f"🔮 Sensitivity Predictor: {len(self.sensitivity_api.predictor.bins_data)} bins loaded")
        logger.info("="*80)

    def get_today_filename(self) -> Path:
        """Get filename for today's trades"""
        today = date.today().strftime('%Y%m%d')
        return self.trades_dir / f"buyhold_trades_{today}.json"

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
                if call_changed:
                    logger.info(f"   CALL: ...{new_call_id[-12:]}")

                self.current_put_id = new_put_id
                self.current_call_id = new_call_id

            self.last_asset_reload = time.time()

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

                # Resync with actual wallet balance
                put_data = read_json(PUT_FILE)
                if not put_data or not put_data.get('best_bid') or not put_data.get('best_ask'):
                    logger.warning("DEEP ITM/OTM PUT - cannot sync")
                else:
                    price = (put_data['best_bid'].get('price', 0.50) + put_data['best_ask'].get('price', 0.50)) / 2

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
                            strike_price=self.strike_price if self.strike_price else 0,
                            period_start=self.current_period_start if self.current_period_start else 0,
                            signal_reason="WALLET_SYNC"
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
                            strike_price=self.strike_price if self.strike_price else 0,
                            period_start=self.current_period_start if self.current_period_start else 0,
                            signal_reason="WALLET_SYNC"
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

                # Resync with actual wallet balance
                call_data = read_json(CALL_FILE)
                if not call_data or not call_data.get('best_bid') or not call_data.get('best_ask'):
                    logger.warning("DEEP ITM/OTM CALL - cannot sync")
                else:
                    price = (call_data['best_bid'].get('price', 0.50) + call_data['best_ask'].get('price', 0.50)) / 2

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
                            strike_price=self.strike_price if self.strike_price else 0,
                            period_start=self.current_period_start if self.current_period_start else 0,
                            signal_reason="WALLET_SYNC"
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
                            strike_price=self.strike_price if self.strike_price else 0,
                            period_start=self.current_period_start if self.current_period_start else 0,
                            signal_reason="WALLET_SYNC"
                        )
                        self.positions.append(synced_position)

                    logger.info(f"✅ Synced CALL: {num_positions} full positions + {remainder:.2f} tokens = {call_balance:.2f} total")
            else:
                logger.info("✅ Cleared CALL positions from tracking")

        self.last_position_check = time.time()

    def execute_buy(self, token_type: str, token_id: str, ask_price: float,
                    btc_price: float, bin_key: str, edge: float, reason: str,
                    strike_price: float, period_start: int) -> bool:
        """Execute buy order - Buy & Hold strategy"""
        try:
            logger.info(f"\n{'='*70}")
            logger.info(f"🛒 EXECUTING BUY ORDER - HOLD UNTIL EXPIRY")
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
                final_price = ask_price + 0.01
                logger.info(f"⚖️  Imbalanced positions (PUT:{put_count} CALL:{call_count}) - adding +$0.01 for faster fill")
            else:
                final_price = ask_price
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

            # Fill-or-Kill: Wait 1 second, then cancel if not filled
            time.sleep(1)

            # Quick check if filled
            balance_1s = self.check_token_balance(token_id)
            if balance_1s < POSITION_SIZE * 0.8:
                logger.warning(f"⚠️  Fill-or-Kill: Order not filled in 1s, canceling...")
                self.trader.cancel_all_orders()
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
                    logger.info(f"   ✅ Position confirmed")
                    break

            # PHASE 3: Final check
            final_balance = self.check_token_balance(token_id)

            if position_confirmed and final_balance >= 0.5:
                logger.info(f"\n✅ SUCCESS: Position opened - HOLDING UNTIL EXPIRY")

                new_position = Position(
                    token_type=token_type,
                    token_id=token_id,
                    entry_price=ask_price,
                    entry_time=time.time(),
                    quantity=final_balance,
                    entry_btc_price=btc_price,
                    entry_bin=bin_key,
                    edge=edge,
                    strike_price=strike_price,
                    period_start=period_start,
                    signal_reason=reason
                )

                self.positions.append(new_position)

                put_count = sum(1 for p in self.positions if p.token_type == 'PUT')
                call_count = sum(1 for p in self.positions if p.token_type == 'CALL')

                logger.info(f"📊 Position: {final_balance:.2f} @ ${ask_price:.4f} | Strike: ${strike_price:.2f}")
                logger.info(f"📦 Total positions: {len(self.positions)}/{MAX_POSITIONS} (PUT:{put_count} CALL:{call_count})")
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



    def can_open_position(self, token_type: str) -> bool:
        """Check if we can open a new position - maintains PUT/CALL balance"""
        # Check max positions
        if len(self.positions) >= MAX_POSITIONS:
            return False

        if time.time() - self.last_position_close_time < MIN_SECONDS_BETWEEN_POSITIONS:
            return False

        # Count PUT and CALL positions
        put_count = sum(1 for p in self.positions if p.token_type == 'PUT')
        call_count = sum(1 for p in self.positions if p.token_type == 'CALL')

        # Allow if difference would not exceed 1
        if token_type == 'CALL':
            return (call_count - put_count) < 1
        else:  # PUT
            return (put_count - call_count) < 1

    def settle_expired_positions(self, current_period_start: int):
        """Settle positions from previous period based on binary expiry"""
        if not self.positions:
            logger.info(f"\n💰 PERIOD END: No positions to settle")
            return

        # Get current strike price
        current_strike = get_strike_price()
        if not current_strike:
            logger.error("❌ Cannot settle: no current strike price")
            return

        # Find positions from previous periods
        expired_positions = [p for p in self.positions if p.period_start != current_period_start]

        if not expired_positions:
            logger.info(f"\n💰 PERIOD END: No expired positions (all current)")
            return

        logger.info(f"\n{'='*80}")
        logger.info(f"💰 SETTLING EXPIRED POSITIONS - BINARY EXPIRY")
        logger.info(f"{'='*80}")
        logger.info(f"Current Strike: ${current_strike:.2f}")
        logger.info(f"Positions to Settle: {len(expired_positions)}")

        # Categorize positions
        call_positions = [p for p in expired_positions if p.token_type == 'CALL']
        put_positions = [p for p in expired_positions if p.token_type == 'PUT']

        total_pnl = 0
        itm_count = 0
        otm_count = 0

        logger.info(f"\n📊 SETTLEMENT DETAILS:")

        for pos in expired_positions:
            # Determine if ITM or OTM
            if pos.token_type == 'CALL':
                # CALL wins if current strike > entry strike
                is_itm = current_strike > pos.strike_price
                comparison = ">" if is_itm else "<="
            else:  # PUT
                # PUT wins if current strike < entry strike
                is_itm = current_strike < pos.strike_price
                comparison = "<" if is_itm else ">="

            # Calculate PNL
            if is_itm:
                # Won: option expires at $1
                exit_price = 1.0
                proceeds = exit_price * pos.quantity
                pnl = proceeds - (pos.entry_price * pos.quantity)
                result = "✅ ITM"
                itm_count += 1
            else:
                # Lost: option expires at $0
                exit_price = 0.0
                proceeds = 0.0
                pnl = -(pos.entry_price * pos.quantity)
                result = "❌ OTM"
                otm_count += 1

            total_pnl += pnl

            logger.info(f"  {pos.token_type}: Entry Strike ${pos.strike_price:.2f} {comparison} Exit ${current_strike:.2f} = {result}")
            logger.info(f"     Signal: {pos.signal_reason}")
            logger.info(f"     Cost: ${pos.entry_price:.3f} × {pos.quantity} = ${pos.entry_price * pos.quantity:.2f}")
            logger.info(f"     Payout: ${exit_price:.2f} × {pos.quantity} = ${proceeds:.2f}")
            logger.info(f"     P&L: ${pnl:+.2f}")

            # Record trade
            trade = {
                'type': pos.token_type,
                'open_time': datetime.fromtimestamp(pos.entry_time).isoformat(),
                'close_time': datetime.now().isoformat(),
                'open_btc': pos.entry_btc_price,
                'close_btc': current_strike,
                'entry_strike': pos.strike_price,
                'exit_strike': current_strike,
                'open_price': pos.entry_price,
                'close_price': exit_price,
                'open_bin': pos.entry_bin,
                'close_bin': 'EXPIRY',
                'quantity': pos.quantity,
                'is_itm': is_itm,
                'signal_reason': pos.signal_reason,
                'edge': pos.edge,
                'pnl': pnl,
                'close_reason': 'BINARY_EXPIRY'
            }

            self.today_trades.append(trade)

        # Remove expired positions
        self.positions = [p for p in self.positions if p.period_start == current_period_start]

        # PERIOD SUMMARY
        put_count = sum(1 for p in self.positions if p.token_type == 'PUT')
        call_count = sum(1 for p in self.positions if p.token_type == 'CALL')

        logger.info(f"\n{'='*60}")
        logger.info(f"📊 PERIOD SUMMARY:")
        logger.info(f"{'='*60}")
        logger.info(f"Settled: {len(expired_positions)} positions ({len(call_positions)} CALL / {len(put_positions)} PUT)")
        logger.info(f"Winners: {itm_count} ITM | Losers: {otm_count} OTM")
        logger.info(f"Period P&L: ${total_pnl:+.2f}")
        logger.info(f"Daily Total: {len(self.today_trades)} trades | P&L: ${self.get_daily_pnl():+.2f}")
        logger.info(f"Remaining: {len(self.positions)} positions (PUT:{put_count} CALL:{call_count})")
        logger.info(f"{'='*60}\n")

        # Save trades
        self.save_trades()


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
        prev_btc_price = None
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
                    # Settle expired positions from previous period
                    if self.current_period_start is not None:
                        self.settle_expired_positions(period_start)

                    # Update period
                    self.current_period_start = period_start

                    # Reset buffer flag for new period
                    self.start_buffer_reload_done = False

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

                # Reload asset IDs at END of start buffer (after 90 seconds)
                if period_start is not None and not self.start_buffer_reload_done:
                    if seconds_into_period > BUFFER_SECONDS and seconds_into_period <= BUFFER_SECONDS + 2:
                        logger.info(f"\n✅ START BUFFER ENDED - Final asset ID reload")
                        self.reload_asset_ids()
                        self.start_buffer_reload_done = True
                        logger.info(f"🟢 Trading now active for remaining period\n")

                # Update strike price
                is_period_start = current_minute in [0, 15, 30, 45]
                if is_period_start and current_second >= 5 and current_second < 10 and self.last_strike_update_minute != current_minute:
                    new_strike = get_strike_price()
                    if new_strike:
                        self.strike_price = new_strike
                        self.last_strike_update_minute = current_minute
                        logger.info(f"✅ Strike: ${self.strike_price:.2f}")

                        put_count = sum(1 for p in self.positions if p.token_type == 'PUT')
                        call_count = sum(1 for p in self.positions if p.token_type == 'CALL')

                        logger.info(f"📦 Positions: {len(self.positions)}/{MAX_POSITIONS} (PUT:{put_count} CALL:{call_count})")
                        logger.info(f"📊 Daily: {len(self.today_trades)} trades | PNL: {self.get_daily_pnl():+.3f}\n")

                # Initialize strike on first run
                if self.strike_price is None:
                    self.strike_price = get_strike_price()
                    if not self.strike_price:
                        btc_data = read_json(BTC_FILE)
                        if btc_data:
                            self.strike_price = btc_data['price']

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
                    self.trader.cancel_all_orders()
                    self.verify_position_from_wallet()
                    self.reload_asset_ids()
                    self.last_maintenance_cycle = time.time()

                # Read prices
                btc_data = read_json(BTC_FILE)
                call_data = read_json(CALL_FILE)
                put_data = read_json(PUT_FILE)

                # Calculate timestamp differences in seconds
                if btc_data and call_data and put_data:
                    btc_ts = btc_data.get('timestamp', 0)
                    call_ts = int(call_data.get('timestamp', 0))
                    put_ts = int(put_data.get('timestamp', 0))

                    btc_call_diff = (btc_ts - call_ts) / 1000.0
                    btc_put_diff = (btc_ts - put_ts) / 1000.0
                    #
                    # if btc_call_diff > 1 or btc_call_diff < -1 :
                    #     print(f"⏱️  Timestamp deltas: BTC-CALL={btc_call_diff:.2f}s, BTC-PUT={btc_put_diff:.2f}s")

                    if btc_call_diff < -1 :
                        #print ("LAG BTC websocket ................  WAIT")
                        #time.sleep(1)
                        continue

                if not all([btc_data, call_data, put_data]):
                    time.sleep(0.1)
                    continue

                # Extract data
                btc_price = btc_data.get('price', 0)
                self.btc_price_history.append(btc_price)

                call_bid_price = call_data.get('best_bid', {}).get('price', 0) if call_data.get('best_bid') else 0
                call_ask_price = call_data.get('best_ask', {}).get('price', 0) if call_data.get('best_ask') else 0

                put_bid_price = put_data.get('best_bid', {}).get('price', 0) if put_data.get('best_bid') else 0
                put_ask_price = put_data.get('best_ask', {}).get('price', 0) if put_data.get('best_ask') else 0

                # Check spreads - suspend trading if spread > MAX_SPREAD
                call_spread = call_ask_price - call_bid_price if (call_ask_price > 0 and call_bid_price > 0) else 0
                put_spread = put_ask_price - put_bid_price if (put_ask_price > 0 and put_bid_price > 0) else 0

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
                    if btc_price > 0:
                        self.last_btc_price = btc_price

                # Calculate volatility
                volatility = calculate_btc_volatility(self.btc_price_history)

                # TRADING LOGIC - Only outside buffer zone and when spread is acceptable
                if not in_buffer_zone and not trading_suspended and self.strike_price and prev_btc_price is not None:
                    distance = abs(btc_price - self.strike_price)
                    bin_key = get_bin_key(distance, seconds_remaining, volatility)

                    # Get dynamic sensitivity prediction from API
                    try:
                        distance = abs(btc_price - self.strike_price) if self.strike_price else 0

                        # Get sensitivity prediction
                        prediction = self.sensitivity_api.get_sensitivity(
                            btc_price=btc_price,
                            strike_price=self.strike_price,
                            time_to_expiry_seconds=seconds_remaining,
                            volatility_percent=volatility
                        )

                        call_sens = prediction['sensitivity']['call']
                        put_sens = prediction['sensitivity']['put']
                        confidence = prediction['confidence']

                        btc_delta = btc_price - prev_btc_price

                        # Dynamic ACTION_THRESHOLD based on position balance (cached for speed)
                        put_count = sum(1 for p in self.positions if p.token_type == 'PUT')
                        call_count = sum(1 for p in self.positions if p.token_type == 'CALL')
                        is_imbalanced = abs(put_count - call_count) > 0

                        ACTION_THRESHOLD = ACTION_THRESHOLD_IMBALANCED if is_imbalanced else ACTION_THRESHOLD_BALANCED

                        # CALL signals
                        if abs(call_sens) > 0.000001 and call_ask_price > 0 and call_bid_price > 0:
                            ideal_call_movement = btc_delta * call_sens * SENS_MULTIPLIER
                            actual_call_ask_movement = call_ask_price - prev_call_ask

                            if ideal_call_movement > 0.01 :
                                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - CALL move: {ideal_call_movement} (conf: {confidence:.2f})")

                            if ideal_call_movement > ACTION_THRESHOLD:
                                # Price constraint: do not buy if price > 0.95 or < MIN_BUY_PRICE
                                # BUY signal - maintain balance
                                if call_ask_price > 0.95:
                                    logger.debug(f"❌ CALL BUY rejected: price {call_ask_price:.2f} too high (>0.95)")
                                elif call_ask_price < MIN_BUY_PRICE:
                                    logger.warning(f"❌ CALL BUY BLOCKED: price {call_ask_price:.3f} < MIN_BUY_PRICE {MIN_BUY_PRICE:.2f}")
                                elif self.can_open_position('CALL'):
                                    edge = ideal_call_movement
                                    self.execute_buy('CALL', self.current_call_id, call_ask_price,
                                                   btc_price, bin_key, edge, f"Edge: {edge:.3f}",
                                                   self.strike_price, period_start)
                                    print("BUY CALL @",call_ask_price)



                        # PUT signals
                        if abs(put_sens) > 0.000001 and put_ask_price > 0 and put_bid_price > 0:
                            ideal_put_movement = btc_delta * put_sens * SENS_MULTIPLIER
                            actual_put_ask_movement = put_ask_price - prev_put_ask

                            if ideal_put_movement > 0.01 :
                                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - PUT move: {ideal_put_movement} (conf: {confidence:.2f})")

                            if ideal_put_movement > ACTION_THRESHOLD:
                                # Price constraint: do not buy if price > 0.95 or < MIN_BUY_PRICE
                                # BUY signal - maintain balance
                                if put_ask_price > 0.95:
                                    logger.debug(f"❌ PUT BUY rejected: price {put_ask_price:.2f} too high (>0.95)")
                                elif put_ask_price < MIN_BUY_PRICE:
                                    logger.warning(f"❌ PUT BUY BLOCKED: price {put_ask_price:.3f} < MIN_BUY_PRICE {MIN_BUY_PRICE:.2f}")
                                elif self.can_open_position('PUT'):
                                    edge = ideal_put_movement
                                    self.execute_buy('PUT', self.current_put_id, put_ask_price,
                                                   btc_price, bin_key, edge, f"Edge: {edge:.3f}",
                                                   self.strike_price, period_start)
                                    print("BUY PUT @",put_ask_price)

                    except Exception as e:
                        logger.error(f"❌ Sensitivity prediction error: {e}")
                        # Continue without trading on this cycle

                # Update previous state
                prev_btc_price = btc_price
                prev_call_bid = call_bid_price
                prev_call_ask = call_ask_price
                prev_put_bid = put_bid_price
                prev_put_ask = put_ask_price

                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("\n\n⏸️  Stopped by user")

            # Close all open positions
            # if len(self.positions) > 0:
            #     logger.info(f"\n⚠️  Closing {len(self.positions)} open positions...")
            #
            #     positions_to_close = self.positions.copy()
            #     for pos in positions_to_close:
            #         data = read_json(CALL_FILE if pos.token_type == 'CALL' else PUT_FILE)
            #
            #         if data and data.get('best_bid'):
            #             exit_price = data['best_bid'].get('price', 0)
            #             btc_data = read_json(BTC_FILE)
            #             btc_price = btc_data.get('price', 0) if btc_data else 0
            #
            #             if exit_price > 0:
            #                 distance = abs(btc_price - self.strike_price) if self.strike_price else 0
            #                 volatility = calculate_btc_volatility(self.btc_price_history)
            #                 seconds_left = get_seconds_to_expiry()
            #                 bin_key = get_bin_key(distance, seconds_left, volatility)
            #
            #                 self.execute_sell(pos.token_type, pos.token_id, exit_price, btc_price, bin_key, "MANUAL_STOP")

            # Final save
            self.save_trades()
            logger.info(f"\n💾 Saved {len(self.today_trades)} trades")
            logger.info(f"📊 Daily PNL: {self.get_daily_pnl():+.3f}")


def main():
    """Main entry point"""
    try:


        # Load credentials
        try:
            env_path = '/home/ubuntu/013_2025_polymarket/keys/keys_ovh38.env'
            credentials = load_credentials_from_env(env_path)
            print(f"✅ Credentials loaded from {env_path}")
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
