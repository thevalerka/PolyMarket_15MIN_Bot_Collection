#!/usr/bin/env python3
"""
Probability-Based Trading Bot - PRODUCTION
Trades based on Black-Scholes probability calculations.

Entry Rules:
1. CALL: call_probability > 0.995 → Place limit buy at 0.991
2. PUT: put_probability > 0.995 → Place limit buy at 0.991
3. Choppiness must be <= 35
4. Hold order unless probability drops below 0.995

Exit:
- At expiry: Compare 15-min candle close vs strike

pm2 start bot_probability_production.py --cron-restart="00 * * * *" --interpreter python3
"""

import json
import time
import sys
import requests
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass
import logging

# Import Polymarket trading core
sys.path.insert(0, '/home/ubuntu')
from polymarket_trading_core_debug import PolymarketTrader, load_credentials_from_env

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths
PUT_FILE = "/home/ubuntu/013_2025_polymarket/15M_PUT.json"
CALL_FILE = "/home/ubuntu/013_2025_polymarket/15M_CALL.json"
BTC_FILE = "/home/ubuntu/013_2025_polymarket/bybit_btc_price.json"
BS_DATA_FILE = "/home/ubuntu/013_2025_polymarket/bot018_Lastsec/BS_data/last_option_data.json"
STATE_FILE = "/home/ubuntu/013_2025_polymarket/bot017_RSI/bot_probability_state.json"
TRADES_DIR = "/home/ubuntu/013_2025_polymarket/bot017_RSI/bot_probability_trades"

# Trading Parameters
POSITION_SIZE = 5.0  # 5 shares
LIMIT_PRICE = 0.99  # Default limit order price
MAX_LIMIT_PRICE = 0.99  # Maximum limit price
BID_PRICE_THRESHOLD = 0.98  # If bid <= this, use bid price instead
PROBABILITY_THRESHOLD = 0.995  # 99.5% probability required
MAX_CHOPPINESS = 35  # Maximum choppiness allowed
MAX_DATA_AGE_MS = 1000  # Maximum age of market data (1 second)
BUFFER_SECONDS = 5  # No trading in last 5s of period
PERIOD_CHANGE_WAIT = 20  # Wait 20s after period change before trading

# ============================================================================
# POSITION DATACLASS
# ============================================================================

@dataclass
class Position:
    """Open position tracker"""
    token_type: str  # 'PUT' or 'CALL'
    token_id: str
    entry_price: float
    entry_time: float
    quantity: float
    entry_btc_price: float
    strike_price: float
    probability_at_entry: float
    choppiness_at_entry: float
    time_to_expiry: float

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def read_json(filepath: str) -> Optional[dict]:
    """Read JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return None

def get_btc_price():
    """Get current BTC price from local JSON file"""
    try:
        data = read_json(BTC_FILE)
        if data:
            return data.get('price')
        return None
    except:
        return None

def get_strike_price() -> Optional[float]:
    """Get strike price from Binance API"""
    try:
        now = datetime.now(timezone.utc)
        current_minute = now.minute

        for start_min in [0, 15, 30, 45]:
            if current_minute >= start_min and current_minute < start_min + 15:
                period_start = now.replace(minute=start_min, second=0, microsecond=0)
                start_timestamp = int(period_start.timestamp() * 1000)

                url = "https://api.binance.com/api/v3/klines"
                params = {
                    'symbol': 'BTCUSDT',
                    'interval': '15m',
                    'startTime': start_timestamp,
                    'limit': 1,
                    'timeZone': '0'  # UTC
                }

                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data and len(data) > 0:
                        return float(data[0][1])
        return None
    except Exception as e:
        logger.error(f"Error getting strike price: {e}")
        return None

def fetch_period_candle(period_start: datetime) -> Optional[Dict]:
    """
    Fetch the 15-minute candle for a specific period from Binance
    Returns: Dict with 'open', 'high', 'low', 'close' or None
    """
    try:
        start_timestamp = int(period_start.timestamp() * 1000)

        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': 'BTCUSDT',
            'interval': '15m',
            'startTime': start_timestamp,
            'limit': 1,
            'timeZone': '0'  # UTC
        }

        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return None

        data = response.json()
        if not data or len(data) == 0:
            return None

        candle = data[0]
        return {
            'open': float(candle[1]),
            'high': float(candle[2]),
            'low': float(candle[3]),
            'close': float(candle[4])
        }
    except Exception as e:
        logger.error(f"Error fetching candle: {e}")
        return None

def get_seconds_to_expiry() -> int:
    """Calculate seconds until next 15-minute mark"""
    now = datetime.now()
    minutes_into_quarter = now.minute % 15
    seconds_into_quarter = minutes_into_quarter * 60 + now.second
    return 900 - seconds_into_quarter

def calculate_choppiness(price_history: list, period: int = 900) -> float:
    """
    Calculate Choppiness Index (0-100) from price history

    Args:
        price_history: List of recent prices (should have ~900 samples for 15 min)
        period: Lookback period in samples

    Returns:
        Choppiness Index value (0-100)
    """
    if len(price_history) < 10:
        return 50.0

    import numpy as np

    prices = np.array(price_history[-period:]) if len(price_history) >= period else np.array(price_history)

    # Calculate true ranges
    true_ranges = np.abs(np.diff(prices))
    sum_tr = np.sum(true_ranges)

    # High and low
    high = np.max(prices)
    low = np.min(prices)
    high_low_range = high - low

    if high_low_range == 0 or sum_tr == 0:
        return 100.0

    # Choppiness Index
    actual_period = len(prices)
    choppiness = 100 * np.log10(sum_tr / high_low_range) / np.log10(actual_period)

    return max(0.0, min(100.0, choppiness))

# ============================================================================
# BOT CLASS
# ============================================================================

class BotProbability:
    """Production probability-based trading bot with Polymarket integration"""

    def __init__(self, credentials: dict):
        # Initialize Polymarket trader
        self.trader = PolymarketTrader(
            clob_api_url=credentials['clob_api_url'],
            private_key=credentials['private_key'],
            api_key=credentials['api_key'],
            api_secret=credentials['api_secret'],
            api_passphrase=credentials['api_passphrase']
        )

        # Position tracking
        self.position: Optional[Position] = None
        self.last_position_close_time = 0

        # Asset IDs
        self.current_put_id: Optional[str] = None
        self.current_call_id: Optional[str] = None

        # Trades
        self.trades_dir = Path(TRADES_DIR)
        self.trades_dir.mkdir(exist_ok=True, parents=True)
        self.today_trades = []
        self.load_today_trades()

        # Period tracking
        self.strike_price: Optional[float] = None
        self.current_period_start: Optional[int] = None
        self.period_change_time: Optional[float] = None  # Track when period changed

        # Pending order tracking
        self.pending_order_id: Optional[str] = None
        self.pending_order_type: Optional[str] = None  # 'CALL' or 'PUT'
        self.pending_order_token_id: Optional[str] = None
        self.pending_order_time: Optional[float] = None
        self.pending_order_limit_price: Optional[float] = None  # Track what price we placed order at

        # Last valid prices
        self.last_call_bid: Optional[float] = None
        self.last_put_bid: Optional[float] = None
        self.last_btc_price: Optional[float] = None

        # Position verification
        self.last_position_check = time.time()
        self.last_asset_reload = time.time()

        # Cached USDC balance
        self.cached_usdc_balance = 0.0
        self.last_usdc_check = 0.0
        self.usdc_check_interval = 10.0

        # Price history for choppiness calculation (900 samples = 15 minutes at 1s intervals)
        self.price_history = []
        self.max_history_length = 900

        logger.info("="*80)
        logger.info("🤖 PROBABILITY-BASED TRADING BOT")
        logger.info("="*80)
        logger.info(f"Position Size: {POSITION_SIZE} shares")
        logger.info(f"Limit Price: ${LIMIT_PRICE}")
        logger.info(f"Probability Threshold: {PROBABILITY_THRESHOLD*100:.1f}%")
        logger.info(f"Max Choppiness: {MAX_CHOPPINESS}")
        logger.info("="*80)

    def get_today_filename(self) -> Path:
        """Get filename for today's trades"""
        today = date.today().strftime('%Y%m%d')
        return self.trades_dir / f"probability_{today}.json"

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
            'strategy': 'Probability_Based',
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
            #
            # ticksize = py_clob_client.__tick_sizes = {}
            #
            # print ("TICKSIZE",ticksize)

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
        """Verify position matches wallet"""
        if not self.current_put_id or not self.current_call_id:
            return

        put_balance = self.check_token_balance(self.current_put_id)
        call_balance = self.check_token_balance(self.current_call_id)

        has_put = put_balance >= 0.5
        has_call = call_balance >= 0.5

        # Case 1: PUT position
        if has_put and not has_call:
            if self.position is None or self.position.token_type != 'PUT':
                logger.warning(f"⚠️  Wallet has PUT ({put_balance:.2f}), tracking: {self.position.token_type if self.position else 'None'}")
                self.position = Position(
                    token_type='PUT',
                    token_id=self.current_put_id,
                    entry_price=LIMIT_PRICE,
                    entry_time=time.time(),
                    quantity=put_balance,
                    entry_btc_price=0,
                    strike_price=self.strike_price if self.strike_price else 0,
                    probability_at_entry=0,
                    choppiness_at_entry=0,
                    time_to_expiry=0
                )

        # Case 2: CALL position
        elif has_call and not has_put:
            if self.position is None or self.position.token_type != 'CALL':
                logger.warning(f"⚠️  Wallet has CALL ({call_balance:.2f}), tracking: {self.position.token_type if self.position else 'None'}")
                self.position = Position(
                    token_type='CALL',
                    token_id=self.current_call_id,
                    entry_price=LIMIT_PRICE,
                    entry_time=time.time(),
                    quantity=call_balance,
                    entry_btc_price=0,
                    strike_price=self.strike_price if self.strike_price else 0,
                    probability_at_entry=0,
                    choppiness_at_entry=0,
                    time_to_expiry=0
                )

        # Case 3: No position
        elif not has_put and not has_call:
            if self.position is not None:
                logger.warning(f"⚠️  Wallet empty but tracking shows {self.position.token_type}")
                self.position = None

        self.last_position_check = time.time()

    def place_limit_order(self, token_type: str, token_id: str, limit_price: float,
                          btc_price: float, strike_price: float, probability: float,
                          choppiness: float, time_to_expiry: float) -> bool:
        """Place LIMIT buy order at specified price"""
        try:
            # Clear tick size cache to get fresh tick size
            try:
                if hasattr(self.trader.client, '_ClobClient__tick_sizes'):
                    self.trader.client._ClobClient__tick_sizes = {}
                    logger.debug("✅ Cleared tick size cache")
            except Exception as e:
                logger.warning(f"⚠️  Could not clear tick size cache: {e}")

            # Enforce maximum limit price
            if limit_price > MAX_LIMIT_PRICE:
                logger.warning(f"⚠️  Limit price ${limit_price:.3f} > max ${MAX_LIMIT_PRICE:.2f}, using ${MAX_LIMIT_PRICE:.2f}")
                limit_price = MAX_LIMIT_PRICE

            logger.info(f"\n{'='*70}")
            logger.info(f"📊 PLACING LIMIT ORDER")
            logger.info(f"{'='*70}")
            logger.info(f"📊 Token: {token_type}")
            logger.info(f"📦 Size: {POSITION_SIZE} shares")
            logger.info(f"💰 Limit Price: ${limit_price:.4f}")
            logger.info(f"📈 Probability: {probability*100:.2f}%")
            logger.info(f"🌊 Choppiness: {choppiness:.1f}")
            logger.info(f"⏱️  Time to Expiry: {time_to_expiry:.0f}s")

            required = limit_price * POSITION_SIZE
            logger.info(f"💵 Expected Cost: ${required:.2f}")

            # Use cached USDC balance
            cached_balance = self.cached_usdc_balance
            logger.info(f"💰 USDC Balance: ${cached_balance:.2f}")

            MIN_BALANCE = 4.90
            if cached_balance < MIN_BALANCE:
                logger.error(f"❌ INSUFFICIENT BALANCE: ${cached_balance:.2f} < ${MIN_BALANCE:.2f}")
                return False

            if cached_balance < required:
                logger.error(f"❌ Need: ${required:.2f}, Have: ${cached_balance:.2f}")
                return False

            # Place limit order
            start_time = time.time()
            try:
                order_id = self.trader.place_buy_order(
                    token_id=token_id,
                    price=limit_price,
                    quantity=POSITION_SIZE
                )
            except Exception as order_error:
                logger.error(f"❌ Order error: {order_error}")
                return False

            if not order_id:
                logger.error(f"❌ Failed to place order")
                return False

            logger.info(f"✅ Limit order placed: {order_id[:16]}... ({time.time() - start_time:.3f}s)")
            logger.info(f"⏳ Order will remain active until probability drops or filled")

            # Track pending order
            self.pending_order_id = order_id
            self.pending_order_type = token_type
            self.pending_order_token_id = token_id
            self.pending_order_time = time.time()
            self.pending_order_limit_price = limit_price  # Remember what price we placed at

            return True

        except Exception as e:
            logger.error(f"❌ Error placing limit order: {e}")
            import traceback
            traceback.print_exc()
            return False

    def check_order_filled(self) -> bool:
        """Check if pending order has been filled"""
        if not self.pending_order_id or not self.pending_order_token_id:
            return False

        balance = self.check_token_balance(self.pending_order_token_id)

        if balance >= POSITION_SIZE * 0.8:
            logger.info(f"\n✅ ORDER FILLED!")
            logger.info(f"📊 Balance: {balance:.2f} shares")

            # Create position
            self.position = Position(
                token_type=self.pending_order_type,
                token_id=self.pending_order_token_id,
                entry_price=LIMIT_PRICE,
                entry_time=time.time(),
                quantity=balance,
                entry_btc_price=self.last_btc_price if self.last_btc_price else 0,
                strike_price=self.strike_price if self.strike_price else 0,
                probability_at_entry=0,  # Will be filled from BS data
                choppiness_at_entry=0,
                time_to_expiry=get_seconds_to_expiry()
            )

            # Clear pending order tracking
            self.pending_order_id = None
            self.pending_order_type = None
            self.pending_order_token_id = None
            self.pending_order_time = None

            return True

        return False

    def close_position_at_expiry(self, candle: Dict):
        """Close position at expiry based on actual 15-min candle"""
        if not self.position:
            return

        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"⏰ EXPIRY - Settling position")
            logger.info(f"{'='*60}")

            final_btc = candle['close']
            strike = self.position.strike_price

            logger.info(f"  Candle Close: ${final_btc:.2f}")
            logger.info(f"  Strike: ${strike:.2f}")

            # Determine final value
            if self.position.token_type == 'CALL':
                final_value = 1.0 if final_btc > strike else 0.0
                result_text = "WIN ✅" if final_btc > strike else "LOSS ❌"
                logger.info(f"  Type: CALL (win if close > strike)")
            else:  # PUT
                final_value = 1.0 if final_btc < strike else 0.0
                result_text = "WIN ✅" if final_btc < strike else "LOSS ❌"
                logger.info(f"  Type: PUT (win if close < strike)")

            logger.info(f"  Result: {result_text}")

            # Calculate PNL
            pnl = (final_value - self.position.entry_price) * self.position.quantity

            logger.info(f"\n  Trade Summary:")
            logger.info(f"    Entry:  ${self.position.entry_price:.2f}")
            logger.info(f"    Exit:   ${final_value:.2f}")
            logger.info(f"    PNL:    ${pnl:+.2f}")

            # Record trade
            trade = {
                'entry_time': datetime.fromtimestamp(self.position.entry_time).isoformat(),
                'expiry_time': datetime.now().isoformat(),
                'type': self.position.token_type,
                'entry_price': self.position.entry_price,
                'exit_price': final_value,
                'entry_btc': self.position.entry_btc_price,
                'exit_btc': final_btc,
                'strike_price': strike,
                'probability_at_entry': self.position.probability_at_entry,
                'choppiness_at_entry': self.position.choppiness_at_entry,
                'time_to_expiry_at_entry': self.position.time_to_expiry,
                'pnl': pnl,
                'result': 'WIN' if pnl > 0 else 'LOSS'
            }

            self.today_trades.append(trade)
            self.save_trades()

            logger.info(f"{'='*60}\n")

            self.position = None
            self.last_position_close_time = time.time()

        except Exception as e:
            logger.error(f"❌ Error closing position: {e}")
            import traceback
            traceback.print_exc()

    def can_open_position(self) -> bool:
        """Check if we can open a new position"""
        return self.position is None and self.pending_order_id is None

    def get_daily_pnl(self) -> float:
        """Get today's total PNL"""
        return sum(t['pnl'] for t in self.today_trades)

    def run(self):
        """Main trading loop"""
        logger.info("\n🚀 Starting Probability-Based Trading Bot\n")

        try:
            while True:
                now = datetime.now()
                current_minute = now.minute
                current_second = now.second

                # Determine current period
                period_start_min = None
                for start_min in [0, 15, 30, 45]:
                    if current_minute >= start_min and current_minute < start_min + 15:
                        period_start_min = start_min
                        break

                if period_start_min is None:
                    time.sleep(1)
                    continue

                # Calculate time remaining
                seconds_into_period = (current_minute - period_start_min) * 60 + current_second
                seconds_remaining = 900 - seconds_into_period
                in_buffer_zone = seconds_remaining <= BUFFER_SECONDS

                # Period start
                period_start = now.replace(minute=period_start_min, second=0, microsecond=0)

                # NEW PERIOD DETECTED
                if period_start_min != self.current_period_start:
                    # CRITICAL: Cancel any pending orders IMMEDIATELY when period changes
                    # This prevents orders from previous period being filled with new period's asset_id
                    if self.pending_order_id:
                        logger.warning(f"\n⚠️  PERIOD CHANGE - Canceling pending order immediately!")
                        self.trader.cancel_all_orders()
                        self.pending_order_id = None
                        self.pending_order_type = None
                        self.pending_order_token_id = None
                        self.pending_order_limit_price = None

                    # Close position from previous period at expiry
                    if self.position is not None and self.current_period_start is not None:
                        logger.info(f"\n⚠️  PERIOD END - Fetching candle for settlement")

                        prev_period_start = now.replace(minute=self.current_period_start, second=0, microsecond=0)
                        candle = fetch_period_candle(prev_period_start)

                        if candle:
                            self.close_position_at_expiry(candle)
                        else:
                            logger.error(f"❌ Could not fetch candle for settlement")
                            self.position = None

                    # Update period
                    self.current_period_start = period_start_min
                    self.period_change_time = time.time()  # Record when period changed

                    logger.info(f"\n{'='*80}")
                    logger.info(f"🔄 NEW PERIOD: {now.strftime('%H:%M')} (:{period_start_min:02d})")
                    logger.info(f"⏳ Waiting {PERIOD_CHANGE_WAIT} seconds before loading new asset IDs...")
                    logger.info(f"{'='*80}\n")

                # Wait 20 seconds after period change before loading new data
                if self.period_change_time and (time.time() - self.period_change_time < PERIOD_CHANGE_WAIT):
                    time_left = PERIOD_CHANGE_WAIT - (time.time() - self.period_change_time)
                    if int(time_left) != int(time_left + 1):  # Log once per second
                        logger.info(f"⏳ Waiting {time_left:.0f}s before loading new period data...")
                    time.sleep(1)
                    continue

                # Load new period data after waiting period
                if self.period_change_time and (time.time() - self.period_change_time >= PERIOD_CHANGE_WAIT):
                    if not self.strike_price:  # Only load once
                        logger.info(f"\n✅ Wait period complete - Loading new period data")

                        # Get strike price
                        self.strike_price = get_strike_price()
                        logger.info(f"✅ Strike: ${self.strike_price:.2f}" if self.strike_price else "❌ Strike: Not available")

                        # Reload asset IDs
                        self.reload_asset_ids()

                        logger.info(f"✅ Ready to trade in new period\n")
                        self.period_change_time = None  # Clear the flag

                # Reload asset IDs every 60s
                if time.time() - self.last_asset_reload >= 60:
                    self.reload_asset_ids()

                # Verify position every 60s
                if time.time() - self.last_position_check >= 60:
                    self.verify_position_from_wallet()

                # Refresh cached USDC balance every 10s
                self.refresh_usdc_balance()

                # Get BTC price
                btc_price = get_btc_price()
                if btc_price:
                    self.last_btc_price = btc_price

                    # Update price history for choppiness calculation
                    self.price_history.append(btc_price)
                    if len(self.price_history) > self.max_history_length:
                        self.price_history.pop(0)

                # Read Black-Scholes data
                bs_data = read_json(BS_DATA_FILE)
                if not bs_data:
                    time.sleep(1)
                    continue

                # Extract probabilities
                call_prob = bs_data.get('call_probability', 0)
                put_prob = bs_data.get('put_probability', 0)

                # Calculate actual choppiness from price history
                choppiness = calculate_choppiness(self.price_history, period=900)

                # Check if pending order should be filled
                if self.pending_order_id:
                    if self.check_order_filled():
                        logger.info(f"✅ Position opened via limit order fill")
                    else:
                        # Check if probability dropped - cancel order
                        if self.pending_order_type == 'CALL' and call_prob < PROBABILITY_THRESHOLD:
                            logger.info(f"\n⚠️  CALL probability dropped: {call_prob*100:.2f}% < {PROBABILITY_THRESHOLD*100:.1f}%")
                            logger.info(f"🔄 Canceling order...")
                            self.trader.cancel_all_orders()
                            self.pending_order_id = None
                            self.pending_order_type = None
                            self.pending_order_token_id = None
                            self.pending_order_limit_price = None
                        elif self.pending_order_type == 'PUT' and put_prob < PROBABILITY_THRESHOLD:
                            logger.info(f"\n⚠️  PUT probability dropped: {put_prob*100:.2f}% < {PROBABILITY_THRESHOLD*100:.1f}%")
                            logger.info(f"🔄 Canceling order...")
                            self.trader.cancel_all_orders()
                            self.pending_order_id = None
                            self.pending_order_type = None
                            self.pending_order_token_id = None
                            self.pending_order_limit_price = None
                        else:
                            # Check if market bid has changed significantly - update order
                            # Get current market data
                            call_data = read_json(CALL_FILE)
                            put_data = read_json(PUT_FILE)

                            current_market_bid = 0

                            if self.pending_order_type == 'CALL' and call_data:
                                call_bid_data = call_data.get('best_bid')
                                current_market_bid = call_bid_data.get('price', 0) if call_bid_data else 0
                            elif self.pending_order_type == 'PUT' and put_data:
                                put_bid_data = put_data.get('best_bid')
                                current_market_bid = put_bid_data.get('price', 0) if put_bid_data else 0

                            # If market bid has improved (increased) and we're using a lower bid price, update order
                            if current_market_bid > 0 and self.pending_order_limit_price:
                                # Determine what our new limit price should be
                                if current_market_bid <= BID_PRICE_THRESHOLD:
                                    new_limit_price = min(current_market_bid, MAX_LIMIT_PRICE)
                                else:
                                    new_limit_price = LIMIT_PRICE

                                new_limit_price = min(new_limit_price, MAX_LIMIT_PRICE)

                                # If new price is different from our current order, update it
                                if abs(new_limit_price - self.pending_order_limit_price) >= 0.01:
                                    logger.info(f"\n⚠️  Market bid changed: ${self.pending_order_limit_price:.3f} → ${current_market_bid:.3f}")
                                    logger.info(f"🔄 Updating order: ${self.pending_order_limit_price:.3f} → ${new_limit_price:.3f}")

                                    # Cancel old order
                                    self.trader.cancel_all_orders()
                                    time.sleep(0.5)  # Brief pause

                                    # Place new order at updated price
                                    success = self.place_limit_order(
                                        self.pending_order_type,
                                        self.pending_order_token_id,
                                        new_limit_price,
                                        btc_price if btc_price else 0,
                                        self.strike_price if self.strike_price else 0,
                                        call_prob if self.pending_order_type == 'CALL' else put_prob,
                                        choppiness,
                                        seconds_remaining
                                    )

                                    if not success:
                                        logger.error(f"❌ Failed to replace order")
                                        self.pending_order_id = None
                                        self.pending_order_type = None
                                        self.pending_order_token_id = None
                                        self.pending_order_limit_price = None

                # Display status every 10 seconds
                if current_second % 10 == 0:
                    pos_str = f"{self.position.token_type}@${self.position.entry_price:.3f}" if self.position else "NONE"
                    pending_str = f"PENDING-{self.pending_order_type}" if self.pending_order_id else "NONE"



                    print(f"[{now.strftime('%H:%M:%S')}] BTC:${btc_price:.2f} | "
                          f"Call-Prob:{call_prob*100:.2f}% Put-Prob:{put_prob*100:.2f}% | "
                          f"Chop:{choppiness:.1f} | TTL:{seconds_remaining:.0f}s | "
                          f"Pos:{pos_str} | Pending:{pending_str} | PNL:{self.get_daily_pnl():+.2f}")

                # Read market data for bid prices
                call_data = read_json(CALL_FILE)
                put_data = read_json(PUT_FILE)

                call_bid = 0
                put_bid = 0

                if call_data:
                    call_bid_data = call_data.get('best_bid')
                    call_bid = call_bid_data.get('price', 0) if call_bid_data else 0

                if put_data:
                    put_bid_data = put_data.get('best_bid')
                    put_bid = put_bid_data.get('price', 0) if put_bid_data else 0

                # TRADING LOGIC: Place order if probability threshold met
                # CRITICAL: Only trade if we have enough time (not in buffer zone)
                # AND not in the 20-second waiting period after period change
                in_waiting_period = self.period_change_time and (time.time() - self.period_change_time < PERIOD_CHANGE_WAIT)

                if (not in_buffer_zone and
                    not in_waiting_period and
                    self.can_open_position() and
                    self.strike_price and
                    choppiness <= MAX_CHOPPINESS):

                    # Check CALL probability
                    if call_prob > PROBABILITY_THRESHOLD:
                        # Determine limit price based on bid
                        if call_bid > 0 and call_bid <= BID_PRICE_THRESHOLD:
                            limit_price = min(call_bid, MAX_LIMIT_PRICE)
                            logger.info(f"\n🔔 CALL SIGNAL: Probability {call_prob*100:.2f}% > {PROBABILITY_THRESHOLD*100:.1f}%")
                            logger.info(f"💰 Bid ${call_bid:.3f} <= ${BID_PRICE_THRESHOLD:.2f}, using bid as limit price")
                        else:
                            limit_price = LIMIT_PRICE
                            logger.info(f"\n🔔 CALL SIGNAL: Probability {call_prob*100:.2f}% > {PROBABILITY_THRESHOLD*100:.1f}%")

                        self.place_limit_order(
                            'CALL', self.current_call_id, limit_price,
                            btc_price, self.strike_price, call_prob,
                            choppiness, seconds_remaining
                        )

                    # Check PUT probability
                    elif put_prob > PROBABILITY_THRESHOLD:
                        # Determine limit price based on bid
                        if put_bid > 0 and put_bid <= BID_PRICE_THRESHOLD:
                            limit_price = min(put_bid, MAX_LIMIT_PRICE)
                            logger.info(f"\n🔔 PUT SIGNAL: Probability {put_prob*100:.2f}% > {PROBABILITY_THRESHOLD*100:.1f}%")
                            logger.info(f"💰 Bid ${put_bid:.3f} <= ${BID_PRICE_THRESHOLD:.2f}, using bid as limit price")
                        else:
                            limit_price = LIMIT_PRICE
                            logger.info(f"\n🔔 PUT SIGNAL: Probability {put_prob*100:.2f}% > {PROBABILITY_THRESHOLD*100:.1f}%")

                        self.place_limit_order(
                            'PUT', self.current_put_id, limit_price,
                            btc_price, self.strike_price, put_prob,
                            choppiness, seconds_remaining
                        )

                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("\n\n⏸️  Stopped by user")

            # Cancel any pending orders
            if self.pending_order_id:
                logger.info(f"🔄 Canceling pending orders...")
                self.trader.cancel_all_orders()

            # If position exists at period end, fetch candle for settlement
            if self.position:
                logger.info(f"\n⚠️  Position still open, fetching candle for settlement...")
                period_start = now.replace(minute=self.current_period_start, second=0, microsecond=0)
                candle = fetch_period_candle(period_start)

                if candle:
                    self.close_position_at_expiry(candle)
                else:
                    logger.error(f"❌ Could not fetch candle")

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
        bot = BotProbability(credentials)
        bot.run()

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
