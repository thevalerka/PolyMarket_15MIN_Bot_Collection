#!/usr/bin/env python3
"""
Last-Minute Volatility Scalping Bot - PRODUCTION
Pure volatility-based strategy trading only in the final 60 seconds.

Entry Rules:
1. Last 60 seconds: Open if |distance| > volatility-long
2. Last 20 seconds: Open if |distance| > volatility-short
3. If no ask (only 0.99 bid): Open at 0.99 if conditions met

Direction:
- distance > 0 (above strike) → CALL
- distance < 0 (below strike) → PUT

Exit:
- At expiry: Compare 15-min candle close vs strike

pm2 start bot_last_minute_production.py --cron-restart="00 * * * *" --interpreter python3
"""

import json
import time
import sys
import requests
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
from typing import Optional, Dict
from collections import deque
from dataclasses import dataclass
import numpy as np
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
STATE_FILE = "/home/ubuntu/013_2025_polymarket/bot017_RSI/bot_last_minute_state.json"
TRADES_DIR = "/home/ubuntu/013_2025_polymarket/bot017_RSI/bot_last_minute_trades"

# Trading Parameters
POSITION_SIZE = 5.0  # 5 shares
MIN_BUY_PRICE = 0.20  # Minimum price to open position
MAX_BUY_PRICE = 0.99  # Can buy at 0.99 if deep ITM
MAX_DATA_AGE_MS = 1000  # Maximum age of market data (1 second)
BUFFER_SECONDS = 2  # No trading in last 10s of period
MAX_CHOPPINESS = 30

# Trading Windows
ENTRY_WINDOW_LONG = 60   # Last 60 seconds
ENTRY_WINDOW_SHORT = 20  # Last 20 seconds

# Volatility Parameters
VOLATILITY_LONG_PERIODS = 12   # 12 minutes of 1-minute candles
VOLATILITY_SHORT_PERIODS = 12  # 120 seconds / 10 seconds = 12 periods

# Choppiness
CHOPPINESS_PERIOD = 900  # 15 minutes in seconds

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
    distance_at_entry: float
    volatility_long: float
    volatility_short: float
    choppiness: float
    time_to_expiry: float
    reason: str

# ============================================================================
# MARKET DATA TRACKER
# ============================================================================

class MarketDataTracker:
    """Track market data for volatility and choppiness calculations"""

    def __init__(self):
        # 1-minute candle data (for volatility-long)
        self.minute_candles = deque(maxlen=VOLATILITY_LONG_PERIODS)
        self.current_minute_high = None
        self.current_minute_low = None
        self.last_minute_timestamp = None

        # 10-second range data (for volatility-short)
        self.short_ranges = deque(maxlen=VOLATILITY_SHORT_PERIODS)
        self.current_10s_high = None
        self.current_10s_low = None
        self.last_10s_timestamp = None

        # Price history for choppiness (15 minutes = 900 samples)
        self.price_history = deque(maxlen=CHOPPINESS_PERIOD)

    def update(self, timestamp: datetime, price: float):
        """Update all tracking data structures"""

        # Add to price history
        self.price_history.append(price)

        # Update 1-minute candles
        current_minute = timestamp.replace(second=0, microsecond=0)
        if self.last_minute_timestamp is None or current_minute > self.last_minute_timestamp:
            # New minute started
            if self.current_minute_high is not None:
                # Save completed candle
                candle_range = self.current_minute_high - self.current_minute_low
                self.minute_candles.append(candle_range)

            # Start new candle
            self.current_minute_high = price
            self.current_minute_low = price
            self.last_minute_timestamp = current_minute
        else:
            # Update current candle
            self.current_minute_high = max(self.current_minute_high, price)
            self.current_minute_low = min(self.current_minute_low, price)

        # Update 10-second ranges
        current_10s = timestamp.replace(second=(timestamp.second // 10) * 10, microsecond=0)
        if self.last_10s_timestamp is None or current_10s > self.last_10s_timestamp:
            # New 10-second period started
            if self.current_10s_high is not None:
                # Save completed range
                range_value = self.current_10s_high - self.current_10s_low
                self.short_ranges.append(range_value)

            # Start new range
            self.current_10s_high = price
            self.current_10s_low = price
            self.last_10s_timestamp = current_10s
        else:
            # Update current range
            self.current_10s_high = max(self.current_10s_high, price)
            self.current_10s_low = min(self.current_10s_low, price)

    def get_volatility_long(self) -> float:
        """Calculate EMA of 1-minute candle ranges"""
        if len(self.minute_candles) < 2:
            return 0.0

        ranges = list(self.minute_candles)

        if len(ranges) < VOLATILITY_LONG_PERIODS:
            # Not enough data, use simple average
            return np.mean(ranges)

        # Calculate EMA
        multiplier = 2.0 / (VOLATILITY_LONG_PERIODS + 1)
        ema = ranges[0]

        for i in range(1, len(ranges)):
            ema = (ranges[i] * multiplier) + (ema * (1 - multiplier))

        return ema

    def get_volatility_short(self) -> float:
        """Calculate EMA of 10-second ranges"""
        if len(self.short_ranges) < 2:
            return 0.0

        ranges = list(self.short_ranges)

        if len(ranges) < VOLATILITY_SHORT_PERIODS:
            # Not enough data, use simple average
            return np.mean(ranges)

        # Calculate EMA
        multiplier = 2.0 / (VOLATILITY_SHORT_PERIODS + 1)
        ema = ranges[0]

        for i in range(1, len(ranges)):
            ema = (ranges[i] * multiplier) + (ema * (1 - multiplier))

        return ema

    def get_choppiness(self) -> float:
        """Calculate Choppiness Index (0-100)"""
        if len(self.price_history) < 10:
            return 50.0

        prices = np.array(list(self.price_history))

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
        period = len(prices)
        choppiness = 100 * np.log10(sum_tr / high_low_range) / np.log10(period)

        return max(0.0, min(100.0, choppiness))

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
    """Get strike price from Bybit API"""
    try:
        now = datetime.now(timezone.utc)
        current_minute = now.minute

        for start_min in [0, 15, 30, 45]:
            if current_minute >= start_min and current_minute < start_min + 15:
                period_start = now.replace(minute=start_min, second=0, microsecond=0)
                start_timestamp = int(period_start.timestamp() * 1000)

                url = "https://api.bybit.com/v5/market/mark-price-kline"
                params = {
                    'category': 'linear',
                    'symbol': 'BTCUSDT',
                    'interval': '15',
                    'start': start_timestamp,
                    'limit': 1
                }

                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('retCode') == 0:
                        kline_list = data.get('result', {}).get('list', [])
                        if kline_list:
                            return float(kline_list[0][1])
        return None
    except:
        return None

def fetch_period_candle(period_start: datetime) -> Optional[Dict]:
    """
    Fetch the 15-minute candle for a specific period from Bybit
    Returns: Dict with 'open', 'high', 'low', 'close' or None
    """
    try:
        url = "https://api.bybit.com/v5/market/mark-price-kline"
        params = {
            'category': 'linear',
            'symbol': 'BTCUSDT',
            'interval': '15',
            'start': int(period_start.timestamp() * 1000),
            'limit': 1
        }

        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if data.get('retCode') != 0:
            return None

        candles = data.get('result', {}).get('list', [])
        if not candles:
            return None

        candle = candles[0]
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

# ============================================================================
# BOT CLASS
# ============================================================================

class BotLastMinute:
    """Production last-minute trading bot with Polymarket integration"""

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

        # Market data tracker
        self.tracker = MarketDataTracker()

        # Period tracking
        self.strike_price: Optional[float] = None
        self.current_period_start: Optional[int] = None

        # Last valid prices
        self.last_call_bid: Optional[float] = None
        self.last_put_bid: Optional[float] = None
        self.last_call_ask: Optional[float] = None
        self.last_put_ask: Optional[float] = None
        self.last_btc_price: Optional[float] = None

        # Position verification
        self.last_position_check = time.time()
        self.last_asset_reload = time.time()

        # Cached USDC balance
        self.cached_usdc_balance = 0.0
        self.last_usdc_check = 0.0
        self.usdc_check_interval = 10.0

        # Load state
        self.load_state()

        logger.info("="*80)
        logger.info("🤖 LAST MINUTE VOLATILITY SCALPING BOT")
        logger.info("="*80)
        logger.info(f"Position Size: {POSITION_SIZE} shares")
        logger.info(f"Entry Window: Last 60s (long vol) / Last 20s (short vol)")
        logger.info(f"Buffer Zone: {BUFFER_SECONDS}s")
        logger.info("="*80)

    def load_state(self):
        """Load bot state from persistent file"""
        state = read_json(STATE_FILE)
        if state:
            saved_history = state.get('price_history', [])
            if saved_history:
                self.tracker.price_history = deque(saved_history, maxlen=CHOPPINESS_PERIOD)
            logger.info(f"📥 Loaded state: {len(self.tracker.price_history)} price samples")
        else:
            logger.info(f"📥 No saved state, starting fresh")

    def save_state(self):
        """Save bot state to persistent file"""
        state = {
            'price_history': list(self.tracker.price_history),
            'timestamp': datetime.now().isoformat()
        }
        try:
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def get_today_filename(self) -> Path:
        """Get filename for today's trades"""
        today = date.today().strftime('%Y%m%d')
        return self.trades_dir / f"last_minute_{today}.json"

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
            'strategy': 'Last_Minute_Volatility',
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
                # Sync position
                put_data = read_json(PUT_FILE)
                price = 0.50
                if put_data and put_data.get('best_bid'):
                    price = (put_data['best_bid'].get('price', 0.50) + put_data['best_ask'].get('price', 0.50)) / 2

                self.position = Position(
                    token_type='PUT',
                    token_id=self.current_put_id,
                    entry_price=price,
                    entry_time=time.time(),
                    quantity=put_balance,
                    entry_btc_price=0,
                    strike_price=self.strike_price if self.strike_price else 0,
                    distance_at_entry=0,
                    volatility_long=0,
                    volatility_short=0,
                    choppiness=50,
                    time_to_expiry=0,
                    reason="SYNCED"
                )

        # Case 2: CALL position
        elif has_call and not has_put:
            if self.position is None or self.position.token_type != 'CALL':
                logger.warning(f"⚠️  Wallet has CALL ({call_balance:.2f}), tracking: {self.position.token_type if self.position else 'None'}")
                # Sync position
                call_data = read_json(CALL_FILE)
                price = 0.50
                if call_data and call_data.get('best_bid'):
                    price = (call_data['best_bid'].get('price', 0.50) + call_data['best_ask'].get('price', 0.50)) / 2

                self.position = Position(
                    token_type='CALL',
                    token_id=self.current_call_id,
                    entry_price=price,
                    entry_time=time.time(),
                    quantity=call_balance,
                    entry_btc_price=0,
                    strike_price=self.strike_price if self.strike_price else 0,
                    distance_at_entry=0,
                    volatility_long=0,
                    volatility_short=0,
                    choppiness=50,
                    time_to_expiry=0,
                    reason="SYNCED"
                )

        # Case 3: No position
        elif not has_put and not has_call:
            if self.position is not None:
                logger.warning(f"⚠️  Wallet empty but tracking shows {self.position.token_type}")
                self.position = None

        self.last_position_check = time.time()

    def execute_buy(self, token_type: str, token_id: str, ask_price: float,
                    btc_price: float, strike_price: float, distance: float,
                    vol_long: float, vol_short: float, choppiness: float,
                    time_to_expiry: float, reason: str) -> bool:
        """Execute buy order with Fill-or-Kill"""
        try:
            logger.info(f"\n{'='*70}")
            logger.info(f"🛒 EXECUTING BUY ORDER")
            logger.info(f"{'='*70}")
            logger.info(f"📊 Token: {token_type}")
            logger.info(f"📦 Size: {POSITION_SIZE} shares")
            logger.info(f"💰 Ask Price: ${ask_price:.4f}")
            logger.info(f"📈 Distance: ${distance:+.2f}")
            logger.info(f"⏱️  Time to Expiry: {time_to_expiry:.0f}s")
            logger.info(f"📝 Reason: {reason}")

            required = ask_price * POSITION_SIZE
            logger.info(f"💵 Expected Cost: ${required:.2f}")

            # Use cached USDC balance
            cached_balance = self.cached_usdc_balance
            logger.info(f"💰 USDC Balance (cached): ${cached_balance:.2f}")

            MIN_BALANCE = 4.90
            if cached_balance < MIN_BALANCE:
                logger.error(f"❌ INSUFFICIENT BALANCE: ${cached_balance:.2f} < ${MIN_BALANCE:.2f}")
                return False

            if cached_balance < required:
                logger.error(f"❌ Need: ${required:.2f}, Have: ${cached_balance:.2f}")
                return False

            # PHASE 1: Place order
            logger.info(f"\n🚀 PHASE 1: Placing Fill-or-Kill buy order...")

            start_time = time.time()
            try:
                order_id = self.trader.place_buy_order(
                    token_id=token_id,
                    price=ask_price,
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
            else:
                logger.info(f"✅ Fill-or-Kill: Order filled in 1s!")

            # PHASE 2: Verify
            logger.info("\n🔍 PHASE 2: Verifying...")

            verification_times = [4, 9]
            position_confirmed = True

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
                logger.info(f"\n✅ SUCCESS: Position opened")

                self.position = Position(
                    token_type=token_type,
                    token_id=token_id,
                    entry_price=ask_price,
                    entry_time=time.time(),
                    quantity=final_balance,
                    entry_btc_price=btc_price,
                    strike_price=strike_price,
                    distance_at_entry=distance,
                    volatility_long=vol_long,
                    volatility_short=vol_short,
                    choppiness=choppiness,
                    time_to_expiry=time_to_expiry,
                    reason=reason
                )

                logger.info(f"📊 Position: {final_balance:.2f} @ ${ask_price:.4f}")
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
                'distance_at_entry': self.position.distance_at_entry,
                'volatility_long': self.position.volatility_long,
                'volatility_short': self.position.volatility_short,
                'choppiness': self.position.choppiness,
                'time_to_expiry_at_entry': self.position.time_to_expiry,
                'pnl': pnl,
                'result': 'WIN' if pnl > 0 else 'LOSS',
                'reason': self.position.reason
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
        return self.position is None

    def get_daily_pnl(self) -> float:
        """Get today's total PNL"""
        return sum(t['pnl'] for t in self.today_trades)

    def determine_entry_signal(self, choppiness: float, distance: float, vol_long: float, vol_short: float,
                               time_to_expiry: float, call_ask: Optional[float], call_bid: Optional[float],
                               put_ask: Optional[float], put_bid: Optional[float]) -> Optional[Dict]:
        """
        Determine if we should enter a position

        Returns: Dict with 'type', 'price', 'reason' or None
        """
        # Must have volatility data
        if vol_long <= 0:
            return None

        # Determine which threshold to use
        if time_to_expiry <= ENTRY_WINDOW_SHORT:
            vol_threshold = vol_short if vol_short > 0 else vol_long
            window_name = "SHORT"
        else:
            vol_threshold = vol_long
            window_name = "LONG"

        # Check if distance exceeds volatility
        if abs(distance) <= vol_threshold:
            return None

        # Check if distance exceeds volatility
        if choppiness > MAX_CHOPPINESS:
            return None

        # Determine direction
        if distance > 0:
            # Price above strike → CALL
            token_type = 'CALL'

            # Check if we have a valid ask price
            if call_ask is not None and MIN_BUY_PRICE <= call_ask <= MAX_BUY_PRICE:
                price = call_ask
                reason = f"{window_name}_VOL: dist={distance:+.2f} > {vol_threshold:.2f}"
            elif call_bid is not None and call_bid >= MAX_BUY_PRICE:
                # Deep ITM - only 0.99 bid available
                price = MAX_BUY_PRICE
                reason = f"{window_name}_VOL: dist={distance:+.2f} > {vol_threshold:.2f} (DEEP_ITM)"
            else:
                return None

        else:
            # Price below strike → PUT
            token_type = 'PUT'

            # Check if we have a valid ask price
            if put_ask is not None and MIN_BUY_PRICE <= put_ask <= MAX_BUY_PRICE:
                price = put_ask
                reason = f"{window_name}_VOL: dist={distance:+.2f} < -{vol_threshold:.2f}"
            elif put_bid is not None and put_bid >= MAX_BUY_PRICE:
                # Deep ITM - only 0.99 bid available
                price = MAX_BUY_PRICE
                reason = f"{window_name}_VOL: dist={distance:+.2f} < -{vol_threshold:.2f} (DEEP_ITM)"
            else:
                return None

        return {
            'type': token_type,
            'price': price,
            'reason': reason
        }

    def run(self):
        """Main trading loop"""
        logger.info("\n🚀 Starting Last Minute Volatility Bot\n")

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
                period_end = period_start + timedelta(minutes=15)

                # NEW PERIOD DETECTED
                if period_start_min != self.current_period_start:
                    # Close position from previous period at expiry
                    if self.position is not None and self.current_period_start is not None:
                        logger.info(f"\n⚠️  PERIOD END - Fetching candle for settlement")

                        # Fetch actual 15-min candle
                        prev_period_start = now.replace(minute=self.current_period_start, second=0, microsecond=0)
                        candle = fetch_period_candle(prev_period_start)

                        if candle:
                            self.close_position_at_expiry(candle)
                        else:
                            logger.error(f"❌ Could not fetch candle for settlement")
                            self.position = None

                    # Update period
                    self.current_period_start = period_start_min

                    # Get strike price
                    self.strike_price = get_strike_price()

                    logger.info(f"\n{'='*80}")
                    logger.info(f"🔄 NEW PERIOD: {now.strftime('%H:%M')} (:{period_start_min:02d})")
                    logger.info(f"✅ Strike: ${self.strike_price:.2f}" if self.strike_price else "❌ Strike: Not available")
                    logger.info(f"{'='*80}\n")

                    self.reload_asset_ids()

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
                if btc_price is None:
                    time.sleep(1)
                    continue

                # Update tracker
                self.tracker.update(now, btc_price)

                # Get metrics
                vol_long = self.tracker.get_volatility_long()
                vol_short = self.tracker.get_volatility_short()
                choppiness = self.tracker.get_choppiness()

                # Calculate distance from strike
                if not self.strike_price:
                    time.sleep(1)
                    continue

                distance = btc_price - self.strike_price

                # Read market data
                call_data = read_json(CALL_FILE)
                put_data = read_json(PUT_FILE)

                if not all([call_data, put_data]):
                    time.sleep(1)
                    continue

                # Check data freshness
                current_time_ms = int(time.time() * 1000)
                call_timestamp = call_data.get('updated_at', 0)
                put_timestamp = put_data.get('updated_at', 0)

                call_age_ms = current_time_ms - call_timestamp
                put_age_ms = current_time_ms - put_timestamp

                if call_age_ms > MAX_DATA_AGE_MS or put_age_ms > MAX_DATA_AGE_MS:
                    logger.debug(f"⚠️  Stale data: CALL={call_age_ms}ms PUT={put_age_ms}ms")
                    time.sleep(1)
                    continue

                # Handle null best_bid/best_ask
                call_bid_data = call_data.get('best_bid')
                call_ask_data = call_data.get('best_ask')
                put_bid_data = put_data.get('best_bid')
                put_ask_data = put_data.get('best_ask')

                call_bid = call_bid_data.get('price', 0) if call_bid_data else 0
                call_ask = call_ask_data.get('price', 0) if call_ask_data else None
                put_bid = put_bid_data.get('price', 0) if put_bid_data else 0
                put_ask = put_ask_data.get('price', 0) if put_ask_data else None

                # Display (every second in last 20s, otherwise every 10s)
                if seconds_remaining <= 20 or (current_second % 10 == 0):
                    pos_str = f"{self.position.token_type}@${self.position.entry_price:.2f}" if self.position else "NONE"
                    prefix = "🔥" if seconds_remaining <= 20 else "  "

                    call_ask_str = f"${call_ask:.2f}" if call_ask else "ITM" if call_bid >= 0.99 else "N/A"
                    put_ask_str = f"${put_ask:.2f}" if put_ask else "ITM" if put_bid >= 0.99 else "N/A"

                    print(f"{prefix}[{now.strftime('%H:%M:%S')}] BTC:${btc_price:.2f} | Dist:${distance:+.2f} | "
                          f"Vol-L:${vol_long:.2f} Vol-S:${vol_short:.2f} | Chop:{choppiness:.1f} | "
                          f"C:{call_bid:.2f}/{call_ask_str} P:{put_bid:.2f}/{put_ask_str} | "
                          f"TTL:{seconds_remaining:.0f}s | Pos:{pos_str}")

                # TRADING LOGIC: Only in last 60 seconds, not in buffer, no position
                if (seconds_remaining <= ENTRY_WINDOW_LONG and
                    not in_buffer_zone and
                    self.position is None and
                    self.strike_price):

                    signal = self.determine_entry_signal(
                        distance, choppiness, vol_long, vol_short, seconds_remaining,
                        call_ask, call_bid, put_ask, put_bid
                    )

                    if signal and self.can_open_position():
                        token_id = self.current_call_id if signal['type'] == 'CALL' else self.current_put_id

                        self.execute_buy(
                            signal['type'], token_id, signal['price'],
                            btc_price, self.strike_price, distance,
                            vol_long, vol_short, choppiness,
                            seconds_remaining, signal['reason']
                        )

                # Save state periodically
                if current_second % 60 == 0:
                    self.save_state()

                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("\n\n⏸️  Stopped by user")

            # Save state
            self.save_state()

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
            env_path = '/home/ubuntu/013_2025_polymarket/keys/keys_ovh39.env'
            credentials = load_credentials_from_env(env_path)
            print(f"✅ Credentials loaded from {env_path}")
        except Exception as e:
            print(f"❌ Error loading credentials: {e}")
            return

        # Create and run bot
        bot = BotLastMinute(credentials)
        bot.run()

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
