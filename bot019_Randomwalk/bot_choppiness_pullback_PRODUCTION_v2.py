#!/usr/bin/env python3
"""
Strategy DBID - Adaptive Choppiness Trading with Bid Execution (PRODUCTION V2)
Enhanced with:
- Order cancellation when conditions change (new max detected)
- 20-second minimum between orders
- Continuous wallet verification for position counting
- Orderbook monitoring to detect if we're the best bid (cancel and re-place)

pm2 start bot_choppiness_pullback_PRODUCTION_v2.py --cron-restart="00 */6 * * *" --interpreter python3
"""

import sys
import time
import json
import requests
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from collections import deque
import logging

# Import Polymarket trading core
sys.path.insert(0, '/home/ubuntu')
from polymarket_trading_core_debug import PolymarketTrader, load_credentials_from_env

# Setup logging
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
STATE_FILE = "/home/ubuntu/013_2025_polymarket/bot_choppiness_pullback/state.json"
TRADES_DIR = "/home/ubuntu/013_2025_polymarket/bot_choppiness_pullback/trades"

# Trading Parameters
CHECK_INTERVAL = 1  # Check every second
CHOPPINESS_THRESHOLD = 30
CHOPPINESS_THRESHOLD_imbalance = 15
PULLBACK_AMOUNT = 0.03  # Initial value, will be adapted
MAX_POSITIONS_PER_SIDE = 5  # Initial value, will be adapted
MAX_POSITIONS_PER_SIDE_HARDCAP = 3  # Hard cap
POSITION_SIZE = 5  # 5 shares per position
MIN_BUY_PRICE = 0.05
MAX_BUY_PRICE = 0.95

# Order management
MIN_SECONDS_BETWEEN_ORDERS = 20  # 20 seconds minimum between orders
WALLET_CHECK_INTERVAL = 10  # Check wallet every 5 seconds
ORDERBOOK_CHECK_INTERVAL = 1 # Check orderbook every 2 seconds

# Buffer times
START_DELAY = 20  # Wait 20s from period start
BUFFER_SECONDS = 20  # No trading in last 20s of period

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

def get_seconds_to_expiry() -> int:
    """Calculate seconds until next 15-minute mark"""
    now = datetime.now()
    minutes_into_quarter = now.minute % 15
    seconds_into_quarter = minutes_into_quarter * 60 + now.second
    return 900 - seconds_into_quarter

def get_seconds_into_period() -> int:
    """Get seconds into current 15-minute period"""
    now = datetime.now()
    minutes_into_quarter = now.minute % 15
    return minutes_into_quarter * 60 + now.second

def get_bin_key(timestamp: float) -> str:
    """Get bin key for current period"""
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    hour = dt.hour
    minute = dt.minute

    for start_min in [0, 15, 30, 45]:
        if minute >= start_min and minute < start_min + 15:
            return f"{hour:02d}:{start_min:02d}"
    return f"{hour:02d}:00"

def calculate_choppiness_index(prices: deque, period: int = 14) -> float:
    """
    Calculate Choppiness Index using True Range method (matches simulator)

    Args:
        prices: Deque of BTC prices
        period: Lookback period (default 14)

    Returns:
        Choppiness Index value (0-100)
    """
    import math
    import numpy as np

    if len(prices) < period + 1:
        return 50.0  # Return neutral value if not enough data

    # Get last period+1 prices to have enough data for True Range calculation
    prices_array = np.array(list(prices)[-(period + 1):])

    # Create highs, lows, closes arrays (for 1-second data, they're all the same price)
    # But we still use the TR formula for consistency with the simulator
    highs = prices_array[1:]  # Last 'period' prices
    lows = prices_array[1:]   # Last 'period' prices
    closes = prices_array[:-1]  # Previous 'period' prices (shifted by 1)

    # Get max high and min low over period
    high_max = np.max(highs)
    low_min = np.min(lows)

    # Calculate ATR sum using True Range formula
    atr_sum = 0.0
    for i in range(len(highs)):
        high_low = highs[i] - lows[i]  # Will be 0 for tick data
        high_close = abs(highs[i] - closes[i])
        low_close = abs(lows[i] - closes[i])
        true_range = max(high_low, high_close, low_close)
        atr_sum += true_range

    # Avoid division by zero
    if high_max - low_min == 0:
        return 100.0  # Completely flat = maximum choppiness

    if atr_sum == 0:
        return 100.0

    # Calculate Choppiness Index
    chop = 100 * math.log10(atr_sum / (high_max - low_min)) / math.log10(period)

    # Clamp to 0-100 range
    chop = max(0.0, min(100.0, chop))

    return chop

# ============================================================================
# POSITION CLASS
# ============================================================================

class Position:
    """Track an open position"""
    def __init__(self, side: str, entry_price: float, size: float, token_id: str,
                 entry_time: Optional[datetime] = None, choppiness: float = 0.0,
                 time_to_expiry_seconds: int = 0, strike_price: float = 0.0):
        self.side = side  # 'CALL' or 'PUT'
        self.entry_price = entry_price
        self.size = size
        self.token_id = token_id
        self.entry_time = entry_time if entry_time else datetime.now(timezone.utc)
        self.choppiness = choppiness
        self.time_to_expiry_seconds = time_to_expiry_seconds
        self.strike_price = strike_price
        self.pnl = 0.0
        self.final_value = 0.0

    def __repr__(self):
        return f"Position({self.side}, entry=${self.entry_price:.3f}, size={self.size}, PNL=${self.pnl:.2f})"

    def to_dict(self):
        """Convert position to dictionary for JSON serialization"""
        return {
            'side': self.side,
            'entry_price': self.entry_price,
            'size': self.size,
            'token_id': self.token_id,
            'entry_time': self.entry_time.isoformat(),
            'choppiness': self.choppiness,
            'time_to_expiry_seconds': self.time_to_expiry_seconds,
            'strike_price': self.strike_price,
            'pnl': self.pnl,
            'final_value': self.final_value
        }

# ============================================================================
# RETRACEMENT TRACKER
# ============================================================================

class RetracementTracker:
    """Track price retracements (swings) for bid and ask prices"""
    def __init__(self, min_retracement: float = 0.04):
        self.min_retracement = min_retracement
        self.reset()

    def reset(self):
        """Reset for new period"""
        self.retracements = []
        self.last_swing_price = None
        self.last_swing_type = None  # 'high' or 'low'
        self.current_high = None
        self.current_low = None

    def add_price(self, price: float, price_type: str):
        """Add new price point and detect retracements"""
        timestamp = datetime.now(timezone.utc)

        # Initialize
        if self.last_swing_price is None:
            self.last_swing_price = price
            self.current_high = price
            self.current_low = price
            return

        # Update current high/low
        if price > self.current_high:
            self.current_high = price
        if price < self.current_low:
            self.current_low = price

        # Check for swing high
        if self.last_swing_type != 'high' and price < self.current_high:
            if self.last_swing_type == 'low':
                retracement_width = self.current_high - self.last_swing_price
                if retracement_width >= self.min_retracement:
                    self.retracements.append({
                        'direction': 'up',
                        'from': self.last_swing_price,
                        'to': self.current_high,
                        'width': retracement_width,
                        'price_type': price_type,
                        'timestamp': timestamp.isoformat()
                    })

            self.last_swing_price = self.current_high
            self.last_swing_type = 'high'
            self.current_low = price

        # Check for swing low
        elif self.last_swing_type != 'low' and price > self.current_low:
            if self.last_swing_type == 'high':
                retracement_width = self.last_swing_price - self.current_low
                if retracement_width >= self.min_retracement:
                    self.retracements.append({
                        'direction': 'down',
                        'from': self.last_swing_price,
                        'to': self.current_low,
                        'width': retracement_width,
                        'price_type': price_type,
                        'timestamp': timestamp.isoformat()
                    })

            self.last_swing_price = self.current_low
            self.last_swing_type = 'low'
            self.current_high = price

    def get_stats(self):
        """Get retracement statistics"""
        if not self.retracements:
            return {
                'count': 0,
                'total_width': 0.0,
                'avg_width': 0.0,
                'max_width': 0.0,
                'min_width': 0.0,
                'up_count': 0,
                'down_count': 0,
                'retracements': []
            }

        widths = [r['width'] for r in self.retracements]
        up_retracements = [r for r in self.retracements if r['direction'] == 'up']
        down_retracements = [r for r in self.retracements if r['direction'] == 'down']

        return {
            'count': len(self.retracements),
            'total_width': sum(widths),
            'avg_width': sum(widths) / len(widths),
            'max_width': max(widths),
            'min_width': min(widths),
            'up_count': len(up_retracements),
            'down_count': len(down_retracements),
            'retracements': self.retracements
        }

# ============================================================================
# TRADING BOT CLASS
# ============================================================================

class ChoppinessPullbackBot:
    """Production trading bot with adaptive choppiness strategy"""

    def __init__(self, credentials: dict):
        # Initialize Polymarket trader
        self.trader = PolymarketTrader(
            clob_api_url=credentials['clob_api_url'],
            private_key=credentials['private_key'],
            api_key=credentials['api_key'],
            api_secret=credentials['api_secret'],
            api_passphrase=credentials['api_passphrase']
        )

        # Asset IDs
        self.current_put_id: Optional[str] = None
        self.current_call_id: Optional[str] = None

        # Position tracking
        self.positions: List[Position] = []
        self.call_positions = 0
        self.put_positions = 0

        # Trades directory
        self.trades_dir = Path(TRADES_DIR)
        self.trades_dir.mkdir(exist_ok=True, parents=True)
        self.today_trades = []
        self.load_today_trades()

        # Period tracking
        self.strike_price: Optional[float] = None
        self.current_period_start: Optional[int] = None
        self.current_bin: Optional[str] = None
        self.start_buffer_reload_done = False

        # Price tracking for max detection
        self.max_call_ask = 0.0
        self.max_put_ask = 0.0

        # Price history for choppiness calculation
        self.price_history = deque(maxlen=900)  # 15 minutes at 1s intervals

        # Retracement trackers
        self.call_bid_retracements = RetracementTracker()
        self.call_ask_retracements = RetracementTracker()
        self.put_bid_retracements = RetracementTracker()
        self.put_ask_retracements = RetracementTracker()

        # Adaptive parameters
        self.current_max_positions = MAX_POSITIONS_PER_SIDE
        self.current_pullback = PULLBACK_AMOUNT

        # Cached USDC balance
        self.cached_usdc_balance = 0.0
        self.last_usdc_check = 0.0
        self.usdc_check_interval = 10.0

        # Last valid prices for period end
        self.last_call_bid: Optional[float] = None
        self.last_put_bid: Optional[float] = None
        self.last_call_ask: Optional[float] = None
        self.last_put_ask: Optional[float] = None

        # Timing
        self.last_order_time = 0  # Track last order placement time
        self.last_position_time = 0
        self.last_save_time = time.time()
        self.last_asset_reload = time.time()
        self.last_position_check = time.time()
        self.last_orderbook_check = time.time()

        # Pending order tracking
        self.pending_order_id: Optional[str] = None
        self.pending_order_side: Optional[str] = None  # 'CALL' or 'PUT'
        self.pending_order_price: Optional[float] = None
        self.pending_order_time: Optional[float] = None

        # Load state
        self.load_state()

        logger.info("="*80)
        logger.info("🤖 BOT CHOPPINESS PULLBACK - PRODUCTION V2")
        logger.info("="*80)
        logger.info(f"Position Size: {POSITION_SIZE} shares")
        logger.info(f"Max Positions: {self.current_max_positions} (adaptive)")
        logger.info(f"Pullback: {self.current_pullback:.3f} (adaptive)")
        logger.info(f"Min Choppiness: {CHOPPINESS_THRESHOLD}")
        logger.info(f"Min Time Between Orders: {MIN_SECONDS_BETWEEN_ORDERS}s")
        logger.info("="*80)

    def load_state(self):
        """Load bot state from file"""
        state = read_json(STATE_FILE)
        if state:
            self.current_max_positions = state.get('current_max_positions', MAX_POSITIONS_PER_SIDE)
            self.current_pullback = state.get('current_pullback', PULLBACK_AMOUNT)
            logger.info(f"📥 Loaded state: max_positions={self.current_max_positions}, pullback={self.current_pullback:.3f}")
        else:
            logger.info(f"📥 No saved state, using defaults")

    def save_state(self):
        """Save bot state to file"""
        state = {
            'current_max_positions': self.current_max_positions,
            'current_pullback': self.current_pullback,
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
        return self.trades_dir / f"choppiness_pullback_{today}.json"

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
            'strategy': 'Choppiness_Pullback_Adaptive',
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

    def check_token_balance(self, token_id: str) -> float:
        """Check balance of specific token"""
        try:
            balance_raw, balance = self.trader.get_token_balance(token_id)
            return balance
        except Exception as e:
            logger.debug(f"Error checking balance for {token_id[:12]}...: {e}")
            return 0.0

    def count_wallet_positions(self) -> Tuple[int, int]:
        """
        Count actual positions in wallet based on token balances
        Returns (call_positions, put_positions) where each position = POSITION_SIZE tokens
        Uses 10% tolerance to handle partial fills and fees
        """
        if not self.current_call_id or not self.current_put_id:
            return 0, 0

        call_balance = self.check_token_balance(self.current_call_id)
        put_balance = self.check_token_balance(self.current_put_id)

        # Calculate number of positions with 10% tolerance
        # Example: 9.929 tokens with POSITION_SIZE=5.2 → 9.929 / (5.2 * 0.9) = 2.12 → 2 positions
        tolerance = 0.9  # 10% tolerance
        min_position_size = POSITION_SIZE * tolerance

        call_positions = int(call_balance / min_position_size)
        put_positions = int(put_balance / min_position_size)

        return call_positions, put_positions

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
        """Verify positions match wallet and sync if needed"""
        if not self.current_put_id or not self.current_call_id:
            return

        # Count actual positions from wallet
        wallet_call_positions, wallet_put_positions = self.count_wallet_positions()

        # Sync internal tracking with wallet and reset max if position increased
        if wallet_call_positions != self.call_positions:
            if wallet_call_positions > self.call_positions:
                # Position filled! Reset max for next entry
                self.max_call_ask = 0.0
                logger.info(f"✅ CALL position filled: {self.call_positions} → {wallet_call_positions}")
            else:
                logger.warning(f"⚠️  CALL position sync: Tracking={self.call_positions}, Wallet={wallet_call_positions}")
            self.call_positions = wallet_call_positions

        if wallet_put_positions != self.put_positions:
            if wallet_put_positions > self.put_positions:
                # Position filled! Reset max for next entry
                self.max_put_ask = 0.0
                logger.info(f"✅ PUT position filled: {self.put_positions} → {wallet_put_positions}")
            else:
                logger.warning(f"⚠️  PUT position sync: Tracking={self.put_positions}, Wallet={wallet_put_positions}")
            self.put_positions = wallet_put_positions

        self.last_position_check = time.time()

    def check_orderbook_for_our_order(self):
        """
        Check if we're the best bid in the orderbook
        If best_bid size matches our POSITION_SIZE, we're likely overpaying
        Cancel and re-place at new best bid
        """
        if not self.pending_order_id or not self.pending_order_side:
            return

        # Only check every 2 seconds
        if time.time() - self.last_orderbook_check < ORDERBOOK_CHECK_INTERVAL:
            return

        self.last_orderbook_check = time.time()

        try:
            # Read current orderbook
            if self.pending_order_side == 'CALL':
                data = read_json(CALL_FILE)
            else:  # PUT
                data = read_json(PUT_FILE)

            if not data:
                return

            best_bid = data.get('best_bid', {})
            best_bid_price = best_bid.get('price')
            best_bid_size = best_bid.get('size')

            if not best_bid_price or not best_bid_size:
                return

            # Check if best bid size is close to our position size
            # This indicates we might be the best bid (overpaying)
            if abs(best_bid_size - POSITION_SIZE) < 0.5:
                logger.info(f"\n⚠️  ORDERBOOK ALERT: We are likely the best bid!")
                logger.info(f"   Best bid size: {best_bid_size:.2f} ≈ Our size: {POSITION_SIZE:.2f}")
                logger.info(f"   Canceling and re-placing at new best bid...")

                # Cancel current order
                self.trader.cancel_all_orders()
                self.pending_order_id = None
                self.pending_order_side = None
                self.pending_order_price = None

                logger.info(f"✅ Orders cancelled, will re-evaluate on next cycle")

        except Exception as e:
            logger.error(f"Error checking orderbook: {e}")

    def should_cancel_pending_orders(self, market_data: Dict) -> bool:
        """
        Check if we should cancel pending orders because conditions changed
        Returns True if orders should be cancelled
        """
        if not self.pending_order_id or not self.pending_order_side:
            return False

        call_ask = market_data.get('call_ask')
        put_ask = market_data.get('put_ask')

        # Check if new max was detected (invalidates old order)
        if self.pending_order_side == 'CALL':
            if call_ask and call_ask > self.max_call_ask:
                logger.info(f"\n⚠️  NEW MAX CALL ASK: ${call_ask:.3f} > ${self.max_call_ask:.3f}")
                logger.info(f"   Canceling pending CALL order (conditions changed)")
                return True
        else:  # PUT
            if put_ask and put_ask > self.max_put_ask:
                logger.info(f"\n⚠️  NEW MAX PUT ASK: ${put_ask:.3f} > ${self.max_put_ask:.3f}")
                logger.info(f"   Canceling pending PUT order (conditions changed)")
                return True

        return False

    def get_market_data(self) -> Optional[Dict]:
        """Get current market data from JSON files"""
        try:
            call_data = read_json(CALL_FILE)
            put_data = read_json(PUT_FILE)
            btc_price = get_btc_price()

            if not call_data or not put_data or not btc_price:
                return None

            # Extract bid/ask prices with safe access
            best_bid = call_data.get('best_bid')
            best_ask = call_data.get('best_ask')

            call_bid = best_bid.get('price') if best_bid else None
            call_ask = best_ask.get('price') if best_ask else None

            best_bid = put_data.get('best_bid')
            best_ask = put_data.get('best_ask')

            put_bid = best_bid.get('price') if best_bid else None
            put_ask = best_ask.get('price') if best_ask else None

            # Update last valid prices
            if call_bid: self.last_call_bid = call_bid
            if call_ask: self.last_call_ask = call_ask
            if put_bid: self.last_put_bid = put_bid
            if put_ask: self.last_put_ask = put_ask

            # Calculate choppiness
            self.price_history.append(btc_price)
            choppiness = calculate_choppiness_index(self.price_history, period=14)

            return {
                'call_bid': call_bid,
                'call_ask': call_ask,
                'put_bid': put_bid,
                'put_ask': put_ask,
                'btc_price': btc_price,
                'choppiness': choppiness
            }

        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None

    def execute_buy(self, token_type: str, token_id: str, bid_price: float,
                    choppiness: float, seconds_remaining: int) -> bool:
        """Execute MAKER buy order - limit order $0.01 below bid"""
        try:
            # Check if we need to wait between orders
            time_since_last_order = time.time() - self.last_order_time
            if time_since_last_order < MIN_SECONDS_BETWEEN_ORDERS:
                wait_time = MIN_SECONDS_BETWEEN_ORDERS - time_since_last_order
                logger.info(f"⏳ Waiting {wait_time:.0f}s before next order (min {MIN_SECONDS_BETWEEN_ORDERS}s between orders)")
                return False

            logger.info(f"\n{'='*70}")
            logger.info(f"🛒 EXECUTING MAKER BUY ORDER - {token_type}")
            logger.info(f"{'='*70}")
            logger.info(f"📦 Size: {POSITION_SIZE} shares")
            logger.info(f"💰 Market Bid: ${bid_price:.4f}")

            # MAKER: Place limit order $0.01 below bid
            limit_price = max(0.01, bid_price - 0.01)
            logger.info(f"💰 Limit Price: ${limit_price:.4f} (bid - $0.01)")
            logger.info(f"🌊 Choppiness: {choppiness:.1f}")

            required = limit_price * POSITION_SIZE
            logger.info(f"💵 Expected Cost: ${required:.2f}")

            # Check USDC balance
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
            logger.info(f"\n🚀 Placing MAKER limit buy order...")
            start_time = time.time()

            order_id = self.trader.place_buy_order(
                token_id=token_id,
                price=limit_price,
                quantity=POSITION_SIZE
            )

            if not order_id:
                logger.error(f"❌ Failed to place order")
                return False

            logger.info(f"✅ Limit order placed: {order_id[:16]}... ({time.time() - start_time:.3f}s)")
            logger.info(f"⏳ Order will fill when market bid reaches ${limit_price:.4f}")

            # Track pending order
            self.pending_order_id = order_id
            self.pending_order_side = token_type
            self.pending_order_price = limit_price
            self.pending_order_time = time.time()
            self.last_order_time = time.time()

            # DON'T update position counts here - wait for wallet verification
            # Position count will be updated by verify_position_from_wallet()

            # Refresh balance after trade
            self.cached_usdc_balance = self.get_usdc_balance()
            logger.info(f"💰 New USDC Balance: ${self.cached_usdc_balance:.2f}")

            return True

        except Exception as e:
            logger.error(f"❌ Error executing buy: {e}")
            import traceback
            traceback.print_exc()
            return False

    def update_adaptive_parameters(self, call_ask_stats: dict, put_ask_stats: dict):
        """Update adaptive parameters based on retracement stats"""
        call_count = call_ask_stats.get('count', 0)
        put_count = put_ask_stats.get('count', 0)

        if call_count > 0 and put_count > 0:
            # Update MAX_POSITIONS: minimum of call and put retracements
            self.current_max_positions = min(call_count, put_count, MAX_POSITIONS_PER_SIDE)

            # Update PULLBACK: average retracement - 0.01
            call_avg = call_ask_stats.get('avg_width', 0.0)
            put_avg = put_ask_stats.get('avg_width', 0.0)
            avg_retracement = (call_avg + put_avg) / 2
            self.current_pullback = max(0.01, avg_retracement - 0.01)

            logger.info(f"📊 Adaptive Parameters Updated:")
            logger.info(f"   MAX_POSITIONS: {self.current_max_positions} (from retracements: CALL={call_count}, PUT={put_count})")
            logger.info(f"   PULLBACK: ${self.current_pullback:.3f} (from avg retracement: ${avg_retracement:.3f})")

    def check_period_end(self):
        """Check if period has ended and close all positions"""
        now = datetime.now(timezone.utc)
        current_minute = now.minute
        current_second = now.second

        # Check if we're at 00, 15, 30, or 45 minutes (within first 5 seconds)
        is_period_start = (current_minute % 15 == 0) and (current_second <= 5)

        # Only process if we have positions/orders OR if it's the first check of new period
        if not is_period_start:
            return

        # Check if we already processed this period
        current_period_key = f"{now.hour:02d}:{current_minute:02d}"
        if hasattr(self, '_last_processed_period') and self._last_processed_period == current_period_key:
            return  # Already processed this period

        self._last_processed_period = current_period_key

        logger.info(f"\n{'='*80}")
        logger.info(f"⚠️  PERIOD END DETECTED - Processing settlements")
        logger.info(f"{'='*80}")

        # Step 1: Cancel all orders IMMEDIATELY
        logger.info(f"🔄 Step 1: Canceling all open orders...")
        try:
            cancelled_count = self.trader.cancel_all_orders()
            logger.info(f"✅ Cancelled {cancelled_count} orders")
            self.pending_order_id = None
            self.pending_order_side = None
            self.pending_order_price = None
        except Exception as e:
            logger.error(f"❌ Error cancelling orders: {e}")

        # Step 2: Verify actual positions from wallet
        wallet_call_positions, wallet_put_positions = self.count_wallet_positions()
        logger.info(f"📊 Wallet positions: CALL={wallet_call_positions}, PUT={wallet_put_positions}")

        # Step 3: Settle positions
        logger.info(f"🔄 Step 2: Settling positions...")

        # Calculate final PNL based on strike price outcome
        if self.strike_price:
            final_btc = get_btc_price()
            if final_btc:
                # Determine winning side
                call_wins = final_btc > self.strike_price
                put_wins = final_btc < self.strike_price

                # Calculate PNL for all positions
                total_call_value = wallet_call_positions * POSITION_SIZE * (1.0 if call_wins else 0.0)
                total_put_value = wallet_put_positions * POSITION_SIZE * (1.0 if put_wins else 0.0)

                # Estimate average entry price (we'll use 0.5 as fallback)
                avg_entry = 0.5
                total_call_cost = wallet_call_positions * POSITION_SIZE * avg_entry
                total_put_cost = wallet_put_positions * POSITION_SIZE * avg_entry

                call_pnl = total_call_value - total_call_cost
                put_pnl = total_put_value - total_put_cost
                total_pnl = call_pnl + put_pnl

                logger.info(f"   CALL PNL: ${call_pnl:+.2f} ({wallet_call_positions} positions)")
                logger.info(f"   PUT PNL: ${put_pnl:+.2f} ({wallet_put_positions} positions)")
                logger.info(f"   TOTAL PNL: ${total_pnl:+.2f}")
                logger.info(f"   Strike: ${self.strike_price:.2f}, Final: ${final_btc:.2f}")

        # Get retracement stats for adaptive learning
        call_ask_stats = self.call_ask_retracements.get_stats()
        put_ask_stats = self.put_ask_retracements.get_stats()

        logger.info(f"\n📊 Retracement Statistics:")
        if call_ask_stats['count'] > 0:
            logger.info(f"   CALL ASK: {call_ask_stats['count']} retracements, avg: ${call_ask_stats['avg_width']:.3f}")
        if put_ask_stats['count'] > 0:
            logger.info(f"   PUT ASK: {put_ask_stats['count']} retracements, avg: ${put_ask_stats['avg_width']:.3f}")

        # Update adaptive parameters
        self.update_adaptive_parameters(call_ask_stats, put_ask_stats)

        # Save state
        self.save_state()

        # Wait 10 seconds before reloading assets
        logger.info(f"\n⏳ Waiting 10 seconds before asset reload...")
        time.sleep(10)

        # Step 4: Reload asset IDs for new period
        logger.info(f"🔄 Step 3: Reloading asset IDs for new period...")
        self.reload_asset_ids()

        logger.info(f"✅ Ready to trade in new period")
        logger.info(f"{'='*80}\n")

        # Reset for new period
        self.positions = []
        self.call_positions = 0
        self.put_positions = 0
        self.max_call_ask = 0.0
        self.max_put_ask = 0.0
        self.strike_price = None

        # Reset retracement trackers
        self.call_bid_retracements.reset()
        self.call_ask_retracements.reset()
        self.put_bid_retracements.reset()
        self.put_ask_retracements.reset()

    def display_status(self, market_data: Dict):
        """Display current status"""
        now = datetime.now()
        seconds_remaining = get_seconds_to_expiry()
        seconds_into_period = get_seconds_into_period()

        # Build status line
        status = f"\r⏰ {now.strftime('%H:%M:%S')} | "
        status += f"BTC: ${market_data['btc_price']:,.2f} | "
        status += f"Strike: ${self.strike_price:.2f} | " if self.strike_price else "Strike: -- | "
        status += f"Chop: {market_data['choppiness']:.0f} | "
        status += f"Pos: C{self.call_positions}/P{self.put_positions} | "

        # Show pending order status
        if self.pending_order_id:
            status += f"⏳ {self.pending_order_side} @ ${self.pending_order_price:.3f} | "

        status += f"Time: {seconds_remaining}s"

        print(status, end='', flush=True)

    def run(self):
        """Main trading loop"""
        logger.info("🚀 Starting Choppiness Pullback Bot V2 (PRODUCTION)...")
        logger.info("=" * 80)

        while True:
            try:
                now = datetime.now()
                current_time = time.time()
                seconds_into_period = get_seconds_into_period()
                seconds_remaining = get_seconds_to_expiry()

                # Buffer zones
                in_start_buffer = seconds_into_period <= START_DELAY
                in_end_buffer = seconds_remaining <= BUFFER_SECONDS
                in_buffer_zone = in_start_buffer or in_end_buffer

                # Update strike price
                if not self.strike_price:
                    self.strike_price = get_strike_price()

                # Check if we're in period transition (first 15 seconds of new period)
                current_minute = now.minute
                current_second = now.second
                in_period_transition = (current_minute % 15 == 0) and (current_second <= 15)

                # Check period end FIRST (at 00, 15, 30, 45 minutes)
                self.check_period_end()

                # Skip all operations during period transition
                if in_period_transition:
                    time.sleep(CHECK_INTERVAL)
                    continue

                # Get market data
                market_data = self.get_market_data()
                if not market_data:
                    time.sleep(CHECK_INTERVAL)
                    continue

                # Track retracements
                if market_data.get('call_bid'):
                    self.call_bid_retracements.add_price(market_data['call_bid'], 'call_bid')
                if market_data.get('call_ask'):
                    self.call_ask_retracements.add_price(market_data['call_ask'], 'call_ask')
                if market_data.get('put_bid'):
                    self.put_bid_retracements.add_price(market_data['put_bid'], 'put_bid')
                if market_data.get('put_ask'):
                    self.put_ask_retracements.add_price(market_data['put_ask'], 'put_ask')

                # Display status
                self.display_status(market_data)

                # Check if we should cancel pending orders (conditions changed)
                if self.should_cancel_pending_orders(market_data):
                    self.trader.cancel_all_orders()
                    self.pending_order_id = None
                    self.pending_order_side = None
                    self.pending_order_price = None
                    logger.info(f"✅ Pending orders cancelled")

                # Check orderbook to see if we're overpaying
                self.check_orderbook_for_our_order()

                # Reload asset IDs every 60s
                if time.time() - self.last_asset_reload >= 60:
                    self.reload_asset_ids()

                # Verify position from wallet every 5s
                if time.time() - self.last_position_check >= WALLET_CHECK_INTERVAL:
                    self.verify_position_from_wallet()

                # Refresh USDC balance every 10s
                self.refresh_usdc_balance()

                # Skip trading in buffer zones
                if in_buffer_zone:
                    time.sleep(CHECK_INTERVAL)
                    continue

                # Skip trading if we have a pending order
                if self.pending_order_id:
                    time.sleep(CHECK_INTERVAL)
                    continue

                # Only trade if choppiness > threshold
                choppiness = market_data['choppiness']
                if choppiness <= CHOPPINESS_THRESHOLD_imbalance:
                    time.sleep(CHECK_INTERVAL)
                    continue

                # Get prices
                call_ask = market_data.get('call_ask')
                put_ask = market_data.get('put_ask')
                call_bid = market_data.get('call_bid')
                put_bid = market_data.get('put_bid')

                # Update max ASK prices
                if call_ask and call_ask > self.max_call_ask:
                    self.max_call_ask = call_ask

                if put_ask and put_ask > self.max_put_ask:
                    self.max_put_ask = put_ask

                # Se abbiamo lo stesso numero di posizioni serve choppiness elevata
                if (self.call_positions ==  self.put_positions and choppiness > CHOPPINESS_THRESHOLD) or (self.call_positions !=  self.put_positions and choppiness > CHOPPINESS_THRESHOLD_imbalance) :
                    # Check for CALL entry (ASK dropped from max, enter at BID)
                    if self.call_positions < self.current_max_positions and self.call_positions < MAX_POSITIONS_PER_SIDE_HARDCAP:

                        pullback_weighted = (self.current_pullback * choppiness / 10)

                        # Dynamic pullback adjustment based on position balance
                        if self.call_positions < self.put_positions:
                            pullback_weighted = pullback_weighted / 2
                            if self.call_positions >= 3:
                                pullback_weighted = pullback_weighted * 1.33
                        elif self.call_positions > self.put_positions:
                            pullback_weighted = pullback_weighted * 1.5

                        if self.max_call_ask > 0 and call_ask and call_ask <= (self.max_call_ask - pullback_weighted):
                            entry_price = call_bid if call_bid is not None else call_ask

                            if MIN_BUY_PRICE <= entry_price <= MAX_BUY_PRICE:
                                print()  # New line
                                logger.info(f"🔵 CALL entry signal: ASK ${call_ask:.3f} dropped ${pullback_weighted:.3f} from max ${self.max_call_ask:.3f}")
                                logger.info(f"   Entering at BID: ${entry_price:.3f}")

                                if self.execute_buy('CALL', self.current_call_id, entry_price,
                                                   choppiness, seconds_remaining):
                                    logger.info(f"🔵 CALL order placed")

                    # Check for PUT entry (ASK dropped from max, enter at BID)
                    if self.put_positions < self.current_max_positions and self.put_positions < MAX_POSITIONS_PER_SIDE_HARDCAP:
                        pullback_weighted = (self.current_pullback * choppiness / 10)

                        # Dynamic pullback adjustment based on position balance
                        if self.put_positions < self.call_positions:
                            pullback_weighted = pullback_weighted / 2
                            if self.put_positions >= 3:
                                pullback_weighted = pullback_weighted * 1.33
                        elif self.put_positions > self.call_positions:
                            pullback_weighted = pullback_weighted * 1.5

                        if self.max_put_ask > 0 and put_ask and put_ask <= (self.max_put_ask - pullback_weighted):
                            entry_price = put_bid if put_bid is not None else put_ask

                            if MIN_BUY_PRICE <= entry_price <= MAX_BUY_PRICE:
                                print()  # New line
                                logger.info(f"🔴 PUT entry signal: ASK ${put_ask:.3f} dropped ${pullback_weighted:.3f} from max ${self.max_put_ask:.3f}")
                                logger.info(f"   Entering at BID: ${entry_price:.3f}")

                                if self.execute_buy('PUT', self.current_put_id, entry_price,
                                                   choppiness, seconds_remaining):
                                    logger.info(f"🔴 PUT order placed")

                time.sleep(CHECK_INTERVAL)

            except KeyboardInterrupt:
                print("\n\n🛑 Bot stopped by user")
                logger.info("💾 Saving final state before shutdown...")
                self.save_state()
                logger.info("✅ State saved successfully")
                break
            except Exception as e:
                print()  # New line
                logger.error(f"Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    # Load credentials

    # Load credentials
    try:
        env_path = '/home/ubuntu/013_2025_polymarket/keys/keys_ovh38.env'
        credentials = load_credentials_from_env(env_path)
        print(f"✅ Credentials loaded from {env_path}")
    except Exception as e:
        print(f"❌ Error loading credentials: {e}")

    # Start bot
    bot = ChoppinessPullbackBot(credentials)
    bot.run()
