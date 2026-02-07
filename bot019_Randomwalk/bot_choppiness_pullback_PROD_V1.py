#!/usr/bin/env python3
"""
Strategy DBID - Adaptive Choppiness Trading with Bid Execution (PRODUCTION)
- Adapts MAX_POSITIONS based on previous period's retracements
- Adapts PULLBACK based on average retracement width
- Entry trigger: when ASK price drops below (max_ask - pullback)
- Execution: Enter at BID price using LIMIT orders
- Period management: Cancel all orders and settle positions at 00, 15, 30, 45 minutes
- Formula: MAX_POSITIONS = min(call_retracements, put_retracements)
- Formula: pullback = avg_retracement - 0.01

pm2 start bot_choppiness_pullback_PRODUCTION.py --cron-restart="00 */6 * * *" --interpreter python3
"""

import sys
import time
import json
import requests
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Optional, Dict, List
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
STATE_FILE = "/home/ubuntu/013_2025_polymarket/bot_choppiness_pullback/state_PROD.json"
TRADES_DIR = "/home/ubuntu/013_2025_polymarket/bot_choppiness_pullback/trades_PROD"

# Trading Parameters
CHECK_INTERVAL = 1  # Check every second
CHOPPINESS_THRESHOLD = 30
PULLBACK_AMOUNT = 0.03  # Initial value, will be adapted
MAX_POSITIONS_PER_SIDE = 5  # Initial value, will be adapted
MAX_POSITIONS_PER_SIDE_HARDCAP = 3  # Hard cap
POSITION_SIZE = 5.2  # 5 shares per position
MIN_BUY_PRICE = 0.05
MAX_BUY_PRICE = 0.95

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

def calculate_choppiness_index(prices: deque, period: int = 900) -> float:
    """Calculate Choppiness Index over the period"""
    import numpy as np

    if len(prices) < period:
        return 50.0  # Return neutral value if not enough data

    prices_array = np.array(list(prices)[-period:])
    true_ranges = np.abs(np.diff(prices_array))
    sum_tr = np.sum(true_ranges)

    high = np.max(prices_array)
    low = np.min(prices_array)
    high_low_range = high - low

    if high_low_range == 0:
        return 100.0  # Completely flat = maximum choppiness

    if sum_tr == 0:
        return 100.0

    choppiness = 100 * np.log10(sum_tr / high_low_range) / np.log10(period)
    choppiness = max(0.0, min(100.0, choppiness))

    return choppiness

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
        self.last_position_time = 0
        self.last_save_time = time.time()
        self.last_asset_reload = time.time()
        self.last_position_check = time.time()

        # Load state
        self.load_state()

        logger.info("="*80)
        logger.info("🤖 BOT CHOPPINESS PULLBACK - PRODUCTION")
        logger.info("="*80)
        logger.info(f"Position Size: {POSITION_SIZE} shares")
        logger.info(f"Max Positions: {self.current_max_positions} (adaptive)")
        logger.info(f"Pullback: {self.current_pullback:.3f} (adaptive)")
        logger.info(f"Min Choppiness: {CHOPPINESS_THRESHOLD}")
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
        """Verify positions match wallet"""
        if not self.current_put_id or not self.current_call_id:
            return

        put_balance = self.check_token_balance(self.current_put_id)
        call_balance = self.check_token_balance(self.current_call_id)

        # Sync call positions
        expected_call = sum(1 for p in self.positions if p.side == 'CALL')
        if call_balance >= 0.5 and expected_call == 0:
            logger.warning(f"⚠️  Wallet has CALL ({call_balance:.2f}) but tracking shows 0")

        # Sync put positions
        expected_put = sum(1 for p in self.positions if p.side == 'PUT')
        if put_balance >= 0.5 and expected_put == 0:
            logger.warning(f"⚠️  Wallet has PUT ({put_balance:.2f}) but tracking shows 0")

        self.last_position_check = time.time()

    def get_market_data(self) -> Optional[Dict]:
        """Get current market data from JSON files"""
        try:
            call_data = read_json(CALL_FILE)
            put_data = read_json(PUT_FILE)
            btc_price = get_btc_price()

            if not call_data or not put_data or not btc_price:
                return None

            # Extract bid/ask prices
            call_bid = call_data.get('best_bid', {}).get('price')
            call_ask = call_data.get('best_ask', {}).get('price')
            put_bid = put_data.get('best_bid', {}).get('price')
            put_ask = put_data.get('best_ask', {}).get('price')

            # Update last valid prices
            if call_bid: self.last_call_bid = call_bid
            if call_ask: self.last_call_ask = call_ask
            if put_bid: self.last_put_bid = put_bid
            if put_ask: self.last_put_ask = put_ask

            # Calculate choppiness
            self.price_history.append(btc_price)
            choppiness = calculate_choppiness_index(self.price_history, period=min(900, len(self.price_history)))

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

            # Create position tracking
            position = Position(
                side=token_type,
                entry_price=limit_price,
                size=POSITION_SIZE,
                token_id=token_id,
                choppiness=choppiness,
                time_to_expiry_seconds=seconds_remaining,
                strike_price=self.strike_price if self.strike_price else 0.0
            )

            self.positions.append(position)

            if token_type == 'CALL':
                self.call_positions += 1
                self.max_call_ask = 0.0  # Reset after entry
            else:
                self.put_positions += 1
                self.max_put_ask = 0.0  # Reset after entry

            self.last_position_time = time.time()

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

        # Check if we're at 00, 15, 30, or 45 minutes (within first 3 seconds)
        is_period_start = (current_minute % 15 == 0) and (current_second <= 3)

        if is_period_start and len(self.positions) > 0:
            logger.info(f"\n{'='*80}")
            logger.info(f"⚠️  PERIOD END DETECTED - Processing settlements")
            logger.info(f"{'='*80}")

            # Step 1: Cancel all orders
            logger.info(f"🔄 Step 1: Canceling all open orders...")
            try:
                cancelled_count = self.trader.cancel_all_orders()
                logger.info(f"✅ Cancelled {cancelled_count} orders")
            except Exception as e:
                logger.error(f"❌ Error cancelling orders: {e}")

            # Step 2: Settle open positions
            logger.info(f"🔄 Step 2: Settling {len(self.positions)} positions...")

            # Calculate final PNL based on strike price outcome
            if self.strike_price:
                final_btc = get_btc_price()
                if final_btc:
                    for position in self.positions:
                        # Determine outcome: 1.0 if win, 0.0 if loss
                        if position.side == 'CALL':
                            position.final_value = 1.0 if final_btc > self.strike_price else 0.0
                        else:  # PUT
                            position.final_value = 1.0 if final_btc < self.strike_price else 0.0

                        position.pnl = (position.final_value - position.entry_price) * position.size

                        # Save to today's trades
                        trade = position.to_dict()
                        trade['final_btc'] = final_btc
                        trade['strike_price'] = self.strike_price
                        trade['outcome'] = 'WIN' if position.final_value > 0 else 'LOSS'
                        self.today_trades.append(trade)

            # Log period results
            total_pnl = sum(p.pnl for p in self.positions)
            logger.info(f"\n📊 Period Results:")
            logger.info(f"   Positions: {len(self.positions)}")
            logger.info(f"   Total PNL: ${total_pnl:+.2f}")
            logger.info(f"   Strike: ${self.strike_price:.2f}" if self.strike_price else "   Strike: Unknown")

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

            # Save trades and state
            self.save_trades()
            self.save_state()

            # Wait 10 seconds before reloading assets
            logger.info(f"\n⏳ Waiting 10 seconds before asset reload...")
            time.sleep(10)

            # Step 3: Reload asset IDs for new period
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
        status += f"Time: {seconds_remaining}s"

        print(status, end='', flush=True)

    def run(self):
        """Main trading loop"""
        logger.info("🚀 Starting Choppiness Pullback Bot (PRODUCTION)...")
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

                # Check period end FIRST (at 00, 15, 30, 45 minutes)
                self.check_period_end()

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

                # Reload asset IDs every 60s
                if time.time() - self.last_asset_reload >= 60:
                    self.reload_asset_ids()

                # Verify position every 60s
                if time.time() - self.last_position_check >= 60:
                    self.verify_position_from_wallet()

                # Refresh USDC balance every 10s
                self.refresh_usdc_balance()

                # Skip trading in buffer zones
                if in_buffer_zone:
                    time.sleep(CHECK_INTERVAL)
                    continue

                # Only trade if choppiness > threshold
                choppiness = market_data['choppiness']
                if choppiness <= CHOPPINESS_THRESHOLD:
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
                                logger.info(f"🔵 CALL position opened")

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
                                logger.info(f"🔴 PUT position opened")

                time.sleep(CHECK_INTERVAL)

            except KeyboardInterrupt:
                print("\n\n🛑 Bot stopped by user")
                logger.info("💾 Saving final state before shutdown...")
                self.save_state()
                self.save_trades()
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
    credentials = load_credentials_from_env('/home/ubuntu/.env')

    # Start bot
    bot = ChoppinessPullbackBot(credentials)
    bot.run()
