#!/usr/bin/env python3
"""
Bot016_Third - Sensitivity-Based Binary Options Trading Bot
Production-ready Polymarket trading with 1-position limit and period management
pm2 start bot016_third.py --cron-restart="00 * * * *" --interpreter python3
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

# Import Polymarket trading core
sys.path.insert(0, '/home/ubuntu')
from polymarket_trading_core_debug import PolymarketTrader, load_credentials_from_env

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
BTC_FILE = "/home/ubuntu/013_2025_polymarket/bybit_btc_price.json"
SENSITIVITY_FILE = "/home/ubuntu/013_2025_polymarket/bot016_react/sensitivity_data/sensitivity_transformed.json"
TRADES_DIR = "/home/ubuntu/013_2025_polymarket/bot016_react/sensitivity_trades"

# Trading parameters
SENS_MULTIPLIER = 1.2
ACTION_THRESHOLD = 0.032
MIN_SECONDS_BETWEEN_POSITIONS = 2
BUFFER_SECONDS = 20  # No trading in last 20s of period
POSITION_SIZE = 5  # 5 shares


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
    """Get strike price from Bybit API"""
    try:
        import requests
        from datetime import timezone

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

        # Position tracking - support multiple positions up to 10 tokens
        self.positions: List[Position] = []  # List of open positions
        self.max_tokens = 10  # Maximum 10 tokens total
        self.last_position_close_time = 0

        # Asset IDs
        self.current_put_id: Optional[str] = None
        self.current_call_id: Optional[str] = None

        # Trades
        self.trades_dir = Path(TRADES_DIR)
        self.trades_dir.mkdir(exist_ok=True, parents=True)
        self.today_trades = []
        self.load_today_trades()

        # Load sensitivity data
        self.sensitivity_data = read_json(SENSITIVITY_FILE)

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

        # Cached USDC balance (refresh every 10s)
        self.cached_usdc_balance = 0.0
        self.last_usdc_check = 0.0
        self.usdc_check_interval = 10.0  # 10 seconds

        logger.info("="*80)
        logger.info("🤖 BOT016_FOURTH - DEV MODE")
        logger.info("="*80)
        logger.info(f"Position Size: {POSITION_SIZE} shares")
        logger.info(f"Sens Multiplier: {SENS_MULTIPLIER}x")
        logger.info(f"Action Threshold: {ACTION_THRESHOLD}")
        logger.info(f"Buffer Zone: {BUFFER_SECONDS}s")
        if self.sensitivity_data:
            logger.info(f"✅ Sensitivity data loaded: {len(self.sensitivity_data.get('bins', {}))} bins")
        logger.info("="*80)

    def get_today_filename(self) -> Path:
        """Get filename for today's trades"""
        today = date.today().strftime('%Y%m%d')
        return self.trades_dir / f"prod_trades_{today}.json"

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
        """Verify positions match wallet (supports multiple positions)"""
        if not self.current_put_id or not self.current_call_id:
            return

        put_balance = self.check_token_balance(self.current_put_id)
        call_balance = self.check_token_balance(self.current_call_id)

        # Count existing positions
        put_positions = [p for p in self.positions if p.token_type == 'PUT']
        call_positions = [p for p in self.positions if p.token_type == 'CALL']

        tracked_put_tokens = sum(p.quantity for p in put_positions)
        tracked_call_tokens = sum(p.quantity for p in call_positions)

        # Sync PUT positions if mismatch
        if abs(put_balance - tracked_put_tokens) > 0.5:
            logger.warning(f"⚠️  PUT mismatch: Wallet={put_balance:.2f}, Tracked={tracked_put_tokens:.2f}")

            if put_balance >= 0.5 and tracked_put_tokens < 0.5:
                # Have tokens but not tracking - sync
                put_data = read_json(PUT_FILE)
                if not put_data or not put_data.get('best_bid') or not put_data.get('best_ask'):
                    logger.warning("DEEP ITM/OTM PUT - cannot sync")
                    return

                price = (put_data['best_bid'].get('price', 0.50) + put_data['best_ask'].get('price', 0.50)) / 2

                synced_position = Position(
                    token_type='PUT',
                    token_id=self.current_put_id,
                    entry_price=price,
                    entry_time=time.time(),
                    quantity=put_balance,
                    entry_btc_price=0,
                    entry_bin="SYNCED",
                    edge=0
                )
                self.positions.append(synced_position)
                logger.info(f"✅ Synced PUT position: {put_balance:.2f} tokens")

            elif put_balance < 0.5 and tracked_put_tokens >= 0.5:
                # Tracking tokens but wallet empty - clear
                self.positions = [p for p in self.positions if p.token_type != 'PUT']
                logger.info("✅ Cleared PUT positions from tracking")

        # Sync CALL positions if mismatch
        if abs(call_balance - tracked_call_tokens) > 0.5:
            logger.warning(f"⚠️  CALL mismatch: Wallet={call_balance:.2f}, Tracked={tracked_call_tokens:.2f}")

            if call_balance >= 0.5 and tracked_call_tokens < 0.5:
                # Have tokens but not tracking - sync
                call_data = read_json(CALL_FILE)
                if not call_data or not call_data.get('best_bid') or not call_data.get('best_ask'):
                    logger.warning("DEEP ITM/OTM CALL - cannot sync")
                    return

                price = (call_data['best_bid'].get('price', 0.50) + call_data['best_ask'].get('price', 0.50)) / 2

                synced_position = Position(
                    token_type='CALL',
                    token_id=self.current_call_id,
                    entry_price=price,
                    entry_time=time.time(),
                    quantity=call_balance,
                    entry_btc_price=0,
                    entry_bin="SYNCED",
                    edge=0
                )
                self.positions.append(synced_position)
                logger.info(f"✅ Synced CALL position: {call_balance:.2f} tokens")

            elif call_balance < 0.5 and tracked_call_tokens >= 0.5:
                # Tracking tokens but wallet empty - clear
                self.positions = [p for p in self.positions if p.token_type != 'CALL']
                logger.info("✅ Cleared CALL positions from tracking")

        self.last_position_check = time.time()

    def execute_buy(self, token_type: str, token_id: str, ask_price: float,
                    btc_price: float, bin_key: str, edge: float, reason: str) -> bool:
        """Execute buy order with Fill-or-Kill (battle-tested with optimizations)"""
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
            final_price = ask_price +0.01
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
                logger.info(f"\n✅ SUCCESS: Position opened")

                new_position = Position(
                    token_type=token_type,
                    token_id=token_id,
                    entry_price=ask_price,
                    entry_time=time.time(),
                    quantity=final_balance,
                    entry_btc_price=btc_price,
                    entry_bin=bin_key,
                    edge=edge
                )

                self.positions.append(new_position)
                total_tokens = self.get_total_tokens()

                logger.info(f"📊 Position: {final_balance:.2f} @ ${ask_price:.4f}")
                logger.info(f"📦 Total positions: {len(self.positions)} | Total tokens: {total_tokens:.2f}/{self.max_tokens}")
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


    def execute_sell(self, token_type: str, token_id: str, bid_price: float, btc_price: float,
                     bin_key: str, reason: str) -> bool:
        """Execute sell order - always sells POSITION_SIZE tokens"""
        try:
            # Find matching position(s) for this token type
            matching_positions = [p for p in self.positions if p.token_type == token_type and p.token_id == token_id]

            if not matching_positions:
                logger.warning(f"⚠️  No {token_type} positions found to sell")
                return False

            # Set sell price
            sell_price = bid_price - 0.01  # aggressive
            sell_price = max(0.01, bid_price - 0.001) if bid_price > 0 else 0.01
            logger.info(f"💰 Sell Price: ${sell_price:.4f}")

            # Place order for POSITION_SIZE
            order_id = self.trader.place_sell_order_FAK(
                token_id=token_id,
                price=sell_price,
                quantity=POSITION_SIZE
            )

            if not order_id:
                logger.error(f"❌ Failed to place sell order")
                return False

            logger.info(f"✅ Sell order placed: {order_id[:16]}...")

            # Verify we have tokens
            actual_balance = self.check_token_balance(token_id)

            if actual_balance < 0.5:
                logger.warning(f"\n⚠️  SELL ABORTED: No tokens in wallet")
                # Remove all matching positions
                self.positions = [p for p in self.positions if not (p.token_type == token_type and p.token_id == token_id)]
                return False

            logger.info(f"\n{'='*60}")
            logger.info(f"💰 EXECUTING SELL ORDER - {reason}")
            logger.info(f"{'='*60}")
            logger.info(f"📦 Size: {POSITION_SIZE} shares")
            logger.info(f"💰 Bid: ${bid_price:.4f}")

            # Calculate P&L from oldest position (FIFO)
            oldest_position = matching_positions[0]
            pnl = (bid_price - oldest_position.entry_price) * POSITION_SIZE
            pnl_pct = ((bid_price / oldest_position.entry_price) - 1) * 100
            logger.info(f"📈 P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")

            # Wait and verify
            time.sleep(3)
            balance_after = self.check_token_balance(token_id)

            if balance_after < actual_balance - (POSITION_SIZE * 0.5):  # Sold at least half of expected
                logger.info(f"✅ Sell FILLED - Position CLOSED")

                # Record trade
                trade = {
                    'type': oldest_position.token_type,
                    'open_time': datetime.fromtimestamp(oldest_position.entry_time).isoformat(),
                    'close_time': datetime.now().isoformat(),
                    'open_btc': oldest_position.entry_btc_price,
                    'close_btc': btc_price,
                    'open_price': oldest_position.entry_price,
                    'close_price': bid_price,
                    'open_bin': oldest_position.entry_bin,
                    'close_bin': bin_key,
                    'edge': oldest_position.edge,
                    'pnl': pnl,
                    'close_reason': reason
                }

                self.today_trades.append(trade)
                self.save_trades()

                # Remove the oldest position (FIFO)
                self.positions.remove(oldest_position)
                self.last_position_close_time = time.time()

                total_tokens = self.get_total_tokens()
                logger.info(f"📦 Remaining positions: {len(self.positions)} | Total tokens: {total_tokens:.2f}/{self.max_tokens}")

                return True
            else:
                logger.warning(f"⚠️  Sell not filled, {balance_after:.2f} tokens remain")
                return False

        except Exception as e:
            logger.error(f"❌ Error executing sell: {e}")
            import traceback
            traceback.print_exc()
            return False

    def can_open_position(self) -> bool:
        """Check if we can open a new position"""
        # Calculate total tokens held
        total_tokens = sum(pos.quantity for pos in self.positions)

        # Can't open if we're at max capacity
        if total_tokens >= self.max_tokens:
            return False

        if time.time() - self.last_position_close_time < MIN_SECONDS_BETWEEN_POSITIONS:
            return False

        return True

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
                    # Close ALL positions from previous period
                    if len(self.positions) > 0 and self.current_period_start is not None:
                        logger.info(f"\n⚠️  PERIOD END - Closing all positions ({len(self.positions)} positions)")

                        # Close each position
                        positions_to_close = self.positions.copy()
                        for pos in positions_to_close:
                            exit_price = self.last_call_bid if pos.token_type == 'CALL' else self.last_put_bid
                            exit_btc = self.last_btc_price if self.last_btc_price else 0

                            if exit_price and exit_price > 0 and exit_btc > 0:
                                distance = abs(exit_btc - self.strike_price) if self.strike_price else 0
                                volatility = calculate_btc_volatility(self.btc_price_history)
                                bin_key = get_bin_key(distance, BUFFER_SECONDS - 1, volatility)

                                self.execute_sell(pos.token_type, pos.token_id, exit_price, exit_btc, bin_key, "PERIOD_END")
                            else:
                                logger.error(f"❌ Cannot close {pos.token_type} - no valid exit price")

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

                # Reload asset IDs at END of start buffer (after 20 seconds)
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
                        total_tokens = self.get_total_tokens()
                        logger.info(f"📦 Positions: {len(self.positions)} | Tokens: {total_tokens:.0f}/{self.max_tokens}")
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

                # Verify position every 60s
                if time.time() - self.last_position_check >= 10:
                    self.verify_position_from_wallet()

                # Refresh cached USDC balance every 10s
                self.refresh_usdc_balance()

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

                    if btc_call_diff > 1 or btc_call_diff < -1 :
                        print(f"⏱️  Timestamp deltas: BTC-CALL={btc_call_diff:.2f}s, BTC-PUT={btc_put_diff:.2f}s")

                    if btc_call_diff < -1 :
                        print ("LAG BTC websocket ................  WAIT")
                        time.sleep(2)
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

                # TRADING LOGIC - Only outside buffer zone
                if not in_buffer_zone and self.strike_price and prev_btc_price is not None:
                    distance = abs(btc_price - self.strike_price)
                    bin_key = get_bin_key(distance, seconds_remaining, volatility)

                    # Check for signals
                    if self.sensitivity_data and bin_key in self.sensitivity_data.get('bins', {}):
                        bin_data = self.sensitivity_data['bins'][bin_key]
                        call_sens = bin_data.get('call_sensitivity', {}).get('avg', 0)
                        put_sens = bin_data.get('put_sensitivity', {}).get('avg', 0)

                        btc_delta = btc_price - prev_btc_price

                        # CALL signals
                        if abs(call_sens) > 0.000001 and call_ask_price > 0 and call_bid_price > 0:
                            ideal_call_movement = btc_delta * call_sens * SENS_MULTIPLIER
                            actual_call_ask_movement = call_ask_price - prev_call_ask

                            if ideal_call_movement > 0.01 :
                                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - CALL move: {ideal_call_movement}")
                                #print("CALL move:", ideal_call_movement)

                            if ideal_call_movement > ACTION_THRESHOLD:
                                # Price constraint: do not buy if price > 0.97 or < 0.03
                                # BUY signal
                                if call_ask_price > 0.95 or call_ask_price < 0.10:
                                    logger.debug(f"❌ CALL BUY rejected: price {call_ask_price:.2f} outside range [0.03, 0.97]")
                                elif self.can_open_position():
                                    edge = ideal_call_movement
                                    self.execute_buy('CALL', self.current_call_id, call_ask_price,
                                                   btc_price, bin_key, edge, f"Edge: {edge:.3f}")
                                    print("BUY CALL @",call_ask_price)

                            if ideal_call_movement < ACTION_THRESHOLD*-1:
                                # SELL signal - check if we have CALL positions
                                call_positions = [p for p in self.positions if p.token_type == 'CALL']
                                if call_positions:
                                    self.execute_sell('CALL', self.current_call_id, call_bid_price,
                                                    btc_price, bin_key, "SELL_SIGNAL")
                                    print("SELL CALL @",call_bid_price)



                        # PUT signals
                        if abs(put_sens) > 0.000001 and put_ask_price > 0 and put_bid_price > 0:
                            ideal_put_movement = btc_delta * put_sens * SENS_MULTIPLIER
                            actual_put_ask_movement = put_ask_price - prev_put_ask

                            if ideal_put_movement > 0.01 :
                                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - PUT move: {ideal_put_movement}")
                                #print("PUT move:", ideal_call_movement)

                            if ideal_put_movement > ACTION_THRESHOLD:
                                # Price constraint: do not buy if price > 0.97 or < 0.03
                                # BUY signal
                                if put_ask_price > 0.95 or put_ask_price < 0.10:
                                    logger.debug(f"❌ PUT BUY rejected: price {put_ask_price:.2f} outside range [0.03, 0.97]")
                                elif self.can_open_position():
                                    edge = ideal_put_movement
                                    self.execute_buy('PUT', self.current_put_id, put_ask_price,
                                                   btc_price, bin_key, edge, f"Edge: {edge:.3f}")
                                    print("BUY PUT @",put_ask_price)

                            if ideal_put_movement < ACTION_THRESHOLD*-1:
                                # SELL signal - check if we have PUT positions
                                put_positions = [p for p in self.positions if p.token_type == 'PUT']
                                if put_positions:
                                    self.execute_sell('PUT', self.current_put_id, put_bid_price,
                                                    btc_price, bin_key, "SELL_SIGNAL")
                                    print("SELL PUT @",put_bid_price)

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
            if len(self.positions) > 0:
                logger.info(f"\n⚠️  Closing {len(self.positions)} open positions...")

                positions_to_close = self.positions.copy()
                for pos in positions_to_close:
                    data = read_json(CALL_FILE if pos.token_type == 'CALL' else PUT_FILE)

                    if data and data.get('best_bid'):
                        exit_price = data['best_bid'].get('price', 0)
                        btc_data = read_json(BTC_FILE)
                        btc_price = btc_data.get('price', 0) if btc_data else 0

                        if exit_price > 0:
                            distance = abs(btc_price - self.strike_price) if self.strike_price else 0
                            volatility = calculate_btc_volatility(self.btc_price_history)
                            seconds_left = get_seconds_to_expiry()
                            bin_key = get_bin_key(distance, seconds_left, volatility)

                            self.execute_sell(pos.token_type, pos.token_id, exit_price, btc_price, bin_key, "MANUAL_STOP")

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
