#!/usr/bin/env python3
"""
Bot016_DryRun - Paper Trading Mode
Simulates all trading without actual execution
Records trades and PNL to daily JSON files
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
TRADES_DIR = "/home/ubuntu/013_2025_polymarket/bot016_react/dryrun_trades"

# Trading parameters
SENS_MULTIPLIER = 1.2
ACTION_THRESHOLD = 0.022
MIN_SECONDS_BETWEEN_POSITIONS = 2
BUFFER_SECONDS = 20  # No trading in last 20s of period
POSITION_SIZE = 6  # 6 shares
STOP_LOSS = 0.06  # Stop if price moves 0.06 against us
TAKE_PROFIT = 0.10  # Take profit if price moves 0.10 in our favor
MAX_SPREAD = 0.03  # Suspend trading if spread > 0.03
MIN_BUY_PRICE = 0.10  # Never buy below this price

# DRY RUN MODE - No actual trading
DRY_RUN = True


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


class Bot016DryRun:
    """Paper trading bot - simulates all operations without actual execution"""

    def __init__(self):
        # Position tracking - support multiple positions up to 10 tokens
        self.positions: List[Position] = []  # List of open positions
        self.max_tokens = 10  # Maximum 10 tokens total
        self.last_position_close_time = 0

        # Asset IDs (simulated)
        self.current_put_id: Optional[str] = "PUT_SIMULATED"
        self.current_call_id: Optional[str] = "CALL_SIMULATED"

        # Paper trading balance
        self.paper_balance = 1000.0  # Start with $1000

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
        logger.info("🤖 BOT016_DRYRUN - PAPER TRADING MODE")
        logger.info("="*80)
        logger.info("⚠️  DRY RUN: No actual trades will be executed")
        logger.info(f"💰 Paper Balance: ${self.paper_balance:.2f}")
        logger.info(f"Position Size: {POSITION_SIZE} shares")
        logger.info(f"Stop Loss: -${STOP_LOSS:.2f} | Take Profit: +${TAKE_PROFIT:.2f}")
        logger.info(f"Max Spread: ${MAX_SPREAD:.2f} (suspend trading if exceeded)")
        logger.info(f"Sens Multiplier: {SENS_MULTIPLIER}x")
        logger.info(f"Action Threshold: {ACTION_THRESHOLD}")
        logger.info(f"Buffer Zone: {BUFFER_SECONDS}s")
        if self.sensitivity_data:
            logger.info(f"✅ Sensitivity data loaded: {len(self.sensitivity_data.get('bins', {}))} bins")
        logger.info("="*80)

    def get_today_filename(self) -> Path:
        """Get filename for today's trades"""
        today = date.today().strftime('%Y%m%d')
        return self.trades_dir / f"dryrun_trades_{today}.json"

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
        """Get paper trading balance"""
        return self.paper_balance

    def refresh_usdc_balance(self):
        """Refresh cached USDC balance if needed"""
        if time.time() - self.last_usdc_check >= self.usdc_check_interval:
            self.cached_usdc_balance = self.paper_balance
            self.last_usdc_check = time.time()
            logger.debug(f"💰 Paper balance: ${self.cached_usdc_balance:.2f}")

    def check_token_balance(self, token_id: str) -> float:
        """Check simulated token balance"""
        # Return quantity from matching positions
        matching = [p for p in self.positions if p.token_id == token_id]
        return sum(p.quantity for p in matching)

    def reload_asset_ids(self):
        """Reload PUT and CALL asset IDs from data files (simulated)"""
        put_data = read_json(PUT_FILE)
        call_data = read_json(CALL_FILE)

        if put_data and call_data:
            new_put_id = put_data.get('asset_id', 'PUT_SIMULATED')
            new_call_id = call_data.get('asset_id', 'CALL_SIMULATED')

            self.current_put_id = new_put_id
            self.current_call_id = new_call_id
            self.last_asset_reload = time.time()

    def verify_position_from_wallet(self):
        """Verify positions (simulated - no wallet sync needed in dry-run)"""
        # In dry-run mode, positions are already accurate
        self.last_position_check = time.time()

    def execute_buy(self, token_type: str, token_id: str, ask_price: float,
                    btc_price: float, bin_key: str, edge: float, reason: str) -> bool:
        """SIMULATED buy order - paper trading only"""
        try:
            logger.info(f"\n{'='*70}")
            logger.info(f"📄 PAPER TRADE - BUY ORDER")
            logger.info(f"{'='*70}")
            logger.info(f"📊 Token: {token_type}")
            logger.info(f"📦 Size: {POSITION_SIZE} shares")
            logger.info(f"💰 Ask Price: ${ask_price:.4f}")
            logger.info(f"📝 Reason: {reason}")

            required = ask_price * POSITION_SIZE
            logger.info(f"💵 Cost: ${required:.2f}")
            logger.info(f"💰 Paper Balance: ${self.paper_balance:.2f}")

            if self.paper_balance < required:
                logger.error(f"❌ INSUFFICIENT PAPER BALANCE")
                return False

            if self.paper_balance < required:
                logger.error(f"❌ INSUFFICIENT PAPER BALANCE")
                return False

            # SIMULATED ORDER - instant execution
            logger.info(f"\n✅ PAPER TRADE SUCCESS: Position opened")

            # Deduct from paper balance
            self.paper_balance -= required

            new_position = Position(
                token_type=token_type,
                token_id=token_id,
                entry_price=ask_price,
                entry_time=time.time(),
                quantity=POSITION_SIZE,
                entry_btc_price=btc_price,
                entry_bin=bin_key,
                edge=edge
            )

            self.positions.append(new_position)
            total_tokens = self.get_total_tokens()

            logger.info(f"📊 Position: {POSITION_SIZE} @ ${ask_price:.4f}")
            logger.info(f"💰 Paper Balance: ${self.paper_balance:.2f}")
            logger.info(f"📦 Total positions: {len(self.positions)} | Total tokens: {total_tokens:.2f}/{self.max_tokens}")
            return True

        except Exception as e:
            logger.error(f"❌ Error executing buy: {e}")
            import traceback
            traceback.print_exc()
            return False


    def execute_sell(self, token_type: str, token_id: str, bid_price: float, btc_price: float,
                     bin_key: str, reason: str, quantity: Optional[float] = None) -> bool:
        """SIMULATED sell order - paper trading only"""
        try:
            # Find matching position(s) for this token type
            matching_positions = [p for p in self.positions if p.token_type == token_type and p.token_id == token_id]

            if not matching_positions:
                logger.warning(f"⚠️  No {token_type} positions found to sell")
                return False

            # Determine sell quantity
            if quantity is None:
                sell_quantity = POSITION_SIZE
            else:
                sell_quantity = quantity

            logger.info(f"\n{'='*60}")
            logger.info(f"📄 PAPER TRADE - SELL ORDER - {reason}")
            logger.info(f"{'='*60}")
            logger.info(f"📦 Size: {sell_quantity:.2f} shares")
            logger.info(f"💰 Bid: ${bid_price:.4f}")

            # Calculate P&L from oldest position (FIFO)
            oldest_position = matching_positions[0]
            pnl = (bid_price - oldest_position.entry_price) * sell_quantity
            pnl_pct = ((bid_price / oldest_position.entry_price) - 1) * 100
            logger.info(f"📈 P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")

            # SIMULATED - instant execution
            logger.info(f"✅ PAPER TRADE: Position CLOSED")

            # Add proceeds to paper balance
            self.paper_balance += bid_price * sell_quantity

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
            logger.info(f"💰 Paper Balance: ${self.paper_balance:.2f}")
            logger.info(f"📦 Remaining positions: {len(self.positions)} | Total tokens: {total_tokens:.2f}/{self.max_tokens}")

            return True

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

                    # PERIOD END SUMMARY - Calculate and save
                    if self.current_period_start is not None:
                        logger.info(f"\n{'='*80}")
                        logger.info(f"📊 PERIOD END SUMMARY")
                        logger.info(f"{'='*80}")

                        # Get period trades (trades closed in last 15 minutes)
                        period_end_time = time.time()
                        period_start_time = period_end_time - 900  # 15 minutes ago

                        period_trades = [
                            t for t in self.today_trades[-20:]  # Check last 20 trades
                            if datetime.fromisoformat(t['close_time']).timestamp() >= period_start_time
                        ]

                        if period_trades:
                            period_pnl = sum(t['pnl'] for t in period_trades)
                            wins = sum(1 for t in period_trades if t['pnl'] > 0)
                            losses = sum(1 for t in period_trades if t['pnl'] <= 0)

                            logger.info(f"Period Strike: ${self.strike_price:.2f}")
                            logger.info(f"Trades: {len(period_trades)} ({wins}W/{losses}L)")
                            logger.info(f"Period P&L: ${period_pnl:+.2f}")
                            logger.info(f"Daily Total: {len(self.today_trades)} trades | ${self.get_daily_pnl():+.2f}")
                            logger.info(f"Paper Balance: ${self.paper_balance:.2f}")

                        # Save trades
                        self.save_trades()
                        logger.info(f"{'='*80}\n")

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

                # 30-SECOND MAINTENANCE CYCLE (simulated)
                if time.time() - self.last_maintenance_cycle >= 30:
                    logger.info("🔧 MAINTENANCE: Verifying positions, reloading tokens...")
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

                    #if btc_call_diff > 1 or btc_call_diff < -1 :
                        #print(f"⏱️  Timestamp deltas: BTC-CALL={btc_call_diff:.2f}s, BTC-PUT={btc_put_diff:.2f}s")

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
                        call_spread_direction = "DOWNWARD"
                    elif ask_move > bid_move:
                        call_spread_direction = "UPWARD"

                if self.prev_put_bid is not None and self.prev_put_ask is not None:
                    bid_move = self.prev_put_bid - put_bid_price
                    ask_move = self.prev_put_ask - put_ask_price
                    if bid_move > ask_move:
                        put_spread_direction = "DOWNWARD"
                    elif ask_move > bid_move:
                        put_spread_direction = "UPWARD"

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

                # CHECK STOP LOSS AND TAKE PROFIT (based on price movement)
                if not in_buffer_zone and len(self.positions) > 0:
                    for position in self.positions[:]:  # Copy list to allow modification
                        current_price = call_bid_price if position.token_type == 'CALL' else put_bid_price

                        if current_price > 0:
                            # Calculate price movement from entry
                            price_movement = current_price - position.entry_price

                            # Check stop loss - AGGRESSIVE with retry
                            if price_movement <= -STOP_LOSS:
                                logger.info(f"🛑 STOP LOSS TRIGGERED: {position.token_type} price moved ${price_movement:.3f} (entry ${position.entry_price:.3f} → ${current_price:.3f})")
                                distance = abs(btc_price - self.strike_price) if self.strike_price else 0
                                bin_key = get_bin_key(distance, seconds_remaining, volatility)

                                # Aggressive stop loss - try to sell immediately
                                success = self.execute_sell(position.token_type, position.token_id, current_price,
                                                btc_price, bin_key, "STOP_LOSS", quantity=position.quantity)

                                # Retry after 5 seconds if unsuccessful
                                if not success:
                                    logger.warning(f"⚠️  STOP LOSS failed, retrying in 5s...")
                                    time.sleep(5)

                                    # Refresh price and retry
                                    retry_data = read_json(CALL_FILE if position.token_type == 'CALL' else PUT_FILE)
                                    if retry_data and retry_data.get('best_bid'):
                                        retry_price = retry_data['best_bid'].get('price', 0)
                                        logger.info(f"🔄 STOP LOSS RETRY at ${retry_price:.4f}")
                                        self.execute_sell(position.token_type, position.token_id, retry_price,
                                                        btc_price, bin_key, "STOP_LOSS_RETRY", quantity=position.quantity)

                            # Check take profit - price moved in our favor by TAKE_PROFIT
                            elif price_movement >= TAKE_PROFIT:
                                logger.info(f"🎯 TAKE PROFIT: {position.token_type} price moved ${price_movement:.3f} (entry ${position.entry_price:.3f} → ${current_price:.3f})")
                                distance = abs(btc_price - self.strike_price) if self.strike_price else 0
                                bin_key = get_bin_key(distance, seconds_remaining, volatility)
                                self.execute_sell(position.token_type, position.token_id, current_price,
                                                btc_price, bin_key, "TAKE_PROFIT", quantity=position.quantity)

                # TRADING LOGIC - Only outside buffer zone and when spread is acceptable
                if not in_buffer_zone and not trading_suspended and self.strike_price and prev_btc_price is not None:
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
                                # Price constraint: do not buy if price > 0.95 or < MIN_BUY_PRICE
                                # BUY signal
                                if call_ask_price > 0.95:
                                    logger.debug(f"❌ CALL BUY rejected: price {call_ask_price:.2f} too high (>0.95)")
                                elif call_ask_price < MIN_BUY_PRICE:
                                    logger.warning(f"❌ CALL BUY BLOCKED: price {call_ask_price:.3f} < MIN_BUY_PRICE {MIN_BUY_PRICE:.2f} (expired period protection)")
                                elif self.can_open_position():
                                    edge = ideal_call_movement
                                    self.execute_buy('CALL', self.current_call_id, call_ask_price,
                                                   btc_price, bin_key, edge, f"Edge: {edge:.3f}")
                                    print("BUY CALL @",call_ask_price)

                            if ideal_call_movement < ACTION_THRESHOLD*-1:
                                # SELL signal - check if we have CALL positions with enough tokens (cached)
                                call_positions = [p for p in self.positions if p.token_type == 'CALL']
                                total_call_tokens = sum(p.quantity for p in call_positions)

                                if total_call_tokens >= POSITION_SIZE:
                                    self.execute_sell('CALL', self.current_call_id, call_bid_price,
                                                    btc_price, bin_key, "SELL_SIGNAL")
                                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - SELL CALL @ {call_bid_price}")
                                elif call_positions and self.can_open_position():
                                    # Have CALL but not enough to sell - buy PUT instead (hedge)
                                    self.execute_buy('PUT', self.current_put_id, put_ask_price,
                                                   btc_price, bin_key, abs(ideal_call_movement), f"HEDGE_PUT")
                                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - HEDGE BUY PUT @ {put_ask_price}")



                        # PUT signals
                        if abs(put_sens) > 0.000001 and put_ask_price > 0 and put_bid_price > 0:
                            ideal_put_movement = btc_delta * put_sens * SENS_MULTIPLIER
                            actual_put_ask_movement = put_ask_price - prev_put_ask

                            if ideal_put_movement > 0.01 :
                                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - PUT move: {ideal_put_movement}")
                                #print("PUT move:", ideal_call_movement)

                            if ideal_put_movement > ACTION_THRESHOLD:
                                # Price constraint: do not buy if price > 0.95 or < MIN_BUY_PRICE
                                # BUY signal
                                if put_ask_price > 0.95:
                                    logger.debug(f"❌ PUT BUY rejected: price {put_ask_price:.2f} too high (>0.95)")
                                elif put_ask_price < MIN_BUY_PRICE:
                                    logger.warning(f"❌ PUT BUY BLOCKED: price {put_ask_price:.3f} < MIN_BUY_PRICE {MIN_BUY_PRICE:.2f} (expired period protection)")
                                elif self.can_open_position():
                                    edge = ideal_put_movement
                                    self.execute_buy('PUT', self.current_put_id, put_ask_price,
                                                   btc_price, bin_key, edge, f"Edge: {edge:.3f}")
                                    print("BUY PUT @",put_ask_price)

                            if ideal_put_movement < ACTION_THRESHOLD*-1:
                                # SELL signal - check if we have PUT positions with enough tokens (cached)
                                put_positions = [p for p in self.positions if p.token_type == 'PUT']
                                total_put_tokens = sum(p.quantity for p in put_positions)

                                if total_put_tokens >= POSITION_SIZE:
                                    self.execute_sell('PUT', self.current_put_id, put_bid_price,
                                                    btc_price, bin_key, "SELL_SIGNAL")
                                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - SELL PUT @ {put_bid_price}")
                                elif put_positions and self.can_open_position():
                                    # Have PUT but not enough to sell - buy CALL instead (hedge)
                                    self.execute_buy('CALL', self.current_call_id, call_ask_price,
                                                   btc_price, bin_key, abs(ideal_put_movement), f"HEDGE_CALL")
                                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} - HEDGE BUY CALL @ {call_ask_price}")

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
        logger.info("🚀 Starting Bot016 DRY RUN - Paper Trading Mode")
        logger.info("⚠️  NO REAL TRADES WILL BE EXECUTED")

        # Create and run bot (no credentials needed for dry-run)
        bot = Bot016DryRun()
        bot.run()

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
