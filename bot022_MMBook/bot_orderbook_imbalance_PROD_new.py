#!/usr/bin/env python3
"""
15-Minute Binary Options Order Book Imbalance Trading Bot - PRODUCTION
=======================================================================
Strategy: Order Book Imbalance with Period Management

Trading Rules:
- BUY when: sum(top4_bids) > sum(top4_asks) × 2 (TAKER order at best ask)
- SELL when: sum(top4_asks) > sum(top4_bids) × 2 OR price >= 0.99
- NO BUY if: price > 0.95 OR price < 0.05
- Period ends at minute 00, 15, 30, 45
- Last 10 seconds: Close all positions, cancel orders, NO new trades
- Reset everything at period end
- Track PNL per period and overall

Order Execution:
- Uses TAKER orders (market price at best_ask)
- Minimum 20 seconds between orders
- Continuous wallet verification for position counting
- 10% tolerance for partial fills

pm2 start bot_orderbook_imbalance_PROD.py --interpreter python3
"""

import sys
import time
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
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

# File paths - ADJUST FOR YOUR VPS
PUT_FILE = "/home/ubuntu/013_2025_polymarket/15M_PUT.json"
CALL_FILE = "/home/ubuntu/013_2025_polymarket/15M_CALL.json"
BTC_FILE = "/home/ubuntu/013_2025_polymarket/bybit_btc_price.json"
TRADES_LOG_DIR = "/home/ubuntu/013_2025_polymarket/bot_orderbook_imbalance/trades"
STATE_FILE = "/home/ubuntu/013_2025_polymarket/bot_orderbook_imbalance/state.json"

# Trading Parameters
CHECK_INTERVAL = 1  # Check every second
IMBALANCE_RATIO = 4.0  # Buy when bids > asks * 2
POSITION_SIZE = 5.2  # 5 shares per position
MAX_POSITIONS_PER_SIDE = 3  # Max 1 CALL and 1 PUT at a time

# Price limits
MIN_BUY_PRICE = 0.10
MAX_BUY_PRICE = 0.90

# Low balance mode
MIN_USDC_TO_TRADE = 5.2  # Minimum USDC to place orders
LOW_BALANCE_CHECK_INTERVAL = 30  # Check every 30s when balance is low

# Order management
MIN_SECONDS_BETWEEN_ORDERS = 20  # 20 seconds minimum between orders
WALLET_CHECK_INTERVAL = 5  # Check wallet every 5 seconds

# Buffer times
PERIOD_END_BUFFER_SECONDS = 10  # No trading in last 10 seconds
PERIOD_START_DELAY = 5  # Wait 5s after period start before trading

# Order book analysis
TOP_N_LEVELS = 4  # Analyze top 4 bids and asks

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def read_json_safe(filepath: str) -> Optional[dict]:
    """Read JSON file with safety for malformed data"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Handle potential trailing characters
        depth = 0
        end = 0
        for i, char in enumerate(content):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end > 0:
            return json.loads(content[:end])
        return None
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return None


def get_period_id(dt: datetime) -> str:
    """Get period identifier (e.g., '2026-01-15_10:15')"""
    minute = (dt.minute // 15) * 15
    period_start = dt.replace(minute=minute, second=0, microsecond=0)
    return period_start.strftime('%Y-%m-%d_%H:%M')


def get_period_end(dt: datetime) -> datetime:
    """Get the end time of current period"""
    minute = (dt.minute // 15) * 15
    period_start = dt.replace(minute=minute, second=0, microsecond=0)
    return period_start + timedelta(minutes=15)


def get_seconds_to_period_end(dt: datetime = None) -> float:
    """Get seconds remaining until period ends"""
    if dt is None:
        dt = datetime.now()
    period_end = get_period_end(dt)
    delta = period_end - dt
    return delta.total_seconds()


def get_seconds_into_period(dt: datetime = None) -> int:
    """Get seconds into current 15-minute period"""
    if dt is None:
        dt = datetime.now()
    minutes_into_quarter = dt.minute % 15
    return minutes_into_quarter * 60 + dt.second


def is_period_boundary(dt: datetime = None) -> bool:
    """Check if we're at a period boundary (00, 15, 30, 45 minutes)"""
    if dt is None:
        dt = datetime.now()
    return dt.minute % 15 == 0 and dt.second <= 5


# ============================================================================
# ORDER BOOK ANALYSIS
# ============================================================================

def analyze_order_book(book_data: dict, n_levels: int = TOP_N_LEVELS) -> Optional[Dict]:
    """
    Analyze order book and return metrics
    Handles None values for deep ITM/OTM options
    """
    try:
        if not book_data:
            return None

        # Get best bid/ask with None handling
        best_bid_data = book_data.get('best_bid')
        best_ask_data = book_data.get('best_ask')

        # Handle None or missing price data
        if best_bid_data is None or best_ask_data is None:
            logger.warning(f"Missing best_bid or best_ask data")
            return None

        best_bid = best_bid_data.get('price') if isinstance(best_bid_data, dict) else None
        best_ask = best_ask_data.get('price') if isinstance(best_ask_data, dict) else None

        if best_bid is None or best_ask is None:
            logger.warning(f"Price is None - option may be deep ITM/OTM")
            return None

        # Get complete book
        complete_book = book_data.get('complete_book', {})
        bids = complete_book.get('bids', [])
        asks = complete_book.get('asks', [])

        if not bids or not asks:
            return None

        # Parse and sort bids (highest first)
        parsed_bids = sorted(
            [(float(l['price']), float(l['size'])) for l in bids if l.get('price') and l.get('size')],
            key=lambda x: x[0], reverse=True
        )[:n_levels]

        # Parse and sort asks (lowest first)
        parsed_asks = sorted(
            [(float(l['price']), float(l['size'])) for l in asks if l.get('price') and l.get('size')],
            key=lambda x: x[0]
        )[:n_levels]

        if not parsed_bids or not parsed_asks:
            return None

        sum_bids = sum(size for _, size in parsed_bids)
        sum_asks = sum(size for _, size in parsed_asks)

        return {
            'best_bid': float(best_bid),
            'best_ask': float(best_ask),
            'bids': parsed_bids,
            'asks': parsed_asks,
            'sum_bids': sum_bids,
            'sum_asks': sum_asks,
            'bid_ask_ratio': sum_bids / sum_asks if sum_asks > 0 else float('inf'),
            'ask_bid_ratio': sum_asks / sum_bids if sum_bids > 0 else float('inf'),
            'spread': float(best_ask) - float(best_bid),
            'asset_id': book_data.get('asset_id'),
            'asset_name': book_data.get('asset_name')
        }
    except Exception as e:
        logger.error(f"Error analyzing order book: {e}")
        return None


# ============================================================================
# TRADE TRACKING
# ============================================================================

class Trade:
    """Trade record"""
    def __init__(self, trade_id: int, period_id: str, asset_type: str,
                 entry_price: float, entry_time: datetime, size: float):
        self.trade_id = trade_id
        self.period_id = period_id
        self.asset_type = asset_type
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.size = size
        self.exit_price: Optional[float] = None
        self.exit_time: Optional[datetime] = None
        self.exit_reason: Optional[str] = None
        self.pnl: float = 0.0
        self.pnl_percent: float = 0.0

    def close(self, exit_price: float, exit_time: datetime, reason: str):
        """Close the trade"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason
        self.pnl = (exit_price - self.entry_price) * self.size
        self.pnl_percent = ((exit_price / self.entry_price) - 1) * 100 if self.entry_price > 0 else 0

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'trade_id': self.trade_id,
            'period_id': self.period_id,
            'asset_type': self.asset_type,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'size': self.size,
            'pnl': round(self.pnl, 2),
            'pnl_percent': round(self.pnl_percent, 2),
            'exit_reason': self.exit_reason
        }


class PeriodStats:
    """Statistics for a trading period"""
    def __init__(self, period_id: str, start_time: datetime):
        self.period_id = period_id
        self.start_time = start_time
        self.end_time: Optional[datetime] = None
        self.trades: List[Trade] = []
        self.pnl: float = 0.0
        self.wins: int = 0
        self.losses: int = 0

    def add_trade(self, trade: Trade):
        """Add a completed trade"""
        self.trades.append(trade)
        self.pnl += trade.pnl
        if trade.pnl > 0:
            self.wins += 1
        else:
            self.losses += 1

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'period_id': self.period_id,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'summary': {
                'total_trades': len(self.trades),
                'wins': self.wins,
                'losses': self.losses,
                'pnl': round(self.pnl, 2),
                'win_rate': round(self.wins / len(self.trades) * 100, 1) if self.trades else 0
            },
            'trades': [t.to_dict() for t in self.trades]
        }


# ============================================================================
# MAIN BOT CLASS
# ============================================================================

class OrderBookImbalanceBot:
    """Production trading bot using order book imbalance strategy"""

    def __init__(self, credentials: dict):
        """Initialize the bot"""
        # Initialize trader
        self.trader = PolymarketTrader(
            clob_api_url=credentials['clob_api_url'],
            private_key=credentials['private_key'],
            api_key=credentials['api_key'],
            api_secret=credentials['api_secret'],
            api_passphrase=credentials['api_passphrase']
        )

        # Create directories
        os.makedirs(TRADES_LOG_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)

        # Asset IDs
        self.current_call_id: Optional[str] = None
        self.current_put_id: Optional[str] = None

        # Position tracking
        self.call_position: Optional[Trade] = None
        self.put_position: Optional[Trade] = None
        self.call_positions: int = 0  # Wallet-verified count
        self.put_positions: int = 0   # Wallet-verified count

        # Period tracking
        self.current_period_id: Optional[str] = None
        self.periods: Dict[str, PeriodStats] = {}
        self.all_trades: List[Trade] = []
        self.trade_counter = 0

        # Timing
        self.last_order_time: float = 0
        self.last_wallet_check: float = 0
        self.last_asset_id_check: float = 0
        self.asset_id_check_interval: float = 60  # Check every 60 seconds

        # Balance caching
        self.cached_usdc_balance: float = 0.0
        self.last_usdc_check: float = 0
        self.usdc_check_interval: float = 10  # 10 second cache
        self.in_low_balance_mode: bool = False
        self.last_low_balance_log: float = 0

        # Load initial asset IDs
        self.reload_asset_ids()

        # Get initial balance
        self.cached_usdc_balance = self.get_usdc_balance()

        logger.info("="*70)
        logger.info("🤖 ORDER BOOK IMBALANCE BOT - PRODUCTION")
        logger.info("="*70)
        logger.info(f"Strategy: Buy when bid_sum > ask_sum × {IMBALANCE_RATIO}")
        logger.info(f"Position Size: {POSITION_SIZE} shares")
        logger.info(f"Max Positions: {MAX_POSITIONS_PER_SIDE} per side")
        logger.info(f"Price Limits: {MIN_BUY_PRICE} - {MAX_BUY_PRICE}")
        logger.info(f"Min USDC to trade: ${MIN_USDC_TO_TRADE:.2f}")
        logger.info(f"Min time between orders: {MIN_SECONDS_BETWEEN_ORDERS}s")
        logger.info(f"Period end buffer: {PERIOD_END_BUFFER_SECONDS}s")
        logger.info(f"USDC Balance: ${self.cached_usdc_balance:.2f}")
        logger.info("="*70)

    # ========================================================================
    # BALANCE AND POSITION FUNCTIONS
    # ========================================================================

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
        Uses 10% tolerance for partial fills
        """
        if not self.current_call_id or not self.current_put_id:
            return 0, 0

        call_balance = self.check_token_balance(self.current_call_id)
        put_balance = self.check_token_balance(self.current_put_id)

        # Calculate with 10% tolerance
        tolerance = 0.9
        min_position_size = POSITION_SIZE * tolerance

        call_positions = int(call_balance / min_position_size) if min_position_size > 0 else 0
        put_positions = int(put_balance / min_position_size) if min_position_size > 0 else 0

        return call_positions, put_positions

    def verify_positions_from_wallet(self):
        """Verify and sync position counts from wallet"""
        if time.time() - self.last_wallet_check < WALLET_CHECK_INTERVAL:
            return

        wallet_call, wallet_put = self.count_wallet_positions()

        if wallet_call != self.call_positions or wallet_put != self.put_positions:
            logger.info(f"📊 Wallet sync: CALL {self.call_positions}→{wallet_call}, PUT {self.put_positions}→{wallet_put}")
            self.call_positions = wallet_call
            self.put_positions = wallet_put

        self.last_wallet_check = time.time()

    def verify_asset_ids_match_files(self):
        """
        Periodically verify that our asset IDs match the JSON files
        This catches cases where files updated but we missed it
        """
        if time.time() - self.last_asset_id_check < self.asset_id_check_interval:
            return

        self.last_asset_id_check = time.time()

        put_data = read_json_safe(PUT_FILE)
        call_data = read_json_safe(CALL_FILE)

        if not put_data or not call_data:
            return

        file_put_id = put_data.get('asset_id')
        file_call_id = call_data.get('asset_id')

        # Check for mismatch
        put_mismatch = file_put_id != self.current_put_id
        call_mismatch = file_call_id != self.current_call_id

        if put_mismatch or call_mismatch:
            logger.warning(f"\n⚠️  ASSET ID MISMATCH DETECTED!")
            if put_mismatch:
                logger.warning(f"   PUT: Memory ...{self.current_put_id[-12:] if self.current_put_id else 'N/A'} ≠ File ...{file_put_id[-12:] if file_put_id else 'N/A'}")
            if call_mismatch:
                logger.warning(f"   CALL: Memory ...{self.current_call_id[-12:] if self.current_call_id else 'N/A'} ≠ File ...{file_call_id[-12:] if file_call_id else 'N/A'}")

            logger.info(f"🔄 Updating to file asset IDs and resetting positions...")

            # Update IDs
            self.current_put_id = file_put_id
            self.current_call_id = file_call_id

            # Reset position tracking - we may have been tracking old tokens
            self.call_position = None
            self.put_position = None
            self.call_positions = 0
            self.put_positions = 0

            # Now check wallet with correct IDs
            wallet_call, wallet_put = self.count_wallet_positions()
            self.call_positions = wallet_call
            self.put_positions = wallet_put

            logger.info(f"   ✅ Corrected: CALL={wallet_call}, PUT={wallet_put}")

    def reload_asset_ids(self, max_retries: int = 3, retry_delay: float = 10) -> bool:
        """
        Reload PUT and CALL asset IDs from data files
        Retries if IDs haven't changed (files not updated yet)
        Returns True if new IDs were loaded
        """
        old_put_id = self.current_put_id
        old_call_id = self.current_call_id

        for attempt in range(max_retries):
            logger.info(f"   Attempt {attempt + 1}/{max_retries}: Reading asset IDs from files...")

            # Force re-read from files
            put_data = read_json_safe(PUT_FILE)
            call_data = read_json_safe(CALL_FILE)

            if not put_data or not call_data:
                logger.warning(f"   ⚠️  Could not read JSON files, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                continue

            new_put_id = put_data.get('asset_id')
            new_call_id = call_data.get('asset_id')

            logger.info(f"   File PUT ID:  ...{new_put_id[-12:] if new_put_id else 'N/A'}")
            logger.info(f"   File CALL ID: ...{new_call_id[-12:] if new_call_id else 'N/A'}")

            # Check if IDs changed
            put_changed = new_put_id != old_put_id
            call_changed = new_call_id != old_call_id

            if put_changed or call_changed:
                logger.info(f"   ✅ Asset IDs CHANGED:")
                if put_changed:
                    logger.info(f"      PUT:  ...{old_put_id[-12:] if old_put_id else 'N/A'} → ...{new_put_id[-12:]}")
                if call_changed:
                    logger.info(f"      CALL: ...{old_call_id[-12:] if old_call_id else 'N/A'} → ...{new_call_id[-12:]}")

                self.current_put_id = new_put_id
                self.current_call_id = new_call_id
                return True
            else:
                logger.warning(f"   ⚠️  Asset IDs unchanged - files may not have updated yet")
                if attempt < max_retries - 1:
                    logger.info(f"   ⏳ Waiting {retry_delay}s before retry...")
                    time.sleep(retry_delay)

        # After all retries, use whatever we have
        logger.warning(f"   ⚠️  Asset IDs did not change after {max_retries} attempts")
        logger.warning(f"   ⚠️  Using current IDs - may cause incorrect position detection!")

        # Still update in case files were read successfully
        if put_data and call_data:
            self.current_put_id = put_data.get('asset_id')
            self.current_call_id = call_data.get('asset_id')

        return False

    # ========================================================================
    # TRADING FUNCTIONS
    # ========================================================================

    def can_place_order(self, price: float) -> Tuple[bool, str]:
        """Check if we can place an order"""
        # Check time between orders
        time_since_last = time.time() - self.last_order_time
        if time_since_last < MIN_SECONDS_BETWEEN_ORDERS:
            wait_time = MIN_SECONDS_BETWEEN_ORDERS - time_since_last
            return False, f"Wait {wait_time:.0f}s between orders"

        # Check if in last 10 seconds of period
        seconds_remaining = get_seconds_to_period_end()
        if seconds_remaining <= PERIOD_END_BUFFER_SECONDS:
            return False, f"Last {PERIOD_END_BUFFER_SECONDS}s of period"

        # Check price limits
        if price > MAX_BUY_PRICE:
            return False, f"Price {price:.2f} > {MAX_BUY_PRICE}"
        if price < MIN_BUY_PRICE:
            return False, f"Price {price:.2f} < {MIN_BUY_PRICE}"

        return True, "OK"

    def should_buy(self, analysis: Dict) -> bool:
        """Check if buy signal is present"""
        return analysis['sum_bids'] > analysis['sum_asks'] * IMBALANCE_RATIO

    def should_sell(self, analysis: Dict) -> bool:
        """Check if sell signal is present"""
        return analysis['sum_asks'] > analysis['sum_bids'] * IMBALANCE_RATIO

    def is_low_balance_mode(self) -> bool:
        """Check if we're in low balance mode"""
        return self.cached_usdc_balance < MIN_USDC_TO_TRADE

    def execute_buy(self, asset_type: str, token_id: str, ask_price: float) -> bool:
        """Execute TAKER buy order at market price (best ask)"""
        try:
            logger.info(f"\n{'='*70}")
            logger.info(f"🛒 EXECUTING TAKER BUY ORDER - {asset_type}")
            logger.info(f"{'='*70}")
            logger.info(f"📦 Size: {POSITION_SIZE} shares")
            logger.info(f"💰 Market Ask: ${ask_price:.4f}")

            # Use ask price for taker order
            limit_price = ask_price
            logger.info(f"💰 Order Price: ${limit_price:.4f} (taker at ask)")

            required = limit_price * POSITION_SIZE
            logger.info(f"💵 Expected Cost: ${required:.2f}")

            # Check USDC balance
            self.refresh_usdc_balance()
            logger.info(f"💰 USDC Balance: ${self.cached_usdc_balance:.2f}")

            MIN_BALANCE = 4.90
            if self.cached_usdc_balance < MIN_BALANCE:
                logger.error(f"❌ INSUFFICIENT BALANCE: ${self.cached_usdc_balance:.2f} < ${MIN_BALANCE:.2f}")
                return False

            if self.cached_usdc_balance < required:
                logger.error(f"❌ Need: ${required:.2f}, Have: ${self.cached_usdc_balance:.2f}")
                return False

            # Place TAKER order
            logger.info(f"\n🚀 Placing TAKER buy order...")
            start_time = time.time()

            order_id = self.trader.place_buy_order(
                token_id=token_id,
                price=limit_price,
                quantity=POSITION_SIZE
            )

            if not order_id:
                logger.error(f"❌ Failed to place order")
                return False

            logger.info(f"✅ Order placed: {order_id[:16]}... ({time.time() - start_time:.3f}s)")

            # Update timing
            self.last_order_time = time.time()

            # Create trade record
            self.trade_counter += 1
            trade = Trade(
                trade_id=self.trade_counter,
                period_id=self.current_period_id,
                asset_type=asset_type,
                entry_price=limit_price,
                entry_time=datetime.now(),
                size=POSITION_SIZE
            )

            if asset_type == 'CALL':
                self.call_position = trade
            else:
                self.put_position = trade

            # Refresh balance after trade
            time.sleep(1)  # Brief wait for settlement
            self.cached_usdc_balance = self.get_usdc_balance()
            logger.info(f"💰 New USDC Balance: ${self.cached_usdc_balance:.2f}")

            return True

        except Exception as e:
            logger.error(f"❌ Error executing buy: {e}")
            import traceback
            traceback.print_exc()
            return False

    def close_position(self, position: Trade, exit_price: float, reason: str) -> bool:
        """Close a position by executing a sell order"""
        try:
            token_id = self.current_call_id if position.asset_type == 'CALL' else self.current_put_id

            # Verify we have tokens
            actual_balance = self.check_token_balance(token_id)

            if actual_balance < 0.5:
                logger.warning(f"\n⚠️  SELL ABORTED: No tokens in wallet for {position.asset_type}")
                # Clear position tracking since we don't have tokens
                if position.asset_type == 'CALL':
                    self.call_position = None
                else:
                    self.put_position = None
                return False

            size = actual_balance

            logger.info(f"\n{'='*60}")
            logger.info(f"💰 EXECUTING SELL ORDER - {position.asset_type} - {reason}")
            logger.info(f"{'='*60}")
            logger.info(f"📦 Size: {size:.2f} shares")
            logger.info(f"💰 Bid: ${exit_price:.4f}")

            # Calculate P&L
            pnl = (exit_price - position.entry_price) * size
            pnl_pct = ((exit_price / position.entry_price) - 1) * 100 if position.entry_price > 0 else 0
            logger.info(f"📈 P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")

            # Set sell price slightly below bid for taker
            sell_price = max(0.01, exit_price - 0.001) if exit_price > 0 else 0.01
            logger.info(f"💰 Sell Price: ${sell_price:.4f}")

            # Place order
            order_id = self.trader.place_sell_order(
                token_id=token_id,
                price=sell_price,
                quantity=size
            )

            if not order_id:
                logger.error(f"❌ Failed to place sell order")
                return False

            logger.info(f"✅ Sell order placed: {order_id[:16]}...")

            # Wait 3 seconds then cancel to avoid bounce back fills
            time.sleep(3)
            try:
                self.trader.cancel_all_orders()
                logger.info(f"🔄 Cancelled remaining orders after 3s")
            except:
                pass

            # Wait for balance to update
            time.sleep(2)
            balance = self.check_token_balance(token_id)

            if balance < 0.5:
                logger.info(f"✅ Sell FILLED - {position.asset_type} Position CLOSED")

                # Record trade
                position.close(exit_price, datetime.now(), reason)
                self.all_trades.append(position)

                # Add to current period
                if self.current_period_id and self.current_period_id in self.periods:
                    self.periods[self.current_period_id].add_trade(position)

                # Clear position
                if position.asset_type == 'CALL':
                    self.call_position = None
                else:
                    self.put_position = None

                return True
            else:
                logger.warning(f"⚠️  Sell not filled, {balance:.2f} tokens remain")
                return False

        except Exception as e:
            logger.error(f"❌ Error closing position: {e}")
            import traceback
            traceback.print_exc()
            return False

    def execute_simultaneous_trade(self, buy_type: str, buy_analysis: Dict,
                                    sell_position: Optional[Trade], sell_analysis: Optional[Dict]) -> dict:
        """
        Execute buy and sell simultaneously when switching positions
        Returns dict with results: {'buy_success': bool, 'sell_success': bool}
        """
        results = {'buy_success': False, 'sell_success': False}

        buy_token_id = self.current_call_id if buy_type == 'CALL' else self.current_put_id
        buy_price = buy_analysis['best_ask']

        logger.info(f"\n{'='*70}")
        logger.info(f"🔄 SIMULTANEOUS TRADE: BUY {buy_type} + SELL {sell_position.asset_type if sell_position else 'N/A'}")
        logger.info(f"{'='*70}")

        # Execute BUY first
        logger.info(f"\n🛒 Placing BUY {buy_type} @ ${buy_price:.4f}...")
        buy_order_id = None
        try:
            buy_order_id = self.trader.place_buy_order(
                token_id=buy_token_id,
                price=buy_price,
                quantity=POSITION_SIZE
            )
            if buy_order_id:
                logger.info(f"✅ BUY order placed: {buy_order_id[:16]}...")
        except Exception as e:
            logger.error(f"❌ BUY order failed: {e}")

        # Execute SELL if we have a position to close
        sell_order_id = None
        if sell_position and sell_analysis:
            sell_token_id = self.current_call_id if sell_position.asset_type == 'CALL' else self.current_put_id
            sell_price = max(0.01, sell_analysis['best_bid'] - 0.001)

            # Verify we have tokens to sell
            actual_balance = self.check_token_balance(sell_token_id)
            if actual_balance >= 0.5:
                logger.info(f"\n💰 Placing SELL {sell_position.asset_type} @ ${sell_price:.4f}...")
                try:
                    sell_order_id = self.trader.place_sell_order(
                        token_id=sell_token_id,
                        price=sell_price,
                        quantity=actual_balance
                    )
                    if sell_order_id:
                        logger.info(f"✅ SELL order placed: {sell_order_id[:16]}...")
                except Exception as e:
                    logger.error(f"❌ SELL order failed: {e}")
            else:
                logger.warning(f"⚠️  No tokens to sell for {sell_position.asset_type}")
                # Clear position tracking since we don't have tokens
                if sell_position.asset_type == 'CALL':
                    self.call_position = None
                else:
                    self.put_position = None

        # Wait 3 seconds then cancel all orders to avoid bounce back
        logger.info(f"\n⏳ Waiting 3s then cancelling remaining orders...")
        time.sleep(3)
        try:
            self.trader.cancel_all_orders()
            logger.info(f"🔄 Cancelled all remaining orders")
        except:
            pass

        # Wait for balances to update (may take several seconds)
        logger.info(f"⏳ Waiting for balance updates...")
        time.sleep(5)

        # Verify BUY result
        if buy_order_id:
            buy_balance = self.check_token_balance(buy_token_id)
            if buy_balance >= POSITION_SIZE * 0.9:  # 10% tolerance
                results['buy_success'] = True
                logger.info(f"✅ BUY {buy_type} FILLED - Balance: {buy_balance:.2f}")

                # Create trade record
                self.trade_counter += 1
                trade = Trade(
                    trade_id=self.trade_counter,
                    period_id=self.current_period_id,
                    asset_type=buy_type,
                    entry_price=buy_price,
                    entry_time=datetime.now(),
                    size=POSITION_SIZE
                )

                if buy_type == 'CALL':
                    self.call_position = trade
                else:
                    self.put_position = trade

                self.last_order_time = time.time()
            else:
                logger.warning(f"⚠️  BUY {buy_type} NOT FILLED - Balance: {buy_balance:.2f}")

        # Verify SELL result
        if sell_position and sell_order_id:
            sell_token_id = self.current_call_id if sell_position.asset_type == 'CALL' else self.current_put_id
            sell_balance = self.check_token_balance(sell_token_id)

            if sell_balance < 0.5:
                results['sell_success'] = True
                logger.info(f"✅ SELL {sell_position.asset_type} FILLED - Position closed")

                # Record trade
                sell_position.close(sell_analysis['best_bid'], datetime.now(), "signal_switch")
                self.all_trades.append(sell_position)

                if self.current_period_id and self.current_period_id in self.periods:
                    self.periods[self.current_period_id].add_trade(sell_position)

                # Clear position
                if sell_position.asset_type == 'CALL':
                    self.call_position = None
                else:
                    self.put_position = None
            else:
                logger.warning(f"⚠️  SELL {sell_position.asset_type} NOT FILLED - Still holding {sell_balance:.2f}")

        # Log final state
        logger.info(f"\n📊 TRADE RESULTS:")
        logger.info(f"   BUY {buy_type}: {'✅ FILLED' if results['buy_success'] else '❌ NOT FILLED'}")
        if sell_position:
            logger.info(f"   SELL {sell_position.asset_type}: {'✅ FILLED' if results['sell_success'] else '❌ NOT FILLED'}")

        # Refresh balance
        self.cached_usdc_balance = self.get_usdc_balance()
        logger.info(f"   USDC Balance: ${self.cached_usdc_balance:.2f}")
        logger.info(f"{'='*70}")

        return results

    # ========================================================================
    # PERIOD MANAGEMENT
    # ========================================================================

    def start_new_period(self):
        """Initialize a new trading period"""
        now = datetime.now()
        period_id = get_period_id(now)

        # Save previous period if exists
        if self.current_period_id and self.current_period_id in self.periods:
            self.periods[self.current_period_id].end_time = now
            self.save_period_trades(self.current_period_id)

        # Start new period
        self.current_period_id = period_id
        period_start = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)

        self.periods[period_id] = PeriodStats(
            period_id=period_id,
            start_time=period_start
        )

        # Reset positions - don't trust old tracking
        self.call_position = None
        self.put_position = None
        # Don't reset call_positions/put_positions here - they should already be set by handle_period_end

        # Reset order timing
        self.last_order_time = 0

        logger.info(f"\n{'='*70}")
        logger.info(f"🆕 NEW PERIOD: {period_id}")
        logger.info(f"   Period ends at: {get_period_end(now).strftime('%H:%M:%S')}")
        logger.info(f"   CALL ID: ...{self.current_call_id[-12:] if self.current_call_id else 'N/A'}")
        logger.info(f"   PUT ID: ...{self.current_put_id[-12:] if self.current_put_id else 'N/A'}")
        logger.info(f"   Positions: CALL={self.call_positions}, PUT={self.put_positions}")
        logger.info(f"{'='*70}")

    def handle_period_end(self):
        """Handle end of period - cancel orders, wait, reload NEW assets"""
        logger.info(f"\n{'='*80}")
        logger.info(f"⚠️  PERIOD END - Processing settlements")
        logger.info(f"{'='*80}")

        # Step 1: Cancel all orders IMMEDIATELY
        logger.info(f"\n🔄 Step 1: Canceling all open orders...")
        try:
            cancelled_count = self.trader.cancel_all_orders()
            logger.info(f"   ✅ Cancelled {cancelled_count} orders")
        except Exception as e:
            logger.error(f"   ❌ Error cancelling orders: {e}")

        # Step 2: Clear ALL position tracking - old positions are EXPIRED
        logger.info(f"\n🔄 Step 2: Clearing expired position tracking...")
        if self.call_position:
            logger.info(f"   Old CALL position @ {self.call_position.entry_price:.2f} - EXPIRED")
        if self.put_position:
            logger.info(f"   Old PUT position @ {self.put_position.entry_price:.2f} - EXPIRED")

        # CRITICAL: Reset everything - old options are GONE
        self.call_position = None
        self.put_position = None
        self.call_positions = 0  # MUST be 0 - old tokens expired
        self.put_positions = 0   # MUST be 0 - old tokens expired

        logger.info(f"   ✅ All position tracking reset to 0")

        # Step 3: Wait for new market data files to be written
        logger.info(f"\n🔄 Step 3: Waiting 10 seconds for new market data...")
        time.sleep(10)

        # Step 4: Reload asset IDs with retry logic
        logger.info(f"\n🔄 Step 4: Loading NEW asset IDs from files...")
        ids_changed = self.reload_asset_ids(max_retries=3, retry_delay=10)

        if not ids_changed:
            logger.error(f"   ❌ CRITICAL: Asset IDs did not change!")
            logger.error(f"   ❌ Will retry in main loop - DO NOT TRADE until IDs update")

        # Step 5: Verify wallet is empty for NEW tokens (should be 0)
        logger.info(f"\n🔄 Step 5: Verifying wallet for NEW asset IDs...")

        # Double check with new IDs
        wallet_call = 0
        wallet_put = 0

        if self.current_call_id:
            call_balance = self.check_token_balance(self.current_call_id)
            wallet_call = int(call_balance / (POSITION_SIZE * 0.9)) if call_balance >= POSITION_SIZE * 0.9 else 0
            logger.info(f"   NEW CALL balance: {call_balance:.2f} tokens = {wallet_call} positions")

        if self.current_put_id:
            put_balance = self.check_token_balance(self.current_put_id)
            wallet_put = int(put_balance / (POSITION_SIZE * 0.9)) if put_balance >= POSITION_SIZE * 0.9 else 0
            logger.info(f"   NEW PUT balance: {put_balance:.2f} tokens = {wallet_put} positions")

        # For NEW period, we should have 0 positions (new options just started)
        if wallet_call > 0 or wallet_put > 0:
            if ids_changed:
                # IDs changed but we have tokens - this is actually a real position
                logger.info(f"   📊 Found existing positions in new period (carried over?)")
                self.call_positions = wallet_call
                self.put_positions = wallet_put
            else:
                # IDs didn't change - this is OLD tokens being detected with OLD IDs
                logger.warning(f"   ⚠️  Detected tokens but IDs didn't change - ignoring (old expired tokens)")
                self.call_positions = 0
                self.put_positions = 0
        else:
            self.call_positions = 0
            self.put_positions = 0
            logger.info(f"   ✅ Wallet empty for new period - ready to trade")

        # Refresh USDC balance
        self.cached_usdc_balance = self.get_usdc_balance()
        logger.info(f"\n💰 USDC Balance: ${self.cached_usdc_balance:.2f}")

        logger.info(f"\n✅ Period end processing complete")
        logger.info(f"{'='*80}\n")

    def save_period_trades(self, period_id: str):
        """Save trades for a period to JSON file"""
        if period_id not in self.periods:
            return

        period = self.periods[period_id]
        period_data = period.to_dict()

        # Create filename
        safe_period_id = period_id.replace(":", "-")
        filename = f"trades_{safe_period_id}.json"
        filepath = os.path.join(TRADES_LOG_DIR, filename)

        try:
            with open(filepath, 'w') as f:
                json.dump(period_data, f, indent=2)
            logger.info(f"💾 Trades saved to: {filepath}")
        except Exception as e:
            logger.error(f"❌ Failed to save trades: {e}")

    def save_overall_summary(self):
        """Save overall trading summary to JSON"""
        total_pnl = sum(t.pnl for t in self.all_trades)
        total_wins = sum(1 for t in self.all_trades if t.pnl > 0)
        total_losses = len(self.all_trades) - total_wins

        # Exit reasons breakdown
        exit_reasons = {}
        for t in self.all_trades:
            if t.exit_reason:
                exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        summary_data = {
            "generated_at": datetime.now().isoformat(),
            "overall": {
                "total_trades": len(self.all_trades),
                "total_pnl": round(total_pnl, 2),
                "wins": total_wins,
                "losses": total_losses,
                "win_rate": round(total_wins / len(self.all_trades) * 100, 1) if self.all_trades else 0
            },
            "exit_reasons": exit_reasons,
            "periods": {pid: p.to_dict()['summary'] for pid, p in self.periods.items()}
        }

        filepath = os.path.join(TRADES_LOG_DIR, "summary_overall.json")
        try:
            with open(filepath, 'w') as f:
                json.dump(summary_data, f, indent=2)
            logger.info(f"💾 Overall summary saved to: {filepath}")
        except Exception as e:
            logger.error(f"❌ Failed to save overall summary: {e}")

    # ========================================================================
    # DISPLAY
    # ========================================================================

    def display_status(self, call_analysis: Optional[Dict], put_analysis: Optional[Dict]):
        """Display current status"""
        now = datetime.now()
        seconds_remaining = get_seconds_to_period_end()

        status = f"\r⏰ {now.strftime('%H:%M:%S')} | "

        # Show low balance warning
        if self.in_low_balance_mode:
            status += f"💸 LOW BAL | "

        status += f"Period: {self.current_period_id or 'N/A'} | "

        if call_analysis:
            ratio = call_analysis['bid_ask_ratio']
            status += f"CALL: ${call_analysis['best_ask']:.2f} ({ratio:.1f}x) "
        else:
            status += "CALL: N/A "

        if put_analysis:
            ratio = put_analysis['bid_ask_ratio']
            status += f"| PUT: ${put_analysis['best_ask']:.2f} ({ratio:.1f}x) "
        else:
            status += "| PUT: N/A "

        status += f"| Pos: C{self.call_positions}/P{self.put_positions} "
        status += f"| ${self.cached_usdc_balance:.0f} "
        status += f"| {seconds_remaining:.0f}s"

        print(status, end='', flush=True)

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    def run(self):
        """Main trading loop"""
        logger.info("\n🚀 Starting Order Book Imbalance Bot...")

        last_period_check = ""

        while True:
            try:
                now = datetime.now()
                current_period = get_period_id(now)
                seconds_remaining = get_seconds_to_period_end()
                seconds_into = get_seconds_into_period()

                # Check for period boundary
                if is_period_boundary() and current_period != last_period_check:
                    self.handle_period_end()
                    self.start_new_period()
                    last_period_check = current_period
                    continue

                # Initialize period if needed
                if self.current_period_id != current_period:
                    self.start_new_period()

                # Skip trading in first few seconds (wait for data)
                if seconds_into < PERIOD_START_DELAY:
                    time.sleep(1)
                    continue

                # Refresh balance
                self.refresh_usdc_balance()

                # === LOW BALANCE MODE ===
                if self.is_low_balance_mode():
                    # Enter low balance mode
                    if not self.in_low_balance_mode:
                        self.in_low_balance_mode = True
                        logger.info(f"\n{'='*70}")
                        logger.info(f"💸 LOW BALANCE MODE ACTIVATED")
                        logger.info(f"   Balance: ${self.cached_usdc_balance:.2f} < ${MIN_USDC_TO_TRADE:.2f} required")
                        logger.info(f"   Waiting for wallet to be refilled...")
                        logger.info(f"   Checking every {LOW_BALANCE_CHECK_INTERVAL}s")
                        logger.info(f"{'='*70}")

                    # Log status periodically (every 60s)
                    if time.time() - self.last_low_balance_log >= 60:
                        logger.info(f"💸 Still waiting... Balance: ${self.cached_usdc_balance:.2f} (need ${MIN_USDC_TO_TRADE:.2f})")
                        self.last_low_balance_log = time.time()

                    # Slow down - check less frequently
                    time.sleep(LOW_BALANCE_CHECK_INTERVAL)
                    continue
                else:
                    # Exit low balance mode
                    if self.in_low_balance_mode:
                        self.in_low_balance_mode = False
                        logger.info(f"\n{'='*70}")
                        logger.info(f"✅ LOW BALANCE MODE DEACTIVATED")
                        logger.info(f"   Balance restored: ${self.cached_usdc_balance:.2f}")
                        logger.info(f"   Resuming normal trading...")
                        logger.info(f"{'='*70}")

                # Check if in last 10 seconds - close mode only
                in_last_10s = seconds_remaining <= PERIOD_END_BUFFER_SECONDS

                # Load market data
                call_data = read_json_safe(CALL_FILE)
                put_data = read_json_safe(PUT_FILE)

                call_analysis = analyze_order_book(call_data) if call_data else None
                put_analysis = analyze_order_book(put_data) if put_data else None

                # Display status
                self.display_status(call_analysis, put_analysis)

                # Verify positions from wallet periodically
                self.verify_positions_from_wallet()

                # Verify asset IDs match files (every 60s)
                self.verify_asset_ids_match_files()

                # === LAST 10 SECONDS - NO NEW TRADES ===
                if in_last_10s:
                    if self.call_position or self.put_position:
                        logger.info(f"\n⚠️  Last {PERIOD_END_BUFFER_SECONDS}s - positions will settle at expiry")
                    time.sleep(CHECK_INTERVAL)
                    continue

                # === TRADING LOGIC ===
                # Determine current state and signals
                have_call = self.call_position is not None or self.call_positions > 0
                have_put = self.put_position is not None or self.put_positions > 0

                call_buy_signal = call_analysis and self.should_buy(call_analysis)
                put_buy_signal = put_analysis and self.should_buy(put_analysis)
                call_sell_signal = call_analysis and self.should_sell(call_analysis)
                put_sell_signal = put_analysis and self.should_sell(put_analysis)

                # === SCENARIO: Have CALL, signal to buy PUT (switch positions) ===
                if have_call and not have_put and put_buy_signal and put_analysis:
                    can_buy, reason = self.can_place_order(put_analysis['best_ask'])
                    if can_buy:
                        logger.info(f"\n🔄 SWITCH: Close CALL, Open PUT")
                        results = self.execute_simultaneous_trade(
                            buy_type='PUT',
                            buy_analysis=put_analysis,
                            sell_position=self.call_position,
                            sell_analysis=call_analysis
                        )
                        # Results handled inside execute_simultaneous_trade
                        time.sleep(CHECK_INTERVAL)
                        continue

                # === SCENARIO: Have PUT, signal to buy CALL (switch positions) ===
                if have_put and not have_call and call_buy_signal and call_analysis:
                    can_buy, reason = self.can_place_order(call_analysis['best_ask'])
                    if can_buy:
                        logger.info(f"\n🔄 SWITCH: Close PUT, Open CALL")
                        results = self.execute_simultaneous_trade(
                            buy_type='CALL',
                            buy_analysis=call_analysis,
                            sell_position=self.put_position,
                            sell_analysis=put_analysis
                        )
                        # Results handled inside execute_simultaneous_trade
                        time.sleep(CHECK_INTERVAL)
                        continue

                # === SCENARIO: No positions, CALL buy signal ===
                if not have_call and call_buy_signal and call_analysis:
                    can_buy, reason = self.can_place_order(call_analysis['best_ask'])
                    if can_buy:
                        logger.info(f"\n🔵 CALL BUY SIGNAL: ratio {call_analysis['bid_ask_ratio']:.2f}x")
                        if self.execute_buy('CALL', self.current_call_id, call_analysis['best_ask']):
                            logger.info(f"✅ CALL order executed")

                # === SCENARIO: No positions, PUT buy signal ===
                if not have_put and put_buy_signal and put_analysis:
                    can_buy, reason = self.can_place_order(put_analysis['best_ask'])
                    if can_buy:
                        logger.info(f"\n🔴 PUT BUY SIGNAL: ratio {put_analysis['bid_ask_ratio']:.2f}x")
                        if self.execute_buy('PUT', self.current_put_id, put_analysis['best_ask']):
                            logger.info(f"✅ PUT order executed")

                # === SCENARIO: Have CALL only, CALL sell signal (no switch, just close) ===
                if have_call and self.call_position and call_sell_signal and call_analysis:
                    if not put_buy_signal:  # Only close if not switching
                        logger.info(f"\n📉 CALL SELL SIGNAL: ask_ratio {call_analysis['ask_bid_ratio']:.2f}x")
                        self.close_position(self.call_position, call_analysis['best_bid'], "signal")

                # === SCENARIO: Have PUT only, PUT sell signal (no switch, just close) ===
                if have_put and self.put_position and put_sell_signal and put_analysis:
                    if not call_buy_signal:  # Only close if not switching
                        logger.info(f"\n📉 PUT SELL SIGNAL: ask_ratio {put_analysis['ask_bid_ratio']:.2f}x")
                        self.close_position(self.put_position, put_analysis['best_bid'], "signal")

                # === SCENARIO: Have BOTH positions, sell signal on one ===
                if have_call and have_put:
                    # Close whichever has sell signal
                    if call_sell_signal and self.call_position and call_analysis:
                        logger.info(f"\n📉 CALL SELL SIGNAL (holding both): ask_ratio {call_analysis['ask_bid_ratio']:.2f}x")
                        self.close_position(self.call_position, call_analysis['best_bid'], "signal")

                    if put_sell_signal and self.put_position and put_analysis:
                        logger.info(f"\n📉 PUT SELL SIGNAL (holding both): ask_ratio {put_analysis['ask_bid_ratio']:.2f}x")
                        self.close_position(self.put_position, put_analysis['best_bid'], "signal")

                time.sleep(CHECK_INTERVAL)

            except KeyboardInterrupt:
                print("\n\n🛑 Bot stopped by user")
                logger.info("💾 Saving final state...")
                self.save_overall_summary()
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(CHECK_INTERVAL)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Load credentials
    try:
        env_path = '/home/ubuntu/013_2025_polymarket/keys/keys_ovh38.env'
        credentials = load_credentials_from_env(env_path)
        print(f"✅ Credentials loaded from {env_path}")
    except Exception as e:
        print(f"❌ Error loading credentials: {e}")
        sys.exit(1)

    # Start bot
    bot = OrderBookImbalanceBot(credentials)
    bot.run()
