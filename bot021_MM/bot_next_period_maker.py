#!/usr/bin/env python3
"""
Bot NEXT Period Market Maker - Trades the NEXT 15-minute binary option
Reads from NEXT_15M_CALL.json and NEXT_15M_PUT.json

Strategy:
- Buy at max 0.49 (maker order)
- Sell at best_ask if above 0.49 (maker order)
- Can have CALL+PUT, CALL only, or PUT only
- Emergency sell at market price in last 20 seconds if imbalanced

pm2 start bot_next_period_maker.py --cron-restart="00 */6 * * *" --interpreter python3
pm2 start cleaner_bot.py --cron-restart="00 */6 * * *" --interpreter python3
"""

import json
import time
import sys
from datetime import datetime, timezone, date
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

# File paths - NEXT period files
PUT_FILE = "/home/ubuntu/013_2025_polymarket/bot021_MM/NEXT_15M_PUT.json"
CALL_FILE = "/home/ubuntu/013_2025_polymarket/bot021_MM/NEXT_15M_CALL.json"
STATE_FILE = "/home/ubuntu/013_2025_polymarket/bot021_MM/bot_next_state.json"
TRADES_DIR = "/home/ubuntu/013_2025_polymarket/bot021_MM/trades"

# Loop intervals
CHECK_INTERVAL = 0.1  # 100ms - trading loop

# Trading Parameters
POSITION_SIZE = 5.2  # 5.2 shares per position
BUFFER_SECONDS = 20  # No trading in first/last 20s of period
POSITION_TOLERANCE = 0.005  # 0.5% tolerance for position matching

# Balance refresh
USDC_CHECK_INTERVAL = 10.0  # Refresh USDC balance every 10s
POSITION_CHECK_INTERVAL = 60.0  # Verify positions every 60s
ASSET_RELOAD_INTERVAL = 10.0  # Reload asset IDs every 10s (important!)
TOKEN_BALANCE_INTERVAL = 5.0  # Refresh token balances every 5s

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

def get_current_quarter_minute() -> int:
    """Get the current quarter minute (0, 15, 30, 45)"""
    now = datetime.now()
    minute = now.minute
    if minute < 15:
        return 0
    elif minute < 30:
        return 15
    elif minute < 45:
        return 30
    else:
        return 45

def get_bin_key(timestamp: float) -> str:
    """Get bin key for current period"""
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    hour = dt.hour
    minute = dt.minute

    for start_min in [0, 15, 30, 45]:
        if minute >= start_min and minute < start_min + 15:
            return f"{hour:02d}:{start_min:02d}"
    return f"{hour:02d}:00"

def positions_are_balanced(call_qty: float, put_qty: float, tolerance: float = POSITION_TOLERANCE) -> bool:
    """
    Check if CALL and PUT positions are balanced (within tolerance).
    Returns True if both positions exist and are within 0.5% of each other.
    """
    if call_qty < 0.5 or put_qty < 0.5:
        return False

    # Check if within tolerance
    ratio = min(call_qty, put_qty) / max(call_qty, put_qty)
    return ratio >= (1 - tolerance)

# ============================================================================
# BOT CLASS
# ============================================================================

class BotNextPeriodMaker:
    """Market Maker bot for NEXT period binary options"""

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
        self.call_position: Optional[Position] = None
        self.put_position: Optional[Position] = None

        # Asset IDs
        self.current_put_id: Optional[str] = None
        self.current_call_id: Optional[str] = None

        # Trades logging
        self.trades_dir = Path(TRADES_DIR)
        self.trades_dir.mkdir(exist_ok=True, parents=True)
        self.today_trades = []
        self.load_today_trades()

        # Period tracking
        self.current_bin: Optional[str] = None
        self.start_buffer_reload_done = False

        # Timing
        self.last_position_check = time.time()
        self.last_asset_reload = time.time()

        # Cached USDC balance
        self.cached_usdc_balance = 0.0
        self.last_usdc_check = 0.0

        # Cached token balances (refresh every 5s)
        self.cached_call_balance = 0.0
        self.cached_put_balance = 0.0
        self.last_token_balance_check = 0.0

        # Pending order tracking - BUY
        self.pending_call_buy_order_id: Optional[str] = None
        self.pending_put_buy_order_id: Optional[str] = None
        self.pending_call_buy_price: Optional[float] = None
        self.pending_put_buy_price: Optional[float] = None

        # Pending order tracking - SELL
        self.pending_call_sell_order_id: Optional[str] = None
        self.pending_put_sell_order_id: Optional[str] = None
        self.pending_call_sell_price: Optional[float] = None
        self.pending_put_sell_price: Optional[float] = None

        # Error cooldown to prevent spam
        self.last_call_error_time: float = 0
        self.last_put_error_time: float = 0
        self.error_cooldown: float = 5.0  # 5 second cooldown after error

        # State save tracking
        self.last_state_save = time.time()

        logger.info("="*80)
        logger.info("🤖 BOT NEXT PERIOD MARKET MAKER")
        logger.info("="*80)
        logger.info(f"Position Size: {POSITION_SIZE} shares")
        logger.info(f"Buy @ best_bid, Sell @ best_ask")
        logger.info(f"Buffer Zone: {BUFFER_SECONDS}s")
        logger.info("="*80)

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

    def get_today_filename(self) -> Path:
        """Get filename for today's trades"""
        today = date.today().strftime('%Y%m%d')
        return self.trades_dir / f"next_period_{today}.json"

    def save_trades(self):
        """Save today's trades to file"""
        filename = self.get_today_filename()

        daily_pnl = sum(t.get('pnl', 0) for t in self.today_trades)
        win_count = sum(1 for t in self.today_trades if t.get('pnl', 0) > 0)
        loss_count = sum(1 for t in self.today_trades if t.get('pnl', 0) < 0)

        data = {
            'date': date.today().isoformat(),
            'strategy': 'NEXT_PERIOD_MAKER',
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
        if time.time() - self.last_usdc_check >= USDC_CHECK_INTERVAL:
            self.cached_usdc_balance = self.get_usdc_balance()
            self.last_usdc_check = time.time()
            logger.debug(f"💰 USDC balance refreshed: ${self.cached_usdc_balance:.2f}")

    def refresh_token_balances(self, force: bool = False):
        """Refresh cached token balances every 5 seconds"""
        if force or time.time() - self.last_token_balance_check >= TOKEN_BALANCE_INTERVAL:
            if self.current_call_id:
                self.cached_call_balance = self.check_token_balance(self.current_call_id)
            else:
                self.cached_call_balance = 0.0

            if self.current_put_id:
                self.cached_put_balance = self.check_token_balance(self.current_put_id)
            else:
                self.cached_put_balance = 0.0

            self.last_token_balance_check = time.time()

    def get_cached_call_balance(self) -> float:
        """Get cached CALL balance"""
        return self.cached_call_balance

    def get_cached_put_balance(self) -> float:
        """Get cached PUT balance"""
        return self.cached_put_balance

    def check_token_balance(self, token_id: str) -> float:
        """Check balance of specific token"""
        try:
            balance_raw, balance = self.trader.get_token_balance(token_id)
            return balance
        except Exception as e:
            logger.debug(f"Error checking balance for {token_id[:12]}...: {e}")
            return 0.0

    def reload_asset_ids(self):
        """Reload PUT and CALL asset IDs from NEXT period data files"""
        put_data = read_json(PUT_FILE)
        call_data = read_json(CALL_FILE)

        if put_data and call_data:
            new_put_id = put_data.get('asset_id')
            new_call_id = call_data.get('asset_id')

            put_changed = new_put_id != self.current_put_id
            call_changed = new_call_id != self.current_call_id

            if put_changed or call_changed:
                logger.info(f"\n   🔄 Asset IDs updated (NEXT period):")
                if put_changed:
                    logger.info(f"   PUT:  ...{new_put_id[-12:] if new_put_id else 'None'}")
                    # Clear PUT position if asset ID changed - it's stale
                    if self.put_position:
                        logger.warning(f"   ⚠️  Clearing stale PUT position (asset ID changed)")
                        self.put_position = None
                    # Cancel pending PUT orders
                    if self.pending_put_buy_order_id or self.pending_put_sell_order_id:
                        logger.warning(f"   ⚠️  Clearing stale PUT pending orders")
                        self.pending_put_buy_order_id = None
                        self.pending_put_sell_order_id = None
                        self.pending_put_buy_price = None
                        self.pending_put_sell_price = None

                if call_changed:
                    logger.info(f"   CALL: ...{new_call_id[-12:] if new_call_id else 'None'}")
                    # Clear CALL position if asset ID changed - it's stale
                    if self.call_position:
                        logger.warning(f"   ⚠️  Clearing stale CALL position (asset ID changed)")
                        self.call_position = None
                    # Cancel pending CALL orders
                    if self.pending_call_buy_order_id or self.pending_call_sell_order_id:
                        logger.warning(f"   ⚠️  Clearing stale CALL pending orders")
                        self.pending_call_buy_order_id = None
                        self.pending_call_sell_order_id = None
                        self.pending_call_buy_price = None
                        self.pending_call_sell_price = None

                self.current_put_id = new_put_id
                self.current_call_id = new_call_id

                # Reset cached balances since asset IDs changed
                self.cached_call_balance = 0.0
                self.cached_put_balance = 0.0
                self.last_token_balance_check = 0  # Force refresh

            self.last_asset_reload = time.time()
        else:
            logger.warning(f"   ⚠️  Could not read asset files")

    def verify_position_from_wallet(self):
        """Verify positions match wallet balances"""
        if not self.current_put_id or not self.current_call_id:
            return

        # Force refresh for verification
        self.refresh_token_balances(force=True)
        put_balance = self.cached_put_balance
        call_balance = self.cached_call_balance

        has_put = put_balance >= 0.5
        has_call = call_balance >= 0.5

        # Sync CALL position
        if has_call:
            if self.call_position is None:
                logger.warning(f"⚠️  Wallet has CALL ({call_balance:.2f}), syncing...")
                call_data = read_json(CALL_FILE)
                price = 0.49
                if call_data and call_data.get('best_bid'):
                    price = call_data['best_bid'].get('price', 0.49)

                self.call_position = Position(
                    token_type='CALL',
                    token_id=self.current_call_id,
                    entry_price=price,
                    entry_time=time.time(),
                    quantity=call_balance
                )
        else:
            if self.call_position is not None:
                logger.warning(f"⚠️  Wallet empty for CALL but tracking shows position")
                self.call_position = None

        # Sync PUT position
        if has_put:
            if self.put_position is None:
                logger.warning(f"⚠️  Wallet has PUT ({put_balance:.2f}), syncing...")
                put_data = read_json(PUT_FILE)
                price = 0.49
                if put_data and put_data.get('best_bid'):
                    price = put_data['best_bid'].get('price', 0.49)

                self.put_position = Position(
                    token_type='PUT',
                    token_id=self.current_put_id,
                    entry_price=price,
                    entry_time=time.time(),
                    quantity=put_balance
                )
        else:
            if self.put_position is not None:
                logger.warning(f"⚠️  Wallet empty for PUT but tracking shows position")
                self.put_position = None

        self.last_position_check = time.time()

    def is_our_order_at_price(self, book_data: dict, price: float, our_size: float) -> bool:
        """
        Check if the order at a given price level is likely ours.
        We check if the size at that price is approximately our position size.
        """
        bids = book_data.get('complete_book', {}).get('bids', [])

        for bid in bids:
            bid_price = float(bid.get('price', 0))
            bid_size = float(bid.get('size', 0))

            if abs(bid_price - price) < 0.001:
                # Check if size matches our order size (within tolerance)
                if abs(bid_size - our_size) / our_size < 0.1:  # 10% tolerance
                    return True
        return False

    def is_our_ask_at_price(self, book_data: dict, price: float, our_size: float) -> bool:
        """Check if the ask order at a given price level is likely ours."""
        asks = book_data.get('complete_book', {}).get('asks', [])

        for ask in asks:
            ask_price = float(ask.get('price', 0))
            ask_size = float(ask.get('size', 0))

            if abs(ask_price - price) < 0.001:
                if abs(ask_size - our_size) / our_size < 0.1:
                    return True
        return False

    def execute_buy_limit(self, token_type: str, token_id: str, limit_price: float) -> bool:
        """Execute a LIMIT buy order (maker)"""
        try:
            # Verify token_id matches current asset ID from file
            if token_type == 'CALL':
                call_data = read_json(CALL_FILE)
                if call_data and call_data.get('asset_id') != token_id:
                    logger.warning(f"   ⚠️  CALL token_id mismatch! Refreshing...")
                    self.reload_asset_ids()
                    return False
            else:
                put_data = read_json(PUT_FILE)
                if put_data and put_data.get('asset_id') != token_id:
                    logger.warning(f"   ⚠️  PUT token_id mismatch! Refreshing...")
                    self.reload_asset_ids()
                    return False

            logger.info(f"\n{'='*70}")
            logger.info(f"🛒 EXECUTING LIMIT BUY ORDER - {token_type}")
            logger.info(f"{'='*70}")
            logger.info(f"   📦 Size: {POSITION_SIZE} shares")
            logger.info(f"   💰 Limit Price: ${limit_price:.4f}")
            logger.info(f"   🎯 Token ID: ...{token_id[-16:] if token_id else 'None'}")

            required = limit_price * POSITION_SIZE
            logger.info(f"   💵 Expected Cost: ${required:.2f}")
            logger.info(f"   💰 USDC Balance: ${self.cached_usdc_balance:.2f}")

            # Check balance
            if self.cached_usdc_balance < required:
                logger.error(f"   ❌ INSUFFICIENT BALANCE: ${self.cached_usdc_balance:.2f} < ${required:.2f}")
                return False

            logger.info(f"   ✅ Balance OK - Placing order...")

            # Place limit order with post_only
            order_id = self.trader.place_buy_order_limit(
                token_id=token_id,
                price=limit_price,
                quantity=POSITION_SIZE
            )

            if not order_id:
                logger.error(f"   ❌ Failed to place order - no order ID returned")
                return False

            logger.info(f"   ✅ ORDER PLACED!")
            logger.info(f"   📋 Order ID: {order_id[:24]}...")
            logger.info(f"   ⏳ Waiting for fill...")
            logger.info(f"{'='*70}\n")

            # Track pending order
            if token_type == 'CALL':
                self.pending_call_buy_order_id = order_id
                self.pending_call_buy_price = limit_price
            else:
                self.pending_put_buy_order_id = order_id
                self.pending_put_buy_price = limit_price

            return True

        except Exception as e:
            logger.error(f"   ❌ Error executing buy: {e}")
            import traceback
            traceback.print_exc()
            return False

    def execute_sell_limit(self, token_type: str, token_id: str, sell_price: float, quantity: float) -> bool:
        """Execute a LIMIT sell order (maker)"""
        try:
            # Verify token_id matches current asset ID from file
            if token_type == 'CALL':
                call_data = read_json(CALL_FILE)
                if call_data and call_data.get('asset_id') != token_id:
                    logger.warning(f"   ⚠️  CALL token_id mismatch! Refreshing...")
                    self.reload_asset_ids()
                    return False
            else:
                put_data = read_json(PUT_FILE)
                if put_data and put_data.get('asset_id') != token_id:
                    logger.warning(f"   ⚠️  PUT token_id mismatch! Refreshing...")
                    self.reload_asset_ids()
                    return False

            logger.info(f"\n{'='*70}")
            logger.info(f"💰 EXECUTING LIMIT SELL ORDER - {token_type}")
            logger.info(f"{'='*70}")
            logger.info(f"   📦 Size: {quantity:.2f} shares")
            logger.info(f"   💰 Limit Price: ${sell_price:.4f}")
            logger.info(f"   🎯 Token ID: ...{token_id[-16:] if token_id else 'None'}")

            # Get entry price for PNL estimate
            position = self.call_position if token_type == 'CALL' else self.put_position
            if position:
                est_pnl = (sell_price - position.entry_price) * quantity
                logger.info(f"   📈 Entry Price: ${position.entry_price:.4f}")
                logger.info(f"   💵 Est. PNL: ${est_pnl:+.4f}")

            # Use cached balance instead of API call
            actual_balance = self.cached_call_balance if token_type == 'CALL' else self.cached_put_balance
            logger.info(f"   🔍 Cached Balance: {actual_balance:.2f}")

            if actual_balance < 0.5:
                logger.warning(f"   ⚠️  SELL ABORTED: No tokens in wallet")
                # Clear position since we have no tokens
                if token_type == 'CALL':
                    self.call_position = None
                else:
                    self.put_position = None
                return False

            logger.info(f"   ✅ Balance OK - Placing order...")

            # Place limit sell order
            from py_clob_client.clob_types import OrderArgs
            from py_clob_client.order_builder.constants import SELL

            order_args = OrderArgs(
                price=sell_price,
                size=quantity,
                side=SELL,
                token_id=token_id
            )

            signed_order = self.trader.client.create_order(order_args)
            response = self.trader.client.post_order(signed_order, post_only=True)

            order_id = response.get('orderID') or response.get('orderId')
            if not order_id:
                logger.error(f"   ❌ Failed to place sell order: {response}")
                return False

            logger.info(f"   ✅ ORDER PLACED!")
            logger.info(f"   📋 Order ID: {order_id[:24]}...")
            logger.info(f"   ⏳ Waiting for fill...")
            logger.info(f"{'='*70}\n")

            # Track pending order
            if token_type == 'CALL':
                self.pending_call_sell_order_id = order_id
                self.pending_call_sell_price = sell_price
            else:
                self.pending_put_sell_order_id = order_id
                self.pending_put_sell_price = sell_price

            return True

        except Exception as e:
            error_msg = str(e)
            logger.error(f"   ❌ Error executing sell: {e}")

            # If balance/allowance error, the position is likely already sold
            if 'not enough balance' in error_msg or 'allowance' in error_msg:
                logger.warning(f"   ⚠️  Position likely already sold - clearing tracking")
                if token_type == 'CALL':
                    self.call_position = None
                    self.cached_call_balance = 0.0
                else:
                    self.put_position = None
                    self.cached_put_balance = 0.0
                # Force refresh balances
                self.last_token_balance_check = 0
            else:
                import traceback
                traceback.print_exc()

            # Refresh asset IDs on any error
            self.reload_asset_ids()
            return False

    def execute_sell_market(self, token_type: str, token_id: str, sell_price: float, quantity: float, reason: str) -> bool:
        """Execute a MARKET sell order (taker) - for emergency exits"""
        try:
            logger.info(f"\n{'='*70}")
            logger.info(f"🚨🚨🚨 EMERGENCY MARKET SELL - {token_type} 🚨🚨🚨")
            logger.info(f"{'='*70}")
            logger.info(f"   ⚠️  REASON: {reason}")
            logger.info(f"   📦 Size: {quantity:.2f} shares")
            logger.info(f"   💰 Price: ${sell_price:.4f}")
            logger.info(f"   🎯 Token ID: ...{token_id[-16:] if token_id else 'None'}")

            # Get entry price for PNL
            position = self.call_position if token_type == 'CALL' else self.put_position
            if position:
                est_pnl = (sell_price - position.entry_price) * quantity
                logger.info(f"   📈 Entry Price: ${position.entry_price:.4f}")
                logger.info(f"   💵 Est. PNL: ${est_pnl:+.4f}")

            # Use cached balance - force refresh for emergency
            self.refresh_token_balances(force=True)
            actual_balance = self.cached_call_balance if token_type == 'CALL' else self.cached_put_balance
            logger.info(f"   🔍 Wallet Balance: {actual_balance:.2f}")

            if actual_balance < 0.5:
                logger.warning(f"   ⚠️  SELL ABORTED: No tokens in wallet")
                return False

            logger.info(f"   🔥 Placing MARKET order (taker)...")

            # Place market sell order (no post_only)
            order_id = self.trader.place_sell_order(
                token_id=token_id,
                price=sell_price,
                quantity=quantity
            )

            if not order_id:
                logger.error(f"   ❌ Failed to place market sell order")
                return False

            logger.info(f"   ✅ MARKET SELL EXECUTED!")
            logger.info(f"   📋 Order ID: {order_id[:24]}...")

            # Record trade
            if position:
                pnl = (sell_price - position.entry_price) * quantity
                trade = {
                    'timestamp': datetime.now().isoformat(),
                    'type': token_type,
                    'action': 'EMERGENCY_SELL',
                    'entry_price': position.entry_price,
                    'exit_price': sell_price,
                    'quantity': quantity,
                    'pnl': pnl,
                    'reason': reason
                }
                self.today_trades.append(trade)
                self.save_trades()
                logger.info(f"   💵 REALIZED PNL: ${pnl:+.4f}")

            # Clear position
            if token_type == 'CALL':
                self.call_position = None
            else:
                self.put_position = None

            logger.info(f"{'='*70}\n")
            return True

        except Exception as e:
            logger.error(f"   ❌ Error executing emergency sell: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_daily_pnl(self) -> float:
        """Get today's total PNL"""
        return sum(t.get('pnl', 0) for t in self.today_trades)

    def settle_period(self):
        """Settle all positions at period end - add to cleaner queue"""
        logger.info(f"\n{'='*70}")
        logger.info(f"⏰⏰⏰ PERIOD END - SETTLING POSITIONS ⏰⏰⏰")
        logger.info(f"{'='*70}")

        # Cancel all open orders
        logger.info(f"   🔄 Canceling all open orders...")
        cancelled = self.trader.cancel_all_orders()
        logger.info(f"   ✅ Cancelled {cancelled} orders")

        # Clear pending order tracking
        self.pending_call_buy_order_id = None
        self.pending_put_buy_order_id = None
        self.pending_call_sell_order_id = None
        self.pending_put_sell_order_id = None

        # Force refresh balances for final check
        self.refresh_token_balances(force=True)
        call_balance = self.cached_call_balance
        put_balance = self.cached_put_balance

        logger.info(f"\n   📊 FINAL POSITIONS:")

        # Add positions to cleaner queue if they exist
        positions_to_clean = []

        # Log CALL position
        if self.call_position and call_balance >= 0.5:
            logger.info(f"      CALL: {call_balance:.2f} shares @ entry ${self.call_position.entry_price:.4f}")
            logger.info(f"            → Adding to cleaner queue")
            positions_to_clean.append({
                'token_type': 'CALL',
                'token_id': self.current_call_id,
                'quantity': call_balance,
                'entry_price': self.call_position.entry_price,
                'added_at': datetime.now().isoformat()
            })
        else:
            logger.info(f"      CALL: None")

        # Log PUT position
        if self.put_position and put_balance >= 0.5:
            logger.info(f"      PUT:  {put_balance:.2f} shares @ entry ${self.put_position.entry_price:.4f}")
            logger.info(f"            → Adding to cleaner queue")
            positions_to_clean.append({
                'token_type': 'PUT',
                'token_id': self.current_put_id,
                'quantity': put_balance,
                'entry_price': self.put_position.entry_price,
                'added_at': datetime.now().isoformat()
            })
        else:
            logger.info(f"      PUT:  None")

        # Save positions to cleaner queue file
        if positions_to_clean:
            cleaner_file = "/home/ubuntu/013_2025_polymarket/bot021_MM/positions_to_sell.json"
            existing_positions = []
            try:
                with open(cleaner_file, 'r') as f:
                    data = json.load(f)
                    existing_positions = data.get('positions', [])
            except:
                pass

            # Add new positions (avoid duplicates)
            existing_ids = {p.get('token_id') for p in existing_positions}
            for pos in positions_to_clean:
                if pos['token_id'] not in existing_ids:
                    existing_positions.append(pos)

            with open(cleaner_file, 'w') as f:
                json.dump({
                    'updated_at': datetime.now().isoformat(),
                    'positions': existing_positions
                }, f, indent=2)

            logger.info(f"\n   🧹 Added {len(positions_to_clean)} position(s) to cleaner queue")

        logger.info(f"\n   💰 Daily PNL so far: ${self.get_daily_pnl():+.4f}")
        logger.info(f"   📋 Total trades today: {len(self.today_trades)}")

        # Clear positions (they will be handled by cleaner bot)
        self.call_position = None
        self.put_position = None

        logger.info(f"{'='*70}\n")

    def run(self):
        """Main trading loop"""
        logger.info("\n🚀 Starting Bot NEXT Period Market Maker\n")

        try:
            while True:
                now = datetime.now()
                current_minute = now.minute
                current_second = now.second
                current_time = time.time()
                timestamp = current_time

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
                    in_start_buffer = seconds_into_period < BUFFER_SECONDS
                    in_end_buffer = seconds_remaining <= BUFFER_SECONDS
                    in_buffer_zone = in_start_buffer or in_end_buffer
                else:
                    seconds_into_period = 0
                    seconds_remaining = 0
                    in_buffer_zone = True

                # Get bin key
                bin_key = get_bin_key(timestamp)

                # ========== NEW PERIOD DETECTED ==========
                if bin_key != self.current_bin:
                    if self.current_bin is not None:
                        # Settle positions from previous period
                        self.settle_period()

                        # Wait 10 seconds before reloading
                        logger.info(f"   ⏳ Waiting 10 seconds before loading new period...")
                        time.sleep(10)

                    # Update period
                    self.current_bin = bin_key
                    self.start_buffer_reload_done = False

                    # Reload asset IDs for NEXT period
                    logger.info(f"\n{'='*70}")
                    logger.info(f"🔄🔄🔄 NEW PERIOD STARTED 🔄🔄🔄")
                    logger.info(f"{'='*70}")
                    logger.info(f"   🕐 Time: {now.strftime('%H:%M:%S')}")
                    logger.info(f"   📅 Period: :{period_start:02d} - :{(period_start+15)%60:02d}")
                    logger.info(f"   ⏳ Buffer: No trading for first {BUFFER_SECONDS}s")
                    logger.info(f"{'='*70}")

                    self.reload_asset_ids()

                    logger.info(f"\n   ✅ Ready to trade NEXT period options")
                    logger.info(f"   📈 CALL ID: ...{self.current_call_id[-16:] if self.current_call_id else 'None'}")
                    logger.info(f"   📉 PUT ID:  ...{self.current_put_id[-16:] if self.current_put_id else 'None'}")
                    logger.info(f"{'='*70}\n")

                # Reload asset IDs at END of start buffer
                if period_start is not None and not self.start_buffer_reload_done:
                    if seconds_into_period >= BUFFER_SECONDS and seconds_into_period <= BUFFER_SECONDS + 2:
                        logger.info(f"\n✅ START BUFFER ENDED - Final asset ID reload")
                        self.reload_asset_ids()
                        self.start_buffer_reload_done = True
                        logger.info(f"🟢 Trading now active for remaining period\n")

                # Periodic refreshes
                if time.time() - self.last_asset_reload >= ASSET_RELOAD_INTERVAL:
                    self.reload_asset_ids()

                if time.time() - self.last_position_check >= POSITION_CHECK_INTERVAL:
                    self.verify_position_from_wallet()

                self.refresh_usdc_balance()

                # Refresh token balances every 5 seconds
                self.refresh_token_balances()

                # Read market data from NEXT period files
                call_data = read_json(CALL_FILE)
                put_data = read_json(PUT_FILE)

                if not all([call_data, put_data]):
                    time.sleep(CHECK_INTERVAL)
                    continue

                # Extract prices
                call_bid_data = call_data.get('best_bid')
                call_ask_data = call_data.get('best_ask')
                put_bid_data = put_data.get('best_bid')
                put_ask_data = put_data.get('best_ask')

                call_bid = call_bid_data.get('price', 0) if call_bid_data else 0
                call_ask = call_ask_data.get('price', 0) if call_ask_data else 0
                call_bid_size = call_bid_data.get('size', 0) if call_bid_data else 0
                put_bid = put_bid_data.get('price', 0) if put_bid_data else 0
                put_ask = put_ask_data.get('price', 0) if put_ask_data else 0
                put_bid_size = put_bid_data.get('size', 0) if put_bid_data else 0

                # Use cached balances (refreshed every 5s)
                call_balance = self.cached_call_balance
                put_balance = self.cached_put_balance

                has_call = call_balance >= 0.5
                has_put = put_balance >= 0.5

                # ========== SKIP TRADING IN BUFFER ZONES ==========
                if in_buffer_zone:
                    time.sleep(CHECK_INTERVAL)
                    continue

                # ========== CHECK PENDING BUY ORDERS ==========
                # Only check pending orders every 5 seconds (using cached balance refresh)
                # Check if CALL buy order filled
                if self.pending_call_buy_order_id:
                    call_balance_now = self.cached_call_balance

                    if call_balance_now >= POSITION_SIZE * 0.9:
                        logger.info(f"\n{'='*70}")
                        logger.info(f"✅✅✅ CALL BUY ORDER FILLED! ✅✅✅")
                        logger.info(f"{'='*70}")
                        logger.info(f"   📦 Quantity: {call_balance_now:.2f} shares")
                        logger.info(f"   💰 Price: ${self.pending_call_buy_price:.4f}")
                        logger.info(f"   💵 Cost: ${self.pending_call_buy_price * call_balance_now:.2f}")
                        logger.info(f"{'='*70}\n")

                        self.call_position = Position(
                            token_type='CALL',
                            token_id=self.current_call_id,
                            entry_price=self.pending_call_buy_price,
                            entry_time=time.time(),
                            quantity=call_balance_now
                        )
                        self.pending_call_buy_order_id = None
                        self.pending_call_buy_price = None
                    else:
                        # Check if we need to cancel and replace (our bid being eaten)
                        if call_bid > 0 and self.pending_call_buy_price:
                            if self.is_our_order_at_price(call_data, self.pending_call_buy_price, POSITION_SIZE):
                                # Our order is the only one at this price - need to go lower
                                new_price = max(0.01, self.pending_call_buy_price - 0.01)
                                if new_price < self.pending_call_buy_price:
                                    logger.info(f"\n⚠️  CALL BUY: Our bid alone at ${self.pending_call_buy_price:.2f}")
                                    logger.info(f"   🔄 Moving bid down to ${new_price:.2f}")
                                    self.trader.cancel_all_orders()
                                    self.pending_call_buy_order_id = None
                                    self.execute_buy_limit('CALL', self.current_call_id, new_price)

                # Check if PUT buy order filled
                if self.pending_put_buy_order_id:
                    put_balance_now = self.cached_put_balance

                    if put_balance_now >= POSITION_SIZE * 0.9:
                        logger.info(f"\n{'='*70}")
                        logger.info(f"✅✅✅ PUT BUY ORDER FILLED! ✅✅✅")
                        logger.info(f"{'='*70}")
                        logger.info(f"   📦 Quantity: {put_balance_now:.2f} shares")
                        logger.info(f"   💰 Price: ${self.pending_put_buy_price:.4f}")
                        logger.info(f"   💵 Cost: ${self.pending_put_buy_price * put_balance_now:.2f}")
                        logger.info(f"{'='*70}\n")

                        self.put_position = Position(
                            token_type='PUT',
                            token_id=self.current_put_id,
                            entry_price=self.pending_put_buy_price,
                            entry_time=time.time(),
                            quantity=put_balance_now
                        )
                        self.pending_put_buy_order_id = None
                        self.pending_put_buy_price = None
                    else:
                        # Check if we need to cancel and replace
                        if put_bid > 0 and self.pending_put_buy_price:
                            if self.is_our_order_at_price(put_data, self.pending_put_buy_price, POSITION_SIZE):
                                new_price = max(0.01, self.pending_put_buy_price - 0.01)
                                if new_price < self.pending_put_buy_price:
                                    logger.info(f"\n⚠️  PUT BUY: Our bid alone at ${self.pending_put_buy_price:.2f}")
                                    logger.info(f"   🔄 Moving bid down to ${new_price:.2f}")
                                    self.trader.cancel_all_orders()
                                    self.pending_put_buy_order_id = None
                                    self.execute_buy_limit('PUT', self.current_put_id, new_price)

                # ========== CHECK PENDING SELL ORDERS ==========
                # Check if CALL sell order filled
                if self.pending_call_sell_order_id:
                    call_balance_now = self.cached_call_balance

                    if call_balance_now < 0.5:
                        logger.info(f"\n{'='*70}")
                        logger.info(f"✅✅✅ CALL SELL ORDER FILLED! ✅✅✅")
                        logger.info(f"{'='*70}")
                        if self.call_position:
                            pnl = (self.pending_call_sell_price - self.call_position.entry_price) * self.call_position.quantity
                            logger.info(f"   📦 Quantity: {self.call_position.quantity:.2f} shares")
                            logger.info(f"   📈 Entry: ${self.call_position.entry_price:.4f}")
                            logger.info(f"   📉 Exit:  ${self.pending_call_sell_price:.4f}")
                            logger.info(f"   💵 REALIZED PNL: ${pnl:+.4f}")
                            trade = {
                                'timestamp': datetime.now().isoformat(),
                                'type': 'CALL',
                                'action': 'SELL',
                                'entry_price': self.call_position.entry_price,
                                'exit_price': self.pending_call_sell_price,
                                'quantity': self.call_position.quantity,
                                'pnl': pnl
                            }
                            self.today_trades.append(trade)
                            self.save_trades()
                        logger.info(f"{'='*70}\n")
                        self.call_position = None
                        self.pending_call_sell_order_id = None
                        self.pending_call_sell_price = None
                    else:
                        # Check if ask price changed - need to update order
                        if call_ask > 0 and self.pending_call_sell_price:
                            if abs(call_ask - self.pending_call_sell_price) >= 0.01:
                                if call_ask > MIN_SELL_PRICE:
                                    logger.info(f"\n⚠️  CALL SELL: Ask price changed ${self.pending_call_sell_price:.2f} → ${call_ask:.2f}")
                                    logger.info(f"   🔄 Updating sell order...")
                                    self.trader.cancel_all_orders()
                                    self.pending_call_sell_order_id = None
                                    self.execute_sell_limit('CALL', self.current_call_id, call_ask, call_balance_now)

                # Check if PUT sell order filled
                if self.pending_put_sell_order_id:
                    put_balance_now = self.cached_put_balance

                    if put_balance_now < 0.5:
                        logger.info(f"\n{'='*70}")
                        logger.info(f"✅✅✅ PUT SELL ORDER FILLED! ✅✅✅")
                        logger.info(f"{'='*70}")
                        if self.put_position:
                            pnl = (self.pending_put_sell_price - self.put_position.entry_price) * self.put_position.quantity
                            logger.info(f"   📦 Quantity: {self.put_position.quantity:.2f} shares")
                            logger.info(f"   📈 Entry: ${self.put_position.entry_price:.4f}")
                            logger.info(f"   📉 Exit:  ${self.pending_put_sell_price:.4f}")
                            logger.info(f"   💵 REALIZED PNL: ${pnl:+.4f}")
                            trade = {
                                'timestamp': datetime.now().isoformat(),
                                'type': 'PUT',
                                'action': 'SELL',
                                'entry_price': self.put_position.entry_price,
                                'exit_price': self.pending_put_sell_price,
                                'quantity': self.put_position.quantity,
                                'pnl': pnl
                            }
                            self.today_trades.append(trade)
                            self.save_trades()
                        logger.info(f"{'='*70}\n")
                        self.put_position = None
                        self.pending_put_sell_order_id = None
                        self.pending_put_sell_price = None
                    else:
                        # Check if ask price changed
                        if put_ask > 0 and self.pending_put_sell_price:
                            if abs(put_ask - self.pending_put_sell_price) >= 0.01:
                                if put_ask > MIN_SELL_PRICE:
                                    logger.info(f"\n⚠️  PUT SELL: Ask price changed ${self.pending_put_sell_price:.2f} → ${put_ask:.2f}")
                                    logger.info(f"   🔄 Updating sell order...")
                                    self.trader.cancel_all_orders()
                                    self.pending_put_sell_order_id = None
                                    self.execute_sell_limit('PUT', self.current_put_id, put_ask, put_balance_now)

                # ========== TRADING LOGIC ==========
                # Use cached balances (already refreshed above)
                call_balance = self.cached_call_balance
                put_balance = self.cached_put_balance
                has_call = call_balance >= 0.5
                has_put = put_balance >= 0.5

                # Buy at best_bid, sell at best_ask (no price limits)
                call_buy_price = call_bid if call_bid > 0 else 0
                put_buy_price = put_bid if put_bid > 0 else 0
                call_sell_price = call_ask if call_ask > 0 else 0
                put_sell_price = put_ask if put_ask > 0 else 0

                # ========== STATUS LOG (every 5 seconds) ==========
                if int(current_time) % 5 == 0 and not hasattr(self, '_last_status_log') or \
                   (hasattr(self, '_last_status_log') and current_time - self._last_status_log >= 5):
                    self._last_status_log = current_time

                    logger.info(f"\n{'─'*70}")
                    logger.info(f"📊 STATUS | {now.strftime('%H:%M:%S')} | Period :{period_start:02d} | {seconds_remaining}s left")
                    logger.info(f"{'─'*70}")
                    logger.info(f"   💵 USDC Balance: ${self.cached_usdc_balance:.2f}")
                    logger.info(f"   ")
                    logger.info(f"   📈 CALL | Bid: ${call_bid:.4f} (sz:{call_bid_size:.1f}) | Ask: ${call_ask:.4f}")
                    logger.info(f"   📉 PUT  | Bid: ${put_bid:.4f} (sz:{put_bid_size:.1f}) | Ask: ${put_ask:.4f}")
                    logger.info(f"   ")
                    logger.info(f"   🎯 POSITIONS:")
                    if has_call:
                        entry_p = self.call_position.entry_price if self.call_position else 0
                        unrealized = (call_bid - entry_p) * call_balance if entry_p > 0 else 0
                        logger.info(f"      CALL: {call_balance:.2f} shares @ ${entry_p:.4f} | Unrealized: ${unrealized:+.4f}")
                    else:
                        logger.info(f"      CALL: None")
                    if has_put:
                        entry_p = self.put_position.entry_price if self.put_position else 0
                        unrealized = (put_bid - entry_p) * put_balance if entry_p > 0 else 0
                        logger.info(f"      PUT:  {put_balance:.2f} shares @ ${entry_p:.4f} | Unrealized: ${unrealized:+.4f}")
                    else:
                        logger.info(f"      PUT:  None")
                    logger.info(f"   ")
                    logger.info(f"   📋 PENDING ORDERS:")
                    if self.pending_call_buy_order_id:
                        logger.info(f"      CALL BUY: ${self.pending_call_buy_price:.4f}")
                    if self.pending_put_buy_order_id:
                        logger.info(f"      PUT BUY:  ${self.pending_put_buy_price:.4f}")
                    if self.pending_call_sell_order_id:
                        logger.info(f"      CALL SELL: ${self.pending_call_sell_price:.4f}")
                    if self.pending_put_sell_order_id:
                        logger.info(f"      PUT SELL:  ${self.pending_put_sell_price:.4f}")
                    if not any([self.pending_call_buy_order_id, self.pending_put_buy_order_id,
                               self.pending_call_sell_order_id, self.pending_put_sell_order_id]):
                        logger.info(f"      None")
                    logger.info(f"   ")
                    logger.info(f"   💰 Daily PNL: ${self.get_daily_pnl():+.4f} ({len(self.today_trades)} trades)")
                    logger.info(f"{'─'*70}")

                # ========== SCENARIO: NO POSITIONS ==========
                if not has_call and not has_put:
                    # Can buy both CALL and PUT at best_bid
                    if not self.pending_call_buy_order_id and self.current_call_id and call_buy_price > 0:
                        if current_time - self.last_call_error_time > self.error_cooldown:
                            logger.info(f"\n📊 STRATEGY: No positions")
                            logger.info(f"   🎯 Action: Buy CALL at best_bid ${call_buy_price:.4f}")
                            self.execute_buy_limit('CALL', self.current_call_id, call_buy_price)

                    if not self.pending_put_buy_order_id and self.current_put_id and put_buy_price > 0:
                        if current_time - self.last_put_error_time > self.error_cooldown:
                            logger.info(f"\n📊 STRATEGY: No positions")
                            logger.info(f"   🎯 Action: Buy PUT at best_bid ${put_buy_price:.4f}")
                            self.execute_buy_limit('PUT', self.current_put_id, put_buy_price)

                # ========== SCENARIO: HAVE CALL ONLY ==========
                elif has_call and not has_put:
                    # MUST have: SELL order for CALL + BUY order for PUT

                    # 1. Sell CALL at best_ask
                    if not self.pending_call_sell_order_id and call_sell_price > 0:
                        if current_time - self.last_call_error_time > self.error_cooldown:
                            logger.info(f"\n📊 STRATEGY: Have CALL only ({call_balance:.2f})")
                            logger.info(f"   🎯 Action: Place SELL CALL at best_ask ${call_sell_price:.4f}")
                            if not self.execute_sell_limit('CALL', self.current_call_id, call_sell_price, call_balance):
                                self.last_call_error_time = current_time

                    # 2. Buy PUT at best_bid
                    if not self.pending_put_buy_order_id and self.current_put_id and put_buy_price > 0:
                        if current_time - self.last_put_error_time > self.error_cooldown:
                            logger.info(f"\n📊 STRATEGY: Have CALL only ({call_balance:.2f})")
                            logger.info(f"   🎯 Action: Place BUY PUT at best_bid ${put_buy_price:.4f}")
                            self.execute_buy_limit('PUT', self.current_put_id, put_buy_price)

                # ========== SCENARIO: HAVE PUT ONLY ==========
                elif has_put and not has_call:
                    # MUST have: SELL order for PUT + BUY order for CALL

                    # 1. Sell PUT at best_ask
                    if not self.pending_put_sell_order_id and put_sell_price > 0:
                        if current_time - self.last_put_error_time > self.error_cooldown:
                            logger.info(f"\n📊 STRATEGY: Have PUT only ({put_balance:.2f})")
                            logger.info(f"   🎯 Action: Place SELL PUT at best_ask ${put_sell_price:.4f}")
                            if not self.execute_sell_limit('PUT', self.current_put_id, put_sell_price, put_balance):
                                self.last_put_error_time = current_time

                    # 2. Buy CALL at best_bid
                    if not self.pending_call_buy_order_id and self.current_call_id and call_buy_price > 0:
                        if current_time - self.last_call_error_time > self.error_cooldown:
                            logger.info(f"\n📊 STRATEGY: Have PUT only ({put_balance:.2f})")
                            logger.info(f"   🎯 Action: Place BUY CALL at best_bid ${call_buy_price:.4f}")
                            self.execute_buy_limit('CALL', self.current_call_id, call_buy_price)

                # ========== SCENARIO: HAVE BOTH CALL AND PUT ==========
                elif has_call and has_put:
                    # Place SELL orders for both at best_ask
                    if not self.pending_call_sell_order_id and call_sell_price > 0:
                        if current_time - self.last_call_error_time > self.error_cooldown:
                            logger.info(f"\n📊 STRATEGY: Have BOTH (CALL:{call_balance:.2f}, PUT:{put_balance:.2f})")
                            logger.info(f"   🎯 Action: Place SELL CALL at best_ask ${call_sell_price:.4f}")
                            if not self.execute_sell_limit('CALL', self.current_call_id, call_sell_price, call_balance):
                                self.last_call_error_time = current_time

                    if not self.pending_put_sell_order_id and put_sell_price > 0:
                        if current_time - self.last_put_error_time > self.error_cooldown:
                            logger.info(f"\n📊 STRATEGY: Have BOTH (CALL:{call_balance:.2f}, PUT:{put_balance:.2f})")
                            logger.info(f"   🎯 Action: Place SELL PUT at best_ask ${put_sell_price:.4f}")
                            if not self.execute_sell_limit('PUT', self.current_put_id, put_sell_price, put_balance):
                                self.last_put_error_time = current_time

                time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            logger.info("\n\n⏸️  Stopped by user")

            # Cancel all orders
            logger.info(f"🔄 Canceling all orders...")
            self.trader.cancel_all_orders()

            # Save trades
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
        bot = BotNextPeriodMaker(credentials)
        bot.run()

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
