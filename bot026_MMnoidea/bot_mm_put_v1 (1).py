#!/usr/bin/env python3
"""
PUT Market Maker Bot - Provides liquidity with limit orders
============================================================
Strategy: Place limit orders on both sides of the market
- BUY: 0.05 below best bid
- SELL: 0.05 above best ask
- Check orderbook every 0.1 seconds
- Cancel and replace orders when market moves
- Last minute of period: only SELL (close positions)
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

# ========================================================================
# CONFIGURATION
# ========================================================================

# Files
CREDENTIALS_FILE = '/home/ubuntu/013_2025_polymarket/keys/keys_ovh39.env'  # PUT bot wallet
PUT_FILE = "/home/ubuntu/013_2025_polymarket/15M_PUT.json"
BALANCE_FILE = "/home/ubuntu/013_2025_polymarket/bot025_Fill_and_Kill/put_mm_status.json"

# Trading parameters
POSITION_SIZE = 5.2  # Tokens per order
BUY_OFFSET = 0.05  # Place buy 0.05 below best bid
SELL_OFFSET = 0.05  # Place sell 0.05 above best ask
MIN_ORDER_VALUE = 1.0  # Platform minimum

# Timing
ORDERBOOK_CHECK_INTERVAL = 0.1  # Check every 0.1 seconds
BALANCE_CHECK_INTERVAL = 10  # Update balance every 10s
ASSET_ID_CHECK_INTERVAL = 10  # Check for new period every 10s
STATUS_REPORT_INTERVAL = 10  # Print status every 10s
PERIOD_END_BUFFER_SECONDS = 60  # Last 60 seconds: only sell

# Position limits
MAX_LONG_POSITIONS = 2  # Maximum positions we hold
MAX_SPREAD = 0.10  # Don't trade if spread > 20 cents

# ========================================================================
# HELPER FUNCTIONS
# ========================================================================

def read_json_safe(filepath: str, max_depth: int = 3) -> Optional[dict]:
    """Safely read JSON with error handling"""
    if not os.path.exists(filepath):
        return None

    for attempt in range(max_depth):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            if attempt < max_depth - 1:
                time.sleep(0.1)
            else:
                logger.error(f"JSON decode error in {filepath}: {e}")
                return None
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            return None
    return None

def write_json_safe(filepath: str, data: dict):
    """Safely write JSON"""
    try:
        temp_file = f"{filepath}.tmp"
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(temp_file, filepath)
    except Exception as e:
        logger.error(f"Error writing {filepath}: {e}")

def get_seconds_to_period_end(now: datetime = None) -> float:
    """Get seconds until next 15-minute period boundary"""
    if now is None:
        now = datetime.now()

    minute = now.minute
    second = now.second
    microsecond = now.microsecond

    minutes_into_period = minute % 15
    seconds_into_period = minutes_into_period * 60 + second + microsecond / 1_000_000
    period_length = 15 * 60

    return period_length - seconds_into_period

# ========================================================================
# MARKET MAKER BOT CLASS
# ========================================================================

class PutMarketMakerBot:
    def __init__(self):
        logger.info("🚀 Initializing PUT Market Maker Bot...")

        # Load credentials
        credentials = load_credentials_from_env(CREDENTIALS_FILE)
        if not credentials:
            raise Exception(f"Failed to load credentials from {CREDENTIALS_FILE}")

        # Initialize trader
        self.trader = PolymarketTrader(
            clob_api_url=credentials['clob_api_url'],
            private_key=credentials['private_key'],
            api_key=credentials['api_key'],
            api_secret=credentials['api_secret'],
            api_passphrase=credentials['api_passphrase']
        )
        logger.info("✅ Credentials loaded")

        # Asset tracking
        self.current_put_id = None

        # Order tracking
        self.active_buy_order_id = None
        self.active_sell_order_id = None
        self.buy_order_price = None
        self.sell_order_price = None

        # Position tracking
        self.long_positions = 0

        # Balance tracking
        self.usdc_balance = 0.0

        # Timing
        self.last_orderbook_check = 0
        self.last_balance_check = 0
        self.last_asset_id_check = 0
        self.last_status_report = 0

        # Token balance cache
        self.token_balance_cache = {}
        self.token_balance_cache_time = {}

        # Statistics
        self.total_buys = 0
        self.total_sells = 0
        self.orders_cancelled = 0

        logger.info("✅ Bot initialized successfully")

    # ========================================================================
    # BALANCE FUNCTIONS
    # ========================================================================

    def check_token_balance(self, token_id: str) -> float:
        """Check balance with 1-second cache"""
        if token_id in self.token_balance_cache:
            cache_age = time.time() - self.token_balance_cache_time.get(token_id, 0)
            if cache_age < 1.0:
                return self.token_balance_cache[token_id]

        try:
            balance_raw, balance = self.trader.get_token_balance(token_id)
            self.token_balance_cache[token_id] = balance
            self.token_balance_cache_time[token_id] = time.time()
            return balance
        except Exception as e:
            logger.debug(f"Error checking balance: {e}")
            return 0.0

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

    def update_positions_and_balance(self):
        """Update positions and balance every 10 seconds"""
        if time.time() - self.last_balance_check < BALANCE_CHECK_INTERVAL:
            return

        self.last_balance_check = time.time()

        # Update USDC
        self.usdc_balance = self.get_usdc_balance()

        # Update positions
        if self.current_put_id:
            balance = self.check_token_balance(self.current_put_id)
            self.long_positions = int(balance / POSITION_SIZE)

        # Save status
        self.save_status()

    def save_status(self):
        """Save current status to JSON"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "usdc_balance": self.usdc_balance,
            "long_positions": self.long_positions,
            "active_buy_order": self.active_buy_order_id is not None,
            "active_sell_order": self.active_sell_order_id is not None,
            "buy_price": self.buy_order_price,
            "sell_price": self.sell_order_price,
            "total_buys": self.total_buys,
            "total_sells": self.total_sells,
            "orders_cancelled": self.orders_cancelled
        }
        write_json_safe(BALANCE_FILE, status)

    # ========================================================================
    # ASSET ID MANAGEMENT
    # ========================================================================

    def load_asset_ids(self) -> bool:
        """Load PUT asset ID (rate limited)"""
        if time.time() - self.last_asset_id_check < ASSET_ID_CHECK_INTERVAL:
            return True

        self.last_asset_id_check = time.time()

        put_data = read_json_safe(PUT_FILE)
        if not put_data:
            logger.error("❌ Could not load PUT asset ID")
            return False

        new_put_id = put_data.get('asset_id')
        if not new_put_id:
            logger.error("❌ PUT asset ID not found")
            return False

        # Check if new period
        if new_put_id != self.current_put_id:
            logger.info(f"🔄 New period detected!")
            logger.info(f"   PUT ID: ...{new_put_id[-12:]}")

            # Cancel any existing orders
            self.cancel_all_orders()

            # Update ID
            self.current_put_id = new_put_id

            # Reset positions
            balance = self.check_token_balance(self.current_put_id)
            self.long_positions = int(balance / POSITION_SIZE)
            logger.info(f"   Initial positions: {self.long_positions}")

        return True

    # ========================================================================
    # ORDER MANAGEMENT
    # ========================================================================

    def get_put_orderbook(self) -> Optional[dict]:
        """Get PUT orderbook"""
        try:
            return self.trader.get_order_book(self.current_put_id)
        except Exception as e:
            logger.debug(f"Error fetching orderbook: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if not order_id:
            return True

        try:
            self.trader.cancel_order(order_id)
            self.orders_cancelled += 1
            return True
        except Exception as e:
            logger.debug(f"Error cancelling order {order_id[:8]}: {e}")
            return False

    def cancel_all_orders(self):
        """Cancel all active orders"""
        if self.active_buy_order_id:
            logger.info(f"🗑️  Cancelling buy order")
            self.cancel_order(self.active_buy_order_id)
            self.active_buy_order_id = None
            self.buy_order_price = None

        if self.active_sell_order_id:
            logger.info(f"🗑️  Cancelling sell order")
            self.cancel_order(self.active_sell_order_id)
            self.active_sell_order_id = None
            self.sell_order_price = None

    def place_buy_limit(self, price: float, quantity: int) -> Optional[str]:
        """Place buy limit order"""
        try:
            order_id = self.trader.place_buy_order_limit(
                token_id=self.current_put_id,
                price=price,
                quantity=quantity
            )
            if order_id:
                self.total_buys += 1
            return order_id
        except Exception as e:
            logger.error(f"Error placing buy limit: {e}")
            return None

    def place_sell_limit(self, price: float, quantity: int) -> Optional[str]:
        """Place sell limit order"""
        try:
            order_id = self.trader.place_sell_order_limit(
                token_id=self.current_put_id,
                price=price,
                quantity=quantity
            )
            if order_id:
                self.total_sells += 1
            return order_id
        except Exception as e:
            logger.error(f"Error placing sell limit: {e}")
            return None

    # ========================================================================
    # MARKET MAKING LOGIC
    # ========================================================================

    def manage_orders(self):
        """Main market making logic - check and update orders"""
        # Rate limit
        if time.time() - self.last_orderbook_check < ORDERBOOK_CHECK_INTERVAL:
            return

        self.last_orderbook_check = time.time()

        # Check if in last minute of period
        seconds_to_end = get_seconds_to_period_end()
        in_last_minute = seconds_to_end < PERIOD_END_BUFFER_SECONDS

        # Get orderbook
        orderbook = self.get_put_orderbook()
        if not orderbook:
            return

        best_bid = orderbook.get('best_bid')
        best_ask = orderbook.get('best_ask')

        # Need valid bid/ask
        if not best_bid or not isinstance(best_bid, dict):
            return
        if not best_ask or not isinstance(best_ask, dict):
            return

        best_bid_price = best_bid.get('price')
        best_ask_price = best_ask.get('price')

        if not best_bid_price or not best_ask_price:
            return

        # Check spread
        spread = best_ask_price - best_bid_price
        if spread > MAX_SPREAD:
            logger.debug(f"Spread too wide: ${spread:.2f}")
            return

        # Calculate our prices
        our_buy_price = round(best_bid_price - BUY_OFFSET, 2)
        our_sell_price = round(best_ask_price + SELL_OFFSET, 2)

        # Clamp prices
        our_buy_price = max(0.01, min(0.95, our_buy_price))
        our_sell_price = max(0.01, min(0.99, our_sell_price))

        # ===== MANAGE BUY ORDERS =====
        if not in_last_minute:  # Only place buys outside last minute
            should_update_buy = False

            # Check if we need to place/update buy order
            if not self.active_buy_order_id:
                should_update_buy = True
            elif self.buy_order_price and abs(self.buy_order_price - our_buy_price) > 0.01:
                # Price moved significantly, cancel and replace
                self.cancel_order(self.active_buy_order_id)
                self.active_buy_order_id = None
                should_update_buy = True

            # Place buy order
            if should_update_buy and self.long_positions < MAX_LONG_POSITIONS:
                # Check minimum order value
                order_value = POSITION_SIZE * our_buy_price
                if order_value >= MIN_ORDER_VALUE:
                    logger.info(f"📗 BUY limit: {POSITION_SIZE} @ ${our_buy_price:.2f} = ${order_value:.2f}")
                    order_id = self.place_buy_limit(our_buy_price, POSITION_SIZE)
                    if order_id:
                        self.active_buy_order_id = order_id
                        self.buy_order_price = our_buy_price
        else:
            # Last minute: cancel buy orders
            if self.active_buy_order_id:
                logger.info("⏰ Last minute - cancelling buy order")
                self.cancel_order(self.active_buy_order_id)
                self.active_buy_order_id = None
                self.buy_order_price = None

        # ===== MANAGE SELL ORDERS =====
        should_update_sell = False

        # Only sell if we have positions
        if self.long_positions > 0:
            if not self.active_sell_order_id:
                should_update_sell = True
            elif self.sell_order_price and abs(self.sell_order_price - our_sell_price) > 0.01:
                # Price moved significantly, cancel and replace
                self.cancel_order(self.active_sell_order_id)
                self.active_sell_order_id = None
                should_update_sell = True

            # Place sell order
            if should_update_sell:
                # Get actual balance
                actual_balance = self.check_token_balance(self.current_put_id)
                quantity_to_sell = int(actual_balance / POSITION_SIZE) * POSITION_SIZE

                if quantity_to_sell >= POSITION_SIZE:
                    # Check minimum order value
                    order_value = quantity_to_sell * our_sell_price
                    if order_value >= MIN_ORDER_VALUE:
                        logger.info(f"📕 SELL limit: {quantity_to_sell} @ ${our_sell_price:.2f} = ${order_value:.2f}")
                        order_id = self.place_sell_limit(our_sell_price, quantity_to_sell)
                        if order_id:
                            self.active_sell_order_id = order_id
                            self.sell_order_price = our_sell_price

    # ========================================================================
    # STATUS REPORTING
    # ========================================================================

    def print_status(self):
        """Print status every 10 seconds"""
        if time.time() - self.last_status_report < STATUS_REPORT_INTERVAL:
            return

        self.last_status_report = time.time()

        seconds_to_end = get_seconds_to_period_end()
        in_last_minute = seconds_to_end < PERIOD_END_BUFFER_SECONDS

        status_icon = "🟢" if not in_last_minute else "🔴"
        mode = "MARKET MAKING" if not in_last_minute else "SELL ONLY (Last Minute)"

        logger.info("=" * 70)
        logger.info(f"{status_icon} PUT MM STATUS: {mode}")
        logger.info(f"💰 USDC: ${self.usdc_balance:.2f} | Positions: {self.long_positions}/{MAX_LONG_POSITIONS}")
        logger.info(f"📊 Buys: {self.total_buys} | Sells: {self.total_sells} | Cancelled: {self.orders_cancelled}")
        logger.info(f"📝 Active Orders: Buy={self.active_buy_order_id is not None} @ ${self.buy_order_price or 0:.2f}, Sell={self.active_sell_order_id is not None} @ ${self.sell_order_price or 0:.2f}")
        logger.info(f"⏰ Time to period end: {seconds_to_end:.0f}s")
        logger.info("=" * 70)

    # ========================================================================
    # MAIN RUN LOOP
    # ========================================================================

    def run(self):
        """Main bot loop"""
        logger.info("=" * 70)
        logger.info("🤖 PUT MARKET MAKER BOT STARTED")
        logger.info("=" * 70)
        logger.info(f"Position size: {POSITION_SIZE}")
        logger.info(f"Max positions: {MAX_LONG_POSITIONS}")
        logger.info(f"Buy offset: ${BUY_OFFSET}")
        logger.info(f"Sell offset: ${SELL_OFFSET}")
        logger.info("=" * 70)

        # Initial asset ID load with retry
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            logger.info(f"Loading asset IDs (attempt {attempt}/{max_retries})...")
            if self.load_asset_ids():
                logger.info("✅ Asset IDs loaded successfully")
                break
            if attempt < max_retries:
                logger.warning(f"⚠️  Retry in 2s...")
                time.sleep(2)
        else:
            logger.error("❌ Failed to load initial asset IDs")
            return

        # Main loop
        try:
            while True:
                # Check for new period
                self.load_asset_ids()

                # Update positions and balance
                self.update_positions_and_balance()

                # Print status
                self.print_status()

                # Manage orders (main logic)
                self.manage_orders()

                # Small sleep
                time.sleep(0.01)

        except KeyboardInterrupt:
            logger.info("\n⏹️  Bot stopped by user")
            self.cancel_all_orders()
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}", exc_info=True)
            self.cancel_all_orders()

# ========================================================================
# MAIN ENTRY POINT
# ========================================================================

if __name__ == "__main__":
    bot = PutMarketMakerBot()
    bot.run()
