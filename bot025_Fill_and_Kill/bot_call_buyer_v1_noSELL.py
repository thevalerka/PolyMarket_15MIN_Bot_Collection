#!/usr/bin/env python3
"""
Polymarket 15M CALL Buyer Bot
Continuously monitors orderbook and places FAK orders 0.03 below best bid
Maintains balance between CALL and PUT positions
"""

import json
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple
from polymarket_trading_core_debug import PolymarketTrader, load_credentials_from_env

# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths
PUT_FILE = "/home/ubuntu/013_2025_polymarket/15M_PUT.json"
CALL_FILE = "/home/ubuntu/013_2025_polymarket/15M_CALL.json"
BALANCE_FILE = "/home/ubuntu/013_2025_polymarket/bot025_Fill_and_Kill/call_put_balance.json"

# Trading parameters
POSITION_SIZE = 5  # Size per order
BID_OFFSET = 0.03   # Buy 0.03 below best bid
POSITION_TOLERANCE = 0.05  # 5% tolerance for position balancing
MIN_USDC_BALANCE = 4.9  # Minimum USDC balance to continue trading
MAX_CALL_POSITIONS = 5  # Maximum CALL positions allowed

# Timing
ORDERBOOK_CHECK_INTERVAL = 0.1  # Check orderbook every 0.1s
BALANCE_CHECK_INTERVAL = 10     # Check positions every 10s
ASSET_ID_CHECK_INTERVAL = 10    # Check for new asset IDs every 10s
PERIOD_END_BUFFER_SECONDS = 20  # Stop trading 20s before period ends
PERIOD_START_DELAY = 15         # Wait 15s after period start

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

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

def write_json_safe(filepath: str, data: dict) -> bool:
    """Write JSON file safely"""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error writing {filepath}: {e}")
        return False

def get_period_end(dt: datetime = None) -> datetime:
    """Get the end time of current period"""
    if dt is None:
        dt = datetime.now()
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
# CALL BUYER BOT CLASS
# ============================================================================

class CallBuyerBot:
    def __init__(self):
        """Initialize the CALL buyer bot"""
        logger.info("🚀 Initializing CALL Buyer Bot...")

        # Load credentials
        try:
            env_path = '/home/ubuntu/013_2025_polymarket/keys/keys_ovh38.env'
            credentials = load_credentials_from_env(env_path)
            print(f"✅ Credentials loaded from {env_path}")


            # Initialize trader
            self.trader = PolymarketTrader(
                clob_api_url=credentials['clob_api_url'],
                private_key=credentials['private_key'],
                api_key=credentials['api_key'],
                api_secret=credentials['api_secret'],
                api_passphrase=credentials['api_passphrase']
            )

        except Exception as e:
            print(f"❌ Error loading credentials: {e}")
            sys.exit(1)

        # Load credentials and initialize trader



        # Asset IDs (will be loaded from files)
        self.current_call_id = None
        self.current_put_id = None

        # Position tracking
        self.call_positions = 0
        self.put_positions = 0
        
        # Separate tracking: presumed (from USDC) vs verified (from wallet)
        self.call_positions_presumed = 0  # Based on USDC spent
        self.call_positions_verified = 0  # Based on wallet tokens
        self.put_positions_presumed = 0   # Read from other bot's JSON
        self.put_positions_verified = 0   # Read from other bot's JSON

        # USDC balance tracking
        self.usdc_balance = 0.0
        self.last_usdc_balance = 0.0
        self.in_sleep_mode = False
        self.expected_positions_from_usdc = 0  # Positions inferred from USDC changes

        # Timing
        self.last_balance_check = 0
        self.last_orderbook_check = 0
        self.last_asset_id_check = 0
        self.imbalance_detected_time = 0  # Track when presumed imbalance was detected
        self.discrepancy_detected_time = 0  # Track when presumed != verified

        # Trading state
        self.trading_enabled = True
        self.total_orders_placed = 0
        self.total_fills = 0

        logger.info("✅ Bot initialized successfully")

    # ========================================================================
    # BALANCE AND POSITION FUNCTIONS
    # ========================================================================

    def check_token_balance(self, token_id: str) -> float:
        """Check balance of specific token"""
        try:
            balance_raw, balance = self.trader.get_token_balance(token_id)
            return balance
        except Exception as e:
            logger.debug(f"Error checking balance for token: {e}")
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

    def count_wallet_positions(self) -> Tuple[int, int]:
        """
        Count actual positions in wallet based on token balances
        Uses tolerance for partial fills
        """
        if not self.current_call_id or not self.current_put_id:
            return 0, 0

        call_balance = self.check_token_balance(self.current_call_id)
        put_balance = self.check_token_balance(self.current_put_id)

        # Calculate positions with tolerance
        tolerance = 0.9
        min_position_size = POSITION_SIZE * tolerance

        call_positions = int(call_balance / min_position_size) if min_position_size > 0 else 0
        put_positions = int(put_balance / min_position_size) if min_position_size > 0 else 0

        return call_positions, put_positions

    def update_balance_file(self):
        """Write current CALL balance to JSON file, preserving PUT data from other bot"""
        # Read existing data to preserve PUT section
        existing_data = read_json_safe(BALANCE_FILE)
        
        # Prepare CALL section with our data
        call_data = {
            "timestamp": datetime.now().isoformat(),
            "usdc_balance": self.usdc_balance,
            "call_balance": self.call_positions * POSITION_SIZE,
            "call_positions": self.call_positions,
            "call_positions_presumed": self.call_positions_presumed,
            "call_positions_verified": self.call_positions_verified,
            "in_sleep_mode": self.in_sleep_mode,
            "trading_enabled": self.trading_enabled,
            "max_positions": MAX_CALL_POSITIONS
        }
        
        # Preserve existing PUT data if it exists, otherwise create placeholder
        if existing_data and "PUT" in existing_data:
            put_data = existing_data["PUT"]
        else:
            put_data = {
                "timestamp": datetime.now().isoformat(),
                "usdc_balance": self.usdc_balance,
                "put_balance": 0,
                "put_positions": 0,
                "put_positions_presumed": 0,
                "put_positions_verified": 0,
                "in_sleep_mode": False,
                "trading_enabled": False,
                "max_positions": 0
            }
        
        # Combine into final structure
        balance_data = {
            "CALL": call_data,
            "PUT": put_data
        }
        
        write_json_safe(BALANCE_FILE, balance_data)

    def check_and_update_positions(self):
        """Check wallet positions and update balance file every 10 seconds"""
        if time.time() - self.last_balance_check < BALANCE_CHECK_INTERVAL:
            return

        # Check USDC balance
        old_usdc = self.usdc_balance
        self.usdc_balance = self.get_usdc_balance()
        
        # Detect USDC decrease (order filled but token balance not updated yet)
        if old_usdc > 0 and self.usdc_balance < old_usdc:
            usdc_spent = old_usdc - self.usdc_balance
            # Estimate positions bought (assuming average price ~0.50)
            estimated_positions = int(usdc_spent / (POSITION_SIZE * 0.50))
            if estimated_positions > 0:
                self.expected_positions_from_usdc += estimated_positions
                logger.info(f"💰 USDC decreased ${usdc_spent:.2f} → expecting +{estimated_positions} CALL position(s)")

        # Check if we need to enter/exit sleep mode
        if self.usdc_balance < MIN_USDC_BALANCE:
            if not self.in_sleep_mode:
                logger.warning(f"💤 ENTERING SLEEP MODE: USDC balance ${self.usdc_balance:.2f} < ${MIN_USDC_BALANCE}")
                self.in_sleep_mode = True
                self.trading_enabled = False
        else:
            if self.in_sleep_mode:
                logger.info(f"☀️  EXITING SLEEP MODE: USDC balance ${self.usdc_balance:.2f} >= ${MIN_USDC_BALANCE}")
                self.in_sleep_mode = False
                # trading_enabled will be set based on position balance below

        # If in sleep mode, still monitor positions but don't enable trading
        # (Position tracking should continue even when low on USDC)

        # ===== VERIFIED POSITIONS (from wallet tokens) =====
        self.call_positions_verified, _ = self.count_wallet_positions()
        
        # ===== PRESUMED POSITIONS (from USDC spent) =====
        # Increment presumed when USDC decreases
        if old_usdc > 0 and self.usdc_balance < old_usdc:
            usdc_spent = old_usdc - self.usdc_balance
            estimated_positions = int(usdc_spent / (POSITION_SIZE * 0.50))
            if estimated_positions > 0:
                self.call_positions_presumed += estimated_positions
                logger.info(f"💰 USDC spent ${usdc_spent:.2f} → presumed positions +{estimated_positions} (total presumed: {self.call_positions_presumed})")
        
        # Sync presumed with verified when verified catches up
        if self.call_positions_verified > self.call_positions_presumed:
            self.call_positions_presumed = self.call_positions_verified
            self.discrepancy_detected_time = 0  # Reset timer
        
        # If presumed differs from verified, start/check timer
        if self.call_positions_presumed != self.call_positions_verified:
            if self.discrepancy_detected_time == 0:
                self.discrepancy_detected_time = time.time()
                logger.info(f"⏱️  Discrepancy detected: presumed={self.call_positions_presumed}, verified={self.call_positions_verified}")
            else:
                time_discrepant = time.time() - self.discrepancy_detected_time
                if time_discrepant >= 60:
                    logger.warning(f"⚠️  1 minute with presumed≠verified, syncing!")
                    logger.info(f"   CALL: presumed {self.call_positions_presumed} → verified {self.call_positions_verified}")
                    self.call_positions_presumed = self.call_positions_verified
                    self.discrepancy_detected_time = 0
        else:
            self.discrepancy_detected_time = 0  # Reset when they match
        
        # ===== READ PUT POSITIONS FROM FILE (managed by other bot) =====
        balance_data = read_json_safe(BALANCE_FILE)
        if balance_data and "PUT" in balance_data:
            self.put_positions_presumed = balance_data["PUT"].get("put_positions_presumed", 0)
            self.put_positions_verified = balance_data["PUT"].get("put_positions_verified", 0)
        else:
            self.put_positions_presumed = 0
            self.put_positions_verified = 0
        
        # Use presumed for position count (more up-to-date)
        old_call = self.call_positions
        self.call_positions = self.call_positions_presumed
        self.put_positions = self.put_positions_presumed
        
        # Log position updates
        if self.call_positions != old_call or self.call_positions_verified != self.call_positions_presumed:
            logger.info(f"📊 CALL: presumed={self.call_positions_presumed}, verified={self.call_positions_verified} | PUT: presumed={self.put_positions_presumed}, verified={self.put_positions_verified}")
            
            if self.call_positions_verified > old_call:
                fills = self.call_positions_verified - old_call
                self.total_fills += fills
                logger.info(f"   ✅ {fills} CALL fill(s) verified in wallet!")
        
        # Update balance file
        self.update_balance_file()
        
        # If in sleep mode, skip trading rules evaluation
        if self.in_sleep_mode:
            self.last_balance_check = time.time()
            return
        
        # ===== TRADING RULES =====
        # Rule 1: MAX_CALL_POSITIONS check (use presumed - more conservative)
        if self.call_positions_presumed >= MAX_CALL_POSITIONS:
            if self.trading_enabled:
                logger.warning(f"🛑 MAX POSITIONS REACHED: CALL presumed={self.call_positions_presumed} >= MAX={MAX_CALL_POSITIONS}")
                self.trading_enabled = False
                self.imbalance_detected_time = 0  # Reset timer
        
        # Rule 2: Stop if PRESUMED CALL > PRESUMED PUT
        elif self.call_positions_presumed > self.put_positions_presumed:
            if self.trading_enabled:
                logger.warning(f"⚠️  PAUSING (PRESUMED): CALL={self.call_positions_presumed} > PUT={self.put_positions_presumed}")
                self.trading_enabled = False
                self.imbalance_detected_time = time.time()  # Start timer
            else:
                # Already paused - check if we should sync presumed to verified after 1 minute
                if self.imbalance_detected_time > 0:
                    time_paused = time.time() - self.imbalance_detected_time
                    
                    # After 1 minute of discrepancy, ALWAYS sync presumed to verified
                    if time_paused >= 60:
                        logger.info(f"⏱️  1 minute elapsed with presumed/verified discrepancy")
                        logger.info(f"   Presumed: CALL={self.call_positions_presumed}, PUT={self.put_positions_presumed}")
                        logger.info(f"   Verified: CALL={self.call_positions_verified}, PUT={self.put_positions_verified}")
                        logger.info(f"   🔄 Syncing presumed to verified (CALL: {self.call_positions_presumed}→{self.call_positions_verified})")
                        
                        self.call_positions_presumed = self.call_positions_verified
                        self.imbalance_detected_time = 0  # Reset timer
                        
                        # Will be re-evaluated in next iteration
        
        # Rule 3: Stop if VERIFIED CALL > VERIFIED PUT
        elif self.call_positions_verified > self.put_positions_verified:
            if self.trading_enabled:
                logger.warning(f"⚠️  PAUSING (VERIFIED): CALL={self.call_positions_verified} > PUT={self.put_positions_verified}")
                self.trading_enabled = False
                self.imbalance_detected_time = 0  # Reset timer
        
        # Resume trading if all conditions are met
        else:
            if not self.trading_enabled:
                logger.info(f"✅ RESUMING: All position checks passed")
                logger.info(f"   Presumed: CALL={self.call_positions_presumed} <= PUT={self.put_positions_presumed}")
                logger.info(f"   Verified: CALL={self.call_positions_verified} <= PUT={self.put_positions_verified}")
                self.trading_enabled = True
            self.imbalance_detected_time = 0  # Reset timer when balanced
        
        self.last_balance_check = time.time()

    # ========================================================================
    # ASSET ID MANAGEMENT
    # ========================================================================

    def load_asset_ids(self) -> bool:
        """Load current CALL and PUT asset IDs from JSON files (rate limited to every 10s)"""
        # Rate limit: only check every ASSET_ID_CHECK_INTERVAL seconds
        if time.time() - self.last_asset_id_check < ASSET_ID_CHECK_INTERVAL:
            return True  # Return True if we have valid IDs cached
        
        self.last_asset_id_check = time.time()
        
        put_data = read_json_safe(PUT_FILE)
        call_data = read_json_safe(CALL_FILE)

        if not put_data or not call_data:
            logger.error("❌ Could not load asset IDs from files")
            return False

        new_put_id = put_data.get('asset_id')
        new_call_id = call_data.get('asset_id')

        if not new_put_id or not new_call_id:
            logger.error("❌ Asset IDs not found in files")
            return False

        # Check if IDs changed (new period)
        if new_call_id != self.current_call_id or new_put_id != self.current_put_id:
            logger.info(f"🔄 New period detected!")
            logger.info(f"   CALL ID: ...{new_call_id[-12:]}")
            logger.info(f"   PUT ID:  ...{new_put_id[-12:]}")

            self.current_call_id = new_call_id
            self.current_put_id = new_put_id

            # Reset ALL position counters for new period
            logger.info(f"   🔄 Resetting all position counters to 0")
            self.call_positions = 0
            self.put_positions = 0
            self.call_positions_presumed = 0
            self.call_positions_verified = 0
            self.put_positions_presumed = 0
            self.put_positions_verified = 0
            self.expected_positions_from_usdc = 0
            self.trading_enabled = True
            self.imbalance_detected_time = 0

            # Check initial balances from wallet
            wallet_call, wallet_put = self.count_wallet_positions()
            self.call_positions_verified = wallet_call
            self.call_positions_presumed = wallet_call
            self.call_positions = wallet_call
            
            logger.info(f"   Initial verified positions: CALL={wallet_call}, PUT={wallet_put}")

            return True

        return True

    # ========================================================================
    # TRADING LOGIC
    # ========================================================================

    def get_call_orderbook(self) -> Optional[dict]:
        """Read CALL orderbook from JSON file"""
        return read_json_safe(CALL_FILE)

    def should_trade_now(self) -> Tuple[bool, str]:
        """
        Check if we should trade right now
        Returns (can_trade, reason)
        """
        # Check sleep mode first
        if self.in_sleep_mode:
            return False, f"Sleep mode (USDC: ${self.usdc_balance:.2f})"

        now = datetime.now()

        # Check if at period boundary
        if is_period_boundary(now):
            return False, "Period boundary"

        # Check if too early in period
        seconds_into = get_seconds_into_period(now)
        if seconds_into < PERIOD_START_DELAY:
            return False, f"Too early ({seconds_into}s < {PERIOD_START_DELAY}s)"

        # Check if too close to period end
        seconds_to_end = get_seconds_to_period_end(now)
        if seconds_to_end < PERIOD_END_BUFFER_SECONDS:
            return False, f"Too close to end ({seconds_to_end:.0f}s remaining)"

        # Check if trading is enabled (position balance)
        if not self.trading_enabled:
            return False, "Positions imbalanced"

        return True, "OK"

    def place_call_order(self):
        """
        Main trading logic: read orderbook and place FAK order 0.03 below best bid
        """
        # Rate limit orderbook checks
        if time.time() - self.last_orderbook_check < ORDERBOOK_CHECK_INTERVAL:
            return

        self.last_orderbook_check = time.time()

        # Check if we should trade
        can_trade, reason = self.should_trade_now()
        if not can_trade:
            return

        # Get orderbook
        call_data = self.get_call_orderbook()
        if not call_data:
            return

        best_bid = call_data.get('best_bid', {})
        best_bid_price = best_bid.get('price')

        if not best_bid_price:
            logger.debug("No best bid available")
            return

        # Calculate our order price (0.03 below best bid)
        our_price = round(best_bid_price - BID_OFFSET, 2)

        # Ensure price is within valid range [0.10, 0.95]
        if our_price < 0.10:
            our_price = 0.10
        elif our_price > 0.95:
            return  # Don't buy if price would be too high

        try:
            # Place FAK order
            logger.info(f"📈 Placing CALL order: size={POSITION_SIZE}, price=${our_price:.2f} (bid=${best_bid_price:.2f})")

            order_id = self.trader.place_buy_order_FAK(
                token_id=self.current_call_id,
                price=our_price,
                quantity=POSITION_SIZE
            )

            self.total_orders_placed += 1

            if order_id:
                logger.info(f"   ✅ Order placed: {order_id}")
            else:
                logger.debug(f"   ⚠️  Order not filled")

        except Exception as e:
            logger.error(f"❌ Error placing order: {e}")

    # ========================================================================
    # MAIN RUN LOOP
    # ========================================================================

    def run(self):
        """Main bot loop"""
        logger.info("=" * 70)
        logger.info("🤖 CALL BUYER BOT STARTED")
        logger.info("=" * 70)
        logger.info(f"Position size: {POSITION_SIZE}")
        logger.info(f"Max CALL positions: {MAX_CALL_POSITIONS}")
        logger.info(f"Bid offset: ${BID_OFFSET}")
        logger.info(f"Position tolerance: {POSITION_TOLERANCE*100}%")
        logger.info(f"Min USDC balance: ${MIN_USDC_BALANCE}")
        logger.info(f"Orderbook check: {ORDERBOOK_CHECK_INTERVAL}s")
        logger.info(f"Balance check: {BALANCE_CHECK_INTERVAL}s")
        logger.info(f"Asset ID check: {ASSET_ID_CHECK_INTERVAL}s")
        logger.info("=" * 70)

        # Initial asset ID load
        if not self.load_asset_ids():
            logger.error("❌ Failed to load initial asset IDs")
            return

        # Main loop
        try:
            while True:
                # Check for new period (asset IDs changed)
                self.load_asset_ids()

                # Update positions and balance file every 10s
                self.check_and_update_positions()

                # Main trading logic - check orderbook and place orders
                self.place_call_order()

                # Small sleep to prevent CPU spinning
                time.sleep(0.01)

        except KeyboardInterrupt:
            logger.info("\n⏹️  Bot stopped by user")
            self.print_stats()
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}", exc_info=True)
            self.print_stats()

    def print_stats(self):
        """Print trading statistics"""
        logger.info("=" * 70)
        logger.info("📊 TRADING STATISTICS")
        logger.info("=" * 70)
        logger.info(f"USDC balance: ${self.usdc_balance:.2f}")
        logger.info(f"Total orders placed: {self.total_orders_placed}")
        logger.info(f"Total fills confirmed: {self.total_fills}")
        if self.total_orders_placed > 0:
            fill_rate = (self.total_fills / self.total_orders_placed) * 100
            logger.info(f"Fill rate: {fill_rate:.1f}%")
        logger.info(f"Final positions: CALL={self.call_positions}, PUT={self.put_positions}")
        logger.info(f"Sleep mode: {self.in_sleep_mode}")
        logger.info("=" * 70)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    bot = CallBuyerBot()
    bot.run()

if __name__ == "__main__":
    main()
