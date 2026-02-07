#!/usr/bin/env python3
"""
Cleaner Bot - Sells remaining positions from CURRENT 15-minute period

This bot reads from:
- /home/ubuntu/013_2025_polymarket/15M_CALL.json (CURRENT period)
- /home/ubuntu/013_2025_polymarket/15M_PUT.json (CURRENT period)

It checks for positions to sell from:
- /home/ubuntu/013_2025_polymarket/bot021_MM/positions_to_sell.json

Purpose: Sell any remaining positions at best_ask, whatever the price is.

pm2 start cleaner_bot.py --interpreter python3
"""

import json
import time
import sys
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Optional, Dict, List
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

# File paths - CURRENT period files (for prices)
CURRENT_PUT_FILE = "/home/ubuntu/013_2025_polymarket/15M_PUT.json"
CURRENT_CALL_FILE = "/home/ubuntu/013_2025_polymarket/15M_CALL.json"

# Positions to sell file
POSITIONS_FILE = "/home/ubuntu/013_2025_polymarket/bot021_MM/positions_to_sell.json"
TRADES_DIR = "/home/ubuntu/013_2025_polymarket/bot021_MM/cleaner_trades"

# Loop interval
CHECK_INTERVAL = 1.0  # 1 second between checks
TOKEN_BALANCE_INTERVAL = 2.0  # Check balances every 2 seconds

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def read_json(filepath: str) -> Optional[dict]:
    """Read JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        return None

def write_json(filepath: str, data: dict) -> bool:
    """Write JSON file"""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error writing {filepath}: {e}")
        return False

# ============================================================================
# CLEANER BOT CLASS
# ============================================================================

class CleanerBot:
    """Bot to sell remaining positions from CURRENT period"""

    def __init__(self, credentials: dict):
        # Initialize Polymarket trader
        self.trader = PolymarketTrader(
            clob_api_url=credentials['clob_api_url'],
            private_key=credentials['private_key'],
            api_key=credentials['api_key'],
            api_secret=credentials['api_secret'],
            api_passphrase=credentials['api_passphrase']
        )

        # Trades logging
        self.trades_dir = Path(TRADES_DIR)
        self.trades_dir.mkdir(exist_ok=True, parents=True)
        self.today_trades = []
        self.load_today_trades()

        # Timing
        self.last_token_balance_check = 0

        # Pending orders
        self.pending_sell_orders: Dict[str, dict] = {}  # token_id -> order info

        logger.info("="*70)
        logger.info("🧹 CLEANER BOT - Sells remaining CURRENT period positions")
        logger.info("="*70)
        logger.info(f"Positions file: {POSITIONS_FILE}")
        logger.info(f"CALL prices: {CURRENT_CALL_FILE}")
        logger.info(f"PUT prices: {CURRENT_PUT_FILE}")
        logger.info("="*70)

    def load_today_trades(self):
        """Load today's trades if they exist"""
        filename = self.get_today_filename()
        if filename.exists():
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                    self.today_trades = data.get('trades', [])
                    logger.info(f"📂 Loaded {len(self.today_trades)} cleaner trades")
            except:
                self.today_trades = []

    def get_today_filename(self) -> Path:
        """Get filename for today's trades"""
        today = date.today().strftime('%Y%m%d')
        return self.trades_dir / f"cleaner_{today}.json"

    def save_trades(self):
        """Save today's trades to file"""
        filename = self.get_today_filename()

        data = {
            'date': date.today().isoformat(),
            'bot': 'CLEANER',
            'total_trades': len(self.today_trades),
            'trades': self.today_trades
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def check_token_balance(self, token_id: str) -> float:
        """Check balance of specific token"""
        try:
            balance_raw, balance = self.trader.get_token_balance(token_id)
            return balance
        except Exception as e:
            logger.debug(f"Error checking balance: {e}")
            return 0.0

    def load_positions_to_sell(self) -> List[dict]:
        """Load positions that need to be sold"""
        data = read_json(POSITIONS_FILE)
        if data and 'positions' in data:
            return data['positions']
        return []

    def save_positions_to_sell(self, positions: List[dict]):
        """Save updated positions list"""
        data = {
            'updated_at': datetime.now().isoformat(),
            'positions': positions
        }
        write_json(POSITIONS_FILE, data)

    def add_position_to_sell(self, token_type: str, token_id: str, quantity: float, entry_price: float = 0):
        """Add a position to the sell queue (called by main bot)"""
        positions = self.load_positions_to_sell()
        
        # Check if already exists
        for pos in positions:
            if pos.get('token_id') == token_id:
                logger.info(f"   Position {token_type} already in queue")
                return
        
        new_pos = {
            'token_type': token_type,
            'token_id': token_id,
            'quantity': quantity,
            'entry_price': entry_price,
            'added_at': datetime.now().isoformat()
        }
        positions.append(new_pos)
        self.save_positions_to_sell(positions)
        logger.info(f"   ✅ Added {token_type} to sell queue: {token_id[-16:]}")

    def get_current_price(self, token_type: str) -> Optional[float]:
        """Get current best_ask price from CURRENT period files"""
        if token_type == 'CALL':
            data = read_json(CURRENT_CALL_FILE)
        else:
            data = read_json(CURRENT_PUT_FILE)
        
        if data and data.get('best_ask'):
            return data['best_ask'].get('price', 0)
        return None

    def get_current_asset_id(self, token_type: str) -> Optional[str]:
        """Get current asset_id from CURRENT period files"""
        if token_type == 'CALL':
            data = read_json(CURRENT_CALL_FILE)
        else:
            data = read_json(CURRENT_PUT_FILE)
        
        if data:
            return data.get('asset_id')
        return None

    def execute_sell(self, token_type: str, token_id: str, sell_price: float, quantity: float) -> bool:
        """Execute a sell order at best_ask"""
        try:
            logger.info(f"\n{'='*70}")
            logger.info(f"🧹 CLEANER: SELLING {token_type}")
            logger.info(f"{'='*70}")
            logger.info(f"   📦 Size: {quantity:.2f} shares")
            logger.info(f"   💰 Price: ${sell_price:.4f}")
            logger.info(f"   🎯 Token ID: ...{token_id[-16:]}")

            # Verify we have tokens
            actual_balance = self.check_token_balance(token_id)
            logger.info(f"   🔍 Actual Balance: {actual_balance:.2f}")
            
            if actual_balance < 0.5:
                logger.warning(f"   ⚠️  No tokens to sell - removing from queue")
                return True  # Return True to remove from queue

            # Place sell order (market order - no post_only)
            order_id = self.trader.place_sell_order(
                token_id=token_id,
                price=sell_price,
                quantity=actual_balance
            )

            if not order_id:
                logger.error(f"   ❌ Failed to place sell order")
                return False

            logger.info(f"   ✅ SELL ORDER PLACED!")
            logger.info(f"   📋 Order ID: {order_id[:24]}...")
            logger.info(f"{'='*70}\n")

            # Track pending order
            self.pending_sell_orders[token_id] = {
                'order_id': order_id,
                'token_type': token_type,
                'price': sell_price,
                'quantity': actual_balance,
                'time': time.time()
            }

            return True

        except Exception as e:
            error_msg = str(e)
            logger.error(f"   ❌ Error executing sell: {e}")
            
            if 'not enough balance' in error_msg:
                logger.warning(f"   ⚠️  Position already sold - removing from queue")
                return True  # Remove from queue
            
            return False

    def check_pending_orders(self):
        """Check if pending orders have filled"""
        to_remove = []
        
        for token_id, order_info in self.pending_sell_orders.items():
            balance = self.check_token_balance(token_id)
            
            if balance < 0.5:
                # Order filled!
                logger.info(f"\n✅ CLEANER: {order_info['token_type']} SOLD!")
                logger.info(f"   Price: ${order_info['price']:.4f}")
                logger.info(f"   Quantity: {order_info['quantity']:.2f}")
                
                # Record trade
                trade = {
                    'timestamp': datetime.now().isoformat(),
                    'type': order_info['token_type'],
                    'action': 'CLEAN_SELL',
                    'price': order_info['price'],
                    'quantity': order_info['quantity']
                }
                self.today_trades.append(trade)
                self.save_trades()
                
                to_remove.append(token_id)
        
        for token_id in to_remove:
            del self.pending_sell_orders[token_id]

    def run(self):
        """Main loop"""
        logger.info("\n🧹 Starting Cleaner Bot...\n")

        last_current_period_check = 0
        CURRENT_PERIOD_CHECK_INTERVAL = 60  # Check current period tokens every 60 seconds

        try:
            while True:
                current_time = time.time()

                # Check pending orders
                if self.pending_sell_orders:
                    self.check_pending_orders()

                # ========== CHECK CURRENT PERIOD TOKENS EVERY MINUTE ==========
                if current_time - last_current_period_check >= CURRENT_PERIOD_CHECK_INTERVAL:
                    last_current_period_check = current_time
                    
                    logger.info(f"\n{'─'*50}")
                    logger.info(f"🔍 Checking CURRENT period tokens...")
                    logger.info(f"{'─'*50}")
                    
                    # Get current period asset IDs
                    call_asset_id = self.get_current_asset_id('CALL')
                    put_asset_id = self.get_current_asset_id('PUT')
                    
                    if call_asset_id:
                        call_balance = self.check_token_balance(call_asset_id)
                        if call_balance >= 0.5:
                            # We have CALL tokens - sell them!
                            call_price = self.get_current_price('CALL')
                            logger.info(f"   📈 Found CALL: {call_balance:.2f} tokens")
                            logger.info(f"   💰 Best Ask: ${call_price:.4f}" if call_price else "   ⚠️  No price")
                            
                            if call_price and call_price > 0 and call_asset_id not in self.pending_sell_orders:
                                self.execute_sell('CALL', call_asset_id, call_price, call_balance)
                        else:
                            logger.info(f"   📈 CALL: No tokens")
                    
                    if put_asset_id:
                        put_balance = self.check_token_balance(put_asset_id)
                        if put_balance >= 0.5:
                            # We have PUT tokens - sell them!
                            put_price = self.get_current_price('PUT')
                            logger.info(f"   📉 Found PUT: {put_balance:.2f} tokens")
                            logger.info(f"   💰 Best Ask: ${put_price:.4f}" if put_price else "   ⚠️  No price")
                            
                            if put_price and put_price > 0 and put_asset_id not in self.pending_sell_orders:
                                self.execute_sell('PUT', put_asset_id, put_price, put_balance)
                        else:
                            logger.info(f"   📉 PUT: No tokens")
                    
                    logger.info(f"{'─'*50}\n")

                # ========== PROCESS QUEUE FILE ==========
                # Load positions to sell from queue
                positions = self.load_positions_to_sell()
                
                if not positions:
                    time.sleep(CHECK_INTERVAL)
                    continue

                # Process each position
                positions_to_keep = []
                
                for pos in positions:
                    token_type = pos.get('token_type')
                    token_id = pos.get('token_id')
                    quantity = pos.get('quantity', 0)
                    
                    if not token_id:
                        continue

                    # Skip if already have pending order for this token
                    if token_id in self.pending_sell_orders:
                        positions_to_keep.append(pos)
                        continue

                    # Get current price from CURRENT period file
                    # First check if the token_id matches current period
                    current_asset_id = self.get_current_asset_id(token_type)
                    
                    if current_asset_id != token_id:
                        logger.warning(f"   ⚠️  {token_type} token_id doesn't match current period")
                        logger.warning(f"      Queue: ...{token_id[-16:]}")
                        logger.warning(f"      Current: ...{current_asset_id[-16:] if current_asset_id else 'None'}")
                        # Keep in queue - might be from previous period that hasn't settled
                        
                        # Check if we actually have balance for this old token
                        balance = self.check_token_balance(token_id)
                        if balance < 0.5:
                            logger.info(f"   🗑️  No balance for old token - removing")
                            continue  # Don't keep - no balance
                        else:
                            # Try to sell at any price we can get
                            logger.info(f"   🔄 Have {balance:.2f} of old token - trying to sell")
                            # Use a reasonable price - try 0.01 as minimum
                            if self.execute_sell(token_type, token_id, 0.01, balance):
                                continue  # Don't keep if sell initiated
                            else:
                                positions_to_keep.append(pos)
                                continue

                    # Get best_ask price
                    sell_price = self.get_current_price(token_type)
                    
                    if not sell_price or sell_price <= 0:
                        logger.warning(f"   ⚠️  No valid price for {token_type}")
                        positions_to_keep.append(pos)
                        continue

                    # Check actual balance
                    balance = self.check_token_balance(token_id)
                    
                    if balance < 0.5:
                        logger.info(f"   🗑️  {token_type} already sold - removing from queue")
                        continue  # Don't keep

                    # Execute sell
                    logger.info(f"\n📊 Processing {token_type} from queue...")
                    logger.info(f"   Token: ...{token_id[-16:]}")
                    logger.info(f"   Balance: {balance:.2f}")
                    logger.info(f"   Best Ask: ${sell_price:.4f}")
                    
                    if self.execute_sell(token_type, token_id, sell_price, balance):
                        # Sell initiated - don't keep in queue
                        pass
                    else:
                        # Failed - keep in queue
                        positions_to_keep.append(pos)

                # Update positions file
                if len(positions_to_keep) != len(positions):
                    self.save_positions_to_sell(positions_to_keep)

                time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            logger.info("\n\n⏸️  Cleaner Bot stopped by user")
            self.save_trades()


# ============================================================================
# UTILITY: Add position to sell queue (can be called from main bot)
# ============================================================================

def add_to_sell_queue(token_type: str, token_id: str, quantity: float, entry_price: float = 0):
    """Utility function to add position to sell queue"""
    positions = []
    data = read_json(POSITIONS_FILE)
    if data and 'positions' in data:
        positions = data['positions']
    
    # Check if already exists
    for pos in positions:
        if pos.get('token_id') == token_id:
            return  # Already in queue
    
    new_pos = {
        'token_type': token_type,
        'token_id': token_id,
        'quantity': quantity,
        'entry_price': entry_price,
        'added_at': datetime.now().isoformat()
    }
    positions.append(new_pos)
    
    data = {
        'updated_at': datetime.now().isoformat(),
        'positions': positions
    }
    write_json(POSITIONS_FILE, data)
    logger.info(f"✅ Added {token_type} to sell queue")


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
        bot = CleanerBot(credentials)
        bot.run()

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
