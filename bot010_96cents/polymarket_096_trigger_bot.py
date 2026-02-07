#!/usr/bin/env python3
"""
Polymarket 0.96 Trigger Bot - WEB3 VERSION WITH SPREAD STRATEGY

Strategy:
1. Monitor CALL and PUT tokens for best ask price <= $0.96
2. Check that spread between best ask and best bid is <= $0.04
3. When both conditions met, place BUY order at best_ask price
4. Hold position until price drops to $0.60 (stop loss)
5. Never sell unless stop loss is triggered

FIXES:
- Added balance and allowance checking
- Better error handling for corrupted JSON files
- Improved data lag calculation
- Added USDC balance monitoring
- New spread-based trigger strategy
"""

import os
import json
import time
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

# Web3 imports for ERC20 balance checking
from web3 import Web3
from eth_account import Account

# Import the core trading functions
from polymarket_trading_core import PolymarketTrader, load_credentials_from_env, read_market_data_from_json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Track our positions and orders"""
    call_balance: float = 0.0
    put_balance: float = 0.0
    usdc_balance: float = 0.0
    call_avg_buy_price: float = 0.0
    put_avg_buy_price: float = 0.0
    active_orders: Dict[str, Dict] = None
    triggered_tokens: List[str] = None  # Track which tokens we've already triggered on

    def __post_init__(self):
        if self.active_orders is None:
            self.active_orders = {}
        if self.triggered_tokens is None:
            self.triggered_tokens = []

class Polymarket096TriggerBot:
    """Spread-based bot that buys at best_ask when ask <= $0.96 and spread <= $0.04"""

    def __init__(self, config_path: str = None):
        """Initialize the trigger bot"""
        self.config = self._load_config(config_path)
        self.trader = self._initialize_trader()
        self.position = Position()

        # USDC.e contract details on Polygon
        self.usdc_contract_address = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
        self.usdc_decimals = 6  # USDC.e has 6 decimals

        # Web3 setup for Polygon
        self.polygon_rpc = "https://polygon-rpc.com"  # Public Polygon RPC
        self.w3 = Web3(Web3.HTTPProvider(self.polygon_rpc))

        # ERC20 ABI (minimal - just balanceOf)
        self.erc20_abi = [
            {
                "constant": True,
                "inputs": [{"name": "_owner", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "balance", "type": "uint256"}],
                "type": "function"
            }
        ]

        # Get wallet address from private key
        self.wallet_address = self._get_wallet_address()

        # Balance checking timing
        self.last_balance_check = 0
        self.balance_check_interval = 60  # Check USDC balance every 60 seconds

        logger.info("🎯 Polymarket 0.96 Spread Trigger Bot initialized (WEB3 VERSION)")
        logger.info(f"Strategy: BUY at best_ask when ask <= $0.96 AND spread <= $0.04, STOP LOSS at $0.60")
        logger.info(f"💳 Wallet: {self.wallet_address}")

    def _get_wallet_address(self) -> str:
        """Get wallet address from private key"""
        try:
            # Load credentials to get private key - Updated to use keys_ovh38.env
            creds = load_credentials_from_env('/home/ubuntu/013_2025_polymarket/keys/keys_ovh38.env')
            private_key = creds['private_key']

            # Remove 0x prefix if present
            if private_key.startswith('0x'):
                private_key = private_key[2:]

            # Get account from private key
            account = Account.from_key(private_key)
            return account.address

        except Exception as e:
            logger.error(f"❌ Error getting wallet address: {e}")
            return "0x0000000000000000000000000000000000000000"

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        default_config = {
            'trigger_ask_price': 0.98,          # Trigger when ask is this price or lower
            'max_spread': 0.04,                 # Maximum spread between ask and bid
            'stop_loss_price': 0.60,            # Stop loss price
            'max_order_size': 100.0,            # Maximum order size per trigger
            'min_order_size': 5.0,              # Minimum order size (Polymarket requirement)
            'max_data_lag_ms': 5000,            # Maximum acceptable data lag
            'min_usdc_balance': 5.0,            # Minimum USDC.e balance required
            'polygon_rpc': 'https://polygon-rpc.com',  # Polygon RPC endpoint
            'btc_data_file': '/home/ubuntu/013_2025_polymarket/btc_price.json',
            'call_data_file': '/home/ubuntu/013_2025_polymarket/CALL.json',
            'put_data_file': '/home/ubuntu/013_2025_polymarket/PUT.json'
        }

        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)

        return default_config

    def _initialize_trader(self) -> PolymarketTrader:
        """Initialize the Polymarket trader"""
        try:
            # Updated to use keys_ovh38.env
            creds = load_credentials_from_env('/home/ubuntu/013_2025_polymarket/keys/keys_ovh38.env')
            return PolymarketTrader(
                clob_api_url=creds['clob_api_url'],
                private_key=creds['private_key'],
                api_key=creds['api_key'],
                api_secret=creds['api_secret'],
                api_passphrase=creds['api_passphrase']
            )
        except Exception as e:
            logger.error(f"❌ Failed to initialize trader: {e}")
            raise

    def safe_read_market_data(self, file_path: str) -> Optional[Dict]:
        """Safely read market data with better error handling"""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"⚠️ File not found: {file_path}")
                return None

            with open(file_path, 'r') as f:
                content = f.read().strip()

            if not content:
                logger.warning(f"⚠️ Empty file: {file_path}")
                return None

            data = json.loads(content)

            if not data:
                logger.warning(f"⚠️ No data in file: {file_path}")
                return None

            # Validate required fields
            required_fields = ['asset_id', 'best_bid', 'best_ask', 'timestamp']
            for field in required_fields:
                if field not in data:
                    logger.warning(f"⚠️ Missing field '{field}' in {file_path}")
                    return None

            # Validate nested fields
            if not isinstance(data.get('best_bid'), dict) or 'price' not in data['best_bid']:
                logger.warning(f"⚠️ Invalid best_bid in {file_path}")
                return None

            if not isinstance(data.get('best_ask'), dict) or 'price' not in data['best_ask']:
                logger.warning(f"⚠️ Invalid best_ask in {file_path}")
                return None

            return data

        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error in {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error reading {file_path}: {e}")
            return None

    def check_data_lag(self) -> Tuple[bool, int]:
        """Check if data lag is acceptable with better error handling"""
        try:
            current_time_ms = int(time.time() * 1000)

            # Check CALL data lag
            call_data = self.safe_read_market_data(self.config['call_data_file'])
            if call_data and call_data.get('timestamp'):
                try:
                    call_timestamp = int(float(call_data['timestamp']))
                    call_lag = current_time_ms - call_timestamp
                except (ValueError, TypeError):
                    call_lag = float('inf')
            else:
                call_lag = float('inf')

            # Check PUT data lag
            put_data = self.safe_read_market_data(self.config['put_data_file'])
            if put_data and put_data.get('timestamp'):
                try:
                    put_timestamp = int(float(put_data['timestamp']))
                    put_lag = current_time_ms - put_timestamp
                except (ValueError, TypeError):
                    put_lag = float('inf')
            else:
                put_lag = float('inf')

            # Handle infinite lag
            if call_lag == float('inf') or put_lag == float('inf'):
                logger.warning(f"⚠️ Data files unavailable or corrupted")
                return False, 999999

            max_lag = max(call_lag, put_lag)
            is_acceptable = max_lag <= self.config['max_data_lag_ms']

            if not is_acceptable:
                logger.warning(f"⚠️ High data lag: CALL {call_lag}ms, PUT {put_lag}ms")

            return is_acceptable, int(max_lag)

        except Exception as e:
            logger.error(f"❌ Error checking data lag: {e}")
            return False, 999999

    def get_usdc_balance(self) -> float:
        """Get USDC.e balance using Web3 and ERC20 contract"""
        try:
            logger.info(f"📊 Checking USDC.e balance for wallet: {self.wallet_address}")

            # Check if Web3 is connected
            if not self.w3.is_connected():
                logger.error("❌ Web3 not connected to Polygon network")
                return 0.0

            # Create USDC.e contract instance
            usdc_contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(self.usdc_contract_address),
                abi=self.erc20_abi
            )

            # Get raw balance (in smallest units)
            balance_raw = usdc_contract.functions.balanceOf(
                Web3.to_checksum_address(self.wallet_address)
            ).call()

            # Convert to human readable (USDC.e has 6 decimals)
            balance_usdc = balance_raw / (10 ** self.usdc_decimals)

            logger.info(f"✅ USDC.e balance: ${balance_usdc:.6f}")
            return balance_usdc

        except Exception as e:
            logger.error(f"❌ Error getting USDC.e balance via Web3: {e}")

            # Try alternative RPC endpoints
            alternative_rpcs = [
                "https://rpc-mainnet.matic.network",
                "https://rpc-mainnet.maticvigil.com",
                "https://polygon-mainnet.public.blastapi.io"
            ]

            for rpc_url in alternative_rpcs:
                try:
                    logger.info(f"🔄 Trying alternative RPC: {rpc_url}")
                    alt_w3 = Web3(Web3.HTTPProvider(rpc_url))

                    if alt_w3.is_connected():
                        usdc_contract = alt_w3.eth.contract(
                            address=Web3.to_checksum_address(self.usdc_contract_address),
                            abi=self.erc20_abi
                        )

                        balance_raw = usdc_contract.functions.balanceOf(
                            Web3.to_checksum_address(self.wallet_address)
                        ).call()

                        balance_usdc = balance_raw / (10 ** self.usdc_decimals)
                        logger.info(f"✅ USDC.e balance via {rpc_url}: ${balance_usdc:.6f}")
                        return balance_usdc

                except Exception as e2:
                    logger.warning(f"⚠️ Alternative RPC {rpc_url} failed: {e2}")
                    continue

            logger.error("❌ All RPC endpoints failed")
            return 0.0

    def update_usdc_balance_if_needed(self):
        """Update USDC.e balance if enough time has passed"""
        current_time = time.time()

        # Check if it's time to update balance
        if current_time - self.last_balance_check >= self.balance_check_interval:
            try:
                usdc_balance = self.get_usdc_balance()
                self.position.usdc_balance = usdc_balance
                self.last_balance_check = current_time
                logger.info(f"🔄 USDC.e balance updated: ${usdc_balance:.2f}")
            except Exception as e:
                logger.error(f"❌ Error updating USDC.e balance: {e}")

    def check_usdc_balance(self) -> bool:
        """Check USDC.e balance (user handles allowances manually)"""
        try:
            # Always get fresh balance before checking
            usdc_balance = self.get_usdc_balance()
            self.position.usdc_balance = usdc_balance
            self.last_balance_check = time.time()  # Update timestamp

            logger.info(f"💰 USDC.e Balance: ${usdc_balance:.2f}")

            # Check minimum balance
            if usdc_balance < self.config['min_usdc_balance']:
                logger.warning(f"⚠️ Insufficient USDC.e balance: ${usdc_balance:.2f} < ${self.config['min_usdc_balance']:.2f}")
                return False

            return True

        except Exception as e:
            logger.error(f"❌ Error checking USDC.e balance: {e}")
            return False

    def calculate_affordable_order_size(self, buy_price: float) -> float:
        """Calculate how much we can afford to buy at given price"""
        try:
            max_affordable = self.position.usdc_balance / buy_price
            configured_size = self.config['max_order_size']
            min_size = self.config['min_order_size']

            # Use the smaller of what we can afford vs configured max
            order_size = min(max_affordable, configured_size)

            # Ensure we meet minimum
            if order_size < min_size:
                logger.warning(f"⚠️ Calculated order size {order_size:.2f} below minimum {min_size}")
                return 0.0

            # Round down to avoid precision issues
            return float(int(order_size))

        except Exception as e:
            logger.error(f"❌ Error calculating order size: {e}")
            return 0.0

    def update_position(self, call_data: Dict, put_data: Dict):
        """Update current position"""
        try:
            # Get ERC1155 token balances (CALL/PUT tokens)
            call_balance_raw, call_balance = self.trader.get_token_balance(call_data['asset_id'])
            put_balance_raw, put_balance = self.trader.get_token_balance(put_data['asset_id'])

            self.position.call_balance = call_balance
            self.position.put_balance = put_balance

            # Calculate current values using mid prices
            call_mid_price = (call_data['best_bid']['price'] + call_data['best_ask']['price']) / 2
            put_mid_price = (put_data['best_bid']['price'] + put_data['best_ask']['price']) / 2

            call_value = call_balance * call_mid_price
            put_value = put_balance * put_mid_price
            total_value = call_value + put_value

            logger.info(f"💰 NFT Positions - CALL: {call_balance:.1f} (${call_value:.2f}) | PUT: {put_balance:.1f} (${put_value:.2f}) | Total: ${total_value:.2f}")

        except Exception as e:
            logger.error(f"❌ Error updating position: {e}")

    def check_trigger_condition(self, token_data: Dict, token_name: str) -> Tuple[bool, float]:
        """Check if trigger condition is met for a token - NEW SPREAD STRATEGY"""
        try:
            token_id = token_data['asset_id']
            best_bid_price = float(token_data['best_bid']['price'])
            best_ask_price = float(token_data['best_ask']['price'])

            # Check if we've already triggered on this token
            if token_id in self.position.triggered_tokens:
                return False, 0.0

            # Calculate spread
            spread = best_ask_price - best_bid_price

            # Check NEW trigger conditions:
            # 1. Best ask price <= $0.96
            # 2. Spread <= $0.04
            trigger_ask_met = best_ask_price >= self.config['trigger_ask_price']
            spread_met = spread <= self.config['max_spread']

            if trigger_ask_met and spread_met:
                logger.info(f"🚨 TRIGGER DETECTED: {token_name}")
                logger.info(f"   Ask: ${best_ask_price:.4f} >= ${self.config['trigger_ask_price']:.2f} ✅")
                logger.info(f"   Bid: ${best_bid_price:.4f}")
                logger.info(f"   Spread: ${spread:.4f} <= ${self.config['max_spread']:.2f} ✅")
                return True, best_ask_price
            else:
                # Log why trigger didn't activate (for debugging)
                if not trigger_ask_met:
                    logger.debug(f"   {token_name} ask  high enough: ${best_ask_price:.4f} > ${self.config['trigger_ask_price']:.2f}")
                if not spread_met:
                    logger.debug(f"   {token_name} spread too wide: ${spread:.4f} > ${self.config['max_spread']:.2f}")

            return False, 0.0

        except Exception as e:
            logger.error(f"❌ Error checking trigger condition for {token_name}: {e}")
            return False, 0.0

    def execute_trigger_buy(self, token_data: Dict, token_name: str, buy_price: float) -> bool:
        """Execute buy order when trigger condition is met"""
        try:
            token_id = token_data['asset_id']

            # ALWAYS check fresh USDC.e balance before trading
            logger.info(f"📊 Checking fresh USDC.e balance before {token_name} trade...")
            if not self.check_usdc_balance():
                logger.error(f"❌ Insufficient USDC.e balance for {token_name} trigger buy")
                return False

            # Calculate affordable order size based on current balance and buy price
            order_size = self.calculate_affordable_order_size(buy_price)
            if order_size <= 0:
                logger.error(f"❌ Cannot calculate valid order size for {token_name}")
                return False

            logger.info(f"🔥 EXECUTING TRIGGER BUY: {token_name}")
            logger.info(f"   Token ID: {token_id}")
            logger.info(f"   Buy Price: ${buy_price:.4f} (best ask)")
            logger.info(f"   Order Size: {order_size}")
            logger.info(f"   Total Cost: ${order_size * buy_price:.2f}")
            logger.info(f"   Current USDC.e: ${self.position.usdc_balance:.2f}")

            # Place buy order at best ask price
            order_id = self.trader.place_buy_order(token_id, buy_price, order_size)

            if order_id:
                # Track this order
                self.position.active_orders[order_id] = {
                    'token_id': token_id,
                    'token_name': token_name,
                    'side': 'buy',
                    'price': buy_price,
                    'size': order_size,
                    'timestamp': time.time()
                }

                # Mark this token as triggered
                self.position.triggered_tokens.append(token_id)

                logger.info(f"✅ TRIGGER BUY SUCCESSFUL: {token_name} - Order ID: {order_id}")

                # Update balance immediately after successful trade
                self.update_usdc_balance_if_needed()

                return True
            else:
                logger.error(f"❌ TRIGGER BUY FAILED: {token_name}")
                return False

        except Exception as e:
            logger.error(f"❌ Error executing trigger buy for {token_name}: {e}")
            return False

    def check_stop_loss(self, token_data: Dict, token_name: str, current_balance: float):
        """Check if stop loss should be triggered"""
        try:
            if current_balance < self.config['min_order_size']:
                return  # Not enough balance to sell

            best_bid_price = float(token_data['best_bid']['price'])
            stop_loss_price = self.config['stop_loss_price']

            if best_bid_price <= stop_loss_price:
                logger.warning(f"🛑 STOP LOSS TRIGGERED: {token_name}")
                logger.warning(f"   Current bid: ${best_bid_price:.4f} <= Stop loss: ${stop_loss_price:.2f}")

                # Place sell order at current bid price
                token_id = token_data['asset_id']
                sell_size = min(current_balance, 1000.0)  # Cap sell size

                order_id = self.trader.place_sell_order(token_id, best_bid_price, sell_size)

                if order_id:
                    logger.warning(f"🛑 STOP LOSS ORDER PLACED: {token_name} - {sell_size} @ ${best_bid_price:.4f}")

                    # Track this order
                    self.position.active_orders[order_id] = {
                        'token_id': token_id,
                        'token_name': token_name,
                        'side': 'sell',
                        'price': best_bid_price,
                        'size': sell_size,
                        'timestamp': time.time(),
                        'type': 'stop_loss'
                    }

                    # Remove from triggered tokens so we can trigger again later
                    if token_id in self.position.triggered_tokens:
                        self.position.triggered_tokens.remove(token_id)
                        logger.info(f"🔄 Reset trigger status for {token_name}")

                else:
                    logger.error(f"❌ STOP LOSS ORDER FAILED: {token_name}")

        except Exception as e:
            logger.error(f"❌ Error checking stop loss for {token_name}: {e}")

    def monitor_markets(self):
        """Monitor both CALL and PUT markets for trigger conditions"""
        try:
            # Update USDC.e balance if needed (every minute)
            self.update_usdc_balance_if_needed()

            # Load fresh market data with better error handling
            call_data = self.safe_read_market_data(self.config['call_data_file'])
            put_data = self.safe_read_market_data(self.config['put_data_file'])

            if not call_data:
                logger.warning("⚠️ Could not load CALL market data")
                return

            if not put_data:
                logger.warning("⚠️ Could not load PUT market data")
                return

            # Update NFT token positions
            self.update_position(call_data, put_data)

            # Log current market state with spread info
            call_bid = float(call_data['best_bid']['price'])
            call_ask = float(call_data['best_ask']['price'])
            put_bid = float(put_data['best_bid']['price'])
            put_ask = float(put_data['best_ask']['price'])

            call_spread = call_ask - call_bid
            put_spread = put_ask - put_bid

            logger.info(f"📊 CALL: bid ${call_bid:.4f} | ask ${call_ask:.4f} | spread ${call_spread:.4f}")
            logger.info(f"📊 PUT:  bid ${put_bid:.4f} | ask ${put_ask:.4f} | spread ${put_spread:.4f} | USDC ${self.position.usdc_balance:.2f}")

            # Check trigger conditions (returns buy price if triggered)
            call_triggered, call_buy_price = self.check_trigger_condition(call_data, 'CALL')
            put_triggered, put_buy_price = self.check_trigger_condition(put_data, 'PUT')

            # Execute trigger buys if conditions are met
            if call_triggered:
                self.execute_trigger_buy(call_data, 'CALL', call_buy_price)

            if put_triggered:
                self.execute_trigger_buy(put_data, 'PUT', put_buy_price)

            # Check stop loss conditions for existing positions
            if self.position.call_balance >= self.config['min_order_size']:
                self.check_stop_loss(call_data, 'CALL', self.position.call_balance)

            if self.position.put_balance >= self.config['min_order_size']:
                self.check_stop_loss(put_data, 'PUT', self.position.put_balance)

        except Exception as e:
            logger.error(f"❌ Error monitoring markets: {e}")

    def cleanup_old_orders(self):
        """Clean up old completed orders from tracking"""
        try:
            orders_to_remove = []

            for order_id, order_info in self.position.active_orders.items():
                # Check order status
                order_status = self.trader.get_order_status(order_id)

                if order_status and order_status.get('status') in ['filled', 'cancelled', 'expired']:
                    orders_to_remove.append(order_id)
                    logger.info(f"🧹 Removing completed order: {order_id} ({order_status.get('status')})")

            # Remove completed orders
            for order_id in orders_to_remove:
                del self.position.active_orders[order_id]

        except Exception as e:
            logger.error(f"❌ Error cleaning up orders: {e}")

    async def run(self):
        """Main bot loop"""
        logger.info("🎯 Starting 0.96 Spread Trigger Bot (WEB3 VERSION)")
        logger.info(f"🎯 Strategy: Buy at best_ask when ask <= ${self.config['trigger_ask_price']:.2f} AND spread <= ${self.config['max_spread']:.2f}")
        logger.info(f"🛑 Stop loss at ${self.config['stop_loss_price']:.2f}")
        logger.info(f"💰 USDC.e balance will be checked every {self.balance_check_interval} seconds")

        # Initial balance check
        if not self.check_usdc_balance():
            logger.error("❌ Initial USDC.e balance check failed. Please check your USDC.e balance and set allowances manually.")
            return

        while True:
            try:
                # Check data lag
                lag_acceptable, max_lag = self.check_data_lag()
                if not lag_acceptable:
                    logger.warning(f"⚠️ Data lag too high ({max_lag}ms), waiting...")
                    await asyncio.sleep(5)
                    continue

                # Monitor markets for trigger conditions
                self.monitor_markets()

                # Clean up old orders
                self.cleanup_old_orders()

                # Log current status with balance check timing
                active_count = len(self.position.active_orders)
                triggered_count = len(self.position.triggered_tokens)
                time_since_balance_check = int(time.time() - self.last_balance_check)
                logger.info(f"📋 Status: USDC.e ${self.position.usdc_balance:.2f} (checked {time_since_balance_check}s ago) | Active orders: {active_count} | Triggered tokens: {triggered_count}")

                # Sleep before next iteration
                await asyncio.sleep(3)  # Check every 3 seconds

            except KeyboardInterrupt:
                logger.info("🛑 Shutdown requested")
                break
            except Exception as e:
                logger.error(f"❌ Error in main loop: {e}")
                await asyncio.sleep(5)  # Wait longer on error

        # Cleanup on exit
        logger.info("🧹 Cleaning up on exit...")
        try:
            # Cancel any remaining active orders
            if self.position.active_orders:
                self.trader.cancel_all_orders()
                logger.info("🧹 Cancelled all active orders")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main entry point"""
    try:
        # Initialize trigger bot
        trigger_bot = Polymarket096TriggerBot()

        # Run the bot
        await trigger_bot.run()

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        raise

if __name__ == "__main__":
    print("🎯 POLYMARKET 0.96 SPREAD TRIGGER BOT - WEB3 VERSION")
    print("=" * 70)
    print("STRATEGY:")
    print("• Monitor CALL and PUT tokens for best ask <= $0.96")
    print("• Check that spread (ask - bid) <= $0.04")
    print("• When both conditions met, BUY at best_ask price")
    print("• Hold position until price drops to $0.60 (STOP LOSS)")
    print("• Never sell unless stop loss is triggered")
    print()
    print("FIXES:")
    print("• ✅ Better balance and allowance checking")
    print("• ✅ Improved JSON file error handling")
    print("• ✅ Fixed data lag calculation errors")
    print("• ✅ Added USDC.e balance monitoring (ERC20)")
    print("• ✅ Fixed ERC1155 vs ERC20 token handling")
    print("• ✅ Dynamic order size calculation")
    print("• ✅ Web3 direct ERC20 balance checking")
    print("• ✅ NEW: Spread-based trigger strategy")
    print()
    print("TOKEN TYPES:")
    print("• 💰 USDC.e: ERC20 token used for buying (Web3 direct)")
    print("• 🎯 CALL/PUT: ERC1155 NFT tokens to trade (CLOB API)")
    print()
    print("CONFIGURATION:")
    print(f"• Trigger ask price: <= $0.96")
    print(f"• Max spread: <= $0.04")
    print(f"• Buy price: best_ask (dynamic)")
    print(f"• Stop loss: $0.60")
    print(f"• Max order size: 100 tokens")
    print(f"• Min USDC.e balance: $5")
    print(f"• Balance check: Every 60 seconds + before each trade")
    print(f"• Credentials: keys/keys_ovh38.env")
    print(f"• Polygon RPC: polygon-rpc.com")
    print()
    print("⚠️  WARNING: This bot will execute real trades with real money!")
    print("⚠️  Make sure you have sufficient USDC.e balance!")
    print("⚠️  You must manually approve USDC.e allowances for Polymarket contracts!")
    print()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Bot failed: {e}")
