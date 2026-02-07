#!/usr/bin/env python3
"""
Bitcoin Hourly Options Trading Bot

Trading Logic:
1. Read probabilities from option_probabilities.json
2. Place buy orders 0.07$ below probability price
3. Place sell orders 0.07$ above probability price
4. Suspend trading if spread > 0.05$
5. Switch to selling mode when holding tokens
6. Fixed order size: 5 shares
7. Refresh asset_id every hour
8. SAFETY: Suspend trading if any data file is older than 2 seconds
"""

import os
import json
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

# Web3 imports for ERC20 balance checking
from web3 import Web3
from eth_account import Account

# Import the core trading functions
from polymarket_trading_core import PolymarketTrader, load_credentials_from_env

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Track our positions and orders"""
    call_balance: float = 0.0
    put_balance: float = 0.0
    usdc_balance: float = 0.0
    active_orders: Dict[str, Dict] = None
    last_asset_refresh: float = 0.0  # Track when we last refreshed asset IDs

    def __post_init__(self):
        if self.active_orders is None:
            self.active_orders = {}

class BitcoinHourlyOptionsBot:
    """Bitcoin hourly options trading bot based on probabilities"""

    def __init__(self, config_path: str = None):
        """Initialize the options trading bot"""
        self.config = self._load_config(config_path)
        self.trader = self._initialize_trader()
        self.position = Position()

        # USDC.e contract details on Polygon
        self.usdc_contract_address = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
        self.usdc_decimals = 6

        # Web3 setup for Polygon
        self.polygon_rpc = "https://polygon-mainnet.core.chainstack.com/d308330baf4e49cfc74c637905858979"
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

        # Current asset IDs (will be refreshed hourly)
        self.current_call_asset_id = None
        self.current_put_asset_id = None

        # Asset refresh interval (1 hour = 3600 seconds)
        self.asset_refresh_interval = 3600

        # Hourly reset tracking
        self.last_hour_processed = -1  # Track which hour we last processed
        self.is_in_hourly_reset = False  # Flag to indicate we're in reset mode

        logger.info("Bitcoin Hourly Options Trading Bot initialized")
        logger.info(f"Trading logic: Buy at probability-0.07, Sell at probability+0.07")
        logger.info(f"Fixed order size: {self.config['order_size']} shares")
        logger.info(f"Max spread: {self.config['max_spread']}")
        logger.info(f"Wallet: {self.wallet_address}")
        logger.info(f"Hourly reset: 30-second pause at 0:00 to reload asset IDs")

    def _get_wallet_address(self) -> str:
        """Get wallet address from private key"""
        try:
            creds = load_credentials_from_env('/home/ubuntu/013_2025_polymarket/keys/keys_ovh39.env')
            private_key = creds['private_key']

            if private_key.startswith('0x'):
                private_key = private_key[2:]

            account = Account.from_key(private_key)
            return account.address
        except Exception as e:
            logger.error(f"Error getting wallet address: {e}")
            return "0x0000000000000000000000000000000000000000"

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        default_config = {
            'order_size': 5.0,              # Fixed order size in shares
            'price_offset': 0.07,
            'sell_offset':0.04,          # Offset from probability price
            'max_spread': 0.05,             # Maximum acceptable bid-ask spread
            'min_usdc_balance': 5.0,        # Minimum USDC.e balance required
            'polygon_rpc': 'https://polygon-mainnet.core.chainstack.com/d308330baf4e49cfc74c637905858979',
            'btc_data_file': '/home/ubuntu/013_2025_polymarket/btc_price.json',
            'call_data_file': '/home/ubuntu/013_2025_polymarket/CALL.json',
            'put_data_file': '/home/ubuntu/013_2025_polymarket/PUT.json',
            'probability_file': '/home/ubuntu/013_2025_polymarket/option_probabilities.json'
        }

        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)

        return default_config

    def _initialize_trader(self) -> PolymarketTrader:
        """Initialize the Polymarket trader"""
        try:
            creds = load_credentials_from_env('/home/ubuntu/013_2025_polymarket/keys/keys_ovh39.env')
            return PolymarketTrader(
                clob_api_url=creds['clob_api_url'],
                private_key=creds['private_key'],
                api_key=creds['api_key'],
                api_secret=creds['api_secret'],
                api_passphrase=creds['api_passphrase']
            )
        except Exception as e:
            logger.error(f"Failed to initialize trader: {e}")
            raise

    def safe_read_json_file(self, file_path: str) -> Optional[Dict]:
        """Safely read JSON file with error handling"""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                return None

            with open(file_path, 'r') as f:
                content = f.read().strip()

            if not content:
                logger.warning(f"Empty file: {file_path}")
                return None

            return json.loads(content)

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return None

    def get_usdc_balance(self) -> float:
        """Get USDC.e balance using Web3"""
        try:
            if not self.w3.is_connected():
                logger.error("Web3 not connected to Polygon network")
                return 0.0

            usdc_contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(self.usdc_contract_address),
                abi=self.erc20_abi
            )

            balance_raw = usdc_contract.functions.balanceOf(
                Web3.to_checksum_address(self.wallet_address)
            ).call()

            balance_usdc = balance_raw / (10 ** self.usdc_decimals)
            return balance_usdc

        except Exception as e:
            logger.error(f"Error getting USDC.e balance: {e}")
            return 0.0

    def update_usdc_balance_if_needed(self):
        """Update USDC.e balance if enough time has passed"""
        current_time = time.time()

        if current_time - self.last_balance_check >= self.balance_check_interval:
            try:
                usdc_balance = self.get_usdc_balance()
                self.position.usdc_balance = usdc_balance
                self.last_balance_check = current_time
                logger.info(f"USDC.e balance updated: ${usdc_balance:.2f}")
            except Exception as e:
                logger.error(f"Error updating USDC.e balance: {e}")

    def check_hourly_reset_needed(self) -> bool:
        """Check if we need to do hourly reset at 0:00"""
        now = datetime.now()
        current_hour = now.hour

        # Check if we're at the start of a new hour (0:00-0:30 seconds)
        is_start_of_hour = (now.minute == 0 and now.second < 30)

        # Check if this is a new hour we haven't processed
        is_new_hour = current_hour != self.last_hour_processed

        needs_reset = is_start_of_hour and is_new_hour and not self.is_in_hourly_reset

        if needs_reset:
            logger.info(f"HOURLY RESET DETECTED: {now.strftime('%H:%M:%S')} - Hour {current_hour}")

        return needs_reset

    async def execute_hourly_reset(self):
        """Execute 30-second hourly reset to handle asset ID changes"""
        try:
            now = datetime.now()
            self.is_in_hourly_reset = True
            self.last_hour_processed = now.hour

            logger.warning(f"🔄 STARTING HOURLY RESET at {now.strftime('%H:%M:%S')}")
            logger.warning("📛 CANCELLING ALL ORDERS - Asset IDs are changing")

            # Step 1: Cancel all active orders
            cancelled_count = self.trader.cancel_all_orders()
            self.position.active_orders.clear()
            logger.warning(f"✅ Cancelled {cancelled_count} orders")

            # Step 2: Wait 30 seconds for new contracts to be available
            logger.warning("⏳ Waiting 30 seconds for new hourly contracts...")
            for i in range(30, 0, -1):
                if i % 5 == 0:  # Log every 5 seconds
                    logger.warning(f"⏱️ Reset countdown: {i} seconds remaining")
                await asyncio.sleep(1)

            # Step 3: Force reload asset IDs from fresh JSON files
            logger.warning("📄 Reloading CALL.json and PUT.json for new asset IDs")

            call_data = self.safe_read_json_file(self.config['call_data_file'])
            put_data = self.safe_read_json_file(self.config['put_data_file'])

            if call_data and 'asset_id' in call_data:
                old_call_id = self.current_call_asset_id
                self.current_call_asset_id = call_data['asset_id']
                logger.warning(f"🔄 CALL asset ID updated")
                logger.warning(f"   Old: {old_call_id}")
                logger.warning(f"   New: {self.current_call_asset_id}")
            else:
                logger.error("❌ Failed to reload CALL asset ID")

            if put_data and 'asset_id' in put_data:
                old_put_id = self.current_put_asset_id
                self.current_put_asset_id = put_data['asset_id']
                logger.warning(f"🔄 PUT asset ID updated")
                logger.warning(f"   Old: {old_put_id}")
                logger.warning(f"   New: {self.current_put_asset_id}")
            else:
                logger.error("❌ Failed to reload PUT asset ID")

            # Step 4: Reset position tracking for new hour
            self.position.last_asset_refresh = time.time()

            # Step 5: Update token balances for new contracts
            self.update_token_balances()

            logger.warning("✅ HOURLY RESET COMPLETED - Resuming trading with new contracts")
            self.is_in_hourly_reset = False

        except Exception as e:
            logger.error(f"❌ Error during hourly reset: {e}")
            self.is_in_hourly_reset = False

    def refresh_asset_ids_if_needed(self):
        """Refresh asset IDs at the beginning of each hour"""
        current_time = time.time()

        # Check if we need to refresh (either first time or hourly)
        if (self.position.last_asset_refresh == 0 or
            current_time - self.position.last_asset_refresh >= self.asset_refresh_interval):

            logger.info("Refreshing asset IDs from market data files...")

            # Read CALL asset ID
            call_data = self.safe_read_json_file(self.config['call_data_file'])
            if call_data and 'asset_id' in call_data:
                self.current_call_asset_id = call_data['asset_id']
                logger.info(f"CALL asset ID: {self.current_call_asset_id}")
            else:
                logger.error("Failed to read CALL asset ID")
                return False

            # Read PUT asset ID
            put_data = self.safe_read_json_file(self.config['put_data_file'])
            if put_data and 'asset_id' in put_data:
                self.current_put_asset_id = put_data['asset_id']
                logger.info(f"PUT asset ID: {self.current_put_asset_id}")
            else:
                logger.error("Failed to read PUT asset ID")
                return False

            self.position.last_asset_refresh = current_time
            logger.info("Asset IDs refreshed successfully")
            return True

        return True  # No refresh needed

    def check_data_freshness(self, data: Dict, file_name: str, max_age_seconds: int = 2) -> bool:
        """Check if data is fresh enough for trading"""
        try:
            current_time_ms = int(time.time() * 1000)

            # Check for timestamp field (could be 'timestamp' or 'updated_at')
            data_timestamp = data.get('timestamp') or data.get('updated_at')

            if not data_timestamp:
                logger.error(f"No timestamp found in {file_name}")
                return False

            # Convert to milliseconds if needed
            data_timestamp = int(float(data_timestamp))
            if data_timestamp < 1e12:  # If timestamp is in seconds, convert to ms
                data_timestamp *= 1000

            age_ms = current_time_ms - data_timestamp
            age_seconds = age_ms / 1000

            if age_seconds > max_age_seconds:
                logger.warning(f"STALE DATA: {file_name} is {age_seconds:.1f}s old (max: {max_age_seconds}s)")
                return False

            logger.debug(f"Data freshness OK: {file_name} is {age_seconds:.1f}s old")
            return True

        except Exception as e:
            logger.error(f"Error checking data freshness for {file_name}: {e}")
            return False

    def cancel_all_active_orders(self):
        """Cancel all active orders and clear tracking"""
        try:
            if not self.position.active_orders:
                logger.debug("No active orders to cancel")
                return

            order_count = len(self.position.active_orders)
            logger.warning(f"SAFETY: Canceling all {order_count} active orders due to stale data")

            # Cancel all orders via trader
            cancelled_count = self.trader.cancel_all_orders()

            # Clear our local tracking
            self.position.active_orders.clear()

            logger.warning(f"Successfully cancelled {cancelled_count} orders")

        except Exception as e:
            logger.error(f"Error canceling orders: {e}")
            # Still clear local tracking even if API call failed
            self.position.active_orders.clear()

    def get_probability_data(self) -> Optional[Dict]:
        """Read probability data from option_probabilities.json"""
        prob_data = self.safe_read_json_file(self.config['probability_file'])

        if not prob_data:
            logger.error("Failed to read probability data")
            return None

        # Check data freshness
        if not self.check_data_freshness(prob_data, 'option_probabilities.json'):
            logger.warning("Probability data is stale, canceling all orders and suspending trading")
            cancelled_count = self.trader.cancel_all_orders()
            self.position.active_orders.clear()
            logger.warning(f"Cancelled {cancelled_count} orders due to stale probability data")
            return None

        required_fields = ['call_probability', 'put_probability']
        for field in required_fields:
            if field not in prob_data:
                logger.error(f"Missing field '{field}' in probability data")
                return None

        return prob_data

    def check_spread_condition(self, market_data: Dict) -> bool:
        """Check if bid-ask spread is acceptable"""
        try:
            best_bid = float(market_data['best_bid']['price'])
            best_ask = float(market_data['best_ask']['price'])
            spread = best_ask - best_bid

            if spread > self.config['max_spread']:
                logger.warning(f"Spread too wide: ${spread:.4f} > ${self.config['max_spread']}")
                return False

            return True
        except Exception as e:
            logger.error(f"Error checking spread: {e}")
            return False

    def calculate_trading_prices(self, prob_data: Dict) -> Dict:
        """Calculate buy and sell prices based on probabilities"""
        call_prob = float(prob_data['call_probability'])
        put_prob = float(prob_data['put_probability'])
        offset = self.config['price_offset']
        sell_offset = self.config['sell_offset']

        prices = {
            'call_buy_price': max(0.01, call_prob - offset),    # Buy below probability
            'call_sell_price': min(0.99, call_prob + sell_offset),   # Sell above probability
            'put_buy_price': max(0.01, put_prob - offset),      # Buy below probability
            'put_sell_price': min(0.99, put_prob + sell_offset),     # Sell above probability
            'call_probability': call_prob,
            'put_probability': put_prob
        }

        return prices

    def update_token_balances(self):
        """Update current token balances"""
        try:
            if self.current_call_asset_id:
                _, call_balance = self.trader.get_token_balance(self.current_call_asset_id)
                self.position.call_balance = call_balance

            if self.current_put_asset_id:
                _, put_balance = self.trader.get_token_balance(self.current_put_asset_id)
                self.position.put_balance = put_balance

            logger.info(f"Token balances - CALL: {self.position.call_balance:.1f}, PUT: {self.position.put_balance:.1f}")

        except Exception as e:
            logger.error(f"Error updating token balances: {e}")

    def should_buy_tokens(self) -> bool:
        """Check if we should be in buying mode (no tokens held)"""
        min_balance = 2.0  # Minimum balance to consider "holding" (increased for safety)
        has_call_tokens = self.position.call_balance >= min_balance
        has_put_tokens = self.position.put_balance >= min_balance

        logger.debug(f"Balance check: CALL {self.position.call_balance:.1f} >= {min_balance}? {has_call_tokens}")
        logger.debug(f"Balance check: PUT {self.position.put_balance:.1f} >= {min_balance}? {has_put_tokens}")

        return not (has_call_tokens or has_put_tokens)

    def execute_buy_order(self, token_name: str, asset_id: str, price: float) -> bool:
        """Execute a buy order"""
        try:
            order_size = self.config['order_size']
            total_cost = order_size * price

            # Check USDC balance - log but don't stop
            if self.position.usdc_balance < total_cost:
                logger.info(f"Insufficient USDC for {token_name} buy: ${self.position.usdc_balance:.2f} < ${total_cost:.2f} - continuing to monitor")
                return False

            logger.info(f"Placing BUY order: {token_name} - {order_size} @ ${price:.4f}")

            order_id = self.trader.place_buy_order(asset_id, price, order_size)

            if order_id:
                self.position.active_orders[order_id] = {
                    'token_name': token_name,
                    'asset_id': asset_id,
                    'side': 'buy',
                    'price': price,
                    'size': order_size,
                    'timestamp': time.time()
                }
                logger.info(f"BUY order placed successfully: {order_id}")
                return True
            else:
                logger.warning(f"BUY order failed for {token_name}")
                return False

        except Exception as e:
            logger.error(f"Error executing buy order for {token_name}: {e}")
            return False

    def validate_sell_balance(self, token_name: str, asset_id: str, current_balance: float) -> Tuple[bool, float]:
        """Validate and get actual sellable balance"""
        try:
            # Get fresh balance from API
            balance_raw, balance_tokens = self.trader.get_token_balance(asset_id)

            # Use the minimum of reported balance and current balance for safety
            actual_balance = min(balance_tokens, current_balance)

            # Use configured order size (5.0) as the target sell amount
            target_sell_size = self.config['order_size']  # 5.0

            # Check if we have enough to sell the target amount
            if actual_balance < target_sell_size:
                logger.debug(f"Insufficient {token_name} to sell target amount: {actual_balance:.2f} < {target_sell_size}")
                return False, 0.0

            # Sell exactly the target amount (5.0) - no buffer needed
            sellable_amount = target_sell_size

            logger.info(f"Validated {token_name} balance: reported={current_balance:.2f}, actual={balance_tokens:.2f}, selling={sellable_amount:.2f}")
            return True, sellable_amount

        except Exception as e:
            logger.error(f"Error validating sell balance for {token_name}: {e}")
            return False, 0.0

    def execute_sell_order(self, token_name: str, asset_id: str, price: float, balance: float) -> bool:
        """Execute a sell order with proper balance validation"""
        try:
            # First validate we actually have tokens to sell
            can_sell, sellable_amount = self.validate_sell_balance(token_name, asset_id, balance)

            if not can_sell:
                logger.debug(f"Skipping {token_name} sell - insufficient balance after validation")
                return False

            # Use the validated sellable amount, but cap at configured order size
            sell_size = min(sellable_amount, self.config['order_size'])

            logger.info(f"Placing SELL order: {token_name} - {sell_size:.2f} @ ${price:.4f} (from balance: {balance:.2f})")

            order_id = self.trader.place_sell_order(asset_id, price, sell_size)

            if order_id:
                self.position.active_orders[order_id] = {
                    'token_name': token_name,
                    'asset_id': asset_id,
                    'side': 'sell',
                    'price': price,
                    'size': sell_size,
                    'timestamp': time.time()
                }
                logger.info(f"SELL order placed successfully: {order_id}")
                return True
            else:
                logger.warning(f"SELL order failed for {token_name}")
                return False

        except Exception as e:
            logger.error(f"Error executing sell order for {token_name}: {e}")
            return False

    def cancel_old_orders(self):
        """Cancel orders older than 5 minutes using reliable method"""
        current_time = time.time()
        orders_to_cancel = []

        for order_id, order_info in self.position.active_orders.items():
            age = current_time - order_info['timestamp']
            if age > 300:  # 5 minutes
                orders_to_cancel.append(order_id)

        if orders_to_cancel:
            logger.info(f"Cancelling {len(orders_to_cancel)} old orders")
            # Use reliable cancel all method
            cancelled_count = self.trader.cancel_all_orders()
            self.position.active_orders.clear()
            logger.info(f"Successfully cancelled {cancelled_count} orders")
        else:
            logger.debug("No old orders to cancel")

    def execute_trading_strategy(self):
        """Execute the main trading strategy"""
        try:
            # Refresh asset IDs if needed (hourly)
            if not self.refresh_asset_ids_if_needed():
                logger.error("Failed to refresh asset IDs, skipping trading cycle")
                return

            # Update balances
            self.update_usdc_balance_if_needed()
            self.update_token_balances()

            # Get probability data
            prob_data = self.get_probability_data()
            if not prob_data:
                logger.error("No probability data available, canceling all orders")
                cancelled_count = self.trader.cancel_all_orders()
                self.position.active_orders.clear()
                logger.warning(f"Cancelled {cancelled_count} orders due to missing probability data")
                return

            # Calculate trading prices
            prices = self.calculate_trading_prices(prob_data)

            # Get market data
            call_data = self.safe_read_json_file(self.config['call_data_file'])
            put_data = self.safe_read_json_file(self.config['put_data_file'])

            if not call_data or not put_data:
                logger.error("Failed to read market data, canceling all orders")
                cancelled_count = self.trader.cancel_all_orders()
                self.position.active_orders.clear()
                logger.warning(f"Cancelled {cancelled_count} orders due to missing market data")
                return

            # Check data freshness for market data
            if not self.check_data_freshness(call_data, 'CALL.json'):
                logger.warning("CALL market data is stale, canceling all orders and suspending trading")
                cancelled_count = self.trader.cancel_all_orders()
                self.position.active_orders.clear()
                logger.warning(f"Cancelled {cancelled_count} orders due to stale CALL data")
                return

            if not self.check_data_freshness(put_data, 'PUT.json'):
                logger.warning("PUT market data is stale, canceling all orders and suspending trading")
                cancelled_count = self.trader.cancel_all_orders()
                self.position.active_orders.clear()
                logger.warning(f"Cancelled {cancelled_count} orders due to stale PUT data")
                return

            # Check spread conditions
            call_spread_ok = self.check_spread_condition(call_data)
            put_spread_ok = self.check_spread_condition(put_data)

            if not call_spread_ok or not put_spread_ok:
                logger.warning("Spread conditions not met, suspending trading")
                return

            # Log current state
            logger.info(f"Probabilities - CALL: {prices['call_probability']:.1%}, PUT: {prices['put_probability']:.1%}")
            logger.info(f"Balances - CALL: {self.position.call_balance:.1f}, PUT: {self.position.put_balance:.1f}")

            # Determine trading mode
            if self.should_buy_tokens():
                # Buying mode - place buy orders for both tokens
                logger.info("BUYING MODE: Placing buy orders")

                # CRITICAL: Cancel all existing orders before placing new ones
                cancelled_count = self.trader.cancel_all_orders()
                self.position.active_orders.clear()
                if cancelled_count > 0:
                    logger.info(f"Cancelled {cancelled_count} existing orders before placing new buy orders")

                self.execute_buy_order('CALL', self.current_call_asset_id, prices['call_buy_price'])
                self.execute_buy_order('PUT', self.current_put_asset_id, prices['put_buy_price'])

            else:
                # Selling mode - place sell orders for held tokens
                logger.info("SELLING MODE: Placing sell orders")

                # CRITICAL: Cancel all existing orders before placing new ones
                cancelled_count = self.trader.cancel_all_orders()
                self.position.active_orders.clear()
                if cancelled_count > 0:
                    logger.info(f"Cancelled {cancelled_count} existing orders before placing new sell orders")

                if self.position.call_balance >= 2.0:
                    self.execute_sell_order('CALL', self.current_call_asset_id,
                                          prices['call_sell_price'], self.position.call_balance)

                if self.position.put_balance >= 2.0:
                    self.execute_sell_order('PUT', self.current_put_asset_id,
                                          prices['put_sell_price'], self.position.put_balance)

            # Note: No need for cancel_old_orders() since we cancel all orders before each cycle

        except Exception as e:
            logger.error(f"Error in trading strategy: {e}")

    async def run(self):
        """Main bot loop"""
        logger.info("Starting Bitcoin Hourly Options Trading Bot")
        logger.info(f"Order size: {self.config['order_size']} shares")
        logger.info(f"Price offset: ${self.config['price_offset']}")
        logger.info(f"Max spread: ${self.config['max_spread']}")

        # Initial setup
        if not self.refresh_asset_ids_if_needed():
            logger.error("Failed initial asset ID refresh")
            return

        # Check initial balance but don't exit - just log
        initial_balance = self.get_usdc_balance()
        if initial_balance < self.config['min_usdc_balance']:
            logger.warning(f"Low USDC balance: ${initial_balance:.2f} - Bot will continue monitoring")
        else:
            logger.info(f"Initial USDC balance: ${initial_balance:.2f}")

        self.position.usdc_balance = initial_balance

        while True:
            try:
                # CRITICAL: Check if we need hourly reset at 0:00
                if self.check_hourly_reset_needed():
                    await self.execute_hourly_reset()
                    continue  # Skip this cycle and start fresh

                # Skip trading if we're in reset mode
                if self.is_in_hourly_reset:
                    await asyncio.sleep(1)
                    continue

                # Execute trading strategy
                self.execute_trading_strategy()

                # Log status
                active_orders = len(self.position.active_orders)
                now = datetime.now()
                logger.info(f"Status [{now.strftime('%H:%M:%S')}]: USDC ${self.position.usdc_balance:.2f} | Active orders: {active_orders}")

                # Wait before next cycle
                await asyncio.sleep(0.3)  # Check every 0.3 seconds for real-time trading

            except KeyboardInterrupt:
                logger.info("Shutdown requested")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(3)  # Wait 3 seconds on error

        # Cleanup
        logger.info("Cleaning up...")
        try:
            cancelled_count = self.trader.cancel_all_orders()
            self.position.active_orders.clear()
            logger.info(f"Cancelled {cancelled_count} orders on shutdown")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main entry point"""
    try:
        bot = BitcoinHourlyOptionsBot()
        await bot.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    print("BITCOIN HOURLY OPTIONS TRADING BOT")
    print("=" * 50)
    print("TRADING STRATEGY:")
    print("• Read probabilities from option_probabilities.json")
    print("• Buy orders: probability - 0.07$")
    print("• Sell orders: probability + 0.07$")
    print("• Fixed order size: 5 shares")
    print("• Suspend trading if spread > 0.05$")
    print("• Switch to selling mode when holding tokens")
    print("• HOURLY RESET: 30-second pause at 0:00 to reload asset IDs")
    print()
    print("CONFIGURATION:")
    print(f"• Order size: 5 shares")
    print(f"• Price offset: $0.07")
    print(f"• Max spread: $0.05")
    print(f"• Min USDC balance: $5")
    print(f"• Trading cycle: Every 0.3 seconds")
    print(f"• Hourly reset: Automatic at 0:00")
    print()
    print("CRITICAL - HOURLY RESET:")
    print("• At 0:00 each hour, bot will:")
    print("  1. Cancel ALL orders")
    print("  2. Wait 30 seconds")
    print("  3. Reload CALL.json and PUT.json")
    print("  4. Get new asset IDs for new contracts")
    print("  5. Resume trading")
    print("• This prevents trading expired contracts!")
    print()
    print("WARNING: This bot executes real trades with real money!")
    print()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"\nBot failed: {e}")
