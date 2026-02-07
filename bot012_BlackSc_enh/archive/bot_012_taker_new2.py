#!/usr/bin/env python3
"""
Bitcoin Hourly Options Market Trading Bot

Trading Logic:
1. Read probabilities from option_probabilities.json
2. BUY: When market ask + offset < probability (for >10 seconds) -> execute market buy
3. SELL: When market bid - offset > probability (for >10 seconds) AND we own the asset -> execute market sell
4. 10 second wait prevents trading on sudden price spikes
5. Fixed order size: 5 shares maximum
6. Initial offset: 0.10$
7. Refresh asset_id every hour
8. SAFETY: Suspend trading if any data file is older than 2 seconds
"""

import os
import json
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
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
class PriceCondition:
    """Track price conditions over time"""
    condition_met: bool = False
    first_met_time: float = 0.0
    condition_duration: float = 0.0

    def update(self, condition_met: bool, current_time: float):
        """Update condition state"""
        if condition_met and not self.condition_met:
            # Condition just became true
            self.condition_met = True
            self.first_met_time = current_time
            self.condition_duration = 0.0
        elif condition_met and self.condition_met:
            # Condition continues to be true
            self.condition_duration = current_time - self.first_met_time
        elif not condition_met:
            # Condition is false
            self.condition_met = False
            self.first_met_time = 0.0
            self.condition_duration = 0.0

@dataclass
class Position:
    """Track our positions and purchase history"""
    call_balance: float = 0.0
    put_balance: float = 0.0
    usdc_balance: float = 0.0

    # Track which assets we've purchased to enable selling
    call_purchased: bool = False
    put_purchased: bool = False

    # Track pending orders to prevent over-buying
    call_pending_buys: float = 0.0  # Total size of pending buy orders
    put_pending_buys: float = 0.0   # Total size of pending buy orders

    # Track price conditions over time
    call_buy_condition: PriceCondition = field(default_factory=PriceCondition)
    call_sell_condition: PriceCondition = field(default_factory=PriceCondition)
    put_buy_condition: PriceCondition = field(default_factory=PriceCondition)
    put_sell_condition: PriceCondition = field(default_factory=PriceCondition)

    last_asset_refresh: float = 0.0

class BitcoinHourlyMarketBot:
    """Bitcoin hourly options market trading bot based on probabilities"""

    def __init__(self, config_path: str = None):
        """Initialize the market trading bot"""
        self.config = self._load_config(config_path)
        self.trader = self._initialize_trader()
        self.position = Position()

        # API call tracking
        self.api_call_stats = {
            'total_calls': 0,
            'by_function': {},
            'start_time': time.time(),
            'last_stats_log': time.time()
        }

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

        # Token balance checking timing
        self.last_token_balance_check = 0
        self.token_balance_check_interval = 60  # Check token balances every 60 seconds

        # Order cleanup timing
        self.last_order_cleanup = 0
        self.order_cleanup_interval = 60  # Check and cancel pending orders every 60 seconds

        # Current asset IDs (will be refreshed hourly)
        self.current_call_asset_id = None
        self.current_put_asset_id = None

        # Asset refresh interval (1 hour = 3600 seconds)
        self.asset_refresh_interval = 3600

        # Hourly reset tracking
        self.last_hour_processed = -1  # Track which hour we last processed
        self.is_in_hourly_reset = False  # Flag to indicate we're in reset mode

        # Time condition threshold
        self.condition_threshold_seconds = 10.0  # 10 seconds

        logger.info("Bitcoin Hourly Options Market Trading Bot initialized")
        logger.info(f"Market order logic: Buy when ask+offset < probability for >10s")
        logger.info(f"                   Sell when bid-offset > probability for >10s (if owned)")
        logger.info(f"Fixed order size: {self.config['order_size']} shares")
        logger.info(f"Max position size: {self.config['max_position_size']} shares per asset")
        logger.info(f"Price offset: ${self.config['price_offset']}")
        logger.info(f"Condition threshold: {self.condition_threshold_seconds}s")
        logger.info(f"Token balance updates: Every {self.token_balance_check_interval}s (+ before trades)")
        logger.info(f"Order cleanup: Every {self.order_cleanup_interval}s (cancel pending orders)")
        logger.info(f"Wallet: {self.wallet_address}")
        logger.info(f"API call tracking: ENABLED")

    def track_api_call(self, function_name: str):
        """Track an API call for rate limit monitoring"""
        self.api_call_stats['total_calls'] += 1

        if function_name not in self.api_call_stats['by_function']:
            self.api_call_stats['by_function'][function_name] = 0

        self.api_call_stats['by_function'][function_name] += 1

    def log_api_stats_if_needed(self):
        """Log API statistics periodically"""
        current_time = time.time()

        # Log stats every 5 minutes
        if current_time - self.api_call_stats['last_stats_log'] >= 300:
            self.log_api_statistics()
            self.api_call_stats['last_stats_log'] = current_time

    def log_api_statistics(self):
        """Log detailed API call statistics"""
        current_time = time.time()
        runtime_hours = (current_time - self.api_call_stats['start_time']) / 3600
        total_calls = self.api_call_stats['total_calls']
        calls_per_hour = total_calls / runtime_hours if runtime_hours > 0 else 0

        logger.info("=" * 60)
        logger.info(f"API CALL STATISTICS (Runtime: {runtime_hours:.1f}h)")
        logger.info(f"Total API calls: {total_calls}")
        logger.info(f"Calls per hour: {calls_per_hour:.1f}")
        logger.info("")
        logger.info("Calls by function:")

        # Sort by call count (descending)
        sorted_functions = sorted(
            self.api_call_stats['by_function'].items(),
            key=lambda x: x[1],
            reverse=True
        )

        for func_name, count in sorted_functions:
            percentage = (count / total_calls * 100) if total_calls > 0 else 0
            logger.info(f"  {func_name:<25} {count:>6} calls ({percentage:>5.1f}%)")

        logger.info("=" * 60)

    def _get_wallet_address(self) -> str:
        """Get wallet address from private key"""
        try:
            creds = load_credentials_from_env('/home/ubuntu/013_2025_polymarket/keys/keys_ovh38.env')
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
            'max_position_size': 5.0,       # Maximum total shares per asset
            'price_offset': 0.05,           # Offset for market order conditions
            'max_spread': 0.06,             # Maximum acceptable bid-ask spread
            'min_usdc_balance': 4.0,        # Minimum USDC.e balance required
            'condition_threshold': 5.0,    # Seconds to wait before executing
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
            creds = load_credentials_from_env('/home/ubuntu/013_2025_polymarket/keys/keys_ovh38.env')
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

            logger.warning(f"STARTING HOURLY RESET at {now.strftime('%H:%M:%S')}")
            logger.warning("CANCELLING ALL ORDERS - Asset IDs are changing")

            # Step 1: Cancel all active orders
            self.track_api_call('cancel_all_orders_hourly_reset')
            cancelled_count = self.trader.cancel_all_orders()
            logger.warning(f"Cancelled {cancelled_count} orders")

            # Step 2: Reset position tracking for new contracts
            self.position.call_purchased = False
            self.position.put_purchased = False

            # Reset pending buy tracking
            self.position.call_pending_buys = 0.0
            self.position.put_pending_buys = 0.0

            # Reset all price conditions
            self.position.call_buy_condition = PriceCondition()
            self.position.call_sell_condition = PriceCondition()
            self.position.put_buy_condition = PriceCondition()
            self.position.put_sell_condition = PriceCondition()

            # Step 3: Wait 30 seconds for new contracts to be available
            logger.warning("Waiting 30 seconds for new hourly contracts...")
            for i in range(30, 0, -1):
                if i % 5 == 0:  # Log every 5 seconds
                    logger.warning(f"Reset countdown: {i} seconds remaining")
                await asyncio.sleep(1)

            # Step 4: Force reload asset IDs from fresh JSON files
            logger.warning("Reloading CALL.json and PUT.json for new asset IDs")

            call_data = self.safe_read_json_file(self.config['call_data_file'])
            put_data = self.safe_read_json_file(self.config['put_data_file'])

            if call_data and 'asset_id' in call_data:
                old_call_id = self.current_call_asset_id
                self.current_call_asset_id = call_data['asset_id']
                logger.warning(f"CALL asset ID updated")
                logger.warning(f"   Old: {old_call_id}")
                logger.warning(f"   New: {self.current_call_asset_id}")
            else:
                logger.error("Failed to reload CALL asset ID")

            if put_data and 'asset_id' in put_data:
                old_put_id = self.current_put_asset_id
                self.current_put_asset_id = put_data['asset_id']
                logger.warning(f"PUT asset ID updated")
                logger.warning(f"   Old: {old_put_id}")
                logger.warning(f"   New: {self.current_put_asset_id}")
            else:
                logger.error("Failed to reload PUT asset ID")

            # Step 5: Reset position tracking for new hour
            self.position.last_asset_refresh = time.time()

            # Step 6: Update token balances for new contracts (forced after reset)
            self.update_token_balances()

            logger.warning("HOURLY RESET COMPLETED - Resuming trading with new contracts")
            self.is_in_hourly_reset = False

        except Exception as e:
            logger.error(f"Error during hourly reset: {e}")
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

    def get_probability_data(self) -> Optional[Dict]:
        """Read probability data from option_probabilities.json"""
        prob_data = self.safe_read_json_file(self.config['probability_file'])

        if not prob_data:
            logger.error("Failed to read probability data")
            return None

        # Check data freshness
        if not self.check_data_freshness(prob_data, 'option_probabilities.json'):
            logger.warning("Probability data is stale, suspending trading")
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

    def update_token_balances_if_needed(self):
        """Update token balances if enough time has passed"""
        current_time = time.time()

        if current_time - self.last_token_balance_check >= self.token_balance_check_interval:
            try:
                if self.current_call_asset_id:
                    self.track_api_call('get_token_balance_call_interval')
                    _, call_balance = self.trader.get_token_balance(self.current_call_asset_id)
                    self.position.call_balance = call_balance

                if self.current_put_asset_id:
                    self.track_api_call('get_token_balance_put_interval')
                    _, put_balance = self.trader.get_token_balance(self.current_put_asset_id)
                    self.position.put_balance = put_balance

                self.last_token_balance_check = current_time
                logger.debug(f"Token balances updated: CALL {self.position.call_balance:.1f}, PUT {self.position.put_balance:.1f}")

            except Exception as e:
                logger.error(f"Error updating token balances: {e}")

    def cleanup_pending_orders_if_needed(self):
        """Cancel any pending orders if enough time has passed"""
        current_time = time.time()

        if current_time - self.last_order_cleanup >= self.order_cleanup_interval:
            try:
                # Cancel all pending orders (market strategy requires immediate execution only)
                self.track_api_call('cancel_all_orders_cleanup')
                cancelled_count = self.trader.cancel_all_orders()

                if cancelled_count > 0:
                    logger.warning(f"Periodic cleanup: Cancelled {cancelled_count} pending orders")
                else:
                    logger.debug("Periodic cleanup: No orders to cancel")

                self.last_order_cleanup = current_time

            except Exception as e:
                logger.error(f"Error during order cleanup: {e}")
                # Still update the last cleanup time to avoid continuous failures
                self.last_order_cleanup = current_time

    def update_token_balances(self):
        """Force update current token balances (used before trading decisions)"""
        try:
            if self.current_call_asset_id:
                self.track_api_call('get_token_balance_call_forced')
                _, call_balance = self.trader.get_token_balance(self.current_call_asset_id)
                self.position.call_balance = call_balance

            if self.current_put_asset_id:
                self.track_api_call('get_token_balance_put_forced')
                _, put_balance = self.trader.get_token_balance(self.current_put_asset_id)
                self.position.put_balance = put_balance

            # Update the last check time since we just did a fresh check
            self.last_token_balance_check = time.time()
            logger.debug(f"Token balances (forced): CALL {self.position.call_balance:.1f}, PUT {self.position.put_balance:.1f}")

        except Exception as e:
            logger.error(f"Error updating token balances: {e}")

    def can_buy_more(self, token_name: str, current_balance: float) -> Tuple[bool, float]:
        """
        Check if we can buy more of an asset without exceeding the maximum position size

        Args:
            token_name: Name of the token ('CALL' or 'PUT')
            current_balance: Current balance of the token

        Returns:
            Tuple of (can_buy_more, available_size_to_buy)
        """
        max_position = self.config['max_position_size']
        order_size = self.config['order_size']

        # Get pending buy orders for this token
        if token_name == 'CALL':
            pending_buys = self.position.call_pending_buys
        elif token_name == 'PUT':
            pending_buys = self.position.put_pending_buys
        else:
            pending_buys = 0.0

        # Calculate total exposure (current balance + pending orders)
        total_exposure = current_balance + pending_buys

        # Calculate how much we can still buy
        available_to_buy = max_position - total_exposure

        if total_exposure >= max_position:
            logger.info(f"❌ Cannot buy more {token_name}: at max position {current_balance:.1f} + pending {pending_buys:.1f} = {total_exposure:.1f}/{max_position}")
            return False, 0.0

        if available_to_buy < order_size:
            logger.info(f"❌ Cannot buy full order of {token_name}: only {available_to_buy:.1f} shares available (current: {current_balance:.1f}, pending: {pending_buys:.1f}, max: {max_position})")
            return False, 0.0

        # Can buy the full order size
        logger.debug(f"✅ Can buy {token_name}: current {current_balance:.1f} + pending {pending_buys:.1f} + order {order_size} = {total_exposure + order_size:.1f}/{max_position}")
        return True, min(order_size, available_to_buy)

    def execute_market_buy_order(self, token_name: str, asset_id: str, market_ask_price: float, current_balance: float) -> bool:
        """Execute a market buy order at current ask price"""
        try:
            # CRITICAL: Fresh balance check right before buying
            if token_name == 'CALL':
                self.track_api_call('get_token_balance_fresh_call')
                _, fresh_balance = self.trader.get_token_balance(asset_id)
                self.position.call_balance = fresh_balance
            elif token_name == 'PUT':
                self.track_api_call('get_token_balance_fresh_put')
                _, fresh_balance = self.trader.get_token_balance(asset_id)
                self.position.put_balance = fresh_balance
            else:
                fresh_balance = current_balance

            # Check if we can buy more with fresh balance
            can_buy, buy_size = self.can_buy_more(token_name, fresh_balance)

            if not can_buy:
                logger.warning(f"BLOCKED: Cannot buy more {token_name} - at position limit")
                return False

            # Double check with hard limit
            if fresh_balance >= self.config['max_position_size']:
                logger.error(f"HARD LIMIT: {token_name} balance {fresh_balance:.1f} >= max {self.config['max_position_size']}")
                return False

            total_cost = buy_size * market_ask_price

            # Check USDC balance
            if self.position.usdc_balance < total_cost:
                logger.info(f"Insufficient USDC for {token_name} market buy: ${self.position.usdc_balance:.2f} < ${total_cost:.2f}")
                return False

            # Track pending order BEFORE placing it
            if token_name == 'CALL':
                self.position.call_pending_buys += buy_size
            elif token_name == 'PUT':
                self.position.put_pending_buys += buy_size

            logger.warning(f"EXECUTING MARKET BUY: {token_name} - {buy_size} shares @ ${market_ask_price:.4f}")
            logger.warning(f"  Current position: {fresh_balance:.1f} -> {fresh_balance + buy_size:.1f} shares (max: {self.config['max_position_size']})")

            # Execute market buy by buying at the ask price
            self.track_api_call(f'place_buy_order_{token_name.lower()}')
            order_id = self.trader.place_buy_order(asset_id, market_ask_price, buy_size)

            if order_id:
                # Give order a moment to potentially execute immediately
                import time
                time.sleep(0.5)  # Wait 0.5 seconds

                # Check if the order executed immediately
                self.track_api_call(f'get_order_status_{token_name.lower()}_buy')
                order_status = self.trader.get_order_status(order_id)

                if order_status and order_status.get('status') not in ['FILLED', 'COMPLETED']:
                    # Order didn't execute immediately - cancel it
                    logger.warning(f"Order {order_id} not executed immediately (status: {order_status.get('status', 'unknown')}) - cancelling for market strategy")
                    self.track_api_call(f'cancel_order_{token_name.lower()}_buy')
                    cancel_success = self.trader.cancel_order(order_id)

                    if cancel_success:
                        logger.warning(f"Successfully cancelled non-executing buy order: {order_id}")

                    # Clear pending and don't mark as purchased
                    if token_name == 'CALL':
                        self.position.call_pending_buys = max(0, self.position.call_pending_buys - buy_size)
                    elif token_name == 'PUT':
                        self.position.put_pending_buys = max(0, self.position.put_pending_buys - buy_size)

                    return False
                else:
                    # Order executed successfully
                    # Mark this asset as purchased
                    if token_name == 'CALL':
                        self.position.call_purchased = True
                        # Clear pending after successful order
                        self.position.call_pending_buys = max(0, self.position.call_pending_buys - buy_size)
                    elif token_name == 'PUT':
                        self.position.put_purchased = True
                        # Clear pending after successful order
                        self.position.put_pending_buys = max(0, self.position.put_pending_buys - buy_size)

                    logger.warning(f"MARKET BUY EXECUTED: {order_id} - {token_name} purchased")

                    # Note: Token balances will be updated on next interval or before next trade

                    return True
            else:
                # Clear pending if order failed
                if token_name == 'CALL':
                    self.position.call_pending_buys = max(0, self.position.call_pending_buys - buy_size)
                elif token_name == 'PUT':
                    self.position.put_pending_buys = max(0, self.position.put_pending_buys - buy_size)

                logger.error(f"Market buy FAILED for {token_name}")
                return False

        except Exception as e:
            # Clear pending if exception occurred
            if token_name == 'CALL':
                self.position.call_pending_buys = max(0, self.position.call_pending_buys - (buy_size if 'buy_size' in locals() else 0))
            elif token_name == 'PUT':
                self.position.put_pending_buys = max(0, self.position.put_pending_buys - (buy_size if 'buy_size' in locals() else 0))

            logger.error(f"Error executing market buy for {token_name}: {e}")
            return False

    def execute_market_sell_order(self, token_name: str, asset_id: str, market_bid_price: float, balance: float) -> bool:
        """Execute a market sell order at current bid price"""
        try:
            # Validate we have tokens to sell
            if balance < self.config['order_size']:
                logger.debug(f"Insufficient {token_name} balance to sell: {balance:.2f} < {self.config['order_size']}")
                return False

            sell_size = min(balance, self.config['order_size'])

            logger.info(f"EXECUTING MARKET SELL: {token_name} - {sell_size:.2f} shares @ ${market_bid_price:.4f}")

            # Execute market sell by selling at the bid price
            self.track_api_call(f'place_sell_order_{token_name.lower()}')
            order_id = self.trader.place_sell_order(asset_id, market_bid_price, sell_size)

            if order_id:
                # Give order a moment to potentially execute immediately
                import time
                time.sleep(0.5)  # Wait 0.5 seconds

                # Check if the order executed immediately
                self.track_api_call(f'get_order_status_{token_name.lower()}_sell')
                order_status = self.trader.get_order_status(order_id)

                if order_status and order_status.get('status') not in ['FILLED', 'COMPLETED']:
                    # Order didn't execute immediately - cancel it
                    logger.warning(f"Sell order {order_id} not executed immediately (status: {order_status.get('status', 'unknown')}) - cancelling for market strategy")
                    self.track_api_call(f'cancel_order_{token_name.lower()}_sell')
                    cancel_success = self.trader.cancel_order(order_id)

                    if cancel_success:
                        logger.warning(f"Successfully cancelled non-executing sell order: {order_id}")

                    return False
                else:
                    # Order executed successfully
                    # Update purchase tracking if we sold everything
                    if sell_size >= balance * 0.9:  # If we sold most of our holdings
                        if token_name == 'CALL':
                            self.position.call_purchased = False
                        elif token_name == 'PUT':
                            self.position.put_purchased = False

                    logger.info(f"MARKET SELL EXECUTED: {order_id} - {token_name} sold")

                    # Note: Token balances will be updated on next interval or before next trade

                    return True
            else:
                logger.warning(f"Market sell failed for {token_name}")
                return False

        except Exception as e:
            logger.error(f"Error executing market sell for {token_name}: {e}")
            return False

    def evaluate_market_conditions(self, prob_data: Dict, call_data: Dict, put_data: Dict):
        """Evaluate market conditions and update price condition tracking"""
        current_time = time.time()
        offset = self.config['price_offset']

        # Get market prices
        call_ask = float(call_data['best_ask']['price'])
        call_bid = float(call_data['best_bid']['price'])
        put_ask = float(put_data['best_ask']['price'])
        put_bid = float(put_data['best_bid']['price'])

        # Get probabilities
        call_prob = float(prob_data['call_probability'])
        put_prob = float(prob_data['put_probability'])

        # CALL BUY CONDITION: ask + offset < probability AND not at position limit
        call_can_buy, _ = self.can_buy_more('CALL', self.position.call_balance)
        call_buy_condition_met = (call_ask + offset) < call_prob and call_can_buy
        self.position.call_buy_condition.update(call_buy_condition_met, current_time)

        # CALL SELL CONDITION: bid - offset > probability AND we own the asset
        call_sell_condition_met = (call_bid - offset) > call_prob and self.position.call_purchased
        self.position.call_sell_condition.update(call_sell_condition_met, current_time)

        # PUT BUY CONDITION: ask + offset < probability AND not at position limit
        put_can_buy, _ = self.can_buy_more('PUT', self.position.put_balance)
        put_buy_condition_met = (put_ask + offset) < put_prob and put_can_buy
        self.position.put_buy_condition.update(put_buy_condition_met, current_time)

        # PUT SELL CONDITION: bid - offset > probability AND we own the asset
        put_sell_condition_met = (put_bid - offset) > put_prob and self.position.put_purchased
        self.position.put_sell_condition.update(put_sell_condition_met, current_time)

        # Log conditions for debugging
        logger.debug(f"CALL: ask=${call_ask:.4f}+${offset:.2f}={call_ask+offset:.4f} < prob=${call_prob:.4f}? {(call_ask + offset) < call_prob} | Can buy more? {call_can_buy} | Combined: {call_buy_condition_met} ({self.position.call_buy_condition.condition_duration:.1f}s)")
        logger.debug(f"CALL: bid=${call_bid:.4f}-${offset:.2f}={call_bid-offset:.4f} > prob=${call_prob:.4f}? {call_sell_condition_met} ({self.position.call_sell_condition.condition_duration:.1f}s)")
        logger.debug(f"PUT:  ask=${put_ask:.4f}+${offset:.2f}={put_ask+offset:.4f} < prob=${put_prob:.4f}? {(put_ask + offset) < put_prob} | Can buy more? {put_can_buy} | Combined: {put_buy_condition_met} ({self.position.put_buy_condition.condition_duration:.1f}s)")
        logger.debug(f"PUT:  bid=${put_bid:.4f}-${offset:.2f}={put_bid-offset:.4f} > prob=${put_prob:.4f}? {put_sell_condition_met} ({self.position.put_sell_condition.condition_duration:.1f}s)")

    def execute_trading_strategy(self):
        """Execute the market trading strategy"""
        try:
            # Refresh asset IDs if needed (hourly)
            if not self.refresh_asset_ids_if_needed():
                logger.error("Failed to refresh asset IDs, skipping trading cycle")
                return

            # Update balances (only if needed - interval based)
            self.update_usdc_balance_if_needed()
            self.update_token_balances_if_needed()

            # Clean up any pending orders (market strategy requires immediate execution)
            self.cleanup_pending_orders_if_needed()

            # Get probability data
            prob_data = self.get_probability_data()
            if not prob_data:
                logger.error("No probability data available, suspending trading")
                return

            # Get market data
            call_data = self.safe_read_json_file(self.config['call_data_file'])
            put_data = self.safe_read_json_file(self.config['put_data_file'])

            if not call_data or not put_data:
                logger.error("Failed to read market data, suspending trading")
                return

            # Check data freshness for market data
            if not self.check_data_freshness(call_data, 'CALL.json'):
                logger.warning("CALL market data is stale, suspending trading")
                return

            if not self.check_data_freshness(put_data, 'PUT.json'):
                logger.warning("PUT market data is stale, suspending trading")
                return

            # Check spread conditions
            call_spread_ok = self.check_spread_condition(call_data)
            put_spread_ok = self.check_spread_condition(put_data)

            if not call_spread_ok or not put_spread_ok:
                logger.debug("Spread conditions not met, monitoring...")
                return

            # Evaluate market conditions and update timers
            self.evaluate_market_conditions(prob_data, call_data, put_data)

            threshold = self.condition_threshold_seconds

            # Execute CALL trades if conditions are met for long enough
            if (self.position.call_buy_condition.condition_met and
                self.position.call_buy_condition.condition_duration >= threshold):

                # Force fresh balance update before trading decision
                self.update_token_balances()

                call_ask = float(call_data['best_ask']['price'])
                if self.execute_market_buy_order('CALL', self.current_call_asset_id, call_ask, self.position.call_balance):
                    # Reset condition after successful purchase
                    self.position.call_buy_condition = PriceCondition()

            if (self.position.call_sell_condition.condition_met and
                self.position.call_sell_condition.condition_duration >= threshold and
                self.position.call_purchased and
                self.position.call_balance >= 2.0):

                # Force fresh balance update before trading decision
                self.update_token_balances()

                call_bid = float(call_data['best_bid']['price'])
                if self.execute_market_sell_order('CALL', self.current_call_asset_id, call_bid, self.position.call_balance):
                    # Reset condition after successful sale
                    self.position.call_sell_condition = PriceCondition()

            # Execute PUT trades if conditions are met for long enough
            if (self.position.put_buy_condition.condition_met and
                self.position.put_buy_condition.condition_duration >= threshold):

                # Force fresh balance update before trading decision
                self.update_token_balances()

                put_ask = float(put_data['best_ask']['price'])
                if self.execute_market_buy_order('PUT', self.current_put_asset_id, put_ask, self.position.put_balance):
                    # Reset condition after successful purchase
                    self.position.put_buy_condition = PriceCondition()

            if (self.position.put_sell_condition.condition_met and
                self.position.put_sell_condition.condition_duration >= threshold and
                self.position.put_purchased and
                self.position.put_balance >= 2.0):

                # Force fresh balance update before trading decision
                self.update_token_balances()

                put_bid = float(put_data['best_bid']['price'])
                if self.execute_market_sell_order('PUT', self.current_put_asset_id, put_bid, self.position.put_balance):
                    # Reset condition after successful sale
                    self.position.put_sell_condition = PriceCondition()

            # Log current status
            call_prob = float(prob_data['call_probability'])
            put_prob = float(prob_data['put_probability'])

            if self.position.call_buy_condition.condition_met or self.position.call_sell_condition.condition_met or self.position.put_buy_condition.condition_met or self.position.put_sell_condition.condition_met:
                logger.info(f"Probabilities - CALL: {call_prob:.1%}, PUT: {put_prob:.1%}")
                logger.info(f"Positions - CALL: {self.position.call_balance:.1f}/{self.config['max_position_size']} shares | PUT: {self.position.put_balance:.1f}/{self.config['max_position_size']} shares")

                if self.position.call_buy_condition.condition_met:
                    logger.info(f"CALL BUY condition met for {self.position.call_buy_condition.condition_duration:.1f}s (need {threshold}s)")
                if self.position.call_sell_condition.condition_met:
                    logger.info(f"CALL SELL condition met for {self.position.call_sell_condition.condition_duration:.1f}s (need {threshold}s)")
                if self.position.put_buy_condition.condition_met:
                    logger.info(f"PUT BUY condition met for {self.position.put_buy_condition.condition_duration:.1f}s (need {threshold}s)")
                if self.position.put_sell_condition.condition_met:
                    logger.info(f"PUT SELL condition met for {self.position.put_sell_condition.condition_duration:.1f}s (need {threshold}s)")

        except Exception as e:
            logger.error(f"Error in trading strategy: {e}")

    async def run(self):
        """Main bot loop"""
        logger.info("Starting Bitcoin Hourly Options Market Trading Bot")
        logger.info(f"Order size: {self.config['order_size']} shares")
        logger.info(f"Price offset: ${self.config['price_offset']}")
        logger.info(f"Condition threshold: {self.condition_threshold_seconds}s")
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

                # Log API statistics periodically
                self.log_api_stats_if_needed()

                # Log status every 30 seconds
                if int(time.time()) % 30 == 0:
                    now = datetime.now()
                    logger.info(f"Status [{now.strftime('%H:%M:%S')}]: USDC ${self.position.usdc_balance:.2f} | CALL: {self.position.call_balance:.1f}/{self.config['max_position_size']} | PUT: {self.position.put_balance:.1f}/{self.config['max_position_size']} | API calls: {self.api_call_stats['total_calls']}")

                # Wait before next cycle (faster for market conditions)
                await asyncio.sleep(0.1)  # Check every 0.1 seconds for real-time market monitoring

            except KeyboardInterrupt:
                logger.info("Shutdown requested")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(3)  # Wait 3 seconds on error

        # Cleanup
        logger.info("Cleaning up...")
        try:
            self.track_api_call('cancel_all_orders_shutdown')
            cancelled_count = self.trader.cancel_all_orders()
            logger.info(f"Cancelled {cancelled_count} orders on shutdown")

            # Log final API statistics
            self.log_api_statistics()

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main entry point"""
    try:
        bot = BitcoinHourlyMarketBot()
        await bot.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    print("BITCOIN HOURLY OPTIONS MARKET TRADING BOT")
    print("=" * 55)
    print("MARKET TRADING STRATEGY:")
    print("• Read probabilities from option_probabilities.json")
    print("• BUY: When market ask + 0.10$ < probability (for >10s)")
    print("• SELL: When market bid - 0.10$ > probability (for >10s) AND asset owned")
    print("• Market orders executed at current ask/bid prices")
    print("• Fixed order size: 5 shares maximum")
    print("• 10-second wait prevents trading on price spikes")
    print("• HOURLY RESET: 30-second pause at 0:00 to reload asset IDs")
    print()
    print("CONFIGURATION:")
    print(f"• Order size: 5 shares per order")
    print(f"• Max position: 5 shares per asset (prevents over-buying)")
    print(f"• Price offset: $0.10")
    print(f"• Condition threshold: 10 seconds")
    print(f"• Max spread: $0.05")
    print(f"• Order execution: Market orders with immediate cancellation")
    print(f"• Order cleanup: Every 60 seconds (cancel any pending orders)")
    print(f"• Min USDC balance: $5")
    print(f"• Trading cycle: Every 0.1 seconds")
    print(f"• Credentials: keys_ovh38.env")
    print()
    print("CRITICAL - ORDER MANAGEMENT:")
    print("• All orders must execute immediately or be cancelled")
    print("• Orders that don't fill in 0.5 seconds are cancelled")
    print("• Periodic cleanup every 60 seconds cancels any pending orders")
    print("• This prevents stale orders from executing at wrong times")
    print("• Market strategy requires immediate execution only")
    print()
    print("CRITICAL - MARKET ORDER LOGIC:")
    print("• BUY TRIGGER: ask_price + $0.10 < probability_price")
    print("• SELL TRIGGER: bid_price - $0.10 > probability_price")
    print("• Both conditions must persist for 10+ seconds")
    print("• Can only sell assets that were previously purchased")
    print("• Orders execute immediately at market prices")
    print()
    print("HOURLY RESET:")
    print("• At 0:00 each hour, bot will:")
    print("  1. Cancel ALL orders")
    print("  2. Wait 30 seconds")
    print("  3. Reload CALL.json and PUT.json")
    print("  4. Get new asset IDs for new contracts")
    print("  5. Reset purchase tracking")
    print("  6. Resume trading")
    print()
    print("WARNING: This bot executes real market trades with real money!")
    print()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"\nBot failed: {e}")
