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
9. NEW: No trading for first 2 minutes of each hour (00:00-02:00)
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

# pm2 start filename.py --cron-restart="*/15 * * * *" --interpreter python3
# pm2 start 15M_polymarket_dynamic_monitor.py --cron-restart="*/15 * * * *" --interpreter python3


# Configure logging to suppress warnings and only show errors
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
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
        self.asset_refresh_interval = 900

        # Hourly reset tracking
        self.last_hour_processed = -1  # Track which hour we last processed
        self.is_in_hourly_reset = False  # Flag to indicate we're in reset mode

        # NEW: No-trading window tracking (first 2 minutes of each hour)
        self.no_trading_window_minutes = 0  # No trading for first 2 minutes of each hour
        self.is_in_no_trading_window = False  # Flag to indicate we're in no-trading period

        # Time condition threshold
        self.condition_threshold_seconds = 1.0  # 10 seconds

        # Print tracking
        self.last_print_time = 0
        self.print_interval = 1  # Print every 1 second

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
        pass  # Suppressed to reduce output

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
            return "0x0000000000000000000000000000000000000000"

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        default_config = {
            'order_size': 5.0,              # Fixed order size in shares
            'max_position_size': 5.0,       # Maximum total shares per asset
            'price_offset': 0.1,           # Offset for market order conditions
            'max_spread': 0.9,             # Maximum acceptable bid-ask spread
            'min_usdc_balance': 4.0,        # Minimum USDC.e balance required
            'condition_threshold': 1.0,    # Seconds to wait before executing
            'polygon_rpc': 'https://polygon-mainnet.core.chainstack.com/d308330baf4e49cfc74c637905858979',
            'btc_data_file': '/home/ubuntu/013_2025_polymarket/btc_price.json',
            'call_data_file': '/home/ubuntu/013_2025_polymarket/15M_CALL.json',
            'put_data_file': '/home/ubuntu/013_2025_polymarket/15M_PUT.json',
            'probability_file': '/home/ubuntu/013_2025_polymarket/option_probabilities_M15.json'
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
            raise

    def safe_read_json_file(self, file_path: str) -> Optional[Dict]:
        """Safely read JSON file with error handling"""
        try:
            if not os.path.exists(file_path):
                return None

            with open(file_path, 'r') as f:
                content = f.read().strip()

            if not content:
                return None

            return json.loads(content)

        except json.JSONDecodeError as e:
            return None
        except Exception as e:
            return None

    def get_usdc_balance(self) -> float:
        """Get USDC.e balance using Web3"""
        try:
            if not self.w3.is_connected():
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
            return 0.0

    def update_usdc_balance_if_needed(self):
        """Update USDC.e balance if enough time has passed"""
        current_time = time.time()

        if current_time - self.last_balance_check >= self.balance_check_interval:
            try:
                usdc_balance = self.get_usdc_balance()
                self.position.usdc_balance = usdc_balance
                self.last_balance_check = current_time
            except Exception as e:
                pass

    def is_in_no_trading_period(self) -> bool:
        """Check if we're currently in the no-trading window (first 2 minutes of each hour)"""
        now = datetime.now()

        # Check if we're in the first X minutes of any hour
        is_no_trade_window = now.minute < self.no_trading_window_minutes

        return is_no_trade_window

    def check_hourly_reset_needed(self) -> bool:
        """Check if we need to do hourly reset at 0:00"""
        now = datetime.now()
        current_hour = now.hour

        # Check if we're at the start of a new hour (0:00-0:30 seconds)
        is_start_of_hour = ((now.minute == 0 or now.minute == 15 or now.minute == 30 or now.minute == 45) and now.second < 15)

        # Check if this is a new hour we haven't processed
        is_new_hour = current_hour != self.last_hour_processed

        needs_reset = is_start_of_hour and is_new_hour and not self.is_in_hourly_reset

        return needs_reset

    async def execute_hourly_reset(self):
        """Execute 30-second hourly reset to handle asset ID changes"""
        try:
            now = datetime.now()
            self.is_in_hourly_reset = True
            self.last_hour_processed = now.hour

            # Step 1: Cancel all active orders
            self.track_api_call('cancel_all_orders_hourly_reset')
            cancelled_count = self.trader.cancel_all_orders()

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

            # Step 3: Wait 10 seconds for new contracts to be available
            for i in range(10, 0, -1):
                await asyncio.sleep(1)

            # Step 4: Force reload asset IDs from fresh JSON files
            call_data = self.safe_read_json_file(self.config['call_data_file'])
            put_data = self.safe_read_json_file(self.config['put_data_file'])

            if call_data and 'asset_id' in call_data:
                old_call_id = self.current_call_asset_id
                self.current_call_asset_id = call_data['asset_id']

            if put_data and 'asset_id' in put_data:
                old_put_id = self.current_put_asset_id
                self.current_put_asset_id = put_data['asset_id']

            # Step 5: Reset position tracking for new hour
            self.position.last_asset_refresh = time.time()

            # Step 6: Update token balances for new contracts (forced after reset)
            self.update_token_balances()

            self.is_in_hourly_reset = False

        except Exception as e:
            self.is_in_hourly_reset = False

    def refresh_asset_ids_if_needed(self):
        """Refresh asset IDs at the beginning of each hour"""
        current_time = time.time()

        # Check if we need to refresh (either first time or hourly)
        if (self.position.last_asset_refresh == 0 or
            current_time - self.position.last_asset_refresh >= self.asset_refresh_interval):

            # Read CALL asset ID
            call_data = self.safe_read_json_file(self.config['call_data_file'])
            if call_data and 'asset_id' in call_data:
                self.current_call_asset_id = call_data['asset_id']
            else:
                return False

            # Read PUT asset ID
            put_data = self.safe_read_json_file(self.config['put_data_file'])
            if put_data and 'asset_id' in put_data:
                self.current_put_asset_id = put_data['asset_id']
            else:
                return False

            self.position.last_asset_refresh = current_time
            return True

        return True  # No refresh needed

    def check_data_freshness(self, data: Dict, file_name: str, max_age_seconds: int = 2) -> bool:
        """Check if data is fresh enough for trading"""
        try:
            current_time_ms = int(time.time() * 1000)

            # Check for timestamp field (could be 'timestamp' or 'updated_at')
            data_timestamp = data.get('timestamp') or data.get('updated_at')

            if not data_timestamp:
                return False

            # Convert to milliseconds if needed
            data_timestamp = int(float(data_timestamp))
            if data_timestamp < 1e12:  # If timestamp is in seconds, convert to ms
                data_timestamp *= 1000

            age_ms = current_time_ms - data_timestamp
            age_seconds = age_ms / 1000

            if age_seconds > max_age_seconds:
                return False

            return True

        except Exception as e:
            return False

    def get_probability_data(self) -> Optional[Dict]:
        """Read probability data from option_probabilities.json"""
        prob_data = self.safe_read_json_file(self.config['probability_file'])

        if not prob_data:
            return None

        # Check data freshness
        if not self.check_data_freshness(prob_data, 'option_probabilities.json'):
            return None

        required_fields = ['call_probability', 'put_probability']
        for field in required_fields:
            if field not in prob_data:
                return None

        return prob_data

    def check_spread_condition(self, market_data: Dict) -> bool:
        """Check if bid-ask spread is acceptable"""
        try:
            best_bid = float(market_data['best_bid']['price'])
            best_ask = float(market_data['best_ask']['price'])
            spread = best_ask - best_bid

            if spread > self.config['max_spread']:
                return False

            return True
        except Exception as e:
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

            except Exception as e:
                pass

    def cleanup_pending_orders_if_needed(self):
        """Cancel any pending orders if enough time has passed"""
        current_time = time.time()

        if current_time - self.last_order_cleanup >= self.order_cleanup_interval:
            try:
                # Cancel all pending orders (market strategy requires immediate execution only)
                self.track_api_call('cancel_all_orders_cleanup')
                cancelled_count = self.trader.cancel_all_orders()

                self.last_order_cleanup = current_time

            except Exception as e:
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

        except Exception as e:
            pass

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
            return False, 0.0

        if available_to_buy < order_size:
            return False, 0.0

        # Can buy the full order size
        return True, min(order_size, available_to_buy)

    def execute_market_buy_order(self, token_name: str, asset_id: str, market_ask_price: float, current_balance: float) -> bool:
        """Execute a market buy order at current ask price"""
        try:
            # CRITICAL: Cancel all existing orders before placing new ones
            self.track_api_call(f'cancel_all_orders_before_buy_{token_name.lower()}')
            cancelled_count = self.trader.cancel_all_orders()

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
                return False

            # Double check with hard limit
            if fresh_balance >= self.config['max_position_size']:
                return False

            total_cost = buy_size * market_ask_price

            # Check USDC balance
            if self.position.usdc_balance < total_cost:
                return False

            # Track pending order BEFORE placing it
            if token_name == 'CALL':
                self.position.call_pending_buys += buy_size
            elif token_name == 'PUT':
                self.position.put_pending_buys += buy_size

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
                    # Order didn't execute immediately - cancel everything again
                    self.track_api_call(f'cancel_all_orders_after_failed_buy_{token_name.lower()}')
                    self.trader.cancel_all_orders()

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

                    # Note: Token balances will be updated on next interval or before next trade

                    return True
            else:
                # Clear pending if order failed
                if token_name == 'CALL':
                    self.position.call_pending_buys = max(0, self.position.call_pending_buys - buy_size)
                elif token_name == 'PUT':
                    self.position.put_pending_buys = max(0, self.position.put_pending_buys - buy_size)

                return False

        except Exception as e:
            # Clear pending if exception occurred
            if token_name == 'CALL':
                self.position.call_pending_buys = max(0, self.position.call_pending_buys - (buy_size if 'buy_size' in locals() else 0))
            elif token_name == 'PUT':
                self.position.put_pending_buys = max(0, self.position.put_pending_buys - (buy_size if 'buy_size' in locals() else 0))

            return False

    def execute_market_sell_order(self, token_name: str, asset_id: str, market_bid_price: float, balance: float) -> bool:
        """Execute a market sell order at current bid price"""
        try:
            # CRITICAL: Cancel all existing orders before placing new ones
            self.track_api_call(f'cancel_all_orders_before_sell_{token_name.lower()}')
            cancelled_count = self.trader.cancel_all_orders()

            # Validate we have tokens to sell
            if balance < self.config['order_size']:
                return False

            sell_size = min(balance, self.config['order_size'])

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
                    # Order didn't execute immediately - cancel everything again
                    self.track_api_call(f'cancel_all_orders_after_failed_sell_{token_name.lower()}')
                    self.trader.cancel_all_orders()

                    return False
                else:
                    # Order executed successfully
                    # Update purchase tracking if we sold everything
                    if sell_size >= balance * 0.9:  # If we sold most of our holdings
                        if token_name == 'CALL':
                            self.position.call_purchased = False
                        elif token_name == 'PUT':
                            self.position.put_purchased = False

                    # Note: Token balances will be updated on next interval or before next trade

                    return True
            else:
                return False

        except Exception as e:
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

    def print_current_data_if_needed(self, prob_data: Dict, call_data: Dict, put_data: Dict):
        """Print current market data if enough time has passed"""
        current_time = time.time()

        if current_time - self.last_print_time >= self.print_interval:
            try:
                call_prob = float(prob_data['call_probability'])
                call_ask = float(call_data['best_ask']['price'])
                put_prob = float(prob_data['put_probability'])
                put_ask = float(put_data['best_ask']['price'])

                print(f"Call prob: {call_prob:.4f} - Call ask: {call_ask:.4f} || Put prob: {put_prob:.4f} - Put ask: {put_ask:.4f}")

                self.last_print_time = current_time
            except Exception as e:
                pass

    def execute_trading_strategy(self):
        """Execute the market trading strategy"""
        try:
            # NEW: Check if we're in the no-trading window
            if self.is_in_no_trading_period():
                return

            # Refresh asset IDs if needed (hourly)
            if not self.refresh_asset_ids_if_needed():
                return

            # Update balances (only if needed - interval based)
            self.update_usdc_balance_if_needed()
            self.update_token_balances_if_needed()

            # Clean up any pending orders (market strategy requires immediate execution)
            self.cleanup_pending_orders_if_needed()

            # Get probability data
            prob_data = self.get_probability_data()
            if not prob_data:
                return

            # Get market data
            call_data = self.safe_read_json_file(self.config['call_data_file'])
            put_data = self.safe_read_json_file(self.config['put_data_file'])

            if not call_data or not put_data:
                return

            # Check data freshness for market data
            if not self.check_data_freshness(call_data, 'CALL.json'):
                return

            if not self.check_data_freshness(put_data, 'PUT.json'):
                return

            # Check spread conditions
            call_spread_ok = self.check_spread_condition(call_data)
            put_spread_ok = self.check_spread_condition(put_data)

            if not call_spread_ok or not put_spread_ok:
                return

            # Print current data
            self.print_current_data_if_needed(prob_data, call_data, put_data)

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

        except Exception as e:
            pass

    async def run(self):
        """Main bot loop"""
        # Initial setup
        if not self.refresh_asset_ids_if_needed():
            return

        # Check initial balance but don't exit - just log
        initial_balance = self.get_usdc_balance()
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

                # Execute trading strategy (includes no-trading window check)
                self.execute_trading_strategy()

                # Log API statistics periodically
                self.log_api_stats_if_needed()

                # Wait before next cycle (faster for market conditions)
                await asyncio.sleep(0.1)  # Check every 0.1 seconds for real-time market monitoring

            except KeyboardInterrupt:
                break
            except Exception as e:
                await asyncio.sleep(3)  # Wait 3 seconds on error

        # Cleanup
        try:
            self.track_api_call('cancel_all_orders_shutdown')
            cancelled_count = self.trader.cancel_all_orders()

            # Log final API statistics
            self.log_api_statistics()

        except Exception as e:
            pass

# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main entry point"""
    try:
        bot = BitcoinHourlyMarketBot()
        await bot.run()
    except Exception as e:
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        pass
