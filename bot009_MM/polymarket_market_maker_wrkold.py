#!/usr/bin/env python3
"""
Polymarket Collar Strategy Market Maker

Advanced market making bot with volatility-based spike detection
and dynamic order repositioning for CALL/PUT binary options.
"""

import os
import json
import time
import asyncio
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

# Import the core trading functions
from polymarket_trading_core import PolymarketTrader, load_credentials_from_env, read_market_data_from_json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MarketState:
    """Current market state"""
    btc_price: float
    btc_volatility: float
    call_token_id: str
    put_token_id: str
    call_best_bid: float
    call_best_ask: float
    put_best_bid: float
    put_best_ask: float
    timestamp: float

@dataclass
class Position:
    """Current position tracking"""
    call_balance: float = 0.0
    put_balance: float = 0.0
    call_exposition: float = 0.0  # USD value of CALL position
    put_exposition: float = 0.0   # USD value of PUT position
    total_exposition: float = 0.0
    active_orders: Dict[str, Dict] = None

    def __post_init__(self):
        if self.active_orders is None:
            self.active_orders = {}

class PolymarketMarketMaker:
    """Advanced Market Maker with Collar Strategy"""

    def __init__(self, config_path: str = None):
        """Initialize the market maker"""
        self.config = self._load_config(config_path)
        self.trader = self._initialize_trader()
        self.position = Position()
        self.market_state = None
        self.last_btc_price = None
        self.volatility_window = []
        self.spike_threshold_multiplier = 2.0  # 2x volatility = spike
        self.last_token_reload = 0
        self.token_reload_interval = 3600  # 1 hour

        logger.info("🎯 Polymarket Market Maker initialized")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        default_config = {
            'max_trade_size': 5.0,  # $5 max per trade
            'max_total_exposition': 20.0,  # $20 max total exposition
            'max_call_exposition': 10.0,  # $10 max CALL exposition
            'max_put_exposition': 10.0,  # $10 max PUT exposition
            'min_book_size': 1010,  # Minimum shares to place orders
            'min_order_size': 5.0,  # Polymarket minimum order size
            'min_buy_price': 0.24,  # Don't buy when ask price is below this
            'spread_threshold': 0.04,  # When to tighten spread
            'emergency_exit_price': 0.99,  # Mandatory sell level
            'volatility_window_size': 60,  # 1 hour of 1-min data
            'order_offset': 0.01,  # Order placement offset
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
            creds = load_credentials_from_env('/home/ubuntu/013_2025_polymarket/keys/keys_ovh13.env')
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

    def get_btc_volatility(self) -> float:
        """Calculate BTC volatility from Binance 1-minute klines"""
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': 'BTCUSDT',
                'interval': '1m',
                'limit': self.config['volatility_window_size']
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            klines = response.json()

            # Extract close prices and calculate returns
            closes = [float(kline[4]) for kline in klines]  # Close price index = 4
            returns = np.diff(np.log(closes))

            # Calculate volatility (annualized)
            volatility = np.std(returns) * np.sqrt(525600)  # 525600 minutes per year

            logger.info(f"📊 BTC Volatility: {volatility:.4f}")
            return volatility

        except Exception as e:
            logger.error(f"❌ Error calculating volatility: {e}")
            return 0.5  # Default volatility if calculation fails

    def load_btc_price(self) -> Optional[float]:
        """Load BTC price from JSON file"""
        try:
            if not os.path.exists(self.config['btc_data_file']):
                logger.warning(f"⚠️ BTC data file not found: {self.config['btc_data_file']}")
                return None

            with open(self.config['btc_data_file'], 'r') as f:
                data = json.load(f)
                return float(data['price'])

        except Exception as e:
            logger.error(f"❌ Error loading BTC price: {e}")
            return None

    def reload_token_addresses(self) -> bool:
        """Reload token addresses from CALL.json and PUT.json"""
        try:
            call_data = read_market_data_from_json(self.config['call_data_file'])
            put_data = read_market_data_from_json(self.config['put_data_file'])

            if not call_data or not put_data:
                logger.error("❌ Failed to load token data")
                return False

            # Update market state
            self.market_state = MarketState(
                btc_price=self.load_btc_price() or 0,
                btc_volatility=self.get_btc_volatility(),
                call_token_id=call_data['asset_id'],
                put_token_id=put_data['asset_id'],
                call_best_bid=call_data['best_bid']['price'],
                call_best_ask=call_data['best_ask']['price'],
                put_best_bid=put_data['best_bid']['price'],
                put_best_ask=put_data['best_ask']['price'],
                timestamp=time.time()
            )

            self.last_token_reload = time.time()
            logger.info(f"🔄 Token addresses reloaded - CALL: {call_data['asset_id'][:8]}... PUT: {put_data['asset_id'][:8]}...")
            return True

        except Exception as e:
            logger.error(f"❌ Error reloading token addresses: {e}")
            return False

    def detect_spike(self, current_price: float, previous_price: float, volatility: float) -> str:
        """
        Detect if current price movement is a spike

        Returns: 'UP', 'DOWN', or 'NONE'
        """
        if previous_price is None or volatility == 0:
            return 'NONE'

        price_change_pct = abs(current_price - previous_price) / previous_price
        spike_threshold = volatility * self.spike_threshold_multiplier

        if price_change_pct > spike_threshold:
            if current_price > previous_price:
                logger.info(f"🚀 SPIKE UP detected: {price_change_pct:.4f} > {spike_threshold:.4f}")
                return 'UP'
            else:
                logger.info(f"📉 SPIKE DOWN detected: {price_change_pct:.4f} > {spike_threshold:.4f}")
                return 'DOWN'

        return 'NONE'

    def find_optimal_bid_ask_levels(self, complete_book: Dict, side: str) -> Tuple[float, float]:
        """Find optimal price levels where book size > min_book_size"""
        orders = complete_book.get(side, [])
        min_size = self.config['min_book_size']

        suitable_levels = []
        for order in orders:
            if isinstance(order, dict):
                price = float(order.get('price', 0))
                size = float(order.get('size', 0))
                if size >= min_size:
                    suitable_levels.append((price, size))

        if not suitable_levels:
            # Fallback to best available level
            if orders and isinstance(orders[0], dict):
                return float(orders[0].get('price', 0)), float(orders[0].get('size', 0))
            return 0.0, 0.0

        # Sort by price (ascending for bids, descending for asks)
        if side == 'bids':
            suitable_levels.sort(reverse=True)  # Highest price first
        else:
            suitable_levels.sort()  # Lowest price first

        return suitable_levels[0]  # Return best price with sufficient size

    def calculate_sell_order_size(self, price: float, current_balance: float) -> float:
        """Calculate sell order size - always try to sell since it reduces exposition"""
        # Polymarket minimum order size
        MIN_ORDER_SIZE = 5.0

        # Check if we have enough balance to sell minimum
        if current_balance < MIN_ORDER_SIZE:
            return 0.0

        # Calculate max sell size based on trade size limit
        max_trade_usd = self.config['max_trade_size']
        max_sell_size_by_trade = max_trade_usd / max(price, 0.01)

        # Sell size should be minimum of: max trade size, current balance, reasonable cap
        sell_size = min(max_sell_size_by_trade, current_balance, 100.0)  # Cap at 100 shares per order

        # Ensure we meet minimum size
        if sell_size < MIN_ORDER_SIZE:
            # If we can't sell max_trade_size worth, but have enough balance, sell minimum
            if current_balance >= MIN_ORDER_SIZE:
                return MIN_ORDER_SIZE
            else:
                return 0.0

        return sell_size

    def calculate_order_size(self, price: float, token_type: str = 'CALL') -> float:
        """Calculate BUY order size based on max trade size and current exposition"""
        # Polymarket minimum order size
        MIN_ORDER_SIZE = 5.0

        max_trade_usd = self.config['max_trade_size']
        max_order_size = max_trade_usd / max(price, 0.01)  # Avoid division by zero

        # Check remaining exposition capacity - per token and total
        if token_type == 'CALL':
            remaining_token_capacity = self.config['max_call_exposition'] - self.position.call_exposition
        else:  # PUT
            remaining_token_capacity = self.config['max_put_exposition'] - self.position.put_exposition

        remaining_total_capacity = self.config['max_total_exposition'] - self.position.total_exposition
        remaining_capacity = min(remaining_token_capacity, remaining_total_capacity)

        max_size_by_capacity = remaining_capacity / max(price, 0.01)

        # Calculate final size
        calculated_size = min(max_order_size, max_size_by_capacity, 1000)  # Cap at 1000 shares

        # Return 0 if below minimum, otherwise return at least minimum size
        if calculated_size < MIN_ORDER_SIZE:
            # Check if we can afford the minimum size
            min_cost = MIN_ORDER_SIZE * price
            if min_cost <= remaining_capacity and min_cost <= max_trade_usd:
                return MIN_ORDER_SIZE
            else:
                logger.debug(f"Can't afford minimum size for {token_type}: need ${min_cost:.2f}, have ${remaining_capacity:.2f}")
                return 0.0  # Can't afford minimum size

        return calculated_size
        """Calculate order size based on max trade size and current exposition"""
        # Polymarket minimum order size
        MIN_ORDER_SIZE = 5.0

        max_trade_usd = self.config['max_trade_size']
        max_order_size = max_trade_usd / max(price, 0.01)  # Avoid division by zero

        # Check remaining exposition capacity - per token and total
        if token_type == 'CALL':
            remaining_token_capacity = self.config['max_call_exposition'] - self.position.call_exposition
        else:  # PUT
            remaining_token_capacity = self.config['max_put_exposition'] - self.position.put_exposition

        remaining_total_capacity = self.config['max_total_exposition'] - self.position.total_exposition
        remaining_capacity = min(remaining_token_capacity, remaining_total_capacity)

        max_size_by_capacity = remaining_capacity / max(price, 0.01)

        # Calculate final size
        calculated_size = min(max_order_size, max_size_by_capacity, 1000)  # Cap at 1000 shares

        # Return 0 if below minimum, otherwise return at least minimum size
        if calculated_size < MIN_ORDER_SIZE:
            # Check if we can afford the minimum size
            min_cost = MIN_ORDER_SIZE * price
            if min_cost <= remaining_capacity and min_cost <= max_trade_usd:
                return MIN_ORDER_SIZE
            else:
                logger.debug(f"Can't afford minimum size for {token_type}: need ${min_cost:.2f}, have ${remaining_capacity:.2f}")
                return 0.0  # Can't afford minimum size

        return calculated_size

    def update_position(self):
        """Update current position and calculate separate expositions"""
        try:
            if not self.market_state:
                return

            # Get balances
            call_balance_raw, call_balance = self.trader.get_token_balance(self.market_state.call_token_id)
            put_balance_raw, put_balance = self.trader.get_token_balance(self.market_state.put_token_id)

            self.position.call_balance = call_balance
            self.position.put_balance = put_balance

            # Calculate separate expositions (using mid prices for more accurate valuation)
            call_mid_price = (self.market_state.call_best_bid + self.market_state.call_best_ask) / 2
            put_mid_price = (self.market_state.put_best_bid + self.market_state.put_best_ask) / 2

            self.position.call_exposition = call_balance * call_mid_price
            self.position.put_exposition = put_balance * put_mid_price
            self.position.total_exposition = self.position.call_exposition + self.position.put_exposition

            logger.info(f"💰 Position - CALL: {call_balance:.1f} (${self.position.call_exposition:.2f}), "
                       f"PUT: {put_balance:.1f} (${self.position.put_exposition:.2f}), "
                       f"Total: ${self.position.total_exposition:.2f}")

            # Log remaining capacity per token
            call_remaining = self.config['max_call_exposition'] - self.position.call_exposition
            put_remaining = self.config['max_put_exposition'] - self.position.put_exposition
            total_remaining = self.config['max_total_exposition'] - self.position.total_exposition

            logger.info(f"📊 Capacity - CALL: ${call_remaining:.2f}, PUT: ${put_remaining:.2f}, Total: ${total_remaining:.2f}")

        except Exception as e:
            logger.error(f"❌ Error updating position: {e}")

    def emergency_exit(self):
        """Emergency exit when price reaches 0.99"""
        try:
            logger.warning("🚨 EMERGENCY EXIT TRIGGERED - Price at 0.99")

            # Cancel all orders first
            self.trader.cancel_all_orders()

            # Sell all CALL positions
            if self.position.call_balance > 0:
                self.trader.place_sell_order(
                    self.market_state.call_token_id,
                    0.98,  # Sell at 0.98
                    self.position.call_balance
                )

            # Sell all PUT positions
            if self.position.put_balance > 0:
                self.trader.place_sell_order(
                    self.market_state.put_token_id,
                    0.98,  # Sell at 0.98
                    self.position.put_balance
                )

            logger.warning("🚨 Emergency exit orders placed")

        except Exception as e:
            logger.error(f"❌ Error in emergency exit: {e}")

    def place_collar_orders(self, spike_direction: str = 'NONE'):
        """Place collar strategy orders based on current market conditions"""
        try:
            if not self.market_state:
                logger.warning("⚠️ No market state available")
                return

            # Cancel existing orders
            self.trader.cancel_all_orders()
            time.sleep(0.5)  # Brief pause

            # Load fresh market data
            call_data = read_market_data_from_json(self.config['call_data_file'])
            put_data = read_market_data_from_json(self.config['put_data_file'])

            if not call_data or not put_data:
                logger.error("❌ Failed to load fresh market data")
                return

            # Check for emergency exit conditions
            if (call_data['best_ask']['price'] >= self.config['emergency_exit_price'] or
                put_data['best_ask']['price'] >= self.config['emergency_exit_price']):
                self.emergency_exit()
                return

            # Check exposition limits before placing any orders
            call_limit_reached = self.position.call_exposition >= self.config['max_call_exposition'] * 0.9
            put_limit_reached = self.position.put_exposition >= self.config['max_put_exposition'] * 0.9
            total_limit_reached = self.position.total_exposition >= self.config['max_total_exposition'] * 0.9

            if call_limit_reached or put_limit_reached or total_limit_reached:
                logger.warning(f"⚠️ Near exposition limits - CALL: {call_limit_reached}, PUT: {put_limit_reached}, Total: {total_limit_reached}")
                # Only place sell orders when near limits to reduce exposition
                self._place_sell_only_orders(call_data, put_data, self.config['order_offset'])
                return

            offset = self.config['order_offset']

            if spike_direction == 'UP':
                # BTC spiked UP: CALL becomes more valuable, PUT less valuable
                self._place_spike_up_orders(call_data, put_data, offset)
            elif spike_direction == 'DOWN':
                # BTC spiked DOWN: PUT becomes more valuable, CALL less valuable
                self._place_spike_down_orders(call_data, put_data, offset)
            else:
                # Normal market conditions
                self._place_normal_orders(call_data, put_data, offset)

        except Exception as e:
            logger.error(f"❌ Error placing collar orders: {e}")

    def _place_spike_up_orders(self, call_data: Dict, put_data: Dict, offset: float):
        """Place orders when BTC spikes UP"""
        try:
            # CALL: Remove ask, place bid at best_bid + offset
            call_bid_price, call_bid_size = self.find_optimal_bid_ask_levels(
                call_data['complete_book'], 'bids'
            )
            call_ask_price = call_data['best_ask']['price']  # Check current ask price

            if call_bid_price > 0:
                bid_price = min(call_bid_price + offset, 0.98)

                # Check minimum buy price condition
                if call_ask_price < self.config['min_buy_price']:
                    logger.warning(f"⚠️ CALL BUY skipped: ask price ${call_ask_price:.4f} < ${self.config['min_buy_price']:.2f} minimum")
                else:
                    order_size = self.calculate_order_size(bid_price, 'CALL')
                    if order_size >= 5.0:  # Only place if meets minimum
                        order_id = self.trader.place_buy_order(call_data['asset_id'], bid_price, order_size)
                        if order_id:
                            logger.info(f"🚀 CALL BUY placed: {order_size} @ ${bid_price:.4f} (ask: ${call_ask_price:.4f})")
                    else:
                        logger.warning(f"⚠️ CALL BUY skipped: size {order_size:.2f} < 5.0 minimum or exposition limit")

            # PUT: Remove bid, place ask at best_ask - offset (SELL)
            put_ask_price, put_ask_size = self.find_optimal_bid_ask_levels(
                put_data['complete_book'], 'asks'
            )
            if put_ask_price > 0:
                ask_price = max(put_ask_price - offset, 0.02)
                order_size = self.calculate_sell_order_size(ask_price, self.position.put_balance)
                if order_size >= 5.0:
                    order_id = self.trader.place_sell_order(put_data['asset_id'], ask_price, order_size)
                    if order_id:
                        logger.info(f"🚀 PUT SELL placed: {order_size} @ ${ask_price:.4f}")
                else:
                    logger.warning(f"⚠️ PUT SELL skipped: size {order_size:.2f} or insufficient balance ({self.position.put_balance:.2f})")

            logger.info("🚀 Spike UP order processing complete")

        except Exception as e:
            logger.error(f"❌ Error placing spike UP orders: {e}")

    def _place_spike_down_orders(self, call_data: Dict, put_data: Dict, offset: float):
        """Place orders when BTC spikes DOWN"""
        try:
            # PUT: Remove ask, place bid at best_bid + offset
            put_bid_price, put_bid_size = self.find_optimal_bid_ask_levels(
                put_data['complete_book'], 'bids'
            )
            put_ask_price = put_data['best_ask']['price']  # Check current ask price

            if put_bid_price > 0:
                bid_price = min(put_bid_price + offset, 0.98)

                # Check minimum buy price condition
                if put_ask_price < self.config['min_buy_price']:
                    logger.warning(f"⚠️ PUT BUY skipped: ask price ${put_ask_price:.4f} < ${self.config['min_buy_price']:.2f} minimum")
                else:
                    order_size = self.calculate_order_size(bid_price, 'PUT')
                    if order_size >= 5.0:  # Only place if meets minimum
                        order_id = self.trader.place_buy_order(put_data['asset_id'], bid_price, order_size)
                        if order_id:
                            logger.info(f"📉 PUT BUY placed: {order_size} @ ${bid_price:.4f} (ask: ${put_ask_price:.4f})")
                    else:
                        logger.warning(f"⚠️ PUT BUY skipped: size {order_size:.2f} < 5.0 minimum or exposition limit")

            # CALL: Remove bid, place ask at best_ask - offset (SELL)
            call_ask_price, call_ask_size = self.find_optimal_bid_ask_levels(
                call_data['complete_book'], 'asks'
            )
            if call_ask_price > 0:
                ask_price = max(call_ask_price - offset, 0.02)
                order_size = self.calculate_sell_order_size(ask_price, self.position.call_balance)
                if order_size >= 5.0:
                    order_id = self.trader.place_sell_order(call_data['asset_id'], ask_price, order_size)
                    if order_id:
                        logger.info(f"📉 CALL SELL placed: {order_size} @ ${ask_price:.4f}")
                else:
                    logger.warning(f"⚠️ CALL SELL skipped: size {order_size:.2f} or insufficient balance ({self.position.call_balance:.2f})")

            logger.info("📉 Spike DOWN order processing complete")

        except Exception as e:
            logger.error(f"❌ Error placing spike DOWN orders: {e}")

    def _place_normal_orders(self, call_data: Dict, put_data: Dict, offset: float):
        """Place orders during normal market conditions"""
        try:
            # Check if we should tighten spread due to low volatility
            call_spread = call_data['best_ask']['price'] - call_data['best_bid']['price']
            put_spread = put_data['best_ask']['price'] - put_data['best_bid']['price']

            # Adjust offset if spread is too wide and volatility is low
            if (call_spread > self.config['spread_threshold'] or
                put_spread > self.config['spread_threshold']):
                if self.market_state.btc_volatility < 0.3:  # Low volatility
                    offset = offset * 0.5  # Tighten spread

            # Place CALL orders
            self._place_token_orders(call_data, 'CALL', offset)

            # Place PUT orders
            self._place_token_orders(put_data, 'PUT', offset)

            logger.info("📊 Normal market orders placed")

        except Exception as e:
            logger.error(f"❌ Error placing normal orders: {e}")

    def _place_sell_only_orders(self, call_data: Dict, put_data: Dict, offset: float):
        """Place only sell orders when near exposition limits"""
        try:
            # Check which tokens are over their individual limits
            call_over_limit = self.position.call_exposition >= self.config['max_call_exposition'] * 0.9
            put_over_limit = self.position.put_exposition >= self.config['max_put_exposition'] * 0.9

            # CALL sells (always try to sell if we have balance ≥ 5)
            if self.position.call_balance >= 5.0:
                call_ask_price, _ = self.find_optimal_bid_ask_levels(
                    call_data['complete_book'], 'asks'
                )
                if call_ask_price > 0:
                    final_ask_price = max(call_ask_price - offset, 0.02)
                    sell_size = self.calculate_sell_order_size(final_ask_price, self.position.call_balance)

                    if sell_size >= 5.0:
                        order_id = self.trader.place_sell_order(call_data['asset_id'], final_ask_price, sell_size)
                        if order_id:
                            priority = "HIGH PRIORITY" if call_over_limit else "reduce exp"
                            logger.info(f"💰 CALL SELL ({priority}): {sell_size} @ ${final_ask_price:.4f}")
                    else:
                        logger.warning(f"⚠️ CALL SELL skipped: size {sell_size:.2f} < 5.0")

            # PUT sells (always try to sell if we have balance ≥ 5)
            if self.position.put_balance >= 5.0:
                put_ask_price, _ = self.find_optimal_bid_ask_levels(
                    put_data['complete_book'], 'asks'
                )
                if put_ask_price > 0:
                    final_ask_price = max(put_ask_price - offset, 0.02)
                    sell_size = self.calculate_sell_order_size(final_ask_price, self.position.put_balance)

                    if sell_size >= 5.0:
                        order_id = self.trader.place_sell_order(put_data['asset_id'], final_ask_price, sell_size)
                        if order_id:
                            priority = "HIGH PRIORITY" if put_over_limit else "reduce exp"
                            logger.info(f"💰 PUT SELL ({priority}): {sell_size} @ ${final_ask_price:.4f}")
                    else:
                        logger.warning(f"⚠️ PUT SELL skipped: size {sell_size:.2f} < 5.0")

            logger.info("💰 Sell-only orders placed to manage exposition limits")

        except Exception as e:
            logger.error(f"❌ Error placing sell-only orders: {e}")

    def _place_token_orders(self, token_data: Dict, token_type: str, offset: float):
        """Place bid and ask orders for a specific token"""
        try:
            token_id = token_data['asset_id']
            current_ask_price = token_data['best_ask']['price']

            # Find optimal bid level and place BUY order
            bid_price, bid_size = self.find_optimal_bid_ask_levels(
                token_data['complete_book'], 'bids'
            )

            if bid_price > 0:
                final_bid_price = min(bid_price + offset, 0.98)

                # Check minimum buy price condition
                if current_ask_price < self.config['min_buy_price']:
                    logger.warning(f"⚠️ {token_type} BUY skipped: ask price ${current_ask_price:.4f} < ${self.config['min_buy_price']:.2f} minimum")
                else:
                    order_size = self.calculate_order_size(final_bid_price, token_type)
                    if order_size >= 5.0:  # Only place if meets minimum
                        order_id = self.trader.place_buy_order(token_id, final_bid_price, order_size)
                        if order_id:
                            logger.info(f"📊 {token_type} BUY placed: {order_size} @ ${final_bid_price:.4f} (ask: ${current_ask_price:.4f})")
                    else:
                        logger.warning(f"⚠️ {token_type} BUY skipped: size {order_size:.2f} < 5.0 minimum or exposition limit")

            # Find optimal ask level and place SELL order
            ask_price, ask_size = self.find_optimal_bid_ask_levels(
                token_data['complete_book'], 'asks'
            )

            if ask_price > 0:
                final_ask_price = max(ask_price - offset, 0.02)

                # Get current balance for this token type
                current_balance = (self.position.call_balance if token_type == 'CALL'
                                 else self.position.put_balance)

                # Calculate sell order size (always tries to sell if we have enough)
                order_size = self.calculate_sell_order_size(final_ask_price, current_balance)

                if order_size >= 5.0:
                    order_id = self.trader.place_sell_order(token_id, final_ask_price, order_size)
                    if order_id:
                        logger.info(f"📊 {token_type} SELL placed: {order_size} @ ${final_ask_price:.4f}")
                else:
                    logger.warning(f"⚠️ {token_type} SELL skipped: size {order_size:.2f} or insufficient balance ({current_balance:.2f})")

        except Exception as e:
            logger.error(f"❌ Error placing {token_type} orders: {e}")

    async def run(self):
        """Main trading loop"""
        logger.info("🎯 Starting Polymarket Market Maker")

        # Initial setup
        if not self.reload_token_addresses():
            logger.error("❌ Failed initial token reload")
            return

        self.last_btc_price = self.market_state.btc_price

        while True:
            try:
                # Check if we need to reload token addresses (hourly)
                if time.time() - self.last_token_reload > self.token_reload_interval:
                    logger.info("🔄 Hourly token reload")
                    self.reload_token_addresses()

                # Update current BTC price
                current_btc_price = self.load_btc_price()
                if current_btc_price is None:
                    logger.warning("⚠️ Could not load BTC price, using last known price")
                    current_btc_price = self.last_btc_price

                # Update volatility
                volatility = self.get_btc_volatility()
                if self.market_state:
                    self.market_state.btc_price = current_btc_price
                    self.market_state.btc_volatility = volatility

                # Detect spike
                spike_direction = self.detect_spike(
                    current_btc_price,
                    self.last_btc_price,
                    volatility
                )

                # Update position
                self.update_position()

                # Place orders based on market conditions and risk limits
                self.place_collar_orders(spike_direction)

                # Update last BTC price
                self.last_btc_price = current_btc_price

                # Log current state with per-token detail
                call_remaining = self.config['max_call_exposition'] - self.position.call_exposition
                put_remaining = self.config['max_put_exposition'] - self.position.put_exposition
                total_remaining = self.config['max_total_exposition'] - self.position.total_exposition
                logger.info(f"🔄 BTC: ${current_btc_price:.2f}, Vol: {volatility:.4f}")
                logger.info(f"💼 CALL: ${call_remaining:.2f} remaining, PUT: ${put_remaining:.2f} remaining, Total: ${total_remaining:.2f} remaining")

                # Sleep before next iteration (adjust for speed)
                await asyncio.sleep(5)  # 5 second intervals for real-time response

            except KeyboardInterrupt:
                logger.info("🛑 Shutdown requested")
                break
            except Exception as e:
                logger.error(f"❌ Error in main loop: {e}")
                await asyncio.sleep(10)  # Wait longer on error

        # Cleanup
        logger.info("🧹 Cleaning up - cancelling all orders")
        self.trader.cancel_all_orders()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main entry point"""
    try:
        # Initialize market maker
        market_maker = PolymarketMarketMaker()

        # Run the trading loop
        await market_maker.run()

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        raise

if __name__ == "__main__":
    print("🎯 POLYMARKET COLLAR STRATEGY MARKET MAKER")
    print("=" * 60)
    print("Features:")
    print("• Volatility-based spike detection")
    print("• Dynamic order repositioning")
    print("• Per-token risk management ($10 CALL + $10 PUT = $20 total max)")
    print("• Emergency exit at 0.99 price")
    print("• Real-time BTC price monitoring")
    print("• Hourly token address reload")
    print("• Minimum 5-share order size compliance")
    print("• Smart exposition management per token")
    print("• Minimum buy price filter (don't buy below $0.24)")
    print()
    print("Configuration:")
    print("• Max trade size: $5 per order")
    print("• Max CALL exposition: $10")
    print("• Max PUT exposition: $10")
    print("• Max total exposition: $20")
    print("• Minimum buy price: $0.24")
    print("• File paths: /home/ubuntu/013_2025_polymarket/")
    print()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Market maker stopped by user")
    except Exception as e:
        print(f"\n❌ Market maker failed: {e}")
