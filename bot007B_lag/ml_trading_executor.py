#!/usr/bin/env python3
"""
 ML Arbitrage Trading Executor

Key improvements:
1. Fixed position sizing calculation
2. Better price validation and buffer handling
3. Enhanced balance checking
4.  error handling and logging
5. More robust market data handling
"""

import time
import json
import os
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, BalanceAllowanceParams, AssetType, OrderArgs
from py_clob_client.order_builder.constants import BUY, SELL

# Import ML detector
from ml_arbitrage_detector import MLArbitrageDetector

# Mandatory sell constants
MANDATORY_SELL_PRICE = 0.99          # Trigger mandatory sell at $0.99
MIN_BALANCE_SELL = 1000000           # Minimum 1 token (in raw units) to sell
MANDATORY_SELL_BUFFER = 0.005        # Small buffer below $0.99 for aggressive selling

@dataclass
class MLTradingConfig:
    """Enhanced configuration with better defaults"""

    # API Configuration
    clob_api_url: str
    private_key: str
    api_key: str
    api_secret: str
    api_passphrase: str
    chain_id: int = 137

    # File paths
    btc_file: str = '/home/ubuntu/013_2025_polymarket/btc_price.json'
    call_file: str = '/home/ubuntu/013_2025_polymarket/CALL.json'
    put_file: str = '/home/ubuntu/013_2025_polymarket/PUT.json'

    # Trading parameters (more conservative defaults)
    max_position_size_usd: float = 10.0      # Reduced from 50
    max_total_exposure_usd: float = 50.0     # Reduced from 200
    min_confidence_threshold: float = 0.75   # Increased from 0.6
    min_profit_threshold: float = 0.025      # Increased from 0.02
    max_trades_per_hour: int = 5             # Reduced from 10

    # Risk management
    stop_loss_threshold: float = -0.03       # Tighter stop loss
    take_profit_threshold: float = 0.06      # Lower take profit
    max_hold_time_minutes: int = 20          # Shorter hold time

    # Order execution (improved)
    order_timeout_seconds: int = 30          # Faster timeout
    price_buffer_pct: float = 0.001          # Use percentage instead of fixed cents
    min_order_size: float = 1.0
    max_price_impact: float = 0.05           # Maximum 5% price impact

class MLPosition:
    """Enhanced position tracking with better validation"""

    def __init__(self, opportunity: Dict, token_id: str, entry_price: float,
                 quantity: float, timestamp: float):
        # Validate inputs
        if entry_price <= 0:
            raise ValueError(f"Invalid entry price: {entry_price}")
        if quantity <= 0:
            raise ValueError(f"Invalid quantity: {quantity}")

        self.opportunity = opportunity
        self.token_id = token_id
        self.entry_price = entry_price
        self.quantity = quantity
        self.timestamp = timestamp

        # Position details
        self.option_type = opportunity['type']
        self.action = opportunity['action']
        self.ml_confidence = opportunity.get('confidence', 0)
        self.expected_profit = opportunity.get('profit_potential', 0)
        self.market_regime = opportunity.get('market_regime', 'unknown')

        # Status tracking
        self.status = 'pending'  # pending, filled, cancelled, closed
        self.fill_price = None
        self.fill_quantity = None
        self.exit_price = None
        self.exit_timestamp = None
        self.realized_pnl = 0.0

    def get_current_pnl(self, current_price: float) -> float:
        """Calculate current unrealized PnL"""
        if self.status != 'filled' or not self.fill_price:
            return 0.0

        if self.action == 'BUY':
            return (current_price - self.fill_price) * self.fill_quantity
        else:  # SELL
            return (self.fill_price - current_price) * self.fill_quantity

    def get_hold_time_minutes(self) -> float:
        """Get position hold time in minutes"""
        return (time.time() - self.timestamp) / 60.0

    def should_close(self, current_price: float, config) -> Tuple[bool, str]:
        """Comprehensive position close logic"""
        if self.status != 'filled':
            return False, ""

        current_pnl = self.get_current_pnl(current_price)
        hold_time = self.get_hold_time_minutes()

        # Stop loss check
        if current_pnl < config.stop_loss_threshold:
            return True, "stop_loss"

        # Take profit check
        if current_pnl > config.take_profit_threshold:
            return True, "take_profit"

        # Max hold time check
        if hold_time > config.max_hold_time_minutes:
            return True, "max_hold_time"

        # Near expiry check (2 minutes before hour end)
        minutes_to_hour = 60 - datetime.now().minute
        if minutes_to_hour <= 2:
            return True, "near_expiry"

        return False, ""

class MLTradingExecutor:

    def check_mandatory_sell_conditions(self, option_type: str, market_data: Dict, token_balance: float) -> bool:
        """Check if mandatory sell conditions are met (market at $0.99)"""

        if token_balance < 1.0:  # Need at least 1 token to sell
            return False

        if not market_data:
            return False

        # Get current market prices
        best_bid_data = market_data.get('best_bid', {})
        best_ask_data = market_data.get('best_ask', {})

        best_bid = best_bid_data.get('price', 0) if best_bid_data else 0
        best_ask = best_ask_data.get('price', 0) if best_ask_data else 0
        market_mid = market_data.get('mid_price', 0) if 'mid_price' in market_data else (best_bid + best_ask) / 2

        # Multiple mandatory sell triggers
        mandatory_conditions = [
            best_bid >= MANDATORY_SELL_PRICE,                                    # Best bid at $0.99+
            market_mid >= MANDATORY_SELL_PRICE,                                  # Mid price at $0.99+
            (best_ask >= MANDATORY_SELL_PRICE and best_bid >= 0.97),            # Ask at $0.99+ and bid very high
            best_bid >= (MANDATORY_SELL_PRICE - MANDATORY_SELL_BUFFER),         # Aggressive: bid at $0.985+
        ]

        if any(mandatory_conditions):
            print(f"🚨 MANDATORY SELL TRIGGERED for {option_type}!")
            print(f"   💰 Token balance: {token_balance:.3f}")
            print(f"   📊 Market: ${best_bid:.3f}/${best_ask:.3f} (mid: ${market_mid:.3f})")
            return True

        return False

    def execute_mandatory_sell(self, option_type: str, market_data: Dict, token_balance: float) -> bool:
        """Execute mandatory market sell at $0.99"""

        token_id = self.get_token_id(option_type)
        if not token_id:
            return False

        # Get current market prices
        best_bid_data = market_data.get('best_bid', {})
        best_bid = best_bid_data.get('price', 0) if best_bid_data else 0

        # Use aggressive pricing: sell at current bid or slightly below
        sell_price = max(best_bid - 0.005, 0.985)  # Sell 0.5 cents below bid, minimum $0.985
        sell_quantity = max(1.0, token_balance * 0.98)  # Sell 98% of balance

        print(f"🚨 EXECUTING MANDATORY SELL:")
        print(f"   🎯 {option_type}: {sell_quantity:.3f} tokens @ ${sell_price:.3f}")
        print(f"   💵 Expected proceeds: ${sell_quantity * sell_price:.2f}")
        print(f"   ⚡ REASON: Market at $0.99 - taking profits before expiry!")

        try:
            order_args = OrderArgs(
                price=sell_price,
                size=sell_quantity,
                side=SELL,
                token_id=token_id
            )

            signed_order = self.client.create_order(order_args)
            response = self.client.post_order(signed_order)
            order_id = response.get('orderId', 'unknown')

            print(f"✅ MANDATORY SELL ORDER PLACED: {order_id}")
            print(f"🎉 Capturing profits before expiry!")
            return True

        except Exception as e:
            print(f"❌ MANDATORY SELL FAILED: {e}")
            return False

    def check_all_mandatory_sells(self):
        """Check and execute mandatory sells for all held tokens"""

        try:
            # Check CALL tokens
            if self.call_token_id:
                call_market_data = self.get_market_data('CALL')
                if call_market_data:
                    call_resp = self.client.get_balance_allowance(
                        params=BalanceAllowanceParams(
                            asset_type=AssetType.CONDITIONAL,
                            token_id=self.call_token_id
                        )
                    )
                    call_balance = int(call_resp.get('balance', 0)) / 10**6

                    if call_balance > 1.0:
                        if self.check_mandatory_sell_conditions('CALL', call_market_data, call_balance):
                            self.execute_mandatory_sell('CALL', call_market_data, call_balance)

            # Check PUT tokens
            if self.put_token_id:
                put_market_data = self.get_market_data('PUT')
                if put_market_data:
                    put_resp = self.client.get_balance_allowance(
                        params=BalanceAllowanceParams(
                            asset_type=AssetType.CONDITIONAL,
                            token_id=self.put_token_id
                        )
                    )
                    put_balance = int(put_resp.get('balance', 0)) / 10**6

                    if put_balance > 1.0:
                        if self.check_mandatory_sell_conditions('PUT', put_market_data, put_balance):
                            self.execute_mandatory_sell('PUT', put_market_data, put_balance)

        except Exception as e:
            print(f"❌ Error in mandatory sell check: {e}")
    """Enhanced ML trading executor with robust error handling"""
    def show_all_balances(self):
        """Show all current balances for debugging"""
        print(f"\n💎 CURRENT BALANCES:")
        print("=" * 30)

        try:
            # USDC balance (collateral)
            usdc_response = self.client.get_balance_allowance(
                params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            )
            usdc_balance = int(usdc_response.get('balance', 0)) / 10**6
            print(f"💵 USDC: ${usdc_balance:.2f}")

            # CALL token balance
            if self.call_token_id:
                call_response = self.client.get_balance_allowance(
                    params=BalanceAllowanceParams(
                        asset_type=AssetType.CONDITIONAL,
                        token_id=self.call_token_id
                    )
                )
                call_balance = int(call_response.get('balance', 0)) / 10**6
                print(f"📊 CALL tokens: {call_balance:.3f}")

            # PUT token balance
            if self.put_token_id:
                put_response = self.client.get_balance_allowance(
                    params=BalanceAllowanceParams(
                        asset_type=AssetType.CONDITIONAL,
                        token_id=self.put_token_id
                    )
                )
                put_balance = int(put_response.get('balance', 0)) / 10**6
                print(f"📊 PUT tokens: {put_balance:.3f}")

            print("=" * 30)

        except Exception as e:
            print(f"❌ Error checking balances: {e}")

    def check_balances_before_trade(self, token_id: str, action: str, quantity: float, price: float):
        """Check balances and adjust quantities before trading"""

        try:
            # Get token balance (conditional token)
            token_response = self.client.get_balance_allowance(
                params=BalanceAllowanceParams(
                    asset_type=AssetType.CONDITIONAL,
                    token_id=token_id
                )
            )
            token_balance = int(token_response.get('balance', 0)) / 10**6

            # Get USDC balance (collateral token) - FIXED: Don't specify token_id for USDC
            try:
                usdc_response = self.client.get_balance_allowance(
                    params=BalanceAllowanceParams(
                        asset_type=AssetType.COLLATERAL
                        # NO token_id for USDC - it's the default collateral
                    )
                )
                usdc_balance = int(usdc_response.get('balance', 0)) / 10**6
            except Exception as usdc_error:
                print(f"⚠️ Could not get USDC balance: {usdc_error}")
                usdc_balance = 0.0

            print(f"💰 Balances - Token: {token_balance:.2f}, USDC: ${usdc_balance:.2f}")

            if action == 'SELL':
                if token_balance < quantity:
                    if token_balance > 1.0:
                        # Adjust to available balance
                        adjusted_qty = token_balance * 0.95
                        print(f"🔧 Adjusting SELL: {quantity:.2f} → {adjusted_qty:.2f}")
                        return adjusted_qty, True
                    else:
                        print(f"🚫 Insufficient tokens for SELL: need {quantity:.2f}, have {token_balance:.2f}")
                        return 0, False

            elif action == 'BUY':
                needed_usdc = quantity * price
                if usdc_balance < needed_usdc:
                    if usdc_balance > 1.0:
                        # Adjust to available USDC
                        adjusted_qty = (usdc_balance * 0.95) / price
                        print(f"🔧 Adjusting BUY: {quantity:.2f} → {adjusted_qty:.2f}")
                        return max(1.0, adjusted_qty), True  # Minimum 1 token
                    else:
                        print(f"🚫 Insufficient USDC for BUY: need ${needed_usdc:.2f}, have ${usdc_balance:.2f}")
                        return 0, False

            return quantity, True

        except Exception as e:
            print(f"❌ Balance check error: {e}")
            return 0, False

    def __init__(self, config: MLTradingConfig):
        self.config = config
        self.client = self._setup_client()

        # ML Detection System
        self.ml_detector = MLArbitrageDetector(
            btc_file=config.btc_file,
            call_file=config.call_file,
            put_file=config.put_file
        )

        # Position tracking
        self.positions: Dict[str, MLPosition] = {}
        self.trade_history: List[Dict] = []
        self.total_pnl = 0.0

        # Rate limiting
        self.trades_this_hour = 0
        self.last_hour_reset = time.time()

        # Token management
        self.call_token_id = None
        self.put_token_id = None
        self._update_token_ids()

        # Performance tracking
        self.total_opportunities = 0
        self.executed_opportunities = 0

        print("🤖 IMPROVED ML Trading Executor initialized")
        print(f"📊 ML Model trained: {self.ml_detector.models_trained}")
        print(f"💰 Max position: ${config.max_position_size_usd}")
        print(f"🎯 Min confidence: {config.min_confidence_threshold:.1%}")
        print(f"⚡ Price buffer: {config.price_buffer_pct:.1%}")

    def _setup_client(self):
        """Initialize CLOB client with error handling"""
        try:
            creds = ApiCreds(
                api_key=self.config.api_key,
                api_secret=self.config.api_secret,
                api_passphrase=self.config.api_passphrase
            )
            client = ClobClient(
                self.config.clob_api_url,
                key=self.config.private_key,
                chain_id=self.config.chain_id,
                creds=creds
            )
            print("✅ CLOB client initialized")
            return client
        except Exception as e:
            print(f"❌ Failed to initialize CLOB client: {e}")
            raise

    def _update_token_ids(self):
        """Update token IDs with validation"""
        try:
            # Get CALL token ID
            if os.path.exists(self.config.call_file):
                with open(self.config.call_file, 'r') as f:
                    call_data = json.load(f)
                    self.call_token_id = call_data.get('asset_id')

            # Get PUT token ID
            if os.path.exists(self.config.put_file):
                with open(self.config.put_file, 'r') as f:
                    put_data = json.load(f)
                    self.put_token_id = put_data.get('asset_id')

            if self.call_token_id and self.put_token_id:
                print(f"📋 Token IDs updated: CALL={self.call_token_id[:8]}..., PUT={self.put_token_id[:8]}...")
            else:
                print("⚠️ Warning: Could not update token IDs")

        except Exception as e:
            print(f"❌ Error updating token IDs: {e}")

    def get_token_id(self, option_type: str) -> Optional[str]:
        """Get token ID for option type"""
        if option_type == 'CALL':
            return self.call_token_id
        elif option_type == 'PUT':
            return self.put_token_id
        return None

    def get_balance(self, token_id: str) -> Tuple[int, float]:
        """Get token balance with better error handling"""
        try:
            response = self.client.get_balance_allowance(
                params=BalanceAllowanceParams(
                    asset_type=AssetType.CONDITIONAL,
                    token_id=token_id
                )
            )
            balance_raw = int(response.get('balance', 0))
            balance_tokens = balance_raw / 10**6
            return balance_raw, balance_tokens

        except Exception as e:
            print(f"❌ Error getting balance for {token_id[:8]}...: {e}")
            return 0, 0.0

    def get_market_data(self, option_type: str) -> Optional[Dict]:
        """Get current market data with validation"""
        try:
            if option_type == 'CALL':
                file_path = self.config.call_file
            elif option_type == 'PUT':
                file_path = self.config.put_file
            else:
                print(f"❌ Invalid option_type: {option_type}")
                return None

            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                return None

            with open(file_path, 'r') as f:
                data = json.load(f)

            # DEBUG: Print what we loaded
            print(f"🐛 Raw data for {option_type}: {data}")

            # Validate required fields
            if not data:
                print(f"❌ Empty data for {option_type}")
                return None

            best_bid = data.get('best_bid')
            best_ask = data.get('best_ask')

            if not best_bid or not best_ask:
                print(f"❌ Missing bid/ask data for {option_type}")
                print(f"   best_bid: {best_bid}")
                print(f"   best_ask: {best_ask}")
                return None

            # Check if price fields exist
            if not best_bid.get('price') or not best_ask.get('price'):
                print(f"❌ Missing price fields for {option_type}")
                print(f"   bid price: {best_bid.get('price')}")
                print(f"   ask price: {best_ask.get('price')}")
                return None

            # Add derived fields safely
            bid_price = best_bid['price']
            ask_price = best_ask['price']
            data['mid_price'] = (bid_price + ask_price) / 2
            data['spread'] = ask_price - bid_price
            data['spread_pct'] = data['spread'] / data['mid_price'] if data['mid_price'] > 0 else 0

            return data

        except Exception as e:
            print(f"❌ Error getting market data for {option_type}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def calculate_position_size(self, opportunity: Dict, target_price: float) -> float:
        """Improved position sizing with better validation"""

        if target_price <= 0:
            return 0

        # Base position size from configuration
        base_usd_amount = self.config.max_position_size_usd

        # Adjust by ML confidence (minimum 0.5x, maximum 1.0x)
        confidence = opportunity.get('confidence', 0.5)
        confidence_multiplier = max(0.5, min(1.0, confidence / self.config.min_confidence_threshold))

        # Adjust by expected profit potential
        profit_potential = opportunity.get('profit_potential', 0)
        profit_multiplier = max(0.3, min(1.0, profit_potential / self.config.min_profit_threshold))

        # Adjust by market regime
        regime = opportunity.get('market_regime', 'normal')
        if regime == 'quiet':
            regime_multiplier = 1.1  # Slightly increase in quiet markets
        elif regime == 'volatile':
            regime_multiplier = 0.7  # Decrease in volatile markets
        else:
            regime_multiplier = 1.0

        # Calculate final USD amount
        adjusted_usd = base_usd_amount * confidence_multiplier * profit_multiplier * regime_multiplier

        # Convert to token quantity (FIXED: proper conversion)
        token_quantity = adjusted_usd / target_price

        # Apply constraints
        token_quantity = max(token_quantity, self.config.min_order_size)
        token_quantity = min(token_quantity, base_usd_amount / target_price * 2)  # Max 2x base size

        return round(token_quantity, 2)

    def calculate_target_price(self, opportunity: Dict, market_data: Dict) -> Optional[float]:
        """Calculate target price with proper validation"""
        action = opportunity['action']
        best_bid = market_data['best_bid']['price']
        best_ask = market_data['best_ask']['price']
        mid_price = market_data['mid_price']

        # Calculate buffer amount
        buffer_amount = mid_price * self.config.price_buffer_pct

        if action == 'BUY':
            # For BUY orders, start from ask and add small buffer
            target_price = best_ask + buffer_amount
        else:  # SELL
            # For SELL orders, start from bid and subtract small buffer
            target_price = best_bid - buffer_amount

        # Validate target price
        if target_price <= 0:
            print(f"⚠️ Invalid target price calculated: {target_price}")
            return None

        # Check for excessive price impact
        price_impact = abs(target_price - mid_price) / mid_price
        if price_impact > self.config.max_price_impact:
            print(f"⚠️ Price impact too high: {price_impact:.2%}")
            return None

        return round(target_price, 4)

    def check_risk_limits(self) -> Tuple[bool, str]:
        """Enhanced risk limit checking"""

        # Check hourly trade limit
        current_time = time.time()
        if current_time - self.last_hour_reset > 3600:
            self.trades_this_hour = 0
            self.last_hour_reset = current_time

        if self.trades_this_hour >= self.config.max_trades_per_hour:
            return False, "hourly_limit"

        # Check total exposure
        total_exposure = 0
        active_positions = 0
        for pos in self.positions.values():
            if pos.status in ['pending', 'filled']:
                total_exposure += abs(pos.entry_price * pos.quantity)
                active_positions += 1

        if total_exposure >= self.config.max_total_exposure_usd:
            return False, f"exposure_limit (${total_exposure:.2f})"

        # Check number of active positions
        if active_positions >= 3:  # Max 3 active positions
            return False, "position_limit"

        return True, ""

    def execute_ml_opportunity(self, opportunity: Dict) -> Optional[str]:
        """Execute ML arbitrage opportunity"""

        print(f"\n🐛 DEBUG: Executing opportunity: {opportunity}")

        option_type = opportunity.get('type')
        action = opportunity.get('action')

        print(f"🐛 DEBUG: option_type={option_type}, action={action}")

        # Get token ID
        token_id = self.get_token_id(option_type)
        print(f"🐛 DEBUG: token_id={token_id}")

        if not token_id:
            print(f"❌ No token ID for {option_type}")
            return None

        # Get market data
        market_data = self.get_market_data(option_type)
        print(f"🐛 DEBUG: market_data={market_data}")

        if not market_data:
            print(f"❌ No market data for {option_type}")
            return None

        # Continue with rest of method...



        self.total_opportunities += 1

        option_type = opportunity['type']
        action = opportunity['action']
        confidence = opportunity.get('confidence', 0)

        print(f"\n🎯 ML OPPORTUNITY #{self.total_opportunities}")
        print(f"   📊 {option_type} {action} | Confidence: {confidence:.1%}")
        print(f"   💰 Expected profit: ${opportunity.get('profit_potential', 0):.3f}")
        print(f"   🌍 Market regime: {opportunity.get('market_regime', 'unknown')}")

        # Check confidence threshold
        if confidence < self.config.min_confidence_threshold:
            print(f"   🚫 SKIPPED: Low confidence ({confidence:.1%} < {self.config.min_confidence_threshold:.1%})")
            return None

        # Check risk limits
        can_trade, limit_reason = self.check_risk_limits()
        if not can_trade:
            print(f"   🚫 SKIPPED: Risk limit - {limit_reason}")
            return None

        # Get token ID
        token_id = self.get_token_id(option_type)
        if not token_id:
            print(f"   ❌ SKIPPED: No token ID for {option_type}")
            return None

        # Get market data
        market_data = self.get_market_data(option_type)
        if not market_data:
            print(f"   ❌ SKIPPED: No market data for {option_type}")
            return None

        # Calculate target price
        target_price = self.calculate_target_price(opportunity, market_data)
        if target_price is None:
            print(f"   ❌ SKIPPED: Invalid target price")
            return None

        # Calculate position size
        quantity = self.calculate_position_size(opportunity, target_price)
        if quantity < self.config.min_order_size:
            print(f"   🚫 SKIPPED: Position too small ({quantity:.2f})")
            return None

        # Check balance for SELL orders
        if action == 'SELL':
            balance_raw, balance_tokens = self.get_balance(token_id)
            if balance_tokens < quantity:
                print(f"   🚫 SKIPPED: Insufficient balance ({balance_tokens:.2f} < {quantity:.2f})")
                return None

        # Calculate trade size in USD
        trade_size_usd = quantity * target_price

        try:

            # BEFORE executing the order, check and adjust balances
            adjusted_quantity, can_proceed = self.check_balances_before_trade(
                token_id, action, quantity, target_price
            )

            if not can_proceed:
                print(f"   🚫 SKIPPED: Balance/allowance insufficient")
                return None

            # Use adjusted quantity
            quantity = adjusted_quantity
            trade_size_usd = quantity * target_price


            print(f"   ✅ EXECUTING new: {quantity:.2f} tokens @ ${target_price:.4f} (${trade_size_usd:.2f})")

            # Create order
            order_args = OrderArgs(
                price=target_price,
                size=quantity,
                side=BUY if action == 'BUY' else SELL,
                token_id=token_id
            )

            signed_order = self.client.create_order(order_args)
            response = self.client.post_order(signed_order)

            order_id = response.get('orderId', 'unknown')

            # Create position tracking
            position_key = f"{option_type}_{order_id}"
            position = MLPosition(
                opportunity=opportunity,
                token_id=token_id,
                entry_price=target_price,
                quantity=quantity,
                timestamp=time.time()
            )

            self.positions[position_key] = position
            self.trades_this_hour += 1
            self.executed_opportunities += 1

            print(f"   🎉 ORDER PLACED: {order_id}")
            print(f"   📈 Execution rate: {self.executed_opportunities}/{self.total_opportunities} ({self.executed_opportunities/self.total_opportunities:.1%})")

            return order_id

        except Exception as e:
            print(f"   ❌ EXECUTION FAILED: {e}")
            return None

    def update_positions(self):
        """Enhanced position management with better status tracking"""
        current_time = time.time()
        positions_to_close = []

        for position_key, position in self.positions.items():
            if position.status in ['cancelled', 'closed']:
                continue

            try:
                # Get current market data
                market_data = self.get_market_data(position.option_type)
                if not market_data:
                    continue

                current_price = market_data['mid_price']

                # Simulate order fill (in reality, check actual order status)
                if position.status == 'pending':
                    # Assume filled after 5 seconds for demo
                    if current_time - position.timestamp > 5:
                        position.status = 'filled'
                        position.fill_price = position.entry_price
                        position.fill_quantity = position.quantity
                        print(f"✅ FILLED: {position.option_type} {position.action} @ ${position.fill_price:.4f}")

                # Risk management for filled positions
                if position.status == 'filled':
                    should_close, reason = position.should_close(current_price, self.config)

                    if should_close:
                        positions_to_close.append((position_key, position, current_price, reason))

            except Exception as e:
                print(f"❌ Error updating position {position_key}: {e}")

        # Close positions that need to be closed
        for position_key, position, exit_price, reason in positions_to_close:
            self._close_position(position_key, position, exit_price, reason)

    def _close_position(self, position_key: str, position: MLPosition,
                       exit_price: float, reason: str):
        """Enhanced position closing with comprehensive tracking"""
        try:
            # Calculate final PnL
            position.exit_price = exit_price
            position.exit_timestamp = time.time()
            position.realized_pnl = position.get_current_pnl(exit_price)
            position.status = 'closed'

            self.total_pnl += position.realized_pnl

            # Create detailed trade record
            trade_record = {
                'timestamp': position.timestamp,
                'exit_timestamp': position.exit_timestamp,
                'option_type': position.option_type,
                'action': position.action,
                'entry_price': position.fill_price,
                'exit_price': exit_price,
                'quantity': position.fill_quantity,
                'pnl': position.realized_pnl,
                'pnl_pct': (position.realized_pnl / (position.fill_price * position.fill_quantity)) * 100,
                'hold_time_minutes': position.get_hold_time_minutes(),
                'ml_confidence': position.ml_confidence,
                'expected_profit': position.expected_profit,
                'market_regime': position.market_regime,
                'close_reason': reason,
                'trade_size_usd': position.fill_price * position.fill_quantity
            }

            self.trade_history.append(trade_record)

            # Enhanced logging
            pnl_emoji = "💚" if position.realized_pnl > 0 else "💔"
            print(f"\n{pnl_emoji} POSITION CLOSED: {reason.upper()}")
            print(f"   📊 {position.option_type} {position.action}: ${position.fill_price:.4f} → ${exit_price:.4f}")
            print(f"   💰 PnL: ${position.realized_pnl:+.4f} ({trade_record['pnl_pct']:+.1f}%)")
            print(f"   ⏱️ Hold time: {position.get_hold_time_minutes():.1f}m")
            print(f"   🎯 Total PnL: ${self.total_pnl:+.4f}")

        except Exception as e:
            print(f"❌ Error closing position {position_key}: {e}")

    def get_performance_stats(self) -> Dict:
        """Get trading performance statistics"""
        if not self.trade_history:
            return {
                'total_opportunities': getattr(self, 'total_opportunities', 0),
                'executed_opportunities': getattr(self, 'executed_opportunities', 0),
                'execution_rate': 0,
                'total_trades': 0,           # ADD THIS LINE
                'profitable_trades': 0,      # ADD THIS LINE
                'win_rate': 0,              # ADD THIS LINE
                'total_pnl': 0,             # ADD THIS LINE
                'avg_pnl_per_trade': 0,     # ADD THIS LINE
                'avg_pnl_pct': 0,           # ADD THIS LINE
                'avg_hold_time_minutes': 0, # ADD THIS LINE
                'best_trade_pnl': 0,        # ADD THIS LINE
                'worst_trade_pnl': 0,       # ADD THIS LINE
                'active_positions': len([p for p in self.positions.values() if p.status in ['pending', 'filled']])
            }

    def print_status_line(self, opportunities: List[Dict]):
        """Enhanced status line with more information"""
        stats = self.get_performance_stats()
        active_pos = stats['active_positions']

        status = (f"\r{datetime.now().strftime('%H:%M:%S')} | "
                 f"ML: {'✅' if self.ml_detector.models_trained else '❌'} | "
                 f"Opps: {len(opportunities)} | "
                 f"Exec: {stats['executed_opportunities']}/{stats['total_opportunities']} "
                 f"({stats['execution_rate']:.1%}) | "
                 f"Active: {active_pos} | "
                 f"PnL: ${self.total_pnl:+.3f}")

        if stats['total_trades'] > 0:
            status += f" | Win: {stats['win_rate']:.1%} ({stats['profitable_trades']}/{stats['total_trades']})"

        print(status, end='', flush=True)

    def run_trading_loop(self):
        """Enhanced main trading loop"""
        print("🚀 Starting IMPROVED ML Trading Loop")
        print("=" * 80)

        iteration = 0
        last_stats_time = 0

        # In your run_trading_loop method, add this every 100 iterations:
        if iteration % 100 == 0:
            self.show_all_balances()

        # Add this inside your main trading loop:
        if iteration % 5 == 0:  # Check every 5 iterations (5 seconds)
            self.check_all_mandatory_sells()

        try:
            while True:
                iteration += 1

                # Update token IDs periodically
                if iteration % 100 == 0:
                    self._update_token_ids()

                # Get ML opportunities
                opportunities = self.ml_detector.analyze_market_data()

                # Execute opportunities
                for opportunity in opportunities:
                    order_id = self.execute_ml_opportunity(opportunity)
                    if order_id:
                        time.sleep(2)  # Brief pause between orders

                # Update existing positions
                self.update_positions()

                # Print status line
                if iteration % 10 == 0:  # Every 10 iterations
                    self.print_status_line(opportunities)

                # Detailed performance report every 5 minutes
                current_time = time.time()
                if current_time - last_stats_time > 300:
                    stats = self.get_performance_stats()
                    print(f"\n\n📊 PERFORMANCE REPORT (Iteration {iteration})")
                    print(f"🎯 Opportunities: {stats['executed_opportunities']}/{stats['total_opportunities']} executed ({stats['execution_rate']:.1%})")
                    if stats['total_trades'] > 0:
                        print(f"📈 Trading: {stats['win_rate']:.1%} win rate, ${stats['avg_pnl_per_trade']:+.4f} avg PnL")
                        print(f"💰 Total PnL: ${stats['total_pnl']:+.4f}")
                        print(f"⏱️ Avg hold time: {stats['avg_hold_time_minutes']:.1f}m")
                    print(f"🔄 Active positions: {stats['active_positions']}")
                    print("=" * 60)
                    last_stats_time = current_time

                time.sleep(1)  # 1 second loop

        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down Improved ML Trading Executor...")
            self._shutdown_gracefully()

    def _shutdown_gracefully(self):
        """Enhanced graceful shutdown"""
        print("📊 Closing all open positions...")

        # Close all open positions
        for position in self.positions.values():
            if position.status == 'filled':
                try:
                    market_data = self.get_market_data(position.option_type)
                    if market_data:
                        current_price = market_data['mid_price']
                        self._close_position(
                            f"{position.option_type}_{position.timestamp}",
                            position, current_price, "shutdown"
                        )
                except Exception as e:
                    print(f"❌ Error closing position during shutdown: {e}")

        # Final performance report
        stats = self.get_performance_stats()
        print("\n" + "="*80)
        print("📊 FINAL PERFORMANCE REPORT")
        print("="*80)

        if stats['total_trades'] > 0:
            print(f"🎯 Opportunity Execution: {stats['executed_opportunities']}/{stats['total_opportunities']} ({stats['execution_rate']:.1%})")
            print(f"📈 Total Trades: {stats['total_trades']}")
            print(f"🏆 Win Rate: {stats['win_rate']:.1%} ({stats['profitable_trades']}/{stats['total_trades']})")
            print(f"💰 Total PnL: ${stats['total_pnl']:+.4f}")
            print(f"📊 Avg PnL/Trade: ${stats['avg_pnl_per_trade']:+.4f} ({stats['avg_pnl_pct']:+.1f}%)")
            print(f"⏱️ Avg Hold Time: {stats['avg_hold_time_minutes']:.1f} minutes")
            print(f"🚀 Best Trade: ${stats['best_trade_pnl']:+.4f}")
            print(f"💔 Worst Trade: ${stats['worst_trade_pnl']:+.4f}")
        else:
            print("No trades completed during this session")

        print("="*80)
        print("✅ Shutdown complete")

def main():
    """Main function with improved configuration"""

    # Enhanced configuration with conservative defaults
    config = MLTradingConfig(
        clob_api_url=os.getenv("CLOB_API_URL", "https://clob.polymarket.com"),
        private_key=os.getenv("PK", ""),
        api_key=os.getenv("CLOB_API_KEY", ""),
        api_secret=os.getenv("CLOB_SECRET", ""),
        api_passphrase=os.getenv("CLOB_PASS_PHRASE", ""),

        # Conservative parameters for improved safety
        max_position_size_usd=5.0,       # Small position size
        max_total_exposure_usd=15.0,     # Limited total exposure
        min_confidence_threshold=0.80,   # High confidence requirement
        min_profit_threshold=0.030,      # Higher profit threshold
        max_trades_per_hour=3,           # Very conservative rate

        # Tighter risk management
        stop_loss_threshold=-0.025,      # 2.5 cent stop loss
        take_profit_threshold=0.075,     # 7.5 cent take profit
        max_hold_time_minutes=15,        # Shorter holds

        # Improved execution parameters
        price_buffer_pct=0.002,          # 0.2% price buffer
        max_price_impact=0.03,           # 3% max price impact
    )

    # Validate configuration
    if not all([config.clob_api_url, config.private_key, config.api_key,
                config.api_secret, config.api_passphrase]):
        print("❌ Missing API credentials. Please set environment variables:")
        print("   CLOB_API_URL, PK, CLOB_API_KEY, CLOB_SECRET, CLOB_PASS_PHRASE")
        return

    print("🎯 Starting Improved ML Trading System")
    print("🔒 Conservative configuration loaded for safety")

    # Start improved trading
    executor = MLTradingExecutor(config)
    executor.run_trading_loop()

if __name__ == "__main__":
    main()
