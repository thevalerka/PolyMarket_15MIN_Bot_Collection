#!/usr/bin/env python3
"""
Bot016_MarketMaker - Binary Options Market Making Strategy
Provides liquidity by placing limit orders on both sides of the book
Buy at bid - $0.01, Sell at ask + $0.01
Quick reaction to BTC price movements
Max 12 shares per side, stop loss on imbalanced positions
Uses OVH39 credentials
"""

import json
import time
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, List
from collections import deque
from dataclasses import dataclass, asdict
import logging
import requests

# Import Polymarket trading core
sys.path.insert(0, '/home/ubuntu')
from polymarket_trading_core_debug import PolymarketTrader, load_credentials_from_env
from sensitivity_api import SensitivityAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# File paths

PUT_FILE = "/home/ubuntu/013_2025_polymarket/15M_BTC_PUT_rest.json"
CALL_FILE = "/home/ubuntu/013_2025_polymarket/15M_BTC_CALL_rest.json"
BTC_FILE = "/home/ubuntu/013_2025_polymarket/bot016_react/coinbase_btc_price.json"
BTC_FILE_COINBASE = "/home/ubuntu/013_2025_polymarket/bot016_react/coinbase_btc_price.json"
BTC_FILE_CHAINLINK = "/home/ubuntu/013_2025_polymarket/chainlink_btc_price.json"
SENSITIVITY_FILE = "/home/ubuntu/013_2025_polymarket/bot016_react/sensitivity_data/sensitivity_transformed.json"
TRADES_DIR = "/home/ubuntu/013_2025_polymarket/bot016_react/mm_trades"

# Credentials
CREDENTIALS_ENV = '/home/ubuntu/013_2025_polymarket/keys/keys_ovh39.env'

# Market making parameters
SENS_MULTIPLIER = 1.2  # Sensitivity multiplier for signal strength
SIGNAL_THRESHOLD = 0.01  # Reposition if signal > $0.01 option price movement
BUY_OFFSET = 0.02  # Buy at bid - $0.01
SELL_OFFSET = 0.01  # Sell at ask + $0.01
MAX_SHARES_PER_SIDE = 12  # Maximum 12 shares per side
STOP_LOSS = 0.20  # Stop loss at -$0.20/share for imbalanced positions
MIN_SPREAD = 0.03  # Minimum spread required to place orders
BUFFER_SECONDS = 30  # No trading in first 30s of period
POSITION_CHECK_INTERVAL = 30  # Check positions every 30s
REPOSITION_DELAY = 2.0  # Wait 1 second after cancel before repositioning


@dataclass
class Position:
    """Position tracker"""
    token_type: str  # 'PUT' or 'CALL'
    token_id: str
    entry_price: float
    entry_time: float
    quantity: float


@dataclass
class OpenOrder:
    """Track open orders"""
    order_id: str
    token_type: str  # 'PUT' or 'CALL'
    side: str  # 'BUY' or 'SELL'
    price: float
    quantity: float
    token_id: str


def read_json(filepath: str) -> Optional[dict]:
    """Read JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        return None


def calculate_btc_volatility(price_history: deque) -> float:
    """Calculate BTC volatility over last minute (price range)"""
    if len(price_history) < 10:
        return 0.0
    prices = list(price_history)
    return max(prices) - min(prices)


class MarketMakerBot:
    """Market maker for 15-minute BTC binary options"""

    def __init__(self, credentials: dict):
        """Initialize market maker bot"""
        self.trader = PolymarketTrader(
            clob_api_url=credentials['clob_api_url'],
            private_key=credentials['private_key'],
            api_key=credentials['api_key'],
            api_secret=credentials['api_secret'],
            api_passphrase=credentials['api_passphrase']
        )

        # Position tracking
        self.positions: List[Position] = []

        # Order tracking
        self.open_orders: List[OpenOrder] = []

        # Asset IDs
        self.current_put_id: Optional[str] = None
        self.current_call_id: Optional[str] = None
        self.previous_put_id: Optional[str] = None
        self.previous_call_id: Optional[str] = None

        # Trades
        self.trades_dir = Path(TRADES_DIR)
        self.trades_dir.mkdir(exist_ok=True, parents=True)
        self.today_trades = []
        self.load_today_trades()

        # Load sensitivity API
        self.sensitivity_api = SensitivityAPI(SENSITIVITY_FILE)

        # Price history
        self.btc_price_history = deque(maxlen=600)

        # Period tracking
        self.current_period_start: Optional[int] = None
        self.strike_price: Optional[float] = None
        self.last_strike_check = 0  # Track last strike price check
        self.buffer_check_done = False  # Track if we checked at end of buffer

        # Last known prices for signal calculation and entry prices
        self.last_btc_price: Optional[float] = None
        self.last_call_ask: Optional[float] = None
        self.last_call_bid: Optional[float] = None
        self.last_put_ask: Optional[float] = None
        self.last_put_bid: Optional[float] = None

        # Last buy order prices (for stop loss entry price)
        self.last_call_buy_price: float = 0.50  # Default to mid
        self.last_put_buy_price: float = 0.50  # Default to mid

        # Timing
        self.last_position_check = time.time()

        self.last_asset_reload = time.time()
        self.last_cancel_time = 0  # Track when we last cancelled for repositioning delay

        # USDC balance cache
        self.cached_usdc_balance = 0
        self.last_usdc_check = 0

        self.print_startup_info()

    def print_startup_info(self):
        """Print startup information"""
        logger.info("="*80)
        logger.info("🏦 BOT016_MARKETMAKER - Liquidity Provider")
        logger.info("="*80)
        logger.info("📡 BTC Price: Coinbase WebSocket")
        logger.info("📡 CALL/PUT Prices: REST API (via JSON files)")
        logger.info(f"🔮 Sensitivity Predictor: {len(self.sensitivity_api.predictor.bins_data)} bins loaded")
        logger.info(f"💰 Buy: bid - ${BUY_OFFSET:.2f}")
        logger.info(f"💰 Sell: ask + ${SELL_OFFSET:.2f}")
        logger.info(f"🛑 Stop Loss: -${STOP_LOSS:.2f}/share (imbalanced only)")
        logger.info(f"🔑 Credentials: OVH39")
        logger.info(f"Max Per Side: {MAX_SHARES_PER_SIDE} shares")
        logger.info(f"Min Spread: ${MIN_SPREAD:.2f}")
        logger.info(f"Signal Threshold: ${SIGNAL_THRESHOLD:.2f} option movement")
        logger.info(f"Sens Multiplier: {SENS_MULTIPLIER}x")
        logger.info(f"Buffer Zone: {BUFFER_SECONDS}s (first 30s of each period)")
        logger.info("="*80)

    def get_today_filename(self) -> Path:
        """Get filename for today's trades"""
        today = date.today().strftime('%Y%m%d')
        return self.trades_dir / f"mm_trades_{today}.json"

    def load_today_trades(self):
        """Load today's trades from file"""
        try:
            filename = self.get_today_filename()
            if filename.exists():
                with open(filename, 'r') as f:
                    self.today_trades = json.load(f)
                logger.info(f"📁 Loaded {len(self.today_trades)} trades from {filename.name}")
        except Exception as e:
            logger.error(f"❌ Error loading trades: {e}")
            self.today_trades = []

    def save_trade(self, trade: dict):
        """Save a trade to file"""
        try:
            self.today_trades.append(trade)
            filename = self.get_today_filename()
            with open(filename, 'w') as f:
                json.dump(self.today_trades, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Error saving trade: {e}")

    def reload_asset_ids(self):
        """Reload PUT and CALL asset IDs from data files"""
        put_data = read_json(PUT_FILE)
        call_data = read_json(CALL_FILE)

        if put_data and call_data:
            new_put_id = put_data.get('asset_id')
            new_call_id = call_data.get('asset_id')

            put_changed = new_put_id != self.current_put_id
            call_changed = new_call_id != self.current_call_id

            if put_changed or call_changed:
                logger.info(f"   🔄 Asset IDs updated:")
                if put_changed:
                    logger.info(f"   PUT:  ...{new_put_id[-12:]}")
                    self.previous_put_id = self.current_put_id
                if call_changed:
                    logger.info(f"   CALL: ...{new_call_id[-12:]}")
                    self.previous_call_id = self.current_call_id

                self.current_put_id = new_put_id
                self.current_call_id = new_call_id

            self.last_asset_reload = time.time()


    def refresh_usdc_balance(self):
        """Refresh cached USDC balance with timeout"""

        if time.time() - self.last_usdc_check >= 10:
            try:
                from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
                import signal

                # Set 2 second timeout
                def timeout_handler(signum, frame):
                    raise TimeoutError("USDC balance check timeout")

                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(2)  # 2 second timeout

                try:
                    response = self.trader.client.get_balance_allowance(
                        params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
                    )
                    balance_raw = int(response.get('balance', 0))
                    self.cached_usdc_balance = balance_raw / 10**6
                    self.last_usdc_check = time.time()
                finally:
                    signal.alarm(0)  # Cancel alarm
                    signal.signal(signal.SIGALRM, old_handler)

            except TimeoutError:
                logger.warning(f"⚠️  USDC balance check timed out")
                self.last_usdc_check = time.time()  # Don't retry immediately
            except Exception as e:
                logger.error(f"❌ Error checking USDC balance: {e}")
                self.last_usdc_check = time.time()  # Don't retry immediately

    def check_token_balance(self, token_id: str) -> float:
        """Get token balance"""
        try:
            _, balance = self.trader.get_token_balance(token_id)
            return balance
        except:
            return 0.0

    def update_strike_price(self):
        """Update strike price from chainlink data"""
        try:
            btc_data_chainlink = read_json(BTC_FILE_CHAINLINK)
            if btc_data_chainlink:
                new_strike = btc_data_chainlink.get('strike')
                if new_strike:
                    old_strike = self.strike_price
                    self.strike_price = new_strike
                    self.last_strike_check = time.time()

                    if old_strike and abs(new_strike - old_strike) > 0.01:
                        logger.info(f"📊 Strike updated: ${old_strike:.2f} → ${new_strike:.2f}")
                    elif not old_strike:
                        logger.info(f"✅ Strike: ${self.strike_price:.2f}")

                    return True
        except Exception as e:
            logger.error(f"❌ Error updating strike price: {e}")
        return False

    def verify_positions_from_wallet(self):
        """Verify positions against wallet balances - sync immediately"""
        try:
            synced = False

            # Check CALL balance
            if self.current_call_id:
                call_balance = self.check_token_balance(self.current_call_id)
                call_positions = [p for p in self.positions if p.token_type == 'CALL']
                tracked_call = sum(p.quantity for p in call_positions)

                if abs(call_balance - tracked_call) > 0.5:
                    logger.info(f"🔄 CALL sync: wallet={call_balance:.2f} tracked={tracked_call:.2f} → syncing to wallet")
                    # Sync to wallet immediately
                    self.positions = [p for p in self.positions if p.token_type != 'CALL']
                    if call_balance >= 1:
                        # Use last buy order price or last market price
                        entry_price = self.last_call_buy_price if self.last_call_buy_price > 0 else (self.last_call_ask if self.last_call_ask else 0.50)
                        self.positions.append(Position(
                            token_type='CALL',
                            token_id=self.current_call_id,
                            entry_price=entry_price,
                            entry_time=time.time(),
                            quantity=call_balance
                        ))
                        logger.info(f"   → Created CALL position: {call_balance:.2f} shares @ ${entry_price:.2f}")
                    else:
                        logger.info(f"   → Cleared CALL positions (wallet empty)")
                    synced = True

            # Check PUT balance
            if self.current_put_id:
                put_balance = self.check_token_balance(self.current_put_id)
                put_positions = [p for p in self.positions if p.token_type == 'PUT']
                tracked_put = sum(p.quantity for p in put_positions)

                if abs(put_balance - tracked_put) > 0.5:
                    logger.info(f"🔄 PUT sync: wallet={put_balance:.2f} tracked={tracked_put:.2f} → syncing to wallet")
                    # Sync to wallet immediately
                    self.positions = [p for p in self.positions if p.token_type != 'PUT']
                    if put_balance >= 1:
                        # Use last buy order price or last market price
                        entry_price = self.last_put_buy_price if self.last_put_buy_price > 0 else (self.last_put_ask if self.last_put_ask else 0.50)
                        self.positions.append(Position(
                            token_type='PUT',
                            token_id=self.current_put_id,
                            entry_price=entry_price,
                            entry_time=time.time(),
                            quantity=put_balance
                        ))
                        logger.info(f"   → Created PUT position: {put_balance:.2f} shares @ ${entry_price:.2f}")
                    else:
                        logger.info(f"   → Cleared PUT positions (wallet empty)")
                    synced = True

            if synced:
                logger.info(f"✅ Position sync complete - continuing operation")

            self.last_position_check = time.time()

        except Exception as e:
            logger.error(f"❌ Error verifying positions: {e}")
            import traceback
            traceback.print_exc()

    def cancel_all_orders(self):
        """Cancel all open orders and clear tracking"""
        try:
            self.trader.cancel_all_orders()
            self.open_orders = []
            logger.info("🗑️  All orders cancelled")
        except Exception as e:
            logger.error(f"❌ Error cancelling orders: {e}")

    def place_buy_order(self, token_type: str, token_id: str, price: float, quantity: float) -> Optional[str]:
        """Place limit buy order"""
        try:
            order_id = self.trader.place_buy_order_limit(token_id, price, quantity)

            if order_id:
                self.open_orders.append(OpenOrder(
                    order_id=order_id,
                    token_type=token_type,
                    side='BUY',
                    price=price,
                    quantity=quantity,
                    token_id=token_id
                ))

                # Track last buy order price for entry price estimation
                if token_type == 'CALL':
                    self.last_call_buy_price = price
                elif token_type == 'PUT':
                    self.last_put_buy_price = price

                logger.info(f"✅ BUY {token_type}: {quantity} @ ${price:.2f} | ID: {order_id[:12]}...")
                return order_id

        except Exception as e:
            logger.error(f"❌ Error placing buy order: {e}")
        return None

    def place_sell_order(self, token_type: str, token_id: str, price: float, quantity: float) -> Optional[str]:
        """Place limit sell order"""
        try:
            order_id = self.trader.place_sell_order_limit(token_id, price, quantity)

            if order_id:
                self.open_orders.append(OpenOrder(
                    order_id=order_id,
                    token_type=token_type,
                    side='SELL',
                    price=price,
                    quantity=quantity,
                    token_id=token_id
                ))
                logger.info(f"✅ SELL {token_type}: {quantity} @ ${price:.2f} | ID: {order_id[:12]}...")
                return order_id

        except Exception as e:
            logger.error(f"❌ Error placing sell order: {e}")
        return None

    def check_signal_for_reposition(self, btc_price: float, btc_delta: float,
                                     call_ask: float, call_bid: float, call_ask_movement: float, call_bid_movement: float,
                                     put_ask: float, put_bid: float, put_ask_movement: float, put_bid_movement: float,
                                     seconds_remaining: float, volatility: float) -> bool:
        """Check if sensitivity signals indicate we should reposition orders

        Returns True if orders should be cancelled and repositioned
        """
        try:
            #print("H ", self.strike_price, " OR ", btc_delta)
            if not self.strike_price or btc_delta is None:
                return False

            # Calculate distance and get sensitivity
            distance = abs(btc_price - self.strike_price)

            prediction = self.sensitivity_api.get_sensitivity(
                btc_price=btc_price,
                strike_price=self.strike_price,
                time_to_expiry_seconds=seconds_remaining,
                volatility_percent=volatility
            )

            call_sens = prediction['sensitivity']['call']
            put_sens = prediction['sensitivity']['put']

            # Calculate ideal movements
            ideal_call_movement = btc_delta * call_sens * SENS_MULTIPLIER
            ideal_put_movement = btc_delta * put_sens * SENS_MULTIPLIER

            print("signal: ",ideal_call_movement," - ", ideal_put_movement)

            should_reposition = False

            # Check CALL signals
            if abs(call_sens) > 0.000001 and abs(ideal_call_movement) > SIGNAL_THRESHOLD:
                # Check if ideal movement exceeds actual movement
                if abs(ideal_call_movement) > abs(call_ask_movement) or abs(ideal_call_movement) > abs(call_bid_movement):
                    logger.warning(f"⚠️  CALL signal: ideal={ideal_call_movement:+.3f} > actual_ask={call_ask_movement:+.3f} or actual_bid={call_bid_movement:+.3f}")
                    should_reposition = True

            # Check PUT signals
            if abs(put_sens) > 0.000001 and abs(ideal_put_movement) > SIGNAL_THRESHOLD:
                # Check if ideal movement exceeds actual movement
                if abs(ideal_put_movement) > abs(put_ask_movement) or abs(ideal_put_movement) > abs(put_bid_movement):
                    logger.warning(f"⚠️  PUT signal: ideal={ideal_put_movement:+.3f} > actual_ask={put_ask_movement:+.3f} or actual_bid={put_bid_movement:+.3f}")
                    should_reposition = True

            if should_reposition:
                logger.info(f"📊 Signal Stats: BTC_Δ=${btc_delta:+.2f} | CALL_sens={call_sens:.4f} PUT_sens={put_sens:.4f}")
                logger.info(f"   Distance={distance:.1f} | Volatility={volatility:.0f} | Time={seconds_remaining}s")

            return should_reposition

        except Exception as e:
            logger.error(f"❌ Error checking signals: {e}")
            return False

    def update_market_making_orders(self, call_bid: float, call_ask: float, put_bid: float, put_ask: float, btc_delta: float = 0):
        """Update market making orders based on current prices"""
        try:
            # Calculate position counts
            call_shares = sum(p.quantity for p in self.positions if p.token_type == 'CALL')
            put_shares = sum(p.quantity for p in self.positions if p.token_type == 'PUT')

            # Calculate spreads
            call_spread = call_ask - call_bid
            put_spread = put_ask - put_bid

            logger.info(f"\n{'='*70}")
            logger.info(f"📊 Market Making Update | BTC_Δ=${btc_delta:+.2f}")
            logger.info(f"   CALL: bid={call_bid:.2f} ask={call_ask:.2f} spread={call_spread:.3f}")
            logger.info(f"   PUT:  bid={put_bid:.2f} ask={put_ask:.2f} spread={put_spread:.3f}")
            logger.info(f"   Positions: CALL={call_shares:.0f}/{MAX_SHARES_PER_SIDE} PUT={put_shares:.0f}/{MAX_SHARES_PER_SIDE}")

            # CALL orders - ALWAYS PLACE (no spread check)
            # Buy order if room
            if call_shares < MAX_SHARES_PER_SIDE:
                buy_price = max(0.01, call_bid - BUY_OFFSET)
                buy_price = round(buy_price, 2)
                buy_quantity = min(6, MAX_SHARES_PER_SIDE - call_shares)

                if buy_price >= 0.01 and buy_price <= 0.99:
                    self.place_buy_order('CALL', self.current_call_id, buy_price, buy_quantity)

            # Sell order if have inventory
            if call_shares >= 1:
                sell_price = min(0.99, call_ask + SELL_OFFSET)
                sell_price = round(sell_price, 2)
                sell_quantity = min(call_shares, 6)

                if sell_price >= 0.01 and sell_price <= 0.99:
                    self.place_sell_order('CALL', self.current_call_id, sell_price, sell_quantity)

            # PUT orders - ALWAYS PLACE (no spread check)
            # Buy order if room
            if put_shares < MAX_SHARES_PER_SIDE:
                buy_price = max(0.01, put_bid - BUY_OFFSET)
                buy_price = round(buy_price, 2)
                buy_quantity = min(6, MAX_SHARES_PER_SIDE - put_shares)

                if buy_price >= 0.01 and buy_price <= 0.99:
                    self.place_buy_order('PUT', self.current_put_id, buy_price, buy_quantity)

            # Sell order if have inventory
            if put_shares >= 1:
                sell_price = min(0.99, put_ask + SELL_OFFSET)
                sell_price = round(sell_price, 2)
                sell_quantity = min(put_shares, 6)

                if sell_price >= 0.01 and sell_price <= 0.99:
                    self.place_sell_order('PUT', self.current_put_id, sell_price, sell_quantity)

            logger.info(f"   Orders placed: {len(self.open_orders)}")
            logger.info(f"{'='*70}")

        except Exception as e:
            logger.error(f"❌ Error updating MM orders: {e}")

    def check_stop_loss(self, call_bid: float, put_bid: float):
        """Check stop loss for imbalanced positions"""
        try:
            call_positions = [p for p in self.positions if p.token_type == 'CALL']
            put_positions = [p for p in self.positions if p.token_type == 'PUT']

            call_shares = sum(p.quantity for p in call_positions)
            put_shares = sum(p.quantity for p in put_positions)

            # Only check if imbalanced
            if abs(call_shares - put_shares) <= 1:
                return  # Balanced, no stop loss check

            # Check CALL stop loss (if we have more CALLs)
            if call_shares > put_shares + 1:
                for position in call_positions:
                    pnl_per_share = call_bid - position.entry_price
                    if pnl_per_share <= -STOP_LOSS:
                        logger.warning(f"🛑 STOP LOSS: CALL imbalance ${pnl_per_share:+.2f}/share")

                        # Check minimum price
                        if call_bid < 0.01:
                            logger.warning(f"   ⚠️  Market price too low (${call_bid:.3f} < $0.01), keeping position")
                            break

                        logger.info(f"   Selling {position.quantity:.0f} CALL @ ${call_bid:.2f}")

                        # Execute market sell (FAK)
                        actual_balance = self.check_token_balance(self.current_call_id)
                        if actual_balance >= 0.5:
                            order_id = self.trader.place_sell_order_FAK(
                                token_id=self.current_call_id,
                                price=call_bid,
                                quantity=actual_balance
                            )

                            if order_id:
                                # Remove from positions
                                self.positions = [p for p in self.positions if p.token_type != 'CALL']

                                # Save trade
                                self.save_trade({
                                    'timestamp': datetime.now().isoformat(),
                                    'type': 'STOP_LOSS',
                                    'token_type': 'CALL',
                                    'quantity': actual_balance,
                                    'price': call_bid,
                                    'pnl': pnl_per_share * actual_balance
                                })
                        break  # Only sell once

            # Check PUT stop loss (if we have more PUTs)
            if put_shares > call_shares + 1:
                for position in put_positions:
                    pnl_per_share = put_bid - position.entry_price
                    if pnl_per_share <= -STOP_LOSS:
                        logger.warning(f"🛑 STOP LOSS: PUT imbalance ${pnl_per_share:+.2f}/share")

                        # Check minimum price
                        if put_bid < 0.01:
                            logger.warning(f"   ⚠️  Market price too low (${put_bid:.3f} < $0.01), keeping position")
                            break

                        logger.info(f"   Selling {position.quantity:.0f} PUT @ ${put_bid:.2f}")

                        # Execute market sell (FAK)
                        actual_balance = self.check_token_balance(self.current_put_id)
                        if actual_balance >= 0.5:
                            order_id = self.trader.place_sell_order_FAK(
                                token_id=self.current_put_id,
                                price=put_bid,
                                quantity=actual_balance
                            )

                            if order_id:
                                # Remove from positions
                                self.positions = [p for p in self.positions if p.token_type != 'PUT']

                                # Save trade
                                self.save_trade({
                                    'timestamp': datetime.now().isoformat(),
                                    'type': 'STOP_LOSS',
                                    'token_type': 'PUT',
                                    'quantity': actual_balance,
                                    'price': put_bid,
                                    'pnl': pnl_per_share * actual_balance
                                })
                        break

        except Exception as e:
            logger.error(f"❌ Error checking stop loss: {e}")

    def run(self):
        """Main market making loop"""
        logger.info("\n🚀 Starting market maker bot...")

        # Initial setup
        self.reload_asset_ids()
        self.refresh_usdc_balance()



        # Update strike price initially
        self.update_strike_price()

        while True:
            try:
                now = datetime.now()

                # Determine period (00, 15, 30, 45)
                minute = now.minute
                if minute < 15:
                    period_start = 0
                elif minute < 30:
                    period_start = 15
                elif minute < 45:
                    period_start = 30
                else:
                    period_start = 45

                # Calculate seconds into period
                seconds_into_period = (minute % 15) * 60 + now.second
                seconds_remaining = 900 - seconds_into_period
                in_buffer_zone = seconds_into_period < BUFFER_SECONDS

                # NEW PERIOD DETECTED
                if period_start != self.current_period_start:
                    logger.info(f"\n{'='*80}")
                    logger.info(f"🔄 NEW PERIOD: {now.strftime('%H:%M')} (:{period_start:02d})")
                    logger.info(f"{'='*80}")

                    # Cancel all orders
                    self.cancel_all_orders()

                    # Update period and get new strike price
                    self.current_period_start = period_start
                    self.reload_asset_ids()
                    self.update_strike_price()
                    self.buffer_check_done = False  # Reset buffer check flag

                    # Reset tracking
                    self.last_btc_price = None
                    self.last_call_ask = None
                    self.last_call_bid = None
                    self.last_put_ask = None
                    self.last_put_bid = None

                # Periodic maintenance
                if time.time() - self.last_asset_reload >= 60:
                    self.reload_asset_ids()

                # Check strike price at end of buffer period (once per period)
                if not in_buffer_zone and not self.buffer_check_done:
                    logger.info("🔄 End of buffer - rechecking strike price")
                    self.update_strike_price()
                    self.buffer_check_done = True

                # Recheck strike price every 60 seconds
                if time.time() - self.last_strike_check >= 60:
                    self.update_strike_price()

                if time.time() - self.last_position_check >= POSITION_CHECK_INTERVAL:
                    # print("We are here0")
                    self.verify_positions_from_wallet()
                    # print("We are here1")
                # print("We are here2")
                # Refresh USDC balance (may block)

                try:
                    self.refresh_usdc_balance()
                    # print("USDCHECKOK")
                except Exception as e:
                    print("usdc check failed")
                    logger.error(f"❌ USDC balance check failed: {e}")

                # print("We are here3")
                # Read prices
                btc_data = read_json(BTC_FILE)
                call_data = read_json(CALL_FILE)
                put_data = read_json(PUT_FILE)

                if not all([btc_data, call_data, put_data]):
                    print("NOTALL data quite")
                    time.sleep(0.1)
                    continue

                # Extract prices
                btc_price = btc_data.get('price', 0)
                self.btc_price_history.append(btc_price)

                call_bid = call_data.get('best_bid', 0)
                call_ask = call_data.get('best_ask', 0)
                put_bid = put_data.get('best_bid', 0)
                put_ask = put_data.get('best_ask', 0)

                print("H ", call_bid,"/", call_ask)

                if not all([btc_price, call_bid, call_ask, put_bid, put_ask]):
                    # print("NOTALL data ALL")
                    time.sleep(0.1)
                    continue

                # Calculate volatility
                volatility = calculate_btc_volatility(self.btc_price_history)

                # Calculate movements if we have previous prices
                should_check_signals = (self.last_btc_price is not None and
                                       self.last_call_ask is not None and
                                       self.last_put_ask is not None and
                                       len(self.open_orders) > 0 and
                                       not in_buffer_zone)

                if should_check_signals:
                    #print("H check signals")
                    btc_delta = btc_price - self.last_btc_price
                    call_ask_movement = call_ask - self.last_call_ask
                    call_bid_movement = call_bid - self.last_call_bid
                    put_ask_movement = put_ask - self.last_put_ask
                    put_bid_movement = put_bid - self.last_put_bid

                    # Check if signals indicate we should reposition
                    if self.check_signal_for_reposition(
                        btc_price, btc_delta,
                        call_ask, call_bid, call_ask_movement, call_bid_movement,
                        put_ask, put_bid, put_ask_movement, put_bid_movement,
                        seconds_remaining, volatility
                    ):
                        logger.info("🔄 Repositioning orders based on signals...")
                        self.cancel_all_orders()
                        self.last_cancel_time = time.time()

                    # else :
                    #     print("H nocheck signale forrepositions")
                # else :
                    # print("should not che signsl")
                # Update tracking for next iteration
                self.last_btc_price = btc_price
                self.last_call_ask = call_ask
                self.last_call_bid = call_bid
                self.last_put_ask = put_ask
                self.last_put_bid = put_bid

                # Check stop loss for imbalanced positions
                if not in_buffer_zone:
                    self.check_stop_loss(call_bid, put_bid)

                # Place orders if:
                # 1. Not in buffer zone
                # 2. No orders currently active
                # 3. Enough time passed since last cancel (repositioning delay)
                can_place_orders = (not in_buffer_zone and
                                   len(self.open_orders) == 0 and
                                   time.time() - self.last_cancel_time >= REPOSITION_DELAY)

                if can_place_orders and self.current_call_id and self.current_put_id:
                    # Calculate btc_delta for display
                    btc_delta_display = 0
                    if self.last_btc_price is not None:
                        btc_delta_display = btc_price - self.last_btc_price
                    self.update_market_making_orders(call_bid, call_ask, put_bid, put_ask, btc_delta_display)

                # Run every 0.1 seconds for fast reaction
                time.sleep(0.1)

            except KeyboardInterrupt:
                logger.info("\n\n⏸️  Shutting down...")
                self.cancel_all_orders()
                break

            except Exception as e:
                logger.error(f"❌ Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)


def main():
    """Main entry point"""
    try:
        # Load credentials
        try:
            credentials = load_credentials_from_env(CREDENTIALS_ENV)
            print(f"✅ Credentials loaded from {CREDENTIALS_ENV}")
        except Exception as e:
            print(f"❌ Error loading credentials: {e}")
            return

        # Create and run bot
        bot = MarketMakerBot(credentials)
        bot.run()

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
