#!/usr/bin/env python3
"""
Binary Options Arbitrage Bot
Monitors BTC price movements and exploits pricing inefficiencies in 15M PUT/CALL tokens
"""

import json
import time
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List
import logging
from scipy.optimize import curve_fit
import warnings

# Suppress all numpy polynomial warnings
warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PriceSnapshot:
    """Single price observation"""
    timestamp: float
    btc_price: float
    put_bid: float
    put_ask: float
    call_bid: float
    call_ask: float
    put_spread: float
    call_spread: float


@dataclass
class Position:
    """Open position tracker"""
    token_type: str  # 'PUT' or 'CALL'
    entry_price: float
    entry_time: float
    quantity: float
    entry_btc_price: float
    target_exit: Optional[float] = None


@dataclass
class Trade:
    """Completed trade record"""
    timestamp: str
    token_type: str
    action: str  # 'BUY' or 'SELL'
    price: float
    quantity: float
    btc_price: float
    pnl: Optional[float] = None
    reason: str = ""


class BinaryArbitrageBot:
    """
    Monitors BTC price and binary options tokens to exploit pricing inefficiencies
    """
    
    def __init__(
        self,
        put_file: str = "/home/ubuntu/013_2025_polymarket/15M_PUT.json",
        call_file: str = "/home/ubuntu/013_2025_polymarket/15M_CALL.json",
        btc_file: str = "/home/ubuntu/013_2025_polymarket/bybit_btc_price.json",
        discrepancy_threshold: float = 0.03,
        position_size: float = 1.0,
        daily_log_dir: str = "/home/ubuntu/013_2025_polymarket/bot016_react"
    ):
        self.put_file = put_file
        self.call_file = call_file
        self.btc_file = btc_file
        self.discrepancy_threshold = discrepancy_threshold
        self.position_size = position_size
        self.daily_log_dir = Path(daily_log_dir)
        self.daily_log_dir.mkdir(exist_ok=True)
        
        # Price history buffers
        self.price_history_5s = deque(maxlen=50)   # 5s at 0.1s intervals
        self.price_history_15s = deque(maxlen=150) # 15s at 0.1s intervals
        self.price_history_30s = deque(maxlen=300) # 30s at 0.1s intervals
        
        # Position tracking
        self.position: Optional[Position] = None
        self.trades_today: List[Trade] = []
        self.daily_pnl: float = 0.0
        
        # Strike price tracking (BTC price at start of 15-min period)
        self.current_strike: Optional[float] = None
        self.strike_timestamp: Optional[float] = None
        
        # Load today's log if exists
        self.load_daily_log()
        
        logger.info("Binary Arbitrage Bot initialized")
        logger.info(f"Discrepancy threshold: ${self.discrepancy_threshold}")
        logger.info(f"Position size: ${self.position_size}")
    
    def get_daily_log_path(self) -> Path:
        """Get path for today's log file"""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.daily_log_dir / f"trading_log_{today}.json"
    
    def load_daily_log(self):
        """Load today's trading log if it exists"""
        log_path = self.get_daily_log_path()
        if log_path.exists():
            try:
                with open(log_path, 'r') as f:
                    data = json.load(f)
                    self.trades_today = [Trade(**t) for t in data.get('trades', [])]
                    self.daily_pnl = data.get('daily_pnl', 0.0)
                    logger.info(f"Loaded {len(self.trades_today)} trades from today's log")
                    logger.info(f"Current daily PNL: ${self.daily_pnl:.2f}")
            except Exception as e:
                logger.error(f"Error loading daily log: {e}")
    
    def save_daily_log(self):
        """Save today's trading activity"""
        log_path = self.get_daily_log_path()
        data = {
            'date': datetime.now().strftime("%Y-%m-%d"),
            'daily_pnl': round(self.daily_pnl, 2),
            'total_trades': len(self.trades_today),
            'trades': [asdict(t) for t in self.trades_today],
            'current_position': asdict(self.position) if self.position else None
        }
        
        try:
            with open(log_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving daily log: {e}")
    
    def read_json_file(self, filepath: str) -> Optional[dict]:
        """Safely read JSON file"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.debug(f"Error reading {filepath}: {e}")
            return None
    
    def get_current_prices(self) -> Optional[PriceSnapshot]:
        """Read current prices from all three files"""
        put_data = self.read_json_file(self.put_file)
        call_data = self.read_json_file(self.call_file)
        btc_data = self.read_json_file(self.btc_file)
        
        if not all([put_data, call_data, btc_data]):
            return None
        
        try:
            # Handle None/null best_bid and best_ask
            put_bid = put_data.get('best_bid')
            put_ask = put_data.get('best_ask')
            call_bid = call_data.get('best_bid')
            call_ask = call_data.get('best_ask')
            
            # Skip if any critical price is missing
            if not put_bid or not put_ask or not call_bid or not call_ask:
                logger.debug("Missing bid/ask data - skipping snapshot")
                return None
            
            snapshot = PriceSnapshot(
                timestamp=time.time(),
                btc_price=btc_data['price'],
                put_bid=put_bid.get('price', 0.0),
                put_ask=put_ask.get('price', 0.0),
                call_bid=call_bid.get('price', 0.0),
                call_ask=call_ask.get('price', 0.0),
                put_spread=put_data.get('spread', 0.0),
                call_spread=call_data.get('spread', 0.0)
            )
            
            # Sanity check prices are in valid range
            if not (0 <= snapshot.put_bid <= 1 and 0 <= snapshot.put_ask <= 1 and
                    0 <= snapshot.call_bid <= 1 and 0 <= snapshot.call_ask <= 1):
                logger.debug("Invalid price range - skipping snapshot")
                return None
            
            return snapshot
        except (KeyError, TypeError) as e:
            logger.debug(f"Error parsing price data: {e}")
            return None
    
    def calculate_btc_momentum(self, window: deque) -> Tuple[float, float]:
        """
        Calculate BTC price momentum and volatility
        Returns: (momentum_pct, volatility)
        """
        if len(window) < 10:
            return 0.0, 0.0
        
        prices = np.array([s.btc_price for s in window])
        
        # Linear regression for momentum
        x = np.arange(len(prices))
        z = np.polyfit(x, prices, 1)
        momentum = z[0]  # slope
        momentum_pct = (momentum / prices[0]) * 100 if prices[0] > 0 else 0
        
        # Volatility as standard deviation
        volatility = np.std(prices) / np.mean(prices) if len(prices) > 1 else 0
        
        return momentum_pct, volatility
    
    def calculate_time_to_expiry(self) -> float:
        """
        Calculate time remaining until next expiry (in minutes)
        Options expire at :00, :15, :30, :45
        """
        now = datetime.now()
        current_minute = now.minute
        
        # Find next expiry minute
        expiry_minutes = [0, 15, 30, 45]
        next_expiry = None
        for exp_min in expiry_minutes:
            if current_minute < exp_min:
                next_expiry = exp_min
                break
        
        if next_expiry is None:
            # Next expiry is at :00 of next hour
            next_expiry_dt = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
        else:
            next_expiry_dt = now.replace(minute=next_expiry, second=0, microsecond=0)
        
        time_to_expiry = (next_expiry_dt - now).total_seconds() / 60.0
        return max(0.1, time_to_expiry)  # Minimum 0.1 minutes
    
    def update_strike_price(self, btc_price: float):
        """
        Update strike price at the start of each 15-minute period
        Strike is set at :00, :15, :30, :45
        """
        now = datetime.now()
        current_minute = now.minute
        current_second = now.second
        
        # Check if we're within first 10 seconds of a new period
        if current_minute in [0, 15, 30, 45] and current_second < 10:
            # Check if we need to set a new strike
            period_start = now.replace(second=0, microsecond=0).timestamp()
            
            if self.strike_timestamp is None or period_start != self.strike_timestamp:
                self.current_strike = btc_price
                self.strike_timestamp = period_start
                logger.info(f"🎯 New period started - Strike set at ${btc_price:.2f}")
    
    def predict_option_price_from_btc_regression(self, current_btc: float, token_type: str) -> Optional[float]:
        """
        Predict what the option price SHOULD BE based on polynomial regression
        of recent BTC price -> option price relationship
        
        This captures the market's normal reaction pattern, allowing us to detect
        when the market hasn't caught up to a BTC price spike yet.
        """
        if token_type == 'PUT':
            history = [(s.btc_price, (s.put_bid + s.put_ask) / 2) for s in self.price_history_5s]
        else:  # CALL
            history = [(s.btc_price, (s.call_bid + s.call_ask) / 2) for s in self.price_history_5s]
        
        if len(history) < 10:
            return None
        
        # Extract BTC prices and option prices
        btc_prices = np.array([h[0] for h in history])
        option_prices = np.array([h[1] for h in history])
        
        # Remove outliers (prices outside 0.01-0.99)
        valid_mask = (option_prices >= 0.01) & (option_prices <= 0.99)
        btc_prices = btc_prices[valid_mask]
        option_prices = option_prices[valid_mask]
        
        if len(btc_prices) < 5:
            return None
        
        try:
            # Fit polynomial regression (degree 2 for capturing curvature)
            # This learns: as BTC rises, how does the option price typically react?
            coeffs = np.polyfit(btc_prices, option_prices, deg=2)
            poly = np.poly1d(coeffs)
            
            # Predict what option price should be at current BTC price
            predicted_price = poly(current_btc)
            
            # Clamp to valid range
            predicted_price = max(0.01, min(0.99, predicted_price))
            
            return round(predicted_price, 2)
            
        except Exception as e:
            logger.debug(f"Regression failed: {e}")
            return None
    
    def check_data_freshness(self, snapshot: PriceSnapshot) -> bool:
        """
        Check if data is fresh (not stale)
        Returns True if fresh, False if stale
        """
        current_time = time.time()
        
        # Check BTC data age (from bybit file timestamp)
        btc_age = current_time - (snapshot.timestamp)
        
        # Check option data age (from snapshot timestamp)
        option_age = current_time - snapshot.timestamp
        
        # Alert if data is more than 5 seconds old
        if btc_age > 5.0:
            logger.warning(f"⚠️  BTC price data is stale ({btc_age:.1f}s old)")
            return False
        
        if option_age > 5.0:
            logger.warning(f"⚠️  Option price data is stale ({option_age:.1f}s old)")
            return False
        
        return True
    
    def detect_btc_spike(self) -> Tuple[bool, float]:
        """
        Detect if BTC just had a significant price spike
        Returns: (has_spike, spike_magnitude_pct)
        """
        if len(self.price_history_5s) < 20:
            return False, 0.0
        
        # Compare most recent price to 2 seconds ago (20 samples at 0.1s)
        recent_prices = list(self.price_history_5s)
        current_price = recent_prices[-1].btc_price
        price_2s_ago = recent_prices[-20].btc_price
        
        spike_pct = ((current_price - price_2s_ago) / price_2s_ago) * 100
        
        # Consider it a spike if > 0.05% in 2 seconds
        has_spike = abs(spike_pct) > 0.05
        
        return has_spike, spike_pct
    
    def calculate_expected_prices(self, current_btc: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate expected PUT and CALL prices based on regression of recent data
        Returns: (expected_put_price, expected_call_price)
        
        This predicts what the market price SHOULD BE based on the recent
        BTC price -> option price relationship
        """
        expected_put = self.predict_option_price_from_btc_regression(current_btc, 'PUT')
        expected_call = self.predict_option_price_from_btc_regression(current_btc, 'CALL')
        
        return expected_put, expected_call
    
    def check_entry_signal(self, snapshot: PriceSnapshot, 
                           expected_put: Optional[float], expected_call: Optional[float]) -> Optional[Tuple[str, float, str]]:
        """
        Check for entry signals based on market lag after BTC spikes
        Returns: (token_type, entry_price, reason) or None
        """
        if expected_put is None or expected_call is None:
            return None
        
        # Don't trade extreme prices (too risky near expiry)
        if snapshot.put_ask >= 0.97 or snapshot.put_ask <= 0.03:
            expected_put = None  # Skip PUT
        if snapshot.call_ask >= 0.97 or snapshot.call_ask <= 0.03:
            expected_call = None  # Skip CALL
        
        # Detect if we just had a BTC spike
        has_spike, spike_pct = self.detect_btc_spike()
        
        # Calculate how much market is lagging
        put_lag = expected_put - snapshot.put_ask if expected_put else -999
        call_lag = expected_call - snapshot.call_ask if expected_call else -999
        
        logger.debug(f"Spike: {has_spike} ({spike_pct:+.3f}%) | PUT lag: {put_lag:.3f} | CALL lag: {call_lag:.3f}")
        
        # If we have a spike and market hasn't caught up yet
        if has_spike:
            # BTC spiked up -> CALL should be more expensive
            if spike_pct > 0.05 and expected_call and call_lag >= self.discrepancy_threshold:
                return ('CALL', snapshot.call_ask, 
                       f"BTC spike {spike_pct:+.3f}%, market lag ${call_lag:.3f}")
            
            # BTC spiked down -> PUT should be more expensive
            if spike_pct < -0.05 and expected_put and put_lag >= self.discrepancy_threshold:
                return ('PUT', snapshot.put_ask,
                       f"BTC spike {spike_pct:+.3f}%, market lag ${put_lag:.3f}")
        
        # Also check for general discrepancies (market slow to react)
        if expected_put and put_lag >= self.discrepancy_threshold:
            return ('PUT', snapshot.put_ask, 
                   f"PUT underpriced by ${put_lag:.3f}, expected={expected_put:.2f}")
        
        if expected_call and call_lag >= self.discrepancy_threshold:
            return ('CALL', snapshot.call_ask,
                   f"CALL underpriced by ${call_lag:.3f}, expected={expected_call:.2f}")
        
        return None
    
    def check_exit_signal(self, snapshot: PriceSnapshot,
                          expected_put: Optional[float], expected_call: Optional[float]) -> Optional[Tuple[float, str]]:
        """
        Check for exit signals for open position
        
        Exit conditions:
        1. Bid hits 0.99 (maximum profit)
        2. Opposite opportunity arises (flip position)
        3. Near expiry (< 1 minute)
        
        Returns: (exit_price, reason) or None
        """
        if not self.position:
            return None
        
        if self.position.token_type == 'PUT':
            # ALWAYS SELL at 0.99
            if snapshot.put_bid >= 0.99:
                return (0.99, "Hit max price 0.99")
            
            # Check if CALL opportunity arises (opposite direction)
            if expected_call:
                call_lag = expected_call - snapshot.call_ask
                # Only flip if it's a significant opportunity
                if call_lag >= self.discrepancy_threshold and snapshot.call_ask <= 0.97:
                    return (snapshot.put_bid, f"Flip to CALL opportunity (lag ${call_lag:.3f})")
        
        else:  # CALL
            # ALWAYS SELL at 0.99
            if snapshot.call_bid >= 0.99:
                return (0.99, "Hit max price 0.99")
            
            # Check if PUT opportunity arises (opposite direction)
            if expected_put:
                put_lag = expected_put - snapshot.put_ask
                # Only flip if it's a significant opportunity
                if put_lag >= self.discrepancy_threshold and snapshot.put_ask <= 0.97:
                    return (snapshot.call_bid, f"Flip to PUT opportunity (lag ${put_lag:.3f})")
        
        # Time-based exit (close to expiry)
        time_to_expiry = self.calculate_time_to_expiry()
        if time_to_expiry < 1.0:  # Less than 1 minute to expiry
            exit_price = snapshot.put_bid if self.position.token_type == 'PUT' else snapshot.call_bid
            return (exit_price, f"Near expiry: {time_to_expiry:.1f}m")
        
        return None
    
    def execute_entry(self, token_type: str, entry_price: float, 
                      btc_price: float, reason: str):
        """Execute entry into position (or flip from opposite position)"""
        
        # If we have an opposite position, close it first
        if self.position and self.position.token_type != token_type:
            logger.info(f"🔄 Flipping position from {self.position.token_type} to {token_type}")
            # Close existing position at current bid
            if self.position.token_type == 'PUT':
                # Get current PUT snapshot for exit price
                put_data = self.read_json_file(self.put_file)
                if put_data and put_data.get('best_bid'):
                    exit_price = put_data['best_bid'].get('price', self.position.entry_price)
                else:
                    exit_price = self.position.entry_price
            else:  # CALL
                call_data = self.read_json_file(self.call_file)
                if call_data and call_data.get('best_bid'):
                    exit_price = call_data['best_bid'].get('price', self.position.entry_price)
                else:
                    exit_price = self.position.entry_price
            
            self.execute_exit(exit_price, btc_price, f"Flip to {token_type}")
        
        quantity = self.position_size / entry_price
        
        self.position = Position(
            token_type=token_type,
            entry_price=entry_price,
            entry_time=time.time(),
            quantity=quantity,
            entry_btc_price=btc_price
        )
        
        trade = Trade(
            timestamp=datetime.now().isoformat(),
            token_type=token_type,
            action='BUY',
            price=entry_price,
            quantity=quantity,
            btc_price=btc_price,
            reason=reason
        )
        
        self.trades_today.append(trade)
        self.save_daily_log()
        
        logger.info(f"🟢 ENTRY: {token_type} @ ${entry_price:.3f} | Qty: {quantity:.2f} | {reason}")
    
    def execute_exit(self, exit_price: float, btc_price: float, reason: str):
        """Execute exit from position"""
        if not self.position:
            return
        
        # Calculate PNL
        pnl = (exit_price - self.position.entry_price) * self.position.quantity
        self.daily_pnl += pnl
        
        trade = Trade(
            timestamp=datetime.now().isoformat(),
            token_type=self.position.token_type,
            action='SELL',
            price=exit_price,
            quantity=self.position.quantity,
            btc_price=btc_price,
            pnl=pnl,
            reason=reason
        )
        
        self.trades_today.append(trade)
        self.save_daily_log()
        
        hold_time = time.time() - self.position.entry_time
        logger.info(f"🔴 EXIT: {self.position.token_type} @ ${exit_price:.3f} | "
                   f"PNL: ${pnl:.2f} | Hold: {hold_time:.1f}s | {reason}")
        logger.info(f"💰 Daily PNL: ${self.daily_pnl:.2f} | Total trades: {len(self.trades_today)}")
        
        self.position = None
    
    def run(self):
        """Main bot loop"""
        logger.info("🚀 Starting Binary Arbitrage Bot")
        logger.info(f"Monitoring: {self.put_file}, {self.call_file}, {self.btc_file}")
        
        iteration = 0
        last_freshness_check = time.time()
        
        try:
            while True:
                # Read current prices
                snapshot = self.get_current_prices()
                
                if snapshot:
                    # Check data freshness every 5 seconds
                    if time.time() - last_freshness_check > 5.0:
                        if not self.check_data_freshness(snapshot):
                            logger.error("❌ Data is stale - skipping trading")
                            time.sleep(1)
                            last_freshness_check = time.time()
                            continue
                        last_freshness_check = time.time()
                    
                    # Add to history buffers
                    self.price_history_5s.append(snapshot)
                    self.price_history_15s.append(snapshot)
                    self.price_history_30s.append(snapshot)
                    
                    # Update strike price if new period started
                    self.update_strike_price(snapshot.btc_price)
                    
                    # Calculate expected prices (regression-based)
                    if len(self.price_history_5s) >= 20:  # Need at least 2 seconds of data
                        expected_put, expected_call = self.calculate_expected_prices(snapshot.btc_price)
                        
                        # Detect spikes
                        has_spike, spike_pct = self.detect_btc_spike()
                        
                        # Log current state periodically
                        if iteration % 50 == 0:
                            time_to_exp = self.calculate_time_to_expiry()
                            spike_indicator = f"⚡ SPIKE {spike_pct:+.3f}%" if has_spike else ""
                            
                            expected_put_str = f"{expected_put:.2f}" if expected_put else "N/A"
                            expected_call_str = f"{expected_call:.2f}" if expected_call else "N/A"
                            
                            # Position status
                            if self.position:
                                pos_info = f"📍 OPEN {self.position.token_type} @ ${self.position.entry_price:.2f}"
                            else:
                                pos_info = "⭕ NO POSITION"
                            
                            logger.info(f"📊 BTC: ${snapshot.btc_price:.2f} {spike_indicator} | "
                                      f"PUT: {snapshot.put_bid:.2f}/{snapshot.put_ask:.2f} (exp: {expected_put_str}) | "
                                      f"CALL: {snapshot.call_bid:.2f}/{snapshot.call_ask:.2f} (exp: {expected_call_str}) | "
                                      f"Exp: {time_to_exp:.1f}m")
                            logger.info(f"💰 Daily PNL: ${self.daily_pnl:.2f} | Trades: {len(self.trades_today)} | {pos_info}")
                        
                        # Check for signals
                        if self.position is None:
                            # Look for entry
                            entry_signal = self.check_entry_signal(snapshot, expected_put, expected_call)
                            if entry_signal:
                                token_type, entry_price, reason = entry_signal
                                self.execute_entry(token_type, entry_price, snapshot.btc_price, reason)
                        else:
                            # Look for exit (includes flip opportunities)
                            exit_signal = self.check_exit_signal(snapshot, expected_put, expected_call)
                            if exit_signal:
                                exit_price, reason = exit_signal
                                
                                # Check if this is a flip signal
                                if "Flip to" in reason:
                                    # Extract which direction to flip to and execute entry for opposite
                                    if "CALL" in reason:
                                        self.execute_exit(exit_price, snapshot.btc_price, reason)
                                        # Now enter CALL
                                        self.execute_entry('CALL', snapshot.call_ask, snapshot.btc_price, 
                                                         f"Flipped from PUT: {reason}")
                                    else:  # Flip to PUT
                                        self.execute_exit(exit_price, snapshot.btc_price, reason)
                                        # Now enter PUT
                                        self.execute_entry('PUT', snapshot.put_ask, snapshot.btc_price,
                                                         f"Flipped from CALL: {reason}")
                                else:
                                    # Regular exit
                                    self.execute_exit(exit_price, snapshot.btc_price, reason)
                
                iteration += 1
                time.sleep(0.1)  # 10 reads per second
                
        except KeyboardInterrupt:
            logger.info("\n⏸️  Bot stopped by user")
            if self.position:
                logger.info(f"⚠️  Open position: {self.position.token_type} @ ${self.position.entry_price:.3f}")
            self.save_daily_log()
        except Exception as e:
            logger.error(f"❌ Error in main loop: {e}", exc_info=True)
            self.save_daily_log()


def main():
    """Entry point"""
    bot = BinaryArbitrageBot(
        discrepancy_threshold=0.03,
        position_size=100.0
    )
    bot.run()


if __name__ == "__main__":
    main()
