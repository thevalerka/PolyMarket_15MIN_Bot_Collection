#!/usr/bin/env python3
"""
Polymarket Data Collection Bot
Collects second-by-second market data for research
Calculates: Greeks, microstructure metrics, volatility, entropy, etc.
"""

import json
import time
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple
import logging
from collections import deque
from scipy import stats
import math

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = "/home/ubuntu/013_2025_polymarket"
OUTPUT_DIR = f"{DATA_DIR}/research_data"
CALL_FILE = f"{DATA_DIR}/15M_CALL.json"
PUT_FILE = f"{DATA_DIR}/15M_PUT.json"
BTC_FILE = f"{DATA_DIR}/bybit_btc_price.json"

# Circular buffers for rolling calculations
BUFFER_SIZE = 300  # 5 minutes of second data


class CircularBuffer:
    """Efficient circular buffer for time-series data"""
    def __init__(self, size: int):
        self.buffer = deque(maxlen=size)
    
    def append(self, value):
        self.buffer.append(value)
    
    def get_array(self) -> np.ndarray:
        return np.array(list(self.buffer))
    
    def __len__(self):
        return len(self.buffer)


class OptionGreeksCalculator:
    """Calculate option Greeks using Black-Scholes approximations"""
    
    @staticmethod
    def calculate_d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
        """Calculate d1 and d2 for Black-Scholes"""
        if T <= 0 or sigma <= 0:
            return 0.0, 0.0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    
    @staticmethod
    def calculate_implied_vol(option_price: float, S: float, K: float, T: float, 
                            option_type: str = 'call', r: float = 0.0) -> float:
        """Calculate implied volatility using Newton-Raphson"""
        if T <= 0 or option_price <= 0:
            return 0.0
        
        # Initial guess
        sigma = 0.5
        
        for _ in range(50):  # Max iterations
            d1, d2 = OptionGreeksCalculator.calculate_d1_d2(S, K, T, r, sigma)
            
            if option_type == 'call':
                price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
            else:  # put
                price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
            
            vega = S * stats.norm.pdf(d1) * np.sqrt(T)
            
            if vega < 1e-10:
                break
            
            diff = option_price - price
            if abs(diff) < 1e-6:
                return sigma
            
            sigma = sigma + diff / vega
            
            if sigma <= 0:
                return 0.5
        
        return sigma
    
    @staticmethod
    def calculate_greeks(S: float, K: float, T: float, sigma: float, 
                        option_type: str = 'call', r: float = 0.0) -> Dict:
        """Calculate all Greeks"""
        if T <= 0 or sigma <= 0:
            return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0}
        
        d1, d2 = OptionGreeksCalculator.calculate_d1_d2(S, K, T, r, sigma)
        
        if option_type == 'call':
            delta = stats.norm.cdf(d1)
            theta = (-S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    - r * K * np.exp(-r * T) * stats.norm.cdf(d2))
        else:  # put
            delta = -stats.norm.cdf(-d1)
            theta = (-S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r * T) * stats.norm.cdf(-d2))
        
        gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * stats.norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% vol change
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta / 365  # Per day
        }


class DataCollector:
    """Main data collection bot"""
    
    def __init__(self):
        # Price history buffers
        self.call_bid_history = CircularBuffer(BUFFER_SIZE)
        self.call_ask_history = CircularBuffer(BUFFER_SIZE)
        self.put_bid_history = CircularBuffer(BUFFER_SIZE)
        self.put_ask_history = CircularBuffer(BUFFER_SIZE)
        self.btc_price_history = CircularBuffer(BUFFER_SIZE)
        
        # Mid price buffers
        self.call_mid_history = CircularBuffer(BUFFER_SIZE)
        self.put_mid_history = CircularBuffer(BUFFER_SIZE)
        
        # Spread buffers
        self.call_spread_history = CircularBuffer(BUFFER_SIZE)
        self.put_spread_history = CircularBuffer(BUFFER_SIZE)
        
        # Greeks history
        self.call_iv_history = CircularBuffer(BUFFER_SIZE)
        self.put_iv_history = CircularBuffer(BUFFER_SIZE)
        
        # Create output directory
        import os
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        self.current_period_start = None
        self.strike_price = None
        
        logger.info("📊 Data Collector Bot initialized")
        logger.info(f"📁 Output directory: {OUTPUT_DIR}")
    
    def read_json_file(self, filepath: str) -> Optional[Dict]:
        """Safely read JSON file, handle empty/malformed files"""
        try:
            with open(filepath, 'r') as f:
                content = f.read().strip()
                if not content:
                    return None
                return json.loads(content)
        except json.JSONDecodeError as e:
            # File might be mid-write, skip this iteration
            return None
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            return None
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            return None
    
    def get_strike_price(self) -> Optional[float]:
        """Get current period strike price from Binance"""
        try:
            now = datetime.now(timezone.utc)
            current_minute = now.minute
            
            for start_min in [0, 15, 30, 45]:
                if current_minute >= start_min and current_minute < start_min + 15:
                    period_start = now.replace(minute=start_min, second=0, microsecond=0)
                    
                    if self.current_period_start != period_start:
                        self.current_period_start = period_start
                        
                        # Get strike from Binance
                        import requests
                        start_timestamp = int(period_start.timestamp() * 1000)
                        url = "https://api.binance.com/api/v3/klines"
                        params = {
                            'symbol': 'BTCUSDT',
                            'interval': '15m',
                            'startTime': start_timestamp,
                            'limit': 1,
                            'timeZone': '0'
                        }
                        
                        response = requests.get(url, params=params, timeout=5)
                        if response.status_code == 200:
                            data = response.json()
                            if data and len(data) > 0:
                                self.strike_price = float(data[0][1])
                                logger.info(f"📍 New period: {period_start.strftime('%H:%M')}, Strike: ${self.strike_price:,.2f}")
                    
                    return self.strike_price
            
            return None
        except Exception as e:
            logger.error(f"Error getting strike: {e}")
            return None
    
    def get_time_to_expiry(self) -> float:
        """Get time to expiry in years"""
        if not self.current_period_start:
            return 0.0
        
        now = datetime.now(timezone.utc)
        expiry = self.current_period_start + timedelta(minutes=15)
        seconds_left = (expiry - now).total_seconds()
        
        return max(0, seconds_left / (365.25 * 24 * 3600))  # Convert to years
    
    def calculate_entropy(self, data: np.ndarray, bins: int = 10) -> float:
        """Calculate Shannon entropy"""
        if len(data) < 2:
            return 0.0
        
        try:
            hist, _ = np.histogram(data, bins=bins, density=True)
            hist = hist[hist > 0]
            return -np.sum(hist * np.log2(hist))
        except:
            return 0.0
    
    def calculate_hurst_exponent(self, prices: np.ndarray) -> float:
        """Calculate Hurst exponent (trending vs mean-reverting)"""
        if len(prices) < 20:
            return 0.5
        
        try:
            lags = range(2, min(20, len(prices) // 2))
            tau = [np.std(np.subtract(prices[lag:], prices[:-lag])) for lag in lags]
            
            # Filter out zeros
            valid_indices = [i for i, t in enumerate(tau) if t > 0]
            if len(valid_indices) < 3:
                return 0.5
            
            lags_valid = [lags[i] for i in valid_indices]
            tau_valid = [tau[i] for i in valid_indices]
            
            poly = np.polyfit(np.log(lags_valid), np.log(tau_valid), 1)
            return poly[0]
        except:
            return 0.5
    
    def calculate_realized_volatility(self, prices: np.ndarray, window: int = 60) -> float:
        """Calculate realized volatility (annualized)"""
        if len(prices) < 2:
            return 0.0
        
        returns = np.diff(np.log(prices[-window:]))
        if len(returns) == 0:
            return 0.0
        
        # Annualize: sqrt(seconds_per_year)
        return np.std(returns) * np.sqrt(365.25 * 24 * 3600)
    
    def calculate_orderbook_imbalance(self, bid_size: float, ask_size: float) -> float:
        """Calculate order book imbalance"""
        total = bid_size + ask_size
        if total == 0:
            return 0.0
        return (bid_size - ask_size) / total
    
    def collect_snapshot(self) -> Dict:
        """Collect one snapshot of all data"""
        timestamp = datetime.now(timezone.utc)
        
        # Read raw data
        call_data = self.read_json_file(CALL_FILE)
        put_data = self.read_json_file(PUT_FILE)
        btc_data = self.read_json_file(BTC_FILE)
        
        if not call_data or not put_data or not btc_data:
            return None
        
        # Get strike and time to expiry
        strike = self.get_strike_price()
        T = self.get_time_to_expiry()
        
        # Extract prices (can be None when deep ITM/OTM)
        call_bid = call_data.get('best_bid', {}).get('price') if call_data.get('best_bid') else None
        call_ask = call_data.get('best_ask', {}).get('price') if call_data.get('best_ask') else None
        call_bid_size = call_data.get('best_bid', {}).get('size', 0) if call_data.get('best_bid') else 0
        call_ask_size = call_data.get('best_ask', {}).get('size', 0) if call_data.get('best_ask') else 0
        
        put_bid = put_data.get('best_bid', {}).get('price') if put_data.get('best_bid') else None
        put_ask = put_data.get('best_ask', {}).get('price') if put_data.get('best_ask') else None
        put_bid_size = put_data.get('best_bid', {}).get('size', 0) if put_data.get('best_bid') else 0
        put_ask_size = put_data.get('best_ask', {}).get('size', 0) if put_data.get('best_ask') else 0
        
        btc_price = btc_data.get('price')
        
        # Skip if we don't have essential data
        if not btc_price or (not call_bid and not call_ask) or (not put_bid and not put_ask):
            return None
        
        # Calculate mid prices (use bid or ask if one is missing)
        call_mid = None
        if call_bid and call_ask:
            call_mid = (call_bid + call_ask) / 2
        elif call_bid:
            call_mid = call_bid
        elif call_ask:
            call_mid = call_ask
        
        put_mid = None
        if put_bid and put_ask:
            put_mid = (put_bid + put_ask) / 2
        elif put_bid:
            put_mid = put_bid
        elif put_ask:
            put_mid = put_ask
        
        # Calculate spreads (only if both sides exist)
        call_spread = (call_ask - call_bid) if (call_bid and call_ask) else None
        put_spread = (put_ask - put_bid) if (put_bid and put_ask) else None
        
        # Update history buffers (only append non-None values)
        if call_bid is not None:
            self.call_bid_history.append(call_bid)
        if call_ask is not None:
            self.call_ask_history.append(call_ask)
        if put_bid is not None:
            self.put_bid_history.append(put_bid)
        if put_ask is not None:
            self.put_ask_history.append(put_ask)
        if btc_price is not None:
            self.btc_price_history.append(btc_price)
        if call_mid is not None:
            self.call_mid_history.append(call_mid)
        if put_mid is not None:
            self.put_mid_history.append(put_mid)
        if call_spread is not None:
            self.call_spread_history.append(call_spread)
        if put_spread is not None:
            self.put_spread_history.append(put_spread)
        
        # Calculate Greeks (only when we have reasonable prices)
        greeks_call = {}
        greeks_put = {}
        call_iv = None
        put_iv = None
        
        # Calculate moneyness to determine if option is tradeable
        moneyness = None
        if btc_price and strike:
            moneyness = btc_price / strike
        
        # Only calculate Greeks if option is reasonably near the money
        # Skip if moneyness is extreme (>1.05 or <0.95 means deep ITM/OTM)
        if call_mid and btc_price and strike and T > 0 and moneyness:
            if 0.95 <= moneyness <= 1.05 and call_mid > 0.01 and call_mid < 0.99:
                try:
                    # Calculate implied volatility
                    call_iv = OptionGreeksCalculator.calculate_implied_vol(
                        call_mid, btc_price, strike, T, 'call'
                    )
                    if call_iv > 0:
                        self.call_iv_history.append(call_iv)
                        
                        # Calculate greeks
                        greeks_call = OptionGreeksCalculator.calculate_greeks(
                            btc_price, strike, T, call_iv, 'call'
                        )
                except Exception as e:
                    logger.debug(f"Error calculating CALL Greeks: {e}")
        
        if put_mid and btc_price and strike and T > 0 and moneyness:
            if 0.95 <= moneyness <= 1.05 and put_mid > 0.01 and put_mid < 0.99:
                try:
                    # Calculate implied volatility
                    put_iv = OptionGreeksCalculator.calculate_implied_vol(
                        put_mid, btc_price, strike, T, 'put'
                    )
                    if put_iv > 0:
                        self.put_iv_history.append(put_iv)
                        
                        # Calculate greeks
                        greeks_put = OptionGreeksCalculator.calculate_greeks(
                            btc_price, strike, T, put_iv, 'put'
                        )
                except Exception as e:
                    logger.debug(f"Error calculating PUT Greeks: {e}")
        
        # Calculate rolling metrics
        btc_prices_array = self.btc_price_history.get_array()
        call_mid_array = self.call_mid_history.get_array()
        put_mid_array = self.put_mid_history.get_array()
        
        # Volatility metrics
        btc_realized_vol = self.calculate_realized_volatility(btc_prices_array)
        call_realized_vol = self.calculate_realized_volatility(call_mid_array)
        put_realized_vol = self.calculate_realized_volatility(put_mid_array)
        
        # Entropy metrics
        btc_entropy = self.calculate_entropy(btc_prices_array)
        call_entropy = self.calculate_entropy(call_mid_array)
        put_entropy = self.calculate_entropy(put_mid_array)
        
        # Hurst exponent (trending vs mean-reverting)
        btc_hurst = self.calculate_hurst_exponent(btc_prices_array)
        call_hurst = self.calculate_hurst_exponent(call_mid_array)
        put_hurst = self.calculate_hurst_exponent(put_mid_array)
        
        # Microstructure metrics
        call_imbalance = self.calculate_orderbook_imbalance(call_bid_size, call_ask_size)
        put_imbalance = self.calculate_orderbook_imbalance(put_bid_size, put_ask_size)
        
        # Put-Call parity deviation
        pcp_deviation = 0
        if call_mid and put_mid and btc_price and strike:
            pcp_deviation = call_mid - put_mid - (btc_price - strike)
        
        # Vol risk premium (handle None IVs)
        call_vrp = (call_iv - call_realized_vol) if (call_iv and call_realized_vol and call_iv > 0) else None
        put_vrp = (put_iv - put_realized_vol) if (put_iv and put_realized_vol and put_iv > 0) else None
        
        # Assemble snapshot
        snapshot = {
            'timestamp': timestamp.isoformat(),
            'timestamp_unix': timestamp.timestamp(),
            
            # Period info
            'period_start': self.current_period_start.isoformat() if self.current_period_start else None,
            'strike_price': strike,
            'time_to_expiry_years': T,
            'time_to_expiry_seconds': T * 365.25 * 24 * 3600,
            'moneyness': moneyness,  # Spot / Strike ratio
            
            # Raw prices
            'btc_spot': btc_price,
            'call_bid': call_bid,
            'call_ask': call_ask,
            'call_mid': call_mid,
            'call_spread': call_spread,
            'put_bid': put_bid,
            'put_ask': put_ask,
            'put_mid': put_mid,
            'put_spread': put_spread,
            
            # Order book
            'call_bid_size': call_bid_size,
            'call_ask_size': call_ask_size,
            'put_bid_size': put_bid_size,
            'put_ask_size': put_ask_size,
            'call_imbalance': call_imbalance,
            'put_imbalance': put_imbalance,
            
            # Implied volatility
            'call_iv': call_iv if call_iv else None,
            'put_iv': put_iv if put_iv else None,
            'iv_spread': (call_iv - put_iv) if (call_iv and put_iv and call_iv > 0 and put_iv > 0) else None,
            
            # Greeks
            'call_delta': greeks_call.get('delta', 0),
            'call_gamma': greeks_call.get('gamma', 0),
            'call_vega': greeks_call.get('vega', 0),
            'call_theta': greeks_call.get('theta', 0),
            'put_delta': greeks_put.get('delta', 0),
            'put_gamma': greeks_put.get('gamma', 0),
            'put_vega': greeks_put.get('vega', 0),
            'put_theta': greeks_put.get('theta', 0),
            
            # Realized volatility
            'btc_realized_vol': btc_realized_vol,
            'call_realized_vol': call_realized_vol,
            'put_realized_vol': put_realized_vol,
            
            # Vol risk premium
            'call_vrp': call_vrp,
            'put_vrp': put_vrp,
            
            # Entropy
            'btc_entropy': btc_entropy,
            'call_entropy': call_entropy,
            'put_entropy': put_entropy,
            
            # Hurst exponent
            'btc_hurst': btc_hurst,
            'call_hurst': call_hurst,
            'put_hurst': put_hurst,
            
            # Derived metrics
            'put_call_ratio': (put_mid / call_mid) if (call_mid and put_mid and call_mid > 0) else None,
            'pcp_deviation': pcp_deviation if pcp_deviation != 0 else None,
            'synthetic_forward': (call_mid - put_mid + strike) if (call_mid and put_mid and strike) else None,
            
            # Buffer sizes (for diagnostics)
            'buffer_size': len(self.btc_price_history)
        }
        
        return snapshot
    
    def save_snapshot(self, snapshot: Dict):
        """Save snapshot to JSON file"""
        if not snapshot:
            return
        
        # Create filename with timestamp
        timestamp = datetime.fromisoformat(snapshot['timestamp'])
        date_str = timestamp.strftime('%Y-%m-%d')
        time_str = timestamp.strftime('%H-%M-%S')
        
        # Save to daily file (append)
        daily_file = f"{OUTPUT_DIR}/data_{date_str}.jsonl"
        
        try:
            with open(daily_file, 'a') as f:
                f.write(json.dumps(snapshot) + '\n')
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")
    
    def run(self):
        """Main collection loop"""
        logger.info("🚀 Starting data collection...")
        logger.info("=" * 80)
        
        iteration = 0
        
        while True:
            try:
                # Collect snapshot
                snapshot = self.collect_snapshot()
                
                if snapshot:
                    # Save to file
                    self.save_snapshot(snapshot)
                    
                    # Log progress every 60 seconds
                    iteration += 1
                    if iteration % 60 == 0:
                        call_iv_str = f"{snapshot['call_iv']:.2%}" if snapshot['call_iv'] else "N/A"
                        put_iv_str = f"{snapshot['put_iv']:.2%}" if snapshot['put_iv'] else "N/A"
                        moneyness_str = f"{snapshot['moneyness']:.4f}" if snapshot['moneyness'] else "N/A"
                        
                        logger.info(
                            f"✓ {iteration} snapshots | "
                            f"BTC: ${snapshot['btc_spot']:,.2f} | "
                            f"M: {moneyness_str} | "
                            f"CALL IV: {call_iv_str} | "
                            f"PUT IV: {put_iv_str} | "
                            f"Buf: {snapshot['buffer_size']}/300"
                        )
                
                # Sleep for 1 second
                time.sleep(1)
            
            except KeyboardInterrupt:
                logger.info("\n🛑 Data collection stopped by user")
                logger.info(f"📊 Total snapshots collected: {iteration}")
                break
            
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(1)


if __name__ == "__main__":
    collector = DataCollector()
    collector.run()
