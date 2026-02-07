#!/usr/bin/env python3
"""
Binary Options Price Behavior Monitor - 24/7
Tracks option price changes across three dimensions:
1. Distance from strike (BTC movement from period start)
2. Time to expiry
3. BTC volatility

Saves data to quickly accessible bins for pattern analysis
"""

import json
import time
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque, defaultdict
from typing import Optional, Tuple, Dict, List
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DimensionBins:
    """Define bins for each dimension"""
    
    # Distance from strike (in dollars)
    DISTANCE_BINS = [
        (0, 1, "0-1"),
        (1, 5, "1-5"),
        (5, 10, "5-10"),
        (10, 20, "10-20"),
        (20, 40, "20-40"),
        (40, 80, "40-80"),
        (80, 160, "80-160"),
        (160, 320, "160-320"),
        (320, 640, "320-640"),
        (640, 1280, "640-1280"),
        (1280, float('inf'), "1280+")
    ]
    
    # Time to expiry (in seconds)
    TIME_BINS = [
        (13*60, 15*60, "15m-13m"),
        (11*60, 13*60, "13m-11m"),
        (10*60, 11*60, "11m-10m"),
        (9*60, 10*60, "10m-9m"),
        (8*60, 9*60, "9m-8m"),
        (7*60, 8*60, "8m-7m"),
        (6*60, 7*60, "7m-6m"),
        (5*60, 6*60, "6m-5m"),
        (4*60, 5*60, "5m-4m"),
        (3*60, 4*60, "4m-3m"),
        (2*60, 3*60, "3m-2m"),
        (90, 120, "120s-90s"),
        (60, 90, "90s-60s"),
        (40, 60, "60s-40s"),
        (30, 40, "40s-30s"),
        (20, 30, "30s-20s"),
        (10, 20, "20s-10s"),
        (5, 10, "10s-5s"),
        (2, 5, "5s-2s"),
        (0, 2, "last-2s")
    ]
    
    # BTC volatility (in dollars - price range over last minute)
    VOLATILITY_BINS = [
        (0, 10, "0-10"),
        (10, 20, "10-20"),
        (20, 30, "20-30"),
        (30, 40, "30-40"),
        (40, 60, "40-60"),
        (60, 90, "60-90"),
        (90, 120, "90-120"),
        (120, float('inf'), "120+")
    ]
    
    @staticmethod
    def get_bin(value: float, bins: List[Tuple]) -> str:
        """Get bin label for a value"""
        for min_val, max_val, label in bins:
            if min_val <= value < max_val:
                return label
        return bins[-1][2]  # Return last bin if not found


class PriceObservation:
    """Single observation of option prices"""
    def __init__(self, timestamp: float, btc_price: float, strike_price: float,
                 put_bid: float, put_ask: float, call_bid: float, call_ask: float,
                 seconds_to_expiry: float, volatility: float):
        self.timestamp = timestamp
        self.btc_price = btc_price
        self.strike_price = strike_price
        self.distance_from_strike = abs(btc_price - strike_price)
        self.put_bid = put_bid
        self.put_ask = put_ask
        self.call_bid = call_bid
        self.call_ask = call_ask
        self.put_mid = (put_bid + put_ask) / 2
        self.call_mid = (call_bid + call_ask) / 2
        self.seconds_to_expiry = seconds_to_expiry
        self.volatility = volatility
        
        # Calculate bins
        self.distance_bin = DimensionBins.get_bin(self.distance_from_strike, DimensionBins.DISTANCE_BINS)
        self.time_bin = DimensionBins.get_bin(seconds_to_expiry, DimensionBins.TIME_BINS)
        self.volatility_bin = DimensionBins.get_bin(volatility, DimensionBins.VOLATILITY_BINS)
        
        # Create composite key
        self.bin_key = f"{self.distance_bin}|{self.time_bin}|{self.volatility_bin}"


class BinData:
    """Accumulated data for a specific bin"""
    def __init__(self):
        self.count = 0
        self.put_prices = []
        self.call_prices = []
        self.put_sum = 0.0
        self.call_sum = 0.0
        self.put_min = float('inf')
        self.put_max = 0.0
        self.call_min = float('inf')
        self.call_max = 0.0
    
    def add_observation(self, put_price: float, call_price: float):
        """Add new observation to bin"""
        self.count += 1
        self.put_prices.append(put_price)
        self.call_prices.append(call_price)
        self.put_sum += put_price
        self.call_sum += call_price
        self.put_min = min(self.put_min, put_price)
        self.put_max = max(self.put_max, put_price)
        self.call_min = min(self.call_min, call_price)
        self.call_max = max(self.call_max, call_price)
        
        # Keep only last 1000 observations per bin to manage memory
        if len(self.put_prices) > 1000:
            removed_put = self.put_prices.pop(0)
            removed_call = self.call_prices.pop(0)
            self.put_sum -= removed_put
            self.call_sum -= removed_call
    
    def get_stats(self) -> Dict:
        """Get statistics for this bin"""
        if self.count == 0:
            return {}
        
        return {
            'count': self.count,
            'put': {
                'avg': self.put_sum / len(self.put_prices),
                'min': self.put_min,
                'max': self.put_max,
                'std': float(np.std(self.put_prices)) if len(self.put_prices) > 1 else 0.0
            },
            'call': {
                'avg': self.call_sum / len(self.call_prices),
                'min': self.call_min,
                'max': self.call_max,
                'std': float(np.std(self.call_prices)) if len(self.call_prices) > 1 else 0.0
            }
        }


class OptionPriceMonitor:
    """24/7 monitoring of option price behavior across multiple dimensions"""
    
    def __init__(
        self,
        put_file: str = "/home/ubuntu/013_2025_polymarket/15M_PUT.json",
        call_file: str = "/home/ubuntu/013_2025_polymarket/15M_CALL.json",
        btc_file: str = "/home/ubuntu/013_2025_polymarket/bybit_btc_price.json",
        data_dir: str = "/home/ubuntu/013_2025_polymarket/bot016_react/monitor_data"
    ):
        self.put_file = put_file
        self.call_file = call_file
        self.btc_file = btc_file
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # BTC price at period start (strike price)
        self.period_start_btc: Optional[float] = None
        self.period_start_minute: Optional[int] = None
        
        # Price history for volatility calculation
        self.btc_price_history = deque(maxlen=600)  # 1 minute @ 10Hz
        
        # Bin storage
        self.bins: Dict[str, BinData] = defaultdict(BinData)
        
        # Stats
        self.total_observations = 0
        self.current_period_obs = 0
        self.last_save_time = time.time()
        self.save_interval = 30  # Save every 30 seconds
        
        # Load existing data
        self.load_existing_data()
        
        logger.info("="*80)
        logger.info("📊 OPTION PRICE BEHAVIOR MONITOR - 24/7")
        logger.info("="*80)
        logger.info("Tracking price behavior across 3 dimensions:")
        logger.info(f"  Distance bins: {len(DimensionBins.DISTANCE_BINS)}")
        logger.info(f"  Time bins: {len(DimensionBins.TIME_BINS)}")
        logger.info(f"  Volatility bins: {len(DimensionBins.VOLATILITY_BINS)}")
        logger.info(f"  Total combinations: {len(DimensionBins.DISTANCE_BINS) * len(DimensionBins.TIME_BINS) * len(DimensionBins.VOLATILITY_BINS)}")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info("="*80)
    
    def get_data_file(self) -> Path:
        """Get current data file path"""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.data_dir / f"monitor_data_{today}.json"
    
    def load_existing_data(self):
        """Load existing data for today"""
        data_file = self.get_data_file()
        if data_file.exists():
            try:
                with open(data_file, 'r') as f:
                    data = json.load(f)
                
                self.total_observations = data.get('total_observations', 0)
                
                # Reload bins
                for bin_key, bin_stats in data.get('bins', {}).items():
                    bin_data = BinData()
                    bin_data.count = bin_stats['count']
                    bin_data.put_sum = bin_stats['put']['avg'] * bin_stats['count']
                    bin_data.call_sum = bin_stats['call']['avg'] * bin_stats['count']
                    bin_data.put_min = bin_stats['put']['min']
                    bin_data.put_max = bin_stats['put']['max']
                    bin_data.call_min = bin_stats['call']['min']
                    bin_data.call_max = bin_stats['call']['max']
                    self.bins[bin_key] = bin_data
                
                logger.info(f"📥 Loaded {len(self.bins)} bins, {self.total_observations} total observations")
            except Exception as e:
                logger.error(f"Error loading data: {e}")
    
    def save_data(self):
        """Save current data to file"""
        data_file = self.get_data_file()
        
        data = {
            'date': datetime.now().strftime("%Y-%m-%d"),
            'last_updated': datetime.now().isoformat(),
            'total_observations': self.total_observations,
            'total_bins': len(self.bins),
            'bins': {}
        }
        
        # Save bin statistics
        for bin_key, bin_data in self.bins.items():
            stats = bin_data.get_stats()
            if stats:
                data['bins'][bin_key] = stats
        
        try:
            # Write to temp file first, then rename (atomic)
            temp_file = data_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            temp_file.rename(data_file)
        except Exception as e:
            logger.error(f"Error saving data: {e}")
    
    def save_bin_index(self):
        """Save quick-lookup index of all bins"""
        index_file = self.data_dir / "bin_index.json"
        
        index = {
            'dimensions': {
                'distance': [label for _, _, label in DimensionBins.DISTANCE_BINS],
                'time': [label for _, _, label in DimensionBins.TIME_BINS],
                'volatility': [label for _, _, label in DimensionBins.VOLATILITY_BINS]
            },
            'populated_bins': list(self.bins.keys()),
            'bin_counts': {k: v.count for k, v in self.bins.items()}
        }
        
        try:
            with open(index_file, 'w') as f:
                json.dump(index, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def read_json_file(self, filepath: str) -> Optional[dict]:
        """Safely read JSON file"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except:
            return None
    
    def calculate_btc_volatility(self) -> float:
        """Calculate BTC volatility over last minute"""
        if len(self.btc_price_history) < 10:
            return 0.0
        
        prices = list(self.btc_price_history)
        return max(prices) - min(prices)
    
    def check_period_start(self) -> bool:
        """Check if new period started and capture strike price"""
        now = datetime.now()
        current_minute = now.minute
        
        period_start_minutes = [0, 15, 30, 45]
        for start_min in period_start_minutes:
            if current_minute >= start_min and current_minute < start_min + 15:
                seconds_into_period = (current_minute - start_min) * 60 + now.second
                
                # New period detected
                if self.period_start_minute != start_min:
                    logger.info(f"\n{'='*80}")
                    logger.info(f"🔄 NEW PERIOD DETECTED - :{start_min:02d}")
                    logger.info(f"{'='*80}")
                    
                    # Capture BTC price at period start
                    btc_data = self.read_json_file(self.btc_file)
                    if btc_data:
                        self.period_start_btc = btc_data['price']
                        logger.info(f"📍 Strike Price (BTC at start): ${self.period_start_btc:,.2f}")
                    
                    self.period_start_minute = start_min
                    self.current_period_obs = 0
                    
                    # Save data at period change
                    self.save_data()
                    self.save_bin_index()
                    
                    return True
        
        return False
    
    def get_seconds_to_expiry(self) -> float:
        """Get seconds remaining in current period"""
        now = datetime.now()
        current_minute = now.minute
        
        period_start_minutes = [0, 15, 30, 45]
        for start_min in period_start_minutes:
            if current_minute >= start_min and current_minute < start_min + 15:
                seconds_into_period = (current_minute - start_min) * 60 + now.second
                return 900 - seconds_into_period
        
        return 0
    
    def record_observation(self):
        """Record current market state"""
        if self.period_start_btc is None:
            return
        
        # Read current prices
        put_data = self.read_json_file(self.put_file)
        call_data = self.read_json_file(self.call_file)
        btc_data = self.read_json_file(self.btc_file)
        
        if not all([put_data, call_data, btc_data]):
            return
        
        try:
            put_bid = put_data['best_bid']
            put_ask = put_data['best_ask']
            call_bid = call_data['best_bid']
            call_ask = call_data['best_ask']
            
            if not all([put_bid, put_ask, call_bid, call_ask]):
                return
            
            current_btc = btc_data['price']
            
            # Update BTC history for volatility
            self.btc_price_history.append(current_btc)
            
            # Calculate dimensions
            seconds_to_expiry = self.get_seconds_to_expiry()
            volatility = self.calculate_btc_volatility()
            
            # Create observation
            obs = PriceObservation(
                timestamp=time.time(),
                btc_price=current_btc,
                strike_price=self.period_start_btc,
                put_bid=put_bid.get('price', 0.0),
                put_ask=put_ask.get('price', 0.0),
                call_bid=call_bid.get('price', 0.0),
                call_ask=call_ask.get('price', 0.0),
                seconds_to_expiry=seconds_to_expiry,
                volatility=volatility
            )
            
            # Validate prices
            if not (0 <= obs.put_mid <= 1 and 0 <= obs.call_mid <= 1):
                return
            
            # Add to appropriate bin
            self.bins[obs.bin_key].add_observation(obs.put_mid, obs.call_mid)
            
            self.total_observations += 1
            self.current_period_obs += 1
            
        except Exception as e:
            logger.debug(f"Error recording observation: {e}")
    
    def print_status(self):
        """Print current monitoring status"""
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 MONITOR STATUS - {datetime.now().strftime('%H:%M:%S')}")
        logger.info(f"{'='*80}")
        logger.info(f"Total Observations: {self.total_observations:,}")
        logger.info(f"Period Observations: {self.current_period_obs:,}")
        logger.info(f"Active Bins: {len(self.bins):,}")
        
        if self.period_start_btc:
            btc_data = self.read_json_file(self.btc_file)
            if btc_data:
                current_btc = btc_data['price']
                distance = abs(current_btc - self.period_start_btc)
                volatility = self.calculate_btc_volatility()
                seconds_left = self.get_seconds_to_expiry()
                
                logger.info(f"\nCurrent State:")
                logger.info(f"  Strike: ${self.period_start_btc:,.2f}")
                logger.info(f"  BTC Now: ${current_btc:,.2f}")
                logger.info(f"  Distance: ${distance:.2f}")
                logger.info(f"  Volatility: ${volatility:.2f}")
                logger.info(f"  Time Left: {seconds_left:.0f}s ({seconds_left/60:.1f}m)")
        
        # Show top 5 most populated bins
        if self.bins:
            sorted_bins = sorted(self.bins.items(), key=lambda x: x[1].count, reverse=True)[:5]
            logger.info(f"\nTop 5 Most Populated Bins:")
            for bin_key, bin_data in sorted_bins:
                stats = bin_data.get_stats()
                logger.info(f"  {bin_key}: {stats['count']} obs")
                logger.info(f"    PUT: {stats['put']['avg']:.3f} ± {stats['put']['std']:.3f}")
                logger.info(f"    CALL: {stats['call']['avg']:.3f} ± {stats['call']['std']:.3f}")
        
        logger.info(f"{'='*80}\n")
    
    def run(self):
        """Main monitoring loop"""
        logger.info("\n🚀 Starting 24/7 Option Price Monitor\n")
        
        iteration = 0
        last_status = time.time()
        
        try:
            while True:
                # Check for new period
                self.check_period_start()
                
                # Record observation
                self.record_observation()
                
                # Auto-save every 30 seconds
                if time.time() - self.last_save_time >= self.save_interval:
                    self.save_data()
                    self.save_bin_index()
                    self.last_save_time = time.time()
                
                # Status update every 60 seconds
                if time.time() - last_status >= 60:
                    self.print_status()
                    last_status = time.time()
                
                iteration += 1
                time.sleep(0.1)  # 10Hz sampling
        
        except KeyboardInterrupt:
            logger.info("\n\n⏸️  Monitor stopped by user")
            logger.info("💾 Saving final data...")
            self.save_data()
            self.save_bin_index()
            logger.info(f"✅ Data saved to: {self.get_data_file()}")


def main():
    """Entry point"""
    monitor = OptionPriceMonitor()
    monitor.run()


if __name__ == "__main__":
    main()
