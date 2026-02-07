#!/usr/bin/env python3
"""
Binary Options LAG Sensitivity Monitor - 24/7
Measures how option prices REACT to BTC movements with 2-second lag

Question: If BTC moves by $X, how much does the option price move in the NEXT 2 seconds?

This is different from instantaneous correlation - we're measuring market reaction time.

pm2 start bot_C2_2.py --cron-restart="00 * * * *" --interpreter python3

"""

import json
import time
import numpy as np
from datetime import datetime
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
        (0, 1, "0-1"), (1, 5, "1-5"), (5, 10, "5-10"), (10, 20, "10-20"),
        (20, 40, "20-40"), (40, 80, "40-80"), (80, 160, "80-160"),
        (160, 320, "160-320"), (320, 640, "320-640"), (640, 1280, "640-1280"),
        (1280, float('inf'), "1280+")
    ]

    # Time to expiry (in seconds)
    TIME_BINS = [
        (13*60, 15*60, "15m-13m"), (11*60, 13*60, "13m-11m"), (10*60, 11*60, "11m-10m"),
        (9*60, 10*60, "10m-9m"), (8*60, 9*60, "9m-8m"), (7*60, 8*60, "8m-7m"),
        (6*60, 7*60, "7m-6m"), (5*60, 6*60, "6m-5m"), (4*60, 5*60, "5m-4m"),
        (3*60, 4*60, "4m-3m"), (2*60, 3*60, "3m-2m"), (90, 120, "120s-90s"),
        (60, 90, "90s-60s"), (40, 60, "60s-40s"), (30, 40, "40s-30s"),
        (20, 30, "30s-20s"), (10, 20, "20s-10s"), (5, 10, "10s-5s"),
        (2, 5, "5s-2s"), (0, 2, "last-2s")
    ]

    # BTC volatility (in dollars - price range over last minute)
    VOLATILITY_BINS = [
        (0, 10, "0-10"), (10, 20, "10-20"), (20, 30, "20-30"), (30, 40, "30-40"),
        (40, 60, "40-60"), (60, 90, "60-90"), (90, 120, "90-120"),(120, 240, "120-240"),
        (240, float('inf'), "240+")
    ]

    @staticmethod
    def get_bin(value: float, bins: List[Tuple]) -> str:
        """Get bin label for a value"""
        for min_val, max_val, label in bins:
            if min_val <= value < max_val:
                return label
        return bins[-1][2]


class PriceSnapshot:
    """Single price observation"""
    def __init__(self, timestamp: float, btc: float, put: float, call: float):
        self.timestamp = timestamp
        self.btc = btc
        self.put = put
        self.call = call


class LagSensitivityData:
    """Accumulated lag sensitivity data for a specific bin"""
    def __init__(self):
        self.lag_sensitivities_put = []  # Option change per $1 BTC move (with 2s lag)
        self.lag_sensitivities_call = []

    def add_lag_measurement(self, btc_move: float, put_reaction: float, call_reaction: float):
        """
        Add lag sensitivity measurement

        btc_move: How much BTC moved (e.g., +$5)
        put_reaction: How much PUT moved in the NEXT 2 seconds
        call_reaction: How much CALL moved in the NEXT 2 seconds

        Sensitivity = reaction / btc_move
        """
        if abs(btc_move) < 1.0:  # Only measure on significant moves
            return

        put_sensitivity = put_reaction / btc_move
        call_sensitivity = call_reaction / btc_move

        # Filter extreme outliers (likely bad data)
        if abs(put_sensitivity) > 0.5 or abs(call_sensitivity) > 0.5:
            return

        self.lag_sensitivities_put.append(put_sensitivity)
        self.lag_sensitivities_call.append(call_sensitivity)

        # Keep last 500
        if len(self.lag_sensitivities_put) > 500:
            self.lag_sensitivities_put.pop(0)
            self.lag_sensitivities_call.pop(0)

    def get_stats(self) -> Dict:
        """Get sensitivity statistics"""
        if not self.lag_sensitivities_put:
            return {}

        # Filter out zeros (0.0, -0.0) from raw data
        put_sens_nonzero = [x for x in self.lag_sensitivities_put if abs(x) > 1e-10]
        call_sens_nonzero = [x for x in self.lag_sensitivities_call if abs(x) > 1e-10]

        # If no non-zero data, return empty
        if not put_sens_nonzero or not call_sens_nonzero:
            return {}

        return {
            'count': len(put_sens_nonzero),
            'put_sensitivity': {
                'avg': float(np.mean(put_sens_nonzero)),
                'median': float(np.median(put_sens_nonzero)),
                'std': float(np.std(put_sens_nonzero)),
                'min': float(np.min(put_sens_nonzero)),
                'max': float(np.max(put_sens_nonzero)),
                'p25': float(np.percentile(put_sens_nonzero, 25)),
                'p75': float(np.percentile(put_sens_nonzero, 75))
            },
            'call_sensitivity': {
                'avg': float(np.mean(call_sens_nonzero)),
                'median': float(np.median(call_sens_nonzero)),
                'std': float(np.std(call_sens_nonzero)),
                'min': float(np.min(call_sens_nonzero)),
                'max': float(np.max(call_sens_nonzero)),
                'p25': float(np.percentile(call_sens_nonzero, 25)),
                'p75': float(np.percentile(call_sens_nonzero, 75))
            },
            'put_sensitivity_raw': put_sens_nonzero,
            'call_sensitivity_raw': call_sens_nonzero
        }


class LagSensitivityMonitor:
    """Monitor option price LAG reactions to BTC movements"""

    def __init__(
        self,
        put_file: str = "/home/ubuntu/013_2025_polymarket/15M_PUT.json",
        call_file: str = "/home/ubuntu/013_2025_polymarket/15M_CALL.json",
        btc_file: str = "/home/ubuntu/013_2025_polymarket/bybit_btc_price.json",
        data_dir: str = "/home/ubuntu/013_2025_polymarket/bot016_react/sensitivity_data"
    ):
        self.put_file = put_file
        self.call_file = call_file
        self.btc_file = btc_file
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Single cumulative data file
        self.data_file = self.data_dir / "sensitivity_master.json"

        # Period tracking
        self.period_start_btc: Optional[float] = None
        self.period_start_minute: Optional[int] = None

        # Price history for lag detection (need snapshots with timestamps)
        self.price_snapshots = deque(maxlen=200)  # 20 seconds @ 10Hz

        # BTC price history for volatility
        self.btc_price_history = deque(maxlen=600)  # 1 minute

        # Sensitivity bins
        self.bins: Dict[str, LagSensitivityData] = defaultdict(LagSensitivityData)

        # Stats
        self.total_measurements = 0
        self.last_save_time = time.time()
        self.save_interval = 30

        # Load existing cumulative data
        self.load_cumulative_data()

        logger.info("="*80)
        logger.info("📊 LAG SENSITIVITY MONITOR - 24/7")
        logger.info("="*80)
        logger.info("Measuring: Option price reaction to BTC moves (2-second lag)")
        logger.info(f"  Distance bins: {len(DimensionBins.DISTANCE_BINS)}")
        logger.info(f"  Time bins: {len(DimensionBins.TIME_BINS)}")
        logger.info(f"  Volatility bins: {len(DimensionBins.VOLATILITY_BINS)}")
        logger.info(f"Master data file: {self.data_file}")
        logger.info("="*80)

    def load_cumulative_data(self):
        """Load existing cumulative data"""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)

                self.total_measurements = data.get('total_measurements', 0)

                # Reload bins
                for bin_key, bin_stats in data.get('bins', {}).items():
                    bin_data = LagSensitivityData()
                    put_sens = bin_stats.get('put_sensitivity_raw', [])
                    call_sens = bin_stats.get('call_sensitivity_raw', [])
                    bin_data.lag_sensitivities_put = put_sens[-500:]
                    bin_data.lag_sensitivities_call = call_sens[-500:]
                    self.bins[bin_key] = bin_data

                logger.info(f"📥 Loaded cumulative data: {len(self.bins)} bins, {self.total_measurements:,} measurements")
            except Exception as e:
                logger.error(f"Error loading data: {e}")

    def save_cumulative_data(self):
        """Save cumulative data"""
        data = {
            'last_updated': datetime.now().isoformat(),
            'total_measurements': self.total_measurements,
            'total_bins': len(self.bins),
            'bins': {}
        }

        for bin_key, bin_data in self.bins.items():
            stats = bin_data.get_stats()
            if stats:
                data['bins'][bin_key] = stats

        try:
            temp_file = self.data_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            temp_file.rename(self.data_file)
        except Exception as e:
            logger.error(f"Error saving data: {e}")

    def save_bin_index(self):
        """Save quick-lookup index"""
        index_file = self.data_dir / "sensitivity_index.json"

        index = {
            'last_updated': datetime.now().isoformat(),
            'dimensions': {
                'distance': [label for _, _, label in DimensionBins.DISTANCE_BINS],
                'time': [label for _, _, label in DimensionBins.TIME_BINS],
                'volatility': [label for _, _, label in DimensionBins.VOLATILITY_BINS]
            },
            'total_measurements': self.total_measurements,
            'populated_bins': list(self.bins.keys()),
            'bin_counts': {k: len(v.lag_sensitivities_put) for k, v in self.bins.items()}
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

    def get_strike_price_from_bybit(self, start_min: int) -> Optional[float]:
        """Get exact opening price from Bybit API"""
        try:
            import requests
            from datetime import timezone

            now = datetime.now(timezone.utc)
            period_start = now.replace(minute=start_min, second=0, microsecond=0)
            start_timestamp = int(period_start.timestamp() * 1000)

            url = "https://api.bybit.com/v5/market/mark-price-kline"
            params = {
                'category': 'linear',
                'symbol': 'BTCUSDT',
                'interval': '15',
                'start': start_timestamp,
                'limit': 1
            }

            response = requests.get(url, params=params, timeout=5)

            if response.status_code == 200:
                data = response.json()
                if data.get('retCode') == 0:
                    kline_list = data.get('result', {}).get('list', [])
                    if kline_list:
                        return float(kline_list[0][1])  # open price
            return None
        except Exception as e:
            logger.debug(f"Error fetching strike: {e}")
            return None

    def calculate_btc_volatility(self) -> float:
        """Calculate BTC volatility over last minute"""
        if len(self.btc_price_history) < 10:
            return 0.0
        prices = list(self.btc_price_history)
        return max(prices) - min(prices)

    def check_period_start(self) -> bool:
        """Check if new period started"""
        now = datetime.now()
        current_minute = now.minute

        period_start_minutes = [0, 15, 30, 45]
        for start_min in period_start_minutes:
            if current_minute >= start_min and current_minute < start_min + 15:
                if self.period_start_minute != start_min:
                    logger.info(f"\n{'='*80}")
                    logger.info(f"🔄 NEW PERIOD - :{start_min:02d}")
                    logger.info(f"{'='*80}")

                    strike_from_api = self.get_strike_price_from_bybit(start_min)
                    if strike_from_api:
                        self.period_start_btc = strike_from_api
                        logger.info(f"📍 Strike: ${self.period_start_btc:,.2f}")
                    else:
                        btc_data = self.read_json_file(self.btc_file)
                        if btc_data:
                            self.period_start_btc = btc_data['price']
                            logger.info(f"📍 Strike (fallback): ${self.period_start_btc:,.2f}")

                    self.period_start_minute = start_min
                    self.price_snapshots.clear()
                    self.save_cumulative_data()
                    self.save_bin_index()
                    return True
        return False

    def get_seconds_to_expiry(self) -> float:
        """Get seconds remaining"""
        now = datetime.now()
        current_minute = now.minute

        for start_min in [0, 15, 30, 45]:
            if current_minute >= start_min and current_minute < start_min + 15:
                seconds_into_period = (current_minute - start_min) * 60 + now.second
                return 900 - seconds_into_period
        return 0

    def get_bin_key(self, distance: float, seconds_to_expiry: float, volatility: float) -> str:
        """Get bin key"""
        dist_label = DimensionBins.get_bin(distance, DimensionBins.DISTANCE_BINS)
        time_label = DimensionBins.get_bin(seconds_to_expiry, DimensionBins.TIME_BINS)
        vol_label = DimensionBins.get_bin(volatility, DimensionBins.VOLATILITY_BINS)
        return f"{dist_label}|{time_label}|{vol_label}"

    def record_snapshot(self):
        """Record current prices"""
        if not self.period_start_btc:
            return

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

            btc_price = btc_data['price']
            put_mid = (put_bid['price'] + put_ask['price']) / 2
            call_mid = (call_bid['price'] + call_ask['price']) / 2

            if not (0 <= put_mid <= 1 and 0 <= call_mid <= 1):
                return

            # Add snapshot
            snapshot = PriceSnapshot(time.time(), btc_price, put_mid, call_mid)
            self.price_snapshots.append(snapshot)
            self.btc_price_history.append(btc_price)

            # Measure lag: If BTC moved 2 seconds ago, how much did options move since then?
            if len(self.price_snapshots) >= 20:  # 2 seconds @ 10Hz
                snapshot_now = self.price_snapshots[-1]
                snapshot_2s_ago = self.price_snapshots[-20]

                btc_move = snapshot_now.btc - snapshot_2s_ago.btc
                put_reaction = snapshot_now.put - snapshot_2s_ago.put
                call_reaction = snapshot_now.call - snapshot_2s_ago.call

                # Only measure if BTC actually moved
                if abs(btc_move) >= 1.0:
                    # Calculate dimensions at time of BTC move (2s ago)
                    distance = abs(snapshot_2s_ago.btc - self.period_start_btc)
                    seconds_remaining = self.get_seconds_to_expiry() + 2  # Was 2s ago
                    volatility = self.calculate_btc_volatility()

                    bin_key = self.get_bin_key(distance, seconds_remaining, volatility)

                    # Record the lag sensitivity
                    self.bins[bin_key].add_lag_measurement(btc_move, put_reaction, call_reaction)
                    self.total_measurements += 1

        except Exception as e:
            logger.debug(f"Error recording: {e}")

    def print_status(self):
        """Print status"""
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 STATUS - {datetime.now().strftime('%H:%M:%S')}")
        logger.info(f"{'='*80}")
        logger.info(f"Total Measurements: {self.total_measurements:,}")
        logger.info(f"Active Bins: {len(self.bins):,}")

        if self.period_start_btc and len(self.price_snapshots) > 0:
            current = self.price_snapshots[-1]
            distance = abs(current.btc - self.period_start_btc)
            volatility = self.calculate_btc_volatility()
            seconds_left = self.get_seconds_to_expiry()

            logger.info(f"\nCurrent:")
            logger.info(f"  Strike: ${self.period_start_btc:,.2f}")
            logger.info(f"  BTC: ${current.btc:,.2f}")
            logger.info(f"  Distance: ${distance:.2f}")
            logger.info(f"  Vol: ${volatility:.2f}")
            logger.info(f"  Time: {seconds_left:.0f}s")

        # Top bins
        if self.bins:
            sorted_bins = sorted(self.bins.items(),
                                key=lambda x: len(x[1].lag_sensitivities_put),
                                reverse=True)[:5]
            logger.info(f"\nTop 5 Bins:")
            for bin_key, bin_data in sorted_bins:
                stats = bin_data.get_stats()
                if stats:
                    logger.info(f"  {bin_key}: {stats['count']} measurements")
                    logger.info(f"    PUT: {stats['put_sensitivity']['median']:+.6f}")
                    logger.info(f"    CALL: {stats['call_sensitivity']['median']:+.6f}")

        logger.info(f"{'='*80}\n")

    def run(self):
        """Main loop"""
        logger.info("\n🚀 Starting Lag Sensitivity Monitor\n")

        last_status = time.time()

        try:
            while True:
                self.check_period_start()
                self.record_snapshot()

                if time.time() - self.last_save_time >= self.save_interval:
                    self.save_cumulative_data()
                    self.save_bin_index()
                    self.last_save_time = time.time()

                if time.time() - last_status >= 60:
                    self.print_status()
                    last_status = time.time()

                time.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("\n\n⏸️  Stopped")
            logger.info("💾 Saving...")
            self.save_cumulative_data()
            self.save_bin_index()
            logger.info(f"✅ Saved to: {self.data_file}")


def main():
    monitor = LagSensitivityMonitor()
    monitor.run()


if __name__ == "__main__":
    main()
