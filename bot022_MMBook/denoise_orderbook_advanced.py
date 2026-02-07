#!/usr/bin/env python3
"""
Advanced Order Book Denoising with Market Maker Detection
- Learns MM size during 30-second calibration period at start of each 15-min window
- Detects MM based on proximity to best bid/ask (0.01-0.03 away)
- Memorizes MM size across periods
- Only operates in normal price range (0.10 < price < 0.90)
"""

import json
import copy
import time
import sys
from typing import Dict, List, Tuple, Optional
from statistics import mean, median
from datetime import datetime, timedelta
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================
UPDATE_INTERVAL = 0.5  # Update every 0.5 seconds
INPUT_DIR = '/home/ubuntu/013_2025_polymarket/'  # Same directory, set to your path if different
OUTPUT_SUFFIX = '_nonoise'

# Market Maker Detection Parameters
CALIBRATION_PERIOD = 30  # Wait 30 seconds at start of each period
MM_DISTANCE_FROM_BEST = (0.01, 0.03)  # MM typically 1-3 cents from best (range)
MM_SIZE_THRESHOLD = 1.5  # MM size must be 1.5x+ larger than neighbors (lowered from 2.5)
NEIGHBOR_RANGE = 0.05  # Look at neighbors within 5 cents
NUM_NEIGHBORS_TO_CHECK = 5  # Check 4-5 neighboring levels
NORMAL_PRICE_RANGE = (0.10, 0.90)  # Only detect MM in this range

# Memory settings
MM_MEMORY_DECAY = 0.95  # Slowly decay old MM estimates (95% retention)
MIN_CONFIDENCE_LEVEL = 3  # Need at least 3 observations to be confident
# ============================================================================

class MarketMakerMemory:
    """Stores and manages market maker size estimates across periods"""

    def __init__(self):
        self.call_mm_sizes = defaultdict(list)  # price -> [sizes]
        self.put_mm_sizes = defaultdict(list)
        self.call_mm_estimate = {}  # price -> estimated_size
        self.put_mm_estimate = {}
        self.last_calibration = None
        self.in_calibration = False

    def start_calibration(self):
        """Begin a new calibration period"""
        self.in_calibration = True
        self.last_calibration = datetime.now()
        print(f"\n🔬 CALIBRATION STARTED at {self.last_calibration.strftime('%H:%M:%S')}")
        print(f"   Learning market maker patterns for {CALIBRATION_PERIOD} seconds...")

    def end_calibration(self, asset_type: str):
        """End calibration and compute MM estimates"""
        self.in_calibration = False

        if asset_type == 'CALL':
            self._compute_estimates(self.call_mm_sizes, self.call_mm_estimate, 'CALL')
        else:
            self._compute_estimates(self.put_mm_sizes, self.put_mm_estimate, 'PUT')

    def _compute_estimates(self, observations: dict, estimates: dict, asset_type: str):
        """Compute robust MM size estimates from observations"""
        print(f"\n📊 Computing {asset_type} MM estimates:")

        for price, sizes in observations.items():
            if len(sizes) >= MIN_CONFIDENCE_LEVEL:
                # Use median for robustness
                estimate = median(sizes)
                estimates[price] = estimate
                print(f"   ${price:.2f}: {estimate:.2f} (from {len(sizes)} observations)")

        # Clear observations for next period
        observations.clear()

        if not estimates:
            print(f"   ⚠️  No confident MM estimates found")

    def add_observation(self, asset_type: str, price: float, size: float):
        """Add a MM size observation during calibration"""
        if asset_type == 'CALL':
            self.call_mm_sizes[price].append(size)
        else:
            self.put_mm_sizes[price].append(size)

    def get_mm_size(self, asset_type: str, price: float) -> Optional[float]:
        """Get estimated MM size for a price level"""
        estimates = self.call_mm_estimate if asset_type == 'CALL' else self.put_mm_estimate
        return estimates.get(price)

    def apply_decay(self):
        """Apply decay to old MM estimates"""
        for price in self.call_mm_estimate:
            self.call_mm_estimate[price] *= MM_MEMORY_DECAY
        for price in self.put_mm_estimate:
            self.put_mm_estimate[price] *= MM_MEMORY_DECAY

    def should_calibrate(self) -> bool:
        """Check if we should start a new calibration period"""
        if self.in_calibration:
            return False

        now = datetime.now()

        # First run
        if self.last_calibration is None:
            return True

        # Check if we're at the start of a 15-minute window
        # (minutes: 00, 15, 30, 45)
        current_minute = now.minute
        if current_minute % 15 == 0:
            # Check if enough time has passed since last calibration
            time_since_last = (now - self.last_calibration).total_seconds()
            if time_since_last > 60:  # At least 1 minute since last
                return True

        return False

    def is_calibrating(self, start_time: datetime) -> bool:
        """Check if still in calibration period"""
        if not self.in_calibration:
            return False
        elapsed = (datetime.now() - start_time).total_seconds()
        return elapsed < CALIBRATION_PERIOD


def load_orderbook(filepath: str) -> Dict:
    """Load orderbook data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_orderbook(data: Dict, filepath: str):
    """Save denoised orderbook data to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def get_neighbor_sizes(orders: List[Dict], target_price: float,
                       num_neighbors: int = NUM_NEIGHBORS_TO_CHECK) -> List[float]:
    """Get sizes of neighboring price levels"""
    prices_and_sizes = [(float(o['price']), float(o['size'])) for o in orders]
    prices_and_sizes.sort(key=lambda x: abs(x[0] - target_price))

    # Skip the target price itself, take next N neighbors
    neighbors = [size for price, size in prices_and_sizes[1:num_neighbors+1]]
    return neighbors


def detect_market_maker_level(orders: List[Dict], best_price: float,
                               side: str, mm_memory: MarketMakerMemory,
                               asset_type: str) -> List[Tuple[str, float]]:
    """
    Detect market maker levels based on:
    1. Distance from best bid/ask (0.01-0.03)
    2. Size significantly larger than neighbors
    3. Price in normal range (0.10 < price < 0.90)

    For BIDS: MM is BELOW best bid (best_bid - price = 0.01-0.03)
    For ASKS: MM is ABOVE best ask (price - best_ask = 0.01-0.03)
    """
    mm_levels = []

    for order in orders:
        price = float(order['price'])
        size = float(order['size'])

        # Rule 1: Must be in normal price range
        if not (NORMAL_PRICE_RANGE[0] < price < NORMAL_PRICE_RANGE[1]):
            continue

        # Rule 2: Must be close to best bid/ask (0.01 to 0.03 away)
        # For bids: MM places below best bid
        # For asks: MM places above best ask
        if side == "BIDS":
            distance = best_price - price  # How much below best bid
        else:  # ASKS
            distance = price - best_price  # How much above best ask

        if not (MM_DISTANCE_FROM_BEST[0] <= distance <= MM_DISTANCE_FROM_BEST[1]):
            continue

        # Rule 3: Must be significantly larger than neighbors
        neighbors = get_neighbor_sizes(orders, price)
        if not neighbors:
            continue

        avg_neighbor = mean(neighbors)
        if size > MM_SIZE_THRESHOLD * avg_neighbor:
            mm_levels.append((f"{price:.2f}", size))

            # During calibration, record observations
            if mm_memory.in_calibration:
                mm_memory.add_observation(asset_type, price, size)
                if side == "BIDS":  # Only print once per asset
                    print(f"    ✓ MM at ${price:.2f}: size={size:.2f}, neighbors_avg={avg_neighbor:.2f}, ratio={size/avg_neighbor:.1f}x, dist={distance:.2f}")

    return mm_levels


def denoise_side_advanced(orders: List[Dict], side_name: str, best_price: float,
                         mm_memory: MarketMakerMemory, asset_type: str) -> List[Dict]:
    """
    Advanced denoising using learned MM patterns
    """
    if not orders:
        return orders

    denoised = []

    # Detect current MM levels
    mm_levels = detect_market_maker_level(orders, best_price, side_name, mm_memory, asset_type)

    if not mm_memory.in_calibration and mm_levels:
        print(f"  {asset_type} {side_name} - Detected {len(mm_levels)} MM levels")

    for order in orders:
        price = order['price']
        price_float = float(price)
        size = float(order['size'])

        # Check if this is a known MM level
        mm_size = mm_memory.get_mm_size(asset_type, price_float)

        if mm_size and mm_size > 10:  # Only subtract if MM size is significant
            # Subtract the learned MM size
            clean_size = max(size - mm_size, 0)
            if not mm_memory.in_calibration and clean_size < size:
                print(f"    💥 Deducting MM at ${price}: {size:.2f} -> {clean_size:.2f} (MM={mm_size:.2f})")

            if clean_size > 0:
                denoised.append({
                    'price': price,
                    'size': f"{clean_size:.2f}"
                })
        else:
            # No MM detected, keep as is
            denoised.append(order)

    return denoised


def denoise_orderbook_advanced(data: Dict, mm_memory: MarketMakerMemory) -> Dict:
    """Main denoising function with MM learning"""
    denoised_data = copy.deepcopy(data)
    asset_type = 'CALL' if 'CALL' in data['asset_name'] else 'PUT'

    bids = data['complete_book']['bids']
    asks = data['complete_book']['asks']

    # Get best prices
    best_bid = data['best_bid']['price'] if data.get('best_bid') else 0.5
    best_ask = data['best_ask']['price'] if data.get('best_ask') else 0.5

    # Denoise both sides
    denoised_bids = denoise_side_advanced(bids, "BIDS", best_bid, mm_memory, asset_type)
    denoised_asks = denoise_side_advanced(asks, "ASKS", best_ask, mm_memory, asset_type)

    # Update the data
    denoised_data['complete_book']['bids'] = denoised_bids
    denoised_data['complete_book']['asks'] = denoised_asks

    # Recalculate best bid/ask
    if denoised_bids:
        best_bid_order = max(denoised_bids, key=lambda x: float(x['price']))
        denoised_data['best_bid'] = {
            'price': float(best_bid_order['price']),
            'size': float(best_bid_order['size'])
        }

    if denoised_asks:
        best_ask_order = min(denoised_asks, key=lambda x: float(x['price']))
        denoised_data['best_ask'] = {
            'price': float(best_ask_order['price']),
            'size': float(best_ask_order['size'])
        }

    return denoised_data


def main():
    print("\n" + "="*80)
    print("🚀 STARTING ADVANCED MM DETECTION & DENOISING")
    print("="*80)
    print(f"Update interval: {UPDATE_INTERVAL} seconds")
    print(f"Calibration period: {CALIBRATION_PERIOD} seconds at :00, :15, :30, :45")
    print(f"MM detection: {MM_DISTANCE_FROM_BEST[0]:.2f}-{MM_DISTANCE_FROM_BEST[1]:.2f} from best bid/ask")
    print(f"MM size threshold: {MM_SIZE_THRESHOLD}x neighbors")
    print(f"Normal price range: ${NORMAL_PRICE_RANGE[0]:.2f} - ${NORMAL_PRICE_RANGE[1]:.2f}")
    print(f"Press Ctrl+C to stop")
    print("="*80 + "\n")

    mm_memory = MarketMakerMemory()
    iteration = 0
    calibration_start = None

    try:
        while True:
            iteration += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Check if we should start calibration
            if mm_memory.should_calibrate():
                mm_memory.start_calibration()
                calibration_start = datetime.now()

            # Check if calibration just ended
            if mm_memory.in_calibration and calibration_start:
                if not mm_memory.is_calibrating(calibration_start):
                    print(f"\n✅ CALIBRATION COMPLETE")
                    mm_memory.end_calibration('CALL')
                    mm_memory.end_calibration('PUT')
                    calibration_start = None

            # Show status
            status = "🔬 CALIBRATING" if mm_memory.in_calibration else "🎯 DENOISING"
            print(f"\n[{timestamp}] #{iteration} - {status}")

            try:
                # Process CALL
                call_input = f'{INPUT_DIR}15M_CALL.json'
                call_output = f'{INPUT_DIR}15M_CALL{OUTPUT_SUFFIX}.json'

                call_data = load_orderbook(call_input)
                denoised_call = denoise_orderbook_advanced(call_data, mm_memory)
                save_orderbook(denoised_call, call_output)

                if not mm_memory.in_calibration:
                    print(f"✅ Updated: {call_output}")

            except FileNotFoundError:
                print(f"⚠️  File not found: {call_input}")
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON error in CALL file: {e}")
            except Exception as e:
                print(f"❌ Error processing CALL: {e}")

            try:
                # Process PUT
                put_input = f'{INPUT_DIR}15M_PUT.json'
                put_output = f'{INPUT_DIR}15M_PUT{OUTPUT_SUFFIX}.json'

                put_data = load_orderbook(put_input)
                denoised_put = denoise_orderbook_advanced(put_data, mm_memory)
                save_orderbook(denoised_put, put_output)

                if not mm_memory.in_calibration:
                    print(f"✅ Updated: {put_output}")

            except FileNotFoundError:
                print(f"⚠️  File not found: {put_input}")
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON error in PUT file: {e}")
            except Exception as e:
                print(f"❌ Error processing PUT: {e}")

            # Apply memory decay periodically (every 100 iterations)
            if iteration % 100 == 0:
                mm_memory.apply_decay()

            # Wait before next iteration
            time.sleep(UPDATE_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("🛑 STOPPING - User interrupted")
        print(f"Total iterations completed: {iteration}")
        print("\n📊 Final MM Estimates:")
        print("\nCALL:")
        for price, size in mm_memory.call_mm_estimate.items():
            print(f"  ${price:.2f}: {size:.2f}")
        print("\nPUT:")
        for price, size in mm_memory.put_mm_estimate.items():
            print(f"  ${price:.2f}: {size:.2f}")
        print("="*80)
        sys.exit(0)


if __name__ == '__main__':
    main()
