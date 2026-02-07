#!/usr/bin/env python3
"""
Order Book Smoothing Denoiser
Removes all noise, anomalies, and spikes by replacing them with interpolated average data
Simple and robust approach using statistical outlier detection
pm2 start denoise_smooth.py --cron-restart="00 */12 * * *" --interpreter python3
"""

import json
import copy
import time
import sys
from typing import Dict, List, Optional
from statistics import mean, median, stdev
from datetime import datetime
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================
UPDATE_INTERVAL = 0.1  # Update every 0.5 seconds
INPUT_DIR = '/home/ubuntu/013_2025_polymarket/'

# Smoothing Parameters
OUTLIER_METHOD = 'iqr'  # 'iqr', 'zscore', or 'percentile'
IQR_MULTIPLIER = 2.0  # Values beyond Q1-2*IQR or Q3+2*IQR are outliers
ZSCORE_THRESHOLD = 2.5  # Values with |z-score| > 2.5 are outliers
PERCENTILE_THRESHOLD = (5, 95)  # Remove values outside 5th-95th percentile

# Replacement method
REPLACEMENT_METHOD = 'local_average'  # 'local_average', 'median', 'interpolate'
NEIGHBOR_WINDOW = 5  # Use 5 neighbors on each side for local average

# Price range filter
PRICE_RANGE = (0.10, 0.90)  # Only smooth in normal range
EDGE_SMOOTHING = False  # Whether to also smooth extreme prices
EXTREME_PRICE_CUTOFF = (0.05, 0.95)  # Delete all levels outside this range
# ============================================================================


def filter_extreme_prices(orders: List[Dict]) -> List[Dict]:
    """
    Remove all order book levels with prices < 0.05 or > 0.95
    """
    filtered = []
    for order in orders:
        price = float(order['price'])
        if EXTREME_PRICE_CUTOFF[0] <= price <= EXTREME_PRICE_CUTOFF[1]:
            filtered.append(order)
    return filtered


def load_orderbook(filepath: str) -> Dict:
    """Load orderbook data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_orderbook(data: Dict, filepath: str):
    """Save smoothed orderbook data to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def detect_outliers_iqr(sizes: List[float]) -> List[bool]:
    """
    Detect outliers using Interquartile Range (IQR) method
    Returns boolean list where True = outlier
    """
    if len(sizes) < 4:
        return [False] * len(sizes)

    sorted_sizes = sorted(sizes)
    n = len(sorted_sizes)

    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    q1_idx = n // 4
    q3_idx = 3 * n // 4
    q1 = sorted_sizes[q1_idx]
    q3 = sorted_sizes[q3_idx]

    iqr = q3 - q1
    lower_bound = q1 - IQR_MULTIPLIER * iqr
    upper_bound = q3 + IQR_MULTIPLIER * iqr

    return [size < lower_bound or size > upper_bound for size in sizes]


def detect_outliers_zscore(sizes: List[float]) -> List[bool]:
    """
    Detect outliers using Z-score method
    Returns boolean list where True = outlier
    """
    if len(sizes) < 3:
        return [False] * len(sizes)

    avg = mean(sizes)
    try:
        std = stdev(sizes)
        if std == 0:
            return [False] * len(sizes)

        z_scores = [(size - avg) / std for size in sizes]
        return [abs(z) > ZSCORE_THRESHOLD for z in z_scores]
    except:
        return [False] * len(sizes)


def detect_outliers_percentile(sizes: List[float]) -> List[bool]:
    """
    Detect outliers using percentile method
    Returns boolean list where True = outlier
    """
    if len(sizes) < 10:
        return [False] * len(sizes)

    sorted_sizes = sorted(sizes)
    n = len(sorted_sizes)

    lower_idx = int(n * PERCENTILE_THRESHOLD[0] / 100)
    upper_idx = int(n * PERCENTILE_THRESHOLD[1] / 100)

    lower_bound = sorted_sizes[lower_idx]
    upper_bound = sorted_sizes[upper_idx]

    return [size < lower_bound or size > upper_bound for size in sizes]


def get_replacement_value(orders: List[Dict], index: int,
                         method: str = REPLACEMENT_METHOD) -> float:
    """
    Calculate replacement value for an outlier
    """
    sizes = [float(o['size']) for o in orders]

    if method == 'median':
        # Use overall median
        return median(sizes)

    elif method == 'local_average':
        # Use average of nearby non-outlier values
        start = max(0, index - NEIGHBOR_WINDOW)
        end = min(len(orders), index + NEIGHBOR_WINDOW + 1)

        neighbors = []
        for i in range(start, end):
            if i != index:
                neighbors.append(sizes[i])

        return mean(neighbors) if neighbors else median(sizes)

    elif method == 'interpolate':
        # Linear interpolation between neighbors
        # Find nearest non-outlier on each side
        left_val = None
        right_val = None

        # Look left
        for i in range(index - 1, -1, -1):
            if sizes[i] != sizes[index]:  # Different value
                left_val = sizes[i]
                break

        # Look right
        for i in range(index + 1, len(sizes)):
            if sizes[i] != sizes[index]:  # Different value
                right_val = sizes[i]
                break

        if left_val and right_val:
            return (left_val + right_val) / 2
        elif left_val:
            return left_val
        elif right_val:
            return right_val
        else:
            return median(sizes)

    # Default: median
    return median(sizes)


def smooth_orderbook_side(orders: List[Dict], side_name: str,
                          asset_type: str, verbose: bool = False) -> tuple:
    """
    Smooth one side of the order book by detecting and replacing outliers
    Returns: (smoothed_orders, num_outliers_detected)
    """
    if not orders:
        return orders, 0

    # Separate orders by price range
    normal_range_orders = []
    edge_orders = []

    for order in orders:
        price = float(order['price'])
        if PRICE_RANGE[0] < price < PRICE_RANGE[1]:
            normal_range_orders.append(order)
        else:
            edge_orders.append(order)

    # Extract sizes from normal range
    sizes = [float(o['size']) for o in normal_range_orders]

    if len(sizes) < 4:
        return orders, 0  # Not enough data to detect outliers

    # Detect outliers
    if OUTLIER_METHOD == 'iqr':
        is_outlier = detect_outliers_iqr(sizes)
    elif OUTLIER_METHOD == 'zscore':
        is_outlier = detect_outliers_zscore(sizes)
    elif OUTLIER_METHOD == 'percentile':
        is_outlier = detect_outliers_percentile(sizes)
    else:
        is_outlier = [False] * len(sizes)

    num_outliers = sum(is_outlier)

    # Replace outliers
    smoothed_orders = []
    for i, order in enumerate(normal_range_orders):
        if is_outlier[i]:
            replacement = get_replacement_value(normal_range_orders, i)
            if verbose:
                print(f"    {side_name} ${order['price']}: {order['size']} → {replacement:.2f}")
            smoothed_orders.append({
                'price': order['price'],
                'size': f"{replacement:.2f}"
            })
        else:
            smoothed_orders.append(order)

    # Add back edge orders (or smooth them too if enabled)
    if EDGE_SMOOTHING and edge_orders:
        edge_sizes = [float(o['size']) for o in edge_orders]
        if len(edge_sizes) >= 4:
            edge_outliers = detect_outliers_iqr(edge_sizes)
            for i, order in enumerate(edge_orders):
                if edge_outliers[i]:
                    replacement = get_replacement_value(edge_orders, i)
                    smoothed_orders.append({
                        'price': order['price'],
                        'size': f"{replacement:.2f}"
                    })
                    num_outliers += 1
                else:
                    smoothed_orders.append(order)
        else:
            smoothed_orders.extend(edge_orders)
    else:
        smoothed_orders.extend(edge_orders)

    # Sort back by price
    smoothed_orders.sort(key=lambda x: float(x['price']))

    return smoothed_orders, num_outliers


def smooth_orderbook(data: Dict, verbose: bool = False) -> Dict:
    """
    Main smoothing function - removes anomalies from entire order book
    """
    smoothed_data = copy.deepcopy(data)
    asset_type = 'CALL' if 'CALL' in data['asset_name'] else 'PUT'

    bids = data['complete_book']['bids']
    asks = data['complete_book']['asks']

    # FIRST: Filter out extreme prices (< 0.05 or > 0.95)
    bids = filter_extreme_prices(bids)
    asks = filter_extreme_prices(asks)

    if verbose:
        orig_bids = len(data['complete_book']['bids'])
        orig_asks = len(data['complete_book']['asks'])
        filtered_bids = len(bids)
        filtered_asks = len(asks)
        if orig_bids != filtered_bids or orig_asks != filtered_asks:
            print(f"  {asset_type}: Filtered extremes - Bids: {orig_bids}→{filtered_bids}, Asks: {orig_asks}→{filtered_asks}")

    # SECOND: Smooth outliers in remaining data
    smoothed_bids, bid_outliers = smooth_orderbook_side(bids, "BID", asset_type, verbose)
    smoothed_asks, ask_outliers = smooth_orderbook_side(asks, "ASK", asset_type, verbose)

    if verbose:
        print(f"  {asset_type}: Smoothed {bid_outliers} bid outliers, {ask_outliers} ask outliers")

    # Update the data
    smoothed_data['complete_book']['bids'] = smoothed_bids
    smoothed_data['complete_book']['asks'] = smoothed_asks

    # Recalculate best bid/ask
    if smoothed_bids:
        best_bid_order = max(smoothed_bids, key=lambda x: float(x['price']))
        smoothed_data['best_bid'] = {
            'price': float(best_bid_order['price']),
            'size': float(best_bid_order['size'])
        }

    if smoothed_asks:
        best_ask_order = min(smoothed_asks, key=lambda x: float(x['price']))
        smoothed_data['best_ask'] = {
            'price': float(best_ask_order['price']),
            'size': float(best_ask_order['size'])
        }

    return smoothed_data


def main():
    print("\n" + "="*80)
    print("🧹 STARTING SMOOTHING-BASED ORDER BOOK DENOISER")
    print("="*80)
    print(f"Update interval: {UPDATE_INTERVAL} seconds")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Extreme price cutoff: Delete levels < ${EXTREME_PRICE_CUTOFF[0]:.2f} or > ${EXTREME_PRICE_CUTOFF[1]:.2f}")
    print(f"Outlier detection: {OUTLIER_METHOD}")
    if OUTLIER_METHOD == 'iqr':
        print(f"IQR multiplier: {IQR_MULTIPLIER}")
    elif OUTLIER_METHOD == 'zscore':
        print(f"Z-score threshold: {ZSCORE_THRESHOLD}")
    else:
        print(f"Percentile range: {PERCENTILE_THRESHOLD[0]}-{PERCENTILE_THRESHOLD[1]}")
    print(f"Replacement method: {REPLACEMENT_METHOD}")
    print(f"Smoothing range: ${PRICE_RANGE[0]:.2f} - ${PRICE_RANGE[1]:.2f}")
    print(f"Press Ctrl+C to stop")
    print("="*80 + "\n")

    iteration = 0

    try:
        while True:
            iteration += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Show periodic updates (every 10 iterations)
            verbose = (iteration % 10 == 0)

            if verbose or iteration == 1:
                print(f"\n[{timestamp}] Iteration #{iteration}")

            try:
                # Process CALL
                call_input = f'{INPUT_DIR}15M_CALL.json'
                call_output = f'{INPUT_DIR}15M_CALL_nonoise.json'

                call_data = load_orderbook(call_input)
                smoothed_call = smooth_orderbook(call_data, verbose)
                save_orderbook(smoothed_call, call_output)

                if verbose:
                    print(f"✅ Updated: {call_output}")

            except FileNotFoundError:
                if verbose:
                    print(f"⚠️  File not found: {call_input}")
            except json.JSONDecodeError as e:
                if verbose:
                    print(f"⚠️  JSON error in CALL: {e}")
            except Exception as e:
                print(f"❌ Error processing CALL: {e}")

            try:
                # Process PUT
                put_input = f'{INPUT_DIR}15M_PUT.json'
                put_output = f'{INPUT_DIR}15M_PUT_nonoise.json'

                put_data = load_orderbook(put_input)
                smoothed_put = smooth_orderbook(put_data, verbose)
                save_orderbook(smoothed_put, put_output)

                if verbose:
                    print(f"✅ Updated: {put_output}")

            except FileNotFoundError:
                if verbose:
                    print(f"⚠️  File not found: {put_input}")
            except json.JSONDecodeError as e:
                if verbose:
                    print(f"⚠️  JSON error in PUT: {e}")
            except Exception as e:
                print(f"❌ Error processing PUT: {e}")

            # Wait before next iteration
            time.sleep(UPDATE_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("🛑 STOPPING - User interrupted")
        print(f"Total iterations completed: {iteration}")
        print("="*80)
        sys.exit(0)


if __name__ == '__main__':
    main()
