#!/usr/bin/env python3
"""
Order Book Denoising Script
Removes market maker noise from Polymarket order book data
Runs continuously, updating denoised files in real-time
"""

import json
import copy
import time
import sys
from typing import Dict, List, Tuple
from statistics import mean, median
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
UPDATE_INTERVAL = 0.1  # Update every 0.5 seconds
INPUT_DIR = '/home/ubuntu/013_2025_polymarket/'  # Same directory, set to your path if different
OUTPUT_SUFFIX = '_nonoise'  # Files will be saved as 15M_CALL_nonoise.json

# Denoising parameters
MM_THRESHOLD_MULTIPLIER = 2.5  # Market maker detection threshold
OUTLIER_THRESHOLD = 8.0  # Cap single outliers at 8x average
# ============================================================================

def load_orderbook(filepath: str) -> Dict:
    """Load orderbook data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_orderbook(data: Dict, filepath: str):
    """Save denoised orderbook data to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def calculate_stats(sizes: List[float]) -> Tuple[float, float, float]:
    """Calculate mean, median, and max of sizes"""
    if not sizes:
        return 0, 0, 0
    return mean(sizes), median(sizes), max(sizes)

def find_market_maker_levels(bids: List[Dict], asks: List[Dict],
                              threshold_multiplier: float = 2.5) -> List[str]:
    """
    Identify market maker levels that appear on BOTH bid and ask sides
    and are significantly larger than neighbors
    """
    # Create dictionaries for quick lookup
    bid_dict = {float(b['price']): float(b['size']) for b in bids}
    ask_dict = {float(a['price']): float(a['size']) for a in asks}

    # Find prices that exist on both sides
    common_prices = set(bid_dict.keys()) & set(ask_dict.keys())

    mm_levels = []

    for price in sorted(common_prices):
        bid_size = bid_dict[price]
        ask_size = ask_dict[price]

        # Get neighbor sizes on bid side
        bid_neighbors = []
        for p in bid_dict.keys():
            if abs(p - price) <= 0.05 and p != price:  # Within 5 cents
                bid_neighbors.append(bid_dict[p])

        # Get neighbor sizes on ask side
        ask_neighbors = []
        for p in ask_dict.keys():
            if abs(p - price) <= 0.05 and p != price:  # Within 5 cents
                ask_neighbors.append(ask_dict[p])

        # Calculate average neighbor size
        if bid_neighbors and ask_neighbors:
            avg_bid_neighbor = mean(bid_neighbors)
            avg_ask_neighbor = mean(ask_neighbors)

            # Check if both sides are significantly larger than neighbors
            if (bid_size > threshold_multiplier * avg_bid_neighbor and
                ask_size > threshold_multiplier * avg_ask_neighbor):
                mm_levels.append(f"{price:.2f}")

    return mm_levels

def denoise_side(orders: List[Dict], side_name: str,
                 mm_levels: List[str], outlier_threshold: float = 8.0) -> List[Dict]:
    """
    Remove market maker noise from one side of the book

    Rules:
    1. If price is in mm_levels (bilateral MM), remove the MM size
    2. If a level is 8x+ average but only 1 such level, cap it
    3. If multiple 8x+ levels exist, keep them (likely real flow)
    """
    if not orders:
        return orders

    denoised = []
    sizes = [float(o['size']) for o in orders]
    avg_size = mean(sizes)

    # Find outlier levels (8x+ average)
    outlier_levels = []
    for order in orders:
        size = float(order['size'])
        if size > outlier_threshold * avg_size:
            outlier_levels.append(order['price'])

    # Count occurrences at each price level
    price_counts = {}
    for order in orders:
        price = order['price']
        price_counts[price] = price_counts.get(price, 0) + 1

    for order in orders:
        price = order['price']
        size = float(order['size'])

        # Rule 1: Remove MM levels (bilateral large orders)
        if price in mm_levels:
            # Reduce by 80% as a heuristic
            reduced_size = size * 0.2
            denoised.append({
                'price': price,
                'size': f"{reduced_size:.2f}"
            })
            continue

        # Rule 2: Multiple orders at same price - keep all
        if price_counts[price] > 1:
            denoised.append(order)
            continue

        # Rule 3: Single outlier (8x+ avg) - cap it
        if price in outlier_levels and len(outlier_levels) == 1:
            capped_size = min(size, outlier_threshold * avg_size)
            denoised.append({
                'price': price,
                'size': f"{capped_size:.2f}"
            })
            continue

        # Rule 4: Multiple outliers - keep them (real flow)
        denoised.append(order)

    return denoised

def denoise_orderbook(data: Dict) -> Dict:
    """Main denoising function"""
    denoised_data = copy.deepcopy(data)

    bids = data['complete_book']['bids']
    asks = data['complete_book']['asks']

    # Find bilateral market maker levels
    mm_levels = find_market_maker_levels(bids, asks, MM_THRESHOLD_MULTIPLIER)

    # Denoise both sides
    denoised_bids = denoise_side(bids, "BIDS", mm_levels, OUTLIER_THRESHOLD)
    denoised_asks = denoise_side(asks, "ASKS", mm_levels, OUTLIER_THRESHOLD)

    # Update the data
    denoised_data['complete_book']['bids'] = denoised_bids
    denoised_data['complete_book']['asks'] = denoised_asks

    # Recalculate best bid/ask
    if denoised_bids:
        best_bid = max(denoised_bids, key=lambda x: float(x['price']))
        denoised_data['best_bid'] = {
            'price': float(best_bid['price']),
            'size': float(best_bid['size'])
        }

    if denoised_asks:
        best_ask = min(denoised_asks, key=lambda x: float(x['price']))
        denoised_data['best_ask'] = {
            'price': float(best_ask['price']),
            'size': float(best_ask['size'])
        }

    return denoised_data

def main():
    print("\n" + "="*80)
    print("🚀 STARTING CONTINUOUS ORDER BOOK DENOISING")
    print("="*80)
    print(f"Update interval: {UPDATE_INTERVAL} seconds")
    print(f"Input directory: {INPUT_DIR if INPUT_DIR else 'current directory'}")
    print(f"Press Ctrl+C to stop")
    print("="*80 + "\n")

    iteration = 0

    try:
        while True:
            iteration += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            print(f"\n{'='*80}")
            print(f"[{timestamp}] Iteration #{iteration}")
            print(f"{'='*80}")

            try:
                # Process CALL
                call_input = f'{INPUT_DIR}15M_CALL.json'
                call_output = f'{INPUT_DIR}15M_CALL{OUTPUT_SUFFIX}.json'

                call_data = load_orderbook(call_input)
                denoised_call = denoise_orderbook(call_data)
                save_orderbook(denoised_call, call_output)
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
                denoised_put = denoise_orderbook(put_data)
                save_orderbook(denoised_put, put_output)
                print(f"✅ Updated: {put_output}")

            except FileNotFoundError:
                print(f"⚠️  File not found: {put_input}")
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON error in PUT file: {e}")
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
