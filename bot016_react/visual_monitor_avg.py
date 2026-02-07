#!/usr/bin/env python3
"""
Visual Monitor (AVG) - Uses Average instead of Median
Shows BTC, CALL, PUT prices and sensitivity status every 0.1s
Uses average sensitivity values instead of median
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from collections import deque

# File paths
PUT_FILE = "/home/ubuntu/013_2025_polymarket/15M_PUT.json"
CALL_FILE = "/home/ubuntu/013_2025_polymarket/15M_CALL.json"
BTC_FILE = "/home/ubuntu/013_2025_polymarket/bybit_btc_price.json"
#SENSITIVITY_FILE = "/home/ubuntu/013_2025_polymarket/bot016_react/sensitivity_data/sensitivity_master.json"
SENSITIVITY_FILE = "/home/ubuntu/013_2025_polymarket/bot016_react/sensitivity_data/sensitivity_transformed.json"
sens_multiplier = 1.2
action_threshold = 0.042


def read_json(filepath: str) -> Optional[dict]:
    """Read JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return None


def get_bin_key(distance: float, seconds_to_expiry: float, volatility: float) -> str:
    """Get bin key"""
    # Distance bins
    distance_bins = [
        (0, 1, "0-1"), (1, 5, "1-5"), (5, 10, "5-10"), (10, 20, "10-20"),
        (20, 40, "20-40"), (40, 80, "40-80"), (80, 160, "80-160"),
        (160, 320, "160-320"), (320, 640, "320-640"), (640, 1280, "640-1280"),
        (1280, float('inf'), "1280+")
    ]

    # Time bins
    time_bins = [
        (13*60, 15*60, "15m-13m"), (11*60, 13*60, "13m-11m"), (10*60, 11*60, "11m-10m"),
        (9*60, 10*60, "10m-9m"), (8*60, 9*60, "9m-8m"), (7*60, 8*60, "8m-7m"),
        (6*60, 7*60, "7m-6m"), (5*60, 6*60, "6m-5m"), (4*60, 5*60, "5m-4m"),
        (3*60, 4*60, "4m-3m"), (2*60, 3*60, "3m-2m"), (90, 120, "120s-90s"),
        (60, 90, "90s-60s"), (40, 60, "60s-40s"), (30, 40, "40s-30s"),
        (20, 30, "30s-20s"), (10, 20, "20s-10s"), (5, 10, "10s-5s"),
        (2, 5, "5s-2s"), (0, 2, "last-2s")
    ]

    # Volatility bins (BTC price range in dollars over last minute)
    vol_bins = [
        (0, 10, "0-10"), (10, 20, "10-20"), (20, 30, "20-30"), (30, 40, "30-40"),
        (40, 60, "40-60"), (60, 90, "60-90"), (90, 120, "90-120"), (120, 240, "120-240"),
        (240, float('inf'), "240+")
    ]

    def get_bin_label(value, bins):
        for min_val, max_val, label in bins:
            if min_val <= value < max_val:
                return label
        return bins[-1][2]

    dist_label = get_bin_label(distance, distance_bins)
    time_label = get_bin_label(seconds_to_expiry, time_bins)
    vol_label = get_bin_label(volatility, vol_bins)

    return f"{dist_label}|{time_label}|{vol_label}"


def get_strike_price() -> Optional[float]:
    """Get strike price from Bybit API"""
    try:
        import requests
        from datetime import timezone

        now = datetime.now(timezone.utc)
        current_minute = now.minute

        # Determine period start
        for start_min in [0, 15, 30, 45]:
            if current_minute >= start_min and current_minute < start_min + 15:
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
                            return float(kline_list[0][1])
        return None
    except:
        return None


def get_seconds_to_expiry() -> float:
    """Get seconds to expiry"""
    now = datetime.now()
    current_minute = now.minute

    for start_min in [0, 15, 30, 45]:
        if current_minute >= start_min and current_minute < start_min + 15:
            seconds_into_period = (current_minute - start_min) * 60 + now.second
            return 900 - seconds_into_period
    return 0


def calculate_btc_volatility(price_history: deque) -> float:
    """Calculate BTC volatility over last minute (price range)"""
    if len(price_history) < 10:
        return 0.0
    prices = list(price_history)
    return max(prices) - min(prices)


def main():
    print("\n" + "="*150)
    print("VISUAL PRICE MONITOR (AVG) - Uses Average Sensitivity")
    print("="*150)
    print(f"{'BTC Price':<12} | {'CALL Bid/Ask':<15} | {'PUT Bid/Ask':<15} | {'Bin Status':<35} | {'Time Diff (s)':<15}")
    print("-"*150)

    # Load sensitivity data once
    sensitivity_data = read_json(SENSITIVITY_FILE)

    # Track previous state for delta calculation
    prev_btc_price = None
    prev_call_bid = None
    prev_call_ask = None
    prev_put_bid = None
    prev_put_ask = None

    # BTC price history for volatility calculation (1 minute at 0.1s intervals = 600 samples)
    btc_price_history = deque(maxlen=600)

    # Track strike price and last update time
    strike_price = None
    last_strike_update_minute = None

    try:
        while True:
            # Check if we need to update strike price (every 15 minutes at 00, 15, 30, 45)
            now = datetime.now()
            current_minute = now.minute
            current_second = now.second

            # Determine if we're at the start of a 15-minute period
            is_period_start = current_minute in [0, 15, 30, 45]

            # Update strike price if:
            # 1. We're at a period start (min 00, 15, 30, 45)
            # 2. We're at least 5 seconds into the period (to ensure correct candle)
            # 3. We haven't updated this minute yet
            if is_period_start and current_second >= 5 and last_strike_update_minute != current_minute:
                print(f"\n🔄 Updating strike price at {now.strftime('%H:%M:%S')}...")
                new_strike = get_strike_price()
                if new_strike:
                    strike_price = new_strike
                    last_strike_update_minute = current_minute
                    print(f"✅ New strike price: ${strike_price:.2f}\n")
                else:
                    print("⚠️  Failed to get strike price, retrying...\n")

            # Initialize strike price on first run if not set
            if strike_price is None:
                strike_price = get_strike_price()
                if not strike_price:
                    btc_data = read_json(BTC_FILE)
                    if btc_data:
                        strike_price = btc_data['price']
                if strike_price:
                    print(f"Initial Strike Price: ${strike_price:.2f}\n")

            # Read prices
            btc_data = read_json(BTC_FILE)
            call_data = read_json(CALL_FILE)
            put_data = read_json(PUT_FILE)

            if not all([btc_data, call_data, put_data]):
                print("Waiting for data...")
                time.sleep(0.1)
                continue

            # Extract data
            btc_price = btc_data.get('price', 0)
            btc_timestamp = btc_data.get('timestamp', 0)

            # Add to price history for volatility calculation
            btc_price_history.append(btc_price)

            call_bid = call_data.get('best_bid', {})
            call_ask = call_data.get('best_ask', {})
            call_bid_price = call_bid.get('price', 0) if call_bid else 0
            call_ask_price = call_ask.get('price', 0) if call_ask else 0
            call_timestamp = call_data.get('timestamp', 0)

            put_bid = put_data.get('best_bid', {})
            put_ask = put_data.get('best_ask', {})
            put_bid_price = put_bid.get('price', 0) if put_bid else 0
            put_ask_price = put_ask.get('price', 0) if put_ask else 0

            # Calculate timestamp difference (BTC - CALL) in seconds
            # Timestamps are in milliseconds, so convert to seconds first
            try:
                btc_ts = float(btc_timestamp) / 1000 if btc_timestamp else 0
                call_ts = float(call_timestamp) / 1000 if call_timestamp else 0
                time_diff_s = btc_ts - call_ts
            except:
                time_diff_s = 0

            # Calculate volatility
            volatility = calculate_btc_volatility(btc_price_history)

            # Get bin status and check for trading signals
            signal_text = ""
            if strike_price and prev_btc_price is not None:
                distance = abs(btc_price - strike_price)
                seconds_remaining = get_seconds_to_expiry()

                bin_key = get_bin_key(distance, seconds_remaining, volatility)

                # Check if bin exists
                if sensitivity_data and bin_key in sensitivity_data.get('bins', {}):
                    bin_data = sensitivity_data['bins'][bin_key]
                    # USE AVERAGE INSTEAD OF MEDIAN
                    call_sens = bin_data.get('call_sensitivity', {}).get('avg', 0)
                    put_sens = bin_data.get('put_sensitivity', {}).get('avg', 0)

                    if abs(call_sens) < 0.000001 and abs(put_sens) < 0.000001:
                        bin_status = f"❌ {bin_key} (avg=0)"
                    else:
                        bin_status = f"✅ {bin_key} (C:{call_sens:+.6f} P:{put_sens:+.6f})"

                        # Calculate BTC delta
                        btc_delta = btc_price - prev_btc_price

                        # Only generate signals if we have valid sensitivities
                        if abs(call_sens) > 0.000001:
                            # Calculate ideal CALL price movement based on BTC movement
                            ideal_call_movement = btc_delta * call_sens * sens_multiplier
                            actual_call_bid_movement = call_bid_price - prev_call_bid
                            actual_call_ask_movement = call_ask_price - prev_call_ask

                            # If actual movement is LARGER than ideal, our BTC price is lagging - skip signal
                            if abs(actual_call_ask_movement) > abs(ideal_call_movement) + 0.02:
                                signal_text += f"\n⚠️  CALL LAG: BTC feed lagging (actual Δ={actual_call_ask_movement:+.3f} > ideal Δ={ideal_call_movement:+.3f})"
                            else:
                                # Calculate ideal CALL prices
                                ideal_call_bid = prev_call_bid + ideal_call_movement
                                ideal_call_ask = prev_call_ask + ideal_call_movement

                                # Check for BUY signal (ideal ask > actual ask by at least ...)
                                # This means market hasn't caught up to BTC movement yet
                                if ideal_call_ask > call_ask_price + action_threshold:
                                    signal_text += f"\n🟢 BUY CALL | Actual: {call_bid_price:.2f}/{call_ask_price:.2f} | Ideal: {ideal_call_bid:.2f}/{ideal_call_ask:.2f} | Edge: {ideal_call_ask - call_ask_price:.3f}"

                                # Check for SELL signal (ideal bid < actual bid by at least ...)
                                # This means market has overshot BTC movement
                                elif ideal_call_bid < call_bid_price - action_threshold:
                                        signal_text += f"\n🔴 SELL CALL | Actual: {call_bid_price:.2f}/{call_ask_price:.2f} | Ideal: {ideal_call_bid:.2f}/{ideal_call_ask:.2f} | Edge: {call_bid_price - ideal_call_bid:.3f}"

                        if abs(put_sens) > 0.000001:
                            # Calculate ideal PUT price movement based on BTC movement
                            ideal_put_movement = btc_delta * put_sens * sens_multiplier
                            actual_put_bid_movement = put_bid_price - prev_put_bid
                            actual_put_ask_movement = put_ask_price - prev_put_ask

                            # If actual movement is LARGER than ideal, our BTC price is lagging - skip signal
                            if abs(actual_put_ask_movement) > abs(ideal_put_movement) + 0.02:
                                signal_text += f"\n⚠️  PUT LAG: BTC feed lagging (actual Δ={actual_put_ask_movement:+.3f} > ideal Δ={ideal_put_movement:+.3f})"
                            else:
                                # Calculate ideal PUT prices
                                ideal_put_bid = prev_put_bid + ideal_put_movement
                                ideal_put_ask = prev_put_ask + ideal_put_movement

                                # Check for BUY signal (ideal ask > actual ask by at least 0.03)
                                if ideal_put_ask > put_ask_price + action_threshold:
                                    signal_text += f"\n🟢 BUY PUT | Actual: {put_bid_price:.2f}/{put_ask_price:.2f} | Ideal: {ideal_put_bid:.2f}/{ideal_put_ask:.2f} | Edge: {ideal_put_ask - put_ask_price:.3f}"

                                # Check for SELL signal (ideal bid < actual bid by at least 0.03)
                                elif ideal_put_bid < put_bid_price - action_threshold:
                                        signal_text += f"\n🔴 SELL PUT | Actual: {put_bid_price:.2f}/{put_ask_price:.2f} | Ideal: {ideal_put_bid:.2f}/{ideal_put_ask:.2f} | Edge: {put_bid_price - ideal_put_bid:.3f}"
                else:
                    bin_status = f"❌ {bin_key} (no data)"
            else:
                bin_status = "N/A (no strike)"

            # Print row
            print(f"${btc_price:<11.2f} | {call_bid_price:.2f}/{call_ask_price:.2f}        | {put_bid_price:.2f}/{put_ask_price:.2f}        | {bin_status:<35} | {time_diff_s:>14.3f}s{signal_text}")

            # Update previous state
            prev_btc_price = btc_price
            prev_call_bid = call_bid_price
            prev_call_ask = call_ask_price
            prev_put_bid = put_bid_price
            prev_put_ask = put_ask_price

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nStopped by user")


if __name__ == "__main__":
    main()
