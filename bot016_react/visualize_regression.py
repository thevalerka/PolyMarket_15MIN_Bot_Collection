#!/usr/bin/env python3
"""
Regression-Based Price Visualizer
Shows how regression predicts option prices and detects market lag
"""

import json
import time
import numpy as np
from datetime import datetime
from collections import deque
import warnings

# Suppress all numpy polynomial warnings
warnings.filterwarnings('ignore', message='Polyfit may be poorly conditioned')


def read_json(filepath):
    """Read JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return None


def predict_price_from_regression(btc_history, option_history, current_btc):
    """
    Predict option price using polynomial regression on recent data
    """
    if len(btc_history) < 10:
        return None
    
    btc_prices = np.array(btc_history)
    option_prices = np.array(option_history)
    
    # Filter valid range
    valid_mask = (option_prices >= 0.01) & (option_prices <= 0.99)
    btc_prices = btc_prices[valid_mask]
    option_prices = option_prices[valid_mask]
    
    if len(btc_prices) < 5:
        return None
    
    try:
        # Polynomial regression (degree 2)
        coeffs = np.polyfit(btc_prices, option_prices, deg=2)
        poly = np.poly1d(coeffs)
        predicted = poly(current_btc)
        return max(0.01, min(0.99, round(predicted, 2)))
    except:
        return None


def detect_spike(price_history):
    """Detect price spike in last 2 seconds"""
    if len(price_history) < 20:
        return False, 0.0
    
    current = price_history[-1]
    two_sec_ago = price_history[-20]
    
    spike_pct = ((current - two_sec_ago) / two_sec_ago) * 100
    has_spike = abs(spike_pct) > 0.05
    
    return has_spike, spike_pct


def main():
    """Visualize regression predictions and market lag"""
    put_file = "/home/ubuntu/013_2025_polymarket/15M_PUT.json"
    call_file = "/home/ubuntu/013_2025_polymarket/15M_CALL.json"
    btc_file = "/home/ubuntu/013_2025_polymarket/bybit_btc_price.json"
    
    print("="*120)
    print("REGRESSION-BASED ARBITRAGE VISUALIZER")
    print("="*120)
    print("\nShows: Expected prices from regression | Market lag | BTC spikes")
    print("Strategy: Buy when market lags behind expected price after BTC spike\n")
    
    # History buffers (5 seconds = 50 samples at 0.1s)
    btc_history = deque(maxlen=50)
    put_mid_history = deque(maxlen=50)
    call_mid_history = deque(maxlen=50)
    
    try:
        while True:
            # Read current prices
            put_data = read_json(put_file)
            call_data = read_json(call_file)
            btc_data = read_json(btc_file)
            
            if not all([put_data, call_data, btc_data]):
                time.sleep(0.1)
                continue
            
            current_btc = btc_data.get('price', 0)
            
            # Get market prices
            put_bid = put_data.get('best_bid', {}).get('price') if put_data.get('best_bid') else None
            put_ask = put_data.get('best_ask', {}).get('price') if put_data.get('best_ask') else None
            call_bid = call_data.get('best_bid', {}).get('price') if call_data.get('best_bid') else None
            call_ask = call_data.get('best_ask', {}).get('price') if call_data.get('best_ask') else None
            
            if None in [put_bid, put_ask, call_bid, call_ask]:
                time.sleep(0.1)
                continue
            
            # Add to history
            btc_history.append(current_btc)
            put_mid_history.append((put_bid + put_ask) / 2)
            call_mid_history.append((call_bid + call_ask) / 2)
            
            # Need sufficient history for regression
            if len(btc_history) < 20:
                print(f"Collecting data... {len(btc_history)}/20")
                time.sleep(0.1)
                continue
            
            # Predict expected prices from regression
            expected_put = predict_price_from_regression(
                list(btc_history), list(put_mid_history), current_btc
            )
            expected_call = predict_price_from_regression(
                list(btc_history), list(call_mid_history), current_btc
            )
            
            if expected_put is None or expected_call is None:
                time.sleep(0.1)
                continue
            
            # Detect spike
            has_spike, spike_pct = detect_spike(list(btc_history))
            
            # Calculate market lag
            put_lag = expected_put - put_ask
            call_lag = expected_call - call_ask
            
            # Skip extreme prices
            skip_put = put_ask >= 0.97 or put_ask <= 0.03
            skip_call = call_ask >= 0.97 or call_ask <= 0.03
            
            # Display
            now = datetime.now()
            timestamp = now.strftime('%H:%M:%S.%f')[:-3]
            
            spike_indicator = ""
            if has_spike:
                direction = "UP" if spike_pct > 0 else "DOWN"
                spike_indicator = f"⚡ SPIKE {direction} {abs(spike_pct):.3f}%"
            
            print(f"\n{timestamp} | BTC: ${current_btc:.2f} | {spike_indicator}")
            
            # PUT analysis
            put_signal = ""
            if skip_put:
                put_signal = "⚠️ EXTREME PRICE - SKIP"
            elif put_bid >= 0.99:
                put_signal = "🎯 SELL @ 0.99!"
            elif has_spike and spike_pct < -0.05 and put_lag >= 0.03:
                put_signal = "🟢 BUY SIGNAL!"
            elif put_lag >= 0.03:
                put_signal = "🟡 OPPORTUNITY"
            
            print(f"  PUT:  Market {put_bid:.2f}/{put_ask:.2f} | "
                  f"Expected {expected_put:.2f} | "
                  f"Lag: {put_lag:+.3f} | {put_signal}")
            
            # CALL analysis
            call_signal = ""
            if skip_call:
                call_signal = "⚠️ EXTREME PRICE - SKIP"
            elif call_bid >= 0.99:
                call_signal = "🎯 SELL @ 0.99!"
            elif has_spike and spike_pct > 0.05 and call_lag >= 0.03:
                call_signal = "🟢 BUY SIGNAL!"
            elif call_lag >= 0.03:
                call_signal = "🟡 OPPORTUNITY"
            
            print(f"  CALL: Market {call_bid:.2f}/{call_ask:.2f} | "
                  f"Expected {expected_call:.2f} | "
                  f"Lag: {call_lag:+.3f} | {call_signal}")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\nStopped by user")


if __name__ == "__main__":
    main()
