#!/usr/bin/env python3
"""
Theoretical Price Visualizer
Shows how theoretical PUT/CALL prices react to BTC movements
"""

import json
import time
import numpy as np
from datetime import datetime


def read_json(filepath):
    """Read JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return None


def calculate_binary_price(current_btc, strike, time_to_expiry_min, token_type):
    """
    Simplified binary option pricing for visualization
    """
    if strike is None or strike == 0:
        return 0.5
    
    price_diff_pct = ((current_btc - strike) / strike) * 100
    
    # Base probability from current position
    if token_type == 'CALL':
        if current_btc > strike:
            base_prob = 0.5 + min(0.45, price_diff_pct * 2.0)
        else:
            base_prob = 0.5 - min(0.45, abs(price_diff_pct) * 2.0)
    else:  # PUT
        if current_btc < strike:
            base_prob = 0.5 + min(0.45, abs(price_diff_pct) * 2.0)
        else:
            base_prob = 0.5 - min(0.45, price_diff_pct * 2.0)
    
    # Time decay effect
    if time_to_expiry_min < 3:
        if abs(price_diff_pct) > 0.3:
            if token_type == 'CALL':
                base_prob = 0.95 if current_btc > strike else 0.05
            else:
                base_prob = 0.95 if current_btc < strike else 0.05
    
    return max(0.01, min(0.99, round(base_prob, 2)))


def main():
    """Visualize theoretical prices"""
    put_file = "/home/ubuntu/013_2025_polymarket/15M_PUT.json"
    call_file = "/home/ubuntu/013_2025_polymarket/15M_CALL.json"
    btc_file = "/home/ubuntu/013_2025_polymarket/bybit_btc_price.json"
    
    print("="*100)
    print("THEORETICAL PRICE VISUALIZER")
    print("="*100)
    print("\nMonitoring price movements... Press Ctrl+C to stop\n")
    
    strike = None
    strike_set_time = None
    
    try:
        while True:
            # Read current prices
            put_data = read_json(put_file)
            call_data = read_json(call_file)
            btc_data = read_json(btc_file)
            
            if not all([put_data, call_data, btc_data]):
                time.sleep(0.5)
                continue
            
            current_btc = btc_data.get('price', 0)
            
            # Check for new period (set strike)
            now = datetime.now()
            if now.minute in [0, 15, 30, 45] and now.second < 10:
                period_start = now.replace(second=0, microsecond=0)
                if strike_set_time is None or period_start != strike_set_time:
                    strike = current_btc
                    strike_set_time = period_start
                    print(f"\n{'='*100}")
                    print(f"🎯 NEW PERIOD STARTED - Strike: ${strike:.2f}")
                    print(f"{'='*100}\n")
            
            # Calculate time to expiry
            current_minute = now.minute
            expiry_minutes = [0, 15, 30, 45]
            next_expiry = None
            for exp_min in expiry_minutes:
                if current_minute < exp_min:
                    next_expiry = exp_min
                    break
            if next_expiry is None:
                next_expiry = 60
            time_to_expiry = (next_expiry - current_minute) + ((60 - now.second) / 60.0)
            
            # Get market prices
            put_bid = put_data.get('best_bid', {}).get('price', 0) if put_data.get('best_bid') else None
            put_ask = put_data.get('best_ask', {}).get('price', 0) if put_data.get('best_ask') else None
            call_bid = call_data.get('best_bid', {}).get('price', 0) if call_data.get('best_bid') else None
            call_ask = call_data.get('best_ask', {}).get('price', 0) if call_data.get('best_ask') else None
            
            if None in [put_bid, put_ask, call_bid, call_ask]:
                print(f"⚠️  Missing bid/ask data - waiting...")
                time.sleep(1)
                continue
            
            # Calculate theoretical prices
            theo_put = calculate_binary_price(current_btc, strike, time_to_expiry, 'PUT')
            theo_call = calculate_binary_price(current_btc, strike, time_to_expiry, 'CALL')
            
            # Calculate discrepancies
            put_disc = theo_put - put_ask
            call_disc = theo_call - call_ask
            
            # Price vs strike
            if strike:
                diff_pct = ((current_btc - strike) / strike) * 100
                position = "ABOVE" if current_btc > strike else "BELOW"
            else:
                diff_pct = 0
                position = "NO STRIKE"
            
            # Display
            print(f"{now.strftime('%H:%M:%S')} | "
                  f"BTC: ${current_btc:.2f} ({position} strike by {abs(diff_pct):.3f}%) | "
                  f"Expiry: {time_to_expiry:.1f}m")
            
            print(f"  PUT:  Market {put_bid:.2f}/{put_ask:.2f} | "
                  f"Theo {theo_put:.2f} | "
                  f"Disc: {put_disc:+.3f} {'🟢 ENTRY!' if put_disc >= 0.03 else ''}")
            
            print(f"  CALL: Market {call_bid:.2f}/{call_ask:.2f} | "
                  f"Theo {theo_call:.2f} | "
                  f"Disc: {call_disc:+.3f} {'🟢 ENTRY!' if call_disc >= 0.03 else ''}")
            
            print()
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n\nStopped by user")


if __name__ == "__main__":
    main()
