#!/usr/bin/env python3
"""
Minimal RSI Testing Bot
- Samples BTC price every 1 second
- Calculates RSI with 60-second period
- Logs RSI value
- That's it!
"""

import requests
import time
import numpy as np
from collections import deque
from datetime import datetime, timezone

# Configuration
RSI_PERIOD = 60  # 60 seconds
SAMPLE_INTERVAL = 1.0  # 1 second

def get_btc_price():
    """Get current BTC price from Bybit"""
    try:
        url = "https://api.bybit.com/v5/market/tickers?category=linear&symbol=BTCUSDT"
        response = requests.get(url, timeout=5)
        data = response.json()
        return float(data['result']['list'][0]['lastPrice'])
    except Exception as e:
        print(f"[ERROR] Failed to get BTC price: {e}")
        return None

def calculate_rsi(prices: deque, period: int):
    """Calculate RSI using proper Wilder's smoothing method"""
    if len(prices) < period + 1:
        return None
    
    # Get last period+1 prices to calculate period deltas
    prices_array = np.array(list(prices)[-(period + 1):])
    deltas = np.diff(prices_array)
    
    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # First average: Simple Moving Average of first 'period' values
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    # Wilder's smoothing for remaining values (if any)
    # This uses EMA with alpha = 1/period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    # Handle division by zero
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def main():
    print("=" * 80)
    print("RSI TESTING BOT")
    print("=" * 80)
    print(f"RSI Period: {RSI_PERIOD} seconds")
    print(f"Sample Interval: {SAMPLE_INTERVAL} seconds")
    print("=" * 80)
    print()
    
    # Price history (1-second samples)
    price_history = deque(maxlen=1000)
    
    # Timing
    last_sample_time = 0
    
    print("Starting data collection...\n")
    
    while True:
        try:
            current_time = time.time()
            
            # Get BTC price
            btc_price = get_btc_price()
            if btc_price is None:
                time.sleep(0.1)
                continue
            
            # Sample price every 1 second
            if current_time - last_sample_time >= SAMPLE_INTERVAL:
                price_history.append(btc_price)
                last_sample_time = current_time
                
                # Calculate RSI
                rsi = calculate_rsi(price_history, RSI_PERIOD)
                
                # Log
                timestamp = datetime.fromtimestamp(current_time, tz=timezone.utc).strftime('%H:%M:%S')
                samples = len(price_history)
                needed = RSI_PERIOD + 1
                
                if rsi is not None:
                    print(f"[{timestamp}] BTC: ${btc_price:,.2f} | RSI{RSI_PERIOD}: {rsi:6.2f} | Samples: {samples}")
                else:
                    progress = (samples / needed * 100) if needed > 0 else 0
                    print(f"[{timestamp}] BTC: ${btc_price:,.2f} | Building data... {samples}/{needed} ({progress:.0f}%)")
            
            time.sleep(0.1)
            
        except KeyboardInterrupt:
            print("\n[SHUTDOWN] Complete!")
            break
        
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

if __name__ == "__main__":
    main()
