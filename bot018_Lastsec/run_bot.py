#!/usr/bin/env python3
"""
Last-Minute Volatility Scalping Bot - Live Runner

Fetches live BTC price data and runs the simulator in real-time.
"""

import sys
import time
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from collections import deque

# Import simulator functions
from sim_last_minute_clean import (
    MarketDataTracker,
    simulate_period,
    save_state,
    load_state,
    INITIAL_BALANCE
)

# ============================================================================
# LIVE DATA COLLECTION
# ============================================================================

def get_current_btc_price() -> float:
    """Get current BTC price from Bybit"""
    try:
        url = "https://api.bybit.com/v5/market/tickers?category=linear&symbol=BTCUSDT"
        response = requests.get(url, timeout=5)
        data = response.json()
        return float(data['result']['list'][0]['lastPrice'])
    except Exception as e:
        print(f"[ERROR] Failed to get BTC price: {e}")
        return None

def get_current_strike() -> float:
    """Get strike price for current 15-minute period"""
    try:
        now = datetime.now(timezone.utc)
        
        # Determine which 15-min period we're in
        minute = now.minute
        if minute < 15:
            start_min = 0
        elif minute < 30:
            start_min = 15
        elif minute < 45:
            start_min = 30
        else:
            start_min = 45
        
        # Get period start time
        period_start = now.replace(minute=start_min, second=0, microsecond=0)
        
        # Fetch the strike (opening price of the period)
        url = "https://api.bybit.com/v5/market/mark-price-kline"
        params = {
            'category': 'linear',
            'symbol': 'BTCUSDT',
            'interval': '15',
            'start': int(period_start.timestamp() * 1000),
            'limit': 1
        }
        
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        
        if data.get('retCode') == 0:
            candles = data.get('result', {}).get('list', [])
            if candles:
                strike = float(candles[0][1])  # Open price
                return strike
        
        return None
    
    except Exception as e:
        print(f"[ERROR] Failed to get strike: {e}")
        return None

def collect_period_data(duration_seconds: int = 900) -> pd.DataFrame:
    """
    Collect live price data for one period
    
    Args:
        duration_seconds: How long to collect (default 900 = 15 minutes)
    
    Returns:
        DataFrame with timestamp and price columns
    """
    
    print(f"\n[DATA COLLECTION] Starting {duration_seconds}s collection...")
    
    data = []
    start_time = datetime.now(timezone.utc)
    end_time = start_time + timedelta(seconds=duration_seconds)
    
    last_log = 0
    
    while datetime.now(timezone.utc) < end_time:
        current_time = datetime.now(timezone.utc)
        
        # Get price
        price = get_current_btc_price()
        
        if price:
            data.append({
                'timestamp': current_time,
                'price': price
            })
            
            # Log progress every 30 seconds
            if int(current_time.timestamp()) % 30 == 0 and int(current_time.timestamp()) != last_log:
                elapsed = (current_time - start_time).total_seconds()
                remaining = (end_time - current_time).total_seconds()
                print(f"[COLLECTING] {elapsed:.0f}s elapsed, {remaining:.0f}s remaining | "
                      f"Samples: {len(data)} | Current: ${price:.2f}")
                last_log = int(current_time.timestamp())
        
        # Sleep 1 second
        time.sleep(1)
    
    print(f"[DATA COLLECTION] Complete! Collected {len(data)} samples\n")
    
    return pd.DataFrame(data)

# ============================================================================
# MAIN BOT LOOP
# ============================================================================

def run_live_bot():
    """Run the bot continuously, period by period"""
    
    print("=" * 80)
    print("LAST-MINUTE VOLATILITY SCALPING BOT - LIVE MODE")
    print("=" * 80)
    print("Collecting live BTC data and simulating trades in real-time")
    print("Press Ctrl+C to stop")
    print("=" * 80)
    
    # Load state
    tracker, balance = load_state()
    if tracker is None:
        tracker = MarketDataTracker()
        balance = INITIAL_BALANCE
    
    print(f"\nStarting Balance: ${balance:.2f}\n")
    
    period_count = 0
    
    try:
        while True:
            period_count += 1
            
            print(f"\n{'#' * 80}")
            print(f"PERIOD #{period_count}")
            print(f"{'#' * 80}\n")
            
            # Get current strike
            strike = get_current_strike()
            if strike is None:
                print("[ERROR] Could not get strike price, skipping period")
                time.sleep(60)
                continue
            
            print(f"Strike Price: ${strike:.2f}")
            
            # Collect 15 minutes of data
            df = collect_period_data(duration_seconds=900)
            
            if len(df) < 100:
                print(f"[ERROR] Not enough data collected ({len(df)} samples), skipping")
                continue
            
            # Run simulation
            balance, trades = simulate_period(df, strike, tracker, balance)
            
            # Save state
            save_state(tracker, balance)
            
            print(f"\n[PERIOD END] Balance: ${balance:.2f} | Trades: {len(trades)}\n")
            
            # Small pause before next period
            time.sleep(5)
    
    except KeyboardInterrupt:
        print("\n\n[SHUTDOWN] Stopping bot...")
        save_state(tracker, balance)
        print(f"[SHUTDOWN] Final balance: ${balance:.2f}")
        print("[SHUTDOWN] State saved. Goodbye!\n")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    
    print("\nLast-Minute Volatility Scalping Bot")
    print("=" * 40)
    print("\nOptions:")
    print("  1. Run LIVE bot (collect real data)")
    print("  2. Test with sample data")
    print()
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == '1':
        print("\n⚠️  WARNING: This will run for 15+ minutes per period!")
        print("⚠️  Make sure you have stable internet connection")
        confirm = input("\nContinue? (yes/no): ").strip().lower()
        
        if confirm == 'yes':
            run_live_bot()
        else:
            print("Cancelled.")
    
    elif choice == '2':
        print("\nRunning test with sample data...")
        from test_logging import *
    
    else:
        print("Invalid choice")
