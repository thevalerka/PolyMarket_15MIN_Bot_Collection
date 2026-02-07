#!/usr/bin/env python3
"""
Quick test to verify logging works throughout the period
"""

import pandas as pd
from datetime import datetime, timedelta

# Generate test data for 15 minutes with 1-second granularity
start_time = datetime(2026, 1, 6, 10, 0, 0)
timestamps = [start_time + timedelta(seconds=i) for i in range(900)]

# Generate some price movement
import numpy as np
np.random.seed(42)
base_price = 91000.0
prices = base_price + np.cumsum(np.random.randn(900) * 10)

# Create DataFrame
df = pd.DataFrame({
    'timestamp': timestamps,
    'price': prices
})

print("Test Data Created:")
print(f"  Period: {timestamps[0]} to {timestamps[-1]}")
print(f"  Samples: {len(df)}")
print(f"  Price range: ${prices.min():.2f} - ${prices.max():.2f}")
print()

# Now test the logging
from sim_last_minute_clean import MarketDataTracker, simulate_period

tracker = MarketDataTracker()
balance = 100.0
strike = 91000.0

print("Starting simulation...")
print("You should see logs every 10 seconds, then every 1 second in last 20s")
print()

# Run simulation
final_balance, trades = simulate_period(df, strike, tracker, balance)

print()
print(f"Simulation complete!")
print(f"Final balance: ${final_balance:.2f}")
print(f"Trades executed: {len(trades)}")
