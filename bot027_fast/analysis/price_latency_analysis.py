#!/usr/bin/env python3
"""
Price Source Latency & Leadership Analysis
Compares Coinbase, Chainlink, and Bybit BTC prices to determine:
1. Which source is fastest (lowest latency)
2. Which is the leading price indicator
3. How fast Polymarket options react to price changes

Run continuously to build statistical evidence.
"""

import json
import time
import os
import statistics
from datetime import datetime
from collections import deque
from pathlib import Path

# Configuration - adjust paths as needed
PATHS = {
    'coinbase': '/home/ubuntu/013_2025_polymarket/bot016_react/coinbase_btc_price.json',
    'chainlink': '/home/ubuntu/013_2025_polymarket/chainlink_btc_price.json',
    'bybit': '/home/ubuntu/013_2025_polymarket/bybit_btc_price.json',
    'call': '/home/ubuntu/013_2025_polymarket/bot016_react/15M_BTC_CALL_rest.json',
    'put': '/home/ubuntu/013_2025_polymarket/bot016_react/15M_BTC_PUT_rest.json',
}

# Data storage for analysis
HISTORY_SIZE = 1000
price_history = {
    'coinbase': deque(maxlen=HISTORY_SIZE),
    'chainlink': deque(maxlen=HISTORY_SIZE),
    'bybit': deque(maxlen=HISTORY_SIZE),
}
options_history = {
    'call': deque(maxlen=HISTORY_SIZE),
    'put': deque(maxlen=HISTORY_SIZE),
}

# For detecting price movements and measuring reaction times
last_prices = {'coinbase': None, 'chainlink': None, 'bybit': None}
price_change_events = deque(maxlen=500)  # Records which source moved first
latency_samples = {'coinbase': [], 'chainlink': [], 'bybit': []}
options_reaction_times = []


def safe_read_json(filepath):
    """Read JSON file with error handling."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return None


def get_current_time_ms():
    """Get current time in milliseconds."""
    return int(time.time() * 1000)


def calculate_latency(data_timestamp, current_time_ms):
    """Calculate latency in milliseconds."""
    return current_time_ms - data_timestamp


def read_all_sources():
    """Read all price sources and options data."""
    current_time = get_current_time_ms()
    
    readings = {
        'read_time': current_time,
        'read_time_readable': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
        'prices': {},
        'options': {},
    }
    
    # Read price sources
    for source in ['coinbase', 'chainlink', 'bybit']:
        data = safe_read_json(PATHS[source])
        if data:
            readings['prices'][source] = {
                'price': data.get('price'),
                'timestamp': data.get('timestamp'),
                'latency_ms': calculate_latency(data.get('timestamp', 0), current_time),
            }
    
    # Read options
    for option in ['call', 'put']:
        data = safe_read_json(PATHS[option])
        if data:
            readings['options'][option] = {
                'best_bid': data.get('best_bid'),
                'best_ask': data.get('best_ask'),
                'mid': (data.get('best_bid', 0) + data.get('best_ask', 0)) / 2,
                'spread': data.get('spread'),
                'timestamp': data.get('timestamp'),
                'delay_ms': data.get('delay_ms'),
                'latency_ms': calculate_latency(data.get('timestamp', 0), current_time),
            }
    
    return readings


def detect_price_leadership(readings):
    """
    Detect which price source moves first.
    Returns the leading source if a significant price change is detected.
    """
    global last_prices
    
    PRICE_CHANGE_THRESHOLD = 5.0  # $5 minimum change to register
    
    leaders = []
    
    for source in ['coinbase', 'chainlink', 'bybit']:
        if source not in readings['prices']:
            continue
            
        current_price = readings['prices'][source]['price']
        if current_price is None:
            continue
            
        if last_prices[source] is not None:
            change = abs(current_price - last_prices[source])
            if change >= PRICE_CHANGE_THRESHOLD:
                leaders.append({
                    'source': source,
                    'price': current_price,
                    'change': current_price - last_prices[source],
                    'timestamp': readings['prices'][source]['timestamp'],
                    'latency': readings['prices'][source]['latency_ms'],
                })
        
        last_prices[source] = current_price
    
    return leaders


def analyze_options_reaction(readings, price_change_direction):
    """
    Analyze how options prices react to underlying price changes.
    """
    if 'call' not in readings['options'] or 'put' not in readings['options']:
        return None
    
    call = readings['options']['call']
    put = readings['options']['put']
    
    # Theoretical: if price goes UP, CALL should increase, PUT should decrease
    # if price goes DOWN, CALL should decrease, PUT should increase
    
    return {
        'call_mid': call['mid'],
        'put_mid': put['mid'],
        'call_spread': call['spread'],
        'put_spread': put['spread'],
        'call_latency': call['latency_ms'],
        'put_latency': put['latency_ms'],
        'implied_direction': 'UP' if call['mid'] > put['mid'] else 'DOWN',
    }


def print_current_snapshot(readings):
    """Print current market snapshot."""
    print(f"\n{'='*70}")
    print(f"📊 MARKET SNAPSHOT - {readings['read_time_readable']}")
    print(f"{'='*70}")
    
    # Price sources comparison
    print(f"\n{'─'*35} PRICE SOURCES {'─'*35}")
    print(f"{'Source':<12} {'Price':>12} {'Timestamp':>15} {'Latency':>12} {'Status'}")
    print(f"{'─'*70}")
    
    prices = readings['prices']
    min_latency = float('inf')
    fastest_source = None
    
    for source in ['coinbase', 'chainlink', 'bybit']:
        if source in prices:
            p = prices[source]
            latency = p['latency_ms']
            if latency < min_latency:
                min_latency = latency
                fastest_source = source
            
            status = "🟢" if latency < 500 else "🟡" if latency < 1000 else "🔴"
            print(f"{source.upper():<12} ${p['price']:>11,.2f} {p['timestamp']:>15} {latency:>10}ms {status}")
        else:
            print(f"{source.upper():<12} {'N/A':>12} {'N/A':>15} {'N/A':>12} ⚪")
    
    if fastest_source:
        print(f"\n⚡ FASTEST SOURCE: {fastest_source.upper()} ({min_latency}ms latency)")
    
    # Price discrepancies
    if len(prices) >= 2:
        price_values = [p['price'] for p in prices.values() if p['price']]
        if price_values:
            max_diff = max(price_values) - min(price_values)
            print(f"📈 Price spread across sources: ${max_diff:.2f}")
    
    # Options data
    print(f"\n{'─'*35} OPTIONS {'─'*35}")
    print(f"{'Option':<8} {'Bid':>8} {'Ask':>8} {'Mid':>8} {'Spread':>8} {'Latency':>10}")
    print(f"{'─'*70}")
    
    opts = readings['options']
    for option in ['call', 'put']:
        if option in opts:
            o = opts[option]
            print(f"{option.upper():<8} {o['best_bid']:>8.4f} {o['best_ask']:>8.4f} "
                  f"{o['mid']:>8.4f} {o['spread']:>8.4f} {o['latency_ms']:>8}ms")
    
    # Market implied direction
    if 'call' in opts and 'put' in opts:
        call_mid = opts['call']['mid']
        put_mid = opts['put']['mid']
        
        if call_mid > put_mid:
            direction = "📈 BULLISH"
            confidence = call_mid
        else:
            direction = "📉 BEARISH"
            confidence = put_mid
        
        print(f"\n🎯 Market Implied Direction: {direction} (confidence: {confidence:.2%})")
        
        # Compare with Chainlink strike
        chainlink = prices.get('chainlink', {})
        if chainlink.get('price'):
            # We need strike from the original chainlink data
            chainlink_data = safe_read_json(PATHS['chainlink'])
            if chainlink_data and 'strike' in chainlink_data:
                strike = chainlink_data['strike']
                current = chainlink_data['price']
                diff = current - strike
                pct_diff = (diff / strike) * 100
                
                actual_direction = "ABOVE" if diff > 0 else "BELOW"
                print(f"\n📍 Strike Price: ${strike:,.2f}")
                print(f"📍 Current Price: ${current:,.2f}")
                print(f"📍 Difference: ${diff:+,.2f} ({pct_diff:+.3f}%) - {actual_direction} strike")


def collect_latency_stats(readings):
    """Collect latency statistics for analysis."""
    for source in ['coinbase', 'chainlink', 'bybit']:
        if source in readings['prices']:
            latency = readings['prices'][source]['latency_ms']
            latency_samples[source].append(latency)


def print_latency_statistics():
    """Print latency statistics summary."""
    print(f"\n{'='*70}")
    print(f"📈 LATENCY STATISTICS (based on {len(latency_samples['bybit'])} samples)")
    print(f"{'='*70}")
    
    print(f"\n{'Source':<12} {'Min':>10} {'Max':>10} {'Mean':>10} {'Median':>10} {'StdDev':>10}")
    print(f"{'─'*70}")
    
    rankings = []
    for source in ['coinbase', 'chainlink', 'bybit']:
        samples = latency_samples[source]
        if len(samples) >= 2:
            min_lat = min(samples)
            max_lat = max(samples)
            mean_lat = statistics.mean(samples)
            median_lat = statistics.median(samples)
            stdev_lat = statistics.stdev(samples)
            rankings.append((source, mean_lat))
            
            print(f"{source.upper():<12} {min_lat:>10.1f} {max_lat:>10.1f} "
                  f"{mean_lat:>10.1f} {median_lat:>10.1f} {stdev_lat:>10.1f}")
    
    if rankings:
        rankings.sort(key=lambda x: x[1])
        print(f"\n🏆 LATENCY RANKING (fastest to slowest):")
        for i, (source, mean) in enumerate(rankings, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            print(f"   {medal} {i}. {source.upper()} - {mean:.1f}ms average latency")


def continuous_monitoring(duration_seconds=60, interval_ms=100):
    """
    Run continuous monitoring for specified duration.
    
    Args:
        duration_seconds: How long to monitor
        interval_ms: Sampling interval in milliseconds
    """
    print(f"\n🔄 Starting continuous monitoring for {duration_seconds} seconds...")
    print(f"   Sampling every {interval_ms}ms")
    
    start_time = time.time()
    sample_count = 0
    
    try:
        while time.time() - start_time < duration_seconds:
            readings = read_all_sources()
            collect_latency_stats(readings)
            
            # Print snapshot every 5 seconds
            if sample_count % (5000 // interval_ms) == 0:
                print_current_snapshot(readings)
            
            sample_count += 1
            time.sleep(interval_ms / 1000)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Monitoring interrupted by user")
    
    print(f"\n✅ Collected {sample_count} samples")
    print_latency_statistics()


def single_snapshot():
    """Take a single snapshot and display analysis."""
    readings = read_all_sources()
    print_current_snapshot(readings)
    return readings


def main():
    """Main entry point."""
    import sys
    
    print("="*70)
    print("   BTC PRICE SOURCE & OPTIONS REACTION ANALYZER")
    print("="*70)
    
    # Check if files exist
    print("\n📁 Checking file paths...")
    for name, path in PATHS.items():
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"   {exists} {name}: {path}")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--monitor':
            duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
            continuous_monitoring(duration_seconds=duration)
        elif sys.argv[1] == '--stats':
            # Run quick sampling for statistics
            continuous_monitoring(duration_seconds=30, interval_ms=50)
    else:
        # Single snapshot mode
        single_snapshot()
        print("\n" + "="*70)
        print("💡 USAGE:")
        print("   python3 price_latency_analysis.py           # Single snapshot")
        print("   python3 price_latency_analysis.py --monitor # Monitor for 60s")
        print("   python3 price_latency_analysis.py --monitor 120  # Monitor for 120s")
        print("   python3 price_latency_analysis.py --stats   # Quick stats (30s)")
        print("="*70)


if __name__ == '__main__':
    main()
