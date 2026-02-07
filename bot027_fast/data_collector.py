#!/usr/bin/env python3
"""
Live Data Collector for MM Backtesting
======================================

Continuously collects real market data from:
- Bybit BTC price
- Chainlink BTC price (with strike)
- Coinbase BTC price
- Polymarket CALL options
- Polymarket PUT options

Saves to CSV for later backtesting.

Usage:
    python3 data_collector.py                    # Collect indefinitely
    python3 data_collector.py --duration 3600   # Collect for 1 hour
    python3 data_collector.py --interval 1000   # Sample every 1000ms
"""

import json
import time
import csv
import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass, asdict

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATHS = {
    'bybit': '/home/ubuntu/013_2025_polymarket/bybit_btc_price.json',
    'chainlink': '/home/ubuntu/013_2025_polymarket/chainlink_btc_price.json',
    'coinbase': '/home/ubuntu/013_2025_polymarket/bot016_react/coinbase_btc_price.json',
    'call': '/home/ubuntu/013_2025_polymarket/15M_BTC_CALL_rest.json',
    'put': '/home/ubuntu/013_2025_polymarket/15M_BTC_PUT_rest.json',
}

OUTPUT_DIR = '/home/ubuntu/013_2025_polymarket/backtest_data'
DEFAULT_INTERVAL_MS = 500  # Sample every 500ms


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MarketTick:
    """Complete market snapshot at a point in time."""
    # Timestamps
    collection_time: int          # When we collected this data
    
    # BTC Prices from different sources
    bybit_price: Optional[float] = None
    bybit_timestamp: Optional[int] = None
    bybit_direction: Optional[str] = None
    
    chainlink_price: Optional[float] = None
    chainlink_timestamp: Optional[int] = None
    strike_price: Optional[float] = None
    strike_timestamp: Optional[int] = None
    
    coinbase_price: Optional[float] = None
    coinbase_timestamp: Optional[int] = None
    
    # CALL options
    call_bid: Optional[float] = None
    call_ask: Optional[float] = None
    call_spread: Optional[float] = None
    call_timestamp: Optional[int] = None
    call_market: Optional[str] = None
    
    # PUT options
    put_bid: Optional[float] = None
    put_ask: Optional[float] = None
    put_spread: Optional[float] = None
    put_timestamp: Optional[int] = None
    put_market: Optional[str] = None
    
    # Derived
    period_start: Optional[int] = None
    seconds_into_period: Optional[int] = None


# =============================================================================
# DATA READING
# =============================================================================

def read_json_safe(filepath: str) -> Optional[dict]:
    """Safely read JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return None


def get_period_start(timestamp_ms: int) -> int:
    """Calculate the start of the current 15-minute period."""
    period_ms = 15 * 60 * 1000
    return (timestamp_ms // period_ms) * period_ms


def collect_tick() -> MarketTick:
    """Collect a complete market snapshot."""
    collection_time = int(time.time() * 1000)
    tick = MarketTick(collection_time=collection_time)
    
    # Read Bybit
    bybit_data = read_json_safe(DATA_PATHS['bybit'])
    if bybit_data:
        tick.bybit_price = bybit_data.get('price')
        tick.bybit_timestamp = bybit_data.get('timestamp')
        tick.bybit_direction = bybit_data.get('direction')
    
    # Read Chainlink (includes strike)
    chainlink_data = read_json_safe(DATA_PATHS['chainlink'])
    if chainlink_data:
        tick.chainlink_price = chainlink_data.get('price')
        tick.chainlink_timestamp = chainlink_data.get('timestamp')
        tick.strike_price = chainlink_data.get('strike')
        tick.strike_timestamp = chainlink_data.get('strike_timestamp')
    
    # Read Coinbase
    coinbase_data = read_json_safe(DATA_PATHS['coinbase'])
    if coinbase_data:
        tick.coinbase_price = coinbase_data.get('price')
        tick.coinbase_timestamp = coinbase_data.get('timestamp')
    
    # Read CALL options
    call_data = read_json_safe(DATA_PATHS['call'])
    if call_data:
        tick.call_bid = call_data.get('best_bid')
        tick.call_ask = call_data.get('best_ask')
        tick.call_spread = call_data.get('spread')
        tick.call_timestamp = call_data.get('timestamp')
        tick.call_market = call_data.get('market')
    
    # Read PUT options
    put_data = read_json_safe(DATA_PATHS['put'])
    if put_data:
        tick.put_bid = put_data.get('best_bid')
        tick.put_ask = put_data.get('best_ask')
        tick.put_spread = put_data.get('spread')
        tick.put_timestamp = put_data.get('timestamp')
        tick.put_market = put_data.get('market')
    
    # Calculate period info
    if tick.strike_timestamp:
        tick.period_start = tick.strike_timestamp
        tick.seconds_into_period = (collection_time - tick.strike_timestamp) // 1000
    else:
        tick.period_start = get_period_start(collection_time)
        tick.seconds_into_period = (collection_time - tick.period_start) // 1000
    
    return tick


# =============================================================================
# DATA STORAGE
# =============================================================================

class DataCollector:
    """Collects and stores market data."""
    
    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_file: Optional[Path] = None
        self.current_writer = None
        self.current_handle = None
        self.current_date: Optional[str] = None
        
        self.ticks_collected = 0
        self.periods_seen = set()
        self.start_time = time.time()
        
    def get_output_filename(self) -> Path:
        """Get output filename for current date."""
        today = datetime.now().strftime('%Y%m%d')
        return self.output_dir / f"market_data_{today}.csv"
    
    def get_csv_headers(self) -> list:
        """Get CSV headers matching MarketTick fields."""
        return [
            'collection_time',
            'bybit_price', 'bybit_timestamp', 'bybit_direction',
            'chainlink_price', 'chainlink_timestamp', 'strike_price', 'strike_timestamp',
            'coinbase_price', 'coinbase_timestamp',
            'call_bid', 'call_ask', 'call_spread', 'call_timestamp', 'call_market',
            'put_bid', 'put_ask', 'put_spread', 'put_timestamp', 'put_market',
            'period_start', 'seconds_into_period',
        ]
    
    def ensure_file_open(self):
        """Ensure output file is open for current date."""
        today = datetime.now().strftime('%Y%m%d')
        
        if today != self.current_date:
            # Close old file
            if self.current_handle:
                self.current_handle.close()
            
            # Open new file
            self.current_file = self.get_output_filename()
            file_exists = self.current_file.exists()
            
            self.current_handle = open(self.current_file, 'a', newline='')
            self.current_writer = csv.DictWriter(
                self.current_handle, 
                fieldnames=self.get_csv_headers()
            )
            
            if not file_exists:
                self.current_writer.writeheader()
                print(f"📁 Created new data file: {self.current_file}")
            else:
                print(f"📁 Appending to: {self.current_file}")
            
            self.current_date = today
    
    def write_tick(self, tick: MarketTick):
        """Write a tick to the CSV file."""
        self.ensure_file_open()
        
        row = asdict(tick)
        self.current_writer.writerow(row)
        self.current_handle.flush()  # Ensure data is written
        
        self.ticks_collected += 1
        if tick.period_start:
            self.periods_seen.add(tick.period_start)
    
    def close(self):
        """Close output file."""
        if self.current_handle:
            self.current_handle.close()
    
    def print_status(self, tick: MarketTick):
        """Print collection status."""
        runtime = time.time() - self.start_time
        rate = self.ticks_collected / runtime if runtime > 0 else 0
        
        print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
              f"Ticks: {self.ticks_collected:,} | "
              f"Periods: {len(self.periods_seen)} | "
              f"Rate: {rate:.1f}/s | "
              f"BTC: ${tick.bybit_price:,.2f}" if tick.bybit_price else "N/A", 
              end='', flush=True)


# =============================================================================
# MAIN COLLECTION LOOP
# =============================================================================

def run_collector(duration_seconds: Optional[int] = None, interval_ms: int = DEFAULT_INTERVAL_MS):
    """Run the data collector."""
    print("="*70)
    print("   MARKET DATA COLLECTOR FOR MM BACKTESTING")
    print("="*70)
    print(f"   Interval: {interval_ms}ms")
    print(f"   Duration: {'Indefinite' if duration_seconds is None else f'{duration_seconds}s'}")
    print(f"   Output: {OUTPUT_DIR}")
    print("="*70)
    
    # Verify data sources exist
    print("\n📡 Checking data sources...")
    all_ok = True
    for name, path in DATA_PATHS.items():
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"   {status} {name}: {path}")
        if not exists:
            all_ok = False
    
    if not all_ok:
        print("\n⚠️  Some data sources missing. Collection may be incomplete.")
    
    print("\n🚀 Starting collection... (Ctrl+C to stop)\n")
    
    collector = DataCollector()
    start_time = time.time()
    last_print = 0
    
    try:
        while True:
            # Check duration
            if duration_seconds and (time.time() - start_time) >= duration_seconds:
                print(f"\n\n✅ Duration reached ({duration_seconds}s)")
                break
            
            # Collect tick
            tick = collect_tick()
            collector.write_tick(tick)
            
            # Print status every second
            if time.time() - last_print >= 1.0:
                collector.print_status(tick)
                last_print = time.time()
            
            # Wait for next interval
            time.sleep(interval_ms / 1000)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Stopped by user")
    
    finally:
        collector.close()
        
        print(f"\n{'='*70}")
        print(f"   COLLECTION SUMMARY")
        print(f"{'='*70}")
        print(f"   Total ticks: {collector.ticks_collected:,}")
        print(f"   Periods captured: {len(collector.periods_seen)}")
        print(f"   Runtime: {time.time() - start_time:.1f}s")
        print(f"   Output file: {collector.current_file}")
        print(f"{'='*70}")


# =============================================================================
# UTILITY: Convert collected data to backtest format
# =============================================================================

def convert_to_backtest_format(input_file: str, output_file: str):
    """
    Convert collected CSV to the format expected by mm_backtest.py
    """
    print(f"Converting {input_file} to backtest format...")
    
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        
        writer = csv.writer(outfile)
        writer.writerow([
            'timestamp', 'btc_price', 'call_bid', 'call_ask',
            'put_bid', 'put_ask', 'strike', 'period_start'
        ])
        
        rows_written = 0
        for row in reader:
            # Use Chainlink price as primary (it's what Polymarket uses)
            btc_price = row.get('chainlink_price') or row.get('bybit_price') or row.get('coinbase_price')
            
            # Skip rows with missing critical data
            if not all([
                btc_price,
                row.get('call_bid'),
                row.get('call_ask'),
                row.get('put_bid'),
                row.get('put_ask'),
                row.get('strike_price'),
                row.get('period_start'),
            ]):
                continue
            
            writer.writerow([
                row['collection_time'],
                btc_price,
                row['call_bid'],
                row['call_ask'],
                row['put_bid'],
                row['put_ask'],
                row['strike_price'],
                row['period_start'],
            ])
            rows_written += 1
        
        print(f"✅ Wrote {rows_written} rows to {output_file}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Market Data Collector')
    parser.add_argument('--duration', type=int, default=None,
                       help='Collection duration in seconds (default: indefinite)')
    parser.add_argument('--interval', type=int, default=DEFAULT_INTERVAL_MS,
                       help=f'Sample interval in ms (default: {DEFAULT_INTERVAL_MS})')
    parser.add_argument('--convert', type=str, default=None,
                       help='Convert collected CSV to backtest format')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for conversion')
    
    args = parser.parse_args()
    
    if args.convert:
        output = args.output or args.convert.replace('.csv', '_backtest.csv')
        convert_to_backtest_format(args.convert, output)
    else:
        run_collector(
            duration_seconds=args.duration,
            interval_ms=args.interval
        )


if __name__ == '__main__':
    main()
