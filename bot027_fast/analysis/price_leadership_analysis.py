#!/usr/bin/env python3
"""
Price Leadership & Options Reaction Analysis
============================================

This script determines:
1. Which price source is the LEADER (moves first, others follow)
2. Latency ranking of each source
3. How fast Polymarket options react to price changes in the underlying

Methodology:
- Continuously sample all price sources
- Detect significant price movements
- Track which source registers the move FIRST
- Measure time delta until other sources catch up
- Measure options reaction time to underlying moves

Run for at least 5-10 minutes during active market hours for best results.
"""

import json
import time
import os
import statistics
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import threading

# ============================================================================
# CONFIGURATION
# ============================================================================

PATHS = {
    'coinbase': '/home/ubuntu/013_2025_polymarket/bot016_react/coinbase_btc_price.json',
    'chainlink': '/home/ubuntu/013_2025_polymarket/chainlink_btc_price.json',
    'bybit': '/home/ubuntu/013_2025_polymarket/bybit_btc_price.json',
    'call': '/home/ubuntu/013_2025_polymarket/15M_BTC_CALL_rest.json',
    'put': '/home/ubuntu/013_2025_polymarket/15M_BTC_PUT_rest.json',
}

# Thresholds
PRICE_MOVE_THRESHOLD = 3.0  # $ change to consider a "move"
LEADERSHIP_WINDOW_MS = 2000  # How long to wait for other sources to follow
SAMPLE_INTERVAL_MS = 50  # Sampling interval

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class PriceReading:
    source: str
    price: float
    timestamp: int  # Source's timestamp
    read_time: int  # When we read it
    latency: int    # read_time - timestamp

@dataclass
class PriceMove:
    """Records a detected price movement event."""
    direction: str  # 'UP' or 'DOWN'
    magnitude: float  # $ amount
    leader_source: str
    leader_timestamp: int
    leader_price: float
    followers: Dict[str, Tuple[int, float]]  # source -> (timestamp, price)
    lag_times: Dict[str, int]  # source -> ms behind leader
    options_reaction_ms: Optional[int] = None
    options_direction_correct: Optional[bool] = None

@dataclass
class AnalysisState:
    """Holds all analysis state."""
    # Price tracking
    last_prices: Dict[str, float] = field(default_factory=dict)
    last_timestamps: Dict[str, int] = field(default_factory=dict)

    # Movement detection
    pending_move: Optional[dict] = None
    confirmed_moves: List[PriceMove] = field(default_factory=list)

    # Latency tracking
    latency_samples: Dict[str, List[int]] = field(default_factory=lambda: {
        'coinbase': [], 'chainlink': [], 'bybit': [], 'call': [], 'put': []
    })

    # Leadership tracking
    leadership_scores: Dict[str, int] = field(default_factory=lambda: {
        'coinbase': 0, 'chainlink': 0, 'bybit': 0
    })

    # Options reaction
    options_reactions: List[dict] = field(default_factory=list)

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def safe_read_json(filepath: str) -> Optional[dict]:
    """Thread-safe JSON file reading."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return None

def now_ms() -> int:
    """Current time in milliseconds."""
    return int(time.time() * 1000)

def read_all_prices(state: AnalysisState) -> Dict[str, PriceReading]:
    """Read all price sources."""
    read_time = now_ms()
    readings = {}

    for source in ['coinbase', 'chainlink', 'bybit']:
        data = safe_read_json(PATHS[source])
        if data and data.get('price') and data.get('timestamp'):
            reading = PriceReading(
                source=source,
                price=data['price'],
                timestamp=data['timestamp'],
                read_time=read_time,
                latency=read_time - data['timestamp']
            )
            readings[source] = reading
            state.latency_samples[source].append(reading.latency)

    return readings

def read_options(state: AnalysisState) -> Dict[str, dict]:
    """Read options data."""
    read_time = now_ms()
    options = {}

    for opt in ['call', 'put']:
        data = safe_read_json(PATHS[opt])
        if data:
            latency = read_time - data.get('timestamp', read_time)
            state.latency_samples[opt].append(latency)
            options[opt] = {
                'bid': data.get('best_bid'),
                'ask': data.get('best_ask'),
                'mid': (data.get('best_bid', 0) + data.get('best_ask', 0)) / 2,
                'timestamp': data.get('timestamp'),
                'latency': latency,
            }

    return options

def detect_price_leader(
    readings: Dict[str, PriceReading],
    state: AnalysisState
) -> Optional[Tuple[str, str, float]]:
    """
    Detect which source registered a price move first.
    Returns: (source, direction, magnitude) or None
    """
    moves_detected = []

    for source, reading in readings.items():
        last_price = state.last_prices.get(source)

        if last_price is not None:
            change = reading.price - last_price
            if abs(change) >= PRICE_MOVE_THRESHOLD:
                moves_detected.append({
                    'source': source,
                    'direction': 'UP' if change > 0 else 'DOWN',
                    'magnitude': abs(change),
                    'timestamp': reading.timestamp,
                    'price': reading.price,
                })

        # Update last price
        state.last_prices[source] = reading.price
        state.last_timestamps[source] = reading.timestamp

    if not moves_detected:
        return None

    # Find the earliest mover (by source timestamp)
    moves_detected.sort(key=lambda x: x['timestamp'])
    leader = moves_detected[0]

    return leader['source'], leader['direction'], leader['magnitude']

def analyze_leadership_event(
    leader_source: str,
    direction: str,
    magnitude: float,
    readings: Dict[str, PriceReading],
    state: AnalysisState
):
    """Record a leadership event and track followers."""
    leader = readings[leader_source]

    move = PriceMove(
        direction=direction,
        magnitude=magnitude,
        leader_source=leader_source,
        leader_timestamp=leader.timestamp,
        leader_price=leader.price,
        followers={},
        lag_times={},
    )

    # Check other sources for lag
    for source, reading in readings.items():
        if source != leader_source:
            lag = reading.timestamp - leader.timestamp
            move.followers[source] = (reading.timestamp, reading.price)
            move.lag_times[source] = lag

    # Update leadership score
    state.leadership_scores[leader_source] += 1
    state.confirmed_moves.append(move)

    return move

def print_live_status(
    readings: Dict[str, PriceReading],
    options: Dict[str, dict],
    state: AnalysisState
):
    """Print live monitoring status."""
    os.system('clear' if os.name == 'posix' else 'cls')

    print("="*75)
    print("   🔴 LIVE: BTC PRICE LEADERSHIP & OPTIONS REACTION MONITOR")
    print("="*75)
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    print(f"   Samples: {len(state.latency_samples['bybit'])} | Moves detected: {len(state.confirmed_moves)}")
    print("="*75)

    # Current prices
    print(f"\n{'─'*30} CURRENT PRICES {'─'*30}")
    print(f"{'Source':<12} {'Price':>14} {'Latency':>12} {'Direction':<10}")
    print(f"{'─'*75}")

    for source in ['coinbase', 'chainlink', 'bybit']:
        if source in readings:
            r = readings[source]
            # Determine if price is up or down from last
            last = state.last_prices.get(source)
            if last:
                if r.price > last:
                    arrow = "📈"
                elif r.price < last:
                    arrow = "📉"
                else:
                    arrow = "➡️"
            else:
                arrow = "➡️"

            print(f"{source.upper():<12} ${r.price:>13,.2f} {r.latency:>10}ms {arrow}")

    # Options
    print(f"\n{'─'*30} OPTIONS {'─'*30}")
    print(f"{'Type':<8} {'Bid':>10} {'Ask':>10} {'Mid':>10} {'Latency':>10}")
    print(f"{'─'*75}")

    for opt in ['call', 'put']:
        if opt in options:
            o = options[opt]
            print(f"{opt.upper():<8} {o['bid']:>10.4f} {o['ask']:>10.4f} "
                  f"{o['mid']:>10.4f} {o['latency']:>8}ms")

    # Implied direction
    if 'call' in options and 'put' in options:
        call_mid = options['call']['mid']
        put_mid = options['put']['mid']
        implied = "📈 BULLISH" if call_mid > put_mid else "📉 BEARISH"
        confidence = max(call_mid, put_mid)
        print(f"\n   Market Implied: {implied} ({confidence:.1%})")

    # Leadership scores
    print(f"\n{'─'*30} LEADERSHIP SCORES {'─'*30}")
    total_moves = sum(state.leadership_scores.values())

    if total_moves > 0:
        sorted_scores = sorted(
            state.leadership_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for i, (source, score) in enumerate(sorted_scores):
            pct = (score / total_moves) * 100
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
            print(f"   {medal} {source.upper():<12} {bar} {score:>4} ({pct:>5.1f}%)")
    else:
        print("   No price movements detected yet...")

    # Latency stats
    print(f"\n{'─'*30} LATENCY STATS {'─'*30}")
    print(f"{'Source':<12} {'Min':>10} {'Avg':>10} {'Max':>10} {'Samples':>10}")
    print(f"{'─'*75}")

    for source in ['coinbase', 'chainlink', 'bybit', 'call', 'put']:
        samples = state.latency_samples[source]
        if samples:
            print(f"{source.upper():<12} {min(samples):>10} {statistics.mean(samples):>10.1f} "
                  f"{max(samples):>10} {len(samples):>10}")

    # Recent moves
    if state.confirmed_moves:
        print(f"\n{'─'*30} RECENT MOVES {'─'*30}")
        for move in state.confirmed_moves[-5:]:
            lag_str = ", ".join(f"{s}: {l:+d}ms" for s, l in move.lag_times.items())
            print(f"   {move.direction} ${move.magnitude:.2f} | Leader: {move.leader_source.upper()} | Lags: {lag_str}")


def run_analysis(duration_seconds: int = 60):
    """Run continuous analysis."""
    state = AnalysisState()
    start_time = time.time()

    print(f"\n🚀 Starting {duration_seconds}s analysis...")
    print("   Press Ctrl+C to stop early\n")

    try:
        while time.time() - start_time < duration_seconds:
            # Read all data
            readings = read_all_prices(state)
            options = read_options(state)

            # Detect price leadership
            leader_info = detect_price_leader(readings, state)
            if leader_info:
                source, direction, magnitude = leader_info
                analyze_leadership_event(source, direction, magnitude, readings, state)

            # Print status
            print_live_status(readings, options, state)

            time.sleep(SAMPLE_INTERVAL_MS / 1000)

    except KeyboardInterrupt:
        print("\n\n⚠️ Stopped by user")

    # Final summary
    print_final_summary(state)
    return state

def print_final_summary(state: AnalysisState):
    """Print final analysis summary."""
    print("\n" + "="*75)
    print("   📊 FINAL ANALYSIS SUMMARY")
    print("="*75)

    # Latency ranking
    print(f"\n{'─'*30} LATENCY RANKING {'─'*30}")
    latencies = []
    for source in ['coinbase', 'chainlink', 'bybit']:
        samples = state.latency_samples[source]
        if samples:
            avg = statistics.mean(samples)
            latencies.append((source, avg, min(samples), max(samples), len(samples)))

    latencies.sort(key=lambda x: x[1])
    print(f"\n   FASTEST SOURCE (by average latency):")
    for i, (source, avg, min_l, max_l, n) in enumerate(latencies):
        medal = ["🥇", "🥈", "🥉"][i]
        print(f"   {medal} {source.upper()}: {avg:.1f}ms avg (min: {min_l}ms, max: {max_l}ms)")

    # Leadership ranking
    print(f"\n{'─'*30} LEADERSHIP RANKING {'─'*30}")
    total = sum(state.leadership_scores.values())

    if total > 0:
        sorted_scores = sorted(
            state.leadership_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        print(f"\n   LEADING INDICATOR (moves first):")
        for i, (source, score) in enumerate(sorted_scores):
            pct = (score / total) * 100
            medal = ["🥇", "🥈", "🥉"][i]
            print(f"   {medal} {source.upper()}: {score}/{total} moves ({pct:.1f}%)")
    else:
        print("   No significant price movements detected.")
        print("   Try running during active market hours or increase duration.")

    # Options reaction
    print(f"\n{'─'*30} OPTIONS CHARACTERISTICS {'─'*30}")
    for opt in ['call', 'put']:
        samples = state.latency_samples[opt]
        if samples:
            print(f"   {opt.upper()} latency: {statistics.mean(samples):.1f}ms avg")

    # Key findings
    print(f"\n{'─'*30} KEY FINDINGS {'─'*30}")

    if latencies:
        fastest = latencies[0][0]
        print(f"   ⚡ Fastest data source: {fastest.upper()}")

    if total > 0:
        leader = max(state.leadership_scores.items(), key=lambda x: x[1])[0]
        print(f"   👑 Price leader (moves first): {leader.upper()}")

        if fastest != leader:
            print(f"   ⚠️  Note: Fastest ≠ Leader! {fastest.upper()} has lower latency")
            print(f"       but {leader.upper()} tends to move first.")
            print(f"       → {leader.upper()} likely receives price updates first at source.")

    print("\n" + "="*75)


def main():
    import sys

    duration = 60
    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
        except:
            pass

    print("="*75)
    print("   BTC PRICE LEADERSHIP ANALYZER")
    print("   Determines which price source is the LEADING INDICATOR")
    print("="*75)

    # Verify files exist
    print("\n📁 Checking data files...")
    all_exist = True
    for name, path in PATHS.items():
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"   {status} {name}: {path}")
        if not exists:
            all_exist = False

    if not all_exist:
        print("\n❌ Some files are missing. Make sure your bots are running.")
        return

    print(f"\n   Running for {duration} seconds...")
    print("   Usage: python3 price_leadership_analysis.py [seconds]")

    run_analysis(duration)


if __name__ == '__main__':
    main()
