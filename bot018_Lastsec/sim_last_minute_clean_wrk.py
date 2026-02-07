#!/usr/bin/env python3
"""
Last-Minute Volatility Scalping Simulator

Pure volatility-based strategy trading only in the final 60 seconds.

Entry Rules:
1. Last 60 seconds: Open if |distance| > volatility-long
2. Last 20 seconds: Open if |distance| > volatility-short
3. If no ask (only 0.99 bid): Open at 0.99 if conditions met

Direction:
- distance > 0 (above strike) → CALL
- distance < 0 (below strike) → PUT

Exit:
- At expiry: Compare 15-min candle close vs strike
"""

import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, timezone, timedelta
from collections import deque
from typing import Dict, List, Optional, Tuple
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

# Simulation Parameters
INITIAL_BALANCE = 100.0
POSITION_SIZE = 10.0  # $ per trade

# Trading Windows
ENTRY_WINDOW_LONG = 60   # Last 60 seconds
ENTRY_WINDOW_SHORT = 20  # Last 20 seconds

# Volatility Parameters
VOLATILITY_LONG_PERIODS = 12   # 12 minutes of 1-minute candles
VOLATILITY_SHORT_PERIODS = 12  # 120 seconds / 10 seconds = 12 periods

# Choppiness
CHOPPINESS_PERIOD = 900  # 15 minutes in seconds

# Special Pricing
MAX_BID_ENTRY = 0.99  # If no ask, can enter at 0.99 bid

# Persistence
STATE_FILE = "sim_last_minute_state.json"
TRADES_DIR = "sim_last_minute_trades"

# ============================================================================
# DATA STRUCTURES
# ============================================================================

class MarketDataTracker:
    """Track market data for volatility and choppiness calculations"""

    def __init__(self):
        # 1-minute candle data (for volatility-long)
        self.minute_candles = deque(maxlen=VOLATILITY_LONG_PERIODS)
        self.current_minute_high = None
        self.current_minute_low = None
        self.last_minute_timestamp = None

        # 10-second range data (for volatility-short)
        self.short_ranges = deque(maxlen=VOLATILITY_SHORT_PERIODS)
        self.current_10s_high = None
        self.current_10s_low = None
        self.last_10s_timestamp = None

        # Price history for choppiness (15 minutes = 900 samples)
        self.price_history = deque(maxlen=CHOPPINESS_PERIOD)

    def update(self, timestamp: datetime, price: float):
        """Update all tracking data structures"""

        # Add to price history
        self.price_history.append(price)

        # Update 1-minute candles
        current_minute = timestamp.replace(second=0, microsecond=0)
        if self.last_minute_timestamp is None or current_minute > self.last_minute_timestamp:
            # New minute started
            if self.current_minute_high is not None:
                # Save completed candle
                candle_range = self.current_minute_high - self.current_minute_low
                self.minute_candles.append(candle_range)

            # Start new candle
            self.current_minute_high = price
            self.current_minute_low = price
            self.last_minute_timestamp = current_minute
        else:
            # Update current candle
            self.current_minute_high = max(self.current_minute_high, price)
            self.current_minute_low = min(self.current_minute_low, price)

        # Update 10-second ranges
        current_10s = timestamp.replace(second=(timestamp.second // 10) * 10, microsecond=0)
        if self.last_10s_timestamp is None or current_10s > self.last_10s_timestamp:
            # New 10-second period started
            if self.current_10s_high is not None:
                # Save completed range
                range_value = self.current_10s_high - self.current_10s_low
                self.short_ranges.append(range_value)

            # Start new range
            self.current_10s_high = price
            self.current_10s_low = price
            self.last_10s_timestamp = current_10s
        else:
            # Update current range
            self.current_10s_high = max(self.current_10s_high, price)
            self.current_10s_low = min(self.current_10s_low, price)

    def get_volatility_long(self) -> float:
        """Calculate EMA of 1-minute candle ranges"""
        if len(self.minute_candles) < 2:
            return 0.0

        ranges = list(self.minute_candles)

        if len(ranges) < VOLATILITY_LONG_PERIODS:
            # Not enough data, use simple average
            return np.mean(ranges)

        # Calculate EMA
        multiplier = 2.0 / (VOLATILITY_LONG_PERIODS + 1)
        ema = ranges[0]

        for i in range(1, len(ranges)):
            ema = (ranges[i] * multiplier) + (ema * (1 - multiplier))

        return ema

    def get_volatility_short(self) -> float:
        """Calculate EMA of 10-second ranges"""
        if len(self.short_ranges) < 2:
            return 0.0

        ranges = list(self.short_ranges)

        if len(ranges) < VOLATILITY_SHORT_PERIODS:
            # Not enough data, use simple average
            return np.mean(ranges)

        # Calculate EMA
        multiplier = 2.0 / (VOLATILITY_SHORT_PERIODS + 1)
        ema = ranges[0]

        for i in range(1, len(ranges)):
            ema = (ranges[i] * multiplier) + (ema * (1 - multiplier))

        return ema

    def get_choppiness(self) -> float:
        """Calculate Choppiness Index (0-100)"""
        if len(self.price_history) < 10:
            return 50.0

        prices = np.array(list(self.price_history))

        # Calculate true ranges
        true_ranges = np.abs(np.diff(prices))
        sum_tr = np.sum(true_ranges)

        # High and low
        high = np.max(prices)
        low = np.min(prices)
        high_low_range = high - low

        if high_low_range == 0 or sum_tr == 0:
            return 100.0

        # Choppiness Index
        period = len(prices)
        choppiness = 100 * np.log10(sum_tr / high_low_range) / np.log10(period)

        return max(0.0, min(100.0, choppiness))

# ============================================================================
# BYBIT API
# ============================================================================

def fetch_period_candle(period_start: datetime) -> Optional[Dict]:
    """
    Fetch the 15-minute candle for a specific period from Bybit

    Returns:
        Dict with 'open', 'high', 'low', 'close' or None
    """
    try:
        url = "https://api.bybit.com/v5/market/mark-price-kline"
        params = {
            'category': 'linear',
            'symbol': 'BTCUSDT',
            'interval': '15',
            'start': int(period_start.timestamp() * 1000),
            'limit': 1
        }

        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        if data.get('retCode') != 0:
            return None

        candles = data.get('result', {}).get('list', [])
        if not candles:
            return None

        # Format: [timestamp, open, high, low, close]
        candle = candles[0]

        return {
            'timestamp': int(candle[0]),
            'open': float(candle[1]),
            'high': float(candle[2]),
            'low': float(candle[3]),
            'close': float(candle[4])
        }

    except Exception as e:
        print(f"[ERROR] Failed to fetch candle: {e}")
        return None

# ============================================================================
# TRADING LOGIC
# ============================================================================

def determine_entry_signal(
    distance: float,
    vol_long: float,
    vol_short: float,
    time_to_expiry: float,
    call_ask: Optional[float],
    call_bid: Optional[float],
    put_ask: Optional[float],
    put_bid: Optional[float]
) -> Optional[Dict]:
    """
    Determine if we should enter a position

    Returns:
        Dict with 'type' (CALL/PUT), 'price', 'reason' or None
    """

    abs_distance = abs(distance)

    # Rule 1: Last 60 seconds - use volatility-long
    if time_to_expiry <= ENTRY_WINDOW_LONG and time_to_expiry > ENTRY_WINDOW_SHORT:
        if abs_distance > vol_long:
            if distance > 0:
                # CALL
                if call_ask is not None and call_ask < MAX_BID_ENTRY:
                    return {
                        'type': 'CALL',
                        'price': call_ask,
                        'reason': f'LONG_VOL (dist ${abs_distance:.2f} > vol_long ${vol_long:.2f})'
                    }
                elif call_ask is None and call_bid is not None and call_bid >= 0.99:
                    # Deep ITM - no ask available, use 0.99 bid
                    return {
                        'type': 'CALL',
                        'price': 0.99,
                        'reason': f'LONG_VOL_DEEP_ITM (dist ${abs_distance:.2f} > vol_long ${vol_long:.2f})'
                    }
            else:
                # PUT
                if put_ask is not None and put_ask < MAX_BID_ENTRY:
                    return {
                        'type': 'PUT',
                        'price': put_ask,
                        'reason': f'LONG_VOL (dist ${abs_distance:.2f} > vol_long ${vol_long:.2f})'
                    }
                elif put_ask is None and put_bid is not None and put_bid >= 0.99:
                    # Deep ITM - no ask available, use 0.99 bid
                    return {
                        'type': 'PUT',
                        'price': 0.99,
                        'reason': f'LONG_VOL_DEEP_ITM (dist ${abs_distance:.2f} > vol_long ${vol_long:.2f})'
                    }

    # Rule 2: Last 20 seconds - use volatility-short
    elif time_to_expiry <= ENTRY_WINDOW_SHORT:
        if abs_distance > vol_short:
            if distance > 0:
                # CALL
                if call_ask is not None and call_ask < MAX_BID_ENTRY:
                    return {
                        'type': 'CALL',
                        'price': call_ask,
                        'reason': f'SHORT_VOL (dist ${abs_distance:.2f} > vol_short ${vol_short:.2f})'
                    }
                elif call_ask is None and call_bid is not None and call_bid >= 0.99:
                    # Deep ITM - no ask available, use 0.99 bid
                    return {
                        'type': 'CALL',
                        'price': 0.99,
                        'reason': f'SHORT_VOL_DEEP_ITM (dist ${abs_distance:.2f} > vol_short ${vol_short:.2f})'
                    }
            else:
                # PUT
                if put_ask is not None and put_ask < MAX_BID_ENTRY:
                    return {
                        'type': 'PUT',
                        'price': put_ask,
                        'reason': f'SHORT_VOL (dist ${abs_distance:.2f} > vol_short ${vol_short:.2f})'
                    }
                elif put_ask is None and put_bid is not None and put_bid >= 0.99:
                    # Deep ITM - no ask available, use 0.99 bid
                    return {
                        'type': 'PUT',
                        'price': 0.99,
                        'reason': f'SHORT_VOL_DEEP_ITM (dist ${abs_distance:.2f} > vol_short ${vol_short:.2f})'
                    }

    return None

# ============================================================================
# PERSISTENCE
# ============================================================================

def save_state(tracker: MarketDataTracker, balance: float):
    """Save simulator state"""
    try:
        state = {
            'balance': balance,
            'minute_candles': list(tracker.minute_candles),
            'short_ranges': list(tracker.short_ranges),
            'price_history': list(tracker.price_history),
            'last_save': datetime.now(timezone.utc).isoformat()
        }

        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"[STATE] Saved: Balance=${balance:.2f}")
    except Exception as e:
        print(f"[ERROR] Failed to save state: {e}")

def load_state() -> Tuple[Optional[MarketDataTracker], float]:
    """Load simulator state"""
    if not os.path.exists(STATE_FILE):
        return None, INITIAL_BALANCE

    try:
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)

        tracker = MarketDataTracker()
        tracker.minute_candles = deque(state.get('minute_candles', []), maxlen=VOLATILITY_LONG_PERIODS)
        tracker.short_ranges = deque(state.get('short_ranges', []), maxlen=VOLATILITY_SHORT_PERIODS)
        tracker.price_history = deque(state.get('price_history', []), maxlen=CHOPPINESS_PERIOD)

        balance = state.get('balance', INITIAL_BALANCE)

        print(f"[STATE] Loaded: Balance=${balance:.2f}, History={len(tracker.price_history)} samples")
        return tracker, balance

    except Exception as e:
        print(f"[ERROR] Failed to load state: {e}")
        return None, INITIAL_BALANCE

def save_trade(trade: Dict):
    """Save trade to daily file"""
    try:
        os.makedirs(TRADES_DIR, exist_ok=True)

        trade_date = datetime.fromisoformat(trade['entry_time']).strftime('%Y-%m-%d')
        trades_file = os.path.join(TRADES_DIR, f"trades_{trade_date}.json")

        # Load existing trades
        trades = []
        if os.path.exists(trades_file):
            with open(trades_file, 'r') as f:
                trades = json.load(f)

        # Add new trade
        trades.append(trade)

        # Save
        with open(trades_file, 'w') as f:
            json.dump(trades, f, indent=2)

        print(f"[TRADE] Saved to {trades_file}")

    except Exception as e:
        print(f"[ERROR] Failed to save trade: {e}")

# ============================================================================
# SIMULATION
# ============================================================================

def simulate_period(
    df: pd.DataFrame,
    strike_price: float,
    tracker: MarketDataTracker,
    balance: float
) -> Tuple[float, List[Dict]]:
    """
    Simulate one 15-minute period

    Args:
        df: DataFrame with 'timestamp' and 'price' columns
        strike_price: Strike price for this period
        tracker: Market data tracker
        balance: Current balance

    Returns:
        Tuple of (final_balance, list_of_trades)
    """

    position = None
    trades = []
    last_log_time = 0

    # Get period boundaries
    period_start = df['timestamp'].iloc[0]
    period_end = period_start + timedelta(minutes=15)

    print(f"\n{'='*80}")
    print(f"PERIOD: {period_start.strftime('%H:%M:%S')} - {period_end.strftime('%H:%M:%S')}")
    print(f"Strike: ${strike_price:.2f} | Balance: ${balance:.2f}")
    print(f"{'='*80}\n")

    for idx, row in df.iterrows():
        timestamp = row['timestamp']
        price = row['price']

        # Update tracker
        tracker.update(timestamp, price)

        # Calculate time to expiry
        time_to_expiry = (period_end - timestamp).total_seconds()

        # Get metrics
        vol_long = tracker.get_volatility_long()
        vol_short = tracker.get_volatility_short()
        choppiness = tracker.get_choppiness()

        # Calculate distance from strike
        distance = price - strike_price

        # Determine logging frequency
        current_second = int(timestamp.timestamp())

        # Log every second in last 20 seconds, otherwise every 10 seconds
        should_log = False
        if time_to_expiry <= 20:
            should_log = True  # Every second
        elif current_second % 10 == 0:
            if current_second != last_log_time:
                should_log = True  # Every 10 seconds
                last_log_time = current_second

        if should_log:
            time_str = timestamp.strftime('%H:%M:%S')

            # Position status
            if position:
                pos_str = f"{position['type']}@${position['entry_price']:.2f}"
                entry_time_str = position['entry_time'].strftime('%H:%M:%S')
                time_in_pos = (timestamp - position['entry_time']).total_seconds()
                pos_info = f" | Pos:{pos_str} ({time_in_pos:.0f}s since {entry_time_str})"
            else:
                pos_info = " | Pos:NONE"

            # Format metrics
            vol_long_str = f"${vol_long:.2f}" if vol_long > 0 else "N/A"
            vol_short_str = f"${vol_short:.2f}" if vol_short > 0 else "N/A"

            # Highlight last 20 seconds
            if time_to_expiry <= 20:
                prefix = "🔥"
                ttl_str = f"{time_to_expiry:.0f}s"
            else:
                prefix = "  "
                ttl_str = f"{time_to_expiry:.0f}s"

            print(f"{prefix}[{time_str}] BTC:${price:.2f} | Dist:${distance:+.2f} | "
                  f"Vol-L:{vol_long_str} Vol-S:{vol_short_str} | Chop:{choppiness:.1f} | "
                  f"TTL:{ttl_str}{pos_info}")

        # Only trade in last 60 seconds
        if time_to_expiry <= ENTRY_WINDOW_LONG and time_to_expiry > 0 and position is None:

            # Simulate option prices (simplified)
            abs_dist = abs(distance)
            time_factor = time_to_expiry / 60.0

            # Simple option pricing based on distance and time
            if distance > 0:
                call_ask = 0.5 + (abs_dist / strike_price * 100) * 0.5
                put_ask = 0.5 - (abs_dist / strike_price * 100) * 0.3
            else:
                call_ask = 0.5 - (abs_dist / strike_price * 100) * 0.3
                put_ask = 0.5 + (abs_dist / strike_price * 100) * 0.5

            call_ask = max(0.01, min(0.98, call_ask))
            put_ask = max(0.01, min(0.98, put_ask))

            # Simulate bids (slightly lower than ask)
            call_bid = call_ask - 0.01 if call_ask > 0.01 else None
            put_bid = put_ask - 0.01 if put_ask > 0.01 else None

            # Sometimes no ask available (only high bid)
            if time_to_expiry < 5:  # Very close to expiry
                if np.random.random() < 0.3:  # 30% chance
                    if distance > 0:
                        call_ask = None
                        call_bid = MAX_BID_ENTRY
                    else:
                        put_ask = None
                        put_bid = MAX_BID_ENTRY

            # Check entry signal
            signal = determine_entry_signal(
                distance, vol_long, vol_short, time_to_expiry,
                call_ask, call_bid, put_ask, put_bid
            )

            if signal and balance >= POSITION_SIZE:
                position = {
                    'type': signal['type'],
                    'entry_price': signal['price'],
                    'entry_time': timestamp,
                    'entry_btc': price,
                    'strike_price': strike_price,
                    'distance_at_entry': distance,
                    'time_to_expiry': time_to_expiry,
                    'reason': signal['reason'],
                    'volatility_long': vol_long,
                    'volatility_short': vol_short,
                    'choppiness': choppiness
                }

                balance -= POSITION_SIZE

                print(f"\n{'='*80}")
                print(f"✅ OPENED {signal['type']} @${signal['price']:.2f}")
                print(f"{'='*80}")
                print(f"  Time: {timestamp.strftime('%H:%M:%S')}")
                print(f"  Reason: {signal['reason']}")
                print(f"  BTC: ${price:.2f} (Distance: ${distance:+.2f} from strike)")
                print(f"  Vol-Long: ${vol_long:.2f} | Vol-Short: ${vol_short:.2f}")
                print(f"  Choppiness: {choppiness:.1f}")
                print(f"  Time to Expiry: {time_to_expiry:.0f}s")
                print(f"  Balance: ${balance:.2f}")
                print(f"{'='*80}\n")

    # At expiry - fetch actual candle to determine result
    if position is not None:
        print(f"\n{'='*80}")
        print(f"⏰ EXPIRY - Fetching 15-min candle from Bybit API...")
        print(f"{'='*80}")

        candle = fetch_period_candle(period_start)

        if candle:
            final_btc = candle['close']

            print(f"  Candle Data:")
            print(f"    Open:  ${candle['open']:.2f}")
            print(f"    High:  ${candle['high']:.2f}")
            print(f"    Low:   ${candle['low']:.2f}")
            print(f"    Close: ${candle['close']:.2f}")
            print(f"\n  Settlement:")
            print(f"    Strike: ${strike_price:.2f}")
            print(f"    Close:  ${final_btc:.2f}")

            # Determine final value
            if position['type'] == 'CALL':
                final_value = 1.0 if final_btc > strike_price else 0.0
                result_text = "WIN ✅" if final_btc > strike_price else "LOSS ❌"
                print(f"    Type: CALL (win if close > strike)")
                print(f"    Result: {result_text}")
            else:  # PUT
                final_value = 1.0 if final_btc < strike_price else 0.0
                result_text = "WIN ✅" if final_btc < strike_price else "LOSS ❌"
                print(f"    Type: PUT (win if close < strike)")
                print(f"    Result: {result_text}")

            # Calculate PNL
            pnl = (final_value - position['entry_price']) * POSITION_SIZE
            balance += (final_value * POSITION_SIZE)

            result = 'WIN' if pnl > 0 else 'LOSS'

            print(f"\n  Trade Summary:")
            print(f"    Entry:  ${position['entry_price']:.2f}")
            print(f"    Exit:   ${final_value:.2f}")
            print(f"    PNL:    ${pnl:+.2f} ({pnl/POSITION_SIZE*100:+.1f}%)")
            print(f"    Balance: ${balance:.2f}")

            # Record trade
            trade = {
                'entry_time': position['entry_time'].isoformat(),
                'expiry_time': period_end.isoformat(),
                'type': position['type'],
                'entry_price': position['entry_price'],
                'exit_price': final_value,
                'entry_btc': position['entry_btc'],
                'exit_btc': final_btc,
                'strike_price': strike_price,
                'distance_at_entry': position['distance_at_entry'],
                'pnl': pnl,
                'result': result,
                'reason': position['reason'],
                'time_to_expiry_at_entry': position['time_to_expiry'],
                'volatility_long': position['volatility_long'],
                'volatility_short': position['volatility_short'],
                'choppiness': position['choppiness']
            }

            trades.append(trade)
            save_trade(trade)

            print(f"{'='*80}\n")

        else:
            print(f"[ERROR] Could not fetch candle - treating as no trade")
            print(f"{'='*80}\n")

    return balance, trades

# ============================================================================
# LIVE DATA COLLECTION
# ============================================================================

# File paths
PUT_FILE = "/home/ubuntu/013_2025_polymarket/15M_PUT.json"
CALL_FILE = "/home/ubuntu/013_2025_polymarket/15M_CALL.json"
BTC_FILE = "/home/ubuntu/013_2025_polymarket/bybit_btc_price.json"

def get_live_btc_price() -> Optional[float]:
    """Get BTC price from local file"""
    try:
        with open(BTC_FILE, 'r') as f:
            data = json.load(f)
            return float(data.get('price', 0))
    except:
        return None

def get_live_option_prices() -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Get CALL and PUT prices from local files

    Returns:
        (call_ask, call_bid, put_ask, put_bid)
    """
    try:
        # Read CALL file
        with open(CALL_FILE, 'r') as f:
            call_data = json.load(f)

        call_ask = call_data.get('best_ask', {}).get('price')
        call_bid = call_data.get('best_bid', {}).get('price')

        # Read PUT file
        with open(PUT_FILE, 'r') as f:
            put_data = json.load(f)

        put_ask = put_data.get('best_ask', {}).get('price')
        put_bid = put_data.get('best_bid', {}).get('price')

        return call_ask, call_bid, put_ask, put_bid

    except:
        return None, None, None, None

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main simulation - runs immediately with live data"""

    print("=" * 80)
    print("LAST-MINUTE VOLATILITY SCALPING SIMULATOR")
    print("=" * 80)
    print(f"Entry Window: Last {ENTRY_WINDOW_LONG}s (vol-long) / Last {ENTRY_WINDOW_SHORT}s (vol-short)")
    print(f"Position Size: ${POSITION_SIZE}")
    print(f"Vol-Long: EMA of {VOLATILITY_LONG_PERIODS} 1-min candles")
    print(f"Vol-Short: EMA of {VOLATILITY_SHORT_PERIODS} 10-sec ranges")
    print("=" * 80)
    print()

    # Load or initialize state
    tracker, balance = load_state()
    if tracker is None:
        tracker = MarketDataTracker()
        balance = INITIAL_BALANCE
        print(f"[STATE] Starting fresh | Balance: ${balance:.2f}")
    else:
        print(f"[STATE] Loaded | Balance: ${balance:.2f}")

    print()

    # Get current period info
    now = datetime.now(timezone.utc)
    minute = now.minute
    if minute < 15:
        start_min = 0
    elif minute < 30:
        start_min = 15
    elif minute < 45:
        start_min = 30
    else:
        start_min = 45

    period_start = now.replace(minute=start_min, second=0, microsecond=0)
    period_end = period_start + timedelta(minutes=15)

    # Fetch strike from Bybit
    print(f"[INIT] Fetching strike price for period starting {period_start.strftime('%H:%M:%S')}...")
    candle = fetch_period_candle(period_start)

    if candle is None:
        print("[ERROR] Could not fetch strike price from Bybit!")
        return

    strike_price = candle['open']

    print(f"\n{'='*80}")
    print(f"PERIOD: {period_start.strftime('%H:%M:%S')} - {period_end.strftime('%H:%M:%S')}")
    print(f"Strike: ${strike_price:.2f} | Balance: ${balance:.2f}")
    print(f"{'='*80}\n")

    # Start data collection
    print(f"[LIVE] Starting live data collection...")
    print(f"[LIVE] Reading from:")
    print(f"  BTC:  {BTC_FILE}")
    print(f"  CALL: {CALL_FILE}")
    print(f"  PUT:  {PUT_FILE}")
    print()

    position = None
    trades = []
    last_log_time = 0

    # Main loop
    try:
        while True:
            timestamp = datetime.now(timezone.utc)

            # Check if period ended
            if timestamp >= period_end:
                print(f"\n[PERIOD END] Reached {period_end.strftime('%H:%M:%S')}")
                break

            # Get live data
            btc_price = get_live_btc_price()
            if btc_price is None:
                time.sleep(0.1)
                continue

            # Update tracker
            tracker.update(timestamp, btc_price)

            # Calculate time to expiry
            time_to_expiry = (period_end - timestamp).total_seconds()

            # Get metrics
            vol_long = tracker.get_volatility_long()
            vol_short = tracker.get_volatility_short()
            choppiness = tracker.get_choppiness()

            # Calculate distance from strike
            distance = btc_price - strike_price

            # Determine logging frequency
            current_second = int(timestamp.timestamp())

            # Log every second in last 20 seconds, otherwise every 10 seconds
            should_log = False
            if time_to_expiry <= 20:
                should_log = True  # Every second
            elif current_second % 10 == 0:
                if current_second != last_log_time:
                    should_log = True  # Every 10 seconds
                    last_log_time = current_second

            if should_log:
                time_str = timestamp.strftime('%H:%M:%S')

                # Get current option prices for display
                call_ask, call_bid, put_ask, put_bid = get_live_option_prices()

                # Format option prices - handle None asks (deep ITM)
                if call_ask is not None:
                    call_ask_str = f"${call_ask:.2f}"
                elif call_bid is not None and call_bid >= 0.99:
                    call_ask_str = "ITM"  # Deep in the money
                else:
                    call_ask_str = "N/A"

                call_bid_str = f"${call_bid:.2f}" if call_bid else "N/A"

                if put_ask is not None:
                    put_ask_str = f"${put_ask:.2f}"
                elif put_bid is not None and put_bid >= 0.99:
                    put_ask_str = "ITM"  # Deep in the money
                else:
                    put_ask_str = "N/A"

                put_bid_str = f"${put_bid:.2f}" if put_bid else "N/A"

                # Position status
                if position:
                    pos_str = f"{position['type']}@${position['entry_price']:.2f}"
                    entry_time_str = position['entry_time'].strftime('%H:%M:%S')
                    time_in_pos = (timestamp - position['entry_time']).total_seconds()
                    pos_info = f" | Pos:{pos_str} ({time_in_pos:.0f}s since {entry_time_str})"
                else:
                    pos_info = " | Pos:NONE"

                # Format metrics
                vol_long_str = f"${vol_long:.2f}" if vol_long > 0 else "N/A"
                vol_short_str = f"${vol_short:.2f}" if vol_short > 0 else "N/A"

                # Highlight last 20 seconds
                if time_to_expiry <= 20:
                    prefix = "🔥"
                    ttl_str = f"{time_to_expiry:.0f}s"
                else:
                    prefix = "  "
                    ttl_str = f"{time_to_expiry:.0f}s"

                print(f"{prefix}[{time_str}] BTC:${btc_price:.2f} | Dist:${distance:+.2f} | "
                      f"Vol-L:{vol_long_str} Vol-S:{vol_short_str} | Chop:{choppiness:.1f} | "
                      f"C:{call_bid_str}/{call_ask_str} P:{put_bid_str}/{put_ask_str} | "
                      f"TTL:{ttl_str}{pos_info}")

            # Only trade in last 60 seconds
            if time_to_expiry <= ENTRY_WINDOW_LONG and time_to_expiry > 0 and position is None:

                # Get live option prices
                call_ask, call_bid, put_ask, put_bid = get_live_option_prices()

                # Check entry signal
                signal = determine_entry_signal(
                    distance, vol_long, vol_short, time_to_expiry,
                    call_ask, call_bid, put_ask, put_bid
                )

                if signal and balance >= POSITION_SIZE:
                    position = {
                        'type': signal['type'],
                        'entry_price': signal['price'],
                        'entry_time': timestamp,
                        'entry_btc': btc_price,
                        'strike_price': strike_price,
                        'distance_at_entry': distance,
                        'time_to_expiry': time_to_expiry,
                        'reason': signal['reason'],
                        'volatility_long': vol_long,
                        'volatility_short': vol_short,
                        'choppiness': choppiness
                    }

                    # Deduct actual cost (entry_price * position_size)
                    actual_cost = signal['price'] * POSITION_SIZE
                    balance -= actual_cost

                    print(f"\n{'='*80}")
                    print(f"✅ OPENED {signal['type']} @${signal['price']:.2f}")
                    print(f"{'='*80}")
                    print(f"  Time: {timestamp.strftime('%H:%M:%S')}")
                    print(f"  Reason: {signal['reason']}")
                    print(f"  BTC: ${btc_price:.2f} (Distance: ${distance:+.2f} from strike)")
                    print(f"  Vol-Long: ${vol_long:.2f} | Vol-Short: ${vol_short:.2f}")
                    print(f"  Choppiness: {choppiness:.1f}")
                    print(f"  Time to Expiry: {time_to_expiry:.0f}s")
                    print(f"  Balance: ${balance:.2f}")
                    print(f"{'='*80}\n")

            # Sleep 1 second
            time.sleep(1)

        # At expiry - fetch actual candle to determine result
        if position is not None:
            print(f"\n{'='*80}")
            print(f"⏰ EXPIRY - Fetching 15-min candle from Bybit API...")
            print(f"{'='*80}")

            candle = fetch_period_candle(period_start)

            if candle:
                final_btc = candle['close']

                print(f"  Candle Data:")
                print(f"    Open:  ${candle['open']:.2f}")
                print(f"    High:  ${candle['high']:.2f}")
                print(f"    Low:   ${candle['low']:.2f}")
                print(f"    Close: ${candle['close']:.2f}")
                print(f"\n  Settlement:")
                print(f"    Strike: ${strike_price:.2f}")
                print(f"    Close:  ${final_btc:.2f}")

                # Determine final value
                if position['type'] == 'CALL':
                    final_value = 1.0 if final_btc > strike_price else 0.0
                    result_text = "WIN ✅" if final_btc > strike_price else "LOSS ❌"
                    print(f"    Type: CALL (win if close > strike)")
                    print(f"    Result: {result_text}")
                else:  # PUT
                    final_value = 1.0 if final_btc < strike_price else 0.0
                    result_text = "WIN ✅" if final_btc < strike_price else "LOSS ❌"
                    print(f"    Type: PUT (win if close < strike)")
                    print(f"    Result: {result_text}")

                # Calculate PNL
                pnl = (final_value - position['entry_price']) * POSITION_SIZE
                balance += (final_value * POSITION_SIZE)

                result = 'WIN' if pnl > 0 else 'LOSS'

                print(f"\n  Trade Summary:")
                print(f"    Entry:  ${position['entry_price']:.2f}")
                print(f"    Exit:   ${final_value:.2f}")
                print(f"    PNL:    ${pnl:+.2f} ({pnl/POSITION_SIZE*100:+.1f}%)")
                print(f"    Balance: ${balance:.2f}")

                # Record trade
                trade = {
                    'entry_time': position['entry_time'].isoformat(),
                    'expiry_time': period_end.isoformat(),
                    'type': position['type'],
                    'entry_price': position['entry_price'],
                    'exit_price': final_value,
                    'entry_btc': position['entry_btc'],
                    'exit_btc': final_btc,
                    'strike_price': strike_price,
                    'distance_at_entry': position['distance_at_entry'],
                    'pnl': pnl,
                    'result': result,
                    'reason': position['reason'],
                    'time_to_expiry_at_entry': position['time_to_expiry'],
                    'volatility_long': position['volatility_long'],
                    'volatility_short': position['volatility_short'],
                    'choppiness': position['choppiness']
                }

                trades.append(trade)
                save_trade(trade)

                print(f"{'='*80}\n")

            else:
                print(f"[ERROR] Could not fetch candle")
                print(f"{'='*80}\n")

        # Save state
        save_state(tracker, balance)

        print(f"\n[COMPLETE] Period finished | Balance: ${balance:.2f} | Trades: {len(trades)}\n")

    except KeyboardInterrupt:
        print(f"\n\n[STOPPED] Interrupted by user")
        save_state(tracker, balance)
        print(f"[STOPPED] State saved | Balance: ${balance:.2f}\n")

if __name__ == '__main__':
    main()
