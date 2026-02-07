#!/usr/bin/env python3
"""
Strategy D3 - Pure EMA Crossover Trading Bot

Differences from D2:
- ZERO tolerance: Trade exactly at EMA crossover
- Minimum price movement filter: Don't close/flip if |close_price - open_price| <= $0.04
- More aggressive: React to every EMA cross
- Reduces noise: Avoids trades with tiny option price movements

Trading Rules:
- OPEN CALL: BTC crosses ABOVE EMA AND RSI < 90
- OPEN PUT: BTC crosses BELOW EMA AND RSI > 10
- CLOSE: RSI extreme (90/10) BUT only if option moved > $0.04
- FLIP: Wrong side of EMA BUT only if option moved > $0.04

Example:
- Entry at $0.50, current at $0.52 → |0.52 - 0.50| = $0.02 ≤ $0.04 → Don't close (noise)
- Entry at $0.50, current at $0.60 → |0.60 - 0.50| = $0.10 > $0.04 → Close ✅
- Entry at $0.50, current at $0.45 → |0.45 - 0.50| = $0.05 > $0.04 → Close ✅
"""

import requests
import time
import json
import os
from datetime import datetime, timezone
from collections import deque
import numpy as np
from typing import Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

# API Configuration
MARKET_ID = "0x013193189491bf0b822ab91096e0ddabe3f8537c76b357be6dfee3d8f15b1ebb"
PUT_FILE = "/home/ubuntu/013_2025_polymarket/15M_PUT.json"
CALL_FILE = "/home/ubuntu/013_2025_polymarket/15M_CALL.json"

# Loop intervals
CHECK_INTERVAL = 0.1  # 100ms - trading loop
SAMPLE_INTERVAL = 1.0  # 1 second - indicator sampling

# Trading Parameters
MIN_BUY_PRICE = 0.03
MAX_BUY_PRICE = 0.97
CONFIRMATION_SECONDS = 1
START_DELAY = 20  # Wait 20s from period start

# RSI/EMA Parameters
RSI_PERIOD_INITIAL = 60  # 60 seconds
RSI_BREACH_UPPER = 80
RSI_BREACH_LOWER = 20
RSI_PERIOD_MIN = 15
RSI_PERIOD_MAX = 120  # Cap at 120 seconds
RSI_PERIOD_ADJUSTMENT = 5
BREACH_WINDOW = 900  # 15 minutes

# Trading Thresholds
RSI_EXIT_CALL = 90
RSI_EXIT_PUT = 10
MIN_TOLERANCE = 0.0  # D3: ZERO tolerance - trade at exact EMA
MIN_PRICE_MOVEMENT = 0.04  # D3: Minimum option price movement to close/flip ($0.04)
MIN_PRICE_MOVEMENT_TP = 0.03  # D3: Minimum option price movement to close/flip ($0.04)
MIN_PRICE_MOVEMENT_SL = 0.1  # D3: Minimum option price movement to close/flip ($0.04)

# Persistence
STATE_FILE = "/home/ubuntu/013_2025_polymarket/bot017_RSI/bot_d3_state.json"
TRADES_DIR = "/home/ubuntu/013_2025_polymarket/bot017_RSI/bot017_RSI_trades"
STATE_SAVE_INTERVAL = 60

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_btc_price():
    """Get current BTC price from Bybit"""
    try:
        url = "https://api.bybit.com/v5/market/tickers?category=linear&symbol=BTCUSDT"
        response = requests.get(url, timeout=5)
        data = response.json()
        return float(data['result']['list'][0]['lastPrice'])
    except Exception as e:
        return None

def get_strike_price() -> Optional[float]:
    """Get the strike price for the current 15-minute period"""
    try:
        now = datetime.now(timezone.utc)
        for start_min in [0, 15, 30, 45]:
            if now.minute >= start_min and now.minute < start_min + 15:
                period_start = now.replace(minute=start_min, second=0, microsecond=0)
                url = "https://api.bybit.com/v5/market/mark-price-kline"
                params = {
                    'category': 'linear',
                    'symbol': 'BTCUSDT',
                    'interval': '15',
                    'start': int(period_start.timestamp() * 1000),
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
    except Exception as e:
        return None

def get_option_prices():
    """Get current option prices from local JSON files"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with open(CALL_FILE, 'r') as f:
                content = f.read()
                if not content.strip():
                    if attempt < max_retries - 1:
                        time.sleep(0.01)
                        continue
                    return None, None, None, None
                call_data = json.loads(content)

            with open(PUT_FILE, 'r') as f:
                content = f.read()
                if not content.strip():
                    if attempt < max_retries - 1:
                        time.sleep(0.01)
                        continue
                    return None, None, None, None
                put_data = json.loads(content)

            call_best_bid = call_data.get('best_bid')
            call_best_ask = call_data.get('best_ask')
            call_bid = float(call_best_bid['price']) if call_best_bid else None
            call_ask = float(call_best_ask['price']) if call_best_ask else None

            put_best_bid = put_data.get('best_bid')
            put_best_ask = put_data.get('best_ask')
            put_bid = float(put_best_bid['price']) if put_best_bid else None
            put_ask = float(put_best_ask['price']) if put_best_ask else None

            return call_ask, call_bid, put_ask, put_bid

        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(0.01)
                continue
            return None, None, None, None
        except Exception as e:
            return None, None, None, None

    return None, None, None, None

def calculate_rsi(prices: deque, period: int) -> Optional[float]:
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

def calculate_ema(prices: deque, period: int) -> Optional[float]:
    """Calculate EMA on 1-second samples"""
    if len(prices) < period:
        return None

    prices_array = np.array(list(prices))[-period:]
    k = 2.0 / (period + 1)
    ema = prices_array[0]

    for price in prices_array[1:]:
        ema = price * k + ema * (1 - k)

    return ema

def calculate_volatility(prices: deque) -> float:
    """Calculate exponentially weighted volatility"""
    if len(prices) < 10:
        return 0.0

    prices_array = np.array(list(prices))
    weights = np.exp(np.linspace(-2, 0, len(prices_array)))
    weights = weights / weights.sum()

    weighted_mean = np.average(prices_array, weights=weights)
    variance = np.average((prices_array - weighted_mean) ** 2, weights=weights)

    return np.sqrt(variance)

def calculate_choppiness_index(prices: deque, period: int = 900) -> float:
    """
    Calculate Choppiness Index over the 15-minute period (900 seconds)

    Choppiness Index measures market choppiness on a scale of 0-100:
    - Values near 100 = Very choppy/ranging market (frequent direction changes)
    - Values near 0 = Strong trending market (consistent direction)
    - Values 38.2-61.8 = Transitional/neutral

    Formula: 100 * log10(sum(TR) / (max(high) - min(low))) / log10(period)
    Where TR = True Range = max(high-low, |high-prev_close|, |low-prev_close|)

    For 1-second price data, we use price as both high and low

    Args:
        prices: Deque of 1-second prices
        period: Lookback period in seconds (default 900 = 15 minutes)

    Returns:
        Choppiness Index value (0-100)
    """
    if len(prices) < period:
        return 50.0  # Return neutral value if not enough data

    # Get last 'period' prices
    prices_array = np.array(list(prices)[-period:])

    # Calculate True Range for each period
    # Since we have 1-second prices, TR is just the absolute price change
    true_ranges = np.abs(np.diff(prices_array))
    sum_tr = np.sum(true_ranges)

    # Get high and low over the period
    high = np.max(prices_array)
    low = np.min(prices_array)
    high_low_range = high - low

    # Avoid division by zero
    if high_low_range == 0:
        return 100.0  # Completely flat = maximum choppiness

    # Calculate Choppiness Index
    if sum_tr == 0:
        return 100.0

    choppiness = 100 * np.log10(sum_tr / high_low_range) / np.log10(period)

    # Clamp to 0-100 range
    choppiness = max(0.0, min(100.0, choppiness))

    return choppiness

def get_bin_key(timestamp):
    """Get 15-minute bin key"""
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    minute_bin = (dt.minute // 15) * 15
    return dt.replace(minute=minute_bin, second=0, microsecond=0).isoformat()

def get_seconds_into_period(timestamp):
    """Get seconds elapsed in current 15-minute period"""
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    minute_in_bin = dt.minute % 15
    return minute_in_bin * 60 + dt.second

def get_seconds_remaining(timestamp):
    """Get seconds remaining in current 15-minute period"""
    return 900 - get_seconds_into_period(timestamp)

# ============================================================================
# PERSISTENCE
# ============================================================================

def load_state():
    """Load bot state"""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                state['price_history_1s'] = deque(state.get('price_history_1s', []), maxlen=1000)
                print(f"[STATE] Loaded: RSI period={state.get('rsi_period', RSI_PERIOD_INITIAL)}s, "
                      f"History={len(state['price_history_1s'])} samples")
                return state
        except Exception as e:
            print(f"[ERROR] Failed to load state: {e}")

    print("[STATE] Starting fresh")
    return {
        'rsi_period': RSI_PERIOD_INITIAL,
        'price_history_1s': deque(maxlen=1000),
        'total_breaches': 0,
        'last_adjustment_time': 0
    }

def save_state(state):
    """Save bot state"""
    try:
        state_copy = state.copy()
        state_copy['price_history_1s'] = list(state['price_history_1s'])

        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, 'w') as f:
            json.dump(state_copy, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to save state: {e}")

def load_trades():
    """Load trade history"""
    os.makedirs(TRADES_DIR, exist_ok=True)

    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    trades_file = os.path.join(TRADES_DIR, f"trades_D3_{today}.json")

    if os.path.exists(trades_file):
        try:
            with open(trades_file, 'r') as f:
                trades = json.load(f)
                print(f"[TRADES] Loaded {len(trades.get('trades', []))} trades, PNL: ${trades.get('total_pnl', 0):.2f}")
                return trades
        except Exception as e:
            print(f"[ERROR] Failed to load trades: {e}")

    print(f"[TRADES] Starting fresh for {today}")
    return {
        'trades': [],
        'total_pnl': 0.0,
        'stats': {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'call_trades': 0,
            'put_trades': 0
        }
    }

def save_trade(trades, trade_record):
    """Save a completed trade"""
    try:
        trades['trades'].append(trade_record)
        trades['total_pnl'] += trade_record['pnl']
        trades['stats']['total_trades'] += 1

        if trade_record['pnl'] > 0:
            trades['stats']['winning_trades'] += 1
        else:
            trades['stats']['losing_trades'] += 1

        if trade_record['type'] == 'CALL':
            trades['stats']['call_trades'] += 1
        else:
            trades['stats']['put_trades'] += 1

        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        trades_file = os.path.join(TRADES_DIR, f"trades_D3_{today}.json")

        with open(trades_file, 'w') as f:
            json.dump(trades, f, indent=2)

        print(f"[TRADE] {trade_record['type']} | PNL: ${trade_record['pnl']:+.3f} | Total: ${trades['total_pnl']:+.2f}")

    except Exception as e:
        print(f"[ERROR] Failed to save trade: {e}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("STRATEGY D2 - EMA BANDS TRADING BOT (CLEAN VERSION)")
    print("=" * 80)
    print(f"Indicator Sampling: 1 second")
    print(f"Trading Loop: 0.1 seconds")
    print(f"RSI Period: {RSI_PERIOD_INITIAL}s (adaptive, min {RSI_PERIOD_MIN}s)")
    print("=" * 80)

    # Load state and trades
    state = load_state()
    trades = load_trades()

    # State
    price_history_1s = state['price_history_1s']
    rsi_period = state['rsi_period']
    total_breaches = state['total_breaches']
    last_adjustment_time = state.get('last_adjustment_time', 0)

    # Trading state
    position = None
    last_signal_type = None
    last_signal_time = 0
    last_d3_type_opened = None
    current_bin = None
    strike_price = None

    # Indicator values (updated every 1 second)
    rsi = None
    ema = None
    volatility = 0.0
    choppiness = 50.0  # Choppiness Index (0-100, neutral start at 50)

    # Timing
    last_sample_time = 0
    last_state_save = time.time()

    print(f"\n[STARTING] RSI Period: {rsi_period}s | History: {len(price_history_1s)} samples\n")

    while True:
        try:
            current_time = time.time()
            timestamp = current_time

            # Get BTC price
            btc_price = get_btc_price()
            if btc_price is None:
                time.sleep(CHECK_INTERVAL)
                continue

            # Sample price every 1 second for indicators
            if current_time - last_sample_time >= SAMPLE_INTERVAL:
                price_history_1s.append(btc_price)
                last_sample_time = current_time

                # Calculate indicators ONLY when new sample arrives (every 1 second)
                rsi = calculate_rsi(price_history_1s, rsi_period)
                ema = calculate_ema(price_history_1s, rsi_period)

                # Calculate volatility
                volatility = calculate_volatility(price_history_1s)

                # Calculate Choppiness Index for the 15-minute period
                choppiness = calculate_choppiness_index(price_history_1s, period=900)

                # Debug: Log RSI value when we have a position
                if position:
                    print(f"[RSI CHECK] {datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime('%H:%M:%S')} "
                          f"RSI{rsi_period}s={rsi:.2f} | Position: {position['type']} | "
                          f"Exit threshold: {'>' if position['type'] == 'CALL' else '<'} "
                          f"{RSI_EXIT_CALL if position['type'] == 'CALL' else RSI_EXIT_PUT}")

                # Count breaches using CURRENT rsi_period
                # This means when period changes, breaches are recalculated with new period
                if len(price_history_1s) >= rsi_period + 1:
                    # Get last 15 minutes of data
                    breach_window_samples = min(BREACH_WINDOW, len(price_history_1s))
                    recent_prices = list(price_history_1s)[-breach_window_samples:]

                    breach_count = 0
                    in_breach = False
                    breach_events = []  # Track all breach events for logging

                    # Slide window of CURRENT rsi_period through data
                    num_windows = len(recent_prices) - rsi_period

                    if num_windows > 0:
                        for i in range(num_windows):
                            window = recent_prices[i:i + rsi_period + 1]
                            window_rsi = calculate_rsi(deque(window), rsi_period)

                            if window_rsi is not None:
                                is_outside = window_rsi > RSI_BREACH_UPPER or window_rsi < RSI_BREACH_LOWER

                                # Count event START (transition from normal to breach)
                                if is_outside and not in_breach:
                                    breach_count += 1
                                    breach_type = "HIGH" if window_rsi > RSI_BREACH_UPPER else "LOW"
                                    breach_events.append({
                                        'window_index': i,
                                        'rsi': window_rsi,
                                        'type': breach_type
                                    })
                                    in_breach = True
                                # Track event END (transition from breach to normal)
                                elif not is_outside and in_breach:
                                    in_breach = False

                    # Log breach details every 10 seconds
                    if current_time - last_adjustment_time >= 10:
                        print(f"\n[BREACH CALC] Period: RSI{rsi_period}s | Window: {breach_window_samples}s ({num_windows} windows)")
                        print(f"[BREACH CALC] Total breaches: {breach_count}")
                        if breach_events:
                            print(f"[BREACH CALC] Events:")
                            for idx, event in enumerate(breach_events[-5:], 1):  # Show last 5
                                window_time = breach_window_samples - event['window_index'] - rsi_period
                                print(f"  {idx}. {event['type']} breach at window {event['window_index']} "
                                      f"(~{window_time}s ago) RSI={event['rsi']:.1f}")

                    # Adjust period every 10 seconds
                    # When period changes, breaches will be recalculated next sample with new period
                    if current_time - last_adjustment_time >= 10:
                        old_period = rsi_period

                        if breach_count <= 3 and rsi_period > RSI_PERIOD_MIN:
                            rsi_period = max(RSI_PERIOD_MIN, rsi_period - RSI_PERIOD_ADJUSTMENT)
                            print(f"[RSI ADJUST] Breaches: {breach_count} ≤ 3 → Period: {old_period}s → {rsi_period}s (DECREASE)\n")
                        elif breach_count >= 8 and rsi_period < RSI_PERIOD_MAX:
                            rsi_period = min(RSI_PERIOD_MAX, rsi_period + RSI_PERIOD_ADJUSTMENT)
                            print(f"[RSI ADJUST] Breaches: {breach_count} ≥ 8 → Period: {old_period}s → {rsi_period}s (INCREASE)\n")
                        else:
                            print(f"[RSI ADJUST] Breaches: {breach_count} (4-7) → Period: {old_period}s (NO CHANGE)\n")

                        last_adjustment_time = current_time
                        total_breaches = breach_count

            # Get option prices
            call_ask, call_bid, put_ask, put_bid = get_option_prices()

            # Check if we have needed prices
            prices_available = True
            if position:
                if position['type'] == 'CALL' and call_bid is None:
                    prices_available = False
                elif position['type'] == 'PUT' and put_bid is None:
                    prices_available = False
            else:
                if call_ask is None or put_ask is None:
                    prices_available = False

            if not prices_available:
                time.sleep(CHECK_INTERVAL)
                continue

            # Time info
            bin_key = get_bin_key(timestamp)
            seconds_into_period = get_seconds_into_period(timestamp)
            seconds_remaining = get_seconds_remaining(timestamp)

            # New period handling
            if current_bin != bin_key:
                # Handle expiration
                if position:
                    next_period_strike = get_strike_price()
                    if next_period_strike is None:
                        next_period_strike = btc_price

                    position_strike = position.get('strike_price')
                    final_btc = next_period_strike

                    if position['type'] == 'CALL':
                        final_value = 1.0 if final_btc > position_strike else 0.0
                        result = "ITM" if final_btc > position_strike else "OTM"
                    else:
                        final_value = 1.0 if final_btc <= position_strike else 0.0
                        result = "ITM" if final_btc <= position_strike else "OTM"

                    pnl = final_value - position['entry_price']
                    volatility = calculate_volatility(price_history_1s)
                    choppiness_exit = calculate_choppiness_index(price_history_1s, period=900)

                    trade_record = {
                        'entry_time': position['entry_time'],
                        'exit_time': datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
                        'type': position['type'],
                        'entry_price': position['entry_price'],
                        'exit_price': final_value,
                        'pnl': pnl,
                        'strike_price': position_strike,
                        'final_btc': final_btc,
                        'btc_at_entry': position.get('btc_at_entry'),
                        'exit_reason': 'EXPIRATION',
                        'result': result,
                        'max_profit': position.get('max_profit', 0.0),
                        'max_loss': position.get('max_loss', 0.0),
                        'time_to_expiry_at_entry': position.get('time_to_expiry_at_entry', 0),
                        'time_to_expiry_at_exit': 0,
                        'volatility_at_entry': position.get('volatility', 0.0),
                        'volatility_at_exit': volatility,
                        'choppiness_at_entry': position.get('choppiness', 50.0),
                        'choppiness_at_exit': choppiness_exit,
                        'rsi_period_at_entry': position.get('rsi_period', 60),
                        'rsi_period_at_exit': rsi_period
                    }
                    save_trade(trades, trade_record)
                    print(f"\n[EXPIRATION] {position['type']} {result} | Strike: ${position_strike:.2f} | Final: ${final_btc:.2f} | PNL: ${pnl:+.3f}\n")
                    position = None

                # Reset
                last_d3_type_opened = None
                current_bin = bin_key
                strike_price = get_strike_price()
                if strike_price:
                    print(f"\n{'='*80}")
                    print(f"[NEW PERIOD] {bin_key}")
                    print(f"[STRIKE] ${strike_price:.2f}")
                    print(f"{'='*80}\n")

            # D3: Zero tolerance - trade at exact EMA crossover
            tolerance = 0.0

            # Calculate bands (in D3, bands = EMA exactly)
            if ema is not None:
                upper_band = ema  # No tolerance
                lower_band = ema  # No tolerance
            else:
                upper_band = lower_band = None

            # Display status every 10 seconds
            if int(current_time) % 10 == 0:
                pos_str = f"{position['type']}@${position['entry_price']:.2f}" if position else "NONE"
                last_str = last_d3_type_opened if last_d3_type_opened else "None"
                rsi_str = f"{rsi:.1f}" if rsi is not None else "N/A"
                ema_str = f"{ema:.2f}" if ema is not None else "N/A"

                if ema is not None and upper_band is not None:
                    btc_vs_ema = "ABOVE" if btc_price > ema else "BELOW" if btc_price < ema else "AT"
                    print(f"[D3 {datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime('%H:%M:%S')}] "
                          f"BTC:${btc_price:.2f} {btc_vs_ema} EMA:${ema:.2f} | "
                          f"Vol:${volatility:.2f} Chop:{choppiness:.1f} RSI{rsi_period}s={rsi_str} | "
                          f"Pos:{pos_str} Last:{last_str} | {seconds_remaining}s")
                else:
                    samples_have = len(price_history_1s)
                    samples_need = rsi_period + 1
                    pct = (samples_have / samples_need * 100) if samples_need > 0 else 0
                    print(f"[D3 {datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime('%H:%M:%S')}] "
                          f"BTC:${btc_price:.2f} | Building data... {samples_have}/{samples_need} ({pct:.0f}%) "
                          f"RSI{rsi_period}s={rsi_str} | {seconds_remaining}s")

            # Skip trading until ready
            if rsi is None or ema is None or seconds_into_period < START_DELAY or strike_price is None:
                time.sleep(CHECK_INTERVAL)
                continue

            # Track max profit/loss
            if position:
                if position['type'] == 'CALL' and call_bid is not None:
                    current_pnl = call_bid - position['entry_price']
                    position['max_profit'] = max(position['max_profit'], current_pnl)
                    position['max_loss'] = min(position['max_loss'], current_pnl)
                elif position['type'] == 'PUT' and put_bid is not None:
                    current_pnl = put_bid - position['entry_price']
                    position['max_profit'] = max(position['max_profit'], current_pnl)
                    position['max_loss'] = min(position['max_loss'], current_pnl)

            # ========== TRADING LOGIC ==========
            signal = None

            # Priority 1: RSI Exits (D3: only if option price moved > $0.04)
            if position and position['type'] == 'CALL' and rsi > RSI_EXIT_CALL:
                if call_bid is not None:
                    price_movement = abs(call_bid - position['entry_price'])
                    if price_movement > MIN_PRICE_MOVEMENT:
                        signal = 'CLOSE_CALL'
                        print(f"\n[SIGNAL] RSI EXIT | RSI={rsi:.1f} > {RSI_EXIT_CALL} | Price moved ${price_movement:.3f}")
                    else:
                        print(f"[D3] RSI exit signal but price movement too small: ${price_movement:.3f} <= ${MIN_PRICE_MOVEMENT}")

            elif position and position['type'] == 'PUT' and rsi < RSI_EXIT_PUT:
                if put_bid is not None:
                    price_movement = abs(put_bid - position['entry_price'])
                    if price_movement > MIN_PRICE_MOVEMENT:
                        signal = 'CLOSE_PUT'
                        print(f"\n[SIGNAL] RSI EXIT | RSI={rsi:.1f} < {RSI_EXIT_PUT} | Price moved ${price_movement:.3f}")
                    else:
                        print(f"[D3] RSI exit signal but price movement too small: ${price_movement:.3f} <= ${MIN_PRICE_MOVEMENT}")

            # Priority 2: Flip wrong position (D3: only if option price moved > $0.04)
            elif position and position['type'] == 'CALL' and btc_price < lower_band and rsi > RSI_EXIT_PUT:
                if call_bid is not None:
                    price_movement = abs(call_bid - position['entry_price'])
                    if price_movement > MIN_PRICE_MOVEMENT:
                        signal = 'FLIP_CALL_TO_PUT'
                        print(f"\n[SIGNAL] FLIP TO PUT | Below EMA | Price moved ${price_movement:.3f}")
                    else:
                        print(f"[D3] Flip signal but price movement too small: ${price_movement:.3f} <= ${MIN_PRICE_MOVEMENT}")

            elif position and position['type'] == 'PUT' and btc_price > upper_band and rsi < RSI_EXIT_CALL:
                if put_bid is not None:
                    price_movement = abs(put_bid - position['entry_price'])
                    if price_movement > MIN_PRICE_MOVEMENT:
                        signal = 'FLIP_PUT_TO_CALL'
                        print(f"\n[SIGNAL] FLIP TO CALL | Above EMA | Price moved ${price_movement:.3f}")
                    else:
                        print(f"[D3] Flip signal but price movement too small: ${price_movement:.3f} <= ${MIN_PRICE_MOVEMENT}")

            # Priority 3: Open new position (D3: exactly at EMA cross)
            elif not position and btc_price > upper_band and rsi < RSI_EXIT_CALL:
                signal = 'OPEN_CALL'
            elif not position and btc_price < lower_band and rsi > RSI_EXIT_PUT:
                signal = 'OPEN_PUT'

            # Execute signals with confirmation
            if signal:
                if last_signal_type != signal:
                    last_signal_time = current_time
                    last_signal_type = signal
                elif current_time - last_signal_time >= CONFIRMATION_SECONDS:

                    if signal == 'OPEN_CALL':
                        if MIN_BUY_PRICE <= call_ask <= MAX_BUY_PRICE:
                            position = {
                                'type': 'CALL',
                                'entry_price': call_ask,
                                'entry_time': datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
                                'strike_price': strike_price,
                                'btc_at_entry': btc_price,
                                'rsi_at_entry': rsi,
                                'ema_at_entry': ema,
                                'tolerance': tolerance,
                                'volatility': volatility,
                                'choppiness': choppiness,
                                'rsi_period': rsi_period,
                                'max_profit': 0.0,
                                'max_loss': 0.0,
                                'time_to_expiry_at_entry': seconds_remaining
                            }
                            last_d3_type_opened = 'CALL'
                            print(f"\n✅ OPENED CALL @${call_ask:.2f} | Chop:{choppiness:.1f} | TTL: {seconds_remaining}s\n")
                        last_signal_type = None

                    elif signal == 'OPEN_PUT':
                        if MIN_BUY_PRICE <= put_ask <= MAX_BUY_PRICE:
                            position = {
                                'type': 'PUT',
                                'entry_price': put_ask,
                                'entry_time': datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
                                'strike_price': strike_price,
                                'btc_at_entry': btc_price,
                                'rsi_at_entry': rsi,
                                'ema_at_entry': ema,
                                'tolerance': tolerance,
                                'volatility': volatility,
                                'choppiness': choppiness,
                                'rsi_period': rsi_period,
                                'max_profit': 0.0,
                                'max_loss': 0.0,
                                'time_to_expiry_at_entry': seconds_remaining
                            }
                            last_d3_type_opened = 'PUT'
                            print(f"\n✅ OPENED PUT @${put_ask:.2f} | Chop:{choppiness:.1f} | TTL: {seconds_remaining}s\n")
                        last_signal_type = None

                    elif signal == 'CLOSE_CALL':
                        pnl = call_bid - position['entry_price']
                        choppiness_exit = calculate_choppiness_index(price_history_1s, period=900)
                        trade_record = {
                            'entry_time': position['entry_time'],
                            'exit_time': datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
                            'type': 'CALL',
                            'entry_price': position['entry_price'],
                            'exit_price': call_bid,
                            'pnl': pnl,
                            'strike_price': position.get('strike_price'),
                            'btc_at_entry': position.get('btc_at_entry'),
                            'exit_reason': 'RSI_EXIT',
                            'max_profit': position.get('max_profit', 0.0),
                            'max_loss': position.get('max_loss', 0.0),
                            'time_to_expiry_at_entry': position.get('time_to_expiry_at_entry', 0),
                            'time_to_expiry_at_exit': seconds_remaining,
                            'volatility_at_entry': position.get('volatility', 0.0),
                            'volatility_at_exit': volatility,
                            'choppiness_at_entry': position.get('choppiness', 50.0),
                            'choppiness_at_exit': choppiness_exit,
                            'rsi_period_at_entry': position.get('rsi_period', 60),
                            'rsi_period_at_exit': rsi_period
                        }
                        save_trade(trades, trade_record)
                        print(f"\n✅ CLOSED CALL @${call_bid:.2f} | Chop:{choppiness_exit:.1f} | RSI{rsi_period}s | PNL: ${pnl:+.3f}\n")
                        position = None
                        last_signal_type = None
                        time.sleep(1)

                    elif signal == 'CLOSE_PUT':
                        pnl = put_bid - position['entry_price']
                        choppiness_exit = calculate_choppiness_index(price_history_1s, period=900)
                        trade_record = {
                            'entry_time': position['entry_time'],
                            'exit_time': datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
                            'type': 'PUT',
                            'entry_price': position['entry_price'],
                            'exit_price': put_bid,
                            'pnl': pnl,
                            'strike_price': position.get('strike_price'),
                            'btc_at_entry': position.get('btc_at_entry'),
                            'exit_reason': 'RSI_EXIT',
                            'max_profit': position.get('max_profit', 0.0),
                            'max_loss': position.get('max_loss', 0.0),
                            'time_to_expiry_at_entry': position.get('time_to_expiry_at_entry', 0),
                            'time_to_expiry_at_exit': seconds_remaining,
                            'volatility_at_entry': position.get('volatility', 0.0),
                            'volatility_at_exit': volatility,
                            'choppiness_at_entry': position.get('choppiness', 50.0),
                            'choppiness_at_exit': choppiness_exit,
                            'rsi_period_at_entry': position.get('rsi_period', 60),
                            'rsi_period_at_exit': rsi_period
                        }
                        save_trade(trades, trade_record)
                        print(f"\n✅ CLOSED PUT @${put_bid:.2f} | Chop:{choppiness_exit:.1f} | RSI{rsi_period}s | PNL: ${pnl:+.3f}\n")
                        position = None
                        last_signal_type = None
                        time.sleep(1)

                    elif signal == 'FLIP_CALL_TO_PUT':
                        # Close CALL
                        pnl = call_bid - position['entry_price']
                        choppiness_exit = calculate_choppiness_index(price_history_1s, period=900)
                        trade_record = {
                            'entry_time': position['entry_time'],
                            'exit_time': datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
                            'type': 'CALL',
                            'entry_price': position['entry_price'],
                            'exit_price': call_bid,
                            'pnl': pnl,
                            'strike_price': position.get('strike_price'),
                            'btc_at_entry': position.get('btc_at_entry'),
                            'exit_reason': 'FLIP_TO_PUT',
                            'max_profit': position.get('max_profit', 0.0),
                            'max_loss': position.get('max_loss', 0.0),
                            'time_to_expiry_at_entry': position.get('time_to_expiry_at_entry', 0),
                            'time_to_expiry_at_exit': seconds_remaining,
                            'volatility_at_entry': position.get('volatility', 0.0),
                            'volatility_at_exit': volatility,
                            'choppiness_at_entry': position.get('choppiness', 50.0),
                            'choppiness_at_exit': choppiness_exit,
                            'rsi_period_at_entry': position.get('rsi_period', 60),
                            'rsi_period_at_exit': rsi_period
                        }
                        save_trade(trades, trade_record)
                        print(f"\n🔄 Closed CALL @${call_bid:.2f} | Chop:{choppiness_exit:.1f} | RSI{rsi_period}s | PNL: ${pnl:+.3f}")
                        position = None
                        time.sleep(1)

                        # Open PUT
                        if MIN_BUY_PRICE <= put_ask <= MAX_BUY_PRICE:
                            position = {
                                'type': 'PUT',
                                'entry_price': put_ask,
                                'entry_time': datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
                                'strike_price': strike_price,
                                'btc_at_entry': btc_price,
                                'rsi_at_entry': rsi,
                                'ema_at_entry': ema,
                                'tolerance': tolerance,
                                'volatility': volatility,
                                'choppiness': choppiness,
                                'rsi_period': rsi_period,
                                'max_profit': 0.0,
                                'max_loss': 0.0,
                                'time_to_expiry_at_entry': seconds_remaining
                            }
                            last_d3_type_opened = 'PUT'
                            print(f"✅ Opened PUT @${put_ask:.2f} | Chop:{choppiness:.1f}\n")
                        last_signal_type = None

                    elif signal == 'FLIP_PUT_TO_CALL':
                        # Close PUT
                        pnl = put_bid - position['entry_price']
                        choppiness_exit = calculate_choppiness_index(price_history_1s, period=900)
                        trade_record = {
                            'entry_time': position['entry_time'],
                            'exit_time': datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
                            'type': 'PUT',
                            'entry_price': position['entry_price'],
                            'exit_price': put_bid,
                            'pnl': pnl,
                            'strike_price': position.get('strike_price'),
                            'btc_at_entry': position.get('btc_at_entry'),
                            'exit_reason': 'FLIP_TO_CALL',
                            'max_profit': position.get('max_profit', 0.0),
                            'max_loss': position.get('max_loss', 0.0),
                            'time_to_expiry_at_entry': position.get('time_to_expiry_at_entry', 0),
                            'time_to_expiry_at_exit': seconds_remaining,
                            'volatility_at_entry': position.get('volatility', 0.0),
                            'volatility_at_exit': volatility,
                            'choppiness_at_entry': position.get('choppiness', 50.0),
                            'choppiness_at_exit': choppiness_exit,
                            'rsi_period_at_entry': position.get('rsi_period', 60),
                            'rsi_period_at_exit': rsi_period
                        }
                        save_trade(trades, trade_record)
                        print(f"\n🔄 Closed PUT @${put_bid:.2f} | Chop:{choppiness_exit:.1f} | RSI{rsi_period}s | PNL: ${pnl:+.3f}")
                        position = None
                        time.sleep(1)

                        # Open CALL
                        if MIN_BUY_PRICE <= call_ask <= MAX_BUY_PRICE:
                            position = {
                                'type': 'CALL',
                                'entry_price': call_ask,
                                'entry_time': datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
                                'strike_price': strike_price,
                                'btc_at_entry': btc_price,
                                'rsi_at_entry': rsi,
                                'ema_at_entry': ema,
                                'tolerance': tolerance,
                                'volatility': volatility,
                                'choppiness': choppiness,
                                'rsi_period': rsi_period,
                                'max_profit': 0.0,
                                'max_loss': 0.0,
                                'time_to_expiry_at_entry': seconds_remaining
                            }
                            last_d3_type_opened = 'CALL'
                            print(f"✅ Opened CALL @${call_ask:.2f} | Chop:{choppiness:.1f}\n")
                        last_signal_type = None
            else:
                last_signal_type = None

            # Save state periodically
            if current_time - last_state_save >= STATE_SAVE_INTERVAL:
                state['rsi_period'] = rsi_period
                state['price_history_1s'] = price_history_1s
                state['total_breaches'] = total_breaches
                state['last_adjustment_time'] = last_adjustment_time
                save_state(state)
                last_state_save = current_time

            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            print("\n[SHUTDOWN] Saving state...")
            state['rsi_period'] = rsi_period
            state['price_history_1s'] = price_history_1s
            state['total_breaches'] = total_breaches
            state['last_adjustment_time'] = last_adjustment_time
            save_state(state)
            print("[SHUTDOWN] Complete!")
            break

        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

if __name__ == "__main__":
    main()
