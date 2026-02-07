#!/usr/bin/env python3
"""
RSI Trading Bot - STRATEGIES A, A2, A3, C, C2, BC

STRATEGY A: Adaptive Threshold (Original)
- Starting: 70/30, Adjustment: ±5, Triggers: >5/<3

STRATEGY A2: Faster Adaptive Threshold
- Starting: 70/30, Adjustment: ±2 (smaller steps)
- Triggers: >7/<3 (wider before adjusting)

STRATEGY A3: Price-Enhanced Adaptive Threshold
- Based on A2 parameters
- RSI_UPPER_enhanced = RSI_UPPER + CALL_ask*10
- RSI_LOWER_enhanced = RSI_LOWER - PUT_ask*10

STRATEGY C: Adaptive RSI Period
- Fixed thresholds: 80/20
- Adaptive RSI period: starts at 30s
- 0-2 breaches → decrease 5s, 3-5 no change, 6+ increase 5s
- Min period: 15s, no max limit

STRATEGY C2: Adaptive RSI Period with TP/SL
- Fixed thresholds: 70/30 (tighter than C)
- Adaptive RSI period: starts at 30s
- 0-3 breaches → decrease 5s, 4-7 no change, 8+ increase 5s
- Take Profit: +$0.035, Stop Loss: -$0.10
- Min period: 15s, no max limit

STRATEGY BC: RSI Slope with Adaptive Period + TP/SL
- Uses Strategy C's adaptive RSI period for BOTH:
  * RSI calculation period
  * Slope calculation period (same period)
- Starts at 30s, adapts with C (15s - unlimited)
- Positive slope → HOLD CALL
- Negative slope → HOLD PUT
- NO position flipping - holds until TP/SL
- Take Profit: +$0.035, Stop Loss: -$0.10

STATE PERSISTENCE:
- All thresholds, RSI periods saved to bot_state.json
- Restored on bot restart (hourly RAM management)

All strategies:
- Binary options expire at 00, 15, 30, 45 minutes
- Price limits: $0.03 - $0.97
- Time window: 20s - 880s into period
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict
from collections import deque
import numpy as np

# File paths
PUT_FILE = "/home/ubuntu/013_2025_polymarket/15M_PUT.json"
CALL_FILE = "/home/ubuntu/013_2025_polymarket/15M_CALL.json"
BTC_FILE = "/home/ubuntu/013_2025_polymarket/bybit_btc_price.json"
TRADES_DIR = "/home/ubuntu/013_2025_polymarket/bot017_RSI/bot017_RSI_trades"
STATE_FILE = "/home/ubuntu/013_2025_polymarket/bot017_RSI/bot_state.json"  # Persistent state

# Trading parameters
RSI_PERIOD = 300  # 30 seconds at 0.1s intervals
CONFIRMATION_SECONDS = 2
CHECK_INTERVAL = 0.1
MAX_BUY_PRICE = 0.97
MIN_BUY_PRICE = 0.03
MIN_SECONDS_REMAINING = 20
MAX_SECONDS_INTO_PERIOD = 20

# Strategy A parameters
RSI_UPPER_INITIAL_A = 70
RSI_LOWER_INITIAL_A = 30
THRESHOLD_ADJUSTMENT_A = 5
THRESHOLD_HIGH_TRIGGER_A = 5
THRESHOLD_LOW_TRIGGER_A = 3
MAX_RSI_UPPER = 95
MIN_RSI_LOWER = 5

# Strategy A2 parameters (faster adaptation)
RSI_UPPER_INITIAL_A2 = 70
RSI_LOWER_INITIAL_A2 = 30
THRESHOLD_ADJUSTMENT_A2 = 2  # Smaller adjustments
THRESHOLD_HIGH_TRIGGER_A2 = 7  # Higher threshold to trigger
THRESHOLD_LOW_TRIGGER_A2 = 3

# Strategy A3 uses A2 parameters + price enhancement
PRICE_MULTIPLIER = 10  # Multiply option price by this

# Strategy C parameters (adaptive RSI period)
RSI_UPPER_C = 80  # Fixed upper threshold
RSI_LOWER_C = 20  # Fixed lower threshold
RSI_PERIOD_INITIAL_C = 30  # Starting period in seconds
RSI_PERIOD_ADJUSTMENT_C = 5  # Seconds to adjust by
RSI_PERIOD_HIGH_TRIGGER_C = 5  # If total breaches > this, increase period
RSI_PERIOD_LOW_TRIGGER_C = 3  # If total breaches < this, decrease period
RSI_PERIOD_MIN_C = 15  # Minimum period in seconds
# No maximum period limit

# Strategy C2 parameters (adaptive RSI period with TP/SL)
RSI_UPPER_C2 = 70  # Fixed upper threshold (tighter than C)
RSI_LOWER_C2 = 30  # Fixed lower threshold (tighter than C)
RSI_PERIOD_INITIAL_C2 = 30  # Starting period in seconds
RSI_PERIOD_ADJUSTMENT_C2 = 5  # Seconds to adjust by
RSI_PERIOD_HIGH_TRIGGER_C2 = 8  # If total breaches > this, increase period
RSI_PERIOD_LOW_TRIGGER_C2 = 4  # If total breaches < this, decrease period
RSI_PERIOD_MIN_C2 = 15  # Minimum period in seconds
TAKE_PROFIT_C2 = 0.035  # Take profit threshold
STOP_LOSS_C2 = 0.10  # Stop loss threshold

# Strategy BC parameters (TP/SL for slope strategy)
TAKE_PROFIT_BC = 0.035  # Take profit threshold
STOP_LOSS_BC = 0.10  # Stop loss threshold


def read_json(filepath: str) -> Optional[dict]:
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return None


def write_json(filepath: str, data: dict):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def get_bin_key(distance: float, seconds_to_expiry: float, volatility: float) -> str:
    distance_bins = [(0, 1, "0-1"), (1, 5, "1-5"), (5, 10, "5-10"), (10, 20, "10-20"),
                     (20, 40, "20-40"), (40, 80, "40-80"), (80, 160, "80-160"),
                     (160, 320, "160-320"), (320, 640, "320-640"), (640, 1280, "640-1280"),
                     (1280, float('inf'), "1280+")]
    time_bins = [(13*60, 15*60, "15m-13m"), (11*60, 13*60, "13m-11m"), (10*60, 11*60, "11m-10m"),
                 (9*60, 10*60, "10m-9m"), (8*60, 9*60, "9m-8m"), (7*60, 8*60, "8m-7m"),
                 (6*60, 7*60, "7m-6m"), (5*60, 6*60, "6m-5m"), (4*60, 5*60, "5m-4m"),
                 (3*60, 4*60, "4m-3m"), (2*60, 3*60, "3m-2m"), (90, 120, "120s-90s"),
                 (60, 90, "90s-60s"), (40, 60, "60s-40s"), (30, 40, "40s-30s"),
                 (20, 30, "30s-20s"), (10, 20, "20s-10s"), (5, 10, "10s-5s"),
                 (2, 5, "5s-2s"), (0, 2, "last-2s")]
    vol_bins = [(0, 10, "0-10"), (10, 20, "10-20"), (20, 30, "20-30"), (30, 40, "30-40"),
                (40, 60, "40-60"), (60, 90, "60-90"), (90, 120, "90-120"), (120, 240, "120-240"),
                (240, float('inf'), "240+")]
    
    def get_bin_label(value, bins):
        for min_val, max_val, label in bins:
            if min_val <= value < max_val:
                return label
        return bins[-1][2]
    
    return f"{get_bin_label(distance, distance_bins)}|{get_bin_label(seconds_to_expiry, time_bins)}|{get_bin_label(volatility, vol_bins)}"


def get_strike_price() -> Optional[float]:
    try:
        import requests
        now = datetime.now(timezone.utc)
        for start_min in [0, 15, 30, 45]:
            if now.minute >= start_min and now.minute < start_min + 15:
                period_start = now.replace(minute=start_min, second=0, microsecond=0)
                url = "https://api.bybit.com/v5/market/mark-price-kline"
                params = {'category': 'linear', 'symbol': 'BTCUSDT', 'interval': '15',
                         'start': int(period_start.timestamp() * 1000), 'limit': 1}
                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('retCode') == 0:
                        kline_list = data.get('result', {}).get('list', [])
                        if kline_list:
                            return float(kline_list[0][1])
        return None
    except:
        return None


def get_seconds_to_expiry() -> float:
    now = datetime.now()
    for start_min in [0, 15, 30, 45]:
        if now.minute >= start_min and now.minute < start_min + 15:
            return 900 - ((now.minute - start_min) * 60 + now.second)
    return 0


def calculate_btc_volatility(price_history: deque) -> float:
    if len(price_history) < 10:
        return 0.0
    prices = list(price_history)
    return max(prices) - min(prices)


def calculate_rsi(prices: deque, period: int = RSI_PERIOD) -> Optional[float]:
    if len(prices) < period + 1:
        return None
    price_array = np.array(list(prices))[-(period + 1):]
    deltas = np.diff(price_array)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain, avg_loss = np.mean(gains), np.mean(losses)
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_rsi_slope(rsi_history: deque, period_seconds: int = 30) -> Optional[float]:
    """
    Calculate slope of RSI over specified period
    Args:
        rsi_history: deque of RSI values (sampled at 0.1s intervals)
        period_seconds: time period in seconds to calculate slope over
    """
    samples_needed = period_seconds * 10  # Convert seconds to samples (0.1s intervals)
    
    # If we don't have enough samples yet, use whatever we have (minimum 50 samples = 5s)
    available_samples = len(rsi_history)
    if available_samples < 50:  # Need at least 5s of data
        return None
    
    # Use requested period if available, otherwise use all available data
    samples_to_use = min(samples_needed, available_samples)
    recent_rsi = list(rsi_history)[-samples_to_use:]
    
    x = np.arange(len(recent_rsi))
    y = np.array(recent_rsi)
    slope = np.polyfit(x, y, 1)[0]  # Linear regression slope
    return slope


def save_bot_state(state: Dict):
    """Save bot state to persistent file"""
    write_json(STATE_FILE, state)


def load_bot_state() -> Optional[Dict]:
    """Load bot state from persistent file"""
    state = read_json(STATE_FILE)
    if state:
        print(f"\n📥 Loaded bot state from previous session:")
        print(f"   A:  Thresholds [{state['a_lower']:.0f}/{state['a_upper']:.0f}]")
        print(f"   A2: Thresholds [{state['a2_lower']:.0f}/{state['a2_upper']:.0f}]")
        print(f"   A3: Thresholds [{state['a3_lower']:.0f}/{state['a3_upper']:.0f}]")
        print(f"   C:  RSI Period {state['c_period']}s")
        print(f"   C2: RSI Period {state.get('c2_period', RSI_PERIOD_INITIAL_C2)}s")
        print(f"   BC: RSI Period {state['c_period']}s (shared with C)")
    return state


def adjust_thresholds(breach_history: deque, current_upper: float, current_lower: float, 
                      adjustment: int, high_trigger: int, low_trigger: int) -> tuple:
    """Adjust RSI thresholds based on breach frequency"""
    if len(breach_history) == 0:
        return current_upper, current_lower, 0, 0
    
    # Handle both tuple format (breach_type, timestamp) and string format
    upper_breaches = 0
    lower_breaches = 0
    for item in breach_history:
        if isinstance(item, tuple):
            breach_type, _ = item
        else:
            breach_type = item
        
        if breach_type == 'UPPER':
            upper_breaches += 1
        elif breach_type == 'LOWER':
            lower_breaches += 1
    
    new_upper = current_upper
    new_lower = current_lower
    
    # Adjust upper threshold
    if upper_breaches > high_trigger:
        new_upper = min(MAX_RSI_UPPER, current_upper + adjustment)
    elif upper_breaches < low_trigger:
        new_lower_bound = current_lower + 10
        new_upper = max(new_lower_bound, current_upper - adjustment)
    
    # Adjust lower threshold
    if lower_breaches > high_trigger:
        new_lower = max(MIN_RSI_LOWER, current_lower - adjustment)
    elif lower_breaches < low_trigger:
        new_upper_bound = current_upper - 10
        new_lower = min(new_upper_bound, current_lower + adjustment)
    
    return new_upper, new_lower, upper_breaches, lower_breaches


def adjust_rsi_period(breach_history: deque, current_period: int, adjustment: int, 
                      high_trigger: int, low_trigger: int, min_period: int) -> tuple:
    """
    Adjust RSI period based on total breach frequency
    More breaches → Increase period (slower, smoother RSI)
    Fewer breaches → Decrease period (faster, more responsive RSI)
    """
    # Count total breaches (upper + lower)
    total_breaches = 0
    for item in breach_history:
        if isinstance(item, tuple):
            breach_type, _ = item
        else:
            breach_type = item
        
        if breach_type in ['UPPER', 'LOWER']:
            total_breaches += 1
    
    new_period = current_period
    
    # Adjust period based on total breaches
    if total_breaches > high_trigger:
        # Too many breaches → increase period (slower RSI, smoother)
        new_period = current_period + adjustment
    elif total_breaches < low_trigger:
        # Too few breaches (including 0) → decrease period (faster RSI, more sensitive)
        new_period = max(min_period, current_period - adjustment)
    
    return new_period, total_breaches


def load_trades(strategy: str) -> Dict:
    today = datetime.now().strftime("%Y-%m-%d")
    if strategy == "A":
        filename = f"trades_{today}.json"
    else:
        filename = f"strategy{strategy}_{today}.json"
    filepath = f"{TRADES_DIR}/{filename}"
    if Path(filepath).exists():
        data = read_json(filepath)
        if data:
            print(f"📂 Strategy {strategy} - Loaded: {len(data.get('trades', []))} trades, PNL: ${data.get('total_pnl', 0):.2f}")
            return data
    return {"strategy": f"{strategy}_Adaptive", "date": today, "total_pnl": 0.0, "trades": [],
            "stats": {"total_trades": 0, "winning_trades": 0, "losing_trades": 0, "call_trades": 0, "put_trades": 0}}


def save_trades(trades_data: Dict, strategy: str):
    today = datetime.now().strftime("%Y-%m-%d")
    if strategy == "A":
        filename = f"trades_{today}.json"
    else:
        filename = f"strategy{strategy}_{today}.json"
    filepath = f"{TRADES_DIR}/{filename}"
    write_json(filepath, trades_data)


def check_tp_sl(position, current_bid: float, take_profit: float, stop_loss: float) -> Optional[str]:
    """
    Check if position hits take profit or stop loss
    Returns: 'TP', 'SL', or None
    """
    if not position:
        return None
    
    current_pnl = current_bid - position['entry_price']
    
    if current_pnl >= take_profit:
        return 'TP'
    elif current_pnl <= -stop_loss:
        return 'SL'
    return None


def close_position_expiration(position, strike_price, final_btc, trades_data, strategy, timestamp):
    if position['type'] == 'CALL':
        exit_price = 1.0 if final_btc > strike_price else 0.0
    else:
        exit_price = 1.0 if final_btc < strike_price else 0.0
    
    pnl = exit_price - position['entry_price']
    trades_data['total_pnl'] += pnl
    
    trade_record = {**position, 'exit_price': exit_price, 'exit_time': timestamp,
                   'exit_btc': final_btc, 'pnl': pnl, 'exit_reason': 'EXPIRATION',
                   'expired_itm': exit_price == 1.0}
    trades_data['trades'].append(trade_record)
    trades_data['stats']['total_trades'] += 1
    if pnl > 0:
        trades_data['stats']['winning_trades'] += 1
    else:
        trades_data['stats']['losing_trades'] += 1
    
    save_trades(trades_data, strategy)
    result_emoji = "✅" if exit_price == 1.0 else "❌"
    print(f"[{strategy}] 💰 {result_emoji} Expired ${exit_price:.2f} | PNL:${pnl:+.3f} | Total:${trades_data['total_pnl']:+.2f}")
    return None


def close_position_signal(position, exit_price, timestamp, trades_data, strategy, exit_reason='RSI_SIGNAL'):
    pnl = exit_price - position['entry_price']
    trades_data['total_pnl'] += pnl
    
    trade_record = {**position, 'exit_price': exit_price, 'exit_time': timestamp,
                   'pnl': pnl, 'exit_reason': exit_reason}
    trades_data['trades'].append(trade_record)
    trades_data['stats']['total_trades'] += 1
    if pnl > 0:
        trades_data['stats']['winning_trades'] += 1
    else:
        trades_data['stats']['losing_trades'] += 1
    
    save_trades(trades_data, strategy)
    reason_emoji = "💰" if exit_reason == 'TP' else "🛑" if exit_reason == 'SL' else "🔴"
    print(f"[{strategy}] {reason_emoji} SOLD {position['type']} @${exit_price:.2f} | PNL:${pnl:+.3f} | {exit_reason}")
    return None


def open_position(opt_type, ask_price, timestamp, strike_price, btc_price, rsi, bin_key, 
                 seconds_remaining, trades_data, strategy, rsi_upper=None, rsi_lower=None, 
                 rsi_upper_enhanced=None, rsi_lower_enhanced=None, rsi_period=None, btc_volatility=None, rsi_slope=None):
    position = {'type': opt_type, 'entry_price': ask_price, 'entry_time': timestamp,
               'entry_strike': strike_price, 'entry_btc': btc_price, 'entry_rsi': rsi,
               'entry_bin': bin_key, 'entry_seconds_remaining': seconds_remaining}
    if rsi_upper is not None and rsi_lower is not None:
        position['entry_rsi_upper'] = rsi_upper
        position['entry_rsi_lower'] = rsi_lower
    if rsi_upper_enhanced is not None and rsi_lower_enhanced is not None:
        position['entry_rsi_upper_enhanced'] = rsi_upper_enhanced
        position['entry_rsi_lower_enhanced'] = rsi_lower_enhanced
    if rsi_period is not None:
        position['entry_rsi_period'] = rsi_period
    if btc_volatility is not None:
        position['entry_btc_volatility'] = btc_volatility
    if rsi_slope is not None:
        position['entry_rsi_slope'] = rsi_slope
    
    trades_data['stats'][f"{opt_type.lower()}_trades"] += 1
    save_trades(trades_data, strategy)
    
    thresh_str = f" [{rsi_lower:.0f}/{rsi_upper:.0f}]" if rsi_upper is not None else ""
    enh_str = f" E[{rsi_lower_enhanced:.0f}/{rsi_upper_enhanced:.0f}]" if rsi_upper_enhanced is not None else ""
    period_str = f" RSI{rsi_period}s" if rsi_period is not None else ""
    vol_str = f" Vol:{btc_volatility:.1f}" if btc_volatility is not None else ""
    slope_str = f" Slope:{rsi_slope:+.4f}" if rsi_slope is not None else ""
    print(f"[{strategy}] 🟢 BOUGHT {opt_type} @${ask_price:.2f} | RSI:{rsi:.1f}{thresh_str}{enh_str}{period_str}{vol_str}{slope_str}")
    return position


def main():
    print("\n" + "="*150)
    print("RSI TRADING BOT - STRATEGIES A, A2, A3, C, C2, BC")
    print(f"A: Original Adaptive (±{THRESHOLD_ADJUSTMENT_A}, >{THRESHOLD_HIGH_TRIGGER_A}/<{THRESHOLD_LOW_TRIGGER_A})")
    print(f"A2: Faster Adaptive (±{THRESHOLD_ADJUSTMENT_A2}, >{THRESHOLD_HIGH_TRIGGER_A2}/<{THRESHOLD_LOW_TRIGGER_A2})")
    print(f"A3: Price-Enhanced (A2 params + price×{PRICE_MULTIPLIER})")
    print(f"C: Adaptive RSI Period ({RSI_LOWER_C}/{RSI_UPPER_C}, 0-2/3-5/6+)")
    print(f"C2: Adaptive RSI Period + TP/SL ({RSI_LOWER_C2}/{RSI_UPPER_C2}, 0-3/4-7/8+, TP:${TAKE_PROFIT_C2} SL:${STOP_LOSS_C2})")
    print(f"BC: RSI Slope with C's period + TP/SL (TP:${TAKE_PROFIT_BC} SL:${STOP_LOSS_BC})")
    print(f"Limits: ${MIN_BUY_PRICE:.2f}-${MAX_BUY_PRICE:.2f} | Window: {MAX_SECONDS_INTO_PERIOD}s-{900-MIN_SECONDS_REMAINING}s")
    print("="*150)
    
    # Load persistent state
    saved_state = load_bot_state()
    
    trades_a = load_trades("A")
    trades_a2 = load_trades("A2")
    trades_a3 = load_trades("A3")
    trades_c = load_trades("C")
    trades_c2 = load_trades("C2")
    trades_bc = load_trades("BC")
    
    position_a, position_a2, position_a3, position_c, position_c2, position_bc = None, None, None, None, None, None
    btc_price_history = deque(maxlen=RSI_PERIOD + 1)
    btc_vol_history = deque(maxlen=600)
    
    # Strategy A state - Load from saved state or use defaults
    rsi_upper_a = saved_state['a_upper'] if saved_state else RSI_UPPER_INITIAL_A
    rsi_lower_a = saved_state['a_lower'] if saved_state else RSI_LOWER_INITIAL_A
    breach_history_a = deque(maxlen=1000)
    currently_breaching_upper_a = False
    currently_breaching_lower_a = False
    last_signal_time_a, last_signal_type_a = 0, None
    
    # Strategy A2 state
    rsi_upper_a2 = saved_state['a2_upper'] if saved_state else RSI_UPPER_INITIAL_A2
    rsi_lower_a2 = saved_state['a2_lower'] if saved_state else RSI_LOWER_INITIAL_A2
    breach_history_a2 = deque(maxlen=1000)
    currently_breaching_upper_a2 = False
    currently_breaching_lower_a2 = False
    last_signal_time_a2, last_signal_type_a2 = 0, None
    
    # Strategy A3 state (same breach tracking as A2)
    rsi_upper_a3 = saved_state['a3_upper'] if saved_state else RSI_UPPER_INITIAL_A2
    rsi_lower_a3 = saved_state['a3_lower'] if saved_state else RSI_LOWER_INITIAL_A2
    breach_history_a3 = deque(maxlen=1000)
    currently_breaching_upper_a3 = False
    currently_breaching_lower_a3 = False
    last_signal_time_a3, last_signal_type_a3 = 0, None
    
    # Strategy C state (adaptive RSI period)
    rsi_period_c = saved_state['c_period'] if saved_state else RSI_PERIOD_INITIAL_C
    rsi_period_samples_c = rsi_period_c * 10  # Convert seconds to samples (0.1s intervals)
    breach_history_c = deque(maxlen=1000)
    currently_breaching_upper_c = False
    currently_breaching_lower_c = False
    last_signal_time_c, last_signal_type_c = 0, None
    btc_price_history_c = deque(maxlen=2000)  # Large enough for any RSI period
    
    # Strategy C2 state (adaptive RSI period with TP/SL)
    rsi_period_c2 = saved_state.get('c2_period', RSI_PERIOD_INITIAL_C2) if saved_state else RSI_PERIOD_INITIAL_C2
    rsi_period_samples_c2 = rsi_period_c2 * 10
    breach_history_c2 = deque(maxlen=1000)
    currently_breaching_upper_c2 = False
    currently_breaching_lower_c2 = False
    last_signal_time_c2, last_signal_type_c2 = 0, None
    btc_price_history_c2 = deque(maxlen=2000)
    
    # Strategy BC state (uses C's RSI period + slope logic + TP/SL)
    rsi_history_bc = deque(maxlen=6000)  # Large enough for any adaptive period (10 min of data)
    prev_slope_sign_bc = None
    last_signal_time_bc, last_signal_type_bc = 0, None
    
    last_threshold_adjustment_time = time.time()
    threshold_check_interval = 60
    
    strike_price, last_strike_update_minute = None, None
    waiting_for_prices = False
    
    try:
        while True:
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            is_period_start = now.minute in [0, 15, 30, 45]
            
            # Handle period end expirations
            if is_period_start and now.second == 0:
                btc_data = read_json(BTC_FILE)
                final_btc = btc_data.get('price', 0) if btc_data else 0
                
                if position_a and strike_price:
                    position_a = close_position_expiration(position_a, strike_price, final_btc, trades_a, "A", timestamp)
                if position_a2 and strike_price:
                    position_a2 = close_position_expiration(position_a2, strike_price, final_btc, trades_a2, "A2", timestamp)
                if position_a3 and strike_price:
                    position_a3 = close_position_expiration(position_a3, strike_price, final_btc, trades_a3, "A3", timestamp)
                if position_c and strike_price:
                    position_c = close_position_expiration(position_c, strike_price, final_btc, trades_c, "C", timestamp)
                if position_c2 and strike_price:
                    position_c2 = close_position_expiration(position_c2, strike_price, final_btc, trades_c2, "C2", timestamp)
                if position_bc and strike_price:
                    position_bc = close_position_expiration(position_bc, strike_price, final_btc, trades_bc, "BC", timestamp)
                
                if position_a or position_a2 or position_a3 or position_c or position_c2 or position_bc:
                    time.sleep(2)
            
            # Update strike price
            if is_period_start and now.second >= 5 and last_strike_update_minute != now.minute:
                new_strike = get_strike_price()
                if new_strike:
                    strike_price = new_strike
                    last_strike_update_minute = now.minute
                    print(f"\n🔄 New strike: ${strike_price:.2f}")
            
            if strike_price is None:
                strike_price = get_strike_price()
                if strike_price:
                    print(f"Initial Strike: ${strike_price:.2f}\n")
            
            # Read market data
            btc_data = read_json(BTC_FILE)
            call_data = read_json(CALL_FILE)
            put_data = read_json(PUT_FILE)
            
            if not all([btc_data, call_data, put_data]):
                time.sleep(CHECK_INTERVAL)
                continue
            
            btc_price = btc_data.get('price', 0)
            btc_price_history.append(btc_price)
            btc_vol_history.append(btc_price)
            
            call_bid_data = call_data.get('best_bid')
            call_ask_data = call_data.get('best_ask')
            put_bid_data = put_data.get('best_bid')
            put_ask_data = put_data.get('best_ask')
            
            call_bid = call_bid_data.get('price', 0) if call_bid_data else 0
            call_ask = call_ask_data.get('price', 0) if call_ask_data else 0
            put_bid = put_bid_data.get('price', 0) if put_bid_data else 0
            put_ask = put_ask_data.get('price', 0) if put_ask_data else 0
            
            has_valid_prices = call_bid > 0 and call_ask > 0 and put_bid > 0 and put_ask > 0
            if not has_valid_prices:
                if not waiting_for_prices:
                    print(f"\n⚠️  Waiting for valid prices...")
                    waiting_for_prices = True
                time.sleep(CHECK_INTERVAL)
                continue
            else:
                if waiting_for_prices:
                    waiting_for_prices = False
            
            # Calculate RSI
            rsi = calculate_rsi(btc_price_history)
            if rsi is None:
                time.sleep(CHECK_INTERVAL)
                continue
            
            # Calculate RSI for Strategy C (with adaptive period)
            btc_price_history_c.append(btc_price)
            rsi_c = calculate_rsi(btc_price_history_c, period=rsi_period_samples_c)
            
            # Calculate RSI for Strategy C2 (with adaptive period)
            btc_price_history_c2.append(btc_price)
            rsi_c2 = calculate_rsi(btc_price_history_c2, period=rsi_period_samples_c2)
            
            # Calculate RSI slope for BC (using C's RSI values and C's period)
            if rsi_c is not None:
                rsi_history_bc.append(rsi_c)
            rsi_slope_bc = calculate_rsi_slope(rsi_history_bc, period_seconds=rsi_period_c)
            
            # Track breaches for all strategies
            # Strategy A
            if rsi > rsi_upper_a:
                if not currently_breaching_upper_a:
                    breach_history_a.append(('UPPER', time.time()))
                    currently_breaching_upper_a = True
                    currently_breaching_lower_a = False
            elif rsi < rsi_lower_a:
                if not currently_breaching_lower_a:
                    breach_history_a.append(('LOWER', time.time()))
                    currently_breaching_lower_a = True
                    currently_breaching_upper_a = False
            else:
                currently_breaching_upper_a = False
                currently_breaching_lower_a = False
            
            # Strategy A2
            if rsi > rsi_upper_a2:
                if not currently_breaching_upper_a2:
                    breach_history_a2.append(('UPPER', time.time()))
                    currently_breaching_upper_a2 = True
                    currently_breaching_lower_a2 = False
            elif rsi < rsi_lower_a2:
                if not currently_breaching_lower_a2:
                    breach_history_a2.append(('LOWER', time.time()))
                    currently_breaching_lower_a2 = True
                    currently_breaching_upper_a2 = False
            else:
                currently_breaching_upper_a2 = False
                currently_breaching_lower_a2 = False
            
            # Strategy A3 (uses same base thresholds as A2)
            if rsi > rsi_upper_a3:
                if not currently_breaching_upper_a3:
                    breach_history_a3.append(('UPPER', time.time()))
                    currently_breaching_upper_a3 = True
                    currently_breaching_lower_a3 = False
            elif rsi < rsi_lower_a3:
                if not currently_breaching_lower_a3:
                    breach_history_a3.append(('LOWER', time.time()))
                    currently_breaching_lower_a3 = True
                    currently_breaching_upper_a3 = False
            else:
                currently_breaching_upper_a3 = False
                currently_breaching_lower_a3 = False
            
            # Strategy C (fixed thresholds, adaptive period)
            if rsi_c is not None:
                if rsi_c > RSI_UPPER_C:
                    if not currently_breaching_upper_c:
                        breach_history_c.append(('UPPER', time.time()))
                        currently_breaching_upper_c = True
                        currently_breaching_lower_c = False
                elif rsi_c < RSI_LOWER_C:
                    if not currently_breaching_lower_c:
                        breach_history_c.append(('LOWER', time.time()))
                        currently_breaching_lower_c = True
                        currently_breaching_upper_c = False
                else:
                    currently_breaching_upper_c = False
                    currently_breaching_lower_c = False
            
            # Strategy C2 (fixed thresholds, adaptive period)
            if rsi_c2 is not None:
                if rsi_c2 > RSI_UPPER_C2:
                    if not currently_breaching_upper_c2:
                        breach_history_c2.append(('UPPER', time.time()))
                        currently_breaching_upper_c2 = True
                        currently_breaching_lower_c2 = False
                elif rsi_c2 < RSI_LOWER_C2:
                    if not currently_breaching_lower_c2:
                        breach_history_c2.append(('LOWER', time.time()))
                        currently_breaching_lower_c2 = True
                        currently_breaching_upper_c2 = False
                else:
                    currently_breaching_upper_c2 = False
                    currently_breaching_lower_c2 = False
            
            # Adjust thresholds every 60 seconds
            current_time_threshold = time.time()
            if current_time_threshold - last_threshold_adjustment_time >= threshold_check_interval:
                print(f"\n⏱️  Running threshold/period checks at {datetime.now().strftime('%H:%M:%S')}...")
                
                # Strategy A
                recent_breaches_a = [b for b in breach_history_a if current_time_threshold - b[1] <= 900]
                recent_breach_types_a = deque([b[0] for b in recent_breaches_a])
                old_upper_a, old_lower_a = rsi_upper_a, rsi_lower_a
                rsi_upper_a, rsi_lower_a, upper_count_a, lower_count_a = adjust_thresholds(
                    recent_breach_types_a, rsi_upper_a, rsi_lower_a, 
                    THRESHOLD_ADJUSTMENT_A, THRESHOLD_HIGH_TRIGGER_A, THRESHOLD_LOW_TRIGGER_A)
                if rsi_upper_a != old_upper_a or rsi_lower_a != old_lower_a:
                    print(f"\n📊 [A] Thresholds: {old_lower_a:.0f}/{old_upper_a:.0f} → {rsi_lower_a:.0f}/{rsi_upper_a:.0f} (Breaches: U={upper_count_a}, L={lower_count_a})")
                
                # Strategy A2
                recent_breaches_a2 = [b for b in breach_history_a2 if current_time_threshold - b[1] <= 900]
                recent_breach_types_a2 = deque([b[0] for b in recent_breaches_a2])
                old_upper_a2, old_lower_a2 = rsi_upper_a2, rsi_lower_a2
                rsi_upper_a2, rsi_lower_a2, upper_count_a2, lower_count_a2 = adjust_thresholds(
                    recent_breach_types_a2, rsi_upper_a2, rsi_lower_a2,
                    THRESHOLD_ADJUSTMENT_A2, THRESHOLD_HIGH_TRIGGER_A2, THRESHOLD_LOW_TRIGGER_A2)
                if rsi_upper_a2 != old_upper_a2 or rsi_lower_a2 != old_lower_a2:
                    print(f"\n📊 [A2] Thresholds: {old_lower_a2:.0f}/{old_upper_a2:.0f} → {rsi_lower_a2:.0f}/{rsi_upper_a2:.0f} (Breaches: U={upper_count_a2}, L={lower_count_a2})")
                
                # Strategy A3 (same as A2)
                recent_breaches_a3 = [b for b in breach_history_a3 if current_time_threshold - b[1] <= 900]
                recent_breach_types_a3 = deque([b[0] for b in recent_breaches_a3])
                old_upper_a3, old_lower_a3 = rsi_upper_a3, rsi_lower_a3
                rsi_upper_a3, rsi_lower_a3, upper_count_a3, lower_count_a3 = adjust_thresholds(
                    recent_breach_types_a3, rsi_upper_a3, rsi_lower_a3,
                    THRESHOLD_ADJUSTMENT_A2, THRESHOLD_HIGH_TRIGGER_A2, THRESHOLD_LOW_TRIGGER_A2)
                if rsi_upper_a3 != old_upper_a3 or rsi_lower_a3 != old_lower_a3:
                    print(f"\n📊 [A3] Thresholds: {old_lower_a3:.0f}/{old_upper_a3:.0f} → {rsi_lower_a3:.0f}/{rsi_upper_a3:.0f} (Breaches: U={upper_count_a3}, L={lower_count_a3})")
                
                # Strategy C - Adjust RSI period
                recent_breaches_c = [b for b in breach_history_c if current_time_threshold - b[1] <= 900]
                recent_breach_types_c = deque([b[0] for b in recent_breaches_c])
                old_period_c = rsi_period_c
                rsi_period_c, total_breaches_c = adjust_rsi_period(
                    recent_breach_types_c, rsi_period_c, RSI_PERIOD_ADJUSTMENT_C,
                    RSI_PERIOD_HIGH_TRIGGER_C, RSI_PERIOD_LOW_TRIGGER_C, RSI_PERIOD_MIN_C)
                rsi_period_samples_c = rsi_period_c * 10  # Convert to samples
                
                # Always show C adjustment check (even if no change)
                if rsi_period_c != old_period_c:
                    print(f"\n📊 [C] RSI Period: {old_period_c}s → {rsi_period_c}s (Total Breaches: {total_breaches_c})")
                else:
                    # Debug: show why no change
                    if total_breaches_c > RSI_PERIOD_HIGH_TRIGGER_C:
                        print(f"\n📊 [C] Period check: {total_breaches_c} breaches (>{RSI_PERIOD_HIGH_TRIGGER_C}) → would increase but already at {rsi_period_c}s")
                    elif total_breaches_c < RSI_PERIOD_LOW_TRIGGER_C:
                        print(f"\n📊 [C] Period check: {total_breaches_c} breaches (<{RSI_PERIOD_LOW_TRIGGER_C}) → would decrease but already at {rsi_period_c}s")
                    else:
                        print(f"\n📊 [C] Period check: {total_breaches_c} breaches (staying at {rsi_period_c}s)")

                # Strategy C2 - Adjust RSI period
                recent_breaches_c2 = [b for b in breach_history_c2 if current_time_threshold - b[1] <= 900]
                recent_breach_types_c2 = deque([b[0] for b in recent_breaches_c2])
                old_period_c2 = rsi_period_c2
                rsi_period_c2, total_breaches_c2 = adjust_rsi_period(
                    recent_breach_types_c2, rsi_period_c2, RSI_PERIOD_ADJUSTMENT_C2,
                    RSI_PERIOD_HIGH_TRIGGER_C2, RSI_PERIOD_LOW_TRIGGER_C2, RSI_PERIOD_MIN_C2)
                rsi_period_samples_c2 = rsi_period_c2 * 10
                
                if rsi_period_c2 != old_period_c2:
                    print(f"\n📊 [C2] RSI Period: {old_period_c2}s → {rsi_period_c2}s (Total Breaches: {total_breaches_c2})")
                
                last_threshold_adjustment_time = current_time_threshold
                
                # Save bot state for persistence
                bot_state = {
                    'a_upper': rsi_upper_a,
                    'a_lower': rsi_lower_a,
                    'a2_upper': rsi_upper_a2,
                    'a2_lower': rsi_lower_a2,
                    'a3_upper': rsi_upper_a3,
                    'a3_lower': rsi_lower_a3,
                    'c_period': rsi_period_c,
                    'c2_period': rsi_period_c2,
                    'timestamp': datetime.now().isoformat()
                }
                save_bot_state(bot_state)
            
            # Calculate A3 enhanced thresholds
            rsi_upper_a3_enhanced = rsi_upper_a3 + (call_ask * PRICE_MULTIPLIER)
            rsi_lower_a3_enhanced = rsi_lower_a3 - (put_ask * PRICE_MULTIPLIER)
            
            # Market metrics
            seconds_remaining = get_seconds_to_expiry()
            seconds_into_period = 900 - seconds_remaining
            distance = abs(btc_price - strike_price) if strike_price else 0
            volatility = calculate_btc_volatility(btc_vol_history)
            bin_key = get_bin_key(distance, seconds_remaining, volatility)
            in_time_window = (seconds_into_period >= MAX_SECONDS_INTO_PERIOD and 
                            seconds_remaining >= MIN_SECONDS_REMAINING)
            
            # Display
            pos_a_str = f"{position_a['type']}@${position_a['entry_price']:.2f}" if position_a else "NONE"
            pos_a2_str = f"{position_a2['type']}@${position_a2['entry_price']:.2f}" if position_a2 else "NONE"
            pos_a3_str = f"{position_a3['type']}@${position_a3['entry_price']:.2f}" if position_a3 else "NONE"
            pos_c_str = f"{position_c['type']}@${position_c['entry_price']:.2f}" if position_c else "NONE"
            pos_c2_str = f"{position_c2['type']}@${position_c2['entry_price']:.2f}" if position_c2 else "NONE"
            pos_bc_str = f"{position_bc['type']}@${position_bc['entry_price']:.2f}" if position_bc else "NONE"
            rsi_c_str = f"{rsi_c:.1f}" if rsi_c is not None else "N/A"
            rsi_c2_str = f"{rsi_c2:.1f}" if rsi_c2 is not None else "N/A"
            slope_bc_str = f"{rsi_slope_bc:+.4f}" if rsi_slope_bc is not None else "N/A"
            bc_samples = len(rsi_history_bc)
            bc_period_str = f"RSI{rsi_period_c}s({bc_samples})" if bc_samples > 0 else f"RSI{rsi_period_c}s"
            
            # Split display across 2 lines for readability
            print(f"[{timestamp}] BTC:${btc_price:.2f} | A:[{rsi_lower_a:.0f}/{rsi_upper_a:.0f}] {pos_a_str:<12} ${trades_a['total_pnl']:+.2f} | A2:[{rsi_lower_a2:.0f}/{rsi_upper_a2:.0f}] {pos_a2_str:<12} ${trades_a2['total_pnl']:+.2f} | A3:E[{rsi_lower_a3_enhanced:.0f}/{rsi_upper_a3_enhanced:.0f}] {pos_a3_str:<12} ${trades_a3['total_pnl']:+.2f}")
            print(f"         C:RSI{rsi_period_c}s={rsi_c_str} {pos_c_str:<12} ${trades_c['total_pnl']:+.2f} | C2:RSI{rsi_period_c2}s={rsi_c2_str} {pos_c2_str:<12} ${trades_c2['total_pnl']:+.2f} | BC:{bc_period_str} S{slope_bc_str} {pos_bc_str:<12} ${trades_bc['total_pnl']:+.2f}", end='\r')
            
            if not in_time_window:
                time.sleep(CHECK_INTERVAL)
                continue
            
            current_time = time.time()
            
            # ===== CHECK TP/SL FOR C2 AND BC =====
            # Check TP/SL for C2
            if position_c2:
                current_bid_c2 = put_bid if position_c2['type'] == 'PUT' else call_bid
                tp_sl_result = check_tp_sl(position_c2, current_bid_c2, TAKE_PROFIT_C2, STOP_LOSS_C2)
                if tp_sl_result:
                    position_c2 = close_position_signal(position_c2, current_bid_c2, timestamp, trades_c2, "C2", tp_sl_result)
                    time.sleep(2)
            
            # Check TP/SL for BC
            if position_bc:
                current_bid_bc = put_bid if position_bc['type'] == 'PUT' else call_bid
                tp_sl_result = check_tp_sl(position_bc, current_bid_bc, TAKE_PROFIT_BC, STOP_LOSS_BC)
                if tp_sl_result:
                    position_bc = close_position_signal(position_bc, current_bid_bc, timestamp, trades_bc, "BC", tp_sl_result)
                    time.sleep(2)
            
            # ===== STRATEGY A =====
            signal_a = None
            if rsi < rsi_lower_a:
                signal_a = 'BUY'
            elif rsi > rsi_upper_a:
                signal_a = 'SELL'
            
            if signal_a:
                if last_signal_type_a != signal_a:
                    last_signal_time_a = current_time
                    last_signal_type_a = signal_a
                elif current_time - last_signal_time_a >= CONFIRMATION_SECONDS:
                    if signal_a == 'BUY' and (not position_a or position_a['type'] == 'PUT'):
                        if position_a:
                            position_a = close_position_signal(position_a, put_bid, timestamp, trades_a, "A")
                            time.sleep(2)
                        if MIN_BUY_PRICE <= call_ask <= MAX_BUY_PRICE:
                            position_a = open_position('CALL', call_ask, timestamp, strike_price, btc_price, rsi, 
                                                      bin_key, seconds_remaining, trades_a, "A", rsi_upper_a, rsi_lower_a)
                        last_signal_type_a = None
                    elif signal_a == 'SELL' and (not position_a or position_a['type'] == 'CALL'):
                        if position_a:
                            position_a = close_position_signal(position_a, call_bid, timestamp, trades_a, "A")
                            time.sleep(2)
                        if MIN_BUY_PRICE <= put_ask <= MAX_BUY_PRICE:
                            position_a = open_position('PUT', put_ask, timestamp, strike_price, btc_price, rsi,
                                                      bin_key, seconds_remaining, trades_a, "A", rsi_upper_a, rsi_lower_a)
                        last_signal_type_a = None
            else:
                last_signal_type_a = None
            
            # ===== STRATEGY A2 =====
            signal_a2 = None
            if rsi < rsi_lower_a2:
                signal_a2 = 'BUY'
            elif rsi > rsi_upper_a2:
                signal_a2 = 'SELL'
            
            if signal_a2:
                if last_signal_type_a2 != signal_a2:
                    last_signal_time_a2 = current_time
                    last_signal_type_a2 = signal_a2
                elif current_time - last_signal_time_a2 >= CONFIRMATION_SECONDS:
                    if signal_a2 == 'BUY' and (not position_a2 or position_a2['type'] == 'PUT'):
                        if position_a2:
                            position_a2 = close_position_signal(position_a2, put_bid, timestamp, trades_a2, "A2")
                            time.sleep(2)
                        if MIN_BUY_PRICE <= call_ask <= MAX_BUY_PRICE:
                            position_a2 = open_position('CALL', call_ask, timestamp, strike_price, btc_price, rsi,
                                                       bin_key, seconds_remaining, trades_a2, "A2", rsi_upper_a2, rsi_lower_a2)
                        last_signal_type_a2 = None
                    elif signal_a2 == 'SELL' and (not position_a2 or position_a2['type'] == 'CALL'):
                        if position_a2:
                            position_a2 = close_position_signal(position_a2, call_bid, timestamp, trades_a2, "A2")
                            time.sleep(2)
                        if MIN_BUY_PRICE <= put_ask <= MAX_BUY_PRICE:
                            position_a2 = open_position('PUT', put_ask, timestamp, strike_price, btc_price, rsi,
                                                       bin_key, seconds_remaining, trades_a2, "A2", rsi_upper_a2, rsi_lower_a2)
                        last_signal_type_a2 = None
            else:
                last_signal_type_a2 = None
            
            # ===== STRATEGY A3 (Price-Enhanced) =====
            signal_a3 = None
            if rsi < rsi_lower_a3_enhanced:
                signal_a3 = 'BUY'
            elif rsi > rsi_upper_a3_enhanced:
                signal_a3 = 'SELL'
            
            if signal_a3:
                if last_signal_type_a3 != signal_a3:
                    last_signal_time_a3 = current_time
                    last_signal_type_a3 = signal_a3
                elif current_time - last_signal_time_a3 >= CONFIRMATION_SECONDS:
                    if signal_a3 == 'BUY' and (not position_a3 or position_a3['type'] == 'PUT'):
                        if position_a3:
                            position_a3 = close_position_signal(position_a3, put_bid, timestamp, trades_a3, "A3")
                            time.sleep(2)
                        if MIN_BUY_PRICE <= call_ask <= MAX_BUY_PRICE:
                            position_a3 = open_position('CALL', call_ask, timestamp, strike_price, btc_price, rsi,
                                                       bin_key, seconds_remaining, trades_a3, "A3", 
                                                       rsi_upper_a3, rsi_lower_a3, 
                                                       rsi_upper_a3_enhanced, rsi_lower_a3_enhanced)
                        last_signal_type_a3 = None
                    elif signal_a3 == 'SELL' and (not position_a3 or position_a3['type'] == 'CALL'):
                        if position_a3:
                            position_a3 = close_position_signal(position_a3, call_bid, timestamp, trades_a3, "A3")
                            time.sleep(2)
                        if MIN_BUY_PRICE <= put_ask <= MAX_BUY_PRICE:
                            position_a3 = open_position('PUT', put_ask, timestamp, strike_price, btc_price, rsi,
                                                       bin_key, seconds_remaining, trades_a3, "A3",
                                                       rsi_upper_a3, rsi_lower_a3,
                                                       rsi_upper_a3_enhanced, rsi_lower_a3_enhanced)
                        last_signal_type_a3 = None
            else:
                last_signal_type_a3 = None
            
            # ===== STRATEGY C (Adaptive RSI Period) =====
            if rsi_c is not None:
                signal_c = None
                if rsi_c < RSI_LOWER_C:
                    signal_c = 'BUY'
                elif rsi_c > RSI_UPPER_C:
                    signal_c = 'SELL'
                
                if signal_c:
                    if last_signal_type_c != signal_c:
                        last_signal_time_c = current_time
                        last_signal_type_c = signal_c
                    elif current_time - last_signal_time_c >= CONFIRMATION_SECONDS:
                        if signal_c == 'BUY' and (not position_c or position_c['type'] == 'PUT'):
                            if position_c:
                                position_c = close_position_signal(position_c, put_bid, timestamp, trades_c, "C")
                                time.sleep(2)
                            if MIN_BUY_PRICE <= call_ask <= MAX_BUY_PRICE:
                                position_c = open_position('CALL', call_ask, timestamp, strike_price, btc_price, rsi_c,
                                                          bin_key, seconds_remaining, trades_c, "C",
                                                          rsi_period=rsi_period_c, btc_volatility=volatility)
                            last_signal_type_c = None
                        elif signal_c == 'SELL' and (not position_c or position_c['type'] == 'CALL'):
                            if position_c:
                                position_c = close_position_signal(position_c, call_bid, timestamp, trades_c, "C")
                                time.sleep(2)
                            if MIN_BUY_PRICE <= put_ask <= MAX_BUY_PRICE:
                                position_c = open_position('PUT', put_ask, timestamp, strike_price, btc_price, rsi_c,
                                                          bin_key, seconds_remaining, trades_c, "C",
                                                          rsi_period=rsi_period_c, btc_volatility=volatility)
                            last_signal_type_c = None
                else:
                    last_signal_type_c = None
            
            # ===== STRATEGY C2 (Adaptive RSI Period with TP/SL) =====
            if rsi_c2 is not None:
                signal_c2 = None
                if rsi_c2 < RSI_LOWER_C2:
                    signal_c2 = 'BUY'
                elif rsi_c2 > RSI_UPPER_C2:
                    signal_c2 = 'SELL'
                
                if signal_c2:
                    if last_signal_type_c2 != signal_c2:
                        last_signal_time_c2 = current_time
                        last_signal_type_c2 = signal_c2
                    elif current_time - last_signal_time_c2 >= CONFIRMATION_SECONDS:
                        if signal_c2 == 'BUY' and (not position_c2 or position_c2['type'] == 'PUT'):
                            if position_c2:
                                position_c2 = close_position_signal(position_c2, put_bid, timestamp, trades_c2, "C2")
                                time.sleep(2)
                            if MIN_BUY_PRICE <= call_ask <= MAX_BUY_PRICE:
                                position_c2 = open_position('CALL', call_ask, timestamp, strike_price, btc_price, rsi_c2,
                                                            bin_key, seconds_remaining, trades_c2, "C2",
                                                            rsi_period=rsi_period_c2, btc_volatility=volatility)
                            last_signal_type_c2 = None
                        elif signal_c2 == 'SELL' and (not position_c2 or position_c2['type'] == 'CALL'):
                            if position_c2:
                                position_c2 = close_position_signal(position_c2, call_bid, timestamp, trades_c2, "C2")
                                time.sleep(2)
                            if MIN_BUY_PRICE <= put_ask <= MAX_BUY_PRICE:
                                position_c2 = open_position('PUT', put_ask, timestamp, strike_price, btc_price, rsi_c2,
                                                            bin_key, seconds_remaining, trades_c2, "C2",
                                                            rsi_period=rsi_period_c2, btc_volatility=volatility)
                            last_signal_type_c2 = None
                else:
                    last_signal_type_c2 = None
            
            # ===== STRATEGY BC (RSI Slope with TP/SL - NO FLIPPING) =====
            if rsi_slope_bc is not None and rsi_c is not None:
                current_slope_sign = 'POS' if rsi_slope_bc > 0 else 'NEG'
                
                # Only open position if we don't have one - NO FLIPPING
                if not position_bc:
                    if current_slope_sign == 'POS' and MIN_BUY_PRICE <= call_ask <= MAX_BUY_PRICE:
                        # Positive slope → Open CALL
                        position_bc = open_position('CALL', call_ask, timestamp, strike_price, btc_price, rsi_c,
                                                   bin_key, seconds_remaining, trades_bc, "BC",
                                                   rsi_period=rsi_period_c, btc_volatility=volatility, rsi_slope=rsi_slope_bc)
                    
                    elif current_slope_sign == 'NEG' and MIN_BUY_PRICE <= put_ask <= MAX_BUY_PRICE:
                        # Negative slope → Open PUT
                        position_bc = open_position('PUT', put_ask, timestamp, strike_price, btc_price, rsi_c,
                                                   bin_key, seconds_remaining, trades_bc, "BC",
                                                   rsi_period=rsi_period_c, btc_volatility=volatility, rsi_slope=rsi_slope_bc)
                
                # Hold position until TP/SL/Expiration (handled by TP/SL checks above)
                prev_slope_sign_bc = current_slope_sign
            
            time.sleep(CHECK_INTERVAL)
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping...")
        
        # Close any open positions
        for pos, data, strat, cb, pb in [(position_a, trades_a, "A", call_bid, put_bid),
                                          (position_a2, trades_a2, "A2", call_bid, put_bid),
                                          (position_a3, trades_a3, "A3", call_bid, put_bid),
                                          (position_c, trades_c, "C", call_bid, put_bid),
                                          (position_c2, trades_c2, "C2", call_bid, put_bid),
                                          (position_bc, trades_bc, "BC", call_bid, put_bid)]:
            if pos:
                exit_price = cb if pos['type'] == 'CALL' else pb
                if exit_price > 0:
                    pos = close_position_signal(pos, exit_price, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), data, strat)
        
        # Print summaries
        for strat, data in [("A", trades_a), ("A2", trades_a2), ("A3", trades_a3), ("C", trades_c), ("C2", trades_c2), ("BC", trades_bc)]:
            print(f"\n{'='*80}\nStrategy {strat} Summary\n{'='*80}")
            print(f"Total PNL: ${data['total_pnl']:+.2f}")
            print(f"Trades: {data['stats']['total_trades']} | Wins: {data['stats']['winning_trades']} | Losses: {data['stats']['losing_trades']}")
            if data['stats']['total_trades'] > 0:
                print(f"Win Rate: {data['stats']['winning_trades']/data['stats']['total_trades']*100:.1f}%")
            print("="*80)


if __name__ == "__main__":
    main()
