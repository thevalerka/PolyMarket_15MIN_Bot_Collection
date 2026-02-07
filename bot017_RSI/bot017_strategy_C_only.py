#!/usr/bin/env python3
"""
RSI Trading Bot - STRATEGY C ONLY

STRATEGY C: Adaptive RSI Period
- Fixed thresholds: 80/20
- Adaptive RSI period: starts at 30s, min 15s, max 90s
- Breach-based adaptation:
  * 0-2 breaches → decrease 5s (calm market, speed up)
  * 3-5 breaches → no change (optimal)
  * 6+ breaches  → increase 5s (volatile, smooth out)
- No TP/SL (holds until expiration)

Binary options expire at 00, 15, 30, 45 minutes
Price limits: $0.03 - $0.97
Time window: 20s - 880s into period
"""

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from collections import deque
import numpy as np


# File paths
BTC_FILE = "/home/ubuntu/013_2025_polymarket/BTC.json"
CALL_FILE = "/home/ubuntu/013_2025_polymarket/15M_CALL.json"
PUT_FILE = "/home/ubuntu/013_2025_polymarket/15M_PUT.json"
TRADES_DIR = "/home/ubuntu/013_2025_polymarket/bot017_RSI"
STATE_FILE = "/home/ubuntu/013_2025_polymarket/bot017_RSI/bot_state_c_only.json"

Path(TRADES_DIR).mkdir(parents=True, exist_ok=True)

# RSI calculation
RSI_PERIOD = 140  # 14 seconds at 10Hz
CHECK_INTERVAL = 0.1

# Strategy C parameters (adaptive RSI period)
RSI_UPPER_C = 80  # Fixed upper threshold
RSI_LOWER_C = 20  # Fixed lower threshold
RSI_PERIOD_INITIAL_C = 30  # Starting period in seconds
RSI_PERIOD_ADJUSTMENT_C = 5  # Seconds to adjust by
RSI_PERIOD_HIGH_TRIGGER_C = 5  # If total breaches > this, increase period
RSI_PERIOD_LOW_TRIGGER_C = 3  # If total breaches < this, decrease period
RSI_PERIOD_MIN_C = 15  # Minimum period in seconds
RSI_PERIOD_MAX_C = 90  # Maximum period in seconds

# Trading limits
MIN_BUY_PRICE = 0.03
MAX_BUY_PRICE = 0.97
MAX_SECONDS_INTO_PERIOD = 20
MIN_SECONDS_REMAINING = 20
CONFIRMATION_SECONDS = 3


def read_json(filepath: str) -> Optional[Dict]:
    """Read JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return None


def save_json(data: Dict, filepath: str):
    """Save JSON file"""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving {filepath}: {e}")


def get_seconds_to_expiry() -> int:
    """Calculate seconds until next 15-minute mark"""
    now = datetime.now()
    minutes_into_quarter = now.minute % 15
    seconds_into_quarter = minutes_into_quarter * 60 + now.second
    return 900 - seconds_into_quarter


def calculate_rsi(prices: deque, period: int = RSI_PERIOD) -> Optional[float]:
    """Calculate RSI"""
    if len(prices) < period + 1:
        return None
    
    prices_array = np.array(list(prices))
    deltas = np.diff(prices_array)
    
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_btc_volatility(price_history: deque, window: int = 600) -> float:
    """Calculate BTC price volatility (standard deviation)"""
    if len(price_history) < 10:
        return 0.0
    prices = list(price_history)[-window:]
    return float(np.std(prices))


def get_bin_key(distance: float, seconds_remaining: int, volatility: float) -> str:
    """Generate bin key for statistics"""
    dist_bin = min(int(distance / 20) * 20, 160)
    time_bin = min(int(seconds_remaining / 60), 14)
    vol_bin = min(int(volatility / 10) * 10, 120)
    return f"{dist_bin}_{time_bin}_{vol_bin}"


def adjust_rsi_period(breach_history: deque, current_period: int, adjustment: int, 
                      high_trigger: int, low_trigger: int, min_period: int, max_period: int = None) -> tuple:
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
        if max_period is not None:
            new_period = min(max_period, new_period)
    elif total_breaches < low_trigger:
        # Too few breaches (including 0) → decrease period (faster RSI, more sensitive)
        new_period = max(min_period, current_period - adjustment)
    
    return new_period, total_breaches


def load_trades() -> Dict:
    """Load Strategy C trades"""
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"strategyC_{today}.json"
    filepath = f"{TRADES_DIR}/{filename}"
    if Path(filepath).exists():
        data = read_json(filepath)
        if data:
            print(f"📂 Strategy C - Loaded: {len(data.get('trades', []))} trades, PNL: ${data.get('total_pnl', 0):.2f}")
            return data
    return {"strategy": "C_Adaptive_Period", "date": today, "total_pnl": 0.0, "trades": [],
            "stats": {"total_trades": 0, "winning_trades": 0, "losing_trades": 0, "call_trades": 0, "put_trades": 0}}


def save_trades(trades_data: Dict):
    """Save Strategy C trades"""
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"strategyC_{today}.json"
    filepath = f"{TRADES_DIR}/{filename}"
    save_json(trades_data, filepath)


def load_bot_state() -> Optional[Dict]:
    """Load bot state from persistent file"""
    state = read_json(STATE_FILE)
    if state:
        print(f"\n📥 Loaded bot state from previous session:")
        print(f"   C: RSI Period {state['c_period']}s")
    return state


def save_bot_state(state: Dict):
    """Save bot state to persistent file"""
    save_json(state, STATE_FILE)


def open_position(position_type: str, entry_price: float, timestamp: str, strike_price: float,
                 btc_price: float, rsi: float, bin_key: str, seconds_remaining: int,
                 trades_data: Dict, rsi_period: int = None, btc_volatility: float = None):
    """Open a new position"""
    position = {
        'type': position_type,
        'entry_price': entry_price,
        'entry_time': timestamp,
        'strike_price': strike_price,
        'btc_at_entry': btc_price,
        'rsi_at_entry': rsi,
        'bin': bin_key,
        'seconds_remaining': seconds_remaining
    }
    
    if rsi_period is not None:
        position['rsi_period'] = rsi_period
    if btc_volatility is not None:
        position['btc_volatility'] = btc_volatility
    
    print(f"\n[C] 🟢 BUY {position_type} @${entry_price:.2f} | RSI={rsi:.1f} | {seconds_remaining}s left | Period={rsi_period}s")
    return position


def close_position_signal(position, exit_price: float, timestamp: str, trades_data: Dict, exit_reason: str = "RSI_SIGNAL"):
    """Close position on signal"""
    pnl = exit_price - position['entry_price']
    trades_data['total_pnl'] += pnl
    
    won = pnl > 0
    trades_data['stats']['total_trades'] += 1
    if won:
        trades_data['stats']['winning_trades'] += 1
    else:
        trades_data['stats']['losing_trades'] += 1
    
    if position['type'] == 'CALL':
        trades_data['stats']['call_trades'] += 1
    else:
        trades_data['stats']['put_trades'] += 1
    
    trade_record = {
        'entry_time': position['entry_time'],
        'exit_time': timestamp,
        'type': position['type'],
        'entry_price': position['entry_price'],
        'exit_price': exit_price,
        'pnl': pnl,
        'strike_price': position.get('strike_price'),
        'btc_at_entry': position.get('btc_at_entry'),
        'rsi_at_entry': position.get('rsi_at_entry'),
        'rsi_period': position.get('rsi_period'),
        'btc_volatility': position.get('btc_volatility'),
        'bin': position.get('bin'),
        'seconds_remaining_at_entry': position.get('seconds_remaining'),
        'exit_reason': exit_reason
    }
    
    trades_data['trades'].append(trade_record)
    save_trades(trades_data)
    
    emoji = "💰" if exit_reason == "TP" else "🛑" if exit_reason == "SL" else "🔴"
    print(f"[C] {emoji} SOLD {position['type']} @${exit_price:.2f} | PNL:{pnl:+.3f} | {exit_reason}")
    return None


def close_position_expiration(position, strike_price: float, final_btc: float, trades_data: Dict, timestamp: str):
    """Close position at expiration"""
    won = (position['type'] == 'CALL' and final_btc >= strike_price) or \
          (position['type'] == 'PUT' and final_btc < strike_price)
    
    exit_price = 0.99 if won else 0.01
    pnl = exit_price - position['entry_price']
    trades_data['total_pnl'] += pnl
    
    trades_data['stats']['total_trades'] += 1
    if won:
        trades_data['stats']['winning_trades'] += 1
    else:
        trades_data['stats']['losing_trades'] += 1
    
    if position['type'] == 'CALL':
        trades_data['stats']['call_trades'] += 1
    else:
        trades_data['stats']['put_trades'] += 1
    
    trade_record = {
        'entry_time': position['entry_time'],
        'exit_time': timestamp,
        'type': position['type'],
        'entry_price': position['entry_price'],
        'exit_price': exit_price,
        'pnl': pnl,
        'strike_price': position.get('strike_price'),
        'btc_at_entry': position.get('btc_at_entry'),
        'final_btc': final_btc,
        'rsi_at_entry': position.get('rsi_at_entry'),
        'rsi_period': position.get('rsi_period'),
        'btc_volatility': position.get('btc_volatility'),
        'bin': position.get('bin'),
        'seconds_remaining_at_entry': position.get('seconds_remaining'),
        'exit_reason': 'EXPIRATION',
        'won': won
    }
    
    trades_data['trades'].append(trade_record)
    save_trades(trades_data)
    
    result = "WON" if won else "LOST"
    print(f"\n[C] ⏰ EXPIRED {position['type']} | {result} | Final BTC: ${final_btc:.2f} vs Strike: ${strike_price:.2f} | PNL:{pnl:+.3f}")
    return None


def main():
    print("\n" + "="*80)
    print("RSI TRADING BOT - STRATEGY C ONLY")
    print(f"Adaptive RSI Period ({RSI_LOWER_C}/{RSI_UPPER_C}, Period: {RSI_PERIOD_INITIAL_C}s-{RSI_PERIOD_MAX_C}s)")
    print(f"Breach triggers: 0-{RSI_PERIOD_LOW_TRIGGER_C-1} decrease, {RSI_PERIOD_LOW_TRIGGER_C}-{RSI_PERIOD_HIGH_TRIGGER_C} hold, {RSI_PERIOD_HIGH_TRIGGER_C+1}+ increase")
    print(f"Limits: ${MIN_BUY_PRICE:.2f}-${MAX_BUY_PRICE:.2f} | Window: {MAX_SECONDS_INTO_PERIOD}s-{900-MIN_SECONDS_REMAINING}s")
    print("="*80)
    
    # Load persistent state
    saved_state = load_bot_state()
    
    trades_c = load_trades()
    position_c = None
    
    btc_price_history = deque(maxlen=RSI_PERIOD + 1)
    btc_vol_history = deque(maxlen=600)
    
    # Strategy C state (adaptive RSI period)
    rsi_period_c = saved_state['c_period'] if saved_state else RSI_PERIOD_INITIAL_C
    rsi_period_samples_c = rsi_period_c * 10  # Convert seconds to samples (0.1s intervals)
    breach_history_c = deque(maxlen=100)  # Max ~100 breaches in 15 min
    currently_breaching_upper_c = False
    currently_breaching_lower_c = False
    last_signal_time_c, last_signal_type_c = 0, None
    btc_price_history_c = deque(maxlen=2000)  # Large enough for any RSI period
    
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
                
                if position_c and strike_price:
                    position_c = close_position_expiration(position_c, strike_price, final_btc, trades_c, timestamp)
                
                if position_c:
                    time.sleep(2)
                
                strike_price = None
                last_strike_update_minute = None
                waiting_for_prices = True
                print(f"\n⏰ Period ended. Waiting for new strike price...")
                time.sleep(2)
                continue
            
            # Update strike price
            current_minute = now.minute
            if last_strike_update_minute != current_minute:
                btc_data = read_json(BTC_FILE)
                if btc_data:
                    strike_price = btc_data.get('price')
                    last_strike_update_minute = current_minute
                    if waiting_for_prices and strike_price:
                        waiting_for_prices = False
                        print(f"✅ Strike price updated: ${strike_price:.2f}")
            
            if not strike_price or waiting_for_prices:
                time.sleep(CHECK_INTERVAL)
                continue
            
            # Read market data
            btc_data = read_json(BTC_FILE)
            call_data = read_json(CALL_FILE)
            put_data = read_json(PUT_FILE)
            
            if not all([btc_data, call_data, put_data]):
                time.sleep(CHECK_INTERVAL)
                continue
            
            btc_price = btc_data.get('price', 0)
            call_bid = call_data.get('best_bid', {}).get('price', 0)
            call_ask = call_data.get('best_ask', {}).get('price', 0)
            put_bid = put_data.get('best_bid', {}).get('price', 0)
            put_ask = put_data.get('best_ask', {}).get('price', 0)
            
            if btc_price == 0 or call_ask == 0 or put_ask == 0:
                time.sleep(CHECK_INTERVAL)
                continue
            
            # Update price histories
            btc_price_history.append(btc_price)
            btc_vol_history.append(btc_price)
            
            # Calculate base RSI
            rsi = calculate_rsi(btc_price_history)
            if rsi is None:
                time.sleep(CHECK_INTERVAL)
                continue
            
            # Calculate RSI for Strategy C (with adaptive period)
            btc_price_history_c.append(btc_price)
            rsi_c = calculate_rsi(btc_price_history_c, period=rsi_period_samples_c)
            
            # Track breaches for Strategy C
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
            
            # Adjust RSI period every 60 seconds
            current_time_threshold = time.time()
            if current_time_threshold - last_threshold_adjustment_time >= threshold_check_interval:
                print(f"\n⏱️  Running period check at {timestamp.split()[1]}...")
                
                # Strategy C - Adjust RSI period
                recent_breaches_c = [b for b in breach_history_c if current_time_threshold - b[1] <= 900]
                recent_breach_types_c = deque([b[0] for b in recent_breaches_c])
                old_period_c = rsi_period_c
                rsi_period_c, total_breaches_c = adjust_rsi_period(
                    recent_breach_types_c, rsi_period_c, RSI_PERIOD_ADJUSTMENT_C,
                    RSI_PERIOD_HIGH_TRIGGER_C, RSI_PERIOD_LOW_TRIGGER_C, RSI_PERIOD_MIN_C, RSI_PERIOD_MAX_C)
                rsi_period_samples_c = rsi_period_c * 10
                
                # Always show C adjustment check
                if rsi_period_c != old_period_c:
                    print(f"📊 [C] RSI Period: {old_period_c}s → {rsi_period_c}s (Total Breaches: {total_breaches_c})")
                else:
                    if total_breaches_c > RSI_PERIOD_HIGH_TRIGGER_C:
                        print(f"📊 [C] Period check: {total_breaches_c} breaches (>{RSI_PERIOD_HIGH_TRIGGER_C}) → at max {rsi_period_c}s")
                    elif total_breaches_c < RSI_PERIOD_LOW_TRIGGER_C:
                        print(f"📊 [C] Period check: {total_breaches_c} breaches (<{RSI_PERIOD_LOW_TRIGGER_C}) → at min {rsi_period_c}s")
                    else:
                        print(f"📊 [C] Period check: {total_breaches_c} breaches (staying at {rsi_period_c}s)")
                
                last_threshold_adjustment_time = current_time_threshold
                
                # Save bot state
                bot_state = {
                    'c_period': rsi_period_c,
                    'timestamp': datetime.now().isoformat()
                }
                save_bot_state(bot_state)
            
            # Market metrics
            seconds_remaining = get_seconds_to_expiry()
            seconds_into_period = 900 - seconds_remaining
            distance = abs(btc_price - strike_price) if strike_price else 0
            volatility = calculate_btc_volatility(btc_vol_history)
            bin_key = get_bin_key(distance, seconds_remaining, volatility)
            in_time_window = (seconds_into_period >= MAX_SECONDS_INTO_PERIOD and 
                            seconds_remaining >= MIN_SECONDS_REMAINING)
            
            # Display
            pos_c_str = f"{position_c['type']}@${position_c['entry_price']:.2f}" if position_c else "NONE"
            rsi_c_str = f"{rsi_c:.1f}" if rsi_c is not None else "N/A"
            
            print(f"[{timestamp}] BTC:${btc_price:.2f} | C:RSI{rsi_period_c}s={rsi_c_str} {pos_c_str:<12} ${trades_c['total_pnl']:+.2f}", end='\r')
            
            if not in_time_window:
                time.sleep(CHECK_INTERVAL)
                continue
            
            current_time = time.time()
            
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
                                position_c = close_position_signal(position_c, put_bid, timestamp, trades_c)
                                time.sleep(2)
                            if MIN_BUY_PRICE <= call_ask <= MAX_BUY_PRICE:
                                position_c = open_position('CALL', call_ask, timestamp, strike_price, btc_price, rsi_c,
                                                          bin_key, seconds_remaining, trades_c,
                                                          rsi_period=rsi_period_c, btc_volatility=volatility)
                            last_signal_type_c = None
                        elif signal_c == 'SELL' and (not position_c or position_c['type'] == 'CALL'):
                            if position_c:
                                position_c = close_position_signal(position_c, call_bid, timestamp, trades_c)
                                time.sleep(2)
                            if MIN_BUY_PRICE <= put_ask <= MAX_BUY_PRICE:
                                position_c = open_position('PUT', put_ask, timestamp, strike_price, btc_price, rsi_c,
                                                          bin_key, seconds_remaining, trades_c,
                                                          rsi_period=rsi_period_c, btc_volatility=volatility)
                            last_signal_type_c = None
                else:
                    last_signal_type_c = None
            
            time.sleep(CHECK_INTERVAL)
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping...")
        
        # Close any open positions
        if position_c:
            exit_price = call_bid if position_c['type'] == 'CALL' else put_bid
            if exit_price > 0:
                position_c = close_position_signal(position_c, exit_price, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), trades_c)
        
        # Print summary
        print(f"\n{'='*80}\nStrategy C Summary\n{'='*80}")
        print(f"Total PNL: ${trades_c['total_pnl']:+.2f}")
        print(f"Trades: {trades_c['stats']['total_trades']} | Wins: {trades_c['stats']['winning_trades']} | Losses: {trades_c['stats']['losing_trades']}")
        if trades_c['stats']['total_trades'] > 0:
            print(f"Win Rate: {trades_c['stats']['winning_trades']/trades_c['stats']['total_trades']*100:.1f}%")
        print("="*80)


if __name__ == "__main__":
    main()
