#!/usr/bin/env python3
"""
RSI Trading Bot for 15M Binary Options - DUAL STRATEGY

STRATEGY A: RSI Threshold Strategy
- Uses RSI(30s) with thresholds: BUY at RSI<15, SELL at RSI>85
- 2 second confirmation period between opposite signals

STRATEGY B: RSI Slope Strategy  
- Calculates slope of RSI over last 30 seconds
- Positive slope → BUY CALL (or hold CALL)
- Negative slope → BUY PUT (or hold PUT)
- Switches positions when slope changes sign

Common Rules:
- One position at a time per strategy
- Binary options expire at 00, 15, 30, 45 minutes
- Price limits: $0.03 - $0.97
- Time window: 20s - 880s into period
- Records trades separately: trades_YYYY-MM-DD.json (A), strategyB_YYYY-MM-DD.json (B)
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

# Trading parameters
RSI_PERIOD = 300  # 30 seconds at 0.1s intervals
RSI_UPPER = 85
RSI_LOWER = 15
CONFIRMATION_SECONDS = 2
CHECK_INTERVAL = 0.1
MAX_BUY_PRICE = 0.97
MIN_BUY_PRICE = 0.03
MIN_SECONDS_REMAINING = 20
MAX_SECONDS_INTO_PERIOD = 20


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
    """Get bin key for trade recording"""
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


def calculate_rsi_slope(rsi_history: deque) -> Optional[float]:
    """Calculate slope of RSI over last 30 seconds"""
    if len(rsi_history) < 50:  # Need at least 5s of data
        return None
    recent_rsi = list(rsi_history)[-300:]  # Last 30s
    x = np.arange(len(recent_rsi))
    y = np.array(recent_rsi)
    slope = np.polyfit(x, y, 1)[0]  # Linear regression slope
    return slope


def load_trades(strategy: str) -> Dict:
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"trades_{today}.json" if strategy == "A" else f"strategyB_{today}.json"
    filepath = f"{TRADES_DIR}/{filename}"
    if Path(filepath).exists():
        data = read_json(filepath)
        if data:
            print(f"📂 Strategy {strategy} - Loaded: {len(data.get('trades', []))} trades, PNL: ${data.get('total_pnl', 0):.2f}")
            return data
    return {"strategy": f"{strategy}_RSI", "date": today, "total_pnl": 0.0, "trades": [],
            "stats": {"total_trades": 0, "winning_trades": 0, "losing_trades": 0, "call_trades": 0, "put_trades": 0}}


def save_trades(trades_data: Dict, strategy: str):
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"trades_{today}.json" if strategy == "A" else f"strategyB_{today}.json"
    filepath = f"{TRADES_DIR}/{filename}"
    write_json(filepath, trades_data)


def close_position_expiration(position, strike_price, final_btc, trades_data, strategy, timestamp):
    """Handle position expiration at period end"""
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
    print(f"[{strategy}] 💰 {result_emoji} Expired ${exit_price:.2f} (BTC:${final_btc:.2f} vs ${strike_price:.2f}) | PNL:${pnl:+.3f} | Total:${trades_data['total_pnl']:+.2f}")
    return None


def close_position_signal(position, exit_price, timestamp, trades_data, strategy, reason="RSI_SIGNAL"):
    """Handle position close due to signal"""
    pnl = exit_price - position['entry_price']
    trades_data['total_pnl'] += pnl
    
    trade_record = {**position, 'exit_price': exit_price, 'exit_time': timestamp,
                   'pnl': pnl, 'exit_reason': reason}
    trades_data['trades'].append(trade_record)
    trades_data['stats']['total_trades'] += 1
    if pnl > 0:
        trades_data['stats']['winning_trades'] += 1
    else:
        trades_data['stats']['losing_trades'] += 1
    
    save_trades(trades_data, strategy)
    print(f"[{strategy}] 🔴 SOLD {position['type']} @${exit_price:.2f} | PNL:${pnl:+.3f}")
    return None


def open_position(opt_type, ask_price, timestamp, strike_price, btc_price, rsi, bin_key, seconds_remaining, trades_data, strategy, rsi_slope=None):
    """Open new position"""
    position = {'type': opt_type, 'entry_price': ask_price, 'entry_time': timestamp,
               'entry_strike': strike_price, 'entry_btc': btc_price, 'entry_rsi': rsi,
               'entry_bin': bin_key, 'entry_seconds_remaining': seconds_remaining}
    if rsi_slope is not None:
        position['entry_rsi_slope'] = rsi_slope
    
    trades_data['stats'][f"{opt_type.lower()}_trades"] += 1
    save_trades(trades_data, strategy)
    slope_str = f" | Slope:{rsi_slope:+.4f}" if rsi_slope is not None else ""
    print(f"[{strategy}] 🟢 BOUGHT {opt_type} @${ask_price:.2f} | RSI:{rsi:.1f}{slope_str}")
    return position


def main():
    print("\n" + "="*150)
    print("RSI TRADING BOT - DUAL STRATEGY")
    print(f"Strategy A: Threshold (RSI<{RSI_LOWER}=BUY, RSI>{RSI_UPPER}=SELL) | Strategy B: Slope (Positive=CALL, Negative=PUT)")
    print(f"Limits: ${MIN_BUY_PRICE:.2f}-${MAX_BUY_PRICE:.2f} | Window: {MAX_SECONDS_INTO_PERIOD}s-{900-MIN_SECONDS_REMAINING}s")
    print("="*150)
    
    trades_a = load_trades("A")
    trades_b = load_trades("B")
    position_a, position_b = None, None
    btc_price_history = deque(maxlen=RSI_PERIOD + 1)
    rsi_history = deque(maxlen=300)
    btc_vol_history = deque(maxlen=600)
    last_signal_time, last_signal_type = 0, None
    prev_slope_sign = None
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
                if position_b and strike_price:
                    position_b = close_position_expiration(position_b, strike_price, final_btc, trades_b, "B", timestamp)
                
                if position_a or position_b:
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
            
            call_bid = call_data.get('best_bid', {}).get('price', 0) if call_data.get('best_bid') else 0
            call_ask = call_data.get('best_ask', {}).get('price', 0) if call_data.get('best_ask') else 0
            put_bid = put_data.get('best_bid', {}).get('price', 0) if put_data.get('best_bid') else 0
            put_ask = put_data.get('best_ask', {}).get('price', 0) if put_data.get('best_ask') else 0
            
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
            
            rsi_history.append(rsi)
            rsi_slope = calculate_rsi_slope(rsi_history)
            
            # Market metrics
            seconds_remaining = get_seconds_to_expiry()
            seconds_into_period = 900 - seconds_remaining
            distance = abs(btc_price - strike_price) if strike_price else 0
            volatility = calculate_btc_volatility(btc_vol_history)
            bin_key = get_bin_key(distance, seconds_remaining, volatility)
            
            # Check trading window
            in_time_window = (seconds_into_period >= MAX_SECONDS_INTO_PERIOD and 
                            seconds_remaining >= MIN_SECONDS_REMAINING)
            
            # Display
            pos_a_str = f"{position_a['type']}@${position_a['entry_price']:.2f}" if position_a else "NONE"
            pos_b_str = f"{position_b['type']}@${position_b['entry_price']:.2f}" if position_b else "NONE"
            slope_str = f"{rsi_slope:+.4f}" if rsi_slope else "N/A"
            print(f"[{timestamp}] BTC:${btc_price:.2f} RSI:{rsi:>5.1f} Slope:{slope_str} | A:{pos_a_str:<12} PNL:${trades_a['total_pnl']:+.2f} | B:{pos_b_str:<12} PNL:${trades_b['total_pnl']:+.2f}", end='\r')
            
            # ===== STRATEGY A: RSI Threshold =====
            if in_time_window:
                current_time = time.time()
                signal_a = None
                if rsi < RSI_LOWER:
                    signal_a = 'BUY'
                elif rsi > RSI_UPPER:
                    signal_a = 'SELL'
                
                if signal_a:
                    if last_signal_type != signal_a:
                        last_signal_time = current_time
                        last_signal_type = signal_a
                    elif current_time - last_signal_time >= CONFIRMATION_SECONDS:
                        if signal_a == 'BUY' and (not position_a or position_a['type'] == 'PUT'):
                            if position_a:
                                position_a = close_position_signal(position_a, put_bid, timestamp, trades_a, "A")
                                time.sleep(2)
                            if MIN_BUY_PRICE <= call_ask <= MAX_BUY_PRICE:
                                position_a = open_position('CALL', call_ask, timestamp, strike_price, btc_price, rsi, bin_key, seconds_remaining, trades_a, "A")
                            last_signal_type = None
                        elif signal_a == 'SELL' and (not position_a or position_a['type'] == 'CALL'):
                            if position_a:
                                position_a = close_position_signal(position_a, call_bid, timestamp, trades_a, "A")
                                time.sleep(2)
                            if MIN_BUY_PRICE <= put_ask <= MAX_BUY_PRICE:
                                position_a = open_position('PUT', put_ask, timestamp, strike_price, btc_price, rsi, bin_key, seconds_remaining, trades_a, "A")
                            last_signal_type = None
                else:
                    last_signal_type = None
            
            # ===== STRATEGY B: RSI Slope =====
            if rsi_slope is not None and in_time_window:
                current_slope_sign = 'POS' if rsi_slope > 0 else 'NEG'
                
                if prev_slope_sign != current_slope_sign:
                    if current_slope_sign == 'POS':
                        # Slope turned positive → Want CALL
                        if position_b and position_b['type'] == 'PUT':
                            position_b = close_position_signal(position_b, put_bid, timestamp, trades_b, "B", "SLOPE_CHANGE")
                            time.sleep(2)
                        if not position_b and MIN_BUY_PRICE <= call_ask <= MAX_BUY_PRICE:
                            position_b = open_position('CALL', call_ask, timestamp, strike_price, btc_price, rsi, bin_key, seconds_remaining, trades_b, "B", rsi_slope)
                    
                    elif current_slope_sign == 'NEG':
                        # Slope turned negative → Want PUT
                        if position_b and position_b['type'] == 'CALL':
                            position_b = close_position_signal(position_b, call_bid, timestamp, trades_b, "B", "SLOPE_CHANGE")
                            time.sleep(2)
                        if not position_b and MIN_BUY_PRICE <= put_ask <= MAX_BUY_PRICE:
                            position_b = open_position('PUT', put_ask, timestamp, strike_price, btc_price, rsi, bin_key, seconds_remaining, trades_b, "B", rsi_slope)
                    
                    prev_slope_sign = current_slope_sign
            
            time.sleep(CHECK_INTERVAL)
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping...")
        
        # Close any open positions
        for pos, data, strat, call_b, put_b in [(position_a, trades_a, "A", call_bid, put_bid), 
                                                  (position_b, trades_b, "B", call_bid, put_bid)]:
            if pos:
                exit_price = call_b if pos['type'] == 'CALL' else put_b
                if exit_price > 0:
                    pos = close_position_signal(pos, exit_price, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                                               data, strat, "MANUAL_STOP")
        
        # Print summaries
        for strat, data in [("A", trades_a), ("B", trades_b)]:
            print(f"\n{'='*80}\nStrategy {strat} Summary\n{'='*80}")
            print(f"Total PNL: ${data['total_pnl']:+.2f}")
            print(f"Trades: {data['stats']['total_trades']} | Wins: {data['stats']['winning_trades']} | Losses: {data['stats']['losing_trades']}")
            if data['stats']['total_trades'] > 0:
                print(f"Win Rate: {data['stats']['winning_trades']/data['stats']['total_trades']*100:.1f}%")
            print("="*80)


if __name__ == "__main__":
    main()
