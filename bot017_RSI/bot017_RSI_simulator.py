#!/usr/bin/env python3
"""
RSI Trading Bot for 15M Binary Options
- Uses RSI(30s) with thresholds: BUY at RSI<15, SELL at RSI>85
- One position at a time
- 2 second confirmation period between opposite signals
- Binary options expire at 00, 15, 30, 45 minutes:
  * CALL pays $1 if BTC > strike at expiration, $0 otherwise
  * PUT pays $1 if BTC < strike at expiration, $0 otherwise
- Positions cannot cross period boundaries - they expire at period end
- Records all trades and PNL in daily JSON
- Handles None bid/ask values when BTC is far from strike
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List
from collections import deque
import numpy as np

# File paths
PUT_FILE = "/home/ubuntu/013_2025_polymarket/15M_PUT.json"
CALL_FILE = "/home/ubuntu/013_2025_polymarket/15M_CALL.json"
BTC_FILE = "/home/ubuntu/013_2025_polymarket/bybit_btc_price.json"
TRADES_DIR = "/home/ubuntu/013_2025_polymarket/bot017_RSI/bot017_RSI_trades"

# Trading parameters
RSI_PERIOD = 300  # 30 seconds at 0.1s intervals = 300 samples
RSI_UPPER = 85
RSI_LOWER = 15
CONFIRMATION_SECONDS = 2
CHECK_INTERVAL = 0.1  # 100ms
MAX_BUY_PRICE = 0.97  # Don't buy above this price
MIN_BUY_PRICE = 0.03  # Don't buy below this price
MIN_SECONDS_REMAINING = 20  # Don't buy in last 20 seconds
MAX_SECONDS_INTO_PERIOD = 20  # Don't buy in first 20 seconds


def read_json(filepath: str) -> Optional[dict]:
    """Read JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return None


def write_json(filepath: str, data: dict):
    """Write JSON file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def get_bin_key(distance: float, seconds_to_expiry: float, volatility: float) -> str:
    """Get bin key for trade recording"""
    distance_bins = [
        (0, 1, "0-1"), (1, 5, "1-5"), (5, 10, "5-10"), (10, 20, "10-20"),
        (20, 40, "20-40"), (40, 80, "40-80"), (80, 160, "80-160"),
        (160, 320, "160-320"), (320, 640, "320-640"), (640, 1280, "640-1280"),
        (1280, float('inf'), "1280+")
    ]

    time_bins = [
        (13*60, 15*60, "15m-13m"), (11*60, 13*60, "13m-11m"), (10*60, 11*60, "11m-10m"),
        (9*60, 10*60, "10m-9m"), (8*60, 9*60, "9m-8m"), (7*60, 8*60, "8m-7m"),
        (6*60, 7*60, "7m-6m"), (5*60, 6*60, "6m-5m"), (4*60, 5*60, "5m-4m"),
        (3*60, 4*60, "4m-3m"), (2*60, 3*60, "3m-2m"), (90, 120, "120s-90s"),
        (60, 90, "90s-60s"), (40, 60, "60s-40s"), (30, 40, "40s-30s"),
        (20, 30, "30s-20s"), (10, 20, "20s-10s"), (5, 10, "10s-5s"),
        (2, 5, "5s-2s"), (0, 2, "last-2s")
    ]

    vol_bins = [
        (0, 10, "0-10"), (10, 20, "10-20"), (20, 30, "20-30"), (30, 40, "30-40"),
        (40, 60, "40-60"), (60, 90, "60-90"), (90, 120, "90-120"), (120, 240, "120-240"),
        (240, float('inf'), "240+")
    ]

    def get_bin_label(value, bins):
        for min_val, max_val, label in bins:
            if min_val <= value < max_val:
                return label
        return bins[-1][2]

    dist_label = get_bin_label(distance, distance_bins)
    time_label = get_bin_label(seconds_to_expiry, time_bins)
    vol_label = get_bin_label(volatility, vol_bins)

    return f"{dist_label}|{time_label}|{vol_label}"


def get_strike_price() -> Optional[float]:
    """Get strike price from Bybit API"""
    try:
        import requests

        now = datetime.now(timezone.utc)
        current_minute = now.minute

        for start_min in [0, 15, 30, 45]:
            if current_minute >= start_min and current_minute < start_min + 15:
                period_start = now.replace(minute=start_min, second=0, microsecond=0)
                start_timestamp = int(period_start.timestamp() * 1000)

                url = "https://api.bybit.com/v5/market/mark-price-kline"
                params = {
                    'category': 'linear',
                    'symbol': 'BTCUSDT',
                    'interval': '15',
                    'start': start_timestamp,
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
    except:
        return None


def get_seconds_to_expiry() -> float:
    """Get seconds to expiry"""
    now = datetime.now()
    current_minute = now.minute

    for start_min in [0, 15, 30, 45]:
        if current_minute >= start_min and current_minute < start_min + 15:
            seconds_into_period = (current_minute - start_min) * 60 + now.second
            return 900 - seconds_into_period
    return 0


def calculate_btc_volatility(price_history: deque) -> float:
    """Calculate BTC volatility over last minute (price range)"""
    if len(price_history) < 10:
        return 0.0
    prices = list(price_history)
    return max(prices) - min(prices)


def calculate_rsi(prices: deque, period: int = RSI_PERIOD) -> Optional[float]:
    """Calculate RSI from price deque"""
    if len(prices) < period + 1:
        return None
    
    # Get last 'period + 1' prices to calculate 'period' changes
    price_array = np.array(list(prices))[-(period + 1):]
    
    # Calculate price changes
    deltas = np.diff(price_array)
    
    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Calculate average gain and loss
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    
    # Avoid division by zero
    if avg_loss == 0:
        return 100.0
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def load_daily_trades() -> Dict:
    """Load today's trades file or create new one"""
    today = datetime.now().strftime("%Y-%m-%d")
    filepath = f"{TRADES_DIR}/trades_{today}.json"
    
    if Path(filepath).exists():
        data = read_json(filepath)
        if data:
            print(f"📂 Loaded existing trades: {len(data.get('trades', []))} trades, PNL: ${data.get('total_pnl', 0):.2f}")
            return data
    
    # Create new trades file
    return {
        "date": today,
        "total_pnl": 0.0,
        "trades": [],
        "stats": {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "call_trades": 0,
            "put_trades": 0
        }
    }


def save_daily_trades(trades_data: Dict):
    """Save trades data to daily file"""
    today = datetime.now().strftime("%Y-%m-%d")
    filepath = f"{TRADES_DIR}/trades_{today}.json"
    write_json(filepath, trades_data)


def main():
    print("\n" + "="*150)
    print("RSI TRADING BOT - 15M Binary Options")
    print(f"RSI Period: 30s | Upper: {RSI_UPPER} | Lower: {RSI_LOWER} | Confirmation: {CONFIRMATION_SECONDS}s")
    print(f"Price Limits: ${MIN_BUY_PRICE:.2f} - ${MAX_BUY_PRICE:.2f} | Time Window: {MAX_SECONDS_INTO_PERIOD}s - {900-MIN_SECONDS_REMAINING}s into period")
    print("="*150)
    
    # Load daily trades
    trades_data = load_daily_trades()
    
    # Track current position
    position = None  # {'type': 'CALL' or 'PUT', 'entry_price': float, 'entry_time': str, 'entry_strike': float, ...}
    
    # Price history for RSI calculation (30s at 0.1s = 300 samples)
    btc_price_history = deque(maxlen=RSI_PERIOD + 1)
    
    # Price history for volatility (1 minute = 600 samples)
    btc_vol_history = deque(maxlen=600)
    
    # Signal confirmation tracking
    last_signal_time = 0
    last_signal_type = None
    
    # Strike price tracking
    strike_price = None
    last_strike_update_minute = None
    current_period_start = None
    
    # Track if we're waiting for valid prices
    waiting_for_prices = False
    
    try:
        while True:
            now = datetime.now()
            current_minute = now.minute
            current_second = now.second
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            
            # Check if we need to update strike price
            is_period_start = current_minute in [0, 15, 30, 45]
            
            # At period boundaries, close any open positions first
            if is_period_start and current_second == 0 and position:
                print(f"\n⏰ PERIOD END - Binary option expiring...")
                
                # Determine final BTC price and strike
                btc_data = read_json(BTC_FILE)
                final_btc = btc_data.get('price', 0) if btc_data else 0
                
                # Determine option value at expiration (0 or 1)
                if position['type'] == 'CALL':
                    # CALL pays $1 if BTC closes above strike, $0 otherwise
                    exit_price = 1.0 if final_btc > strike_price else 0.0
                elif position['type'] == 'PUT':
                    # PUT pays $1 if BTC closes below strike, $0 otherwise
                    exit_price = 1.0 if final_btc < strike_price else 0.0
                
                pnl = exit_price - position['entry_price']
                trades_data['total_pnl'] += pnl
                
                # Record trade
                trade_record = {
                    **position,
                    'exit_price': exit_price,
                    'exit_time': timestamp,
                    'exit_btc': final_btc,
                    'pnl': pnl,
                    'exit_reason': 'EXPIRATION',
                    'expired_itm': exit_price == 1.0  # In The Money
                }
                trades_data['trades'].append(trade_record)
                
                # Update stats
                trades_data['stats']['total_trades'] += 1
                if pnl > 0:
                    trades_data['stats']['winning_trades'] += 1
                else:
                    trades_data['stats']['losing_trades'] += 1
                
                # Save immediately after closing trade
                save_daily_trades(trades_data)
                
                result_emoji = "✅" if exit_price == 1.0 else "❌"
                print(f"💰 {result_emoji} Expired at ${exit_price:.2f} (BTC: ${final_btc:.2f} vs Strike: ${strike_price:.2f}) | PNL: ${pnl:+.3f} | Total PNL: ${trades_data['total_pnl']:+.2f}")
                position = None
                
                # Wait 2 seconds before allowing new positions
                print("⏸️  Waiting 2 seconds before next period...")
                time.sleep(2)
            
            # Update strike price if needed (after force close check)
            if is_period_start and current_second >= 5 and last_strike_update_minute != current_minute:
                new_strike = get_strike_price()
                if new_strike:
                    strike_price = new_strike
                    last_strike_update_minute = current_minute
                    current_period_start = now.replace(second=0, microsecond=0)
                    print(f"\n🔄 New period strike price: ${strike_price:.2f}")
            
            # Initialize strike price on first run
            if strike_price is None:
                strike_price = get_strike_price()
                if strike_price:
                    current_period_start = now.replace(second=0, microsecond=0)
                    print(f"Initial Strike Price: ${strike_price:.2f}\n")
            
            # Read current prices
            btc_data = read_json(BTC_FILE)
            call_data = read_json(CALL_FILE)
            put_data = read_json(PUT_FILE)
            
            if not all([btc_data, call_data, put_data]):
                time.sleep(CHECK_INTERVAL)
                continue
            
            btc_price = btc_data.get('price', 0)
            btc_price_history.append(btc_price)
            btc_vol_history.append(btc_price)
            
            # Handle None values for best_bid/best_ask (when BTC is far from strike)
            call_bid_data = call_data.get('best_bid')
            call_ask_data = call_data.get('best_ask')
            put_bid_data = put_data.get('best_bid')
            put_ask_data = put_data.get('best_ask')
            
            call_bid = call_bid_data.get('price', 0) if call_bid_data else 0
            call_ask = call_ask_data.get('price', 0) if call_ask_data else 0
            put_bid = put_bid_data.get('price', 0) if put_bid_data else 0
            put_ask = put_ask_data.get('price', 0) if put_ask_data else 0
            
            # Check if we have valid prices for trading
            has_valid_prices = call_bid > 0 and call_ask > 0 and put_bid > 0 and put_ask > 0
            
            if not has_valid_prices:
                if not waiting_for_prices:
                    print(f"\n⚠️  Missing bid/ask prices (BTC far from strike: ${btc_price:.2f} vs ${strike_price:.2f} if strike_price else 0) - waiting for valid prices...")
                    waiting_for_prices = True
                time.sleep(CHECK_INTERVAL)
                continue
            else:
                if waiting_for_prices:
                    print(f"✅ Valid prices restored")
                    waiting_for_prices = False
            
            # Calculate RSI
            rsi = calculate_rsi(btc_price_history)
            
            if rsi is None:
                print(f"[{timestamp}] Collecting data... {len(btc_price_history)}/{RSI_PERIOD + 1} samples", end='\r')
                time.sleep(CHECK_INTERVAL)
                continue
            
            # Calculate market metrics
            seconds_remaining = get_seconds_to_expiry()
            distance = abs(btc_price - strike_price) if strike_price else 0
            volatility = calculate_btc_volatility(btc_vol_history)
            bin_key = get_bin_key(distance, seconds_remaining, volatility)
            
            # Display current state
            pos_str = f"{position['type']} @${position['entry_price']:.2f}" if position else "NONE"
            print(f"[{timestamp}] BTC: ${btc_price:.2f} | RSI: {rsi:>5.1f} | C: {call_bid:.2f}/{call_ask:.2f} | P: {put_bid:.2f}/{put_ask:.2f} | Pos: {pos_str:<15} | PNL: ${trades_data['total_pnl']:+.2f}", end='\r')
            
            # Trading logic
            current_time = time.time()
            
            # Generate signals
            signal = None
            if rsi < RSI_LOWER:
                signal = 'BUY'  # BUY CALL / SELL PUT
            elif rsi > RSI_UPPER:
                signal = 'SELL'  # BUY PUT / SELL CALL
            
            # Note: When we exit a position due to opposite RSI signal, we sell at current market price (bid)
            # When a position expires at period end, it settles at $0 or $1 based on BTC vs strike
            
            # Check if we have a confirmed signal
            if signal:
                # New signal or same signal continuing
                if last_signal_type != signal:
                    # New signal started
                    last_signal_time = current_time
                    last_signal_type = signal
                elif current_time - last_signal_time >= CONFIRMATION_SECONDS:
                    # Signal confirmed for 2 seconds
                    
                    if signal == 'BUY' and (not position or position['type'] == 'PUT'):
                        # Close PUT if holding, then BUY CALL
                        if position and position['type'] == 'PUT':
                            pnl = put_bid - position['entry_price']
                            trades_data['total_pnl'] += pnl
                            
                            trade_record = {
                                **position,
                                'exit_price': put_bid,
                                'exit_time': timestamp,
                                'pnl': pnl,
                                'exit_reason': 'RSI_SIGNAL'
                            }
                            trades_data['trades'].append(trade_record)
                            trades_data['stats']['total_trades'] += 1
                            if pnl > 0:
                                trades_data['stats']['winning_trades'] += 1
                            else:
                                trades_data['stats']['losing_trades'] += 1
                            
                            # Save immediately after closing trade
                            save_daily_trades(trades_data)
                            
                            print(f"\n🔴 SOLD PUT @${put_bid:.2f} | PNL: ${pnl:+.3f}                                    ")
                            position = None
                            
                            # Wait 2 seconds before opening new position
                            print("⏸️  Waiting 2 seconds before next trade...")
                            time.sleep(2)
                        
                        # BUY CALL - check price limits and time window
                        seconds_into_period = 900 - seconds_remaining
                        
                        if call_ask < MIN_BUY_PRICE or call_ask > MAX_BUY_PRICE:
                            print(f"\n⚠️  CALL price ${call_ask:.2f} outside limits (${MIN_BUY_PRICE:.2f}-${MAX_BUY_PRICE:.2f}) - skipping                    ")
                            last_signal_type = None
                        elif seconds_into_period < MAX_SECONDS_INTO_PERIOD:
                            print(f"\n⚠️  Too early in period ({seconds_into_period:.0f}s < {MAX_SECONDS_INTO_PERIOD}s) - skipping                    ")
                            last_signal_type = None
                        elif seconds_remaining < MIN_SECONDS_REMAINING:
                            print(f"\n⚠️  Too close to expiration ({seconds_remaining:.0f}s < {MIN_SECONDS_REMAINING}s) - skipping                    ")
                            last_signal_type = None
                        else:
                            position = {
                                'type': 'CALL',
                                'entry_price': call_ask,
                                'entry_time': timestamp,
                                'entry_strike': strike_price,
                                'entry_btc': btc_price,
                                'entry_rsi': rsi,
                                'entry_bin': bin_key,
                                'entry_seconds_remaining': seconds_remaining
                            }
                            trades_data['stats']['call_trades'] += 1
                            save_daily_trades(trades_data)
                            print(f"\n🟢 BOUGHT CALL @${call_ask:.2f} | RSI: {rsi:.1f} | {seconds_remaining:.0f}s remaining                                    ")
                            last_signal_type = None  # Reset to avoid immediate re-trigger
                    
                    elif signal == 'SELL' and (not position or position['type'] == 'CALL'):
                        # Close CALL if holding, then BUY PUT
                        if position and position['type'] == 'CALL':
                            pnl = call_bid - position['entry_price']
                            trades_data['total_pnl'] += pnl
                            
                            trade_record = {
                                **position,
                                'exit_price': call_bid,
                                'exit_time': timestamp,
                                'pnl': pnl,
                                'exit_reason': 'RSI_SIGNAL'
                            }
                            trades_data['trades'].append(trade_record)
                            trades_data['stats']['total_trades'] += 1
                            if pnl > 0:
                                trades_data['stats']['winning_trades'] += 1
                            else:
                                trades_data['stats']['losing_trades'] += 1
                            
                            # Save immediately after closing trade
                            save_daily_trades(trades_data)
                            
                            print(f"\n🔴 SOLD CALL @${call_bid:.2f} | PNL: ${pnl:+.3f}                                    ")
                            position = None
                            
                            # Wait 2 seconds before opening new position
                            print("⏸️  Waiting 2 seconds before next trade...")
                            time.sleep(2)
                        
                        # BUY PUT - check price limits and time window
                        seconds_into_period = 900 - seconds_remaining
                        
                        if put_ask < MIN_BUY_PRICE or put_ask > MAX_BUY_PRICE:
                            print(f"\n⚠️  PUT price ${put_ask:.2f} outside limits (${MIN_BUY_PRICE:.2f}-${MAX_BUY_PRICE:.2f}) - skipping                    ")
                            last_signal_type = None
                        elif seconds_into_period < MAX_SECONDS_INTO_PERIOD:
                            print(f"\n⚠️  Too early in period ({seconds_into_period:.0f}s < {MAX_SECONDS_INTO_PERIOD}s) - skipping                    ")
                            last_signal_type = None
                        elif seconds_remaining < MIN_SECONDS_REMAINING:
                            print(f"\n⚠️  Too close to expiration ({seconds_remaining:.0f}s < {MIN_SECONDS_REMAINING}s) - skipping                    ")
                            last_signal_type = None
                        else:
                            position = {
                                'type': 'PUT',
                                'entry_price': put_ask,
                                'entry_time': timestamp,
                                'entry_strike': strike_price,
                                'entry_btc': btc_price,
                                'entry_rsi': rsi,
                                'entry_bin': bin_key,
                                'entry_seconds_remaining': seconds_remaining
                            }
                            trades_data['stats']['put_trades'] += 1
                            save_daily_trades(trades_data)
                            print(f"\n🟢 BOUGHT PUT @${put_ask:.2f} | RSI: {rsi:.1f} | {seconds_remaining:.0f}s remaining                                    ")
                            last_signal_type = None
            else:
                # No signal, reset tracking
                last_signal_type = None
                last_signal_time = 0
            
            time.sleep(CHECK_INTERVAL)
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping bot...")
        
        # Close any open position at current market price (not expiration)
        if position:
            print(f"Closing open {position['type']} position at market price...")
            if position['type'] == 'CALL':
                data = read_json(CALL_FILE)
                bid_data = data.get('best_bid') if data else None
                exit_price = bid_data.get('price', 0) if bid_data else 0
            else:
                data = read_json(PUT_FILE)
                bid_data = data.get('best_bid') if data else None
                exit_price = bid_data.get('price', 0) if bid_data else 0
            
            if exit_price > 0:
                pnl = exit_price - position['entry_price']
                trades_data['total_pnl'] += pnl
                
                btc_data = read_json(BTC_FILE)
                current_btc = btc_data.get('price', 0) if btc_data else 0
                
                trade_record = {
                    **position,
                    'exit_price': exit_price,
                    'exit_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'exit_btc': current_btc,
                    'pnl': pnl,
                    'exit_reason': 'MANUAL_STOP'
                }
                trades_data['trades'].append(trade_record)
                trades_data['stats']['total_trades'] += 1
                if pnl > 0:
                    trades_data['stats']['winning_trades'] += 1
                else:
                    trades_data['stats']['losing_trades'] += 1
                
                # Save immediately after closing trade
                save_daily_trades(trades_data)
                print(f"Closed at ${exit_price:.2f} | PNL: ${pnl:+.3f}")
        
        # Print summary
        print("\n" + "="*80)
        print("SESSION SUMMARY")
        print("="*80)
        print(f"Total PNL: ${trades_data['total_pnl']:+.2f}")
        print(f"Total Trades: {trades_data['stats']['total_trades']}")
        print(f"Winning: {trades_data['stats']['winning_trades']} | Losing: {trades_data['stats']['losing_trades']}")
        print(f"CALL Trades: {trades_data['stats']['call_trades']} | PUT Trades: {trades_data['stats']['put_trades']}")
        if trades_data['stats']['total_trades'] > 0:
            win_rate = trades_data['stats']['winning_trades'] / trades_data['stats']['total_trades'] * 100
            print(f"Win Rate: {win_rate:.1f}%")
        print("="*80)


if __name__ == "__main__":
    main()
