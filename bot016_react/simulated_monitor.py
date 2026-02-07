#!/usr/bin/env python3
"""
Simulated Trading Monitor - 1 Token Position Tracking
Executes trades based on sensitivity signals and tracks PNL
"""

import json
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict
from collections import deque

# File paths
PUT_FILE = "/home/ubuntu/013_2025_polymarket/15M_PUT.json"
CALL_FILE = "/home/ubuntu/013_2025_polymarket/15M_CALL.json"
BTC_FILE = "/home/ubuntu/013_2025_polymarket/bybit_btc_price.json"
SENSITIVITY_FILE = "/home/ubuntu/013_2025_polymarket/bot016_react/sensitivity_data/sensitivity_master.json"
TRADES_DIR = "/home/ubuntu/013_2025_polymarket/bot016_react/sensitivity_trades"

# Trading parameters
sens_multiplier = 3
action_threshold = 0.03
min_seconds_between_positions = 2


def read_json(filepath: str) -> Optional[dict]:
    """Read JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return None


def get_bin_key(distance: float, seconds_to_expiry: float, volatility: float) -> str:
    """Get bin key"""
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
        from datetime import timezone
        
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


class SimulatedTrader:
    """Manages simulated trading positions and PNL"""
    
    def __init__(self, trades_dir: str):
        self.trades_dir = Path(trades_dir)
        self.trades_dir.mkdir(exist_ok=True, parents=True)
        
        # Current position
        self.position = None  # {'type': 'CALL'/'PUT', 'entry_price': float, 'entry_btc': float, 
                              #  'entry_bin': str, 'edge': float, 'entry_time': float}
        self.last_position_close_time = 0
        
        # Today's trades
        self.today_trades = []
        self.load_today_trades()
        
    def get_today_filename(self) -> Path:
        """Get filename for today's trades"""
        today = date.today().strftime('%Y%m%d')
        return self.trades_dir / f"trades_{today}.json"
    
    def load_today_trades(self):
        """Load today's trades if they exist"""
        filename = self.get_today_filename()
        if filename.exists():
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                    self.today_trades = data.get('trades', [])
                    print(f"📂 Loaded {len(self.today_trades)} trades from {filename}")
            except:
                self.today_trades = []
        else:
            self.today_trades = []
    
    def save_trades(self):
        """Save today's trades to file"""
        filename = self.get_today_filename()
        
        # Calculate daily PNL
        daily_pnl = sum(t['pnl'] for t in self.today_trades)
        win_count = sum(1 for t in self.today_trades if t['pnl'] > 0)
        loss_count = sum(1 for t in self.today_trades if t['pnl'] < 0)
        
        data = {
            'date': date.today().isoformat(),
            'total_trades': len(self.today_trades),
            'wins': win_count,
            'losses': loss_count,
            'win_rate': win_count / len(self.today_trades) if self.today_trades else 0,
            'daily_pnl': daily_pnl,
            'trades': self.today_trades
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def can_open_position(self) -> bool:
        """Check if we can open a new position"""
        if self.position is not None:
            return False
        
        # Must wait min_seconds_between_positions after closing previous position
        if time.time() - self.last_position_close_time < min_seconds_between_positions:
            return False
        
        return True
    
    def open_position(self, position_type: str, entry_price: float, btc_price: float, 
                     bin_key: str, edge: float):
        """Open a new position"""
        if not self.can_open_position():
            return False
        
        self.position = {
            'type': position_type,
            'entry_price': entry_price,
            'entry_btc': btc_price,
            'entry_bin': bin_key,
            'edge': edge,
            'entry_time': time.time(),
            'entry_timestamp': datetime.now().isoformat()
        }
        
        print(f"\n🟢 OPENED {position_type} @ {entry_price:.3f} | BTC: ${btc_price:.2f} | Edge: {edge:.3f}")
        return True
    
    def close_position(self, exit_price: float, btc_price: float, bin_key: str, reason: str = "SIGNAL"):
        """Close current position"""
        if self.position is None:
            return
        
        # Calculate PNL
        if self.position['type'] == 'CALL':
            pnl = exit_price - self.position['entry_price']
        else:  # PUT
            pnl = exit_price - self.position['entry_price']
        
        # Record trade
        trade = {
            'type': self.position['type'],
            'open_time': self.position['entry_timestamp'],
            'close_time': datetime.now().isoformat(),
            'open_btc': self.position['entry_btc'],
            'close_btc': btc_price,
            'open_price': self.position['entry_price'],
            'close_price': exit_price,
            'open_bin': self.position['entry_bin'],
            'close_bin': bin_key,
            'edge': self.position['edge'],
            'pnl': pnl,
            'close_reason': reason
        }
        
        self.today_trades.append(trade)
        
        pnl_symbol = "📈" if pnl > 0 else "📉"
        print(f"\n🔴 CLOSED {self.position['type']} @ {exit_price:.3f} | BTC: ${btc_price:.2f} | PNL: {pnl:+.3f} {pnl_symbol}")
        
        self.position = None
        self.last_position_close_time = time.time()
        
        # Save after each trade
        self.save_trades()
    
    def get_daily_pnl(self) -> float:
        """Get today's total PNL"""
        return sum(t['pnl'] for t in self.today_trades)


def main():
    print("\n" + "="*150)
    print("SIMULATED TRADING MONITOR - 1 Token Position Tracking")
    print("="*150)
    print(f"Sensitivity Multiplier: {sens_multiplier}x | Action Threshold: {action_threshold}")
    print("-"*150)
    
    # Initialize trader
    trader = SimulatedTrader(TRADES_DIR)
    
    # Load sensitivity data
    sensitivity_data = read_json(SENSITIVITY_FILE)
    
    # Track previous state for delta calculation
    prev_btc_price = None
    prev_call_bid = None
    prev_call_ask = None
    prev_put_bid = None
    prev_put_ask = None
    
    # BTC price history for volatility
    btc_price_history = deque(maxlen=600)
    
    # Track strike price and period
    strike_price = None
    last_strike_update_minute = None
    current_period_start = None
    
    # Buffer zone: no trading in last 20 seconds of period
    BUFFER_SECONDS = 20
    
    # Store last valid prices before period end
    last_call_bid = None
    last_put_bid = None
    last_btc_price = None
    
    try:
        while True:
            now = datetime.now()
            current_minute = now.minute
            current_second = now.second
            
            # Determine current period
            period_start = None
            for start_min in [0, 15, 30, 45]:
                if current_minute >= start_min and current_minute < start_min + 15:
                    period_start = start_min
                    break
            
            # Calculate seconds into current period
            if period_start is not None:
                seconds_into_period = (current_minute - period_start) * 60 + current_second
                seconds_remaining = 900 - seconds_into_period
                in_buffer_zone = seconds_remaining <= BUFFER_SECONDS
            else:
                seconds_into_period = 0
                seconds_remaining = 0
                in_buffer_zone = True
            
            # NEW PERIOD DETECTED
            if period_start != current_period_start:
                # Close any open position from previous period using LAST VALID PRICE
                if trader.position is not None and current_period_start is not None:
                    print(f"\n⚠️  PERIOD END - Closing position with last valid price from previous period")
                    
                    # Use last valid bid price from the ending period
                    if trader.position['type'] == 'CALL':
                        exit_price = last_call_bid if last_call_bid and last_call_bid > 0 else 0
                    else:  # PUT
                        exit_price = last_put_bid if last_put_bid and last_put_bid > 0 else 0
                    
                    exit_btc = last_btc_price if last_btc_price else 0
                    
                    if exit_price > 0 and exit_btc > 0:
                        # Use bin from end of previous period
                        distance = abs(exit_btc - strike_price) if strike_price else 0
                        volatility = calculate_btc_volatility(btc_price_history)
                        bin_key = get_bin_key(distance, BUFFER_SECONDS - 1, volatility)  # ~19 seconds left
                        
                        trader.close_position(exit_price, exit_btc, bin_key, "PERIOD_END")
                    else:
                        print(f"❌ Cannot close - no valid exit price from previous period")
                        # Force close with 0 PNL
                        trader.close_position(trader.position['entry_price'], exit_btc, 
                                            trader.position['entry_bin'], "PERIOD_END_NO_PRICE")
                
                # Update period tracking
                current_period_start = period_start
                
                # Reset last valid prices for new period
                last_call_bid = None
                last_put_bid = None
                last_btc_price = None
                
                print(f"\n{'='*80}")
                print(f"🔄 NEW PERIOD STARTED: {now.strftime('%H:%M')} (:{period_start:02d})")
                print(f"{'='*80}")
            
            # Update strike price (wait 5 seconds into period)
            is_period_start = current_minute in [0, 15, 30, 45]
            if is_period_start and current_second >= 5 and current_second < 10 and last_strike_update_minute != current_minute:
                print(f"\n🔄 Updating strike price at {now.strftime('%H:%M:%S')}...")
                new_strike = get_strike_price()
                if new_strike:
                    strike_price = new_strike
                    last_strike_update_minute = current_minute
                    print(f"✅ Strike: ${strike_price:.2f}")
                    print(f"📊 Daily Stats: {len(trader.today_trades)} trades | PNL: {trader.get_daily_pnl():+.3f}\n")
            
            # Initialize strike price on first run
            if strike_price is None:
                strike_price = get_strike_price()
                if not strike_price:
                    btc_data = read_json(BTC_FILE)
                    if btc_data:
                        strike_price = btc_data['price']
                if strike_price:
                    print(f"Initial Strike: ${strike_price:.2f}\n")
            
            # Read prices
            btc_data = read_json(BTC_FILE)
            call_data = read_json(CALL_FILE)
            put_data = read_json(PUT_FILE)
            
            if not all([btc_data, call_data, put_data]):
                time.sleep(0.1)
                continue
            
            # Extract data
            btc_price = btc_data.get('price', 0)
            btc_price_history.append(btc_price)
            
            call_bid_price = call_data.get('best_bid', {}).get('price', 0) if call_data.get('best_bid') else 0
            call_ask_price = call_data.get('best_ask', {}).get('price', 0) if call_data.get('best_ask') else 0
            
            put_bid_price = put_data.get('best_bid', {}).get('price', 0) if put_data.get('best_bid') else 0
            put_ask_price = put_data.get('best_ask', {}).get('price', 0) if put_data.get('best_ask') else 0
            
            # Store last valid prices (before buffer zone)
            if not in_buffer_zone:
                if call_bid_price > 0:
                    last_call_bid = call_bid_price
                if put_bid_price > 0:
                    last_put_bid = put_bid_price
                if btc_price > 0:
                    last_btc_price = btc_price
            
            # Calculate volatility
            volatility = calculate_btc_volatility(btc_price_history)
            
            # TRADING LOGIC - Only outside buffer zone
            if not in_buffer_zone and strike_price and prev_btc_price is not None:
                distance = abs(btc_price - strike_price)
                bin_key = get_bin_key(distance, seconds_remaining, volatility)
                
                # Check for signals
                if sensitivity_data and bin_key in sensitivity_data.get('bins', {}):
                    bin_data = sensitivity_data['bins'][bin_key]
                    call_sens = bin_data.get('call_sensitivity', {}).get('avg', 0)
                    put_sens = bin_data.get('put_sensitivity', {}).get('avg', 0)
                    
                    btc_delta = btc_price - prev_btc_price
                    
                    # CALL signals
                    if abs(call_sens) > 0.000001 and call_ask_price > 0 and call_bid_price > 0:
                        ideal_call_movement = btc_delta * call_sens * sens_multiplier
                        actual_call_ask_movement = call_ask_price - prev_call_ask
                        
                        # Check if not lagging
                        if abs(actual_call_ask_movement) <= abs(ideal_call_movement) + 0.02:
                            ideal_call_ask = prev_call_ask + ideal_call_movement
                            ideal_call_bid = prev_call_bid + ideal_call_movement
                            
                            # SELL signal - close CALL position
                            if ideal_call_bid < call_bid_price - action_threshold:
                                if trader.position and trader.position['type'] == 'CALL':
                                    trader.close_position(call_bid_price, btc_price, bin_key, "SELL_SIGNAL")
                            
                            # BUY signal - open CALL position
                            elif ideal_call_ask > call_ask_price + action_threshold:
                                if trader.can_open_position():
                                    edge = ideal_call_ask - call_ask_price
                                    trader.open_position('CALL', call_ask_price, btc_price, bin_key, edge)
                    
                    # PUT signals
                    if abs(put_sens) > 0.000001 and put_ask_price > 0 and put_bid_price > 0:
                        ideal_put_movement = btc_delta * put_sens * sens_multiplier
                        actual_put_ask_movement = put_ask_price - prev_put_ask
                        
                        # Check if not lagging
                        if abs(actual_put_ask_movement) <= abs(ideal_put_movement) + 0.02:
                            ideal_put_ask = prev_put_ask + ideal_put_movement
                            ideal_put_bid = prev_put_bid + ideal_put_movement
                            
                            # SELL signal - close PUT position
                            if ideal_put_bid < put_bid_price - action_threshold:
                                if trader.position and trader.position['type'] == 'PUT':
                                    trader.close_position(put_bid_price, btc_price, bin_key, "SELL_SIGNAL")
                            
                            # BUY signal - open PUT position
                            elif ideal_put_ask > put_ask_price + action_threshold:
                                if trader.can_open_position():
                                    edge = ideal_put_ask - put_ask_price
                                    trader.open_position('PUT', put_ask_price, btc_price, bin_key, edge)
            
            elif in_buffer_zone and trader.position is not None:
                # In buffer zone with open position - show warning
                if seconds_remaining % 5 == 0:  # Show every 5 seconds
                    print(f"⏳ Buffer zone: {seconds_remaining}s remaining - position will close at period end")
            
            # Update previous state
            prev_btc_price = btc_price
            prev_call_bid = call_bid_price
            prev_call_ask = call_ask_price
            prev_put_bid = put_bid_price
            prev_put_ask = put_ask_price
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\n⏸️  Stopped by user")
        
        # Close any open position
        if trader.position:
            print(f"\n⚠️  Closing open {trader.position['type']} position...")
            if trader.position['type'] == 'CALL':
                data = read_json(CALL_FILE)
            else:
                data = read_json(PUT_FILE)
            
            if data and data.get('best_bid'):
                exit_price = data['best_bid'].get('price', 0)
                btc_data = read_json(BTC_FILE)
                btc_price = btc_data.get('price', 0) if btc_data else 0
                
                if exit_price > 0:
                    distance = abs(btc_price - strike_price) if strike_price else 0
                    volatility = calculate_btc_volatility(btc_price_history)
                    seconds_left = get_seconds_to_expiry()
                    bin_key = get_bin_key(distance, seconds_left, volatility)
                    
                    trader.close_position(exit_price, btc_price, bin_key, "MANUAL_STOP")
        
        # Final save
        trader.save_trades()
        print(f"\n💾 Saved {len(trader.today_trades)} trades")
        print(f"📊 Daily PNL: {trader.get_daily_pnl():+.3f}")


if __name__ == "__main__":
    main()
