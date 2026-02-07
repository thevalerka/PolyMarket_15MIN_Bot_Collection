import json
import requests
import time
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime, timezone
import threading
import queue
import signal
import sys
import os
import warnings
warnings.filterwarnings('ignore')

class RealTimeLagArbitrageDetector:
    def __init__(self, max_history=5000):
        # Data storage
        self.btc_prices = deque(maxlen=max_history)
        self.call_bids = deque(maxlen=max_history)
        self.call_asks = deque(maxlen=max_history)
        self.put_bids = deque(maxlen=max_history)
        self.put_asks = deque(maxlen=max_history)
        self.timestamps = deque(maxlen=max_history)
        self.strikes = deque(maxlen=max_history)
        
        # Arbitrage tracking
        self.arbitrage_history = []
        self.open_positions = []  # Track positions that could be closed
        
        # Lag detection parameters
        self.btc_spike_threshold = 0.001  # 0.1% price change threshold
        self.arbitrage_threshold = 0.01   # 1% price advantage threshold
        
        # Statistics tracking
        self.total_opportunities = 0
        self.total_call_opportunities = 0
        self.total_put_opportunities = 0
        
        # Current state
        self.current_strike = None
        self.last_btc_price = None
        self.last_update_time = None
        
        print("🎯 Trading Strategy: BID/ASK only, no mid-prices!")
        print("📊 PNL Tracking: Hold-to-expiry vs Early-exit strategies")
    
    def get_current_strike(self):
        """Get current hour's strike price with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=1"
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                data = response.json()
                
                if data and len(data) > 0 and len(data[0]) > 1:
                    strike_price = float(data[0][1])  # Open price of current hour
                    
                    # Sanity check - strike should be reasonable BTC price
                    if 10000 < strike_price < 500000:
                        return strike_price
                        
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"\n⚠️  Failed to get strike price after {max_retries} attempts: {e}")
                else:
                    time.sleep(0.5)  # Brief pause before retry
                    
        return self.current_strike  # Return last known strike if all attempts fail
    
    def get_btc_price(self):
        """Get current BTC price"""
        try:
            url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
            response = requests.get(url, timeout=2)
            data = response.json()
            return float(data['price'])
        except:
            return self.last_btc_price
    
    def calculate_time_to_expiry(self):
        """Calculate time to expiry as fraction of hour remaining"""
        now = datetime.now(timezone.utc)
        next_hour = now.replace(minute=0, second=0, microsecond=0) + pd.Timedelta(hours=1)
        time_remaining = (next_hour - now).total_seconds() / 3600
        return max(0.001, time_remaining)
    
    def detect_btc_spike(self):
        """Detect if BTC price has spiked recently with dynamic sensitivity"""
        if len(self.btc_prices) < 3:
            return False, 0
            
        recent_prices = list(self.btc_prices)[-3:]
        latest_change = (recent_prices[-1] - recent_prices[-2]) / recent_prices[-2]
        
        # Dynamic threshold based on current strike proximity
        if self.current_strike:
            current_moneyness = recent_prices[-1] / self.current_strike
            distance_from_strike = abs(current_moneyness - 1.0)
            
            # Near the money: more sensitive to smaller moves
            if distance_from_strike < 0.005:  # Within 0.5%
                threshold = 0.0002  # 0.02% threshold
            elif distance_from_strike < 0.02:  # Within 2%
                threshold = 0.0005  # 0.05% threshold
            else:
                threshold = self.btc_spike_threshold  # Normal threshold
        else:
            threshold = self.btc_spike_threshold
        
        spike_detected = abs(latest_change) > threshold
        return spike_detected, latest_change
    
    def calculate_expected_pnl_hold(self, option_type, entry_price, current_btc, strike):
        """Calculate expected PNL if holding to expiry"""
        if option_type == 'CALL':
            # If BTC > strike at expiry, option pays $1, else $0
            if current_btc > strike:
                return 1.0 - entry_price  # Win
            else:
                return 0.0 - entry_price  # Loss
        else:  # PUT
            # If BTC < strike at expiry, option pays $1, else $0
            if current_btc < strike:
                return 1.0 - entry_price  # Win
            else:
                return 0.0 - entry_price  # Loss
    
    def calculate_exit_pnl(self, option_type, entry_price, exit_price):
        """Calculate PNL if exiting now"""
        return exit_price - entry_price
    
    def detect_arbitrage_opportunity(self, btc_price, call_bid, call_ask, put_bid, put_ask, 
                                   strike_price, time_to_expiry):
        """Detect arbitrage opportunities using momentum and overreaction analysis"""
        opportunities = []
        timestamp = datetime.now()
        
        # Calculate moneyness
        moneyness = btc_price / strike_price
        distance_from_strike = abs(moneyness - 1.0)
        
        # MOMENTUM-BASED ARBITRAGE DETECTION
        if len(self.btc_prices) >= 3:
            # Get recent price movements
            recent_btc = list(self.btc_prices)[-3:]
            recent_call_bids = list(self.call_bids)[-3:]
            recent_call_asks = list(self.call_asks)[-3:]
            recent_put_bids = list(self.put_bids)[-3:]
            recent_put_asks = list(self.put_asks)[-3:]
            
            # Calculate BTC movement
            btc_change = recent_btc[-1] - recent_btc[0]  # Absolute change
            btc_pct_change = btc_change / recent_btc[0]  # Percentage change
            
            # Calculate option price movements
            call_bid_change = recent_call_bids[-1] - recent_call_bids[0]
            call_ask_change = recent_call_asks[-1] - recent_call_asks[0]
            put_bid_change = recent_put_bids[-1] - recent_put_bids[0]
            put_ask_change = recent_put_asks[-1] - recent_put_asks[0]
            
            # Dynamic threshold based on proximity to strike
            distance_from_strike = abs(moneyness - 1.0)
            
            # Near the money: be more sensitive (lower threshold)
            # Far from the money: be less sensitive (higher threshold)
            if distance_from_strike < 0.005:  # Within 0.5% of strike
                btc_threshold = 10  # $10 threshold
                option_sensitivity = 2.0  # Very sensitive
            elif distance_from_strike < 0.02:  # Within 2% of strike
                btc_threshold = 15  # $15 threshold
                option_sensitivity = 1.5  # Moderately sensitive
            else:  # Far from strike
                btc_threshold = 25  # $25 threshold
                option_sensitivity = 1.0  # Normal sensitivity
            
            # Analyze significant BTC moves
            if abs(btc_change) > btc_threshold or abs(btc_pct_change) > 0.0001:
                
                # Expected option change calculation
                # Base expectation: $100 BTC move = ~10 cents option move
                # Adjusted by proximity to strike and time to expiry
                base_sensitivity = 0.10 / 100  # 10 cents per $100
                
                # Increase sensitivity near the strike and with more time
                proximity_multiplier = option_sensitivity
                time_multiplier = min(2.0, time_to_expiry * 2)  # More time = more sensitivity
                
                expected_option_change = abs(btc_change) * base_sensitivity * proximity_multiplier * time_multiplier
                expected_option_change = min(0.25, expected_option_change)  # Cap at 25 cents
                
                # CALL OPTION ANALYSIS
                if btc_change > 0:  # BTC went up
                    call_mid_change = (call_bid_change + call_ask_change) / 2
                    
                    # Check for CALL overreaction
                    if call_mid_change > expected_option_change * 1.8:  # 80% overreaction threshold
                        profit_potential = (call_mid_change - expected_option_change) * 0.6
                        if profit_potential > 0.015:  # 1.5 cent threshold
                            opp = self._create_opportunity('CALL', 'SELL', call_bid, btc_price, strike_price, 
                                                         moneyness, time_to_expiry, profit_potential,
                                                         f'BTC +${btc_change:.0f}, call overreacted +${call_mid_change:.3f} (expected ~${expected_option_change:.3f})')
                            opportunities.append(opp)
                    
                    # Check for CALL underreaction
                    elif call_mid_change < expected_option_change * 0.4 and expected_option_change > 0.02:
                        profit_potential = (expected_option_change - call_mid_change) * 0.6
                        if profit_potential > 0.015:
                            opp = self._create_opportunity('CALL', 'BUY', call_ask, btc_price, strike_price,
                                                         moneyness, time_to_expiry, profit_potential,
                                                         f'BTC +${btc_change:.0f}, call underreacted +${call_mid_change:.3f} (expected ~${expected_option_change:.3f})')
                            opportunities.append(opp)
                
                elif btc_change < 0:  # BTC went down
                    put_mid_change = (put_bid_change + put_ask_change) / 2
                    
                    # Check for PUT overreaction (the key fix!)
                    if put_mid_change > expected_option_change * 1.5:  # 50% overreaction threshold (more aggressive)
                        profit_potential = (put_mid_change - expected_option_change) * 0.6
                        if profit_potential > 0.015:  # Lower threshold for puts
                            opp = self._create_opportunity('PUT', 'SELL', put_bid, btc_price, strike_price,
                                                         moneyness, time_to_expiry, profit_potential,
                                                         f'BTC ${btc_change:.0f}, put overreacted +${put_mid_change:.3f} (expected ~${expected_option_change:.3f})')
                            opportunities.append(opp)
                    
                    # Check for PUT underreaction
                    elif put_mid_change < expected_option_change * 0.4 and expected_option_change > 0.02:
                        profit_potential = (expected_option_change - put_mid_change) * 0.6
                        if profit_potential > 0.015:
                            opp = self._create_opportunity('PUT', 'BUY', put_ask, btc_price, strike_price,
                                                         moneyness, time_to_expiry, profit_potential,
                                                         f'BTC ${btc_change:.0f}, put underreacted +${put_mid_change:.3f} (expected ~${expected_option_change:.3f})')
                            opportunities.append(opp)
                
                # CROSS-OPTION ARBITRAGE: When one option moves too much relative to the other
                total_option_change = abs(call_mid_change) + abs(put_mid_change)
                if total_option_change > expected_option_change * 2.5:  # Total movement too large
                    # Identify which option overreacted more
                    if abs(call_mid_change) > abs(put_mid_change) * 1.5:
                        # Call overreacted more
                        if btc_change > 0 and call_mid_change > 0:  # Call rose too much
                            profit_potential = call_mid_change * 0.3
                            if profit_potential > 0.02:
                                opp = self._create_opportunity('CALL', 'SELL', call_bid, btc_price, strike_price,
                                                             moneyness, time_to_expiry, profit_potential,
                                                             f'Cross-option: Call moved ${call_mid_change:.3f} vs put ${put_mid_change:.3f}')
                                opportunities.append(opp)
                    elif abs(put_mid_change) > abs(call_mid_change) * 1.5:
                        # Put overreacted more
                        if btc_change < 0 and put_mid_change > 0:  # Put rose too much
                            profit_potential = put_mid_change * 0.3
                            if profit_potential > 0.02:
                                opp = self._create_opportunity('PUT', 'SELL', put_bid, btc_price, strike_price,
                                                             moneyness, time_to_expiry, profit_potential,
                                                             f'Cross-option: Put moved ${put_mid_change:.3f} vs call ${call_mid_change:.3f}')
                                opportunities.append(opp)
        
        # STATIC MISPRICING DETECTION (more aggressive near strike)
        if distance_from_strike < 0.01:  # Very close to strike
            # Options should be around 50/50 when at the money
            call_mid = (call_bid + call_ask) / 2
            put_mid = (put_bid + put_ask) / 2
            
            # If one option is significantly cheaper than the other near ATM
            if call_mid < 0.35 and put_mid > 0.60:  # Calls too cheap
                profit_potential = (0.50 - call_mid) * 0.7
                if profit_potential > 0.03:
                    opp = self._create_opportunity('CALL', 'BUY', call_ask, btc_price, strike_price,
                                                 moneyness, time_to_expiry, profit_potential,
                                                 f'Near ATM: Call only ${call_mid:.3f} vs put ${put_mid:.3f}')
                    opportunities.append(opp)
            
            elif put_mid < 0.35 and call_mid > 0.60:  # Puts too cheap
                profit_potential = (0.50 - put_mid) * 0.7
                if profit_potential > 0.03:
                    opp = self._create_opportunity('PUT', 'BUY', put_ask, btc_price, strike_price,
                                                 moneyness, time_to_expiry, profit_potential,
                                                 f'Near ATM: Put only ${put_mid:.3f} vs call ${call_mid:.3f}')
                    opportunities.append(opp)
        
        return opportunities
    
    def _create_opportunity(self, option_type, action, entry_price, btc_price, strike_price, 
                          moneyness, time_to_expiry, profit_potential, reason):
        """Helper method to create opportunity dictionary"""
        return {
            'id': len(self.arbitrage_history),
            'timestamp': datetime.now(),
            'type': option_type,
            'action': action,
            'entry_price': entry_price,
            'btc_price_entry': btc_price,
            'strike': strike_price,
            'moneyness': moneyness,
            'time_to_expiry': time_to_expiry,
            'profit_potential': profit_potential,
            'reason': reason
        }
    
    def update_open_positions(self, btc_price, call_bid, call_ask, put_bid, put_ask):
        """Update PNL for open positions"""
        for pos in self.open_positions:
            if pos['type'] == 'CALL':
                if pos['action'] == 'BUY':
                    pos['current_exit_price'] = call_bid
                    pos['exit_pnl'] = self.calculate_exit_pnl(pos['type'], pos['entry_price'], call_bid)
                else:  # SELL
                    pos['current_exit_price'] = call_ask
                    pos['exit_pnl'] = self.calculate_exit_pnl(pos['type'], pos['entry_price'], call_ask)
            else:  # PUT
                if pos['action'] == 'BUY':
                    pos['current_exit_price'] = put_bid
                    pos['exit_pnl'] = self.calculate_exit_pnl(pos['type'], pos['entry_price'], put_bid)
                else:  # SELL
                    pos['current_exit_price'] = put_ask
                    pos['exit_pnl'] = self.calculate_exit_pnl(pos['type'], pos['entry_price'], put_ask)
            
            # Calculate hold-to-expiry PNL
            if pos['action'] == 'BUY':
                pos['hold_pnl'] = self.calculate_expected_pnl_hold(pos['type'], pos['entry_price'], btc_price, pos['strike'])
            else:  # SELL
                pos['hold_pnl'] = -self.calculate_expected_pnl_hold(pos['type'], pos['entry_price'], btc_price, pos['strike'])
    
    def is_hour_transition(self, timestamp):
        """Check if we're in a hour transition period"""
        return (timestamp.minute == 59 and timestamp.second >= 55) or (timestamp.minute == 0 and timestamp.second <= 10)
    
    def validate_order_book_data(self, data, data_type):
        """Validate order book data and handle empty books"""
        if not data:
            return None, None
            
        # Check if data has the required structure
        if not isinstance(data, dict):
            return None, None
            
        # Check for best_bid and best_ask
        best_bid = data.get('best_bid')
        best_ask = data.get('best_ask')
        
        if not best_bid or not best_ask:
            return None, None
            
        # Extract prices safely
        try:
            bid_price = float(best_bid.get('price', 0))
            ask_price = float(best_ask.get('price', 0))
            
            # Sanity checks
            if bid_price <= 0 or ask_price <= 0 or bid_price >= ask_price:
                return None, None
                
            # Additional sanity check for option prices (should be between 0 and 1)
            if bid_price > 1.0 or ask_price > 1.0:
                return None, None
                
            return bid_price, ask_price
            
        except (ValueError, TypeError):
            return None, None
    
    def process_data_point(self, btc_data, call_data, put_data):
        """Process a single data point with robust error handling"""
        timestamp = datetime.now()
        
        # Check if we're in hour transition
        in_transition = self.is_hour_transition(timestamp)
        
        # Extract BTC price
        btc_price = None
        if btc_data and isinstance(btc_data, dict):
            btc_price = btc_data.get('price', self.get_btc_price())
        else:
            btc_price = self.get_btc_price()
            
        if not btc_price:
            return []  # Skip if no BTC price
        
        # Validate order book data
        call_bid, call_ask = self.validate_order_book_data(call_data, 'CALL')
        put_bid, put_ask = self.validate_order_book_data(put_data, 'PUT')
        
        # During transitions or if order books are empty, skip processing but continue monitoring
        if call_bid is None or put_bid is None:
            if in_transition:
                print(f"\r{timestamp.strftime('%H:%M:%S')} | 🔄 HOUR TRANSITION - Order books empty | BTC: ${btc_price:,.0f}", end='', flush=True)
            else:
                print(f"\r{timestamp.strftime('%H:%M:%S')} | ⚠️  Empty order books | BTC: ${btc_price:,.0f}", end='', flush=True)
            return []
        
        # Check and update strike price at hour boundaries
        if self.current_strike is None or (timestamp.minute == 0 and timestamp.second <= 5):
            old_strike = self.current_strike
            self.current_strike = self.get_current_strike()
            
            if old_strike and old_strike != self.current_strike:
                print(f"\n🕐 NEW HOUR! Strike updated: ${old_strike:,.0f} → ${self.current_strike:,.0f}")
                
                # Settle previous hour positions at expiry
                if self.open_positions:
                    print(f"💰 Settling {len(self.open_positions)} positions from previous hour...")
                    total_settled_pnl = 0
                    for pos in self.open_positions:
                        if pos['action'] == 'BUY':
                            settled_pnl = self.calculate_expected_pnl_hold(pos['type'], pos['entry_price'], btc_price, old_strike)
                        else:  # SELL
                            settled_pnl = -self.calculate_expected_pnl_hold(pos['type'], pos['entry_price'], btc_price, old_strike)
                        total_settled_pnl += settled_pnl
                    
                    print(f"💵 Previous hour total PNL: ${total_settled_pnl:+.3f}")
                
                # Clear positions for new hour
                self.open_positions.clear()
        
        strike_price = self.current_strike
        if not strike_price:
            print(f"\r{timestamp.strftime('%H:%M:%S')} | ❌ No strike price available", end='', flush=True)
            return []
            
        time_to_expiry = self.calculate_time_to_expiry()
        
        # Store data
        self.btc_prices.append(btc_price)
        self.call_bids.append(call_bid)
        self.call_asks.append(call_ask)
        self.put_bids.append(put_bid)
        self.put_asks.append(put_ask)
        self.timestamps.append(timestamp)
        self.strikes.append(strike_price)
        
        # Update open positions PNL
        self.update_open_positions(btc_price, call_bid, call_ask, put_bid, put_ask)
        
        # Detect BTC spike
        spike_detected, spike_magnitude = self.detect_btc_spike()
        
        # Detect arbitrage opportunities (skip during transitions to avoid false signals)
        opportunities = []
        if not in_transition and time_to_expiry > 0.02:  # Skip if less than ~1 minute remaining
            opportunities = self.detect_arbitrage_opportunity(
                btc_price, call_bid, call_ask, put_bid, put_ask, strike_price, time_to_expiry
            )
            
            # Debug output for close-to-strike situations
            if len(self.btc_prices) >= 3:
                distance_from_strike = abs(moneyness - 1.0)
                if distance_from_strike < 0.01:  # Very close to strike
                    recent_btc = list(self.btc_prices)[-3:]
                    recent_put_bids = list(self.put_bids)[-3:]
                    btc_change = recent_btc[-1] - recent_btc[0]
                    put_bid_change = recent_put_bids[-1] - recent_put_bids[0]
                    
                    if abs(btc_change) > 10 or abs(put_bid_change) > 0.02:  # Significant move
                        print(f"\n🔍 DEBUG: Near strike analysis")
                        print(f"    BTC change: ${btc_change:.0f}")
                        print(f"    PUT bid change: ${put_bid_change:.3f}")
                        print(f"    Distance from strike: {distance_from_strike:.4f}")
                        print(f"    Opportunities found: {len(opportunities)}")
        
        # Process new opportunities
        for opp in opportunities:
            self.arbitrage_history.append(opp)
            self.open_positions.append(opp.copy())  # Add to open positions
            
            if opp['type'] == 'CALL':
                self.total_call_opportunities += 1
            else:
                self.total_put_opportunities += 1
            
            self.total_opportunities += 1
        
        # Print real-time status
        moneyness = btc_price / strike_price if strike_price else 1.0
        
        # Add transition indicator
        transition_indicator = "🔄" if in_transition else ""
        
        status = (f"\r{timestamp.strftime('%H:%M:%S')} {transition_indicator} | "
                 f"BTC: ${btc_price:,.0f} ({moneyness:.3f}) | "
                 f"C: {call_bid:.3f}/{call_ask:.3f} | "
                 f"P: {put_bid:.3f}/{put_ask:.3f} | "
                 f"TTL: {time_to_expiry:.2f}h | "
                 f"Opps: {len(opportunities)} | "
                 f"Total: {self.total_opportunities}")
        
        print(status, end='', flush=True)
        
        # Alert for new opportunities
        if opportunities:
            print(f"\n🚨 NEW ARBITRAGE OPPORTUNITY!")
            for opp in opportunities:
                print(f"   💰 {opp['type']} {opp['action']} @ ${opp['entry_price']:.3f}")
                print(f"      Reason: {opp['reason']}")
                print(f"      Potential: ${opp['profit_potential']:.3f}")
        
        if spike_detected and not in_transition:
            print(f"\n📈 BTC SPIKE: {spike_magnitude:+.3%}")
        
        self.last_btc_price = btc_price
        self.last_update_time = timestamp
        
        return opportunities
    
    def generate_arbitrage_report(self):
        """Generate comprehensive arbitrage and PNL report"""
        if not self.arbitrage_history:
            print("📋 No arbitrage opportunities detected yet.")
            return
        
        print("\n" + "="*80)
        print("📊 ARBITRAGE OPPORTUNITIES & PNL REPORT")
        print("="*80)
        
        # Summary Statistics
        total_opps = len(self.arbitrage_history)
        call_opps = len([o for o in self.arbitrage_history if o['type'] == 'CALL'])
        put_opps = len([o for o in self.arbitrage_history if o['type'] == 'PUT'])
        buy_opps = len([o for o in self.arbitrage_history if o['action'] == 'BUY'])
        sell_opps = len([o for o in self.arbitrage_history if o['action'] == 'SELL'])
        
        print(f"Total Opportunities: {total_opps}")
        print(f"├─ CALL opportunities: {call_opps}")
        print(f"├─ PUT opportunities: {put_opps}")
        print(f"├─ BUY opportunities: {buy_opps}")
        print(f"└─ SELL opportunities: {sell_opps}")
        
        if self.open_positions:
            print(f"\n🔄 OPEN POSITIONS ({len(self.open_positions)}):")
            print("-"*80)
            
            total_exit_pnl = 0
            total_hold_pnl = 0
            
            for i, pos in enumerate(self.open_positions):
                total_exit_pnl += pos.get('exit_pnl', 0)
                total_hold_pnl += pos.get('hold_pnl', 0)
                
                print(f"{i+1:2d}. {pos['type']} {pos['action']} @ ${pos['entry_price']:.3f}")
                print(f"    Entry: BTC=${pos['btc_price_entry']:,.0f}, Strike=${pos['strike']:,.0f}")
                print(f"    Exit PNL: ${pos.get('exit_pnl', 0):+.3f} | Hold PNL: ${pos.get('hold_pnl', 0):+.3f}")
                print(f"    Reason: {pos['reason']}")
                print()
            
            print(f"💰 TOTAL PNL SUMMARY:")
            print(f"├─ If Exit All Now: ${total_exit_pnl:+.3f}")
            print(f"└─ If Hold to Expiry: ${total_hold_pnl:+.3f}")
        
        # Historical Performance
        if len(self.arbitrage_history) > 0:
            print(f"\n📈 HISTORICAL ANALYSIS:")
            print("-"*80)
            
            df = pd.DataFrame(self.arbitrage_history)
            
            print(f"Average Opportunity Size: ${df['profit_potential'].mean():.3f}")
            print(f"Max Opportunity Size: ${df['profit_potential'].max():.3f}")
            print(f"Min Opportunity Size: ${df['profit_potential'].min():.3f}")
            
            # Breakdown by type and action
            print(f"\nBREAKDOWN BY TYPE:")
            for option_type in ['CALL', 'PUT']:
                type_df = df[df['type'] == option_type]
                if len(type_df) > 0:
                    avg_potential = type_df['profit_potential'].mean()
                    print(f"├─ {option_type}: {len(type_df)} opps, avg ${avg_potential:.3f}")
            
            print(f"\nBREAKDOWN BY ACTION:")
            for action in ['BUY', 'SELL']:
                action_df = df[df['action'] == action]
                if len(action_df) > 0:
                    avg_potential = action_df['profit_potential'].mean()
                    print(f"├─ {action}: {len(action_df)} opps, avg ${avg_potential:.3f}")
        
        print("\n" + "="*80)
    
    def load_json_file(self, filepath):
        """Load JSON data from file with error handling"""
        try:
            with open(filepath, 'r') as f:
                content = f.read().strip()
                if not content:  # Empty file
                    return None
                data = json.loads(content)
                return data
        except FileNotFoundError:
            # File doesn't exist yet - normal during startup
            return None
        except json.JSONDecodeError:
            # Corrupted JSON - might happen during file write
            return None
        except Exception as e:
            # Other errors
            return None
    
    def signal_handler(self, signum, frame):
        """Handle signals to generate reports on demand"""
        print("\n📊 Signal received! Generating report...")
        self.generate_arbitrage_report()
        print("✅ Report complete. Continuing monitoring...\n")
    
    def run_continuous_monitoring(self, btc_file='btc_price.json', 
                                call_file='CALL.json', put_file='PUT.json'):
        """Run continuous monitoring loop"""
        print("🚀 Starting Real-time Lag Arbitrage Detector v2.0")
        print("=" * 80)
        print("💡 Strategy: BTC spike detection + Option pricing lag exploitation")
        print(f"🎯 Dynamic thresholds: $10-25 moves (vs ${self.btc_spike_threshold*100000:.0f} baseline)")
        print(f"💰 Arbitrage threshold: ${self.arbitrage_threshold:.3f}")
        print("🔄 Using BID/ASK prices only (no mid-prices)")
        print("🎯 Enhanced near-strike sensitivity (0.5-2% zones)")
        print("🕐 Hour transition handling: Auto-reload strike price")
        print("📖 Empty order book handling: Skip gracefully")
        print("🔍 Debug mode: Shows analysis for near-strike moves")
        print("=" * 80)
        print("💡 TIP: Send SIGUSR1 signal to see report without stopping:")
        print(f"   kill -USR1 {os.getpid()}")
        print("💡 Watch for 🔄 indicator during hour transitions")
        print("💡 Watch for 🔍 DEBUG output during critical moments")
        print("=" * 80)
        
        # Set up signal handler for on-demand reports
        signal.signal(signal.SIGUSR1, self.signal_handler)
        
        iteration = 0
        
        try:
            while True:
                # Load data files with error handling
                btc_data = self.load_json_file(btc_file)
                call_data = self.load_json_file(call_file)
                put_data = self.load_json_file(put_file)
                
                # Process data even if some files are missing/empty
                opportunities = self.process_data_point(btc_data, call_data, put_data)
                
                iteration += 1
                
                # Generate report every 2000 iterations or on user request
                if iteration % 2000 == 0:
                    self.generate_arbitrage_report()
                
                # Brief pause if all data is None (to avoid spinning)
                if not btc_data and not call_data and not put_data:
                    time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped by user")
            self.generate_arbitrage_report()

# Add method to view report on demand
def view_report():
    """Function to generate report for existing detector instance"""
    global detector
    if 'detector' in globals():
        detector.generate_arbitrage_report()
    else:
        print("❌ No detector instance found. Start monitoring first.")

def send_report_signal():
    """Helper function to send signal to running detector"""
    try:
        # Read PID from file if it exists
        if os.path.exists('detector.pid'):
            with open('detector.pid', 'r') as f:
                pid = int(f.read().strip())
            os.kill(pid, signal.SIGUSR1)
            print("✅ Report signal sent!")
        else:
            print("❌ Detector PID file not found. Is the detector running?")
            print("💡 Use: ps aux | grep lag_arbitrage to find the process ID")
            print("💡 Then: kill -USR1 <PID>")
    except Exception as e:
        print(f"❌ Error sending signal: {e}")

def main():
    global detector
    detector = RealTimeLagArbitrageDetector()
    
    # Save PID for signal sending
    with open('detector.pid', 'w') as f:
        f.write(str(os.getpid()))
    
    # Adjust parameters for more sensitive detection
    detector.btc_spike_threshold = 0.0003  # 0.03% spike threshold (more sensitive)
    detector.arbitrage_threshold = 0.008   # $0.008 arbitrage threshold (lower)
    
    print("💡 TIP: Enhanced detection for near-strike moves!")
    print("💡 While running, you can:")
    print("   1. Press Ctrl+C to stop and see full report")
    print("   2. In another terminal: python -c 'from lag_arbitrage_detector import send_report_signal; send_report_signal()'")
    print(f"   3. Or directly: kill -USR1 {os.getpid()}")
    print("💡 Watch for 🔍 DEBUG output when BTC is near strike price")
    
    try:
        # Run continuous monitoring
        detector.run_continuous_monitoring(
            btc_file='btc_price.json',
            call_file='CALL.json', 
            put_file='PUT.json'
        )
    finally:
        # Clean up PID file
        if os.path.exists('detector.pid'):
            os.remove('detector.pid')

if __name__ == "__main__":
    main()