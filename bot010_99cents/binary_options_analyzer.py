#!/usr/bin/env python3
"""
Binary Options Trading Strategy Analyzer
Analyzes OHLC data to find optimal buy/sell strategies for maximum PNL
"""

import json
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from collections import defaultdict
import itertools
from typing import Dict, List, Tuple, Any

class BinaryOptionsAnalyzer:
    def __init__(self, call_file_path: str, put_file_path: str):
        self.call_file_path = call_file_path
        self.put_file_path = put_file_path
        self.call_data = None
        self.put_data = None
        self.results = []
        
    def load_data(self):
        """Load and preprocess OHLC data from JSON files"""
        try:
            with open(self.call_file_path, 'r') as f:
                call_raw = json.load(f)
            with open(self.put_file_path, 'r') as f:
                put_raw = json.load(f)
                
            # Convert to DataFrames
            self.call_data = pd.DataFrame(call_raw)
            self.put_data = pd.DataFrame(put_raw)
            
            # Convert datetime strings to datetime objects
            for df in [self.call_data, self.put_data]:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df['hour'] = df['datetime'].dt.hour
                df['minute'] = df['datetime'].dt.minute
                df['date'] = df['datetime'].dt.date
                
            print(f"Loaded {len(self.call_data)} CALL records and {len(self.put_data)} PUT records")
            return True
            
        except FileNotFoundError as e:
            print(f"Error loading files: {e}")
            return False
    
    def process_hourly_sessions(self, data: pd.DataFrame) -> List[Dict]:
        """Process data into hourly sessions, handling expiry logic"""
        sessions = []
        
        # Group by date and hour
        for (date, hour), group in data.groupby(['date', 'hour']):
            session = {
                'date': date,
                'hour': hour,
                'data': group.sort_values('minute').copy(),
                'reset_price': 0.50,  # Price at minute 0
                'expiry_price': None,
                'expired': False
            }
            
            # Find the last minute with data
            last_minute = group['minute'].max()
            
            # If session ends before minute 59, it expired
            if last_minute < 59:
                session['expired'] = True
                last_price = group[group['minute'] == last_minute]['close'].iloc[0]
                # Binary option expires at 0 or 1 based on last valid price
                session['expiry_price'] = 1.0 if last_price > 0.5 else 0.0
            else:
                # Session completed normally
                session['expiry_price'] = group[group['minute'] == 59]['close'].iloc[0]
            
            sessions.append(session)
            
        return sessions
    
    def simulate_fixed_price_strategy(self, sessions: List[Dict], buy_price: float, 
                                    sell_price: float = None, option_type: str = 'CALL') -> Dict:
        """Simulate buying at fixed price and selling at fixed price or expiry"""
        trades = []
        total_pnl = 0
        winning_trades = 0
        losing_trades = 0
        
        for session in sessions:
            data = session['data']
            expiry_price = session['expiry_price']
            
            # Check if we can buy at the specified price
            buy_opportunities = data[
                (data['low'] <= buy_price) & (data['high'] >= buy_price)
            ]
            
            if len(buy_opportunities) == 0:
                continue
                
            # Take first buy opportunity
            buy_minute = buy_opportunities.iloc[0]['minute']
            
            # Determine sell scenario
            if sell_price is not None:
                # Look for sell opportunity after buy
                sell_opportunities = data[
                    (data['minute'] > buy_minute) & 
                    (data['low'] <= sell_price) & 
                    (data['high'] >= sell_price)
                ]
                
                if len(sell_opportunities) > 0:
                    # Sold before expiry
                    sell_minute = sell_opportunities.iloc[0]['minute']
                    pnl = sell_price - buy_price
                    trade_type = 'sold_early'
                else:
                    # Held until expiry
                    pnl = expiry_price - buy_price
                    trade_type = 'held_to_expiry'
            else:
                # Always hold to expiry
                pnl = expiry_price - buy_price
                trade_type = 'held_to_expiry'
            
            trades.append({
                'date': session['date'],
                'hour': session['hour'],
                'buy_price': buy_price,
                'sell_price': sell_price if sell_price and trade_type == 'sold_early' else expiry_price,
                'pnl': pnl,
                'trade_type': trade_type
            })
            
            total_pnl += pnl
            if pnl > 0:
                winning_trades += 1
            else:
                losing_trades += 1
        
        return {
            'strategy': f'{option_type} - Buy@{buy_price:.2f}, Sell@{sell_price if sell_price else "Expiry"}',
            'total_trades': len(trades),
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / len(trades) if trades else 0,
            'win_rate': winning_trades / len(trades) if trades else 0,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'trades': trades,
            'buy_price': buy_price,
            'sell_price': sell_price,
            'option_type': option_type
        }
    
    def simulate_threshold_strategy(self, sessions: List[Dict], threshold: float, 
                                  above_below: str, option_type: str = 'CALL') -> Dict:
        """Simulate buying above or below threshold strategies"""
        trades = []
        total_pnl = 0
        winning_trades = 0
        losing_trades = 0
        
        for session in sessions:
            data = session['data']
            expiry_price = session['expiry_price']
            
            # Find buy opportunities based on threshold
            if above_below == 'above':
                buy_opportunities = data[data['close'] > threshold]
            else:  # below
                buy_opportunities = data[data['close'] < threshold]
            
            if len(buy_opportunities) == 0:
                continue
            
            # Take first opportunity
            buy_row = buy_opportunities.iloc[0]
            buy_price = buy_row['close']
            
            # Hold to expiry
            pnl = expiry_price - buy_price
            
            trades.append({
                'date': session['date'],
                'hour': session['hour'],
                'buy_price': buy_price,
                'sell_price': expiry_price,
                'pnl': pnl,
                'threshold': threshold,
                'condition': above_below
            })
            
            total_pnl += pnl
            if pnl > 0:
                winning_trades += 1
            else:
                losing_trades += 1
        
        return {
            'strategy': f'{option_type} - Buy when price {above_below} {threshold:.2f}',
            'total_trades': len(trades),
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / len(trades) if trades else 0,
            'win_rate': winning_trades / len(trades) if trades else 0,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'trades': trades,
            'threshold': threshold,
            'condition': above_below,
            'option_type': option_type
        }
    
    def run_comprehensive_analysis(self, quick_mode=False):
        """Run all possible trading scenarios with progress feedback
        
        Args:
            quick_mode (bool): If True, use even fewer iterations for quick testing
        """
        if not self.load_data():
            return
        
        # Process sessions
        call_sessions = self.process_hourly_sessions(self.call_data)
        put_sessions = self.process_hourly_sessions(self.put_data)
        
        print(f"Processing {len(call_sessions)} CALL sessions and {len(put_sessions)} PUT sessions")
        
        # Test price ranges - REDUCED ITERATIONS
        if quick_mode:
            buy_prices = np.arange(0.05, 0.96, 0.05)  # every 0.05 for quick testing
            sell_prices = np.arange(0.05, 0.96, 0.05)  # every 0.05 for quick testing
            threshold_prices = np.arange(0.20, 0.81, 0.10)  # fewer thresholds
            print("QUICK MODE: Using reduced price ranges for faster testing")
        else:
            buy_prices = np.arange(0.01, 1.00, 0.02)  # every 0.02 instead of 0.01
            sell_prices = np.arange(0.01, 1.00, 0.02)  # every 0.02 instead of 0.01
            threshold_prices = np.arange(0.10, 0.90, 0.05)
        
        total_buy_prices = len(buy_prices)
        total_sell_prices = len(sell_prices)
        
        # Estimate total operations
        fixed_price_strategies = total_buy_prices * 2  # hold to expiry for CALL and PUT
        buy_sell_strategies = sum(1 for b in buy_prices for s in sell_prices if s > b) * 2  # profitable combos only
        threshold_strategies = len(threshold_prices) * 2 * 2  # 2 conditions x 2 option types
        total_strategies = fixed_price_strategies + buy_sell_strategies + threshold_strategies
        
        print(f"Analysis Plan:")
        print(f"• Buy prices: {total_buy_prices} (0.01 to 0.99, step {'0.05' if quick_mode else '0.02'})")
        print(f"• Sell prices: {total_sell_prices} (0.01 to 0.99, step {'0.05' if quick_mode else '0.02'})")
        print(f"• Hold-to-expiry strategies: {fixed_price_strategies}")
        print(f"• Buy-sell combinations: {buy_sell_strategies}")
        print(f"• Threshold strategies: {threshold_strategies}")
        print(f"• Total strategies to test: ~{total_strategies:,}")
        print(f"• Estimated time: {total_strategies/1000:.1f}-{total_strategies/500:.1f} minutes")
        print("=" * 60)
        
        current_combo = 0
        start_time = time.time()
        strategies_found = 0
        
        print("Running CALL option strategies...")
        
        # Test all buy/sell combinations for CALL options
        for i, buy_price in enumerate(buy_prices):
            # Buy and hold to expiry
            result = self.simulate_fixed_price_strategy(call_sessions, buy_price, None, 'CALL')
            if result['total_trades'] > 0:
                self.results.append(result)
                strategies_found += 1
            current_combo += 1
            
            # Buy and sell at various prices
            for sell_price in sell_prices:
                if sell_price > buy_price:  # Only sell for profit
                    result = self.simulate_fixed_price_strategy(call_sessions, buy_price, sell_price, 'CALL')
                    if result['total_trades'] > 0:
                        self.results.append(result)
                        strategies_found += 1
                current_combo += 1
                
                # Progress update every 50 combinations
                if current_combo % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = current_combo / elapsed if elapsed > 0 else 0
                    estimated_total = total_strategies / rate if rate > 0 else 0
                    remaining = estimated_total - elapsed if estimated_total > elapsed else 0
                    print(f"CALL Progress: {current_combo:4d} tested, {strategies_found:3d} valid strategies found. "
                          f"Rate: {rate:.1f}/s, ETA: {remaining/60:.1f}min")
        
        call_strategies = len([r for r in self.results if r['option_type'] == 'CALL'])
        print(f"\nCALL strategies completed: {call_strategies} strategies found")
        
        print("Running PUT option strategies...")
        put_start = current_combo
        
        # Test all buy/sell combinations for PUT options
        for i, buy_price in enumerate(buy_prices):
            # Buy and hold to expiry
            result = self.simulate_fixed_price_strategy(put_sessions, buy_price, None, 'PUT')
            if result['total_trades'] > 0:
                self.results.append(result)
                strategies_found += 1
            current_combo += 1
            
            # Buy and sell at various prices
            for sell_price in sell_prices:
                if sell_price > buy_price:  # Only sell for profit
                    result = self.simulate_fixed_price_strategy(put_sessions, buy_price, sell_price, 'PUT')
                    if result['total_trades'] > 0:
                        self.results.append(result)
                        strategies_found += 1
                current_combo += 1
                
                # Progress update every 50 combinations
                if (current_combo - put_start) % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = current_combo / elapsed if elapsed > 0 else 0
                    estimated_total = total_strategies / rate if rate > 0 else 0
                    remaining = estimated_total - elapsed if estimated_total > elapsed else 0
                    print(f"PUT Progress: {current_combo:4d} tested, {strategies_found:3d} valid strategies found. "
                          f"Rate: {rate:.1f}/s, ETA: {remaining/60:.1f}min")
        
        put_strategies = len([r for r in self.results if r['option_type'] == 'PUT'])
        print(f"\nPUT strategies completed: {put_strategies} strategies found")
        
        print("Running threshold strategies...")
        threshold_start = current_combo
        
        # Test threshold strategies
        for threshold in threshold_prices:
            for condition in ['above', 'below']:
                for option_type, sessions in [('CALL', call_sessions), ('PUT', put_sessions)]:
                    result = self.simulate_threshold_strategy(sessions, threshold, condition, option_type)
                    if result['total_trades'] > 0:
                        self.results.append(result)
                        strategies_found += 1
                    current_combo += 1
                    
            threshold_progress = current_combo - threshold_start
            print(f"Threshold progress: {threshold_progress:2d}/{threshold_strategies} - Testing {threshold:.2f}", end='\r')
        
        total_time = time.time() - start_time
        print(f"\n\nAnalysis completed in {total_time:.1f} seconds!")
        print(f"Tested {current_combo:,} strategy combinations")
        print(f"Found {strategies_found:,} valid strategies with trades")
        print(f"Average time per test: {(total_time/current_combo)*1000:.2f}ms")
        print(f"Success rate: {(strategies_found/current_combo)*100:.1f}%")
    
    def get_top_strategies(self, n: int = 10, sort_by: str = 'total_pnl') -> List[Dict]:
        """Get top N strategies sorted by specified metric"""
        sorted_results = sorted(self.results, key=lambda x: x[sort_by], reverse=True)
        return sorted_results[:n]
    
    def print_strategy_analysis(self):
        """Print comprehensive analysis results"""
        if not self.results:
            print("No results to analyze. Run comprehensive analysis first.")
            return
        
        print("\n" + "="*80)
        print("BINARY OPTIONS TRADING STRATEGY ANALYSIS")
        print("="*80)
        
        # Top 10 by total PNL
        print(f"\nTOP 10 STRATEGIES BY TOTAL PNL:")
        print("-"*80)
        top_pnl = self.get_top_strategies(10, 'total_pnl')
        for i, strategy in enumerate(top_pnl, 1):
            print(f"{i:2d}. {strategy['strategy']}")
            print(f"    Total PNL: {strategy['total_pnl']:.3f} | Avg PNL: {strategy['avg_pnl']:.3f}")
            print(f"    Trades: {strategy['total_trades']} | Win Rate: {strategy['win_rate']:.1%}")
            print(f"    Wins: {strategy['winning_trades']} | Losses: {strategy['losing_trades']}")
            print()
        
        # Top 10 by average PNL
        print(f"\nTOP 10 STRATEGIES BY AVERAGE PNL:")
        print("-"*80)
        top_avg = self.get_top_strategies(10, 'avg_pnl')
        for i, strategy in enumerate(top_avg, 1):
            print(f"{i:2d}. {strategy['strategy']}")
            print(f"    Total PNL: {strategy['total_pnl']:.3f} | Avg PNL: {strategy['avg_pnl']:.3f}")
            print(f"    Trades: {strategy['total_trades']} | Win Rate: {strategy['win_rate']:.1%}")
            print()
        
        # Top 10 by win rate (minimum 10 trades)
        print(f"\nTOP 10 STRATEGIES BY WIN RATE (min 10 trades):")
        print("-"*80)
        high_volume = [s for s in self.results if s['total_trades'] >= 10]
        top_winrate = sorted(high_volume, key=lambda x: x['win_rate'], reverse=True)[:10]
        for i, strategy in enumerate(top_winrate, 1):
            print(f"{i:2d}. {strategy['strategy']}")
            print(f"    Win Rate: {strategy['win_rate']:.1%} | Total PNL: {strategy['total_pnl']:.3f}")
            print(f"    Trades: {strategy['total_trades']} | Avg PNL: {strategy['avg_pnl']:.3f}")
            print()
    
    def suggest_additional_analyses(self):
        """Suggest additional analyses for PNL optimization"""
        print("\n" + "="*80)
        print("SUGGESTED ADDITIONAL ANALYSES")
        print("="*80)
        
        suggestions = [
            "1. TIME-BASED ANALYSIS:",
            "   • Performance by hour of day",
            "   • Performance by day of week", 
            "   • Seasonal patterns",
            "   • Market opening/closing effects",
            "",
            "2. VOLATILITY ANALYSIS:",
            "   • Buy during high volatility periods (high-low spread)",
            "   • Momentum strategies (price direction)",
            "   • Mean reversion strategies",
            "",
            "3. TECHNICAL INDICATORS:",
            "   • Moving averages (buy above/below MA)",
            "   • RSI-based entries",
            "   • Bollinger Bands strategies",
            "   • Volume-weighted strategies",
            "",
            "4. RISK MANAGEMENT:",
            "   • Maximum drawdown analysis",
            "   • Position sizing optimization",
            "   • Stop-loss strategies",
            "   • Portfolio correlation (CALL vs PUT)",
            "",
            "5. ADVANCED STRATEGIES:",
            "   • Delta hedging with CALL/PUT combinations",
            "   • Arbitrage opportunities",
            "   • Market maker strategies",
            "   • Machine learning pattern recognition",
            "",
            "6. MARKET MICROSTRUCTURE:",
            "   • Bid-ask spread analysis",
            "   • Order book depth impact",
            "   • Liquidity patterns",
            "   • Slippage modeling"
        ]
        
        for suggestion in suggestions:
            print(suggestion)
    
    def save_results_to_json(self, filename: str = "trading_analysis_results.json"):
        """Save analysis results to JSON file"""
        output = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_strategies_tested': len(self.results),
            'top_10_by_pnl': self.get_top_strategies(10, 'total_pnl'),
            'top_10_by_avg_pnl': self.get_top_strategies(10, 'avg_pnl'),
            'all_results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"\nResults saved to {filename}")

def main():
    # File paths
    call_file = "/home/ubuntu/013_2025_polymarket/bot010_99cents/data/CALL_OHLC.json"
    put_file = "/home/ubuntu/013_2025_polymarket/bot010_99cents/data/PUT_OHLC.json"
    
    # Initialize analyzer
    analyzer = BinaryOptionsAnalyzer(call_file, put_file)
    
    # Ask user for mode selection
    print("Binary Options Trading Strategy Analyzer")
    print("=" * 50)
    print("Select analysis mode:")
    print("1. Quick mode (5-minute test with reduced iterations)")
    print("2. Full mode (comprehensive analysis)")
    
    try:
        choice = input("Enter choice (1 or 2, default=1): ").strip()
        quick_mode = choice != "2"
    except:
        quick_mode = True
    
    mode_text = "QUICK" if quick_mode else "FULL"
    print(f"\nStarting {mode_text} binary options analysis...")
    
    # Run comprehensive analysis
    analyzer.run_comprehensive_analysis(quick_mode=quick_mode)
    
    # Print results
    analyzer.print_strategy_analysis()
    
    # Print suggestions
    analyzer.suggest_additional_analyses()
    
    # Save results
    filename = f"trading_analysis_results_{'quick' if quick_mode else 'full'}.json"
    analyzer.save_results_to_json(filename)

if __name__ == "__main__":
    main()
