#!/usr/bin/env python3
"""
Polymarket PNL Calculator
Calculates profit/loss for each trading period (slug) for a specific user.
Only processes today's trades.
"""

import requests
from collections import defaultdict
from datetime import datetime, timezone
from typing import List, Dict, Any
import json


class PolymarketPNLCalculator:
    def __init__(self, user_address: str):
        self.user_address = user_address
        self.base_url = "https://data-api.polymarket.com/activity"
        self.today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ).timestamp()
        
    def is_today(self, timestamp: int) -> bool:
        """Check if timestamp is from today"""
        return timestamp >= self.today_start
    
    def fetch_all_today_trades(self) -> List[Dict[str, Any]]:
        """Fetch latest trades (up to 1000) and filter for today"""
        print("Fetching latest trades...")
        print("=" * 80)
        
        # Fetch up to 1000 latest trades in descending order
        params = {
            'limit': 1000,
            'sortBy': 'TIMESTAMP',
            'sortDirection': 'DESC',
            'user': self.user_address,
            'offset': 0
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Handle both list and dict responses
            if isinstance(data, list):
                all_trades = data
            elif isinstance(data, dict) and 'data' in data:
                all_trades = data['data']
            else:
                all_trades = []
            
            print(f"📥 Loaded {len(all_trades)} trades")
            
            # Analyze date distribution
            if all_trades:
                trades_by_day = defaultdict(int)
                for t in all_trades:
                    ts = t.get('timestamp', 0)
                    if ts:
                        day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d')
                        trades_by_day[day] += 1
                
                print(f"\n📅 Trades by Day:")
                for day in sorted(trades_by_day.keys(), reverse=True):
                    today_marker = " ⭐ TODAY" if datetime.fromtimestamp(self.today_start, tz=timezone.utc).strftime('%Y-%m-%d') == day else ""
                    print(f"   {day}: {trades_by_day[day]} trades{today_marker}")
            
            # Filter for today's trades
            today_trades = [t for t in all_trades if self.is_today(t.get('timestamp', 0))]
            
            print("\n" + "=" * 80)
            print(f"📊 TOTAL TRADES FROM TODAY: {len(today_trades)}")
            print("=" * 80 + "\n")
            
            return today_trades
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching data: {e}")
            return []
    
    def calculate_pnl_by_slug(self, trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Calculate PNL for each trading period (slug)"""
        
        # Group trades by slug
        trades_by_slug = defaultdict(list)
        for trade in trades:
            slug = trade.get('slug')
            if slug:
                trades_by_slug[slug].append(trade)
        
        # Calculate PNL for each slug
        pnl_results = {}
        
        for slug, slug_trades in trades_by_slug.items():
            # Sort by timestamp to get last trade
            sorted_trades = sorted(slug_trades, key=lambda x: x.get('timestamp', 0))
            
            pnl_data = {
                'slug': slug,
                'title': slug_trades[0].get('title', 'N/A'),
                'trades': [],
                'total_buys': 0,
                'total_sells': 0,
                'total_redeems': 0,
                'net_pnl': 0,
                'trade_count': len(slug_trades),
                # New fields for outcome-based calculation
                'final_outcome': None,
                'final_price': None,
                'outcome_based_pnl': 0,
                'outcome_based_trades': []
            }
            
            # Determine final outcome from last trade
            last_trade = sorted_trades[-1]
            last_price = last_trade.get('price', 0)
            last_outcome = last_trade.get('outcome', '')
            
            pnl_data['final_price'] = last_price
            
            # Determine which side won
            if last_price < 0.5:
                # This side lost (value = 0), opposite side won (value = 1)
                if last_outcome == 'Up':
                    pnl_data['final_outcome'] = 'Down Won (Up price < 0.5)'
                    winning_side = 'Down'
                else:
                    pnl_data['final_outcome'] = 'Up Won (Down price < 0.5)'
                    winning_side = 'Up'
            else:
                # This side won (value = 1)
                pnl_data['final_outcome'] = f'{last_outcome} Won (price > 0.5)'
                winning_side = last_outcome
            
            # Process trades for both calculations
            for trade in sorted_trades:
                trade_type = trade.get('type')
                size = trade.get('size', 0)
                usdc_size = trade.get('usdcSize', 0)
                side = trade.get('side', '')
                timestamp = trade.get('timestamp', 0)
                price = trade.get('price', 0)
                outcome = trade.get('outcome', '')
                
                trade_info = {
                    'timestamp': datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                    'type': trade_type,
                    'side': side,
                    'outcome': outcome,
                    'size': size,
                    'usdc_size': usdc_size,
                    'price': price
                }
                
                # Original PNL calculation
                if trade_type == 'TRADE':
                    if side == 'BUY':
                        pnl_data['total_buys'] += usdc_size
                        pnl_data['net_pnl'] -= usdc_size
                        trade_info['pnl_impact'] = -usdc_size
                    elif side == 'SELL':
                        pnl_data['total_sells'] += usdc_size
                        pnl_data['net_pnl'] += usdc_size
                        trade_info['pnl_impact'] = usdc_size
                        
                elif trade_type == 'REDEEM':
                    pnl_data['total_redeems'] += usdc_size
                    pnl_data['net_pnl'] += usdc_size
                    trade_info['pnl_impact'] = usdc_size
                
                # Outcome-based PNL calculation (IGNORE SELLS, only BUYs)
                if trade_type == 'TRADE' and side == 'BUY':
                    if outcome == winning_side:
                        # Side won: PNL = (1.0 - entry_price) * size
                        outcome_pnl = (1.0 - price) * size
                    else:
                        # Side lost: PNL = (0.0 - entry_price) * size = -entry_price * size
                        outcome_pnl = (0.0 - price) * size
                    
                    pnl_data['outcome_based_pnl'] += outcome_pnl
                    trade_info['outcome_based_pnl'] = outcome_pnl
                    
                    pnl_data['outcome_based_trades'].append({
                        'timestamp': trade_info['timestamp'],
                        'outcome': outcome,
                        'size': size,
                        'entry_price': price,
                        'won': outcome == winning_side,
                        'pnl': outcome_pnl
                    })
                elif trade_type == 'TRADE' and side == 'SELL':
                    trade_info['outcome_based_pnl'] = None  # Ignored in outcome-based calc
                
                pnl_data['trades'].append(trade_info)
            
            pnl_results[slug] = pnl_data
        
        return pnl_results
    
    def print_results(self, pnl_results: Dict[str, Dict[str, Any]]):
        """Print PNL results in a readable format"""
        
        print("\n" + "="*80)
        print(f"PNL SUMMARY FOR USER: {self.user_address}")
        print(f"DATE: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}")
        print("="*80)
        
        total_pnl = 0
        total_outcome_pnl = 0
        
        for slug, data in sorted(pnl_results.items(), key=lambda x: x[1]['slug']):
            print(f"\n{'─'*80}")
            print(f"PERIOD: {data['slug']}")
            print(f"TITLE: {data['title']}")
            print(f"{'─'*80}")
            print(f"📊 FINAL OUTCOME: {data['final_outcome']} (Last price: ${data['final_price']:.2f})")
            print(f"\n{'─'*40}")
            print(f"💰 ORIGINAL PNL CALCULATION (Buy/Sell/Redeem):")
            print(f"{'─'*40}")
            print(f"Trade Count: {data['trade_count']}")
            print(f"Total Buys:   ${data['total_buys']:.2f}")
            print(f"Total Sells:  ${data['total_sells']:.2f}")
            print(f"Total Redeems: ${data['total_redeems']:.2f}")
            print(f"NET PNL:      ${data['net_pnl']:.2f}")
            
            print(f"\n{'─'*40}")
            print(f"🎯 OUTCOME-BASED PNL (Only BUYs, based on final outcome):")
            print(f"{'─'*40}")
            print(f"NET PNL:      ${data['outcome_based_pnl']:.2f}")
            
            if data['outcome_based_trades']:
                print(f"\nOutcome-based trades breakdown:")
                for t in data['outcome_based_trades']:
                    status = "✅ WON" if t['won'] else "❌ LOST"
                    pnl_sign = "+" if t['pnl'] >= 0 else ""
                    print(f"  {t['timestamp']} | [{t['outcome']:4}] | {status} | "
                          f"Size: {t['size']:6.2f} | Entry: ${t['entry_price']:.2f} | "
                          f"PNL: {pnl_sign}${t['pnl']:.2f}")
            
            print(f"\n{'─'*40}")
            print(f"📝 DETAILED TRADES (All operations):")
            print(f"{'─'*40}")
            for trade in data['trades']:
                pnl_sign = "+" if trade.get('pnl_impact', 0) >= 0 else ""
                outcome_display = f"[{trade.get('outcome', 'N/A'):4}]"
                
                # Show outcome-based PNL for BUYs
                outcome_pnl_str = ""
                if trade.get('outcome_based_pnl') is not None:
                    ob_sign = "+" if trade['outcome_based_pnl'] >= 0 else ""
                    outcome_pnl_str = f" | Outcome PNL: {ob_sign}${trade['outcome_based_pnl']:.2f}"
                elif trade.get('side') == 'SELL':
                    outcome_pnl_str = " | Outcome PNL: (ignored)"
                
                print(f"  {trade['timestamp']} | {trade['type']:6} | "
                      f"{trade.get('side', 'N/A'):4} | {outcome_display} | "
                      f"Size: {trade['size']:6.2f} | "
                      f"Price: ${trade['price']:.2f} | USDC: ${trade['usdc_size']:.2f} | "
                      f"PNL Impact: {pnl_sign}${trade.get('pnl_impact', 0):.2f}{outcome_pnl_str}")
            
            total_pnl += data['net_pnl']
            total_outcome_pnl += data['outcome_based_pnl']
        
        print(f"\n{'='*80}")
        print(f"📊 TOTAL SUMMARY")
        print(f"{'='*80}")
        print(f"💰 Original PNL (Buy/Sell/Redeem):  ${total_pnl:.2f}")
        print(f"🎯 Outcome-based PNL (Only BUYs):   ${total_outcome_pnl:.2f}")
        print(f"{'='*80}\n")
        
        return total_pnl, total_outcome_pnl
    
    def save_to_file(self, pnl_results: Dict[str, Dict[str, Any]], total_pnl: float, total_outcome_pnl: float, filename: str = "pnl_results.json"):
        """Save results to a JSON file"""
        output = {
            'user_address': self.user_address,
            'date': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
            'periods': pnl_results,
            'total_original_pnl': total_pnl,
            'total_outcome_based_pnl': total_outcome_pnl
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Results saved to {filename}")


def main():
    import sys
    
    # User address
    user_address = "0x51A02132Af17252e6993C3b668EA462D46234a02"
    
    # Create calculator instance
    calculator = PolymarketPNLCalculator(user_address)
    
    # Check if a JSON file was provided as argument
    if len(sys.argv) > 1:
        print(f"Loading trades from file: {sys.argv[1]}")
        try:
            with open(sys.argv[1], 'r') as f:
                trades = json.load(f)
                
            # If the JSON has a 'data' key, use that
            if isinstance(trades, dict) and 'data' in trades:
                trades = trades['data']
                
            # Filter for today's trades
            trades = [t for t in trades if calculator.is_today(t.get('timestamp', 0))]
            print(f"Loaded {len(trades)} trades from today")
        except Exception as e:
            print(f"Error loading file: {e}")
            return
    else:
        # Fetch all today's trades from API
        trades = calculator.fetch_all_today_trades()
    
    if not trades:
        print("No trades found for today.")
        return
    
    # Calculate PNL by slug
    pnl_results = calculator.calculate_pnl_by_slug(trades)
    
    # Print results
    total_pnl, total_outcome_pnl = calculator.print_results(pnl_results)
    
    # Save to file
    calculator.save_to_file(pnl_results, total_pnl, total_outcome_pnl)


if __name__ == "__main__":
    main()
