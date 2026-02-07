#!/usr/bin/env python3
"""
Debug Market Maker - Find Why No Trades Are Being Placed

This script will analyze:
1. Order book structure and data
2. Dynamic thresholds vs actual price movements
3. Risk limits and balances
4. Token IDs and file accessibility
5. Market making opportunities detection
"""

import os
import json
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime

def check_json_files():
    """Check the structure of our JSON files"""
    files = {
        'btc': '/home/ubuntu/013_2025_polymarket/btc_price.json',
        'call': '/home/ubuntu/013_2025_polymarket/CALL.json',
        'put': '/home/ubuntu/013_2025_polymarket/PUT.json'
    }
    
    print("🔍 CHECKING JSON FILE STRUCTURES")
    print("=" * 60)
    
    for name, path in files.items():
        print(f"\n📁 {name.upper()} FILE: {path}")
        
        if not os.path.exists(path):
            print(f"❌ File does not exist!")
            continue
            
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            print(f"✅ File loaded successfully")
            print(f"📊 Keys: {list(data.keys())}")
            
            # Show detailed structure
            for key, value in data.items():
                if isinstance(value, dict):
                    print(f"   📋 {key}: {list(value.keys())}")
                elif isinstance(value, list):
                    print(f"   📋 {key}: list with {len(value)} items")
                    if value:
                        print(f"      Sample: {value[0]}")
                else:
                    print(f"   📋 {key}: {value}")
                    
        except Exception as e:
            print(f"❌ Error reading file: {e}")

def check_binance_volatility():
    """Check Binance data and calculate current volatility"""
    print("\n🌐 CHECKING BINANCE VOLATILITY DATA")
    print("=" * 60)
    
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': 'BTCUSDT',
            'interval': '1m',
            'limit': 60
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Fetched {len(data)} klines from Binance")
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_timestamp', 'quote_volume', 'count', 'taker_buy_volume', 
            'taker_buy_quote_volume', 'ignore'
        ])
        
        # Convert price columns to float
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)
        
        # Calculate returns and volatility
        df['returns'] = df['close'].pct_change()
        current_vol = df['returns'].std()
        
        print(f"📈 Current BTC Price: ${df['close'].iloc[-1]:,.2f}")
        print(f"📊 1-Hour Volatility: {current_vol:.6f} ({current_vol:.2%})")
        
        # Calculate dynamic thresholds
        directional_threshold = current_vol * 1.5
        strong_threshold = current_vol * 2.5
        spike_threshold = current_vol * 3.0
        
        print(f"🎯 DYNAMIC THRESHOLDS:")
        print(f"   Directional: {directional_threshold:.6f} ({directional_threshold:.2%})")
        print(f"   Strong Signal: {strong_threshold:.6f} ({strong_threshold:.2%})")
        print(f"   Spike Detection: {spike_threshold:.6f} ({spike_threshold:.2%})")
        
        # Recent price movements
        recent_returns = df['returns'].tail(10).dropna()
        print(f"\n📉 RECENT PRICE MOVEMENTS (last 10 minutes):")
        for i, ret in enumerate(recent_returns):
            direction = "UP" if ret > directional_threshold else "DOWN" if ret < -directional_threshold else "NEUTRAL"
            print(f"   {i+1}: {ret:+.4%} [{direction}]")
        
        return current_vol, directional_threshold, strong_threshold
        
    except Exception as e:
        print(f"❌ Error fetching Binance data: {e}")
        return 0.001, 0.0015, 0.0025

def analyze_order_books():
    """Analyze order book structure and find trading opportunities"""
    print("\n📚 ANALYZING ORDER BOOKS")
    print("=" * 60)
    
    files = ['CALL.json', 'PUT.json']
    
    for filename in files:
        filepath = f'/home/ubuntu/013_2025_polymarket/{filename}'
        option_type = filename.replace('.json', '')
        
        print(f"\n📋 {option_type} ORDER BOOK:")
        
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            continue
            
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Check for order book data
            has_bids = 'bids' in data
            has_asks = 'asks' in data
            has_best_bid = 'best_bid' in data
            has_best_ask = 'best_ask' in data
            
            print(f"   Order Book Structure:")
            print(f"   ├─ Has 'bids' array: {'✅' if has_bids else '❌'}")
            print(f"   ├─ Has 'asks' array: {'✅' if has_asks else '❌'}")
            print(f"   ├─ Has 'best_bid': {'✅' if has_best_bid else '❌'}")
            print(f"   └─ Has 'best_ask': {'✅' if has_best_ask else '❌'}")
            
            # Show best bid/ask if available
            if has_best_bid and data['best_bid']:
                bid_price = data['best_bid'].get('price', 0)
                bid_size = data['best_bid'].get('size', 0)
                print(f"   💰 Best Bid: ${bid_price:.4f} x {bid_size}")
            
            if has_best_ask and data['best_ask']:
                ask_price = data['best_ask'].get('price', 0)
                ask_size = data['best_ask'].get('size', 0)
                print(f"   💰 Best Ask: ${ask_price:.4f} x {ask_size}")
            
            # Calculate spread
            if has_best_bid and has_best_ask and data['best_bid'] and data['best_ask']:
                bid_price = data['best_bid'].get('price', 0)
                ask_price = data['best_ask'].get('price', 0)
                if bid_price > 0 and ask_price > 0:
                    spread = ask_price - bid_price
                    mid_price = (bid_price + ask_price) / 2
                    spread_pct = spread / mid_price * 100
                    print(f"   📊 Spread: ${spread:.4f} ({spread_pct:.2f}%)")
            
            # Check for deep order book
            if has_bids and data['bids']:
                print(f"   📈 Bids in book: {len(data['bids'])}")
                large_bids = [bid for bid in data['bids'] if isinstance(bid, dict) and bid.get('size', 0) >= 1111]
                print(f"   🎯 Bids with size >= 1111: {len(large_bids)}")
                
                if large_bids:
                    for i, bid in enumerate(large_bids[:3]):
                        print(f"      {i+1}: ${bid['price']:.4f} x {bid['size']}")
            
            if has_asks and data['asks']:
                print(f"   📉 Asks in book: {len(data['asks'])}")
                large_asks = [ask for ask in data['asks'] if isinstance(ask, dict) and ask.get('size', 0) >= 1111]
                print(f"   🎯 Asks with size >= 1111: {len(large_asks)}")
                
                if large_asks:
                    for i, ask in enumerate(large_asks[:3]):
                        print(f"      {i+1}: ${ask['price']:.4f} x {ask['size']}")
            
        except Exception as e:
            print(f"❌ Error analyzing {option_type}: {e}")

def simulate_trading_logic(current_vol, directional_threshold, strong_threshold):
    """Simulate the trading logic to see why no trades are placed"""
    print("\n🎯 SIMULATING TRADING LOGIC")
    print("=" * 60)
    
    # Get current BTC price and simulate momentum
    try:
        with open('/home/ubuntu/013_2025_polymarket/btc_price.json', 'r') as f:
            btc_data = json.load(f)
        current_btc = btc_data['price']
        print(f"📈 Current BTC: ${current_btc:,.2f}")
    except:
        print("❌ Could not read BTC price")
        return
    
    # Simulate different momentum scenarios
    test_momentums = [0.0001, 0.001, 0.002, 0.005, 0.01]  # 0.01% to 1%
    
    print(f"\n🧮 MOMENTUM CLASSIFICATION TEST:")
    print(f"   Directional threshold: {directional_threshold:.4%}")
    print(f"   Strong signal threshold: {strong_threshold:.4%}")
    
    for momentum in test_momentums:
        if momentum > strong_threshold:
            direction = 'STRONG_UP'
        elif momentum > directional_threshold:
            direction = 'UP'
        else:
            direction = 'NEUTRAL'
        
        print(f"   {momentum:.2%} movement → {direction}")
    
    # Check risk limits
    print(f"\n⚖️ RISK LIMITS CHECK:")
    max_position = 25.0
    max_exposure = 100.0
    min_book_quantity = 1111
    
    print(f"   Max position size: ${max_position}")
    print(f"   Max total exposure: ${max_exposure}")
    print(f"   Min book quantity: {min_book_quantity}")
    print(f"   Current active orders: 0 (simulated)")
    print(f"   Current exposure: $0 (simulated)")
    print(f"   ✅ Risk limits would allow trading")

def check_token_ids():
    """Check if token IDs can be extracted"""
    print("\n🪙 CHECKING TOKEN IDs")
    print("=" * 60)
    
    files = {'CALL': 'CALL.json', 'PUT': 'PUT.json'}
    
    for option_type, filename in files.items():
        filepath = f'/home/ubuntu/013_2025_polymarket/{filename}'
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Look for token ID in various possible fields
            possible_fields = ['asset_id', 'token_id', 'id', 'tokenId']
            token_id = None
            
            for field in possible_fields:
                if field in data:
                    token_id = data[field]
                    print(f"✅ {option_type} token ID found in '{field}': {token_id}")
                    break
            
            if not token_id:
                print(f"❌ {option_type} token ID not found in any field")
                print(f"   Available fields: {list(data.keys())}")
        
        except Exception as e:
            print(f"❌ Error checking {option_type} token ID: {e}")

def main():
    """Main debugging function"""
    print("🐛 MARKET MAKER DEBUG ANALYSIS")
    print("=" * 80)
    print(f"⏰ Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Check JSON files
    check_json_files()
    
    # Step 2: Check Binance volatility
    current_vol, directional_threshold, strong_threshold = check_binance_volatility()
    
    # Step 3: Analyze order books
    analyze_order_books()
    
    # Step 4: Check token IDs
    check_token_ids()
    
    # Step 5: Simulate trading logic
    simulate_trading_logic(current_vol, directional_threshold, strong_threshold)
    
    # Summary and recommendations
    print("\n💡 RECOMMENDATIONS TO FIX TRADING ISSUES")
    print("=" * 80)
    
    print("1. 📚 ORDER BOOK STRUCTURE:")
    print("   - Verify that JSON files contain 'bids' and 'asks' arrays")
    print("   - Check if any orders have size >= 1111")
    print("   - Consider lowering min_book_quantity if no large orders exist")
    
    print("\n2. 🎯 THRESHOLDS:")
    print("   - Current dynamic thresholds might be too high")
    print("   - Consider using lower multipliers (1.0x, 1.5x instead of 1.5x, 2.5x)")
    print("   - Add fallback to fixed thresholds if volatility is too low")
    
    print("\n3. 🪙 TOKEN IDs:")
    print("   - Ensure token IDs are correctly extracted from JSON files")
    print("   - Verify the field name contains the actual token ID")
    
    print("\n4. 💰 BALANCES:")
    print("   - Check if you have sufficient token balances for SELL orders")
    print("   - Verify API connection is working for balance queries")
    
    print("\n5. 📊 IMMEDIATE FIXES:")
    print("   - Add more debug logging to see exact failure points")
    print("   - Lower the min_book_quantity from 1111 to 100 temporarily")
    print("   - Use fixed thresholds as backup (0.1%, 0.3%, 0.5%)")
    print("   - Test with market orders first to verify execution pipeline")
    
    print("\n🔧 NEXT STEPS:")
    print("1. Run this debug script to identify the main issue")
    print("2. Fix the highest priority issue first")
    print("3. Add debug prints to the main trading loop")
    print("4. Test with very small position sizes initially")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
