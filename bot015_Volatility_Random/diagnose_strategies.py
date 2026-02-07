import json
import sys
from datetime import datetime
from typing import Dict, Optional

def read_file(filepath: str) -> Optional[Dict]:
    """Read a JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Error decoding {filepath}: {e}")
        return None
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return None

def diagnose_strategies():
    """Diagnose why each strategy isn't opening positions"""
    
    print("="*80)
    print("STRATEGY DIAGNOSTIC TOOL")
    print("="*80)
    print(f"Diagnostic Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Data files
    data_files = {
        'BTC_CALL': '/home/ubuntu/013_2025_polymarket/15M_CALL.json',
        'BTC_PUT': '/home/ubuntu/013_2025_polymarket/15M_PUT.json',
    }
    
    oscillation_file = '/home/ubuntu/013_2025_polymarket/bot004_blackScholes/data/latest_oscillation.json'
    
    # Read market data
    print("1. CHECKING DATA FILES")
    print("-" * 80)
    
    data = {}
    for asset_name, filepath in data_files.items():
        file_data = read_file(filepath)
        
        if file_data is None:
            data[asset_name] = None
            print(f"❌ {asset_name}: FILE NOT FOUND")
        else:
            print(f"✅ {asset_name}: File loaded")
            
            # Normalize the structure - check both formats
            normalized = {}
            
            # Check for best_bid/best_ask structure (new format)
            if 'best_bid' in file_data and file_data['best_bid']:
                normalized['bid'] = file_data['best_bid'].get('price')
            else:
                # Fallback to direct bid field (old format)
                normalized['bid'] = file_data.get('bid')
            
            if 'best_ask' in file_data and file_data['best_ask']:
                normalized['ask'] = file_data['best_ask'].get('price')
            else:
                # Fallback to direct ask field (old format)
                normalized['ask'] = file_data.get('ask')
            
            data[asset_name] = normalized
            
            # Check for required fields
            bid_value = normalized.get('bid')
            ask_value = normalized.get('ask')
            
            print(f"   - Bid price: {bid_value}")
            print(f"   - Ask price: {ask_value}")
            
            if ask_value is None:
                print(f"   ⚠️  ASK PRICE IS NULL - CANNOT ENTER POSITIONS")
            if bid_value is None:
                print(f"   ⚠️  BID PRICE IS NULL - CANNOT EXIT POSITIONS")
    
    print()
    
    # Read oscillation data
    print("2. CHECKING OSCILLATION DATA")
    print("-" * 80)
    
    oscillation_data = read_file(oscillation_file)
    
    if oscillation_data is None:
        print(f"❌ Oscillation file not found or invalid")
    else:
        print(f"✅ Oscillation file loaded")
        
        # Check for required fields
        if 'oscillation_60min' in oscillation_data:
            osc_60 = oscillation_data['oscillation_60min']
            choppiness = osc_60.get('choppiness_index')
            print(f"   - 60min Choppiness Index: {choppiness}")
            
            if choppiness is None:
                print(f"   ⚠️  CHOPPINESS INDEX IS NULL")
        else:
            print(f"   ⚠️  'oscillation_60min' field missing")
            choppiness = None
    
    print()
    
    # Check existing positions
    print("3. CHECKING EXISTING POSITIONS")
    print("-" * 80)
    
    date_str = datetime.now().strftime('%Y%m%d')
    results_file = f'trading_results_dynamic_{date_str}.json'
    
    results = read_file(results_file)
    
    open_positions = []
    if results and 'open_positions_detail' in results:
        open_positions = results['open_positions_detail']
        print(f"✅ Found {len(open_positions)} open positions")
        
        for pos in open_positions:
            print(f"   - Strategy {pos['strategy']}: {pos['asset_type']} @ ${pos['entry_price']:.4f}")
    else:
        print(f"ℹ️  No existing results file or no open positions")
    
    print()
    
    # Now diagnose each strategy
    print("4. STRATEGY-BY-STRATEGY DIAGNOSIS")
    print("="*80)
    
    # Strategy A
    print("\n📊 STRATEGY A - BTC_CALL only, TP +$0.06, SL -$0.09")
    print("-" * 80)
    
    has_strategy_a = any(p['strategy'] == 'A' and p['asset_type'] == 'BTC_CALL' for p in open_positions)
    
    if has_strategy_a:
        print("❌ ALREADY HAS OPEN POSITION")
        print("   Reason: Strategy A already has an active BTC_CALL position")
    else:
        btc_call_data = data.get('BTC_CALL')
        
        if btc_call_data is None:
            print("❌ CANNOT OPEN POSITION")
            print("   Reason: BTC_CALL data file not found or invalid")
        elif 'ask' not in btc_call_data:
            print("❌ CANNOT OPEN POSITION")
            print("   Reason: 'ask' field missing in BTC_CALL data")
        elif btc_call_data['ask'] is None:
            print("❌ CANNOT OPEN POSITION")
            print("   Reason: BTC_CALL ask price is NULL")
        else:
            ask_price = btc_call_data['ask']
            tp = ask_price + 0.06
            sl = ask_price - 0.09
            print("✅ READY TO OPEN POSITION")
            print(f"   Entry: ${ask_price:.4f}")
            print(f"   TP: ${tp:.4f} (+$0.06)")
            print(f"   SL: ${sl:.4f} (-$0.09)")
    
    # Strategy A2
    print("\n📊 STRATEGY A2 - BTC_CALL when choppiness > 30")
    print("-" * 80)
    
    has_strategy_a2 = any(p['strategy'] == 'A2' and p['asset_type'] == 'BTC_CALL' for p in open_positions)
    
    if has_strategy_a2:
        print("❌ ALREADY HAS OPEN POSITION")
        print("   Reason: Strategy A2 already has an active BTC_CALL position")
    else:
        btc_call_data = data.get('BTC_CALL')
        
        if btc_call_data is None:
            print("❌ CANNOT OPEN POSITION")
            print("   Reason: BTC_CALL data file not found or invalid")
        elif 'ask' not in btc_call_data:
            print("❌ CANNOT OPEN POSITION")
            print("   Reason: 'ask' field missing in BTC_CALL data")
        elif btc_call_data['ask'] is None:
            print("❌ CANNOT OPEN POSITION")
            print("   Reason: BTC_CALL ask price is NULL")
        else:
            ask_price = btc_call_data['ask']
            
            # Check choppiness condition
            if oscillation_data is None:
                print("❌ CANNOT OPEN POSITION")
                print("   Reason: Oscillation data not available")
            elif 'oscillation_60min' not in oscillation_data:
                print("❌ CANNOT OPEN POSITION")
                print("   Reason: 'oscillation_60min' field missing")
            elif oscillation_data['oscillation_60min'].get('choppiness_index') is None:
                print("❌ CANNOT OPEN POSITION")
                print("   Reason: Choppiness index is NULL")
            else:
                choppiness = oscillation_data['oscillation_60min']['choppiness_index']
                
                if choppiness <= 30:
                    print("❌ CONDITION NOT MET")
                    print(f"   Current Choppiness: {choppiness:.2f}")
                    print(f"   Required: > 30")
                    print(f"   Need {30 - choppiness:.2f} more points")
                else:
                    tp = ask_price + 0.06
                    sl = ask_price - 0.09
                    print("✅ READY TO OPEN POSITION")
                    print(f"   Choppiness: {choppiness:.2f} (> 30 ✓)")
                    print(f"   Entry: ${ask_price:.4f}")
                    print(f"   TP: ${tp:.4f} (+$0.06)")
                    print(f"   SL: ${sl:.4f} (-$0.09)")
    
    # Strategy B
    print("\n📊 STRATEGY B - Both BTC_CALL and BTC_PUT")
    print("-" * 80)
    
    has_strategy_b_call = any(p['strategy'] == 'B' and p['asset_type'] == 'BTC_CALL' for p in open_positions)
    has_strategy_b_put = any(p['strategy'] == 'B' and p['asset_type'] == 'BTC_PUT' for p in open_positions)
    
    if has_strategy_b_call or has_strategy_b_put:
        print("❌ ALREADY HAS OPEN POSITION(S)")
        if has_strategy_b_call:
            print("   - BTC_CALL position exists")
        if has_strategy_b_put:
            print("   - BTC_PUT position exists")
        print("   Reason: Strategy B needs both positions to be closed before re-entering")
    else:
        btc_call_data = data.get('BTC_CALL')
        btc_put_data = data.get('BTC_PUT')
        
        call_ok = False
        put_ok = False
        
        # Check CALL
        if btc_call_data is None:
            print("❌ BTC_CALL: Data file not found")
        elif 'ask' not in btc_call_data:
            print("❌ BTC_CALL: 'ask' field missing")
        elif btc_call_data['ask'] is None:
            print("❌ BTC_CALL: Ask price is NULL")
        else:
            call_ok = True
            call_ask = btc_call_data['ask']
            print(f"✅ BTC_CALL: Ready at ${call_ask:.4f}")
        
        # Check PUT
        if btc_put_data is None:
            print("❌ BTC_PUT: Data file not found")
        elif 'ask' not in btc_put_data:
            print("❌ BTC_PUT: 'ask' field missing")
        elif btc_put_data['ask'] is None:
            print("❌ BTC_PUT: Ask price is NULL")
        else:
            put_ok = True
            put_ask = btc_put_data['ask']
            print(f"✅ BTC_PUT: Ready at ${put_ask:.4f}")
        
        if call_ok and put_ok:
            print("\n✅ READY TO OPEN BOTH POSITIONS")
            call_tp = call_ask + 0.06
            call_sl = call_ask - 0.09
            put_tp = put_ask + 0.06
            put_sl = put_ask - 0.09
            print(f"   CALL: Entry ${call_ask:.4f}, TP ${call_tp:.4f}, SL ${call_sl:.4f}")
            print(f"   PUT:  Entry ${put_ask:.4f}, TP ${put_tp:.4f}, SL ${put_sl:.4f}")
        else:
            print("\n❌ CANNOT OPEN POSITIONS - Both CALL and PUT must be available")
    
    # Strategy C
    print("\n📊 STRATEGY C - Dynamic TP/SL based on instantaneous choppiness")
    print("-" * 80)
    
    has_strategy_c = any(p['strategy'] == 'C' and p['asset_type'] == 'BTC_CALL' for p in open_positions)
    
    if has_strategy_c:
        print("❌ ALREADY HAS OPEN POSITION")
        print("   Reason: Strategy C already has an active BTC_CALL position")
    else:
        btc_call_data = data.get('BTC_CALL')
        
        if btc_call_data is None:
            print("❌ CANNOT OPEN POSITION")
            print("   Reason: BTC_CALL data file not found or invalid")
        elif 'ask' not in btc_call_data:
            print("❌ CANNOT OPEN POSITION")
            print("   Reason: 'ask' field missing in BTC_CALL data")
        elif btc_call_data['ask'] is None:
            print("❌ CANNOT OPEN POSITION")
            print("   Reason: BTC_CALL ask price is NULL")
        else:
            ask_price = btc_call_data['ask']
            
            # Check if we have enough price history
            print("   ⚠️  INSTANTANEOUS CHOPPINESS CHECK:")
            print("   Note: This requires 5-15 seconds of price history")
            print("   The simulator builds this buffer over time")
            print("   If simulator just started, may need to wait 10-15 seconds")
            
            print(f"\n   If choppiness data available, would enter at ${ask_price:.4f}")
            print("   TP/SL would be dynamically calculated:")
            print("   - Higher choppiness (60-100) → Larger TP/SL")
            print("   - Lower choppiness (0-30) → Smaller TP/SL")
            print("   - Always maintains 2:3 ratio (TP:SL)")
    
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)
    
    # Count issues
    issues = []
    
    if data.get('BTC_CALL') is None:
        issues.append("BTC_CALL data file missing")
    elif data['BTC_CALL'].get('ask') is None:
        issues.append("BTC_CALL ask price is NULL")
    
    if data.get('BTC_PUT') is None:
        issues.append("BTC_PUT data file missing")
    elif data['BTC_PUT'].get('ask') is None:
        issues.append("BTC_PUT ask price is NULL")
    
    if oscillation_data is None:
        issues.append("Oscillation data file missing")
    elif oscillation_data.get('oscillation_60min', {}).get('choppiness_index') is None:
        issues.append("Choppiness index is NULL")
    
    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
        
        print("\n💡 RECOMMENDATIONS:")
        print("   1. Check that all data files exist and are being updated")
        print("   2. Verify the data collection scripts are running")
        print("   3. Check file permissions")
        print("   4. Wait a few seconds and run diagnosis again")
    else:
        print("\n✅ ALL DATA FILES OK")
        print("   If positions still not opening, check:")
        print("   1. Are there already open positions preventing new entries?")
        print("   2. For Strategy C: Has enough price history been collected?")
        print("   3. For Strategy A2: Is choppiness > 30?")
    
    print()

if __name__ == "__main__":
    diagnose_strategies()
