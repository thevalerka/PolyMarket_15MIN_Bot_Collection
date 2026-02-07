import json
import glob
from datetime import datetime
from collections import defaultdict
import os
import gc

# Configuration
data_folder = "/home/ubuntu/013_2025_polymarket/bot004_blackScholes/daily"
file_pattern = "option_probabilities_M15_*.json"

# Volatility thresholds
VOLATILITY_BANDS = [
    (0.0, 0.1, "extremely_low"),
    (0.1, 0.22, "low"),
    (0.22, 0.36, "moderate"),
    (0.36, 0.54, "sustained"),
    (0.54, 0.80, "high"),
    (0.80, float('inf'), "extreme")
]

# Probability bands (5% blocks)
PROBABILITY_BANDS = []
for i in range(20):
    min_prob = i * 0.05
    max_prob = (i + 1) * 0.05
    band_name = f"{min_prob:.2f}-{max_prob:.2f}"
    PROBABILITY_BANDS.append((min_prob, max_prob, band_name))

def get_volatility_band(volatility):
    """Classify volatility into bands"""
    for min_vol, max_vol, band_name in VOLATILITY_BANDS:
        if min_vol <= volatility < max_vol:
            return band_name
    return "extreme"

def get_probability_band(probability):
    """Classify probability into 5% bands"""
    for min_prob, max_prob, band_name in PROBABILITY_BANDS:
        if min_prob <= probability < max_prob:
            return band_name
    # Handle edge case for probability = 1.0
    if probability >= 0.95:
        return "0.95-1.00"
    return "0.00-0.05"

# Get all file paths
file_paths = glob.glob(os.path.join(data_folder, file_pattern))
file_paths.sort()

print(f"Found {len(file_paths)} files to process")

# Structure: minute -> volatility_band -> probability_band -> stats
pnl_by_minute_vol_prob = {
    minute: {
        vol_band_name: {
            prob_band_name: {'total_pnl': 0, 'num_trades': 0, 'wins': 0, 'losses': 0}
            for _, _, prob_band_name in PROBABILITY_BANDS
        }
        for _, _, vol_band_name in VOLATILITY_BANDS
    }
    for minute in range(15)
}

total_periods_analyzed = 0
total_records_processed = 0
date_range_start = None
date_range_end = None

previous_last_period = None

def get_period_key(dt_str):
    """Get the period key for a datetime string (rounds down to :00, :15, :30, :45)"""
    dt = datetime.fromisoformat(dt_str)
    minute = (dt.minute // 15) * 15
    period_dt = dt.replace(minute=minute, second=0, microsecond=0)
    return period_dt.isoformat()

def get_period_start_datetime(period_key):
    """Convert period key back to datetime object"""
    return datetime.fromisoformat(period_key)

def get_minute_bucket(record_dt, period_start_dt):
    """Get the minute bucket (0-14) for a record within its period"""
    time_diff = (record_dt - period_start_dt).total_seconds()
    minute_bucket = int(time_diff // 60)
    return max(0, min(14, minute_bucket))

def process_file(file_path, previous_last_period):
    """Process a single file and return aggregated statistics"""
    print(f"Processing: {os.path.basename(file_path)}")
    
    file_records = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                file_records.append(record)
            except json.JSONDecodeError as e:
                print(f"  Error parsing line: {e}")
                continue
    
    print(f"  Loaded {len(file_records)} records")
    
    file_records.sort(key=lambda x: x['datetime'])
    
    periods = defaultdict(list)
    for record in file_records:
        period_key = get_period_key(record['datetime'])
        periods[period_key].append(record)
    
    sorted_periods = sorted(periods.keys())
    period_outcomes = []
    
    for i, period_key in enumerate(sorted_periods):
        period_records = periods[period_key]
        period_records.sort(key=lambda x: x['datetime'])
        
        last_record = period_records[-1]
        period_start_dt = get_period_start_datetime(period_key)
        
        # Collect probability bands reached during this period by minute and volatility
        bands_by_minute_and_vol = defaultdict(lambda: defaultdict(set))
        
        for record in period_records:
            call_prob = record['call_probability']
            prob_band = get_probability_band(call_prob)
            record_dt = datetime.fromisoformat(record['datetime'])
            minute_bucket = get_minute_bucket(record_dt, period_start_dt)
            volatility = record.get('volatility', 0)
            vol_band = get_volatility_band(volatility)
            
            bands_by_minute_and_vol[minute_bucket][vol_band].add(prob_band)
        
        # Determine outcome
        call_won = None
        if i < len(sorted_periods) - 1:
            next_period_key = sorted_periods[i + 1]
            next_period_records = periods[next_period_key]
            next_period_records.sort(key=lambda x: x['datetime'])
            first_next = next_period_records[0]
            
            settlement_price = first_next['period_start_price']
            strike_price = last_record['strike_price']
            call_won = settlement_price > strike_price
        
        period_outcomes.append({
            'period': period_key,
            'bands_by_minute_and_vol': bands_by_minute_and_vol,
            'call_won': call_won,
            'last_record': last_record
        })
    
    if previous_last_period and sorted_periods:
        first_period_records = periods[sorted_periods[0]]
        first_period_records.sort(key=lambda x: x['datetime'])
        first_next = first_period_records[0]
        
        settlement_price = first_next['period_start_price']
        strike_price = previous_last_period['last_record']['strike_price']
        previous_last_period['call_won'] = settlement_price > strike_price
        
        period_outcomes.insert(0, previous_last_period)
    
    # Calculate statistics for this file
    file_stats = {
        minute: {
            vol_band_name: {
                prob_band_name: {'pnl': 0, 'trades': 0, 'wins': 0, 'losses': 0}
                for _, _, prob_band_name in PROBABILITY_BANDS
            }
            for _, _, vol_band_name in VOLATILITY_BANDS
        }
        for minute in range(15)
    }
    
    periods_with_outcomes = 0
    for outcome in period_outcomes:
        if outcome['call_won'] is not None:
            periods_with_outcomes += 1
            
            for minute_bucket, vol_bands in outcome['bands_by_minute_and_vol'].items():
                for vol_band, prob_bands_in_vol in vol_bands.items():
                    for prob_band in prob_bands_in_vol:
                        # Use middle of probability band as cost
                        min_prob_str, max_prob_str = prob_band.split('-')
                        min_prob = float(min_prob_str)
                        max_prob = float(max_prob_str)
                        cost = (min_prob + max_prob) / 2
                        
                        if outcome['call_won']:
                            profit = 1.0 - cost
                            file_stats[minute_bucket][vol_band][prob_band]['wins'] += 1
                        else:
                            profit = -cost
                            file_stats[minute_bucket][vol_band][prob_band]['losses'] += 1
                        
                        file_stats[minute_bucket][vol_band][prob_band]['pnl'] += profit
                        file_stats[minute_bucket][vol_band][prob_band]['trades'] += 1
    
    print(f"  Analyzed {periods_with_outcomes} periods with outcomes")
    
    last_period_for_next = period_outcomes[-1] if period_outcomes and period_outcomes[-1]['call_won'] is None else None
    
    return {
        'file_stats': file_stats,
        'last_period': last_period_for_next,
        'num_records': len(file_records),
        'num_periods': periods_with_outcomes,
        'date_range': (sorted_periods[0] if sorted_periods else None, 
                      sorted_periods[-1] if sorted_periods else None)
    }

# Process each file
for file_path in file_paths:
    try:
        result = process_file(file_path, previous_last_period)
        
        # Aggregate statistics
        for minute in range(15):
            for _, _, vol_band_name in VOLATILITY_BANDS:
                for _, _, prob_band_name in PROBABILITY_BANDS:
                    pnl_by_minute_vol_prob[minute][vol_band_name][prob_band_name]['total_pnl'] += \
                        result['file_stats'][minute][vol_band_name][prob_band_name]['pnl']
                    pnl_by_minute_vol_prob[minute][vol_band_name][prob_band_name]['num_trades'] += \
                        result['file_stats'][minute][vol_band_name][prob_band_name]['trades']
                    pnl_by_minute_vol_prob[minute][vol_band_name][prob_band_name]['wins'] += \
                        result['file_stats'][minute][vol_band_name][prob_band_name]['wins']
                    pnl_by_minute_vol_prob[minute][vol_band_name][prob_band_name]['losses'] += \
                        result['file_stats'][minute][vol_band_name][prob_band_name]['losses']
        
        total_records_processed += result['num_records']
        total_periods_analyzed += result['num_periods']
        
        if result['date_range'][0]:
            if date_range_start is None:
                date_range_start = result['date_range'][0]
            date_range_end = result['date_range'][1]
        
        previous_last_period = result['last_period']
        
        gc.collect()
        
    except Exception as e:
        print(f"  ERROR processing file: {e}")
        continue

print(f"\nTotal records processed: {total_records_processed}")
print(f"Total periods analyzed: {total_periods_analyzed}")

# Calculate win rates and averages
for minute in range(15):
    for _, _, vol_band_name in VOLATILITY_BANDS:
        for _, _, prob_band_name in PROBABILITY_BANDS:
            data = pnl_by_minute_vol_prob[minute][vol_band_name][prob_band_name]
            if data['num_trades'] > 0:
                data['win_rate'] = data['wins'] / data['num_trades']
                data['avg_pnl_per_trade'] = data['total_pnl'] / data['num_trades']
            else:
                data['win_rate'] = 0
                data['avg_pnl_per_trade'] = 0

# Prepare JSON output
json_output = {
    'summary': {
        'total_periods_analyzed': total_periods_analyzed,
        'total_records_processed': total_records_processed,
        'total_files_processed': len(file_paths),
        'date_range_start': date_range_start,
        'date_range_end': date_range_end,
        'volatility_bands': [
            {
                'name': band_name,
                'min': min_vol,
                'max': max_vol if max_vol != float('inf') else 'infinity'
            }
            for min_vol, max_vol, band_name in VOLATILITY_BANDS
        ],
        'probability_bands': [
            {
                'name': band_name,
                'min': min_prob,
                'max': max_prob
            }
            for min_prob, max_prob, band_name in PROBABILITY_BANDS
        ]
    },
    'analysis': {}
}

# Structure the output
for minute in range(15):
    json_output['analysis'][f'minute_{minute}'] = {}
    
    for _, _, vol_band_name in VOLATILITY_BANDS:
        json_output['analysis'][f'minute_{minute}'][vol_band_name] = {}
        
        for _, _, prob_band_name in PROBABILITY_BANDS:
            data = pnl_by_minute_vol_prob[minute][vol_band_name][prob_band_name]
            
            if data['num_trades'] > 0:
                json_output['analysis'][f'minute_{minute}'][vol_band_name][prob_band_name] = {
                    'num_trades': data['num_trades'],
                    'wins': data['wins'],
                    'losses': data['losses'],
                    'win_rate': round(data['win_rate'], 4),
                    'total_pnl': round(data['total_pnl'], 4),
                    'avg_pnl_per_trade': round(data['avg_pnl_per_trade'], 6)
                }

# Find best combinations
json_output['best_by_minute_and_volatility'] = {}

for minute in range(15):
    json_output['best_by_minute_and_volatility'][f'minute_{minute}'] = {}
    
    for _, _, vol_band_name in VOLATILITY_BANDS:
        profitable_combos = []
        
        for _, _, prob_band_name in PROBABILITY_BANDS:
            data = pnl_by_minute_vol_prob[minute][vol_band_name][prob_band_name]
            if data['num_trades'] > 0:
                profitable_combos.append({
                    'probability_band': prob_band_name,
                    'total_pnl': data['total_pnl'],
                    'avg_pnl_per_trade': data['avg_pnl_per_trade'],
                    'num_trades': data['num_trades'],
                    'win_rate': data['win_rate']
                })
        
        if profitable_combos:
            best_total = max(profitable_combos, key=lambda x: x['total_pnl'])
            best_avg = max(profitable_combos, key=lambda x: x['avg_pnl_per_trade'])
            
            json_output['best_by_minute_and_volatility'][f'minute_{minute}'][vol_band_name] = {
                'best_total_pnl': {
                    'probability_band': best_total['probability_band'],
                    'total_pnl': round(best_total['total_pnl'], 4),
                    'num_trades': best_total['num_trades'],
                    'win_rate': round(best_total['win_rate'], 4)
                },
                'best_avg_pnl': {
                    'probability_band': best_avg['probability_band'],
                    'avg_pnl_per_trade': round(best_avg['avg_pnl_per_trade'], 6),
                    'num_trades': best_avg['num_trades'],
                    'win_rate': round(best_avg['win_rate'], 4)
                },
                'total_opportunities': len(profitable_combos)
            }

# Write JSON output
json_output_file = "binary_option_analysis_minute_volatility.json"
with open(json_output_file, 'w') as f:
    json.dump(json_output, f, indent=2)

print(f"\nJSON results written to {json_output_file}")

# Write TXT report
txt_output_file = "binary_option_analysis_minute_volatility.txt"
with open(txt_output_file, 'w') as f:
    f.write("=" * 100 + "\n")
    f.write("BINARY OPTION ANALYSIS - MINUTE, VOLATILITY & PROBABILITY BANDS\n")
    f.write("=" * 100 + "\n\n")
    
    f.write(f"Analysis Summary:\n")
    f.write(f"  Total periods analyzed: {total_periods_analyzed}\n")
    f.write(f"  Total records processed: {total_records_processed}\n")
    f.write(f"  Total files processed: {len(file_paths)}\n")
    f.write(f"  Date range: {date_range_start} to {date_range_end}\n\n")
    
    f.write("Volatility Bands:\n")
    for min_vol, max_vol, band_name in VOLATILITY_BANDS:
        max_str = f"{max_vol:.2f}" if max_vol != float('inf') else "infinity"
        f.write(f"  {band_name.replace('_', ' ').title()}: {min_vol:.2f} - {max_str}\n")
    
    f.write("\nProbability Bands (5% blocks):\n")
    f.write("  20 bands from 0.00-0.05 to 0.95-1.00\n\n")
    
    f.write("=" * 100 + "\n\n")
    
    # Overall summary by volatility
    f.write("SUMMARY BY VOLATILITY BAND (All Minutes Combined):\n")
    f.write("-" * 100 + "\n")
    
    for _, _, vol_band_name in VOLATILITY_BANDS:
        total_trades = 0
        total_pnl = 0
        total_wins = 0
        
        for minute in range(15):
            for _, _, prob_band_name in PROBABILITY_BANDS:
                data = pnl_by_minute_vol_prob[minute][vol_band_name][prob_band_name]
                total_trades += data['num_trades']
                total_pnl += data['total_pnl']
                total_wins += data['wins']
        
        if total_trades > 0:
            win_rate = total_wins / total_trades
            avg_pnl = total_pnl / total_trades
            
            f.write(f"\n{vol_band_name.replace('_', ' ').upper()}:\n")
            f.write(f"  Total trades: {total_trades}\n")
            f.write(f"  Win rate: {win_rate:.2%}\n")
            f.write(f"  Total PNL: ${total_pnl:.2f}\n")
            f.write(f"  Average PNL per trade: ${avg_pnl:.6f}\n")
    
    f.write("\n" + "=" * 100 + "\n\n")
    
    # Best combinations by minute and volatility
    f.write("BEST PROBABILITY BANDS BY MINUTE AND VOLATILITY:\n")
    f.write("-" * 100 + "\n\n")
    
    for minute in range(15):
        f.write(f"MINUTE {minute} ({minute*60}-{(minute+1)*60-1} seconds):\n")
        f.write("-" * 100 + "\n")
        
        for _, _, vol_band_name in VOLATILITY_BANDS:
            best_data = json_output['best_by_minute_and_volatility'].get(f'minute_{minute}', {}).get(vol_band_name)
            
            if best_data:
                f.write(f"\n  {vol_band_name.replace('_', ' ').title()}:\n")
                f.write(f"    Best Total PNL: {best_data['best_total_pnl']['probability_band']} "
                       f"(${best_data['best_total_pnl']['total_pnl']:.2f} from "
                       f"{best_data['best_total_pnl']['num_trades']} trades, "
                       f"win rate: {best_data['best_total_pnl']['win_rate']:.2%})\n")
                f.write(f"    Best Avg PNL: {best_data['best_avg_pnl']['probability_band']} "
                       f"(${best_data['best_avg_pnl']['avg_pnl_per_trade']:.6f} per trade from "
                       f"{best_data['best_avg_pnl']['num_trades']} trades)\n")
                f.write(f"    Total opportunities: {best_data['total_opportunities']} probability bands\n")
        
        f.write("\n")
    
    f.write("=" * 100 + "\n\n")
    
    # Detailed breakdown
    f.write("DETAILED BREAKDOWN BY MINUTE, VOLATILITY AND PROBABILITY:\n")
    f.write("=" * 100 + "\n\n")
    
    for minute in range(15):
        f.write(f"\n{'='*100}\n")
        f.write(f"MINUTE {minute} ({minute*60}-{(minute+1)*60-1} seconds)\n")
        f.write(f"{'='*100}\n\n")
        
        for _, _, vol_band_name in VOLATILITY_BANDS:
            # Check if this volatility band has any data for this minute
            has_data = False
            for _, _, prob_band_name in PROBABILITY_BANDS:
                if pnl_by_minute_vol_prob[minute][vol_band_name][prob_band_name]['num_trades'] > 0:
                    has_data = True
                    break
            
            if not has_data:
                continue
            
            f.write(f"\n{vol_band_name.replace('_', ' ').upper()}:\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Prob Band':<15} {'Trades':<10} {'Wins':<10} {'Losses':<10} "
                   f"{'Win Rate':<12} {'Total PNL':<15} {'Avg PNL':<15}\n")
            f.write("-" * 100 + "\n")
            
            for _, _, prob_band_name in PROBABILITY_BANDS:
                data = pnl_by_minute_vol_prob[minute][vol_band_name][prob_band_name]
                
                if data['num_trades'] > 0:
                    f.write(f"{prob_band_name:<15} {data['num_trades']:<10} {data['wins']:<10} "
                           f"{data['losses']:<10} {data['win_rate']:<12.2%} "
                           f"${data['total_pnl']:<14.2f} ${data['avg_pnl_per_trade']:<14.6f}\n")
            
            f.write("\n")
    
    f.write("=" * 100 + "\n")
    f.write("END OF REPORT\n")
    f.write("=" * 100 + "\n")

print(f"TXT report written to {txt_output_file}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nOutput files:")
print(f"  1. {json_output_file} (JSON format)")
print(f"  2. {txt_output_file} (Text report)")
print("="*80)