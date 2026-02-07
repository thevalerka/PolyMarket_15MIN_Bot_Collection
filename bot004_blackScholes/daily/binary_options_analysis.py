import json
import glob
from datetime import datetime
from collections import defaultdict
import os
import gc

# Configuration
data_folder = "/home/ubuntu/013_2025_polymarket/bot004_blackScholes/daily"
file_pattern = "option_probabilities_M15_*.json"

# Get all file paths
file_paths = glob.glob(os.path.join(data_folder, file_pattern))
file_paths.sort()

print(f"Found {len(file_paths)} files to process")

# Initialize probability level tracking (0.01 to 0.99 in 0.01 steps)
probability_levels = [round(i/100, 2) for i in range(1, 100)]
pnl_by_probability = {prob: {'total_pnl': 0, 'num_trades': 0, 'wins': 0, 'losses': 0} 
                      for prob in probability_levels}

total_periods_analyzed = 0
total_records_processed = 0
date_range_start = None
date_range_end = None

# Storage for the last period from previous file (for continuity)
previous_last_period = None

def get_period_key(dt_str):
    """Get the period key for a datetime string (rounds down to :00, :15, :30, :45)"""
    dt = datetime.fromisoformat(dt_str)
    minute = (dt.minute // 15) * 15
    period_dt = dt.replace(minute=minute, second=0, microsecond=0)
    return period_dt.isoformat()

def process_file(file_path, previous_last_period):
    """Process a single file and return aggregated statistics"""
    print(f"Processing: {os.path.basename(file_path)}")
    
    # Load records from this file
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
    
    # Sort records by datetime
    file_records.sort(key=lambda x: x['datetime'])
    
    # Group records by 15-minute periods
    periods = defaultdict(list)
    for record in file_records:
        period_key = get_period_key(record['datetime'])
        periods[period_key].append(record)
    
    # Analyze periods and calculate outcomes
    sorted_periods = sorted(periods.keys())
    period_outcomes = []
    
    for i, period_key in enumerate(sorted_periods):
        period_records = periods[period_key]
        period_records.sort(key=lambda x: x['datetime'])
        
        last_record = period_records[-1]
        
        # Collect all probability levels reached during this period
        levels_reached = set()
        for record in period_records:
            rounded_prob = round(record['call_probability'] * 100) / 100
            levels_reached.add(rounded_prob)
        
        # Determine outcome
        call_won = None
        if i < len(sorted_periods) - 1:
            # Next period is in this file
            next_period_key = sorted_periods[i + 1]
            next_period_records = periods[next_period_key]
            next_period_records.sort(key=lambda x: x['datetime'])
            first_next = next_period_records[0]
            
            settlement_price = first_next['period_start_price']
            strike_price = last_record['strike_price']
            call_won = settlement_price > strike_price
        
        period_outcomes.append({
            'period': period_key,
            'levels_reached': levels_reached,
            'call_won': call_won,
            'last_record': last_record
        })
    
    # Check if we can resolve the previous file's last period
    if previous_last_period and sorted_periods:
        first_period_records = periods[sorted_periods[0]]
        first_period_records.sort(key=lambda x: x['datetime'])
        first_next = first_period_records[0]
        
        settlement_price = first_next['period_start_price']
        strike_price = previous_last_period['last_record']['strike_price']
        previous_last_period['call_won'] = settlement_price > strike_price
        
        # Add this resolved period to outcomes
        period_outcomes.insert(0, previous_last_period)
    
    # Calculate statistics for this file
    file_stats = {prob: {'pnl': 0, 'trades': 0, 'wins': 0, 'losses': 0} 
                  for prob in probability_levels}
    
    periods_with_outcomes = 0
    for outcome in period_outcomes:
        if outcome['call_won'] is not None:
            periods_with_outcomes += 1
            
            # For each probability level reached in this period, record a trade
            for prob_level in probability_levels:
                if prob_level in outcome['levels_reached']:
                    cost = prob_level
                    
                    if outcome['call_won']:
                        profit = 1.0 - cost
                        file_stats[prob_level]['wins'] += 1
                    else:
                        profit = -cost
                        file_stats[prob_level]['losses'] += 1
                    
                    file_stats[prob_level]['pnl'] += profit
                    file_stats[prob_level]['trades'] += 1
    
    print(f"  Analyzed {periods_with_outcomes} periods with outcomes")
    
    # Return the last period for next file and stats
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
        for prob_level in probability_levels:
            pnl_by_probability[prob_level]['total_pnl'] += result['file_stats'][prob_level]['pnl']
            pnl_by_probability[prob_level]['num_trades'] += result['file_stats'][prob_level]['trades']
            pnl_by_probability[prob_level]['wins'] += result['file_stats'][prob_level]['wins']
            pnl_by_probability[prob_level]['losses'] += result['file_stats'][prob_level]['losses']
        
        total_records_processed += result['num_records']
        total_periods_analyzed += result['num_periods']
        
        if result['date_range'][0]:
            if date_range_start is None:
                date_range_start = result['date_range'][0]
            date_range_end = result['date_range'][1]
        
        previous_last_period = result['last_period']
        
        # Force garbage collection after each file
        gc.collect()
        
    except Exception as e:
        print(f"  ERROR processing file: {e}")
        continue

print(f"\nTotal records processed: {total_records_processed}")
print(f"Total periods analyzed: {total_periods_analyzed}")

# Calculate win rates and averages
for prob_level in probability_levels:
    data = pnl_by_probability[prob_level]
    if data['num_trades'] > 0:
        data['win_rate'] = data['wins'] / data['num_trades']
        data['avg_pnl_per_trade'] = data['total_pnl'] / data['num_trades']
    else:
        data['win_rate'] = 0
        data['avg_pnl_per_trade'] = 0

# Find best probability level (highest total PNL)
profitable_levels = {p: data for p, data in pnl_by_probability.items() if data['num_trades'] > 0}

if profitable_levels:
    best_prob = max(profitable_levels.keys(), key=lambda p: profitable_levels[p]['total_pnl'])
    worst_prob = min(profitable_levels.keys(), key=lambda p: profitable_levels[p]['total_pnl'])
    best_avg_pnl_prob = max(profitable_levels.keys(), key=lambda p: profitable_levels[p]['avg_pnl_per_trade'])
else:
    best_prob = None
    worst_prob = None
    best_avg_pnl_prob = None

# Write detailed results to file
output_file = "binary_option_pnl_analysis.txt"
with open(output_file, 'w') as f:
    f.write("=" * 100 + "\n")
    f.write("BINARY OPTION PNL ANALYSIS - 15 MINUTE OPTIONS\n")
    f.write("=" * 100 + "\n\n")
    
    f.write(f"Analysis Summary:\n")
    f.write(f"  Total periods analyzed: {total_periods_analyzed}\n")
    f.write(f"  Total records processed: {total_records_processed}\n")
    f.write(f"  Total files processed: {len(file_paths)}\n")
    f.write(f"  Date range: {date_range_start if date_range_start else 'N/A'} to {date_range_end if date_range_end else 'N/A'}\n\n")
    
    f.write("Strategy: When option price reaches a probability level during the 15-min period, buy $1\n")
    f.write("  - Cost: $probability (e.g., $0.50 at 50% probability)\n")
    f.write("  - Payout if call wins at expiry: $1.00\n")
    f.write("  - Profit if win: $1.00 - cost\n")
    f.write("  - Loss if lose: -cost\n")
    f.write("  - Each period can generate multiple trades (one per level reached)\n\n")
    
    f.write("=" * 100 + "\n\n")
    
    f.write("PNL BY PROBABILITY LEVEL:\n")
    f.write("-" * 100 + "\n")
    f.write(f"{'Prob':<8} {'Trades':<8} {'Wins':<8} {'Losses':<8} {'Win Rate':<10} {'Total PNL':<12} {'Avg PNL':<12}\n")
    f.write("-" * 100 + "\n")
    
    for prob in sorted(pnl_by_probability.keys()):
        data = pnl_by_probability[prob]
        if data['num_trades'] > 0:
            f.write(f"{prob:<8.2f} {data['num_trades']:<8} {data['wins']:<8} {data['losses']:<8} "
                   f"{data['win_rate']:<10.2%} ${data['total_pnl']:<11.2f} ${data['avg_pnl_per_trade']:<11.4f}\n")
    
    f.write("\n" + "=" * 100 + "\n\n")
    
    if best_prob:
        f.write("BEST PROBABILITY LEVEL (Highest Total PNL):\n")
        f.write(f"  Probability: {best_prob:.2f}\n")
        f.write(f"  Total PNL: ${pnl_by_probability[best_prob]['total_pnl']:.2f}\n")
        f.write(f"  Number of trades: {pnl_by_probability[best_prob]['num_trades']}\n")
        f.write(f"  Wins: {pnl_by_probability[best_prob]['wins']}\n")
        f.write(f"  Losses: {pnl_by_probability[best_prob]['losses']}\n")
        f.write(f"  Win rate: {pnl_by_probability[best_prob]['win_rate']:.2%}\n")
        f.write(f"  Average PNL per trade: ${pnl_by_probability[best_prob]['avg_pnl_per_trade']:.4f}\n\n")
        
        f.write("WORST PROBABILITY LEVEL (Lowest Total PNL):\n")
        f.write(f"  Probability: {worst_prob:.2f}\n")
        f.write(f"  Total PNL: ${pnl_by_probability[worst_prob]['total_pnl']:.2f}\n")
        f.write(f"  Number of trades: {pnl_by_probability[worst_prob]['num_trades']}\n")
        f.write(f"  Win rate: {pnl_by_probability[worst_prob]['win_rate']:.2%}\n\n")
        
        f.write("BEST AVERAGE PNL PER TRADE:\n")
        f.write(f"  Probability: {best_avg_pnl_prob:.2f}\n")
        f.write(f"  Average PNL per trade: ${pnl_by_probability[best_avg_pnl_prob]['avg_pnl_per_trade']:.4f}\n")
        f.write(f"  Total PNL: ${pnl_by_probability[best_avg_pnl_prob]['total_pnl']:.2f}\n")
        f.write(f"  Number of trades: {pnl_by_probability[best_avg_pnl_prob]['num_trades']}\n")
        f.write(f"  Win rate: {pnl_by_probability[best_avg_pnl_prob]['win_rate']:.2%}\n")
    else:
        f.write("No trades found across any probability levels.\n")
    
    f.write("\n" + "=" * 100 + "\n")

print(f"\nResults written to {output_file}\n")

if best_prob:
    print(f"BEST PROBABILITY LEVEL (Total PNL): {best_prob:.2f}")
    print(f"  Total PNL: ${pnl_by_probability[best_prob]['total_pnl']:.2f}")
    print(f"  Number of trades: {pnl_by_probability[best_prob]['num_trades']}")
    print(f"  Win rate: {pnl_by_probability[best_prob]['win_rate']:.2%}")
    print(f"  Average PNL per trade: ${pnl_by_probability[best_prob]['avg_pnl_per_trade']:.4f}\n")
    
    print(f"BEST AVERAGE PNL PER TRADE: {best_avg_pnl_prob:.2f}")
    print(f"  Average PNL per trade: ${pnl_by_probability[best_avg_pnl_prob]['avg_pnl_per_trade']:.4f}")
    print(f"  Total PNL: ${pnl_by_probability[best_avg_pnl_prob]['total_pnl']:.2f}")
    print(f"  Number of trades: {pnl_by_probability[best_avg_pnl_prob]['num_trades']}")
else:
    print("No profitable trades found.")

print(f"\nFull results saved to: {output_file}")
