import json
import glob
from datetime import datetime
from collections import defaultdict
import os

# Configuration
data_folder = "/home/ubuntu/013_2025_polymarket/bot004_blackScholes/daily"
file_pattern = "option_probabilities_M15_*.json"

# Storage for results
all_records = []

# Read all JSON files
file_paths = glob.glob(os.path.join(data_folder, file_pattern))
file_paths.sort()

print(f"Found {len(file_paths)} files to process")

for file_path in file_paths:
    print(f"Processing: {os.path.basename(file_path)}")
    with open(file_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                all_records.append(record)
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")
                continue

print(f"\nLoaded {len(all_records)} total records")

# Sort all records by datetime
all_records.sort(key=lambda x: x['datetime'])

# Group records by 15-minute periods
def get_period_key(dt_str):
    """Get the period key for a datetime string (rounds down to :00, :15, :30, :45)"""
    dt = datetime.fromisoformat(dt_str)
    minute = (dt.minute // 15) * 15
    period_dt = dt.replace(minute=minute, second=0, microsecond=0)
    return period_dt.isoformat()

periods = defaultdict(list)
for record in all_records:
    period_key = get_period_key(record['datetime'])
    periods[period_key].append(record)

print(f"Found {len(periods)} distinct 15-minute periods")

# Analyze each period and determine outcome
period_results = []
sorted_periods = sorted(periods.keys())

for i, period_key in enumerate(sorted_periods):
    period_records = periods[period_key]
    period_records.sort(key=lambda x: x['datetime'])
    
    if i < len(sorted_periods) - 1:
        # Get the last record of current period and first of next period
        last_record = period_records[-1]
        next_period_key = sorted_periods[i + 1]
        next_period_records = periods[next_period_key]
        next_period_records.sort(key=lambda x: x['datetime'])
        
        if next_period_records:
            first_next = next_period_records[0]
            
            # Determine if call won:
            # Settlement price (period_start_price of next period) > strike_price of expiring option
            settlement_price = first_next['period_start_price']
            strike_price = last_record['strike_price']
            call_won = settlement_price > strike_price
            
            # Get the last available call probability before expiry
            last_call_prob = last_record['call_probability']
            
            period_results.append({
                'period': period_key,
                'call_probability': last_call_prob,
                'call_won': call_won,
                'strike_price': strike_price,
                'settlement_price': settlement_price,
                'num_records': len(period_records)
            })

print(f"Analyzed {len(period_results)} complete periods with outcomes\n")

# Calculate PNL for each probability level (0.01 to 0.99 in 0.01 steps)
probability_levels = [round(i/100, 2) for i in range(1, 100)]  # 0.01 to 0.99
pnl_by_probability = {}

for prob_level in probability_levels:
    total_pnl = 0
    num_trades = 0
    wins = 0
    losses = 0
    
    for period in period_results:
        # Round the period's call probability to nearest 0.01
        rounded_prob = round(period['call_probability'] * 100) / 100
        
        # Only trade if probability matches this level exactly
        if abs(rounded_prob - prob_level) < 0.005:  # Within rounding tolerance
            # Cost of buying $1 contract at this probability
            cost = prob_level
            
            # Outcome
            if period['call_won']:
                # Win: receive $1, profit = $1 - cost
                profit = 1.0 - cost
                wins += 1
            else:
                # Lose: lose the investment
                profit = -cost
                losses += 1
            
            total_pnl += profit
            num_trades += 1
    
    if num_trades > 0:
        win_rate = wins / num_trades
    else:
        win_rate = 0
    
    pnl_by_probability[prob_level] = {
        'total_pnl': total_pnl,
        'num_trades': num_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'avg_pnl_per_trade': total_pnl / num_trades if num_trades > 0 else 0
    }

# Find best probability level (highest total PNL)
profitable_levels = {p: data for p, data in pnl_by_probability.items() if data['num_trades'] > 0}

if profitable_levels:
    best_prob = max(profitable_levels.keys(), key=lambda p: profitable_levels[p]['total_pnl'])
    worst_prob = min(profitable_levels.keys(), key=lambda p: profitable_levels[p]['total_pnl'])
else:
    best_prob = None
    worst_prob = None

# Write detailed results to file
output_file = "binary_option_pnl_analysis.txt"
with open(output_file, 'w') as f:
    f.write("=" * 100 + "\n")
    f.write("BINARY OPTION PNL ANALYSIS - 15 MINUTE OPTIONS\n")
    f.write("=" * 100 + "\n\n")
    
    f.write(f"Analysis Summary:\n")
    f.write(f"  Total periods analyzed: {len(period_results)}\n")
    f.write(f"  Total records processed: {len(all_records)}\n")
    f.write(f"  Date range: {sorted_periods[0] if sorted_periods else 'N/A'} to {sorted_periods[-1] if sorted_periods else 'N/A'}\n\n")
    
    f.write("Strategy: Buy $1 call option at specific probability levels\n")
    f.write("  - Cost: $probability (e.g., $0.50 at 50% probability)\n")
    f.write("  - Payout if win: $1.00\n")
    f.write("  - Profit if win: $1.00 - cost\n")
    f.write("  - Loss if lose: -cost\n\n")
    
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
        f.write(f"  Win rate: {pnl_by_probability[worst_prob]['win_rate']:.2%}\n")
    else:
        f.write("No trades found across any probability levels.\n")
    
    f.write("\n" + "=" * 100 + "\n")

print(f"Results written to {output_file}\n")

if best_prob:
    print(f"BEST PROBABILITY LEVEL: {best_prob:.2f}")
    print(f"Total PNL: ${pnl_by_probability[best_prob]['total_pnl']:.2f}")
    print(f"Number of trades: {pnl_by_probability[best_prob]['num_trades']}")
    print(f"Win rate: {pnl_by_probability[best_prob]['win_rate']:.2%}")
    print(f"Average PNL per trade: ${pnl_by_probability[best_prob]['avg_pnl_per_trade']:.4f}")
else:
    print("No profitable trades found.")

print(f"\nFull results saved to: {output_file}")
