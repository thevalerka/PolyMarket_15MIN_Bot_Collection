import json
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob

# Find all trading results JSON files in the current directory
json_files = glob.glob('trading_results_*.json')

if not json_files:
    print("No trading_results_*.json files found in the current directory")
    exit()

print(f"Found {len(json_files)} JSON files:")
for f in json_files:
    print(f"  - {f}")

# Load and combine all trading data
all_trades_list = []
daily_summaries = []

for json_file in sorted(json_files):
    print(f"\nProcessing {json_file}...")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Store daily summary
    daily_summaries.append({
        'file': json_file,
        'date': data.get('date'),
        'last_updated': data.get('last_updated'),
        'total_trades': data.get('total_trades'),
        'open_positions': data.get('open_positions'),
        **{f'pnl_{k}': v for k, v in data.get('total_pnl', {}).items()}
    })
    
    # Extract and flatten nested JSON structure
    for trade in data.get('trades', []):
        flat_trade = {
            'date': data.get('date'),
            'timestamp': trade.get('timestamp'),
            'period_end': trade.get('period_end'),
            'strategy': trade.get('strategy'),
            'pair_type': trade.get('pair_type'),
            'asset1': trade.get('asset1'),
            'asset1_entry': trade.get('asset1_entry'),
            'asset1_exit': trade.get('asset1_exit'),
            'asset2': trade.get('asset2'),
            'asset2_entry': trade.get('asset2_entry'),
            'asset2_exit': trade.get('asset2_exit'),
            'total_pnl': trade.get('total_pnl'),
        }
        
        # Add market conditions
        mc = trade.get('market_conditions', {})
        flat_trade.update({
            'volatility': mc.get('volatility'),
            'volatility_method': mc.get('volatility_method'),
            'time_to_expiry_minutes': mc.get('time_to_expiry_minutes'),
            'ma_5min_crossings_last_15min': mc.get('ma_5min_crossings_last_15min'),
            'max_swing_distance': mc.get('max_swing_distance'),
            'avg_last_3_swings_pct_atr': mc.get('avg_last_3_swings_pct_atr'),
            'data_freshness_pct': mc.get('data_freshness_pct'),
        })
        
        # Add oscillation metrics
        osc = mc.get('oscillation_60min', {})
        flat_trade.update({
            'osc_efficiency_ratio': osc.get('efficiency_ratio'),
            'osc_choppiness_index': osc.get('choppiness_index'),
            'osc_zero_crossing_rate': osc.get('zero_crossing_rate'),
            'osc_fractal_dimension': osc.get('fractal_dimension'),
        })
        
        all_trades_list.append(flat_trade)

trades = pd.DataFrame(all_trades_list)
daily_df = pd.DataFrame(daily_summaries)

print("\n" + "="*80)
print("MULTI-DAY TRADING RESULTS ANALYSIS")
print("="*80)
print(f"\nTotal Days Analyzed: {len(json_files)}")
print(f"Total Trades: {len(trades)}")
print(f"\nDaily Summary:")
print(daily_df[['date', 'total_trades', 'pnl_ALL']].to_string(index=False))

# Basic statistics
print(f"\n{'='*80}")
print("OVERALL PNL STATISTICS")
print("="*80)
print(f"Total PnL: ${trades['total_pnl'].sum():.2f}")
print(f"Mean PnL per trade: ${trades['total_pnl'].mean():.3f}")
print(f"Median PnL per trade: ${trades['total_pnl'].median():.3f}")
print(f"Std Dev: ${trades['total_pnl'].std():.3f}")
print(f"Min PnL: ${trades['total_pnl'].min():.3f}")
print(f"Max PnL: ${trades['total_pnl'].max():.3f}")
print(f"Win Rate: {(trades['total_pnl'] > 0).sum() / len(trades) * 100:.1f}%")

# Strategy performance
print(f"\n{'='*80}")
print("STRATEGY PERFORMANCE")
print("="*80)
strategy_stats = trades.groupby('strategy')['total_pnl'].agg([
    'count', 'mean', 'median', 'std', 'min', 'max',
    lambda x: (x > 0).sum() / len(x) * 100
])
strategy_stats.columns = ['Count', 'Mean', 'Median', 'Std', 'Min', 'Max', 'Win Rate %']
print(strategy_stats.round(3))

# Pair type performance
print(f"\n{'='*80}")
print("PAIR TYPE PERFORMANCE")
print("="*80)
pair_stats = trades.groupby('pair_type')['total_pnl'].agg([
    'count', 'mean', 'median', 
    lambda x: (x > 0).sum() / len(x) * 100
])
pair_stats.columns = ['Count', 'Mean', 'Median', 'Win Rate %']
print(pair_stats.round(3))

# Correlation analysis (excluding asset1_pnl and asset2_pnl)
print(f"\n{'='*80}")
print("CORRELATION ANALYSIS WITH PNL")
print("="*80)

# Select numeric columns for correlation, excluding individual asset PnLs and exits
numeric_cols = trades.select_dtypes(include=[np.number]).columns
excluded_cols = ['asset1_pnl', 'asset2_pnl', 'asset1_exit', 'asset2_exit', 'timestamp']
numeric_cols = [col for col in numeric_cols if col not in excluded_cols]

correlations = trades[numeric_cols].corr()['total_pnl'].sort_values(ascending=False)
print("\nTop positive correlations:")
print(correlations[1:11].round(3))
print("\nTop negative correlations:")
print(correlations[-10:].round(3))

# Statistical significance testing
print(f"\n{'='*80}")
print("STATISTICAL SIGNIFICANCE (p-values < 0.05)")
print("="*80)

significant_vars = []
for col in numeric_cols:
    if col != 'total_pnl' and trades[col].notna().sum() > 3:
        try:
            corr, p_value = stats.pearsonr(trades[col].dropna(), 
                                          trades.loc[trades[col].notna(), 'total_pnl'])
            if p_value < 0.05:
                significant_vars.append({
                    'Variable': col,
                    'Correlation': corr,
                    'P-value': p_value
                })
        except:
            pass

if significant_vars:
    sig_df = pd.DataFrame(significant_vars).sort_values('Correlation', 
                                                        key=abs, 
                                                        ascending=False)
    print(sig_df.to_string(index=False))
else:
    print("No statistically significant correlations found (all p > 0.05)")
    print("\nNote: With limited sample size, correlations may not reach statistical significance.")

# Threshold analysis for key variables
print(f"\n{'='*80}")
print("THRESHOLD ANALYSIS")
print("="*80)

key_vars = ['volatility', 'time_to_expiry_minutes', 'osc_efficiency_ratio', 
            'osc_choppiness_index', 'max_swing_distance', 'data_freshness_pct',
            'ma_5min_crossings_last_15min', 'avg_last_3_swings_pct_atr',
            'osc_zero_crossing_rate', 'osc_fractal_dimension']

for var in key_vars:
    if var in trades.columns and trades[var].notna().sum() > 0:
        print(f"\n{var.upper()}:")
        
        # Find optimal threshold using median split
        median_val = trades[var].median()
        above_median = trades[trades[var] > median_val]['total_pnl'].mean()
        below_median = trades[trades[var] <= median_val]['total_pnl'].mean()
        
        print(f"  Median: {median_val:.3f}")
        print(f"  Avg PnL above median: ${above_median:.3f}")
        print(f"  Avg PnL below median: ${below_median:.3f}")
        print(f"  Difference: ${above_median - below_median:.3f}")
        
        # Quartile analysis
        q1, q3 = trades[var].quantile([0.25, 0.75])
        pnl_q1 = trades[trades[var] < q1]['total_pnl'].mean()
        pnl_q3 = trades[trades[var] > q3]['total_pnl'].mean()
        
        print(f"  Q1 ({q1:.3f}): Avg PnL = ${pnl_q1:.3f} (n={len(trades[trades[var] < q1])})")
        print(f"  Q3 ({q3:.3f}): Avg PnL = ${pnl_q3:.3f} (n={len(trades[trades[var] > q3])})")

# Entry price analysis
print(f"\n{'='*80}")
print("ENTRY PRICE ANALYSIS")
print("="*80)

for asset_type in ['asset1', 'asset2']:
    entry_col = f'{asset_type}_entry'
    
    print(f"\n{asset_type.upper()}:")
    
    # Bin entry prices
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
    trades[f'{asset_type}_entry_bin'] = pd.cut(trades[entry_col], bins=bins, labels=labels)
    
    entry_analysis = trades.groupby(f'{asset_type}_entry_bin')['total_pnl'].agg([
        'count', 'mean', lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0
    ])
    entry_analysis.columns = ['Count', 'Avg PnL', 'Win Rate %']
    print(entry_analysis.round(3))

# Summary insights
print(f"\n{'='*80}")
print("KEY INSIGHTS")
print("="*80)

print("\n1. MOST INFLUENTIAL VARIABLES:")
if significant_vars:
    top_5 = sig_df.head(5)
    for idx, row in top_5.iterrows():
        direction = "positive" if row['Correlation'] > 0 else "negative"
        print(f"   - {row['Variable']}: {direction} correlation (r={row['Correlation']:.3f}, p={row['P-value']:.4f})")
else:
    print("   - No statistically significant correlations found")
    print(f"   - Top correlations by magnitude:")
    for var, corr in correlations[1:6].items():
        print(f"     * {var}: r={corr:.3f}")

print("\n2. STRATEGY RECOMMENDATIONS:")
best_strategy = strategy_stats['Mean'].idxmax()
worst_strategy = strategy_stats['Mean'].idxmin()
print(f"   - Best performing: {best_strategy} (${strategy_stats.loc[best_strategy, 'Mean']:.3f} avg, {strategy_stats.loc[best_strategy, 'Win Rate %']:.1f}% win rate)")
print(f"   - Worst performing: {worst_strategy} (${strategy_stats.loc[worst_strategy, 'Mean']:.3f} avg, {strategy_stats.loc[worst_strategy, 'Win Rate %']:.1f}% win rate)")

print("\n3. PAIR TYPE RECOMMENDATIONS:")
best_pair = pair_stats['Mean'].idxmax()
worst_pair = pair_stats['Mean'].idxmin()
print(f"   - Best performing: {best_pair} (${pair_stats.loc[best_pair, 'Mean']:.3f} avg)")
print(f"   - Worst performing: {worst_pair} (${pair_stats.loc[worst_pair, 'Mean']:.3f} avg)")

print("\n4. OPTIMAL THRESHOLDS:")
# Find variables with biggest PnL differences
threshold_impacts = []
for var in key_vars:
    if var in trades.columns and trades[var].notna().sum() > 0:
        median_val = trades[var].median()
        above = trades[trades[var] > median_val]['total_pnl'].mean()
        below = trades[trades[var] <= median_val]['total_pnl'].mean()
        impact = abs(above - below)
        threshold_impacts.append((var, median_val, impact, above, below))

threshold_impacts.sort(key=lambda x: x[2], reverse=True)
for var, threshold, impact, above, below in threshold_impacts[:3]:
    better = "above" if above > below else "below"
    print(f"   - {var}: Trade when {better} {threshold:.3f} (${impact:.3f} avg difference)")

# Save detailed results
output_file = 'trading_analysis_results.csv'
trades.to_csv(output_file, index=False)
print(f"\n\nDetailed results saved to: {output_file}")

# Save daily summary
daily_output = 'daily_summary.csv'
daily_df.to_csv(daily_output, index=False)
print(f"Daily summary saved to: {daily_output}")
