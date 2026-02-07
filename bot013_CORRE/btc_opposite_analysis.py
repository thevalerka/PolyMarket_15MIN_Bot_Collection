import json
import pandas as pd
import numpy as np
from scipy import stats
import glob

# Find all trading results JSON files in the current directory
json_files = glob.glob('trading_results_*.json')

if not json_files:
    print("No trading_results_*.json files found in the current directory")
    exit()

print(f"Found {len(json_files)} JSON files:")
for f in json_files:
    print(f"  - {f}")

# Load and combine all trading data - ONLY BTC pairs with opposite asset
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
    
    # Extract and flatten nested JSON structure - ONLY BTC PAIRS
    for trade in data.get('trades', []):
        pair_type = trade.get('pair_type', '')
        
        # Only process pairs that include BTC
        if 'BTC' not in pair_type:
            continue
        
        asset1 = trade.get('asset1', '')
        asset2 = trade.get('asset2', '')
        
        # Determine which asset is BTC and which is the opposite
        if 'BTC' in asset1:
            # BTC is asset1, we want asset2 (opposite)
            opposite_asset = asset2
            opposite_entry = trade.get('asset2_entry')
            opposite_exit = trade.get('asset2_exit')
            opposite_pnl = opposite_exit - opposite_entry if opposite_exit is not None and opposite_entry is not None else None
            
            # Extract the coin name (e.g., "XRP" from "XRP_CALL")
            opposite_coin = asset2.split('_')[0] if '_' in asset2 else asset2
            opposite_direction = asset2.split('_')[1] if '_' in asset2 else 'UNKNOWN'
            
        elif 'BTC' in asset2:
            # BTC is asset2, we want asset1 (opposite)
            opposite_asset = asset1
            opposite_entry = trade.get('asset1_entry')
            opposite_exit = trade.get('asset1_exit')
            opposite_pnl = opposite_exit - opposite_entry if opposite_exit is not None and opposite_entry is not None else None
            
            # Extract the coin name
            opposite_coin = asset1.split('_')[0] if '_' in asset1 else asset1
            opposite_direction = asset1.split('_')[1] if '_' in asset1 else 'UNKNOWN'
        else:
            continue
        
        flat_trade = {
            'date': data.get('date'),
            'timestamp': trade.get('timestamp'),
            'period_end': trade.get('period_end'),
            'strategy': trade.get('strategy'),
            'pair_type': pair_type,
            'opposite_coin': opposite_coin,
            'opposite_asset': opposite_asset,
            'opposite_direction': opposite_direction,
            'opposite_entry': opposite_entry,
            'opposite_exit': opposite_exit,
            'opposite_pnl': opposite_pnl,
            'correlation': trade.get('correlation'),
        }
        
        # Add market conditions
        mc = trade.get('market_conditions', {})
        if mc is None:
            mc = {}
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
        if osc is None:
            osc = {}
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
print("BTC OPPOSITE ASSET ANALYSIS")
print("="*80)
print(f"\nTotal Days Analyzed: {len(json_files)}")
print(f"Total BTC Pair Trades: {len(trades)}")

# Convert timestamp to datetime for time analysis
trades['datetime'] = pd.to_datetime(trades['timestamp'])
trades['hour_utc'] = trades['datetime'].dt.hour

# Define time zones
def get_timezone(hour):
    if 0 <= hour < 9.5:
        return 'Asia'
    elif 9.5 <= hour < 14.5:
        return 'Europe'
    elif 14.5 <= hour < 22:
        return 'America'
    else:
        return 'Sleep'

trades['timezone'] = trades['hour_utc'].apply(get_timezone)

# Basic statistics
print(f"\n{'='*80}")
print("OVERALL PNL STATISTICS (Opposite Asset Only)")
print("="*80)
print(f"Total PnL: ${trades['opposite_pnl'].sum():.2f}")
print(f"Mean PnL per trade: ${trades['opposite_pnl'].mean():.3f}")
print(f"Median PnL per trade: ${trades['opposite_pnl'].median():.3f}")
print(f"Std Dev: ${trades['opposite_pnl'].std():.3f}")
print(f"Min PnL: ${trades['opposite_pnl'].min():.3f}")
print(f"Max PnL: ${trades['opposite_pnl'].max():.3f}")
print(f"Win Rate: {(trades['opposite_pnl'] > 0).sum() / len(trades) * 100:.1f}%")

# Opposite coin performance
print(f"\n{'='*80}")
print("PERFORMANCE BY OPPOSITE COIN")
print("="*80)
coin_stats = trades.groupby('opposite_coin')['opposite_pnl'].agg([
    'count', 'mean', 'median', 'sum',
    lambda x: (x > 0).sum() / len(x) * 100
])
coin_stats.columns = ['Count', 'Mean', 'Median', 'Total', 'Win Rate %']
coin_stats = coin_stats.sort_values('Mean', ascending=False)
print(coin_stats.round(3))

# Direction performance (CALL vs PUT)
print(f"\n{'='*80}")
print("PERFORMANCE BY DIRECTION (CALL vs PUT)")
print("="*80)
direction_stats = trades.groupby('opposite_direction')['opposite_pnl'].agg([
    'count', 'mean', 'median',
    lambda x: (x > 0).sum() / len(x) * 100
])
direction_stats.columns = ['Count', 'Mean', 'Median', 'Win Rate %']
print(direction_stats.round(3))

# Strategy performance
print(f"\n{'='*80}")
print("STRATEGY PERFORMANCE")
print("="*80)
strategy_stats = trades.groupby('strategy')['opposite_pnl'].agg([
    'count', 'mean', 'median', 'std', 'min', 'max',
    lambda x: (x > 0).sum() / len(x) * 100
])
strategy_stats.columns = ['Count', 'Mean', 'Median', 'Std', 'Min', 'Max', 'Win Rate %']
print(strategy_stats.round(3))

# Correlation statistics
if 'correlation' in trades.columns and trades['correlation'].notna().sum() > 0:
    print(f"\n{'='*80}")
    print("CORRELATION STATISTICS")
    print("="*80)
    print(f"Trades with correlation data: {trades['correlation'].notna().sum()} / {len(trades)}")
    print(f"Mean correlation: {trades['correlation'].mean():.4f}")
    print(f"Median correlation: {trades['correlation'].median():.4f}")
    print(f"Std Dev: {trades['correlation'].std():.4f}")
    print(f"Min: {trades['correlation'].min():.4f}")
    print(f"Max: {trades['correlation'].max():.4f}")
    
    # Correlation with PnL
    valid_corr = trades[trades['correlation'].notna()]
    if len(valid_corr) > 3:
        corr_pnl, p_corr = stats.pearsonr(valid_corr['correlation'], valid_corr['opposite_pnl'])
        print(f"\nCorrelation coefficient vs PnL: r={corr_pnl:.4f}, p={p_corr:.4f}")
        if p_corr < 0.05:
            print(f"  *** STATISTICALLY SIGNIFICANT ***")
    
    # Correlation bins analysis
    print(f"\nPnL by Correlation Ranges:")
    corr_bins = [-1.0, -0.5, 0, 0.25, 0.5, 0.75, 1.0]
    corr_labels = ['Strong Neg', 'Weak Neg', 'Neutral', 'Weak Pos', 'Moderate Pos', 'Strong Pos']
    trades['correlation_bin'] = pd.cut(trades['correlation'], bins=corr_bins, labels=corr_labels)
    
    corr_bin_stats = trades.groupby('correlation_bin')['opposite_pnl'].agg([
        'count', 'mean', 'sum',
        lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0
    ])
    corr_bin_stats.columns = ['Count', 'Avg PnL', 'Total PnL', 'Win Rate %']
    corr_bin_stats = corr_bin_stats[corr_bin_stats['Count'] > 0]
    print(corr_bin_stats.round(3))

# Entry price analysis
print(f"\n{'='*80}")
print("ENTRY PRICE ANALYSIS (0.05 BINS)")
print("="*80)

bins = np.arange(0, 1.05, 0.05)
trades['entry_bin'] = pd.cut(trades['opposite_entry'], bins=bins)

entry_analysis = trades.groupby('entry_bin')['opposite_pnl'].agg([
    'count', 'mean', lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0
])
entry_analysis.columns = ['Count', 'Avg PnL', 'Win Rate %']
entry_analysis = entry_analysis[entry_analysis['Count'] > 0]
print(entry_analysis.round(3))

# Time zone analysis
print(f"\n{'='*80}")
print("TIME ZONE ANALYSIS")
print("="*80)

timezone_stats = trades.groupby('timezone')['opposite_pnl'].agg([
    'count', 'mean', 'median', 'sum',
    lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0
])
timezone_stats.columns = ['Count', 'Mean', 'Median', 'Total', 'Win Rate %']

timezone_order = ['Asia', 'Europe', 'America', 'Sleep']
timezone_stats = timezone_stats.reindex([tz for tz in timezone_order if tz in timezone_stats.index])

print("\nPerformance by Time Zone:")
print(timezone_stats.round(3))

# Correlation analysis
print(f"\n{'='*80}")
print("CORRELATION ANALYSIS WITH PNL")
print("="*80)

# Select numeric columns for correlation
numeric_cols = trades.select_dtypes(include=[np.number]).columns
excluded_cols = ['timestamp', 'hour_utc', 'opposite_exit']
numeric_cols = [col for col in numeric_cols if col not in excluded_cols]

correlations = trades[numeric_cols].corr()['opposite_pnl'].sort_values(ascending=False)
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
    if col != 'opposite_pnl' and trades[col].notna().sum() > 3:
        try:
            corr, p_value = stats.pearsonr(trades[col].dropna(), 
                                          trades.loc[trades[col].notna(), 'opposite_pnl'])
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

# Threshold analysis
print(f"\n{'='*80}")
print("THRESHOLD ANALYSIS")
print("="*80)

key_vars = ['volatility', 'time_to_expiry_minutes', 'osc_efficiency_ratio', 
            'osc_choppiness_index', 'max_swing_distance', 'data_freshness_pct',
            'ma_5min_crossings_last_15min', 'avg_last_3_swings_pct_atr',
            'osc_zero_crossing_rate', 'osc_fractal_dimension', 'correlation',
            'opposite_entry']

for var in key_vars:
    if var in trades.columns and trades[var].notna().sum() > 0:
        print(f"\n{var.upper()}:")
        
        median_val = trades[var].median()
        above_median = trades[trades[var] > median_val]['opposite_pnl'].mean()
        below_median = trades[trades[var] <= median_val]['opposite_pnl'].mean()
        
        print(f"  Median: {median_val:.3f}")
        print(f"  Avg PnL above median: ${above_median:.3f}")
        print(f"  Avg PnL below median: ${below_median:.3f}")
        print(f"  Difference: ${above_median - below_median:.3f}")
        
        q1, q3 = trades[var].quantile([0.25, 0.75])
        pnl_q1 = trades[trades[var] < q1]['opposite_pnl'].mean()
        pnl_q3 = trades[trades[var] > q3]['opposite_pnl'].mean()
        
        print(f"  Q1 ({q1:.3f}): Avg PnL = ${pnl_q1:.3f} (n={len(trades[trades[var] < q1])})")
        print(f"  Q3 ({q3:.3f}): Avg PnL = ${pnl_q3:.3f} (n={len(trades[trades[var] > q3])})")

# PER-STRATEGY ANALYSIS
print(f"\n\n{'='*80}")
print("PER-STRATEGY DETAILED ANALYSIS")
print("="*80)

strategies = trades['strategy'].unique()

for strategy in sorted(strategies):
    strat_trades = trades[trades['strategy'] == strategy].copy()
    
    print(f"\n\n{'#'*80}")
    print(f"STRATEGY: {strategy}")
    print(f"{'#'*80}")
    
    print(f"\nBasic Statistics:")
    print(f"  Total Trades: {len(strat_trades)}")
    print(f"  Total PnL: ${strat_trades['opposite_pnl'].sum():.2f}")
    print(f"  Mean PnL: ${strat_trades['opposite_pnl'].mean():.3f}")
    print(f"  Median PnL: ${strat_trades['opposite_pnl'].median():.3f}")
    print(f"  Win Rate: {(strat_trades['opposite_pnl'] > 0).sum() / len(strat_trades) * 100:.1f}%")
    
    # Opposite coin performance for this strategy
    if len(strat_trades['opposite_coin'].unique()) > 1:
        print(f"\nOpposite Coin Performance:")
        coin_stats_strat = strat_trades.groupby('opposite_coin')['opposite_pnl'].agg([
            'count', 'mean', lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0
        ])
        coin_stats_strat.columns = ['Count', 'Mean', 'Win Rate %']
        print(coin_stats_strat.round(3))
    
    # Time zone performance
    print(f"\nTime Zone Performance:")
    tz_stats_strat = strat_trades.groupby('timezone')['opposite_pnl'].agg([
        'count', 'mean', lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0
    ])
    tz_stats_strat.columns = ['Count', 'Mean', 'Win Rate %']
    tz_stats_strat = tz_stats_strat.reindex([tz for tz in timezone_order if tz in tz_stats_strat.index])
    print(tz_stats_strat.round(3))
    
    # Correlation analysis
    if 'correlation' in strat_trades.columns and strat_trades['correlation'].notna().sum() > 3:
        print(f"\nCorrelation Statistics:")
        print(f"  Mean: {strat_trades['correlation'].mean():.4f}")
        print(f"  Median: {strat_trades['correlation'].median():.4f}")
        corr_pnl_s, p_pnl_s = stats.pearsonr(
            strat_trades['correlation'].dropna(), 
            strat_trades.loc[strat_trades['correlation'].notna(), 'opposite_pnl']
        )
        print(f"  Correlation with PnL: r={corr_pnl_s:.4f}, p={p_pnl_s:.4f}")

# Hourly accumulated PnL by strategy
print(f"\n\n{'='*80}")
print("HOURLY ACCUMULATED PNL BY STRATEGY")
print("="*80)

trades['hour'] = trades['datetime'].dt.floor('h')
hourly_pnl = trades.groupby(['hour', 'strategy'])['opposite_pnl'].sum().reset_index()
hourly_pnl_pivot = hourly_pnl.pivot(index='hour', columns='strategy', values='opposite_pnl').fillna(0)

# Calculate cumulative PnL
hourly_cumulative = hourly_pnl_pivot.cumsum()

# Save to CSV
hourly_output = 'btc_opposite_hourly_accumulated_pnl_by_strategy.csv'
hourly_cumulative.to_csv(hourly_output)
print(f"\nHourly accumulated PnL saved to: {hourly_output}")
print(f"\nPreview (first 10 hours):")
print(hourly_cumulative.head(10).round(2))

print(f"\nFinal accumulated PnL by strategy:")
print(hourly_cumulative.iloc[-1].round(2))

# Save detailed results
output_file = 'btc_opposite_analysis_results.csv'
trades.to_csv(output_file, index=False)
print(f"\n\nDetailed results saved to: {output_file}")

# Summary insights
print(f"\n{'='*80}")
print("KEY INSIGHTS - BTC OPPOSITE ASSET")
print("="*80)

print("\n1. BEST PERFORMING OPPOSITE COIN:")
best_coin = coin_stats['Mean'].idxmax()
print(f"   - {best_coin}: ${coin_stats.loc[best_coin, 'Mean']:.3f} avg, {coin_stats.loc[best_coin, 'Win Rate %']:.1f}% win rate")

print("\n2. BEST STRATEGY:")
best_strategy = strategy_stats['Mean'].idxmax()
print(f"   - {best_strategy}: ${strategy_stats.loc[best_strategy, 'Mean']:.3f} avg, {strategy_stats.loc[best_strategy, 'Win Rate %']:.1f}% win rate")

print("\n3. DIRECTION PREFERENCE:")
if len(direction_stats) > 0:
    best_direction = direction_stats['Mean'].idxmax()
    print(f"   - {best_direction}: ${direction_stats.loc[best_direction, 'Mean']:.3f} avg")

if significant_vars:
    print("\n4. MOST INFLUENTIAL VARIABLES:")
    for idx, row in sig_df.head(3).iterrows():
        direction = "positive" if row['Correlation'] > 0 else "negative"
        print(f"   - {row['Variable']}: {direction} correlation (r={row['Correlation']:.3f}, p={row['P-value']:.4f})")

# SPECIAL ANALYSIS: High Volatility + High Correlation
print(f"\n\n{'='*80}")
print("SPECIAL ANALYSIS: HIGH VOLATILITY + HIGH CORRELATION")
print("="*80)
print("Conditions: volatility > 0.5 AND correlation > 0.8")

# Filter for high volatility and high correlation
high_vol_high_corr = trades[
    (trades['volatility'] > 0.5) & 
    (trades['correlation'] > 0.8)
].copy()

if len(high_vol_high_corr) > 0:
    print(f"\nTrades matching criteria: {len(high_vol_high_corr)} / {len(trades)} ({len(high_vol_high_corr)/len(trades)*100:.1f}%)")
    
    # Basic statistics
    print(f"\nPnL Statistics:")
    print(f"  Total PnL: ${high_vol_high_corr['opposite_pnl'].sum():.2f}")
    print(f"  Mean PnL: ${high_vol_high_corr['opposite_pnl'].mean():.3f}")
    print(f"  Median PnL: ${high_vol_high_corr['opposite_pnl'].median():.3f}")
    print(f"  Std Dev: ${high_vol_high_corr['opposite_pnl'].std():.3f}")
    print(f"  Win Rate: {(high_vol_high_corr['opposite_pnl'] > 0).sum() / len(high_vol_high_corr) * 100:.1f}%")
    
    # Compare to overall performance
    overall_mean = trades['opposite_pnl'].mean()
    overall_winrate = (trades['opposite_pnl'] > 0).sum() / len(trades) * 100
    special_mean = high_vol_high_corr['opposite_pnl'].mean()
    special_winrate = (high_vol_high_corr['opposite_pnl'] > 0).sum() / len(high_vol_high_corr) * 100
    
    print(f"\nComparison to Overall Performance:")
    print(f"  Mean PnL: ${special_mean:.3f} vs ${overall_mean:.3f} (diff: ${special_mean - overall_mean:+.3f})")
    print(f"  Win Rate: {special_winrate:.1f}% vs {overall_winrate:.1f}% (diff: {special_winrate - overall_winrate:+.1f}%)")
    
    # Strategy breakdown
    print(f"\nStrategy Performance in High Vol + High Corr:")
    special_strategy = high_vol_high_corr.groupby('strategy')['opposite_pnl'].agg([
        'count', 'mean', 'sum',
        lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0
    ])
    special_strategy.columns = ['Count', 'Mean', 'Total', 'Win Rate %']
    print(special_strategy.round(3))
    
    # Opposite coin breakdown
    if len(high_vol_high_corr['opposite_coin'].unique()) > 1:
        print(f"\nOpposite Coin Performance in High Vol + High Corr:")
        special_coin = high_vol_high_corr.groupby('opposite_coin')['opposite_pnl'].agg([
            'count', 'mean',
            lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0
        ])
        special_coin.columns = ['Count', 'Mean', 'Win Rate %']
        print(special_coin.round(3))
    
    # Direction analysis
    if len(high_vol_high_corr['opposite_direction'].unique()) > 1:
        print(f"\nDirection Performance in High Vol + High Corr:")
        special_direction = high_vol_high_corr.groupby('opposite_direction')['opposite_pnl'].agg([
            'count', 'mean',
            lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0
        ])
        special_direction.columns = ['Count', 'Mean', 'Win Rate %']
        print(special_direction.round(3))
    
    # Time zone analysis
    print(f"\nTime Zone Performance in High Vol + High Corr:")
    special_tz = high_vol_high_corr.groupby('timezone')['opposite_pnl'].agg([
        'count', 'mean',
        lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0
    ])
    special_tz.columns = ['Count', 'Mean', 'Win Rate %']
    special_tz = special_tz.reindex([tz for tz in timezone_order if tz in special_tz.index])
    print(special_tz.round(3))
    
    # Entry price analysis
    print(f"\nEntry Price Statistics in High Vol + High Corr:")
    print(f"  Mean Entry: {high_vol_high_corr['opposite_entry'].mean():.3f}")
    print(f"  Median Entry: {high_vol_high_corr['opposite_entry'].median():.3f}")
    
    # Entry bins
    entry_bins_special = [0, 0.3, 0.5, 0.7, 1.0]
    high_vol_high_corr['entry_bin_special'] = pd.cut(high_vol_high_corr['opposite_entry'], bins=entry_bins_special)
    entry_special = high_vol_high_corr.groupby('entry_bin_special')['opposite_pnl'].agg([
        'count', 'mean',
        lambda x: (x > 0).sum() / len(x) * 100 if len(x) > 0 else 0
    ])
    entry_special.columns = ['Count', 'Mean', 'Win Rate %']
    entry_special = entry_special[entry_special['Count'] > 0]
    if len(entry_special) > 0:
        print(f"\n  Entry Price Ranges:")
        print(entry_special.round(3))
    
    # Key correlations within this subset
    print(f"\nKey Variable Correlations in High Vol + High Corr:")
    key_subset_vars = ['time_to_expiry_minutes', 'osc_efficiency_ratio', 
                       'osc_choppiness_index', 'max_swing_distance', 'opposite_entry']
    
    for var in key_subset_vars:
        if var in high_vol_high_corr.columns and high_vol_high_corr[var].notna().sum() > 3:
            try:
                corr_val, p_val = stats.pearsonr(
                    high_vol_high_corr[var].dropna(),
                    high_vol_high_corr.loc[high_vol_high_corr[var].notna(), 'opposite_pnl']
                )
                print(f"  {var}: r={corr_val:.3f}, p={p_val:.4f}")
            except:
                pass
    
    # Recommendation
    print(f"\n{'*'*80}")
    print("RECOMMENDATION:")
    if special_mean > overall_mean and special_winrate > overall_winrate:
        print(f"  ✓ HIGH VOL + HIGH CORR conditions show BETTER performance")
        print(f"    Consider PRIORITIZING trades under these conditions")
    elif special_mean < overall_mean and special_winrate < overall_winrate:
        print(f"  ✗ HIGH VOL + HIGH CORR conditions show WORSE performance")
        print(f"    Consider AVOIDING trades under these conditions")
    else:
        print(f"  ≈ HIGH VOL + HIGH CORR conditions show MIXED results")
        print(f"    Further analysis by strategy/coin recommended")
    print(f"{'*'*80}")
    
    # Save special subset
    special_output = 'btc_opposite_high_vol_high_corr.csv'
    high_vol_high_corr.to_csv(special_output, index=False)
    print(f"\nHigh Vol + High Corr trades saved to: {special_output}")
    
else:
    print("\nNo trades found matching criteria (volatility > 0.5 AND correlation > 0.8)")
    print("Consider adjusting thresholds or checking if correlation data is available.")
