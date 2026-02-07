import json
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict
from collections import defaultdict

def parse_open_bin(open_bin_str: str) -> Dict[str, float]:
    """
    Parses the open_bin string into numerical features:
    'Distance from Strike', 'Minutes to Expiry', 'Volatility'
    Returns average values for ranges.
    """
    try:
        parts = open_bin_str.split('|')

        # Distance from Strike
        dist_range = parts[0].split('-')
        distance_strike = (int(dist_range[0]) + int(dist_range[1])) / 2

        # Minutes to Expiry
        time_range = parts[1].replace('m', '').split('-')
        minutes_expiry = (int(time_range[0]) + int(time_range[1])) / 2

        # Volatility
        vol_range = parts[2].split('-')
        volatility = (int(vol_range[0]) + int(vol_range[1])) / 2

        return {
            'distance_strike': distance_strike,
            'minutes_expiry': minutes_expiry,
            'volatility': volatility
        }
    except Exception:
        # Return None to indicate parsing failure
        return None

def get_15min_period(close_time_str: str) -> str:
    """
    Extract 15-minute period from close_time string.
    Format: YYYY-MM-DDTHH:MM -> YYYY-MM-DDTHH:MM (15-min intervals)
    """
    try:
        # Parse datetime
        dt = datetime.fromisoformat(close_time_str.replace('Z', '+00:00') if close_time_str.endswith('Z') else close_time_str)
        # Get 15-minute period (0, 15, 30, 45 minutes)
        minute_period = (dt.minute // 15) * 15
        period_key = f"{dt.strftime('%Y-%m-%d %H')}:"
        if minute_period == 0:
            period_key += "00"
        elif minute_period == 15:
            period_key += "15"
        elif minute_period == 30:
            period_key += "30"
        else:  # minute_period == 45
            period_key += "45"
        return period_key
    except Exception:
        return "unknown"

def is_late_trade(open_bin_str: str) -> bool:
    """
    Check if trade is a "late" trade (not in first 2 minutes).
    Returns True if minutes to expiry is NOT 15m-13m (which represents first 2 minutes).
    """
    try:
        parts = open_bin_str.split('|')
        if len(parts) < 2:
            return False
        time_range = parts[1].replace('m', '')  # Remove 'm' suffix
        # Return True if it's NOT the first 2 minutes (15m-13m)
        return time_range != "15-13"
    except Exception:
        return True  # If we can't parse, assume it's a late trade

def process_single_buyonly_file(file_path: str) -> tuple:
    """
    Process a single buyonly JSON file and return three DataFrames:
    1. All valid trades
    2. First 8 trades per 15-minute period
    3. First 8 "late" trades per 15-minute period (excluding first 2 minutes)
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        trades = data.get('trades', [])

        # Collect all valid trades
        all_valid_trades = []

        # First pass: collect all valid trades with required fields
        for trade in trades:
            # Skip trades without required fields
            if 'open_bin' not in trade or 'edge' not in trade or 'close_time' not in trade:
                continue

            # Try to parse open_bin
            bin_features = parse_open_bin(trade.get('open_bin', ''))
            if bin_features is None:
                continue

            # Add parsed features to trade
            trade.update(bin_features)
            all_valid_trades.append(trade)

        # Group valid trades by 15-minute periods
        trades_by_period = defaultdict(list)
        late_trades_by_period = defaultdict(list)

        for trade in all_valid_trades:
            period = get_15min_period(trade['close_time'])
            trades_by_period[period].append(trade)

            # Also collect late trades (excluding first 2 minutes)
            if is_late_trade(trade['open_bin']):
                late_trades_by_period[period].append(trade)

        # Select first 8 trades per period for both categories
        first_8_trades = []
        first_8_late_trades = []

        # Regular first 8 trades
        for period, period_trades in trades_by_period.items():
            # Sort by open_time within each period
            period_trades.sort(key=lambda x: x.get('open_time', ''))
            # Take first 8 trades (or all if less than 8)
            first_8_trades.extend(period_trades[:8])

        # Late first 8 trades (excluding first 2 minutes)
        for period, period_trades in late_trades_by_period.items():
            # Sort by open_time within each period
            period_trades.sort(key=lambda x: x.get('open_time', ''))
            # Take first 8 trades (or all if less than 8)
            first_8_late_trades.extend(period_trades[:8])

        # Convert to DataFrames
        def trades_to_dataframe(trades_list, data_date):
            processed_data = []
            for trade in trades_list:
                row = {}
                row['date'] = data_date
                row['type'] = trade.get('type', '')
                row['open_time'] = trade.get('open_time', '')
                row['close_time'] = trade.get('close_time', '')
                row['open_price'] = trade.get('open_price', 0)
                row['close_price'] = trade.get('close_price', 0)
                row['edge'] = trade.get('edge', 0)
                row['pnl'] = trade.get('pnl', 0)
                row['close_reason'] = trade.get('close_reason', '')

                # Add parsed open_bin features
                row['distance_strike'] = trade.get('distance_strike', 0)
                row['minutes_expiry'] = trade.get('minutes_expiry', 0)
                row['volatility'] = trade.get('volatility', 0)

                processed_data.append(row)
            return pd.DataFrame(processed_data)

        all_df = trades_to_dataframe(all_valid_trades, data.get('date', ''))
        first_8_df = trades_to_dataframe(first_8_trades, data.get('date', ''))
        first_8_late_df = trades_to_dataframe(first_8_late_trades, data.get('date', ''))

        return all_df, first_8_df, first_8_late_df

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def load_buyonly_trades(pattern: str = "buyonly_trades_*.json") -> tuple:
    """
    Load all buyonly trades from JSON files matching the pattern.
    Returns tuple of (all_trades_df, first_8_trades_df, first_8_late_trades_df)
    """
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")

    print(f"Found {len(files)} buyonly trade files")

    all_dfs = []
    first_8_dfs = []
    first_8_late_dfs = []

    for file in files:
        try:
            all_df, first_8_df, first_8_late_df = process_single_buyonly_file(file)
            if not all_df.empty or not first_8_df.empty or not first_8_late_df.empty:
                all_dfs.append(all_df)
                first_8_dfs.append(first_8_df)
                first_8_late_dfs.append(first_8_late_df)
                print(f"Processed {file}: {len(all_df)} total, {len(first_8_df)} first-8, {len(first_8_late_df)} first-8-late")
            else:
                print(f"Skipped {file}: no valid trades")
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    # Combine all dataframes
    def combine_dfs(dfs):
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()

    combined_all_df = combine_dfs(all_dfs)
    combined_first_8_df = combine_dfs(first_8_dfs)
    combined_first_8_late_df = combine_dfs(first_8_late_dfs)

    print(f"Total combined trades - All: {len(combined_all_df)}, First 8: {len(combined_first_8_df)}, First 8 Late: {len(combined_first_8_late_df)}")
    return combined_all_df, combined_first_8_df, combined_first_8_late_df

def create_equal_clusters(series: pd.Series, n_clusters: int = 10) -> pd.Series:
    """
    Create n_clusters of equal size based on the number of trades.
    Returns cluster labels for each data point.
    """
    # Handle empty series
    if len(series) == 0:
        return pd.Series([], dtype=int)

    # Handle case where all values are the same
    if series.nunique() <= 1:
        return pd.Series([0] * len(series), index=series.index)

    # Handle case with insufficient data for clustering
    if len(series) < n_clusters:
        return pd.Series([0] * len(series), index=series.index)

    # Sort values and create quantile-based bins
    sorted_series = series.sort_values()
    quantiles = np.linspace(0, 1, n_clusters + 1)
    bins = sorted_series.quantile(quantiles).unique()

    # Handle case where quantiles produce identical bins
    if len(bins) <= 1:
        return pd.Series([0] * len(series), index=series.index)

    # Assign clusters
    clusters = pd.cut(
        series,
        bins=bins,
        labels=range(len(bins) - 1),
        include_lowest=True
    )

    return clusters

def analyze_cluster_performance(df: pd.DataFrame, column: str, cluster_col: str) -> pd.DataFrame:
    """
    Analyze PNL performance across clusters of a given feature.
    """
    if cluster_col not in df.columns or df.empty:
        return pd.DataFrame()

    # Group by cluster and calculate statistics
    cluster_stats = df.groupby(cluster_col)['pnl'].agg([
        'count',
        'mean',
        'sum',
        'std',
        lambda x: (x > 0).mean()  # win rate
    ]).round(6)

    cluster_stats.columns = ['trade_count', 'avg_pnl', 'total_pnl', 'pnl_std', 'win_rate']

    # Add range information
    ranges = []
    for i in cluster_stats.index:
        if len(df[df[cluster_col] == i]) > 0:
            min_val = df[df[cluster_col] == i][column].min()
            max_val = df[df[cluster_col] == i][column].max()
            ranges.append(f"{min_val:.2f}-{max_val:.2f}")
        else:
            ranges.append("N/A")

    cluster_stats[column] = ranges
    cluster_stats = cluster_stats.reset_index()

    return cluster_stats

def perform_buyonly_clustered_analysis(df: pd.DataFrame, title_suffix: str = ""):
    """
    Perform clustering analysis on key features for buyonly trades.
    """
    if df.empty:
        print("No data available for analysis")
        return

    title = f"BUYONLY TRADES ANALYSIS {title_suffix}".strip()
    print(f"\n=== {title} ===")
    print(f"Total trades analyzed: {len(df)}")

    # Features to cluster
    features_to_cluster = ['distance_strike', 'minutes_expiry', 'volatility', 'edge']

    # Create cluster columns
    cluster_columns = {}
    for feature in features_to_cluster:
        if (feature in df.columns and
            not df[feature].isnull().all() and
            df[feature].nunique() > 1 and
            len(df) >= 10):  # Need at least 10 trades for 10 clusters
            try:
                cluster_col = f'{feature}_cluster'
                df[cluster_col] = create_equal_clusters(df[feature], min(10, len(df)))
                cluster_columns[feature] = cluster_col
            except Exception:
                continue

    # Show cluster analysis for each feature
    for feature, cluster_col in cluster_columns.items():
        if cluster_col in df.columns:
            print(f"\n=== {feature.upper()} CLUSTER ANALYSIS ===")
            cluster_performance = analyze_cluster_performance(df, feature, cluster_col)
            if not cluster_performance.empty:
                print(cluster_performance.to_string(index=False))

    # Overall correlation matrix
    print("\n=== OVERALL CORRELATION MATRIX ===")
    cols_of_interest = ['open_price', 'edge', 'distance_strike', 'minutes_expiry', 'volatility', 'pnl']
    available_cols = [col for col in cols_of_interest if col in df.columns and not df[col].isnull().all()]

    if available_cols and len(df) > 1:
        subset_df = df[available_cols].dropna()
        if not subset_df.empty and len(subset_df) > 1:
            try:
                corr_matrix = subset_df.corr()
                if 'pnl' in corr_matrix.columns:
                    print(corr_matrix[['pnl']].sort_values(by='pnl', ascending=False).to_string())
                else:
                    print("PNL column not found in correlation matrix")
            except Exception as e:
                print(f"Could not compute correlation matrix: {e}")
        else:
            print("Not enough data for correlation analysis")
    else:
        print("No suitable columns for correlation analysis")

def compare_analyses(all_df: pd.DataFrame, first_8_df: pd.DataFrame, first_8_late_df: pd.DataFrame):
    """
    Compare the analyses side by side
    """
    print("\n" + "="*100)
    print("COMPARISON SUMMARY")
    print("="*100)
    print(f"All Trades: {len(all_df)} trades")
    print(f"First 8 Trades: {len(first_8_df)} trades")
    print(f"First 8 Late Trades (excl. 1st 2 mins): {len(first_8_late_df)} trades")

    # Calculate total PNL for each category
    all_pnl = all_df['pnl'].sum() if not all_df.empty else 0
    first_8_pnl = first_8_df['pnl'].sum() if not first_8_df.empty else 0
    first_8_late_pnl = first_8_late_df['pnl'].sum() if not first_8_late_df.empty else 0

    print(f"\nTotal PNL Comparison:")
    print(f"  All Trades - Total PNL: {all_pnl:.6f}, Avg PNL: {all_df['pnl'].mean() if not all_df.empty else 0:.6f}, Win Rate: {(all_df['pnl'] > 0).mean() if not all_df.empty else 0:.2%}")
    print(f"  First 8 Trades - Total PNL: {first_8_pnl:.6f}, Avg PNL: {first_8_df['pnl'].mean() if not first_8_df.empty else 0:.6f}, Win Rate: {(first_8_df['pnl'] > 0).mean() if not first_8_df.empty else 0:.2%}")
    print(f"  First 8 Late Trades - Total PNL: {first_8_late_pnl:.6f}, Avg PNL: {first_8_late_df['pnl'].mean() if not first_8_late_df.empty else 0:.6f}, Win Rate: {(first_8_late_df['pnl'] > 0).mean() if not first_8_late_df.empty else 0:.2%}")

def main():
    try:
        # Load all buyonly trades from JSON files
        all_df, first_8_df, first_8_late_df = load_buyonly_trades("buyonly_trades_*.json")

        if not all_df.empty or not first_8_df.empty or not first_8_late_df.empty:
            # Analyze all trades
            if not all_df.empty:
                perform_buyonly_clustered_analysis(all_df, "(ALL TRADES)")

            # Analyze first 8 trades
            if not first_8_df.empty:
                perform_buyonly_clustered_analysis(first_8_df, "(FIRST 8 PER PERIOD)")

            # Analyze first 8 late trades
            if not first_8_late_df.empty:
                perform_buyonly_clustered_analysis(first_8_late_df, "(FIRST 8 LATE PER PERIOD)")

            # Compare analyses
            compare_analyses(all_df, first_8_df, first_8_late_df)

            # Optional: Save processed data
            # if not all_df.empty:
            #     all_df.to_csv("buyonly_all_trades.csv", index=False)
            # if not first_8_df.empty:
            #     first_8_df.to_csv("buyonly_first8_trades.csv", index=False)
            # if not first_8_late_df.empty:
            #     first_8_late_df.to_csv("buyonly_first8_late_trades.csv", index=False)
        else:
            print("No valid trades found in any files")

    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
