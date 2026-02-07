import json
import os
import glob
from typing import List, Dict
import pandas as pd
import numpy as np

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
    except Exception as e:
        # Return default values if parsing fails
        return {
            'distance_strike': 0,
            'minutes_expiry': 0,
            'volatility': 0
        }

def process_single_file(file_path: str) -> pd.DataFrame:
    """
    Process a single JSON file and return a DataFrame with relevant columns.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    trades = data.get('trades', [])
    processed_data = []

    for trade in trades:
        row = {}
        row['date'] = data.get('date', '')
        row['type'] = trade.get('type', '')
        row['open_price'] = trade.get('open_price', 0)
        row['edge'] = trade.get('edge', 0)
        row['pnl'] = trade.get('pnl', 0)
        row['close_reason'] = trade.get('close_reason', '')

        # Parse open_bin
        open_bin_str = trade.get('open_bin', '')
        bin_features = parse_open_bin(open_bin_str)
        row.update(bin_features)

        processed_data.append(row)

    return pd.DataFrame(processed_data)

def load_all_trades(pattern: str = "dryrun_trades_*.json") -> pd.DataFrame:
    """
    Load all trades from JSON files matching the pattern.
    """
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")

    dfs = []
    for file in files:
        try:
            df = process_single_file(file)
            dfs.append(df)
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    if not dfs:
        raise ValueError("No valid data frames created from files")

    return pd.concat(dfs, ignore_index=True)

def create_equal_clusters(series: pd.Series, n_clusters: int = 10) -> pd.Series:
    """
    Create n_clusters of equal size based on the number of trades.
    Returns cluster labels for each data point.
    """
    # Sort values and create quantile-based bins
    sorted_series = series.sort_values()
    quantiles = np.linspace(0, 1, n_clusters + 1)
    bins = sorted_series.quantile(quantiles).unique()  # Remove duplicates

    # Handle case where quantiles produce identical bins
    if len(bins) <= 1:
        return pd.Series([0] * len(series), index=series.index)

    # Create labels with range names
    labels = range(len(bins) - 1)

    # Assign clusters
    clusters = pd.cut(
        series,
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    return clusters

def analyze_cluster_performance(df: pd.DataFrame, column: str, cluster_col: str) -> pd.DataFrame:
    """
    Analyze PNL performance across clusters of a given feature.
    """
    # Group by cluster and calculate statistics
    cluster_stats = df.groupby(cluster_col)['pnl'].agg([
        'count',
        'mean',
        'sum',
        'std',
        lambda x: (x > 0).mean()  # win rate
    ]).round(6)

    cluster_stats.columns = ['trade_count', 'avg_pnl', 'total_pnl', 'pnl_std', 'win_rate']
    cluster_stats[column] = [f"{df[df[cluster_col] == i][column].min():.2f}-{df[df[cluster_col] == i][column].max():.2f}"
                            for i in cluster_stats.index]

    return cluster_stats.reset_index()

def perform_clustered_analysis(df: pd.DataFrame):
    """
    Perform clustering analysis on key features and show PNL correlations.
    """
    # Features to cluster
    features_to_cluster = ['distance_strike', 'minutes_expiry', 'volatility', 'edge']

    # Create cluster columns
    cluster_columns = {}
    for feature in features_to_cluster:
        if feature in df.columns:
            cluster_col = f'{feature}_cluster'
            df[cluster_col] = create_equal_clusters(df[feature], 10)
            cluster_columns[feature] = cluster_col

    # Show cluster analysis for each feature
    for feature, cluster_col in cluster_columns.items():
        print(f"\n=== {feature.upper()} CLUSTER ANALYSIS ===")
        cluster_performance = analyze_cluster_performance(df, feature, cluster_col)
        print(cluster_performance.to_string(index=False))

    # Overall correlation matrix
    print("\n=== OVERALL CORRELATION MATRIX ===")
    cols_of_interest = ['open_price', 'edge', 'distance_strike', 'minutes_expiry', 'volatility', 'pnl']
    available_cols = [col for col in cols_of_interest if col in df.columns]

    subset_df = df[available_cols].dropna()
    if not subset_df.empty:
        corr_matrix = subset_df.corr()
        print(corr_matrix[['pnl']].sort_values(by='pnl', ascending=False))
    else:
        print("Not enough data for correlation analysis")

def main():
    try:
        # Load all trades from JSON files
        df = load_all_trades("dryrun_trades_*.json")
        print(f"Loaded {len(df)} trades from {df['date'].nunique()} days")

        # Clean data
        df = df.dropna(subset=['pnl', 'edge', 'distance_strike', 'minutes_expiry', 'volatility'])
        print(f"After cleaning: {len(df)} trades")

        # Perform clustered analysis
        perform_clustered_analysis(df)

    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()
