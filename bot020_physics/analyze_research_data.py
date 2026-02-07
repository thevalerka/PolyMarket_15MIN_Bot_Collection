#!/usr/bin/env python3
"""
Research Data Analysis Script
Analyze collected data to find predictive relationships
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import glob
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "/home/ubuntu/013_2025_polymarket/research_data"


def load_data(date: str = None) -> pd.DataFrame:
    """Load data from JSONL files"""
    if date:
        files = [f"{DATA_DIR}/data_{date}.jsonl"]
    else:
        files = sorted(glob.glob(f"{DATA_DIR}/data_*.jsonl"))
    
    data = []
    for file in files:
        try:
            with open(file, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    df = pd.DataFrame(data)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df


def calculate_oscillation_metrics(df: pd.DataFrame, window: int = 300) -> pd.DataFrame:
    """Calculate oscillation amplitude and frequency"""
    df = df.copy()
    
    # Rolling max and min for oscillation amplitude
    for col in ['call_mid', 'put_mid', 'btc_spot']:
        if col in df.columns:
            df[f'{col}_amplitude'] = (
                df[col].rolling(window).max() - df[col].rolling(window).min()
            )
            
            # Oscillation frequency (zero-crossings of detrended data)
            detrended = df[col] - df[col].rolling(window).mean()
            df[f'{col}_oscillations'] = (
                (detrended.shift(1) * detrended < 0).rolling(window).sum()
            )
    
    return df


def correlation_analysis(df: pd.DataFrame, target: str = 'call_mid_amplitude'):
    """Analyze correlations with target variable"""
    print(f"\n{'='*80}")
    print(f"CORRELATION ANALYSIS: {target}")
    print(f"{'='*80}\n")
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Remove target and related columns
    feature_cols = [col for col in numeric_cols 
                   if col != target and not col.endswith('_amplitude')]
    
    # Calculate correlations
    correlations = df[feature_cols + [target]].corr()[target].sort_values(ascending=False)
    
    print("Top 20 Positive Correlations:")
    print("-" * 80)
    for feat, corr in correlations.head(20).items():
        if feat != target:
            print(f"{feat:40s}: {corr:+.4f}")
    
    print("\nTop 20 Negative Correlations:")
    print("-" * 80)
    for feat, corr in correlations.tail(20).items():
        if feat != target:
            print(f"{feat:40s}: {corr:+.4f}")
    
    return correlations


def regime_analysis(df: pd.DataFrame):
    """Analyze market regimes"""
    print(f"\n{'='*80}")
    print("MARKET REGIME ANALYSIS")
    print(f"{'='*80}\n")
    
    # Define regimes
    df['high_vol_regime'] = df['btc_realized_vol'] > df['btc_realized_vol'].quantile(0.75)
    df['high_entropy_regime'] = df['btc_entropy'] > df['btc_entropy'].quantile(0.75)
    df['trending_regime'] = df['btc_hurst'] > 0.55
    df['mean_reverting_regime'] = df['btc_hurst'] < 0.45
    
    # Calculate metrics by regime
    regimes = {
        'High Volatility': df['high_vol_regime'],
        'High Entropy': df['high_entropy_regime'],
        'Trending': df['trending_regime'],
        'Mean Reverting': df['mean_reverting_regime']
    }
    
    for regime_name, regime_mask in regimes.items():
        if regime_mask.sum() > 0:
            print(f"{regime_name} Regime:")
            print(f"  Samples: {regime_mask.sum()} ({regime_mask.sum()/len(df)*100:.1f}%)")
            
            # Average metrics in this regime
            if 'call_mid_amplitude' in df.columns:
                print(f"  Avg CALL amplitude: ${df.loc[regime_mask, 'call_mid_amplitude'].mean():.4f}")
            if 'put_mid_amplitude' in df.columns:
                print(f"  Avg PUT amplitude: ${df.loc[regime_mask, 'put_mid_amplitude'].mean():.4f}")
            if 'call_gamma' in df.columns:
                print(f"  Avg Gamma: {df.loc[regime_mask, 'call_gamma'].mean():.6f}")
            print()


def gamma_hypothesis_test(df: pd.DataFrame):
    """Test: Higher gamma → larger oscillations"""
    print(f"\n{'='*80}")
    print("GAMMA HYPOTHESIS TEST")
    print(f"{'='*80}\n")
    
    if 'call_gamma' not in df.columns or 'call_mid_amplitude' not in df.columns:
        print("Required columns not found")
        return
    
    # Remove NaN values
    valid_data = df[['call_gamma', 'call_mid_amplitude']].dropna()
    
    if len(valid_data) < 10:
        print("Insufficient data")
        return
    
    # Calculate correlation
    corr = valid_data.corr().iloc[0, 1]
    print(f"Correlation (Gamma vs Amplitude): {corr:+.4f}")
    
    # Split into quartiles
    valid_data['gamma_quartile'] = pd.qcut(valid_data['call_gamma'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    
    print("\nAmplitude by Gamma Quartile:")
    print("-" * 80)
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        amp = valid_data[valid_data['gamma_quartile'] == q]['call_mid_amplitude'].mean()
        print(f"  {q}: ${amp:.4f}")
    
    # Hypothesis: Q4 > Q1
    q4_amp = valid_data[valid_data['gamma_quartile'] == 'Q4']['call_mid_amplitude'].mean()
    q1_amp = valid_data[valid_data['gamma_quartile'] == 'Q1']['call_mid_amplitude'].mean()
    
    if q4_amp > q1_amp:
        print(f"\n✓ Hypothesis SUPPORTED: High gamma → larger amplitude")
        print(f"  Q4 amplitude is {(q4_amp/q1_amp - 1)*100:.1f}% higher than Q1")
    else:
        print(f"\n✗ Hypothesis NOT SUPPORTED")


def entropy_hypothesis_test(df: pd.DataFrame):
    """Test: Higher entropy → more oscillations"""
    print(f"\n{'='*80}")
    print("ENTROPY HYPOTHESIS TEST")
    print(f"{'='*80}\n")
    
    if 'btc_entropy' not in df.columns or 'btc_spot_oscillations' not in df.columns:
        print("Required columns not found")
        return
    
    valid_data = df[['btc_entropy', 'btc_spot_oscillations']].dropna()
    
    if len(valid_data) < 10:
        print("Insufficient data")
        return
    
    corr = valid_data.corr().iloc[0, 1]
    print(f"Correlation (Entropy vs Oscillations): {corr:+.4f}")
    
    # Split into high/low entropy
    median_entropy = valid_data['btc_entropy'].median()
    high_entropy = valid_data['btc_entropy'] > median_entropy
    
    high_osc = valid_data[high_entropy]['btc_spot_oscillations'].mean()
    low_osc = valid_data[~high_entropy]['btc_spot_oscillations'].mean()
    
    print(f"\nHigh Entropy: {high_osc:.2f} oscillations per 5min")
    print(f"Low Entropy: {low_osc:.2f} oscillations per 5min")
    
    if high_osc > low_osc:
        print(f"\n✓ Hypothesis SUPPORTED: High entropy → more oscillations")


def time_decay_analysis(df: pd.DataFrame):
    """Analyze theta decay impact"""
    print(f"\n{'='*80}")
    print("TIME DECAY ANALYSIS")
    print(f"{'='*80}\n")
    
    if 'time_to_expiry_seconds' not in df.columns:
        print("Required columns not found")
        return
    
    # Group by time remaining
    df['minutes_to_expiry'] = df['time_to_expiry_seconds'] / 60
    df['time_bucket'] = pd.cut(df['minutes_to_expiry'], 
                                bins=[0, 3, 6, 9, 12, 15], 
                                labels=['0-3', '3-6', '6-9', '9-12', '12-15'])
    
    print("Metrics by Time to Expiry:")
    print("-" * 80)
    
    for bucket in ['12-15', '9-12', '6-9', '3-6', '0-3']:
        mask = df['time_bucket'] == bucket
        if mask.sum() > 0:
            print(f"\n{bucket} minutes to expiry:")
            if 'call_mid' in df.columns:
                print(f"  Avg CALL price: ${df.loc[mask, 'call_mid'].mean():.4f}")
            if 'put_mid' in df.columns:
                print(f"  Avg PUT price: ${df.loc[mask, 'put_mid'].mean():.4f}")
            if 'call_spread' in df.columns:
                print(f"  Avg CALL spread: ${df.loc[mask, 'call_spread'].mean():.4f}")


def summary_statistics(df: pd.DataFrame):
    """Print summary statistics"""
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")
    
    print(f"Total snapshots: {len(df)}")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Duration: {(df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600:.1f} hours")
    
    print("\nPrice Ranges:")
    for col in ['btc_spot', 'call_mid', 'put_mid']:
        if col in df.columns:
            print(f"  {col}: ${df[col].min():.2f} - ${df[col].max():.2f}")
    
    print("\nVolatility Ranges:")
    for col in ['btc_realized_vol', 'call_iv', 'put_iv']:
        if col in df.columns:
            print(f"  {col}: {df[col].min():.2%} - {df[col].max():.2%}")
    
    print("\nGreeks Ranges:")
    for col in ['call_gamma', 'call_vega', 'call_theta']:
        if col in df.columns:
            print(f"  {col}: {df[col].min():.6f} - {df[col].max():.6f}")


def main():
    """Main analysis pipeline"""
    print("=" * 80)
    print("POLYMARKET RESEARCH DATA ANALYSIS")
    print("=" * 80)
    
    # Load data
    print("\n📊 Loading data...")
    df = load_data()
    
    if len(df) == 0:
        print("❌ No data found. Run data collector first!")
        return
    
    print(f"✓ Loaded {len(df)} snapshots")
    
    # Calculate oscillation metrics
    print("\n📈 Calculating oscillation metrics...")
    df = calculate_oscillation_metrics(df)
    
    # Run analyses
    summary_statistics(df)
    correlation_analysis(df, 'call_mid_amplitude')
    regime_analysis(df)
    gamma_hypothesis_test(df)
    entropy_hypothesis_test(df)
    time_decay_analysis(df)
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}\n")
    
    return df


if __name__ == "__main__":
    df = main()
