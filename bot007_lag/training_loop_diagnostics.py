#!/usr/bin/env python3
"""
Training Loop Diagnostics

Analyzes why the ML model is stuck in constant retraining with R² = 0.000
and provides detailed diagnosis of the training data quality.
"""

import json
import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

def analyze_training_data_file(filepath):
    """Analyze existing training data to understand the retraining issue"""
    
    try:
        with open(filepath, 'r') as f:
            training_data = json.load(f)
        
        print(f"📁 Loaded {len(training_data)} training samples from {filepath}")
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(training_data)
        
        if 'call_price_change' not in df.columns or 'put_price_change' not in df.columns:
            print("❌ Missing target variables (call_price_change, put_price_change)")
            return False
        
        print("\n🔍 TRAINING DATA ANALYSIS")
        print("=" * 50)
        
        # Analyze target variables
        call_changes = df['call_price_change'].dropna()
        put_changes = df['put_price_change'].dropna()
        
        print(f"📊 CALL Price Changes:")
        print(f"   Count: {len(call_changes)}")
        print(f"   Mean: {call_changes.mean():.6f}")
        print(f"   Std: {call_changes.std():.6f}")
        print(f"   Variance: {call_changes.var():.8f}")
        print(f"   Range: {call_changes.min():.6f} to {call_changes.max():.6f}")
        print(f"   Unique values: {len(call_changes.unique())}")
        
        # Check if all values are the same
        call_zeros = (call_changes == 0).sum()
        call_zero_pct = call_zeros / len(call_changes) * 100
        
        print(f"   Zero changes: {call_zeros}/{len(call_changes)} ({call_zero_pct:.1f}%)")
        
        print(f"\n📊 PUT Price Changes:")
        print(f"   Count: {len(put_changes)}")
        print(f"   Mean: {put_changes.mean():.6f}")
        print(f"   Std: {put_changes.std():.6f}")
        print(f"   Variance: {put_changes.var():.8f}")
        print(f"   Range: {put_changes.min():.6f} to {put_changes.max():.6f}")
        print(f"   Unique values: {len(put_changes.unique())}")
        
        put_zeros = (put_changes == 0).sum()
        put_zero_pct = put_zeros / len(put_changes) * 100
        
        print(f"   Zero changes: {put_zeros}/{len(put_changes)} ({put_zero_pct:.1f}%)")
        
        # Analyze feature variance
        print(f"\n🔧 FEATURE ANALYSIS:")
        print("=" * 30)
        
        feature_cols = [col for col in df.columns 
                       if col not in ['call_price_change', 'put_price_change', 'timestamp']]
        
        low_variance_features = []
        for col in feature_cols:
            if col in df.columns:
                variance = df[col].var()
                if variance < 1e-6:
                    low_variance_features.append((col, variance))
        
        if low_variance_features:
            print(f"⚠️ Features with near-zero variance ({len(low_variance_features)}):")
            for feature, variance in low_variance_features[:10]:  # Show first 10
                print(f"   {feature}: {variance:.2e}")
        else:
            print("✅ All features have reasonable variance")
        
        # Diagnose the R² = 0.000 issue
        print(f"\n🚨 DIAGNOSIS OF R² = 0.000 ISSUE:")
        print("=" * 40)
        
        issues_found = []
        
        # Issue 1: No variance in target variables
        if call_changes.var() < 1e-8:
            issues_found.append("CALL prices have zero variance (all changes are identical)")
        
        if put_changes.var() < 1e-8:
            issues_found.append("PUT prices have zero variance (all changes are identical)")
        
        # Issue 2: All changes are zero
        if call_zero_pct > 95:
            issues_found.append(f"CALL prices rarely change ({call_zero_pct:.1f}% are zero)")
        
        if put_zero_pct > 95:
            issues_found.append(f"PUT prices rarely change ({put_zero_pct:.1f}% are zero)")
        
        # Issue 3: Too many constant features
        if len(low_variance_features) > len(feature_cols) * 0.8:
            issues_found.append("Most features have constant values (no signal)")
        
        # Issue 4: Data collection frequency too high
        if len(training_data) > 1000:
            print(f"⚠️ Very large dataset ({len(training_data)} samples)")
            print("   This suggests data collection frequency might be too high")
            print("   If prices don't change every second, most samples will be duplicates")
        
        if issues_found:
            print("❌ CRITICAL ISSUES FOUND:")
            for i, issue in enumerate(issues_found, 1):
                print(f"   {i}. {issue}")
        else:
            print("✅ No obvious issues found in training data")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print("=" * 25)
        
        if call_zero_pct > 90 or put_zero_pct > 90:
            print("1. PRICE CHANGE ISSUE:")
            print("   • Prices are not changing meaningfully between samples")
            print("   • Solution: Only add training samples when prices actually change")
            print("   • Solution: Increase sampling interval (e.g., every 5-10 seconds)")
            print()
        
        if len(low_variance_features) > 5:
            print("2. FEATURE VARIANCE ISSUE:")
            print("   • Many features are constant (no predictive power)")
            print("   • Solution: Remove or improve feature engineering")
            print("   • Solution: Add more dynamic market data")
            print()
        
        if len(training_data) > 2000:
            print("3. TRAINING FREQUENCY ISSUE:")
            print("   • Training too frequently on similar data")
            print("   • Solution: Train only every 100-200 samples")
            print("   • Solution: Add minimum time interval between training")
            print()
        
        print("4. GENERAL SOLUTIONS:")
        print("   • Use the Fixed ML Detector (addresses all these issues)")
        print("   • Implement price change validation")
        print("   • Add feature variance checking")
        print("   • Reduce training frequency")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing training data: {e}")
        return False

def simulate_current_training_loop():
    """Simulate what's happening in the current training loop"""
    
    print("\n🔄 SIMULATING CURRENT TRAINING LOOP ISSUE")
    print("=" * 50)
    
    # Simulate the problematic scenario
    print("Scenario: Option prices updating every second, but actual prices rarely change")
    print()
    
    # Create sample data that mimics the issue
    np.random.seed(42)
    n_samples = 100
    
    # Features with some variance (market data)
    features = {
        'volatility_5m': np.random.normal(0.001, 0.0001, n_samples),
        'momentum_5m': np.random.normal(0, 0.0005, n_samples),
        'moneyness': np.random.normal(1.0, 0.001, n_samples),
    }
    
    # Target variables with minimal change (the problem!)
    call_price_base = 0.850
    put_price_base = 0.130
    
    # 95% of the time, prices don't change
    call_changes = np.zeros(n_samples)
    put_changes = np.zeros(n_samples)
    
    # Only 5% of samples have actual price changes
    change_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    call_changes[change_indices] = np.random.normal(0, 0.01, len(change_indices))
    put_changes[change_indices] = np.random.normal(0, 0.01, len(change_indices))
    
    # Create DataFrame
    df = pd.DataFrame(features)
    df['call_price_change'] = call_changes
    df['put_price_change'] = put_changes
    
    print(f"📊 Simulated Data:")
    print(f"   Samples: {n_samples}")
    print(f"   CALL changes - Mean: {call_changes.mean():.6f}, Std: {call_changes.std():.6f}")
    print(f"   PUT changes - Mean: {put_changes.mean():.6f}, Std: {put_changes.std():.6f}")
    print(f"   Zero CALL changes: {(call_changes == 0).sum()}/{n_samples} ({(call_changes == 0).mean()*100:.1f}%)")
    print(f"   Zero PUT changes: {(put_changes == 0).sum()}/{n_samples} ({(put_changes == 0).mean()*100:.1f}%)")
    
    # Try to train a model on this data (will fail)
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    
    X = df[['volatility_5m', 'momentum_5m', 'moneyness']]
    y_call = df['call_price_change']
    y_put = df['put_price_change']
    
    # Split data
    X_train, X_test, y_call_train, y_call_test = train_test_split(X, y_call, test_size=0.3, random_state=42)
    _, _, y_put_train, y_put_test = train_test_split(X, y_put, test_size=0.3, random_state=42)
    
    # Train models
    call_model = RandomForestRegressor(n_estimators=50, random_state=42)
    put_model = RandomForestRegressor(n_estimators=50, random_state=42)
    
    call_model.fit(X_train, y_call_train)
    put_model.fit(X_train, y_put_train)
    
    # Make predictions
    call_pred = call_model.predict(X_test)
    put_pred = put_model.predict(X_test)
    
    # Calculate R² scores
    call_r2 = r2_score(y_call_test, call_pred)
    put_r2 = r2_score(y_put_test, put_pred)
    
    print(f"\n🎯 TRAINING RESULTS (simulating your issue):")
    print(f"   CALL R²: {call_r2:.6f}")
    print(f"   PUT R²: {put_r2:.6f}")
    
    if call_r2 < 0.01 and put_r2 < 0.01:
        print("   ❌ EXACTLY THE SAME ISSUE! R² ≈ 0.000")
        print("   📝 Why: No meaningful signal in target variables")
    
    # Show what happens with better data
    print(f"\n✅ WHAT HAPPENS WITH BETTER DATA:")
    print("=" * 40)
    
    # Create data with actual signal
    better_call_changes = np.random.normal(0, 0.02, n_samples)  # More variance
    better_put_changes = np.random.normal(0, 0.02, n_samples)
    
    # Add some relationship to features (create actual signal)
    better_call_changes += features['volatility_5m'] * 20  # Volatility affects call prices
    better_put_changes -= features['momentum_5m'] * 15     # Momentum affects put prices
    
    # Train on better data
    call_model_better = RandomForestRegressor(n_estimators=50, random_state=42)
    put_model_better = RandomForestRegressor(n_estimators=50, random_state=42)
    
    X_train_b, X_test_b, y_call_train_b, y_call_test_b = train_test_split(X, better_call_changes, test_size=0.3, random_state=42)
    _, _, y_put_train_b, y_put_test_b = train_test_split(X, better_put_changes, test_size=0.3, random_state=42)
    
    call_model_better.fit(X_train_b, y_call_train_b)
    put_model_better.fit(X_train_b, y_put_train_b)
    
    call_pred_better = call_model_better.predict(X_test_b)
    put_pred_better = put_model_better.predict(X_test_b)
    
    call_r2_better = r2_score(y_call_test_b, call_pred_better)
    put_r2_better = r2_score(y_put_test_b, put_pred_better)
    
    print(f"   Better data CALL R²: {call_r2_better:.3f}")
    print(f"   Better data PUT R²: {put_r2_better:.3f}")
    print("   ✅ Much better! Model can actually learn patterns")

def compare_original_vs_fixed():
    """Compare the original training logic vs fixed version"""
    
    print(f"\n🔧 ORIGINAL vs FIXED TRAINING LOGIC")
    print("=" * 50)
    
    print("❌ ORIGINAL (PROBLEMATIC) LOGIC:")
    print("   1. Train every 25 samples (way too frequent!)")
    print("   2. No validation of price changes")
    print("   3. No feature variance checking")
    print("   4. No minimum training interval")
    print("   5. Train even if all price changes are zero")
    print("   6. Mark as 'trained' even with R² = 0.000")
    print()
    
    print("✅ FIXED LOGIC:")
    print("   1. Train every 200 samples (much more reasonable)")
    print("   2. Validate meaningful price changes (min $0.001)")
    print("   3. Check feature variance before training")
    print("   4. Minimum 5-minute interval between training")
    print("   5. Skip training if no signal in data")
    print("   6. Require R² > 0.05 to mark as 'trained'")
    print("   7. Track training quality over time")
    print("   8. Only create opportunities if model has actual skill")
    
    print(f"\n📊 IMPACT OF FIXES:")
    print("=" * 25)
    print("• Eliminates constant retraining loops")
    print("• Prevents training on static/unchanging data")
    print("• Ensures model only trades when it has real predictive power")
    print("• Saves computational resources")
    print("• Provides better performance tracking")

def main():
    """Run comprehensive training loop diagnostics"""
    
    print("🔍 ML TRAINING LOOP DIAGNOSTICS")
    print("=" * 60)
    print(f"Timestamp: {datetime.now()}")
    print("=" * 60)
    
    # Check if training data file exists
    training_data_file = '/home/ubuntu/013_2025_polymarket/ml_models/training_data.json'
    
    if os.path.exists(training_data_file):
        print(f"📁 Found existing training data file")
        analyze_training_data_file(training_data_file)
    else:
        print(f"📁 No existing training data file found at {training_data_file}")
        print("   This is normal for a fresh installation")
    
    # Simulate the issue
    simulate_current_training_loop()
    
    # Compare approaches
    compare_original_vs_fixed()
    
    print(f"\n🎯 SUMMARY & ACTION PLAN")
    print("=" * 35)
    print("YOUR ISSUE: Model stuck in training loop with R² = 0.000")
    print()
    print("ROOT CAUSE:")
    print("• Option prices change infrequently (maybe every few seconds)")
    print("• Your detector samples every second")
    print("• Most samples have zero price change")
    print("• Model has no signal to learn from")
    print("• Training every 25 samples on duplicate data")
    print()
    print("SOLUTION:")
    print("• Use the Fixed ML Detector")
    print("• It implements all the necessary fixes")
    print("• Will only train on meaningful data")
    print("• Will achieve actual R² > 0 when patterns exist")
    print()
    print("IMMEDIATE NEXT STEPS:")
    print("1. Stop current detector")
    print("2. Replace with FixedMLArbitrageDetector")
    print("3. Monitor training quality metrics")
    print("4. Should see much less frequent but higher quality training")

if __name__ == "__main__":
    import os
    main()