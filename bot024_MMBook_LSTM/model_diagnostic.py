#!/usr/bin/env python3
"""
Diagnostic script to check model status and optionally force training
"""

import sys
import os
sys.path.insert(0, '/home/ubuntu/013_2025_polymarket')

from orderbook_ml_predictor import OrderBookPredictor, MODEL_DIR
import torch

def check_model_status():
    """Check if model exists and is properly loaded"""
    print("\n" + "="*80)
    print("🔍 MODEL DIAGNOSTIC")
    print("="*80 + "\n")
    
    predictor = OrderBookPredictor()
    
    # Check model file
    model_path = os.path.join(MODEL_DIR, 'orderbook_lstm.pth')
    if os.path.exists(model_path):
        print(f"✅ Model file exists: {model_path}")
        size = os.path.getsize(model_path) / 1024
        print(f"   Size: {size:.2f} KB")
    else:
        print(f"❌ Model file NOT found: {model_path}")
    
    # Check if model is loaded
    if predictor.model is not None:
        print(f"✅ Model loaded in memory")
        print(f"   Feature size: {predictor.feature_size}")
        print(f"   Parameters: {sum(p.numel() for p in predictor.model.parameters()):,}")
        
        # Test prediction
        import numpy as np
        test_seq = np.random.randn(SEQUENCE_LENGTH, predictor.feature_size).astype(np.float32)
        test_pred = predictor.predict(test_seq)
        print(f"\n   Test prediction: {test_pred:.6f}")
        
        if abs(test_pred) < 0.001:
            print(f"   ⚠️  WARNING: Model outputs near-zero predictions!")
            print(f"   This suggests the model needs training or is poorly initialized")
    else:
        print(f"❌ Model NOT loaded in memory")
    
    # Check scaler
    if predictor.scaler_mean is not None:
        print(f"✅ Scaler loaded")
        print(f"   Mean shape: {predictor.scaler_mean.shape}")
    else:
        print(f"❌ Scaler NOT loaded")
    
    # Check training data
    print(f"\n📊 Training Data:")
    print(f"   Collected snapshots: {len(predictor.training_data)}")
    print(f"   Buffer size: {len(predictor.snapshot_buffer)}")
    
    # Check state file
    state_file = '/home/ubuntu/013_2025_polymarket/live_predictor_state.pkl'
    if os.path.exists(state_file):
        size = os.path.getsize(state_file) / 1024
        print(f"✅ State file exists: {size:.2f} KB")
    else:
        print(f"⚠️  No state file (first run)")
    
    print("\n" + "="*80)
    
    return predictor


def force_initial_training(predictor):
    """Force training if enough data available"""
    print("\n" + "="*80)
    print("🎓 ATTEMPTING INITIAL TRAINING")
    print("="*80 + "\n")
    
    if len(predictor.training_data) < 120:
        print(f"❌ Not enough data: {len(predictor.training_data)} < 120")
        print(f"   Continue collecting data...")
        return False
    
    print(f"Preparing training data from {len(predictor.training_data)} snapshots...")
    sequences, targets = predictor.prepare_training_data()
    
    if sequences is None or len(sequences) < 10:
        print(f"❌ Could not prepare training samples")
        return False
    
    print(f"✅ Prepared {len(sequences)} training samples")
    print(f"   Sequence shape: {sequences.shape}")
    print(f"   Target shape: {targets.shape}")
    
    # Train
    try:
        predictor.train_model(sequences, targets)
        print(f"\n✅ Training complete!")
        
        # Test the model
        test_pred = predictor.predict(sequences[0])
        print(f"   Test prediction after training: {test_pred:.6f}")
        
        return True
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Model diagnostic and training tool')
    parser.add_argument('--train', action='store_true', help='Force initial training if enough data')
    args = parser.parse_args()
    
    predictor = check_model_status()
    
    if args.train:
        force_initial_training(predictor)
    elif predictor.model is None:
        print("\n💡 TIP: Run with --train to force initial training")
        print("   python3 model_diagnostic.py --train")


if __name__ == '__main__':
    from orderbook_ml_predictor import SEQUENCE_LENGTH
    main()
