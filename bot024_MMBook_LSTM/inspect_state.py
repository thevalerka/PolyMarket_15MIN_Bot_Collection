#!/usr/bin/env python3
"""
Inspect and repair the state file
"""

import pickle
import os

STATE_FILE = '/home/ubuntu/013_2025_polymarket/live_predictor_state.pkl'

print("="*80)
print("🔍 STATE FILE INSPECTOR")
print("="*80)

if not os.path.exists(STATE_FILE):
    print(f"\n❌ State file not found: {STATE_FILE}")
    exit(1)

# Check file size
size_mb = os.path.getsize(STATE_FILE) / (1024 * 1024)
print(f"\n📊 File Info:")
print(f"   Path: {STATE_FILE}")
print(f"   Size: {size_mb:.2f} MB")

# Try to load it
try:
    print(f"\n📦 Loading state file...")
    with open(STATE_FILE, 'rb') as f:
        state = pickle.load(f)
    
    print(f"✅ State loaded successfully!")
    print(f"\n📋 State Contents:")
    
    for key, value in state.items():
        if key == 'training_data':
            print(f"   {key}: {len(value)} snapshots")
            if len(value) > 0:
                print(f"      First snapshot timestamp: {value[0].get('timestamp', 'N/A')}")
                print(f"      Last snapshot timestamp: {value[-1].get('timestamp', 'N/A')}")
                print(f"      Sample keys: {list(value[0].keys())}")
        elif key == 'predictions_log':
            print(f"   {key}: {len(value)} predictions")
        elif key == 'timestamp':
            print(f"   {key}: {value}")
        elif key == 'iteration':
            print(f"   {key}: {value}")
        elif key == 'last_retrain':
            print(f"   {key}: {value}")
        else:
            print(f"   {key}: {type(value)}")
    
    # Check if training_data has the required fields
    if state.get('training_data') and len(state['training_data']) > 0:
        print(f"\n✅ Training data is valid!")
        print(f"   Total snapshots: {len(state['training_data'])}")
        
        # Test if we can create a predictor and load this
        print(f"\n🧪 Testing state loading in predictor...")
        
        import sys
        sys.path.insert(0, '/home/ubuntu/013_2025_polymarket')
        from orderbook_ml_predictor import OrderBookPredictor
        
        predictor = OrderBookPredictor()
        predictor.training_data = state['training_data']
        
        print(f"   Loaded {len(predictor.training_data)} snapshots into predictor")
        
        # Try to prepare training data
        sequences, targets = predictor.prepare_training_data()
        
        if sequences is not None:
            print(f"   ✅ Can prepare {len(sequences)} training samples!")
            print(f"   Sequence shape: {sequences.shape}")
            
            # Offer to train
            response = input(f"\n💡 Do you want to train the model now? (y/n): ")
            if response.lower() == 'y':
                print(f"\n🎓 Training model...")
                predictor.train_model(sequences, targets)
                print(f"\n✅ Model trained and saved!")
        else:
            print(f"   ❌ Cannot prepare training samples")
    
except Exception as e:
    print(f"\n❌ Error loading state: {e}")
    import traceback
    traceback.print_exc()
    
    print(f"\n🔧 Attempting to diagnose...")
    
    # Try to inspect the pickle without fully loading
    try:
        with open(STATE_FILE, 'rb') as f:
            unpickler = pickle.Unpickler(f)
            
            # This might give us more info
            print(f"   Pickle protocol: {unpickler.load.__code__.co_code[:10]}")
    except:
        pass

print("\n" + "="*80)
