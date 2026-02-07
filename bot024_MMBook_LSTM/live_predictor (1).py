#!/usr/bin/env python3
"""
Live Order Book Data Collector and Predictor
Continuously collects order book snapshots, trains model, and makes predictions
"""

import sys
import time
import json
import pickle
import os
from datetime import datetime, timedelta
sys.path.insert(0, '/home/ubuntu/013_2025_polymarket')

from orderbook_ml_predictor import OrderBookPredictor, DATA_DIR, SNAPSHOT_INTERVAL
from orderbook_ml_predictor import SEQUENCE_LENGTH, MAX_HOLDING_PERIOD

# ============================================================================
# CONFIGURATION
# ============================================================================
COLLECTION_MODE = True  # Set to True to collect data
PREDICTION_MODE = True  # Set to True to make predictions
RETRAIN_INTERVAL = 3600  # Retrain every hour (in seconds)
MIN_TRAINING_SAMPLES = 500  # Minimum samples before training

CALL_FILE = DATA_DIR + '15M_CALL_nonoise.json'
# ============================================================================


class LivePredictor:
    """Live prediction system with continuous data collection"""
    
    def __init__(self):
        self.predictor = OrderBookPredictor()
        self.last_retrain = datetime.now()
        self.predictions_log = []
        self.iteration = 0
        self.state_file = DATA_DIR + 'live_predictor_state.pkl'
        
        # Load previous state if exists
        self.load_state()
    
    def save_state(self):
        """Save current state to disk"""
        state = {
            'training_data': self.predictor.training_data,
            'last_retrain': self.last_retrain,
            'predictions_log': self.predictions_log,
            'iteration': self.iteration,
            'timestamp': datetime.now()
        }
        
        try:
            import pickle
            with open(self.state_file, 'wb') as f:
                pickle.dump(state, f)
            print(f"💾 State saved: {len(self.predictor.training_data)} snapshots, {len(self.predictions_log)} predictions")
        except Exception as e:
            print(f"⚠️  Error saving state: {e}")
    
    def load_state(self):
        """Load previous state from disk"""
        
        if not os.path.exists(self.state_file):
            print("ℹ️  No previous state found, starting fresh")
            return
        
        try:
            print(f"📦 Loading state from {self.state_file}...")
            with open(self.state_file, 'rb') as f:
                state = pickle.load(f)
            
            # Load the state data
            training_data = state.get('training_data', [])
            self.last_retrain = state.get('last_retrain', datetime.now())
            self.predictions_log = state.get('predictions_log', [])
            self.iteration = state.get('iteration', 0)
            saved_time = state.get('timestamp', 'unknown')
            
            print(f"✅ State loaded: {len(training_data)} snapshots from {saved_time}")
            
            # IMPORTANT: Actually assign the training data to the predictor
            self.predictor.training_data = training_data
            
            # Rebuild snapshot buffer from last SEQUENCE_LENGTH samples
            if self.predictor.training_data:
                recent_data = self.predictor.training_data[-SEQUENCE_LENGTH:]
                self.predictor.snapshot_buffer.clear()
                for snapshot in recent_data:
                    self.predictor.snapshot_buffer.append(snapshot)
                print(f"   Restored buffer: {len(self.predictor.snapshot_buffer)} snapshots")
            
            print(f"   Iteration counter: {self.iteration}")
            print(f"   Predictions logged: {len(self.predictions_log)}")
            
        except Exception as e:
            print(f"⚠️  Error loading state: {e}")
            print(f"   Starting fresh")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """Main loop"""
        print("\n" + "="*80)
        print("🚀 STARTING LIVE ORDER BOOK PREDICTOR")
        print("="*80)
        print(f"Collection mode: {COLLECTION_MODE}")
        print(f"Prediction mode: {PREDICTION_MODE}")
        print(f"Retrain interval: {RETRAIN_INTERVAL}s ({RETRAIN_INTERVAL/3600:.1f} hours)")
        print(f"Snapshot interval: {SNAPSHOT_INTERVAL}s")
        print(f"Press Ctrl+C to stop")
        print("="*80 + "\n")
        
        try:
            while True:
                self.iteration += 1
                timestamp = datetime.now()
                
                # Collect snapshot
                if COLLECTION_MODE:
                    success = self.predictor.collect_snapshot(CALL_FILE)
                    
                    if success:
                        # Add to training data
                        snapshot = self.predictor.snapshot_buffer[-1].copy()
                        self.predictor.training_data.append(snapshot)
                        
                        # Show collection status every 10 iterations
                        if self.iteration % 10 == 0:
                            mid_price = snapshot['mid_price']
                            buffer_size = len(self.predictor.snapshot_buffer)
                            total_data = len(self.predictor.training_data)
                            seconds_to_expiry = snapshot.get('seconds_to_expiry', 0)
                            minutes_to_expiry = seconds_to_expiry // 60
                            secs_remainder = seconds_to_expiry % 60
                            
                            print(f"[{timestamp.strftime('%H:%M:%S')}] "
                                  f"Collected: {total_data} | "
                                  f"Buffer: {buffer_size}/{SEQUENCE_LENGTH} | "
                                  f"Mid: ${mid_price:.4f} | "
                                  f"Expiry: {minutes_to_expiry:02d}:{secs_remainder:02d}")
                
                # Make prediction
                if PREDICTION_MODE and len(self.predictor.snapshot_buffer) >= SEQUENCE_LENGTH:
                    # Check if model is trained
                    if self.predictor.model is None:
                        if self.iteration % 100 == 0:
                            print(f"  ⚠️  No trained model - predictions disabled")
                            print(f"     Collect {MIN_TRAINING_SAMPLES} samples to enable auto-training")
                    else:
                        prediction = self.predictor.get_current_prediction()
                        
                        if prediction:
                            self.predictions_log.append(prediction)
                            
                            # Show prediction every 20 iterations
                            if self.iteration % 20 == 0:
                                signal_color = {
                                    'BUY': '\033[92m',   # Green
                                    'SELL': '\033[91m',  # Red
                                    'HOLD': '\033[93m'   # Yellow
                                }
                                reset_color = '\033[0m'
                                
                                color = signal_color.get(prediction['signal'], '')
                                print(f"  🎯 Prediction: {color}{prediction['signal']}{reset_color} "
                                      f"(score: {prediction['prediction']:+.3f}, "
                                      f"price: ${prediction['mid_price']:.4f})")
                
                # Save state periodically (every 100 iterations = ~100 seconds)
                if self.iteration % 100 == 0:
                    self.save_state()
                
                # Check if we should do initial training
                total_samples = len(self.predictor.training_data)
                
                if (self.predictor.model is None and 
                    total_samples >= MIN_TRAINING_SAMPLES and 
                    total_samples % 100 == 0):  # Check every 100 samples
                    
                    print(f"\n{'='*60}")
                    print(f"🎓 INITIAL TRAINING TRIGGERED")
                    print(f"   Samples collected: {total_samples}")
                    print(f"{'='*60}")
                    
                    self.retrain_model()
                
                # Check if we should retrain (only if model exists)
                elif self.predictor.model is not None:
                    time_since_retrain = (datetime.now() - self.last_retrain).total_seconds()
                    
                    if (time_since_retrain >= RETRAIN_INTERVAL and 
                        total_samples >= MIN_TRAINING_SAMPLES):
                        
                        print(f"\n{'='*60}")
                        print(f"⏰ Retraining model...")
                        print(f"   Total samples: {total_samples}")
                        print(f"   Time since last train: {time_since_retrain/3600:.1f} hours")
                        print(f"{'='*60}")
                        
                        self.retrain_model()
                        self.last_retrain = datetime.now()
                
                # Sleep until next snapshot
                time.sleep(SNAPSHOT_INTERVAL)
        
        except KeyboardInterrupt:
            print("\n\n" + "="*80)
            print("🛑 STOPPING - User interrupted")
            print(f"Total iterations: {self.iteration}")
            print(f"Total data collected: {len(self.predictor.training_data)}")
            print(f"Total predictions made: {len(self.predictions_log)}")
            
            # Save state before exiting
            print("\n💾 Saving state...")
            self.save_state()
            
            # Save everything
            self.predictor.save_history()
            
            # Show prediction statistics
            if self.predictions_log:
                self.show_prediction_stats()
            
            print("="*80)
            sys.exit(0)
    
    def retrain_model(self):
        """Retrain the model with accumulated data"""
        sequences, targets = self.predictor.prepare_training_data()
        
        if sequences is not None and len(sequences) >= MIN_TRAINING_SAMPLES:
            try:
                self.predictor.train_model(sequences, targets)
                print(f"✅ Model retrained successfully")
            except Exception as e:
                print(f"❌ Error during training: {e}")
        else:
            print(f"⚠️  Not enough samples for training: {len(sequences) if sequences is not None else 0}")
    
    def show_prediction_stats(self):
        """Show statistics about predictions made"""
        print("\n📊 PREDICTION STATISTICS:")
        
        signals = [p['signal'] for p in self.predictions_log]
        buy_count = signals.count('BUY')
        sell_count = signals.count('SELL')
        hold_count = signals.count('HOLD')
        
        total = len(signals)
        print(f"  Total predictions: {total}")
        print(f"  BUY:  {buy_count} ({buy_count/total*100:.1f}%)")
        print(f"  SELL: {sell_count} ({sell_count/total*100:.1f}%)")
        print(f"  HOLD: {hold_count} ({hold_count/total*100:.1f}%)")
        
        # Average prediction scores
        scores = [p['prediction'] for p in self.predictions_log]
        avg_score = sum(scores) / len(scores)
        print(f"  Average score: {avg_score:+.3f}")


def main():
    """Main entry point"""
    predictor = LivePredictor()
    predictor.run()


if __name__ == '__main__':
    main()
