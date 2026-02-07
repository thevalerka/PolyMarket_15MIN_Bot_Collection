#!/usr/bin/env python3
"""
Live Order Book Data Collector and Predictor with Trade Tracking
Continuously collects order book snapshots, trains model, makes predictions, and tracks P&L
"""

import sys
import time
import json
import pickle
import os
from datetime import datetime, timedelta
from typing import Dict, Optional
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
TRADES_DIR = DATA_DIR + 'bot024_MMBook_LSTM/trades/'

# Trading thresholds
BUY_THRESHOLD = 0.75   # Score >= 0.75 to BUY
SELL_THRESHOLD = 0.25  # Score <= 0.25 to SELL
# Score between 0.25 and 0.75 = HOLD

# Trade management
MAX_POSITION_HOLD_TIME = 300  # Max 5 minutes (300 seconds)
TRADE_SIZE = 100  # $100 per trade
# ============================================================================


class TradeTracker:
    """Track hypothetical trades and calculate P&L"""
    
    def __init__(self, trades_dir: str):
        self.trades_dir = trades_dir
        os.makedirs(trades_dir, exist_ok=True)
        
        self.current_position = None  # None, 'BUY', or 'SELL'
        self.entry_price = None
        self.entry_time = None
        self.entry_score = None
        self.closed_trades = []
        
    def open_position(self, signal: str, price: float, score: float, timestamp: datetime):
        """Open a new position"""
        if self.current_position is not None:
            return False  # Already in position
        
        self.current_position = signal  # 'BUY' or 'SELL'
        self.entry_price = price
        self.entry_time = timestamp
        self.entry_score = score
        
        print(f"\n  📈 OPENED {signal} @ ${price:.4f} (score: {score:.3f})")
        return True
    
    def close_position(self, exit_price: float, timestamp: datetime, reason: str = "normal"):
        """Close current position and calculate P&L"""
        if self.current_position is None:
            return None
        
        hold_time = (timestamp - self.entry_time).total_seconds()
        
        # Calculate P&L for binary options
        if self.current_position == 'BUY':
            # Long: bought at entry_price
            contracts = TRADE_SIZE / self.entry_price
            pnl = contracts * (exit_price - self.entry_price)
        else:  # SELL
            # Short: sold at entry_price
            contracts = TRADE_SIZE / self.entry_price
            pnl = contracts * (self.entry_price - exit_price)
        
        # Create trade record
        trade = {
            'entry_time': self.entry_time.isoformat(),
            'exit_time': timestamp.isoformat(),
            'signal': self.current_position,
            'entry_price': round(self.entry_price, 4),
            'exit_price': round(exit_price, 4),
            'entry_score': round(self.entry_score, 3),
            'hold_time_seconds': round(hold_time, 1),
            'pnl': round(pnl, 2),
            'exit_reason': reason,
            'trade_size': TRADE_SIZE,
            'contracts': round(contracts, 2)
        }
        
        self.closed_trades.append(trade)
        
        # Log to daily file
        self._log_trade(trade)
        
        # Print trade result
        pnl_color = '\033[92m' if pnl > 0 else '\033[91m'
        reset_color = '\033[0m'
        print(f"  📉 CLOSED {self.current_position} @ ${exit_price:.4f}")
        print(f"     P&L: {pnl_color}${pnl:+.2f}{reset_color} | "
              f"Hold: {hold_time:.0f}s | Reason: {reason}")
        
        # Reset position
        self.current_position = None
        self.entry_price = None
        self.entry_time = None
        self.entry_score = None
        
        return trade
    
    def check_exit_conditions(self, current_orderbook: Dict, timestamp: datetime) -> Optional[float]:
        """Check if we should exit current position"""
        if self.current_position is None:
            return None
        
        # Get current prices
        best_bid = current_orderbook.get('best_bid')
        best_ask = current_orderbook.get('best_ask')
        
        if not best_bid or not best_ask:
            return None
        
        current_bid = best_bid.get('price')
        current_ask = best_ask.get('price')
        
        if current_bid is None or current_ask is None:
            return None
        
        # Check time limit
        hold_time = (timestamp - self.entry_time).total_seconds()
        if hold_time >= MAX_POSITION_HOLD_TIME:
            # Exit at market price
            return current_bid if self.current_position == 'BUY' else current_ask
        
        # Check expiration proximity (from orderbook or snapshot)
        seconds_to_expiry = 999
        if hasattr(current_orderbook, 'get'):
            seconds_to_expiry = current_orderbook.get('seconds_to_expiry', 999)
        
        if seconds_to_expiry <= 30:
            return current_bid if self.current_position == 'BUY' else current_ask
        
        return None
    
    def _log_trade(self, trade: Dict):
        """Log trade to daily file"""
        try:
            date_str = datetime.fromisoformat(trade['entry_time']).strftime('%Y%m%d')
            log_file = os.path.join(self.trades_dir, f"trades_{date_str}.jsonl")
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(trade) + '\n')
        except Exception as e:
            print(f"⚠️  Error logging trade: {e}")
    
    def get_stats(self) -> Optional[Dict]:
        """Get trading statistics"""
        if not self.closed_trades:
            return None
        
        total_trades = len(self.closed_trades)
        winning_trades = [t for t in self.closed_trades if t['pnl'] > 0]
        losing_trades = [t for t in self.closed_trades if t['pnl'] < 0]
        
        total_pnl = sum(t['pnl'] for t in self.closed_trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'current_position': self.current_position
        }


class LivePredictor:
    """Live prediction system with continuous data collection and trade tracking"""
    
    def __init__(self):
        self.predictor = OrderBookPredictor()
        self.last_retrain = datetime.now()
        self.predictions_log = []
        self.iteration = 0
        self.state_file = DATA_DIR + 'live_predictor_state.pkl'
        
        # Initialize trade tracker
        self.trade_tracker = TradeTracker(TRADES_DIR)
        
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
            with open(self.state_file, 'wb') as f:
                pickle.dump(state, f)
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
            
            training_data = state.get('training_data', [])
            self.last_retrain = state.get('last_retrain', datetime.now())
            self.predictions_log = state.get('predictions_log', [])
            self.iteration = state.get('iteration', 0)
            saved_time = state.get('timestamp', 'unknown')
            
            print(f"✅ State loaded: {len(training_data)} snapshots from {saved_time}")
            
            self.predictor.training_data = training_data
            
            if self.predictor.training_data:
                recent_data = self.predictor.training_data[-SEQUENCE_LENGTH:]
                self.predictor.snapshot_buffer.clear()
                for snapshot in recent_data:
                    self.predictor.snapshot_buffer.append(snapshot)
                print(f"   Restored buffer: {len(self.predictor.snapshot_buffer)} snapshots")
            
        except Exception as e:
            print(f"⚠️  Error loading state: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """Main loop"""
        print("\n" + "="*80)
        print("🚀 LIVE ORDER BOOK PREDICTOR WITH TRADE TRACKING")
        print("="*80)
        print(f"Collection: {COLLECTION_MODE} | Prediction: {PREDICTION_MODE}")
        print(f"BUY threshold: ≥{BUY_THRESHOLD} | SELL threshold: ≤{SELL_THRESHOLD}")
        print(f"Trade size: ${TRADE_SIZE} | Max hold: {MAX_POSITION_HOLD_TIME}s")
        print(f"Trades log: {TRADES_DIR}")
        print("="*80 + "\n")
        
        try:
            while True:
                self.iteration += 1
                timestamp = datetime.now()
                
                # Collect snapshot
                if COLLECTION_MODE:
                    success = self.predictor.collect_snapshot(CALL_FILE)
                    
                    if success:
                        snapshot = self.predictor.snapshot_buffer[-1].copy()
                        self.predictor.training_data.append(snapshot)
                        
                        # Show status every 10 iterations
                        if self.iteration % 10 == 0:
                            mid_price = snapshot['mid_price']
                            total_data = len(self.predictor.training_data)
                            seconds_to_expiry = snapshot.get('seconds_to_expiry', 0)
                            mins = seconds_to_expiry // 60
                            secs = seconds_to_expiry % 60
                            
                            # Get actual bid/ask from orderbook
                            orderbook = snapshot['orderbook']
                            best_bid_price = orderbook.get('best_bid', {}).get('price', 'N/A')
                            best_ask_price = orderbook.get('best_ask', {}).get('price', 'N/A')
                            
                            status = (f"[{timestamp.strftime('%H:%M:%S')}] "
                                     f"Data: {total_data} | "
                                     f"Bid: ${best_bid_price:.4f if isinstance(best_bid_price, float) else best_bid_price} | "
                                     f"Mid: ${mid_price:.4f} | "
                                     f"Ask: ${best_ask_price:.4f if isinstance(best_ask_price, float) else best_ask_price} | "
                                     f"Exp: {mins:02d}:{secs:02d}")
                            
                            # Add position info if in trade
                            if self.trade_tracker.current_position:
                                hold_time = (timestamp - self.trade_tracker.entry_time).total_seconds()
                                status += f" | POS: {self.trade_tracker.current_position} @${self.trade_tracker.entry_price:.4f} ({hold_time:.0f}s)"
                            
                            print(status)
                
                # Make prediction and manage trades
                if PREDICTION_MODE and len(self.predictor.snapshot_buffer) >= SEQUENCE_LENGTH:
                    if self.predictor.model is None:
                        if self.iteration % 100 == 0:
                            print(f"  ⚠️  No trained model - waiting for {MIN_TRAINING_SAMPLES} samples")
                    else:
                        prediction = self.predictor.get_current_prediction()
                        
                        if prediction:
                            self.predictions_log.append(prediction)
                            current_orderbook = self.predictor.snapshot_buffer[-1]['orderbook']
                            
                            # Check exit conditions first
                            exit_price = self.trade_tracker.check_exit_conditions(current_orderbook, timestamp)
                            if exit_price:
                                self.trade_tracker.close_position(exit_price, timestamp, "auto_exit")
                            
                            # Entry logic
                            if self.trade_tracker.current_position is None:
                                score = prediction['prediction']
                                signal = prediction['signal']
                                trade_price = prediction.get('trade_price')
                                
                                # Show prediction details every 20 iterations
                                if self.iteration % 20 == 0:
                                    signal_color = {
                                        'BUY': '\033[92m',
                                        'SELL': '\033[91m',
                                        'HOLD': '\033[93m'
                                    }
                                    color = signal_color.get(signal, '')
                                    reset = '\033[0m'
                                    
                                    print(f"  🎯 Score: {score:.3f} → {color}{signal}{reset} "
                                          f"(BUY≥{BUY_THRESHOLD}, SELL≤{SELL_THRESHOLD})")
                                    
                                    if signal != 'HOLD' and trade_price:
                                        print(f"     Would trade @ ${trade_price:.4f}")
                                
                                # Execute trade
                                if signal == 'BUY' and trade_price:
                                    self.trade_tracker.open_position('BUY', trade_price, score, timestamp)
                                elif signal == 'SELL' and trade_price:
                                    self.trade_tracker.open_position('SELL', trade_price, score, timestamp)
                
                # Save state periodically
                if self.iteration % 100 == 0:
                    self.save_state()
                    
                    # Show summary
                    print(f"\n{'─'*60}")
                    print(f"📊 SUMMARY (Iteration {self.iteration})")
                    print(f"   Training data: {len(self.predictor.training_data)} snapshots")
                    print(f"   Model: {'✅ Trained' if self.predictor.model else '❌ Not trained'}")
                    
                    if self.predictions_log:
                        recent_preds = self.predictions_log[-100:]
                        signals = [p['signal'] for p in recent_preds]
                        scores = [p['prediction'] for p in recent_preds]
                        print(f"   Recent predictions (last 100):")
                        print(f"     BUY:  {signals.count('BUY')} ({signals.count('BUY')/len(signals)*100:.1f}%)")
                        print(f"     SELL: {signals.count('SELL')} ({signals.count('SELL')/len(signals)*100:.1f}%)")
                        print(f"     HOLD: {signals.count('HOLD')} ({signals.count('HOLD')/len(signals)*100:.1f}%)")
                        print(f"     Score range: [{min(scores):.3f}, {max(scores):.3f}]")
                    
                    stats = self.trade_tracker.get_stats()
                    if stats:
                        print(f"   Trades: {stats['total_trades']} | "
                              f"Wins: {stats['winning_trades']} ({stats['win_rate']*100:.1f}%) | "
                              f"P&L: ${stats['total_pnl']:+.2f}")
                    
                    print(f"{'─'*60}\n")
                
                # Training logic
                total_samples = len(self.predictor.training_data)
                
                if (self.predictor.model is None and 
                    total_samples >= MIN_TRAINING_SAMPLES and 
                    total_samples % 100 == 0):
                    
                    print(f"\n{'='*60}")
                    print(f"🎓 INITIAL TRAINING (samples: {total_samples})")
                    print(f"{'='*60}")
                    self.retrain_model()
                
                elif self.predictor.model is not None:
                    time_since_retrain = (datetime.now() - self.last_retrain).total_seconds()
                    
                    if (time_since_retrain >= RETRAIN_INTERVAL and 
                        total_samples >= MIN_TRAINING_SAMPLES):
                        
                        print(f"\n{'='*60}")
                        print(f"⏰ RETRAINING (samples: {total_samples})")
                        print(f"{'='*60}")
                        self.retrain_model()
                        self.last_retrain = datetime.now()
                
                time.sleep(SNAPSHOT_INTERVAL)
        
        except KeyboardInterrupt:
            print("\n\n" + "="*80)
            print("🛑 SHUTTING DOWN")
            print("="*80)
            
            # Close any open position
            if self.trade_tracker.current_position:
                current_orderbook = self.predictor.snapshot_buffer[-1]['orderbook']
                best_bid = current_orderbook.get('best_bid', {})
                best_ask = current_orderbook.get('best_ask', {})
                
                exit_price = best_bid.get('price') if self.trade_tracker.current_position == 'BUY' else best_ask.get('price')
                if exit_price:
                    self.trade_tracker.close_position(exit_price, datetime.now(), "shutdown")
            
            # Save state
            print("\n💾 Saving state...")
            self.save_state()
            self.predictor.save_history()
            
            # Show stats
            self.show_stats()
            print("="*80)
            sys.exit(0)
    
    def retrain_model(self):
        """Retrain the model"""
        print(f"\n{'='*60}")
        print(f"📚 PREPARING TRAINING DATA")
        print(f"   Total snapshots: {len(self.predictor.training_data)}")
        print(f"{'='*60}")
        
        sequences, targets = self.predictor.prepare_training_data(max_samples=3000)
        
        if sequences is not None and len(sequences) >= 100:
            print(f"\n✅ Training data ready:")
            print(f"   Sequences: {len(sequences)}")
            print(f"   Shape: {sequences.shape}")
            print(f"   Target range: [{targets.min():.3f}, {targets.max():.3f}]")
            print(f"   Target mean: {targets.mean():.3f}")
            
            try:
                self.predictor.train_model(sequences, targets)
                print(f"\n✅ Training complete")
            except Exception as e:
                print(f"\n❌ Training error: {e}")
                import traceback
                traceback.print_exc()
        else:
            samples = len(sequences) if sequences is not None else 0
            print(f"\n⚠️  Insufficient samples: {samples}")
            print(f"{'='*60}")
    
    def show_stats(self):
        """Show trading and prediction statistics"""
        print("\n📊 STATISTICS:")
        
        # Prediction stats
        if self.predictions_log:
            signals = [p['signal'] for p in self.predictions_log]
            print(f"\nPredictions: {len(signals)}")
            print(f"  BUY:  {signals.count('BUY')} ({signals.count('BUY')/len(signals)*100:.1f}%)")
            print(f"  SELL: {signals.count('SELL')} ({signals.count('SELL')/len(signals)*100:.1f}%)")
            print(f"  HOLD: {signals.count('HOLD')} ({signals.count('HOLD')/len(signals)*100:.1f}%)")
        
        # Trade stats
        stats = self.trade_tracker.get_stats()
        if stats:
            print(f"\nTrades: {stats['total_trades']}")
            print(f"  Wins:  {stats['winning_trades']} ({stats['win_rate']*100:.1f}%)")
            print(f"  Losses: {stats['losing_trades']}")
            print(f"  Total P&L: ${stats['total_pnl']:+.2f}")
            print(f"  Avg Win:  ${stats['avg_win']:+.2f}")
            print(f"  Avg Loss: ${stats['avg_loss']:+.2f}")


def main():
    """Main entry point"""
    predictor = LivePredictor()
    predictor.run()


if __name__ == '__main__':
    main()
