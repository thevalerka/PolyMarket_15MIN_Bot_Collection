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
CLOSE_THRESHOLD = 0.25  # Score <= 0.25 to CLOSE position
# We only BUY (go long), never SELL/short

# Trade management
TRADE_SIZE = 100  # $100 per trade
# Buffer zone: no trading in last 10 seconds before period end
# and first 30 seconds after period start
BUFFER_BEFORE_END = 10  # seconds before period end
BUFFER_AFTER_START = 30  # seconds after period start
PNL_DISPLAY_INTERVAL = 60  # Display total PNL every 60 seconds
# ============================================================================


def get_next_period_end(current_time):
    """Calculate the next 15-minute period end time (00, 15, 30, 45 minutes)"""
    # Round up to next 15-minute boundary
    next_minute = ((current_time.minute // 15) + 1) * 15
    if next_minute >= 60:
        next_hour = current_time.hour + 1
        next_minute = 0
        if next_hour >= 24:
            next_hour = 0
            next_day = current_time.day + 1
        else:
            next_day = current_time.day
    else:
        next_hour = current_time.hour
        next_day = current_time.day

    try:
        next_period_end = current_time.replace(
            day=next_day,
            hour=next_hour,
            minute=next_minute,
            second=0,
            microsecond=0
        )
    except ValueError:  # Handle month/year rollover
        next_period_end = current_time.replace(
            hour=next_hour,
            minute=next_minute,
            second=0,
            microsecond=0
        )
        next_period_end += timedelta(days=1)
        next_period_end = next_period_end.replace(day=1)

    return next_period_end


def get_current_period_start(current_time):
    """Calculate the current 15-minute period start time (00, 15, 30, 45 minutes)"""
    current_minute = (current_time.minute // 15) * 15
    return current_time.replace(minute=current_minute, second=0, microsecond=0)


def get_seconds_to_period_end(current_time):
    """Calculate seconds remaining to next 15-minute period end"""
    next_period_end = get_next_period_end(current_time)
    return (next_period_end - current_time).total_seconds()


def get_seconds_from_period_start(current_time):
    """Calculate seconds elapsed from current 15-minute period start"""
    current_period_start = get_current_period_start(current_time)
    return (current_time - current_period_start).total_seconds()


def is_in_buffer_zone(current_time):
    """Check if current time is in buffer zone (no trading allowed)"""
    seconds_to_end = get_seconds_to_period_end(current_time)
    seconds_from_start = get_seconds_from_period_start(current_time)

    # Buffer zone: last 10 seconds before period end OR
    # first 30 seconds after period start
    if seconds_to_end <= BUFFER_BEFORE_END:
        return True
    if seconds_from_start <= BUFFER_AFTER_START:
        return True
    return False


class TradeTracker:
    """Track hypothetical trades and calculate P&L"""

    def __init__(self, trades_dir: str):
        self.trades_dir = trades_dir
        os.makedirs(trades_dir, exist_ok=True)

        self.current_position = None  # None or 'BUY' (we only go long)
        self.entry_price = None
        self.entry_time = None
        self.entry_score = None
        self.closed_trades = []
        self.entry_period_end = None  # Track which period this position belongs to
        self.total_pnl = 0.0  # Running total of all P&L

    def open_position(self, price: float, score: float, timestamp: datetime):
        """Open a new BUY position"""
        if self.current_position is not None:
            return False  # Already in position

        self.current_position = 'BUY'  # Only BUY positions
        self.entry_price = price
        self.entry_time = timestamp
        self.entry_score = score
        self.entry_period_end = get_next_period_end(timestamp)

        print(f"\n  📈 OPENED BUY @ ${price:.4f} (score: {score:.3f})")
        print(f"     Period expires: {self.entry_period_end.strftime('%H:%M:%S')}")
        return True

    def close_position(self, exit_price: float, timestamp: datetime, reason: str = "normal"):
        """Close current position and calculate P&L"""
        if self.current_position is None:
            return None

        hold_time = (timestamp - self.entry_time).total_seconds()

        # Calculate P&L for binary options (only BUY positions)
        contracts = TRADE_SIZE / self.entry_price
        pnl = contracts * (exit_price - self.entry_price)

        # Update total PNL
        self.total_pnl += pnl

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
            'cumulative_pnl': round(self.total_pnl, 2),
            'exit_reason': reason,
            'trade_size': TRADE_SIZE,
            'contracts': round(contracts, 2),
            'period_end': self.entry_period_end.isoformat() if self.entry_period_end else None
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
        print(f"     Cumulative P&L: ${self.total_pnl:+.2f}")

        # Reset position
        self.current_position = None
        self.entry_price = None
        self.entry_time = None
        self.entry_score = None
        self.entry_period_end = None

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

        # Check if we need to close position before period ends
        if self.entry_period_end:
            seconds_to_period_end = (self.entry_period_end - timestamp).total_seconds()
            if seconds_to_period_end <= BUFFER_BEFORE_END:
                # Close at market bid price
                print(f"  ⏰ Closing position before period end ({seconds_to_period_end:.0f}s remaining)")
                return current_bid  # Close at bid for BUY position

        return None

    def get_unrealized_pnl(self, current_orderbook: Dict) -> Optional[float]:
        """Calculate unrealized P&L for current position"""
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

        # Calculate unrealized P&L (only for BUY positions)
        contracts = TRADE_SIZE / self.entry_price
        unrealized_pnl = contracts * (current_bid - self.entry_price)

        return unrealized_pnl

    def check_period_expiration(self, current_orderbook: Dict, timestamp: datetime) -> Optional[float]:
        """Check if current period is expiring and close position"""
        if self.current_position is None:
            return None

        seconds_to_period_end = get_seconds_to_period_end(timestamp)

        # Close position if less than buffer time to period end
        if seconds_to_period_end <= BUFFER_BEFORE_END:
            # Close at market bid price
            best_bid = current_orderbook.get('best_bid')
            if best_bid and best_bid.get('price') is not None:
                exit_price = best_bid.get('price')
                print(f"  ⏰ Period ending - closing position at bid ${exit_price:.4f}")
                return exit_price
            else:
                # Fallback to expiration logic if no bid available
                # If best_bid is NULL/NONE, option is OTM, settle at $0
                print(f"  ⏰ Period ending - no bid available, settling at $0.00")
                return 0.0

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

        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0

        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
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
        self.last_prediction_score = None  # Store last prediction score
        self.last_pnl_display = datetime.now()  # Track when we last displayed PNL

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
        print(f"BUY threshold: ≥{BUY_THRESHOLD} | CLOSE threshold: ≤{CLOSE_THRESHOLD}")
        print(f"Trade size: ${TRADE_SIZE} | Only LONG positions")
        print(f"Buffer zone: {BUFFER_BEFORE_END}s before end, {BUFFER_AFTER_START}s after start")
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

                        # Show status every iteration (every second)
                        mid_price = snapshot['mid_price']
                        total_data = len(self.predictor.training_data)
                        seconds_to_expiry = get_seconds_to_period_end(timestamp)
                        mins = int(seconds_to_expiry // 60)
                        secs = int(seconds_to_expiry % 60)

                        # Get actual bid/ask from orderbook
                        orderbook = snapshot['orderbook']
                        best_bid_obj = orderbook.get('best_bid')
                        best_ask_obj = orderbook.get('best_ask')

                        # Handle NULL/None bid/ask (happens at extreme ITM/OTM)
                        if best_bid_obj and best_bid_obj.get('price') is not None:
                            best_bid_price = best_bid_obj['price']
                            bid_str = f"${best_bid_price:.4f}"
                        else:
                            bid_str = "NULL"

                        if best_ask_obj and best_ask_obj.get('price') is not None:
                            best_ask_price = best_ask_obj['price']
                            ask_str = f"${best_ask_price:.4f}"
                        else:
                            ask_str = "NULL"

                        # Build status string with score
                        status = (f"[{timestamp.strftime('%H:%M:%S')}] "
                                 f"Data: {total_data} | "
                                 f"Bid: {bid_str} | "
                                 f"Mid: ${mid_price:.4f} | "
                                 f"Ask: {ask_str} | "
                                 f"Exp: {mins:02d}:{secs:02d}")

                        # Add buffer zone indicator
                        if is_in_buffer_zone(timestamp):
                            status += " | ⚠️ BUFFER ZONE"

                        # Add score if we have one
                        if self.last_prediction_score is not None:
                            status += f" | Score: {self.last_prediction_score:.3f}"

                        # Add position info if in trade
                        if self.trade_tracker.current_position:
                            hold_time = (timestamp - self.trade_tracker.entry_time).total_seconds()
                            unrealized_pnl = self.trade_tracker.get_unrealized_pnl(orderbook)
                            if unrealized_pnl is not None:
                                pnl_color = '\033[92m' if unrealized_pnl > 0 else '\033[91m'
                                reset_color = '\033[0m'
                                status += f" | POS: BUY @${self.trade_tracker.entry_price:.4f} ({hold_time:.0f}s) Unrealized P&L: {pnl_color}${unrealized_pnl:+.2f}{reset_color}"
                            else:
                                status += f" | POS: BUY @${self.trade_tracker.entry_price:.4f} ({hold_time:.0f}s)"

                        print(status)

                # Display cumulative PNL every 60 seconds
                if (timestamp - self.last_pnl_display).total_seconds() >= PNL_DISPLAY_INTERVAL:
                    print(f"[{timestamp.strftime('%H:%M:%S')}] 📊 Cumulative P&L: ${self.trade_tracker.total_pnl:+.2f}")
                    self.last_pnl_display = timestamp

                # Make prediction and manage trades (skip in buffer zone)
                if PREDICTION_MODE and len(self.predictor.snapshot_buffer) >= SEQUENCE_LENGTH:
                    if self.predictor.model is None:
                        if self.iteration % 100 == 0:
                            print(f"  ⚠️  No trained model - waiting for {MIN_TRAINING_SAMPLES} samples")
                    else:
                        # Skip trading in buffer zone
                        if is_in_buffer_zone(timestamp):
                            if self.iteration % 20 == 0:  # Show occasional message
                                print(f"  ⏸️  Buffer zone - no trading")
                        else:
                            prediction = self.predictor.get_current_prediction()

                            if prediction:
                                self.predictions_log.append(prediction)
                                score = prediction['prediction']
                                self.last_prediction_score = score  # Store for display
                                trade_price = prediction.get('trade_price')
                                current_orderbook = self.predictor.snapshot_buffer[-1]['orderbook']

                                # Check exit conditions first (period ending or score < CLOSE_THRESHOLD)
                                exit_price = self.trade_tracker.check_exit_conditions(current_orderbook, timestamp)
                                if exit_price:
                                    self.trade_tracker.close_position(exit_price, timestamp, "period_close")
                                elif (self.trade_tracker.current_position == 'BUY' and
                                      score <= CLOSE_THRESHOLD and
                                      current_orderbook.get('best_bid') and
                                      current_orderbook['best_bid'].get('price') is not None):
                                    # Close position due to low score
                                    exit_price = current_orderbook['best_bid']['price']
                                    self.trade_tracker.close_position(exit_price, timestamp, "low_score")

                                # Entry logic (only BUY when score > BUY_THRESHOLD and no position)
                                elif self.trade_tracker.current_position is None and score >= BUY_THRESHOLD:
                                    if trade_price is not None:
                                        self.trade_tracker.open_position(trade_price, score, timestamp)
                                    elif self.iteration % 20 == 0:  # Show occasional message
                                        bid_price = prediction.get('bid_price')
                                        if bid_price is None:
                                            print(f"  ⚠️  Score: {score:.3f} ≥ {BUY_THRESHOLD} but Ask is NULL (extreme ITM/OTM) - NO TRADE")
                                        else:
                                            # Handle case where trade_price is None but bid_price exists
                                            print(f"  🎯 Score: {score:.3f} ≥ {BUY_THRESHOLD} - Would BUY but price unavailable")

                # Check for period expiration exit
                if len(self.predictor.snapshot_buffer) >= 1:
                    current_orderbook = self.predictor.snapshot_buffer[-1]['orderbook']
                    expiration_exit_price = self.trade_tracker.check_period_expiration(current_orderbook, timestamp)
                    if expiration_exit_price is not None:
                        self.trade_tracker.close_position(expiration_exit_price, timestamp, "period_expiration")

                # Save state periodically
                if self.iteration % 100 == 0:
                    self.save_state()

                    # Show summary
                    print(f"\n{'─'*60}")
                    print(f"📊 SUMMARY (Iteration {self.iteration})")
                    print(f"   Training data: {len(self.predictor.training_data)} snapshots")
                    print(f"   Model: {'✅ Trained' if self.predictor.model else '❌ Not trained'}")

                    if self.predictions_log:
                        scores = [p['prediction'] for p in self.predictions_log[-100:]]
                        if scores:
                            print(f"   Recent scores (last 100): [{min(scores):.3f}, {max(scores):.3f}]")

                    stats = self.trade_tracker.get_stats()
                    if stats:
                        print(f"   Trades: {stats['total_trades']} | "
                              f"Wins: {stats['winning_trades']} ({stats['win_rate']*100:.1f}%) | "
                              f"Cumulative P&L: ${stats['total_pnl']:+.2f}")

                    # Print current score and unrealized P&L if in position
                    if self.predictor.model is not None and len(self.predictor.snapshot_buffer) >= SEQUENCE_LENGTH:
                        prediction = self.predictor.get_current_prediction()
                        if prediction:
                            current_score = prediction['prediction']
                            current_orderbook = self.predictor.snapshot_buffer[-1]['orderbook']

                            print(f"   Current Score: {current_score:.3f}")

                            if self.trade_tracker.current_position:
                                unrealized_pnl = self.trade_tracker.get_unrealized_pnl(current_orderbook)
                                if unrealized_pnl is not None:
                                    pnl_color = '\033[92m' if unrealized_pnl > 0 else '\033[91m'
                                    reset_color = '\033[0m'
                                    print(f"   Unrealized P&L: {pnl_color}${unrealized_pnl:+.2f}{reset_color}")

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

                exit_price = best_bid.get('price') if best_bid.get('price') is not None else 0.0
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
            scores = [p['prediction'] for p in self.predictions_log]
            high_scores = [s for s in scores if s >= BUY_THRESHOLD]
            low_scores = [s for s in scores if s <= CLOSE_THRESHOLD]
            print(f"\nPredictions: {len(scores)}")
            print(f"  High scores (≥{BUY_THRESHOLD}): {len(high_scores)} ({len(high_scores)/len(scores)*100:.1f}%)")
            print(f"  Low scores (≤{CLOSE_THRESHOLD}): {len(low_scores)} ({len(low_scores)/len(scores)*100:.1f}%)")
            print(f"  Score range: [{min(scores):.3f}, {max(scores):.3f}]")

        # Trade stats
        stats = self.trade_tracker.get_stats()
        if stats:
            print(f"\nTrades: {stats['total_trades']}")
            print(f"  Wins:  {stats['winning_trades']} ({stats['win_rate']*100:.1f}%)")
            print(f"  Losses: {stats['losing_trades']}")
            print(f"  Cumulative P&L: ${stats['total_pnl']:+.2f}")
            print(f"  Avg Win:  ${stats['avg_win']:+.2f}")
            print(f"  Avg Loss: ${stats['avg_loss']:+.2f}")


def main():
    """Main entry point"""
    predictor = LivePredictor()
    predictor.run()


if __name__ == '__main__':
    main()
