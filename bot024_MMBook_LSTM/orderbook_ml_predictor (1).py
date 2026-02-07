#!/usr/bin/env python3
"""
Order Book Price Movement Prediction System
Uses LSTM neural network to predict price movements from order book sequences
Trains on historical P&L feedback
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
import os
import pickle
from typing import List, Dict, Tuple, Optional

# Deep learning imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️  PyTorch not installed. Install with: pip3 install torch --break-system-packages")
    TORCH_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = '/home/ubuntu/013_2025_polymarket/'
MODEL_DIR = DATA_DIR + 'models/'
HISTORY_DIR = DATA_DIR + 'history/'

# Data collection
SEQUENCE_LENGTH = 60  # Use last 20 order book snapshots
TOP_N_LEVELS = 20  # Use top 20 bids and asks
SNAPSHOT_INTERVAL = 0.1  # Collect snapshot every 1 second
PRICE_RANGE = (0.10, 0.90)  # Only use data when price is in this range

# Binary option parameters
OPTION_PERIOD_MINUTES = 15  # Options expire every 15 minutes
EXPIRATION_MINUTES = [0, 15, 30, 45]  # Expiration at :00, :15, :30, :45
SETTLEMENT_PRICE_ITM = 1.0  # In-the-money settlement
SETTLEMENT_PRICE_OTM = 0.0  # Out-of-the-money settlement
NEAR_EXPIRY_THRESHOLD = 10  # Seconds before expiry to predict settlement
ITM_PROBABILITY_THRESHOLD = 0.5  # If price > 0.5, likely settles at $1

# Model architecture (reduced for memory efficiency)
HIDDEN_SIZE = 64  # Reduced from 128
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.001
BATCH_SIZE = 16  # Reduced from 32
EPOCHS = 30  # Reduced from 50

# Trading simulation for P&L feedback
TRADE_SIZE = 100  # Fixed trade size in dollars
MAX_HOLDING_PERIOD = 60  # Max hold for 60 seconds (or until expiry)
HOLDING_PERIOD = 60  # Max hold for 60 seconds (or until expiry)
# ============================================================================


class OrderBookFeatureExtractor:
    """Extract features from order book snapshot"""

    @staticmethod
    def get_next_expiration(timestamp: datetime) -> datetime:
        """Get the next option expiration time"""
        current_minute = timestamp.minute

        # Find next expiration minute
        next_exp_minute = None
        for exp_min in EXPIRATION_MINUTES:
            if current_minute < exp_min:
                next_exp_minute = exp_min
                break

        # If past all expirations this hour, next is :00 of next hour
        if next_exp_minute is None:
            next_exp = timestamp.replace(minute=0, second=0, microsecond=0)
            next_exp += timedelta(hours=1)
        else:
            next_exp = timestamp.replace(minute=next_exp_minute, second=0, microsecond=0)

        return next_exp

    @staticmethod
    def seconds_to_expiry(timestamp: datetime) -> int:
        """Calculate seconds until next expiration"""
        next_exp = OrderBookFeatureExtractor.get_next_expiration(timestamp)
        seconds = (next_exp - timestamp).total_seconds()
        return int(seconds)

    @staticmethod
    def predict_settlement_price(orderbook: Dict, seconds_to_expiry: int) -> Optional[float]:
        """
        Predict settlement price based on current book state
        Returns 1.0 (ITM) or 0.0 (OTM) if near expiry, None otherwise
        """
        if seconds_to_expiry > NEAR_EXPIRY_THRESHOLD:
            return None

        # Near expiry - predict settlement
        best_ask = orderbook.get('best_ask')
        best_bid = orderbook.get('best_bid')

        # Use ask price if available (more reliable near expiry)
        if best_ask and best_ask['price'] is not None:
            if best_ask['price'] > ITM_PROBABILITY_THRESHOLD:
                return SETTLEMENT_PRICE_ITM
            else:
                return SETTLEMENT_PRICE_OTM

        # Fall back to bid price
        if best_bid and best_bid['price'] is not None:
            if best_bid['price'] > ITM_PROBABILITY_THRESHOLD:
                return SETTLEMENT_PRICE_ITM
            else:
                return SETTLEMENT_PRICE_OTM

        # If both null, try mid price
        mid_price = OrderBookFeatureExtractor.get_mid_price(orderbook)
        if mid_price is not None:
            if mid_price > ITM_PROBABILITY_THRESHOLD:
                return SETTLEMENT_PRICE_ITM
            else:
                return SETTLEMENT_PRICE_OTM

        # Cannot determine
        return None

    @staticmethod
    def extract_features(orderbook: Dict, top_n: int = TOP_N_LEVELS) -> Optional[np.ndarray]:
        """
        Extract feature vector from order book
        Returns None if price is out of range
        """
        if not orderbook or 'complete_book' not in orderbook:
            return None

        # Check price range
        mid_price = OrderBookFeatureExtractor.get_mid_price(orderbook)
        if mid_price is None or not (PRICE_RANGE[0] < mid_price < PRICE_RANGE[1]):
            return None

        bids = orderbook['complete_book']['bids']
        asks = orderbook['complete_book']['asks']

        # Sort and get top N
        bids_sorted = sorted(bids, key=lambda x: float(x['price']), reverse=True)[:top_n]
        asks_sorted = sorted(asks, key=lambda x: float(x['price']))[:top_n]

        features = []

        # 1. Bid features (price and size for top N)
        for i in range(top_n):
            if i < len(bids_sorted):
                features.append(float(bids_sorted[i]['price']))
                features.append(float(bids_sorted[i]['size']))
            else:
                features.append(0.0)
                features.append(0.0)

        # 2. Ask features (price and size for top N)
        for i in range(top_n):
            if i < len(asks_sorted):
                features.append(float(asks_sorted[i]['price']))
                features.append(float(asks_sorted[i]['size']))
            else:
                features.append(0.0)
                features.append(0.0)

        # 3. Aggregate features
        features.extend(OrderBookFeatureExtractor.calculate_aggregate_features(
            bids_sorted, asks_sorted, mid_price
        ))

        return np.array(features, dtype=np.float32)

    @staticmethod
    def get_mid_price(orderbook: Dict) -> Optional[float]:
        """Calculate mid price"""
        if orderbook.get('best_bid') and orderbook.get('best_ask'):
            return (orderbook['best_bid']['price'] + orderbook['best_ask']['price']) / 2
        return None

    @staticmethod
    def calculate_aggregate_features(bids: List[Dict], asks: List[Dict],
                                     mid_price: float) -> List[float]:
        """Calculate aggregate order book features"""
        features = []

        # Spread
        if bids and asks:
            best_bid = float(bids[0]['price'])
            best_ask = float(asks[0]['price'])
            spread = best_ask - best_bid
            spread_bps = (spread / mid_price) * 10000 if mid_price > 0 else 0
        else:
            spread = 0
            spread_bps = 0

        features.append(spread)
        features.append(spread_bps)

        # Total volume
        bid_volume = sum(float(b['size']) for b in bids)
        ask_volume = sum(float(a['size']) for a in asks)
        total_volume = bid_volume + ask_volume

        features.append(bid_volume)
        features.append(ask_volume)
        features.append(total_volume)

        # Volume imbalance
        volume_imbalance = (bid_volume - ask_volume) / (total_volume + 1e-8)
        features.append(volume_imbalance)

        # Weighted mid price
        if bid_volume + ask_volume > 0:
            weighted_mid = (best_bid * ask_volume + best_ask * bid_volume) / (bid_volume + ask_volume)
        else:
            weighted_mid = mid_price
        features.append(weighted_mid)

        # Price impact (depth at different levels)
        for level in [5, 10, 20]:
            bid_depth = sum(float(b['size']) for b in bids[:min(level, len(bids))])
            ask_depth = sum(float(a['size']) for a in asks[:min(level, len(asks))])
            features.append(bid_depth)
            features.append(ask_depth)

        return features


class OrderBookDataset(Dataset):
    """PyTorch Dataset for order book sequences"""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class OrderBookLSTM(nn.Module):
    """LSTM model for order book prediction"""

    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
        super(OrderBookLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)

        # Attention weights
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)

        # Fully connected layers
        out = self.relu(self.fc1(context))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.tanh(self.fc3(out))  # Output between -1 and 1

        return out


class OrderBookPredictor:
    """Main predictor class that manages data collection, training, and inference"""

    def __init__(self):
        self.feature_extractor = OrderBookFeatureExtractor()
        self.snapshot_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.training_data = []

        # Create directories
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(HISTORY_DIR, exist_ok=True)

        self.model = None
        self.feature_size = None
        self.scaler_mean = None
        self.scaler_std = None

        # Load existing model if available
        self.load_model()

    def collect_snapshot(self, filepath: str) -> bool:
        """Collect a single order book snapshot"""
        try:
            with open(filepath, 'r') as f:
                orderbook = json.load(f)

            timestamp = datetime.now()

            # Calculate time to expiry
            seconds_to_expiry = self.feature_extractor.seconds_to_expiry(timestamp)

            # Skip if too close to expiry (unstable market)
            if seconds_to_expiry < 30:
                return False

            features = self.feature_extractor.extract_features(orderbook)

            if features is not None:
                mid_price = self.feature_extractor.get_mid_price(orderbook)

                self.snapshot_buffer.append({
                    'timestamp': timestamp,
                    'features': features,
                    'mid_price': mid_price,
                    'orderbook': orderbook,
                    'seconds_to_expiry': seconds_to_expiry
                })
                return True

        except Exception as e:
            print(f"Error collecting snapshot: {e}")

        return False

    def calculate_pnl(self, entry_snapshot: Dict, exit_snapshot: Dict,
                      direction: str) -> float:
        """
        Calculate P&L for a binary option trade
        Handles expiration: if exit is at/near expiry, use settlement price
        direction: 'long' (buy) or 'short' (sell)
        """
        entry_price = entry_snapshot['mid_price']

        # Check if exit is at expiration
        exit_seconds_to_expiry = exit_snapshot.get('seconds_to_expiry', 999)

        if exit_seconds_to_expiry <= NEAR_EXPIRY_THRESHOLD:
            # At expiration - determine settlement
            settlement = self.feature_extractor.predict_settlement_price(
                exit_snapshot['orderbook'],
                exit_seconds_to_expiry
            )

            if settlement is None:
                # Cannot determine settlement, use mid price
                exit_price = exit_snapshot['mid_price']
            else:
                exit_price = settlement
        else:
            # Normal exit before expiration
            exit_price = exit_snapshot['mid_price']

        # Calculate P&L
        if direction == 'long':
            # Bought at entry_price, value is now exit_price
            # P&L per dollar = (exit_price - entry_price)
            # For trade size: multiply by number of contracts bought
            contracts = TRADE_SIZE / entry_price
            pnl = contracts * (exit_price - entry_price)
        else:  # short
            # Sold at entry_price, must buy back at exit_price
            contracts = TRADE_SIZE / (1 - entry_price)  # Selling the "NO" side
            pnl = contracts * (entry_price - exit_price)

        return pnl

    def create_training_sample(self, sequence_idx: int) -> Optional[Tuple]:
        """
        Create a training sample from historical data
        Respects option expiration - exits at expiry or holding period, whichever comes first
        """
        if len(self.training_data) < sequence_idx + SEQUENCE_LENGTH:
            return None

        # Get sequence
        sequence = []
        for i in range(sequence_idx, sequence_idx + SEQUENCE_LENGTH):
            sequence.append(self.training_data[i]['features'])

        sequence = np.array(sequence)

        # Entry point (last snapshot in sequence)
        entry_snapshot = self.training_data[sequence_idx + SEQUENCE_LENGTH - 1]
        entry_time = entry_snapshot['timestamp']
        seconds_to_expiry_at_entry = entry_snapshot.get('seconds_to_expiry', 0)

        # Determine exit time: MIN(holding_period, seconds_to_expiry - 30)
        # We exit 30 seconds before expiry to avoid illiquid conditions
        max_hold = min(MAX_HOLDING_PERIOD, max(seconds_to_expiry_at_entry - 30, 10))

        # Find exit snapshot
        exit_snapshot = None
        for i in range(sequence_idx + SEQUENCE_LENGTH, len(self.training_data)):
            time_held = (self.training_data[i]['timestamp'] - entry_time).total_seconds()

            if time_held >= max_hold:
                exit_snapshot = self.training_data[i]
                break

            # Also check if we've crossed expiration
            if self.training_data[i].get('seconds_to_expiry', 999) < 5:
                exit_snapshot = self.training_data[i]
                break

        if exit_snapshot is None:
            return None

        # Calculate P&L for both directions
        long_pnl = self.calculate_pnl(entry_snapshot, exit_snapshot, 'long')
        short_pnl = self.calculate_pnl(entry_snapshot, exit_snapshot, 'short')

        # Target is the better P&L normalized to [-1, 1]
        best_pnl = max(long_pnl, short_pnl)
        target = np.tanh(best_pnl / 50.0)  # Scale for binary options (higher variance)

        return sequence, target

    def prepare_training_data(self, max_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences and targets for training
        max_samples: Maximum number of training samples to create (for memory efficiency)
        """
        sequences = []
        targets = []

        # Need at least SEQUENCE_LENGTH + some buffer for exits
        min_samples = SEQUENCE_LENGTH + 100

        if len(self.training_data) < min_samples:
            print(f"Not enough data: {len(self.training_data)} < {min_samples}")
            return None, None

        # Use recent data only (more relevant)
        # If we have more than max_samples worth of data, use the most recent
        data_to_use = len(self.training_data) - SEQUENCE_LENGTH - 100

        # Sample every N snapshots if we have too much data
        if data_to_use > max_samples:
            step = data_to_use // max_samples
            print(f"📊 Using recent subset: sampling every {step} snapshots for efficiency")
            indices = range(0, data_to_use, step)
        else:
            indices = range(data_to_use)

        # Create training samples
        for i in indices:
            sample = self.create_training_sample(i)
            if sample:
                seq, target = sample
                sequences.append(seq)
                targets.append(target)

                # Safety limit
                if len(sequences) >= max_samples:
                    break

        if not sequences:
            return None, None

        print(f"✅ Created {len(sequences)} training samples")

        sequences = np.array(sequences, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32).reshape(-1, 1)

        # Normalize features
        if self.scaler_mean is None:
            self.scaler_mean = sequences.mean(axis=(0, 1))
            self.scaler_std = sequences.std(axis=(0, 1)) + 1e-8

        sequences = (sequences - self.scaler_mean) / self.scaler_std

        return sequences, targets

    def train_model(self, sequences: np.ndarray, targets: np.ndarray):
        """Train the LSTM model"""
        if not TORCH_AVAILABLE:
            print("PyTorch not available, cannot train model")
            return

        print(f"🧠 Training with reduced memory footprint...")
        print(f"   Hidden size: {HIDDEN_SIZE}, Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}")

        # Initialize model
        self.feature_size = sequences.shape[2]
        self.model = OrderBookLSTM(input_size=self.feature_size)

        # Create dataset and dataloader
        dataset = OrderBookDataset(sequences, targets)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        # Training loop
        print(f"\n{'='*60}")
        print(f"Training: {train_size} samples, Validation: {val_size} samples")
        print(f"{'='*60}\n")

        best_val_loss = float('inf')

        for epoch in range(EPOCHS):
            # Training
            self.model.train()
            train_loss = 0
            for batch_seq, batch_target in train_loader:
                optimizer.zero_grad()
                output = self.model(batch_seq)
                loss = criterion(output, batch_target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_seq, batch_target in val_loader:
                    output = self.model(batch_seq)
                    loss = criterion(output, batch_target)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            # Print progress every 5 epochs
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} - Train: {train_loss:.6f}, Val: {val_loss:.6f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model()

        print(f"\n✅ Training complete! Best val loss: {best_val_loss:.6f}")

        # Clean up memory
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, sequence: np.ndarray) -> float:
        """Make a prediction from a sequence"""
        if self.model is None or not TORCH_AVAILABLE:
            return 0.0

        self.model.eval()

        # Normalize
        if self.scaler_mean is not None:
            sequence = (sequence - self.scaler_mean) / self.scaler_std

        with torch.no_grad():
            seq_tensor = torch.FloatTensor(sequence).unsqueeze(0)
            prediction = self.model(seq_tensor)
            return prediction.item()

    def get_current_prediction(self) -> Optional[Dict]:
        """Get prediction from current buffer"""
        if len(self.snapshot_buffer) < SEQUENCE_LENGTH:
            return None

        # Extract features from buffer
        sequence = np.array([s['features'] for s in self.snapshot_buffer])

        # Get prediction
        prediction = self.predict(sequence)

        return {
            'timestamp': datetime.now(),
            'prediction': prediction,
            'mid_price': self.snapshot_buffer[-1]['mid_price'],
            'signal': 'BUY' if prediction > 0.1 else ('SELL' if prediction < -0.1 else 'HOLD')
        }

    def save_model(self):
        """Save model and scaler"""
        if self.model is None:
            return

        model_path = os.path.join(MODEL_DIR, 'orderbook_lstm.pth')
        scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_size': self.feature_size,
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS
        }, model_path)

        with open(scaler_path, 'wb') as f:
            pickle.dump({
                'mean': self.scaler_mean,
                'std': self.scaler_std
            }, f)

        print(f"✅ Model saved to {model_path}")

    def load_model(self):
        """Load model and scaler if they exist"""
        model_path = os.path.join(MODEL_DIR, 'orderbook_lstm.pth')
        scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')

        if not os.path.exists(model_path) or not TORCH_AVAILABLE:
            return

        try:
            checkpoint = torch.load(model_path)
            self.feature_size = checkpoint['feature_size']

            self.model = OrderBookLSTM(
                input_size=self.feature_size,
                hidden_size=checkpoint.get('hidden_size', HIDDEN_SIZE),
                num_layers=checkpoint.get('num_layers', NUM_LAYERS)
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
                self.scaler_mean = scaler['mean']
                self.scaler_std = scaler['std']

            print(f"✅ Model loaded from {model_path}")

        except Exception as e:
            print(f"⚠️  Could not load model: {e}")

    def save_history(self):
        """Save collected snapshots to disk"""
        if not self.training_data:
            return

        history_file = os.path.join(
            HISTORY_DIR,
            f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        )

        with open(history_file, 'wb') as f:
            pickle.dump(self.training_data, f)

        print(f"✅ History saved: {len(self.training_data)} snapshots")


def main():
    """Main function for testing and training"""
    print("\n" + "="*80)
    print("🧠 ORDER BOOK PRICE PREDICTION SYSTEM")
    print("="*80)
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Top N levels: {TOP_N_LEVELS}")
    print(f"Price range: ${PRICE_RANGE[0]:.2f} - ${PRICE_RANGE[1]:.2f}")
    print(f"Model: LSTM with {NUM_LAYERS} layers, {HIDDEN_SIZE} hidden units")
    print("="*80 + "\n")

    predictor = OrderBookPredictor()

    # Example: Collect data and train
    print("Collecting sample data...")

    call_file = os.path.join(DATA_DIR, '15M_CALL_nonoise.json')

    # Simulate collecting snapshots over time
    for i in range(100):
        if predictor.collect_snapshot(call_file):
            snapshot = predictor.snapshot_buffer[-1].copy()
            predictor.training_data.append(snapshot)

    print(f"Collected {len(predictor.training_data)} snapshots")

    # Prepare training data
    sequences, targets = predictor.prepare_training_data()
    
    if sequences is not None and len(sequences) > 0:
        print(f"Prepared {len(sequences)} training samples")
        print(f"Feature size: {sequences.shape[2]}")

        # Train model
        if TORCH_AVAILABLE:
            predictor.train_model(sequences, targets)
        else:
            print("Install PyTorch to train the model")
    else:
        print("Not enough data to train")


if __name__ == '__main__':
    main()
