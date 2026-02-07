#!/usr/bin/env python3
"""
FIXED ML Arbitrage Detector - Resolves Training Loop Issues

Key fixes:
1. Intelligent training frequency (only when sufficient signal exists)
2. Price change validation (skip if no meaningful changes)
3. Feature variance checking (prevent training on static data)
4. Training data quality monitoring
5. Proper model performance validation
"""

import numpy as np
import pandas as pd
import requests
import json
import time
import os
from datetime import datetime, timezone, timedelta
from collections import deque
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import threading
import warnings
warnings.filterwarnings('ignore')

class MLArbitrageDetector:
    def __init__(self,
                 btc_file='/home/ubuntu/013_2025_polymarket/btc_price.json',
                 call_file='/home/ubuntu/013_2025_polymarket/CALL.json',
                 put_file='/home/ubuntu/013_2025_polymarket/PUT.json',
                 model_dir='/home/ubuntu/013_2025_polymarket/ml_models'):
        """
        Fixed ML Arbitrage Detector with intelligent training
        """

        # File paths for real-time data
        self.btc_file = btc_file
        self.call_file = call_file
        self.put_file = put_file

        self.call_token_id = None
        self.put_token_id = None

        # Model persistence
        self.model_dir = model_dir
        self.model_files = {
            'call_model': os.path.join(model_dir, 'call_model.joblib'),
            'put_model': os.path.join(model_dir, 'put_model.joblib'),
            'anomaly_detector': os.path.join(model_dir, 'anomaly_detector.joblib'),
            'scaler': os.path.join(model_dir, 'scaler.joblib'),
            'training_data': os.path.join(model_dir, 'training_data.json'),
            'model_metadata': os.path.join(model_dir, 'model_metadata.json'),
            'recent_predictions': os.path.join(model_dir, 'recent_predictions.json')
        }

        os.makedirs(model_dir, exist_ok=True)

        # Historical kline data for context
        self.btc_klines = None
        self.last_kline_update = None
        self.kline_update_interval = 60

        # FIXED: Better training data management
        self.training_data = deque(maxlen=5000)  # Increased capacity
        self.recent_predictions = deque(maxlen=200)

        # FIXED: Training control parameters
        self.min_training_samples = 100          # Minimum samples before training
        self.training_frequency = 200            # Train every 200 samples (not 25!)
        self.min_price_change_threshold = 0.001  # Minimum change to consider (0.1 cents)
        self.min_feature_variance = 1e-6        # Minimum variance in features
        self.last_training_time = 0              # Track training frequency
        self.min_training_interval = 300        # Minimum 5 minutes between training

        # FIXED: Model quality tracking
        self.training_attempts = 0
        self.successful_trainings = 0
        self.training_quality_history = deque(maxlen=10)

        # ML Models
        self.call_model = RandomForestRegressor(
            n_estimators=30,      # Reduced for faster training
            max_depth=6,          # Reduced depth to prevent overfitting
            min_samples_split=20, # Require more samples for splits
            min_samples_leaf=10,   # Prevent overfitting on small datasets
            random_state=42
        )
        self.put_model = RandomForestRegressor(
            n_estimators=30,
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        self.anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
        self.scaler = StandardScaler()

        # Model state
        self.models_trained = False
        self.model_performance = {'call_mae': 0, 'put_mae': 0, 'call_r2': 0, 'put_r2': 0}
        self.training_samples = 0
        self.model_version = 1
        self.last_save_time = None

        # Current market context
        self.market_context = {
            'volatility_1m': 0, 'volatility_5m': 0, 'volatility_15m': 0,
            'momentum_1m': 0, 'momentum_5m': 0, 'momentum_15m': 0,
            'volume_ratio': 1, 'trend_strength': 0, 'price_range': 0,
            'market_regime': 'normal'
        }

        # FIXED: Price tracking for change calculation
        self.previous_option_state = None
        self.price_history = deque(maxlen=50)  # Track recent prices
        self.arbitrage_opportunities = []

        # Load existing models and data
        self._load_saved_state()

        # Background thread for kline updates
        self.update_thread = threading.Thread(target=self._background_kline_updater, daemon=True)
        self.running = True

        print("🧠 FIXED ML ARBITRAGE DETECTOR v3.0")
        print("🔧 Training Issues Resolved:")
        print(f"   • Intelligent training: Every {self.training_frequency} samples")
        print(f"   • Min training interval: {self.min_training_interval}s")
        print(f"   • Min price change: ${self.min_price_change_threshold:.3f}")
        print(f"   • Feature variance checking enabled")

        if self.models_trained:
            print(f"✅ Loaded existing models (v{self.model_version}) with {self.training_samples} samples")
            print(f"📊 Model Performance: CALL R²={self.model_performance.get('call_r2', 0):.3f}, "
                  f"PUT R²={self.model_performance.get('put_r2', 0):.3f}")
        else:
            print("🆕 Starting fresh - will train intelligently as quality data comes in")

        # Start background kline updates
        self.update_thread.start()

    def _background_kline_updater(self):
        """Background thread to keep kline data fresh"""
        while self.running:
            try:
                self._update_kline_context()
                time.sleep(self.kline_update_interval)
            except Exception as e:
                print(f"Kline update error: {e}")
                time.sleep(10)

    def _update_kline_context(self):
        """Update market context from fresh kline data"""
        try:
            klines_1m = self._fetch_klines('1m', 60)
            klines_5m = self._fetch_klines('5m', 60)
            klines_15m = self._fetch_klines('15m', 96)

            if klines_1m is not None and len(klines_1m) > 20:
                context = self._calculate_market_context(klines_1m, klines_5m, klines_15m)
                self.market_context.update(context)
                self.last_kline_update = datetime.now()

        except Exception as e:
            print(f"Context update error: {e}")

    def _fetch_klines(self, interval, limit):
        """Fetch kline data from Binance"""
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval={interval}&limit={limit}"
            response = requests.get(url, timeout=5)
            response.raise_for_status()

            data = response.json()
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_timestamp', 'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
            ])

            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            return df

        except Exception as e:
            return None

    def _calculate_market_context(self, klines_1m, klines_5m, klines_15m):
        """Calculate comprehensive market context from klines"""
        context = {}

        if klines_1m is not None and len(klines_1m) > 5:
            returns_1m = klines_1m['close'].pct_change().dropna()
            context['volatility_1m'] = returns_1m.tail(10).std() if len(returns_1m) >= 10 else 0
            context['momentum_1m'] = returns_1m.tail(1).iloc[0] if len(returns_1m) >= 1 else 0

            recent_high = klines_1m['high'].tail(15).max()
            recent_low = klines_1m['low'].tail(15).min()
            current_price = klines_1m['close'].iloc[-1]
            context['price_range'] = (recent_high - recent_low) / current_price

            context['volume_ratio'] = klines_1m['volume'].iloc[-1] / klines_1m['volume'].tail(20).mean()

        if klines_5m is not None and len(klines_5m) > 10:
            returns_5m = klines_5m['close'].pct_change().dropna()
            context['volatility_5m'] = returns_5m.tail(12).std() if len(returns_5m) >= 12 else 0
            context['momentum_5m'] = (klines_5m['close'].iloc[-1] - klines_5m['close'].iloc[-6]) / klines_5m['close'].iloc[-6] if len(klines_5m) >= 6 else 0

        if klines_15m is not None and len(klines_15m) > 20:
            returns_15m = klines_15m['close'].pct_change().dropna()
            context['volatility_15m'] = returns_15m.tail(24).std() if len(returns_15m) >= 24 else 0
            context['momentum_15m'] = (klines_15m['close'].iloc[-1] - klines_15m['close'].iloc[-8]) / klines_15m['close'].iloc[-8] if len(klines_15m) >= 8 else 0

            sma_short = klines_15m['close'].tail(8).mean()
            sma_long = klines_15m['close'].tail(24).mean()
            context['trend_strength'] = (sma_short - sma_long) / sma_long

        # Market regime classification
        vol_score = context.get('volatility_5m', 0)
        momentum_score = abs(context.get('momentum_5m', 0))

        if vol_score < 0.0003 and momentum_score < 0.0002:
            context['market_regime'] = 'quiet'
        elif vol_score > 0.002 or momentum_score > 0.001:
            context['market_regime'] = 'volatile'
        elif abs(context.get('trend_strength', 0)) > 0.003:
            context['market_regime'] = 'trending'
        else:
            context['market_regime'] = 'normal'

        return context

    def _load_real_time_data(self):
        """Load current real-time data from JSON files"""
        try:
            with open(self.btc_file, 'r') as f:
                btc_data = json.load(f)

            with open(self.call_file, 'r') as f:
                call_data = json.load(f)

            with open(self.put_file, 'r') as f:
                put_data = json.load(f)

            return btc_data, call_data, put_data

        except Exception as e:
            return None, None, None

    def _create_features(self, btc_price, strike_price, time_to_expiry):
        """Create feature vector combining market context + option specifics"""

        moneyness = btc_price / strike_price if strike_price else 1.0
        log_moneyness = np.log(moneyness) if moneyness > 0 else 0
        distance_from_strike = abs(moneyness - 1.0)
        time_decay = 1 - time_to_expiry if time_to_expiry <= 1 else 0

        features = {
            # Market context (from klines)
            'volatility_1m': self.market_context.get('volatility_1m', 0),
            'volatility_5m': self.market_context.get('volatility_5m', 0),
            'volatility_15m': self.market_context.get('volatility_15m', 0),
            'momentum_1m': self.market_context.get('momentum_1m', 0),
            'momentum_5m': self.market_context.get('momentum_5m', 0),
            'momentum_15m': self.market_context.get('momentum_15m', 0),
            'volume_ratio': self.market_context.get('volume_ratio', 1),
            'trend_strength': self.market_context.get('trend_strength', 0),
            'price_range': self.market_context.get('price_range', 0),

            # Option features
            'moneyness': moneyness,
            'log_moneyness': log_moneyness,
            'distance_from_strike': distance_from_strike,
            'time_to_expiry': time_to_expiry,
            'time_decay': time_decay,

            # Interaction features
            'vol_distance': self.market_context.get('volatility_5m', 0) * distance_from_strike,
            'momentum_distance': self.market_context.get('momentum_5m', 0) * distance_from_strike,
            'vol_time': self.market_context.get('volatility_15m', 0) * time_to_expiry,
        }

        # Market regime encoding
        regime = self.market_context.get('market_regime', 'normal')
        features['regime_quiet'] = 1 if regime == 'quiet' else 0
        features['regime_volatile'] = 1 if regime == 'volatile' else 0
        features['regime_trending'] = 1 if regime == 'trending' else 0

        return features

    def _validate_training_data(self, df):
        """FIXED: Validate training data quality before training"""

        # Check for sufficient variance in target variables
        call_changes = df['call_price_change'].dropna()
        put_changes = df['put_price_change'].dropna()

        if len(call_changes) < self.min_training_samples:
            return False, "Insufficient CALL data samples"

        if len(put_changes) < self.min_training_samples:
            return False, "Insufficient PUT data samples"

        # Check for meaningful price changes
        call_variance = call_changes.var()
        put_variance = put_changes.var()

        if call_variance < self.min_feature_variance:
            return False, f"CALL price changes have no variance ({call_variance:.6f})"

        if put_variance < self.min_feature_variance:
            return False, f"PUT price changes have no variance ({put_variance:.6f})"

        # Check for reasonable price change ranges
        call_range = call_changes.max() - call_changes.min()
        put_range = put_changes.max() - put_changes.min()

        if call_range < self.min_price_change_threshold:
            return False, f"CALL price range too small ({call_range:.4f})"

        if put_range < self.min_price_change_threshold:
            return False, f"PUT price range too small ({put_range:.4f})"

        # Check feature variance
        feature_cols = [col for col in df.columns
                       if col not in ['call_price_change', 'put_price_change', 'timestamp']]

        X = df[feature_cols].fillna(0)
        feature_variances = X.var()

        low_variance_features = feature_variances[feature_variances < self.min_feature_variance]
        if len(low_variance_features) > len(feature_cols) * 0.8:  # More than 80% have low variance
            return False, "Too many features have insufficient variance"

        return True, "Training data validation passed"

    def _should_train_model(self):
        """FIXED: Intelligent decision on when to train the model"""

        current_time = time.time()

        # Check minimum sample count
        if len(self.training_data) < self.min_training_samples:
            return False, "Not enough samples"

        # Check training frequency (don't train too often)
        if len(self.training_data) % self.training_frequency != 0:
            return False, "Training frequency not reached"

        # Check minimum time interval between trainings
        if current_time - self.last_training_time < self.min_training_interval:
            return False, "Too soon since last training"

        # Check if we have meaningful new data since last training
        if self.models_trained:
            recent_samples = list(self.training_data)[-50:]  # Last 50 samples
            if len(recent_samples) < 20:
                return False, "Not enough recent samples"

        return True, "Ready to train"

    def _update_models(self):
        """FIXED: Enhanced model training with quality validation"""

        should_train, reason = self._should_train_model()
        if not should_train:
            return False

        # Convert to DataFrame
        df = pd.DataFrame(list(self.training_data))

        # FIXED: Validate training data quality
        is_valid, validation_message = self._validate_training_data(df)
        if not is_valid:
            print(f"\n⚠️ Skipping training: {validation_message}")
            return False

        # Separate features and targets
        feature_cols = [col for col in df.columns
                       if col not in ['call_price_change', 'put_price_change', 'timestamp']]

        X = df[feature_cols].fillna(0)
        y_call = df['call_price_change'].fillna(0)
        y_put = df['put_price_change'].fillna(0)

        try:
            self.training_attempts += 1
            print(f"\n🔄 Training models with {len(X)} samples (Attempt #{self.training_attempts})...")
            print(f"📊 CALL variance: {y_call.var():.6f}, PUT variance: {y_put.var():.6f}")

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Split for validation
            test_size = min(0.3, 30/len(X))  # Use more data for validation
            X_train, X_test, y_call_train, y_call_test = train_test_split(
                X_scaled, y_call, test_size=test_size, random_state=42
            )
            _, _, y_put_train, y_put_test = train_test_split(
                X_scaled, y_put, test_size=test_size, random_state=42
            )

            # Train models
            self.call_model.fit(X_train, y_call_train)
            self.put_model.fit(X_train, y_put_train)
            self.anomaly_detector.fit(X_scaled)

            # Evaluate on test set
            if len(X_test) > 5:  # Need minimum samples for meaningful evaluation
                call_pred = self.call_model.predict(X_test)
                put_pred = self.put_model.predict(X_test)

                # Calculate performance metrics
                call_mae = mean_absolute_error(y_call_test, call_pred)
                put_mae = mean_absolute_error(y_put_test, put_pred)
                call_r2 = r2_score(y_call_test, call_pred)
                put_r2 = r2_score(y_put_test, put_pred)

                self.model_performance = {
                    'call_mae': call_mae,
                    'put_mae': put_mae,
                    'call_r2': call_r2,
                    'put_r2': put_r2
                }

                # FIXED: Quality threshold - only mark as trained if model shows some skill
                min_r2_threshold = 0.05  # Require at least 5% explained variance

                if call_r2 > min_r2_threshold or put_r2 > min_r2_threshold:
                    self.models_trained = True
                    self.successful_trainings += 1
                    self.training_samples = len(X)
                    self.model_version += 1
                    self.last_training_time = time.time()

                    # Track training quality
                    quality_score = max(call_r2, put_r2)
                    self.training_quality_history.append(quality_score)

                    print(f"✅ Models trained successfully!")
                    print(f"📊 Performance: CALL R²={call_r2:.3f}, PUT R²={put_r2:.3f}")
                    print(f"🎯 Quality score: {quality_score:.3f}")
                    print(f"📈 Success rate: {self.successful_trainings}/{self.training_attempts} ({self.successful_trainings/self.training_attempts:.1%})")

                    # Auto-save after successful training
                    self._save_model_state()
                    return True

                else:
                    print(f"⚠️ Training completed but model quality insufficient:")
                    print(f"   CALL R²={call_r2:.3f}, PUT R²={put_r2:.3f} (need > {min_r2_threshold:.3f})")
                    print("   Keeping previous model if available")
                    return False
            else:
                print("⚠️ Insufficient test samples for evaluation")
                return False

        except Exception as e:
            print(f"❌ Model training error: {e}")
            return False

    def _get_current_strike(self):
        """Get current strike price from Binance"""
        try:
            url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=1"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            return float(data[0][1])
        except:
            return 115000  # Fallback strike

    def _predict_option_response(self, features):
        """Predict expected option price changes"""
        if not self.models_trained:
            return None

        try:
            feature_cols = sorted([k for k in features.keys() if k not in ['call_price_change', 'put_price_change', 'timestamp']])
            X = np.array([features[col] for col in feature_cols]).reshape(1, -1)
            X_scaled = self.scaler.transform(X)

            call_pred = self.call_model.predict(X_scaled)[0]
            put_pred = self.put_model.predict(X_scaled)[0]

            # Get confidence (prediction variance across trees)
            call_tree_preds = [tree.predict(X_scaled)[0] for tree in self.call_model.estimators_[:10]]
            put_tree_preds = [tree.predict(X_scaled)[0] for tree in self.put_model.estimators_[:10]]

            call_std = np.std(call_tree_preds)
            put_std = np.std(put_tree_preds)

            call_confidence = 1 / (1 + call_std)
            put_confidence = 1 / (1 + put_std)

            # Anomaly score
            anomaly_score = self.anomaly_detector.decision_function(X_scaled)[0]

            return {
                'call_prediction': call_pred,
                'put_prediction': put_pred,
                'call_confidence': call_confidence,
                'put_confidence': put_confidence,
                'anomaly_score': anomaly_score
            }

        except Exception as e:
            return None

    def _detect_opportunities(self, actual_call_change, actual_put_change, predictions):

        # Add this at the start of your _detect_opportunities method:

        # Check current token holdings for selling bias
        # Fixed holdings check for selling bias
        call_balance = 0
        put_balance = 0

        try:
            if self.call_token_id:
                call_resp = self.client.get_balance_allowance(
                    params=BalanceAllowanceParams(
                        asset_type=AssetType.CONDITIONAL,
                        token_id=self.call_token_id
                    )
                )
                call_balance = int(call_resp.get('balance', 0)) / 10**6

            if self.put_token_id:
                put_resp = self.client.get_balance_allowance(
                    params=BalanceAllowanceParams(
                        asset_type=AssetType.CONDITIONAL,
                        token_id=self.put_token_id
                    )
                )
                put_balance = int(put_resp.get('balance', 0)) / 10**6
        except Exception as e:
            print(f"⚠️ Could not check token balances for selling bias: {e}")

        print(f"💎 Holdings: CALL={call_balance:.1f}, PUT={put_balance:.1f}")

        # Apply selling bias
        if call_balance > 2.0:
            call_threshold = call_threshold * 0.7  # 30% easier to SELL
            print(f"🎯 CALL SELL bias active")

        if put_balance > 2.0:
            put_threshold = put_threshold * 0.7  # 30% easier to SELL
            print(f"🎯 PUT SELL bias active")


        """Detect arbitrage opportunities based on prediction errors"""
        if not predictions:
            return []

        opportunities = []
        timestamp = datetime.now()

        # FIXED: More conservative thresholds based on model quality
        avg_quality = np.mean(self.training_quality_history) if self.training_quality_history else 0.1
        base_threshold = 0.020 * (1 + avg_quality)  # Adjust threshold based on model quality

        regime = self.market_context.get('market_regime', 'normal')
        if regime == 'quiet':
            regime_multiplier = 0.8
        elif regime == 'volatile':
            regime_multiplier = 1.3
        else:
            regime_multiplier = 1.0

        # CALL opportunity detection
        call_error = actual_call_change - predictions['call_prediction']
        call_threshold = base_threshold * regime_multiplier / predictions['call_confidence']

        if abs(call_error) > call_threshold:
            action = 'SELL' if actual_call_change > predictions['call_prediction'] else 'BUY'
            profit_potential = abs(call_error) - (call_threshold * 0.5)

            if profit_potential > 0.010:  # Minimum 1 cent profit
                opportunities.append({
                    'timestamp': timestamp,
                    'type': 'CALL',
                    'action': action,
                    'actual_change': actual_call_change,
                    'predicted_change': predictions['call_prediction'],
                    'error': call_error,
                    'threshold': call_threshold,
                    'profit_potential': profit_potential,
                    'confidence': predictions['call_confidence'],
                    'market_regime': regime,
                    'model_quality': avg_quality,
                    'reason': f"ML: {actual_call_change:+.3f} vs predicted {predictions['call_prediction']:+.3f}"
                })

        # PUT opportunity detection
        put_error = actual_put_change - predictions['put_prediction']
        put_threshold = base_threshold * regime_multiplier / predictions['put_confidence']

        if abs(put_error) > put_threshold:
            action = 'SELL' if actual_put_change > predictions['put_prediction'] else 'BUY'
            profit_potential = abs(put_error) - (put_threshold * 0.5)

            if profit_potential > 0.010:
                opportunities.append({
                    'timestamp': timestamp,
                    'type': 'PUT',
                    'action': action,
                    'actual_change': actual_put_change,
                    'predicted_change': predictions['put_prediction'],
                    'error': put_error,
                    'threshold': put_threshold,
                    'profit_potential': profit_potential,
                    'confidence': predictions['put_confidence'],
                    'market_regime': regime,
                    'model_quality': avg_quality,
                    'reason': f"ML: {actual_put_change:+.3f} vs predicted {predictions['put_prediction']:+.3f}"
                })

        return opportunities

    def analyze_market_data(self):
        """FIXED: Main analysis function with better price change tracking"""
        try:
            btc_data, call_data, put_data = self._load_real_time_data()

            if not all([btc_data, call_data, put_data]):
                return []

            current_btc = btc_data['price']
            call_bid = call_data['best_bid']['price']
            call_ask = call_data['best_ask']['price']
            put_bid = put_data['best_bid']['price']
            put_ask = put_data['best_ask']['price']

            # Calculate mid prices
            call_mid = (call_bid + call_ask) / 2
            put_mid = (put_bid + put_ask) / 2

            # Store current prices in history
            current_prices = {
                'timestamp': time.time(),
                'call_mid': call_mid,
                'put_mid': put_mid,
                'btc_price': current_btc
            }
            self.price_history.append(current_prices)

            # Get strike price
            current_strike = self._get_current_strike()

            # Calculate time to expiry
            now = datetime.now()
            time_to_expiry = max(0.01, (60 - now.minute) / 60.0)

            # Create features
            features = self._create_features(current_btc, current_strike, time_to_expiry)
            features['timestamp'] = now

            opportunities = []

            # FIXED: Better price change calculation
            if self.previous_option_state and len(self.price_history) >= 2:
                # Use recent price history for more stable change calculation
                prev_prices = self.price_history[-2]
                current_prices = self.price_history[-1]

                # Calculate price changes
                call_change = current_prices['call_mid'] - prev_prices['call_mid']
                put_change = current_prices['put_mid'] - prev_prices['put_mid']

                # FIXED: Only proceed if we have meaningful price changes
                total_change = abs(call_change) + abs(put_change)

                if total_change > self.min_price_change_threshold:
                    # Add changes to features for training
                    features['call_price_change'] = call_change
                    features['put_price_change'] = put_change

                    # Get ML predictions
                    predictions = self._predict_option_response(features)

                    # Detect opportunities
                    if predictions:
                        opportunities = self._detect_opportunities(call_change, put_change, predictions)

                        # Store prediction for accuracy tracking
                        self.recent_predictions.append({
                            'timestamp': now,
                            'call_actual': call_change,
                            'call_predicted': predictions['call_prediction'],
                            'put_actual': put_change,
                            'put_predicted': predictions['put_prediction']
                        })

                    # Add to training data
                    self.training_data.append(features)

                    # FIXED: Intelligent model updates
                    if self._should_train_model()[0]:
                        self._update_models()
                else:
                    # Prices haven't changed meaningfully
                    pass

            # Update previous state
            self.previous_option_state = {
                'call_bid': call_bid,
                'call_ask': call_ask,
                'put_bid': put_bid,
                'put_ask': put_ask,
                'btc_price': current_btc
            }

            # Store opportunities
            self.arbitrage_opportunities.extend(opportunities)

            # Print status
            self._print_status(current_btc, call_bid, call_ask, put_bid, put_ask, opportunities)

            return opportunities

        except Exception as e:
            print(f"\nAnalysis error: {e}")
            return []

    def _print_status(self, btc_price, call_bid, call_ask, put_bid, put_ask, opportunities):
        """FIXED: Enhanced status printing with training quality info"""
        regime = self.market_context.get('market_regime', 'unknown')
        vol_5m = self.market_context.get('volatility_5m', 0)
        momentum_5m = self.market_context.get('momentum_5m', 0)

        current_strike = self._get_current_strike()
        moneyness = btc_price / current_strike if current_strike else 1.0

        # Model quality indicator
        if self.training_quality_history:
            avg_quality = np.mean(self.training_quality_history)
            quality_indicator = f"Q:{avg_quality:.2f}"
        else:
            quality_indicator = "Q:--"

        # Training status
        if self.models_trained:
            ml_status = f"✅({self.training_samples})"
        else:
            ml_status = f"🔄({len(self.training_data)})"

        status = (f"\r{datetime.now().strftime('%H:%M:%S')} | "
                 f"BTC: ${btc_price:,.0f} ({moneyness:.3f}) | "
                 f"C: {call_bid:.3f}/{call_ask:.3f} | "
                 f"P: {put_bid:.3f}/{put_ask:.3f} | "
                 f"{regime} | Vol: {vol_5m:.4f} | Mom: {momentum_5m:+.4f} | "
                 f"ML: {ml_status} {quality_indicator} | "
                 f"Train: {self.successful_trainings}/{self.training_attempts} | "
                 f"Opps: {len(opportunities)} | Total: {len(self.arbitrage_opportunities)}")

        print(status, end='', flush=True)

        # Alert for opportunities
        for opp in opportunities:
            print(f"\n🚨 {opp['market_regime'].upper()} MARKET ARBITRAGE!")
            print(f"   💰 {opp['type']} {opp['action']} | Confidence: {opp['confidence']:.2f}")
            print(f"   📊 {opp['reason']}")
            print(f"   💵 Profit potential: ${opp['profit_potential']:.3f}")
            print(f"   🎯 Model quality: {opp.get('model_quality', 0):.2f}")

    def _save_model_state(self):
        """Save trained models and state to disk"""
        if not self.models_trained:
            return False

        try:
            print(f"\n💾 Saving ML models (v{self.model_version})...")

            joblib.dump(self.call_model, self.model_files['call_model'])
            joblib.dump(self.put_model, self.model_files['put_model'])
            joblib.dump(self.anomaly_detector, self.model_files['anomaly_detector'])
            joblib.dump(self.scaler, self.model_files['scaler'])

            # Save training data
            training_data_list = []
            for item in list(self.training_data):
                json_item = {}
                for key, value in item.items():
                    if isinstance(value, datetime):
                        json_item[key] = value.isoformat()
                    elif isinstance(value, (np.integer, np.floating)):
                        json_item[key] = float(value)
                    elif pd.isna(value):
                        json_item[key] = None
                    else:
                        json_item[key] = value
                training_data_list.append(json_item)

            with open(self.model_files['training_data'], 'w') as f:
                json.dump(training_data_list, f, indent=2)

            # Save recent predictions
            predictions_list = []
            for item in list(self.recent_predictions):
                json_item = {}
                for key, value in item.items():
                    if isinstance(value, datetime):
                        json_item[key] = value.isoformat()
                    elif isinstance(value, (np.integer, np.floating)):
                        json_item[key] = float(value)
                    elif pd.isna(value):
                        json_item[key] = None
                    else:
                        json_item[key] = value
                predictions_list.append(json_item)

            with open(self.model_files['recent_predictions'], 'w') as f:
                json.dump(predictions_list, f, indent=2)

            # Save metadata with training quality info
            metadata = {
                'model_version': self.model_version,
                'models_trained': self.models_trained,
                'training_samples': self.training_samples,
                'model_performance': self.model_performance,
                'training_attempts': self.training_attempts,
                'successful_trainings': self.successful_trainings,
                'training_quality_history': list(self.training_quality_history),
                'last_save_time': datetime.now().isoformat(),
                'total_training_points': len(self.training_data),
                'total_predictions': len(self.recent_predictions),
                'total_opportunities': len(self.arbitrage_opportunities)
            }

            with open(self.model_files['model_metadata'], 'w') as f:
                json.dump(metadata, f, indent=2)

            self.last_save_time = datetime.now()
            print(f"✅ Models saved successfully to {self.model_dir}")
            return True

        except Exception as e:
            print(f"❌ Error saving models: {e}")
            return False

    def _load_saved_state(self):
        """Load existing models and training data from disk"""
        try:
            if not os.path.exists(self.model_files['model_metadata']):
                print("🆕 No existing models found - starting fresh")
                return False

            print("📄 Loading existing ML models...")

            with open(self.model_files['model_metadata'], 'r') as f:
                metadata = json.load(f)

            self.model_version = metadata.get('model_version', 1)
            self.models_trained = metadata.get('models_trained', False)
            self.training_samples = metadata.get('training_samples', 0)
            self.model_performance = metadata.get('model_performance', {})
            self.training_attempts = metadata.get('training_attempts', 0)
            self.successful_trainings = metadata.get('successful_trainings', 0)

            # Load training quality history
            quality_history = metadata.get('training_quality_history', [])
            self.training_quality_history.extend(quality_history)

            if not self.models_trained:
                return False

            # Load sklearn models
            if all(os.path.exists(f) for f in [
                self.model_files['call_model'],
                self.model_files['put_model'],
                self.model_files['anomaly_detector'],
                self.model_files['scaler']
            ]):
                self.call_model = joblib.load(self.model_files['call_model'])
                self.put_model = joblib.load(self.model_files['put_model'])
                self.anomaly_detector = joblib.load(self.model_files['anomaly_detector'])
                self.scaler = joblib.load(self.model_files['scaler'])

                print(f"✅ Loaded ML models (v{self.model_version})")
            else:
                print("⚠️ Model files incomplete - will retrain")
                self.models_trained = False
                return False

            # Load training data
            if os.path.exists(self.model_files['training_data']):
                with open(self.model_files['training_data'], 'r') as f:
                    training_data_list = json.load(f)

                for item in training_data_list:
                    if 'timestamp' in item and isinstance(item['timestamp'], str):
                        try:
                            item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                        except:
                            item['timestamp'] = datetime.now()

                    self.training_data.append(item)

                print(f"✅ Loaded {len(self.training_data)} training samples")

            # Load recent predictions
            if os.path.exists(self.model_files['recent_predictions']):
                with open(self.model_files['recent_predictions'], 'r') as f:
                    predictions_list = json.load(f)

                for item in predictions_list:
                    if 'timestamp' in item and isinstance(item['timestamp'], str):
                        try:
                            item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                        except:
                            item['timestamp'] = datetime.now()

                    self.recent_predictions.append(item)

                print(f"✅ Loaded {len(self.recent_predictions)} recent predictions")

            return True

        except Exception as e:
            print(f"❌ Error loading saved state: {e}")
            print("🆕 Starting fresh...")
            self.models_trained = False
            return False

    def get_performance_report(self):
        """Enhanced performance report with training quality info"""
        if not self.recent_predictions:
            return "No predictions yet"

        df = pd.DataFrame(list(self.recent_predictions))

        call_mae = mean_absolute_error(df['call_actual'], df['call_predicted'])
        put_mae = mean_absolute_error(df['put_actual'], df['put_predicted'])

        # Training quality metrics
        avg_quality = np.mean(self.training_quality_history) if self.training_quality_history else 0
        training_success_rate = self.successful_trainings / max(self.training_attempts, 1)

        save_status = "✅ Auto-saved" if self.last_save_time else "❌ Not saved"
        save_age = ""
        if self.last_save_time:
            age_minutes = (datetime.now() - self.last_save_time).seconds // 60
            save_age = f"({age_minutes}m ago)" if age_minutes > 0 else "(just now)"

        report = f"""
🧠 FIXED ML ARBITRAGE DETECTOR PERFORMANCE REPORT
{'='*70}
🗃️  MODEL PERSISTENCE STATUS:
├─ Models Trained: {'✅' if self.models_trained else '❌'}
├─ Model Version: v{self.model_version}
├─ Save Status: {save_status} {save_age}
├─ Model Directory: {self.model_dir}
└─ Training Samples: {len(self.training_data)} / {self.training_samples}

🔄 TRAINING QUALITY METRICS:
├─ Training Attempts: {self.training_attempts}
├─ Successful Trainings: {self.successful_trainings}
├─ Training Success Rate: {training_success_rate:.1%}
├─ Average Quality Score: {avg_quality:.3f}
└─ Quality History: {len(self.training_quality_history)} records

📊 LEARNING PERFORMANCE:
├─ Recent Predictions: {len(self.recent_predictions)}
├─ CALL MAE: {call_mae:.4f} (lower is better)
├─ PUT MAE: {put_mae:.4f} (lower is better)
├─ CALL R²: {self.model_performance.get('call_r2', 0):.3f} (higher is better)
└─ PUT R²: {self.model_performance.get('put_r2', 0):.3f} (higher is better)

🌍 MARKET CONTEXT:
├─ Regime: {self.market_context.get('market_regime', 'unknown')}
├─ 1m Volatility: {self.market_context.get('volatility_1m', 0):.5f}
├─ 5m Volatility: {self.market_context.get('volatility_5m', 0):.5f}
├─ 5m Momentum: {self.market_context.get('momentum_5m', 0):+.5f}
├─ Trend Strength: {self.market_context.get('trend_strength', 0):+.5f}
└─ Last Kline Update: {self.last_kline_update}

💰 ARBITRAGE RESULTS:
└─ Total Opportunities: {len(self.arbitrage_opportunities)}

🔧 TRAINING CONFIGURATION:
├─ Min Training Samples: {self.min_training_samples}
├─ Training Frequency: Every {self.training_frequency} samples
├─ Min Training Interval: {self.min_training_interval}s
├─ Min Price Change Threshold: ${self.min_price_change_threshold:.3f}
└─ Min Feature Variance: {self.min_feature_variance:.2e}

📧 AVAILABLE COMMANDS:
├─ detector.save_models() - Manual save
├─ detector.backup_models('name') - Create backup
├─ detector.list_backups() - Show available backups
└─ detector.restore_backup('name') - Restore from backup
{'='*70}
        """

        return report

    def stop(self):
        """Stop background threads and save final state"""
        print("\n🛑 Shutting down Fixed ML Arbitrage Detector...")
        self.running = False

        if self.models_trained:
            print("💾 Saving final model state...")
            self._save_model_state()

        print("✅ Shutdown complete")

def main():
    """Example usage with fixed training"""
    detector = MLArbitrageDetector()

    print("🧠 Fixed ML Arbitrage Detector - Intelligent Training System")
    print("🔧 Training Issues Resolved:")
    print("   • No more constant retraining loops")
    print("   • Quality validation before training")
    print("   • Intelligent training frequency")
    print("   • Price change validation")
    print("=" * 80)

    try:
        iteration = 0
        while True:
            opportunities = detector.analyze_market_data()

            iteration += 1

            # Show performance report every 1000 iterations
            if iteration % 1000 == 0:
                print(detector.get_performance_report())

            time.sleep(0.5)  # Slightly slower for better stability

    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user")
        print(detector.get_performance_report())
        detector.stop()

if __name__ == "__main__":
    main()
