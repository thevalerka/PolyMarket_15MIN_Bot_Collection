# collar_strategy.py - Dynamic Spread Calculation with Polynomial Regression
import json
import time
import math
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Try to import sklearn, fallback to simple calculations if not available
try:
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ scikit-learn not available - using fallback spread calculations")
    SKLEARN_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

from collar_config import state, MIN_QUANTITY_FILTER, MIN_SPREAD, MAX_SPREAD, SPREAD_BUFFER, SPREAD_FORMULA_PATH

class MarketDataProcessor:
    """Process and filter market data from JSON files."""
    
    def __init__(self, min_quantity: float = MIN_QUANTITY_FILTER):
        self.min_quantity = min_quantity
        
    def load_book_data(self, file_path: str) -> Optional[Dict]:
        """Load and filter order book data."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Filter bids and asks by minimum quantity
            filtered_bids = []
            filtered_asks = []
            
            if 'complete_book' in data:
                bids = data['complete_book'].get('bids', [])
                asks = data['complete_book'].get('asks', [])
                
                for bid in bids:
                    if float(bid['size']) >= self.min_quantity:
                        filtered_bids.append({
                            'price': float(bid['price']),
                            'size': float(bid['size'])
                        })
                
                for ask in asks:
                    if float(ask['size']) >= self.min_quantity:
                        filtered_asks.append({
                            'price': float(ask['price']),
                            'size': float(ask['size'])
                        })
            
            # Sort for best prices
            filtered_bids.sort(key=lambda x: x['price'], reverse=True)  # Highest first
            filtered_asks.sort(key=lambda x: x['price'])  # Lowest first
            
            result = {
                'timestamp': data.get('timestamp'),
                'delay_ms': data.get('delay_ms', 0),
                'filtered_bids': filtered_bids,
                'filtered_asks': filtered_asks,
                'total_bids_filtered': len(filtered_bids),
                'total_asks_filtered': len(filtered_asks),
                'best_filtered_bid': filtered_bids[0] if filtered_bids else None,
                'best_filtered_ask': filtered_asks[0] if filtered_asks else None,
                'original_best_bid': data.get('best_bid', {}),
                'original_best_ask': data.get('best_ask', {})
            }
            
            # Calculate market spread and mid
            if result['best_filtered_bid'] and result['best_filtered_ask']:
                result['market_spread'] = result['best_filtered_ask']['price'] - result['best_filtered_bid']['price']
                result['market_mid'] = (result['best_filtered_bid']['price'] + result['best_filtered_ask']['price']) / 2
            else:
                result['market_spread'] = None
                result['market_mid'] = None
            
            # Find base levels with sufficient liquidity (900+ shares)
            result['base_bid_level'] = self._find_base_level(filtered_bids, 900, 'bid')
            result['base_ask_level'] = self._find_base_level(filtered_asks, 900, 'ask')
            
            return result
            
        except Exception as e:
            print(f"❌ Error loading book data: {e}")
            return None
    
    def _find_base_level(self, orders: List[Dict], min_size: float, side: str) -> Optional[Dict]:
        """Find the price level with sufficient liquidity (900+ shares)."""
        for order in orders:
            if order['size'] >= min_size:
                return order
        
        # If no single order has 900+ shares, return the best available
        if orders:
            return orders[0]
        
        return None
    
    def load_btc_data(self, file_path: str) -> Optional[Dict]:
        """Load BTC price data."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            return {
                'price': float(data.get('price', 0)),
                'timestamp': data.get('timestamp'),
                'symbol': data.get('symbol', 'BTCUSDT')
            }
            
        except Exception as e:
            print(f"❌ Error loading BTC data: {e}")
            return None

class MarketMetricsCalculator:
    """Calculate market metrics for spread determination."""
    
    @staticmethod
    def calculate_price_volatility(price_history: List[Tuple[float, float]], window_minutes: int = 30) -> float:
        """Calculate price volatility over specified window."""
        if len(price_history) < 5:
            return 0.02  # Default low volatility
        
        current_time = time.time()
        cutoff_time = current_time - (window_minutes * 60)
        recent_prices = [price for timestamp, price in price_history if timestamp >= cutoff_time]
        
        if len(recent_prices) < 2:
            return 0.02
        
        # Calculate standard deviation of percentage changes
        pct_changes = []
        for i in range(1, len(recent_prices)):
            if recent_prices[i-1] > 0:
                pct_change = (recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
                pct_changes.append(pct_change)
        
        if not pct_changes:
            return 0.02
        
        volatility = np.std(pct_changes)
        return max(0.01, min(0.50, volatility))  # Clamp between 1% and 50%
    
    @staticmethod
    def calculate_price_change_short(price_history: List[Tuple[float, float]], window_minutes: int = 10) -> float:
        """Calculate short-term price change momentum."""
        if len(price_history) < 3:
            return 0.0
        
        current_time = time.time()
        cutoff_time = current_time - (window_minutes * 60)
        recent_prices = [price for timestamp, price in price_history if timestamp >= cutoff_time]
        
        if len(recent_prices) < 2:
            return 0.0
        
        # Calculate momentum as (latest - earliest) / earliest
        if recent_prices[0] > 0:
            momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            return max(-0.20, min(0.20, momentum))  # Clamp between -20% and +20%
        
        return 0.0
    
    @staticmethod
    def calculate_price_change_very_short(price_history: List[Tuple[float, float]], window_minutes: int = 1) -> float:
        """Calculate very short-term price change momentum (last minute)."""
        if len(price_history) < 2:
            return 0.0
        
        current_time = time.time()
        cutoff_time = current_time - (window_minutes * 60)
        recent_prices = [price for timestamp, price in price_history if timestamp >= cutoff_time]
        
        if len(recent_prices) < 2:
            return 0.0
        
        # Calculate very recent momentum (last minute)
        if recent_prices[0] > 0:
            momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            return max(-0.15, min(0.15, momentum))  # Clamp between -15% and +15%
        
        return 0.0
    
    @staticmethod
    def calculate_latest_btc_change(btc_price_history: List[Tuple[float, float]]) -> float:
        """Calculate the very latest BTC price change (last vs previous)."""
        if len(btc_price_history) < 2:
            return 0.0
        
        # Get the last two BTC price points
        latest_price = btc_price_history[-1][1]
        previous_price = btc_price_history[-2][1]
        
        if previous_price > 0:
            change = (latest_price - previous_price) / previous_price
            return max(-0.10, min(0.10, change))  # Clamp between -10% and +10%
        
        return 0.0
    
    @staticmethod
    def calculate_time_decay() -> float:
        """Calculate time decay factor (higher as expiry approaches)."""
        now = datetime.now()
        next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        time_remaining_minutes = (next_hour - now).total_seconds() / 60
        
        if time_remaining_minutes <= 0:
            return 2.0  # Maximum decay at expiry
        elif time_remaining_minutes > 40:
            return 0.1   # Minimal decay early in hour
        elif time_remaining_minutes > 20:
            return 0.3   # Low decay mid-hour
        elif time_remaining_minutes > 10:
            return 0.6   # Moderate decay
        elif time_remaining_minutes > 5:
            return 1.0   # High decay
        else:
            return 1.5   # Maximum decay in final minutes

class DynamicSpreadCalculator:
    """Calculate optimal spreads using polynomial regression."""
    
    def __init__(self):
        self.model = None
        self.feature_names = ['btc_volatility', 'token_volatility', 'short_momentum', 'very_short_momentum', 'time_decay', 'latest_btc_change']
        self.training_data = []
        self.last_calibration = 0
        
    def load_formula_from_file(self) -> bool:
        """Load spread formula from JSON file."""
        try:
            with open(SPREAD_FORMULA_PATH, 'r') as f:
                data = json.load(f)
            
            # Check if formula is from current hour
            current_hour = datetime.now().hour
            formula_hour = data.get('hour', -1)
            
            if formula_hour == current_hour:
                self.model = self._deserialize_model(data.get('model_params', {}))
                self.training_data = data.get('training_data', [])
                self.last_calibration = data.get('last_calibration', 0)
                print(f"✅ Loaded spread formula from file (hour {current_hour})")
                return True
            else:
                print(f"🔄 Formula from different hour ({formula_hour} vs {current_hour}) - will recalibrate")
                return False
                
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Could not load spread formula: {e}")
            return False
    
    def save_formula_to_file(self):
        """Save current spread formula to JSON file."""
        try:
            data = {
                'hour': datetime.now().hour,
                'last_calibration': self.last_calibration,
                'model_params': self._serialize_model(),
                'training_data': self.training_data[-100:],  # Keep last 100 samples
                'performance_score': state.formula_performance_score,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(SPREAD_FORMULA_PATH, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"💾 Spread formula saved (score: {state.formula_performance_score:.3f})")
            
        except Exception as e:
            print(f"❌ Could not save spread formula: {e}")
    
    def _serialize_model(self) -> Dict:
        """Serialize model parameters for JSON storage."""
        if self.model is None:
            return {}
        
        try:
            # Extract polynomial features and linear regression coefficients
            poly_features = self.model.named_steps['poly']
            linear_reg = self.model.named_steps['regressor']
            
            return {
                'degree': poly_features.degree,
                'coefficients': linear_reg.coef_.tolist(),
                'intercept': float(linear_reg.intercept_),
                'n_features': len(self.feature_names)
            }
        except:
            return {}
    
    def _deserialize_model(self, params: Dict):
        """Recreate model from saved parameters."""
        try:
            if not params:
                return None
            
            # Recreate the polynomial pipeline
            model = Pipeline([
                ('poly', PolynomialFeatures(degree=params['degree'], include_bias=False)),
                ('regressor', LinearRegression())
            ])
            
            # Set the learned parameters
            model.named_steps['regressor'].coef_ = np.array(params['coefficients'])
            model.named_steps['regressor'].intercept_ = params['intercept']
            
            return model
        except:
            return None
    
    def add_training_sample(self, features: List[float], actual_spread: float):
        """Add a new training sample."""
        if len(features) == len(self.feature_names) and actual_spread > 0:
            self.training_data.append({
                'timestamp': time.time(),
                'features': features,
                'spread': actual_spread
            })
            
            # Keep only last 200 samples
            self.training_data = self.training_data[-200:]
    
    def calibrate_model(self) -> bool:
        """Calibrate the polynomial regression model."""
        if len(self.training_data) < 10:
            print("⚠️ Insufficient training data for calibration (need 10+)")
            return False
        
        if not SKLEARN_AVAILABLE:
            print("⚠️ Sklearn not available - using simple linear model")
            return self._calibrate_simple_model()
        
        try:
            # Prepare training data
            X = np.array([sample['features'] for sample in self.training_data])
            y = np.array([sample['spread'] for sample in self.training_data])
            
            # Create polynomial regression pipeline (degree 2 for non-linear relationships)
            self.model = Pipeline([
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('regressor', LinearRegression())
            ])
            
            # Fit the model
            self.model.fit(X, y)
            
            self.last_calibration = time.time()
            
            # Calculate training accuracy
            predictions = self.model.predict(X)
            mae = np.mean(np.abs(predictions - y))
            r2_score = self.model.score(X, y)
            
            print(f"🎯 Model calibrated: MAE={mae:.4f}, R²={r2_score:.3f}, samples={len(self.training_data)}")
            
            # Save the updated formula
            self.save_formula_to_file()
            
            return True
            
        except Exception as e:
            print(f"❌ Model calibration failed: {e}")
            return False
    
    def _calibrate_simple_model(self) -> bool:
        """Fallback simple linear model when sklearn is not available."""
        try:
            # Simple weighted average of recent spreads with some trend adjustment
            recent_data = self.training_data[-50:]  # Last 50 samples
            
            if len(recent_data) < 5:
                return False
            
            # Create a simple linear model using numpy
            features = np.array([sample['features'] for sample in recent_data])
            spreads = np.array([sample['spread'] for sample in recent_data])
            
            # Simple multiple linear regression: spread = w0*btc_vol + w1*token_vol + w2*momentum + ... + bias
            X = np.column_stack([features, np.ones(len(features))])  # Add bias term
            
            # Solve using least squares: weights = (X^T * X)^-1 * X^T * y
            try:
                weights = np.linalg.solve(X.T @ X, X.T @ spreads)
                self.model = {'weights': weights.tolist(), 'type': 'simple_linear'}
                self.last_calibration = time.time()
                
                # Calculate simple accuracy
                predictions = X @ weights
                mae = np.mean(np.abs(predictions - spreads))
                
                print(f"🎯 Simple model calibrated: MAE={mae:.4f}, samples={len(recent_data)}")
                self.save_formula_to_file()
                return True
                
            except np.linalg.LinAlgError:
                print("❌ Matrix inversion failed - using average")
                avg_spread = np.mean(spreads)
                self.model = {'average': float(avg_spread), 'type': 'average'}
                self.last_calibration = time.time()
                return True
                
        except Exception as e:
            print(f"❌ Simple model calibration failed: {e}")
            return False
    
    def predict_optimal_spread(self, btc_volatility: float, token_volatility: float, 
                             short_momentum: float, very_short_momentum: float, 
                             time_decay: float, latest_btc_change: float) -> float:
        """Predict optimal spread using the trained model."""
        if self.model is None:
            # Fallback to simple heuristic
            return self._fallback_spread_calculation(btc_volatility, token_volatility, short_momentum, very_short_momentum, time_decay, latest_btc_change)
        
        try:
            if isinstance(self.model, dict):
                # Handle simple model
                return self._predict_simple_model(btc_volatility, token_volatility, short_momentum, very_short_momentum, time_decay, latest_btc_change)
            else:
                # Handle sklearn model
                features = np.array([[btc_volatility, token_volatility, short_momentum, very_short_momentum, time_decay, latest_btc_change]])
                predicted_spread = self.model.predict(features)[0]
                
                # Clamp to reasonable bounds
                return max(MIN_SPREAD, min(MAX_SPREAD, predicted_spread))
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            # Fallback to simple heuristic
            return self._fallback_spread_calculation(btc_volatility, token_volatility, short_momentum, very_short_momentum, time_decay, latest_btc_change)
    
    def _predict_simple_model(self, btc_vol: float, token_vol: float, short_mom: float, very_short_mom: float, time_decay: float, latest_btc_change: float) -> float:
        """Predict using simple linear model."""
        try:
            if self.model.get('type') == 'average':
                return self.model['average']
            elif self.model.get('type') == 'simple_linear':
                features = np.array([btc_vol, token_vol, short_mom, very_short_mom, time_decay, latest_btc_change, 1.0])  # Include bias
                weights = np.array(self.model['weights'])
                predicted_spread = np.dot(features, weights)
                return max(MIN_SPREAD, min(MAX_SPREAD, predicted_spread))
            else:
                return self._fallback_spread_calculation(btc_vol, token_vol, short_mom, very_short_mom, time_decay, latest_btc_change)
        except:
            return self._fallback_spread_calculation(btc_vol, token_vol, short_mom, very_short_mom, time_decay, latest_btc_change)
    
    def _fallback_spread_calculation(self, btc_vol: float, token_vol: float, short_mom: float, very_short_mom: float, time_decay: float, latest_btc_change: float) -> float:
        """Fallback spread calculation when model is not available."""
        base_spread = (btc_vol + token_vol) * 0.5
        momentum_factor = abs(short_mom) + abs(very_short_mom)
        btc_impact = abs(latest_btc_change) * 2.0  # BTC changes have significant impact
        decay_spread = base_spread * (1 + time_decay + momentum_factor + btc_impact)
        return max(MIN_SPREAD, min(MAX_SPREAD, decay_spread))
    
    def detect_market_manipulation(self, predicted_spread: float, actual_spread: float, 
                                 market_conditions: Dict) -> bool:
        """Detect potential market manipulation."""
        if actual_spread <= 0 or predicted_spread <= 0:
            return False
        
        # Check for extreme deviations
        deviation_ratio = abs(predicted_spread - actual_spread) / predicted_spread
        
        # Manipulation indicators
        extreme_deviation = deviation_ratio > 2.0  # 200% deviation
        unusual_spread = actual_spread > 0.30  # Spread > 30 cents
        low_volume = market_conditions.get('total_volume', 1000) < 100  # Very low volume
        
        manipulation_score = sum([extreme_deviation, unusual_spread, low_volume])
        
        if manipulation_score >= 2:
            alert = {
                'timestamp': time.time(),
                'predicted_spread': predicted_spread,
                'actual_spread': actual_spread,
                'deviation_ratio': deviation_ratio,
                'volume': market_conditions.get('total_volume', 0),
                'reason': f"Manipulation score: {manipulation_score}/3"
            }
            state.manipulation_alerts.append(alert)
            
            # Keep only last 10 alerts
            state.manipulation_alerts = state.manipulation_alerts[-10:]
            
            print(f"🚨 MARKET MANIPULATION DETECTED!")
            print(f"   Predicted: ${predicted_spread:.3f}, Actual: ${actual_spread:.3f}")
            print(f"   Deviation: {deviation_ratio:.1%}, Volume: {market_conditions.get('total_volume', 0)}")
            
            return True
        
        return False

class MarketAnalyzer:
    """Advanced market analyzer with dynamic spread calculation."""
    
    def __init__(self, book_path: str, btc_path: str):
        self.book_path = book_path
        self.btc_path = btc_path
        self.data_processor = MarketDataProcessor()
        self.metrics_calculator = MarketMetricsCalculator()
        self.spread_calculator = DynamicSpreadCalculator()
        
        # Initialize spread formula
        self._initialize_spread_system()
    
    def _initialize_spread_system(self):
        """Initialize the dynamic spread calculation system."""
        print("🔧 Initializing dynamic spread system...")
        
        # Check if it's a new hour (binary option reset)
        if state.is_new_hour():
            print("🆕 New hour detected - binary option reset")
            # Clear historical data for fresh start
            state.btc_price_history = []
            state.token_price_history = []
            state.market_spread_history = []
            state.formula_accuracy_history = []
        
        # Try to load existing formula
        if not self.spread_calculator.load_formula_from_file():
            print("🎯 Starting with default spread calculation")
    
    def analyze_market(self) -> Optional[Dict]:
        """Perform comprehensive market analysis with dynamic spreads."""
        # Load market data
        book_data = self.data_processor.load_book_data(self.book_path)
        btc_data = self.data_processor.load_btc_data(self.btc_path)
        
        if not book_data or not btc_data:
            return None
        
        # Add price data to history
        market_mid = book_data.get('market_mid')
        market_spread = book_data.get('market_spread')
        state.add_price_data(btc_data['price'], market_mid, market_spread)
        
        # Calculate market metrics
        btc_volatility = self.metrics_calculator.calculate_price_volatility(state.btc_price_history, 30)
        token_volatility = self.metrics_calculator.calculate_price_volatility(state.token_price_history, 15)
        short_momentum = self.metrics_calculator.calculate_price_change_short(state.token_price_history, 10)
        very_short_momentum = self.metrics_calculator.calculate_price_change_very_short(state.token_price_history, 1)  # Last minute
        time_decay = self.metrics_calculator.calculate_time_decay()
        latest_btc_change = self.metrics_calculator.calculate_latest_btc_change(state.btc_price_history)
        
        # Predict optimal spread
        predicted_spread = self.spread_calculator.predict_optimal_spread(
            btc_volatility, token_volatility, short_momentum, very_short_momentum, time_decay, latest_btc_change
        )
        
        # Validate prediction against market reality
        if market_spread is not None and market_spread > 0:
            state.validate_formula_accuracy(predicted_spread, market_spread)
            
            # Add training sample for future calibration
            features = [btc_volatility, token_volatility, short_momentum, very_short_momentum, time_decay, latest_btc_change]
            self.spread_calculator.add_training_sample(features, market_spread)
            
            # Check for market manipulation
            market_conditions = {
                'total_volume': book_data.get('total_bids_filtered', 0) + book_data.get('total_asks_filtered', 0),
                'best_bid': book_data.get('best_filtered_bid', {}).get('price', 0),
                'best_ask': book_data.get('best_filtered_ask', {}).get('price', 0)
            }
            self.spread_calculator.detect_market_manipulation(predicted_spread, market_spread, market_conditions)
        
        # Recalibrate model if needed
        if state.should_recalibrate_formula():
            print("🔄 Recalibrating spread formula...")
            self.spread_calculator.calibrate_model()
        
        # Calculate target prices with dynamic spread
        target_spread = predicted_spread + SPREAD_BUFFER
        target_bid = None
        target_ask = None
        
        # Check if market manipulation is detected
        manipulation_detected = len(state.manipulation_alerts) > 0 and time.time() - state.manipulation_alerts[-1]['timestamp'] < 300
        
        if manipulation_detected:
            # During manipulation, use base levels with sufficient liquidity
            print("🔧 Using base levels due to market manipulation")
            
            base_bid = book_data.get('base_bid_level')
            if base_bid:
                target_bid = round(base_bid['price'], 3)
                target_ask = round(base_bid['price'] + predicted_spread + 0.01, 3)
                
                print(f"   📊 Base bid level: ${base_bid['price']:.3f} ({base_bid['size']:.0f} shares)")
                print(f"   🎯 Manipulation targets: ${target_bid:.3f} x ${target_ask:.3f}")
        else:
            # Normal conditions: use market mid with spread
            if market_mid is not None:
                half_spread = target_spread / 2
                target_bid = max(0.01, round(market_mid - half_spread, 3))
                target_ask = min(0.99, round(market_mid + half_spread, 3))
        
        # Ensure valid price range
        if target_bid: target_bid = max(0.01, min(0.98, target_bid))
        if target_ask: target_ask = max(0.02, min(0.99, target_ask))
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'book_data': book_data,
            'btc_data': btc_data,
            'market_metrics': {
                'btc_volatility': btc_volatility,
                'token_volatility': token_volatility,
                'short_momentum': short_momentum,
                'very_short_momentum': very_short_momentum,
                'time_decay': time_decay,
                'latest_btc_change': latest_btc_change
            },
            'spread_analysis': {
                'predicted_spread': predicted_spread,
                'target_spread': target_spread,
                'market_spread': market_spread,
                'formula_performance': state.formula_performance_score,
                'manipulation_detected': len(state.manipulation_alerts) > 0 and 
                                       time.time() - state.manipulation_alerts[-1]['timestamp'] < 300  # Last 5 min
            },
            'target_prices': {
                'bid': target_bid,
                'ask': target_ask
            }
        }
        
        # Store in global state
        state.current_analysis = analysis
        state.last_update_time = time.time()
        
        return analysis

def display_analysis(analysis: Dict):
    """Display comprehensive market analysis results."""
    if not analysis:
        print("❌ No analysis available")
        return
    
    book = analysis.get('book_data', {})
    btc = analysis.get('btc_data', {})
    metrics = analysis.get('market_metrics', {})
    spread_info = analysis.get('spread_analysis', {})
    targets = analysis.get('target_prices', {})
    
    if not book or not btc:
        print("❌ Incomplete analysis data - skipping display")
        return
    
    print("=" * 90)
    print("📊 DYNAMIC SPREAD MARKET ANALYSIS")
    print("=" * 90)
    
    # Market Data
    print(f"💰 BTC Price: ${btc.get('price', 0):,.2f}")
    print(f"⏱️ Book Delay: {book.get('delay_ms', 9999)}ms")
    
    # Filtered Order Book - with safe None handling
    best_bid_data = book.get('best_filtered_bid')
    best_ask_data = book.get('best_filtered_ask')
    
    if best_bid_data and best_ask_data:
        print(f"📊 Market: ${best_bid_data['price']:.3f} x ${best_ask_data['price']:.3f}")
        market_mid = book.get('market_mid')
        market_spread = book.get('market_spread')
        
        if market_mid and market_mid > 0:
            print(f"📊 Market Mid: ${market_mid:.3f}")
        if market_spread:
            print(f"📊 Market Spread: ${market_spread:.3f}")
        print(f"📊 Valid Orders: {book.get('total_bids_filtered', 0)} bids, {book.get('total_asks_filtered', 0)} asks")
        
        # Show base levels for manipulation handling
        base_bid = book.get('base_bid_level')
        if base_bid:
            print(f"🔧 Base Bid Level (900+ shares): ${base_bid['price']:.3f} ({base_bid['size']:.0f} shares)")
        
        base_ask = book.get('base_ask_level')
        if base_ask:
            print(f"🔧 Base Ask Level (900+ shares): ${base_ask['price']:.3f} ({base_ask['size']:.0f} shares)")
    else:
        print("⚠️ Empty order book detected!")
        if not best_bid_data:
            print("   📊 No bids available")
        if not best_ask_data:
            print("   📊 No asks available")
    
    print()
    
    # Market Metrics
    if metrics:
        print("📈 MARKET METRICS")
        print(f"🔸 BTC Volatility: {metrics.get('btc_volatility', 0):.3f} ({metrics.get('btc_volatility', 0)*100:.1f}%)")
        print(f"🔸 Token Volatility: {metrics.get('token_volatility', 0):.3f} ({metrics.get('token_volatility', 0)*100:.1f}%)")
        print(f"🔸 Short Momentum (10min): {metrics.get('short_momentum', 0):+.3f} ({metrics.get('short_momentum', 0)*100:+.1f}%)")
        print(f"🔸 Very Short Momentum (1min): {metrics.get('very_short_momentum', 0):+.3f} ({metrics.get('very_short_momentum', 0)*100:+.1f}%)")
        print(f"🔸 Latest BTC Change: {metrics.get('latest_btc_change', 0):+.3f} ({metrics.get('latest_btc_change', 0)*100:+.1f}%)")
        print(f"🔸 Time Decay: {metrics.get('time_decay', 0):.3f}")
        print()
    
    # Spread Analysis
    if spread_info:
        print("🎯 DYNAMIC SPREAD ANALYSIS")
        print(f"🔸 Predicted Spread: ${spread_info.get('predicted_spread', 0):.3f}")
        print(f"🔸 Target Spread (+buffer): ${spread_info.get('target_spread', 0):.3f}")
        
        market_spread = spread_info.get('market_spread')
        predicted_spread = spread_info.get('predicted_spread', 0)
        if market_spread and predicted_spread > 0:
            accuracy = 1.0 - abs(predicted_spread - market_spread) / predicted_spread
            print(f"🔸 Market Spread: ${market_spread:.3f} (accuracy: {accuracy:.1%})")
        print(f"🔸 Formula Performance: {spread_info.get('formula_performance', 0):.1%}")
        
        if spread_info.get('manipulation_detected'):
            print("🚨 MARKET MANIPULATION DETECTED!")
        
        print()
    
    # Target Prices
    target_bid = targets.get('bid')
    target_ask = targets.get('ask')
    
    if target_bid and target_ask:
        print("🎯 DYNAMIC TARGET PRICES")
        print(f"🟢 Buy Target: ${target_bid:.3f}")
        print(f"🔴 Sell Target: ${target_ask:.3f}")
        
        # Show positioning vs market
        if best_bid_data and best_ask_data:
            bid_diff = target_bid - best_bid_data['price']
            ask_diff = best_ask_data['price'] - target_ask
            print(f"🔸 Bid Position: {bid_diff:+.3f} from market")
            print(f"🔸 Ask Position: {ask_diff:+.3f} from market")
    else:
        print("⚠️ No valid target prices available")
        if not target_bid:
            print("   📊 No target bid calculated")
        if not target_ask:
            print("   📊 No target ask calculated")
    
    # Historical Performance
    if hasattr(state, 'formula_accuracy_history') and state.formula_accuracy_history:
        recent_accuracy = [acc for _, _, _, acc in state.formula_accuracy_history[-10:]]
        if recent_accuracy:
            avg_accuracy = sum(recent_accuracy) / len(recent_accuracy)
            print(f"📊 Recent 10 predictions: {avg_accuracy:.1%} average accuracy")
    
    print("=" * 90)