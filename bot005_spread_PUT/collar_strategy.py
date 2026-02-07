# collar_strategy.py - Dynamic Spread Calculation with Polynomial Regression
"""
Dynamic Spread Calculation System

QUICK START (No ML dependencies):
    python3 run_collar_bot.py  # Uses simple rule-based calculations

ENHANCED VERSION (With ML):
    pip3 install numpy scikit-learn
    python3 run_collar_bot.py  # Uses polynomial regression + fallback

The system gracefully degrades:
- Full ML: Polynomial regression with advanced features
- Basic: Simple volatility-based calculations
- Always works regardless of installed packages
"""

import json
import time
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Try to import numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
    print("✅ Numpy available - enhanced mathematical operations")
except ImportError:
    print("⚠️ Numpy not available - using basic Python math")
    print("💡 For better performance: pip3 install numpy")
    NUMPY_AVAILABLE = False
    # Create a minimal numpy substitute
    class MockNumpy:
        @staticmethod
        def mean(arr): return sum(arr) / len(arr) if arr else 0
        @staticmethod
        def std(arr): 
            if not arr or len(arr) < 2: return 0
            mean_val = sum(arr) / len(arr)
            variance = sum((x - mean_val) ** 2 for x in arr) / len(arr)
            return variance ** 0.5
        @staticmethod
        def array(arr): return list(arr)
        @staticmethod
        def column_stack(arrays): return list(zip(*arrays))
        class linalg:
            @staticmethod
            def solve(A, b): 
                # Very basic 1D solve fallback
                if len(A) == 1 and len(A[0]) == 1: return [b[0] / A[0][0]]
                return [1.0]  # Fallback
            class LinAlgError(Exception): pass
    np = MockNumpy()

# Try to import sklearn, fallback to simple calculations if not available
try:
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
    print("✅ Scikit-learn available - using advanced polynomial regression")
except ImportError:
    print("⚠️ Scikit-learn not available - using simple spread calculations")
    print("💡 For ML features: pip3 install scikit-learn")
    SKLEARN_AVAILABLE = False

# Suppress warnings if sklearn is available
if SKLEARN_AVAILABLE:
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
        
        volatility = np.std(pct_changes) if NUMPY_AVAILABLE else MarketMetricsCalculator._calculate_std(pct_changes)
        return max(0.01, min(0.50, volatility))  # Clamp between 1% and 50%
    
    @staticmethod
    def _calculate_std(values):
        """Calculate standard deviation without numpy."""
        if not values or len(values) < 2:
            return 0.02
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
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
        self.last_comprehensive_export = 0  # Track last comprehensive export
        
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
    
    def export_comprehensive_analysis(self, current_analysis: Dict, force: bool = False) -> bool:
        """Export comprehensive real-time analysis every 5 minutes."""
        current_time = time.time()
        
        # Check if 5 minutes have passed or force export
        if not force and current_time - self.last_comprehensive_export < 300:  # 5 minutes
            return False
        
        try:
            # Build comprehensive analysis
            comprehensive_data = {
                # Basic metadata
                'export_timestamp': datetime.now().isoformat(),
                'export_time_unix': current_time,
                'hour': datetime.now().hour,
                'minute': datetime.now().minute,
                'uptime_minutes': (current_time - state.last_update_time) / 60 if state.last_update_time > 0 else 0,
                
                # Model performance metrics
                'model_performance': {
                    'formula_accuracy': state.formula_performance_score,
                    'recent_predictions': self._get_recent_prediction_analysis(),
                    'training_samples': len(self.training_data),
                    'last_calibration_ago_minutes': (current_time - self.last_calibration) / 60,
                    'model_type': 'polynomial_regression' if SKLEARN_AVAILABLE else 'simple_fallback',
                    'sklearn_available': SKLEARN_AVAILABLE,
                    'numpy_available': NUMPY_AVAILABLE
                },
                
                # Current market conditions
                'current_market': self._analyze_current_market(current_analysis),
                
                # Feature analysis
                'feature_analysis': self._analyze_features(),
                
                # Market regime detection
                'market_regime': self._detect_detailed_market_regime(),
                
                # Prediction insights
                'prediction_insights': self._get_prediction_insights(current_analysis),
                
                # Performance trends
                'performance_trends': self._calculate_performance_trends(),
                
                # Risk indicators
                'risk_indicators': self._calculate_risk_indicators(current_analysis),
                
                # Trading statistics
                'trading_stats': self._get_trading_statistics(),
                
                # Model health
                'model_health': self._assess_model_health(),
                
                # Market manipulation detection
                'manipulation_analysis': self._analyze_manipulation_signals(),
                
                # Recommendations
                'recommendations': self._generate_recommendations(current_analysis)
            }
            
            # Export to file
            export_path = SPREAD_FORMULA_PATH.replace('spread_formula.json', 'comprehensive_analysis.json')
            with open(export_path, 'w') as f:
                json.dump(comprehensive_data, f, indent=2)
            
            self.last_comprehensive_export = current_time
            
            print(f"📊 Comprehensive analysis exported: {export_path}")
            print(f"   Model Accuracy: {state.formula_performance_score:.1%}")
            print(f"   Market Regime: {comprehensive_data['market_regime']['current_regime']}")
            print(f"   Risk Level: {comprehensive_data['risk_indicators']['overall_risk_level']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to export comprehensive analysis: {e}")
            return False
    
    def _get_recent_prediction_analysis(self) -> Dict:
        """Analyze recent prediction performance."""
        if not hasattr(state, 'formula_accuracy_history') or not state.formula_accuracy_history:
            return {'status': 'no_data'}
        
        recent_data = state.formula_accuracy_history[-20:]  # Last 20 predictions
        
        if len(recent_data) < 5:
            return {'status': 'insufficient_data', 'sample_count': len(recent_data)}
        
        accuracies = [acc for _, _, _, acc in recent_data]
        predictions = [pred for _, pred, _, _ in recent_data]
        actuals = [actual for _, _, actual, _ in recent_data]
        
        if NUMPY_AVAILABLE:
            avg_accuracy = np.mean(accuracies)
            accuracy_trend = np.mean(accuracies[-5:]) - np.mean(accuracies[:5]) if len(accuracies) >= 10 else 0
            prediction_volatility = np.std(predictions)
            actual_volatility = np.std(actuals)
        else:
            avg_accuracy = sum(accuracies) / len(accuracies)
            accuracy_trend = 0
            prediction_volatility = 0.05
            actual_volatility = 0.05
        
        return {
            'status': 'active',
            'sample_count': len(recent_data),
            'average_accuracy': avg_accuracy,
            'accuracy_trend': accuracy_trend,
            'best_accuracy': max(accuracies),
            'worst_accuracy': min(accuracies),
            'prediction_volatility': prediction_volatility,
            'actual_volatility': actual_volatility,
            'latest_predictions': [
                {
                    'timestamp': datetime.fromtimestamp(ts).isoformat(),
                    'predicted': pred,
                    'actual': actual,
                    'accuracy': acc
                }
                for ts, pred, actual, acc in recent_data[-5:]
            ]
        }
    
    def _analyze_current_market(self, analysis: Dict) -> Dict:
        """Analyze current market conditions."""
        if not analysis:
            return {'status': 'no_data'}
        
        book_data = analysis.get('book_data', {})
        btc_data = analysis.get('btc_data', {})
        metrics = analysis.get('market_metrics', {})
        
        return {
            'status': 'active',
            'btc_price': btc_data.get('price', 0),
            'market_mid': book_data.get('market_mid', 0),
            'market_spread': book_data.get('market_spread', 0),
            'book_delay_ms': book_data.get('delay_ms', 0),
            'order_count': {
                'bids': book_data.get('total_bids_filtered', 0),
                'asks': book_data.get('total_asks_filtered', 0)
            },
            'volatility': {
                'btc': metrics.get('btc_volatility', 0),
                'token': metrics.get('token_volatility', 0)
            },
            'momentum': {
                'short_term': metrics.get('short_momentum', 0),
                'very_short_term': metrics.get('very_short_momentum', 0),
                'btc_latest': metrics.get('latest_btc_change', 0)
            },
            'time_decay': metrics.get('time_decay', 0),
            'data_quality': 'good' if book_data.get('delay_ms', 0) < 1000 else 'degraded'
        }
    
    def _analyze_features(self) -> Dict:
        """Analyze feature importance and ranges."""
        if not self.training_data:
            return {'status': 'no_training_data'}
        
        # Calculate feature statistics from recent training data
        recent_data = self.training_data[-50:]  # Last 50 samples
        
        if len(recent_data) < 10:
            return {'status': 'insufficient_data'}
        
        feature_stats = {}
        
        for i, feature_name in enumerate(self.feature_names):
            values = [sample['features'][i] for sample in recent_data if len(sample['features']) > i]
            
            if values:
                if NUMPY_AVAILABLE:
                    stats = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'range': np.max(values) - np.min(values),
                        'latest': values[-1]
                    }
                else:
                    stats = {
                        'mean': sum(values) / len(values),
                        'std': (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5,
                        'min': min(values),
                        'max': max(values),
                        'range': max(values) - min(values),
                        'latest': values[-1]
                    }
                
                # Determine feature activity level
                if stats['range'] > stats['std'] * 2:
                    activity = 'high'
                elif stats['range'] > stats['std']:
                    activity = 'moderate'
                else:
                    activity = 'low'
                
                stats['activity_level'] = activity
                feature_stats[feature_name] = stats
        
        return {
            'status': 'analyzed',
            'sample_count': len(recent_data),
            'features': feature_stats,
            'most_active_feature': max(feature_stats.keys(), key=lambda k: feature_stats[k]['range']) if feature_stats else None,
            'least_active_feature': min(feature_stats.keys(), key=lambda k: feature_stats[k]['range']) if feature_stats else None
        }
    
    def _detect_detailed_market_regime(self) -> Dict:
        """Detect current market regime with detailed analysis."""
        if not state.btc_price_history or not state.token_price_history:
            return {'current_regime': 'unknown', 'confidence': 0}
        
        # Get recent data
        recent_btc = state.btc_price_history[-10:] if len(state.btc_price_history) >= 10 else state.btc_price_history
        recent_token = state.token_price_history[-10:] if len(state.token_price_history) >= 10 else state.token_price_history
        
        if len(recent_btc) < 3 or len(recent_token) < 3:
            return {'current_regime': 'insufficient_data', 'confidence': 0}
        
        # Calculate regime indicators
        btc_prices = [p for _, p in recent_btc]
        token_prices = [p for _, p in recent_token]
        
        if NUMPY_AVAILABLE:
            btc_volatility = np.std(btc_prices) / np.mean(btc_prices) if np.mean(btc_prices) > 0 else 0
            token_volatility = np.std(token_prices) / np.mean(token_prices) if np.mean(token_prices) > 0 else 0
        else:
            btc_mean = sum(btc_prices) / len(btc_prices)
            token_mean = sum(token_prices) / len(token_prices)
            btc_volatility = (sum((p - btc_mean)**2 for p in btc_prices) / len(btc_prices))**0.5 / btc_mean if btc_mean > 0 else 0
            token_volatility = (sum((p - token_mean)**2 for p in token_prices) / len(token_prices))**0.5 / token_mean if token_mean > 0 else 0
        
        # Time until expiry
        now = datetime.now()
        next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        minutes_to_expiry = (next_hour - now).total_seconds() / 60
        
        # Determine regime
        if btc_volatility < 0.02 and token_volatility < 0.05 and minutes_to_expiry > 30:
            regime = 'ultra_stable'
            confidence = 0.9
        elif btc_volatility < 0.05 and token_volatility < 0.10 and minutes_to_expiry > 15:
            regime = 'stable'
            confidence = 0.8
        elif btc_volatility < 0.10 and token_volatility < 0.20 and minutes_to_expiry > 5:
            regime = 'moderate'
            confidence = 0.7
        elif minutes_to_expiry < 5:
            regime = 'final_minutes'
            confidence = 0.9
        elif btc_volatility > 0.15 or token_volatility > 0.25:
            regime = 'high_volatility'
            confidence = 0.8
        else:
            regime = 'transitional'
            confidence = 0.6
        
        return {
            'current_regime': regime,
            'confidence': confidence,
            'btc_volatility': btc_volatility,
            'token_volatility': token_volatility,
            'minutes_to_expiry': minutes_to_expiry,
            'regime_factors': {
                'time_pressure': 1.0 - (minutes_to_expiry / 60),
                'market_stress': btc_volatility + token_volatility,
                'uncertainty': abs(token_prices[-1] - 0.5) if token_prices else 0
            }
        }
    
    def _get_prediction_insights(self, analysis: Dict) -> Dict:
        """Get insights about current predictions."""
        if not analysis:
            return {'status': 'no_current_analysis'}
        
        spread_analysis = analysis.get('spread_analysis', {})
        target_prices = analysis.get('target_prices', {})
        
        insights = {
            'status': 'active',
            'predicted_spread': spread_analysis.get('predicted_spread', 0),
            'target_spread': spread_analysis.get('target_spread', 0),
            'buffer_added': SPREAD_BUFFER,
            'target_prices': target_prices,
            'manipulation_detected': spread_analysis.get('manipulation_detected', False)
        }
        
        # Calculate spread reasonableness
        predicted = insights['predicted_spread']
        if 0.02 <= predicted <= 0.08:
            insights['spread_assessment'] = 'reasonable'
        elif 0.08 < predicted <= 0.15:
            insights['spread_assessment'] = 'elevated'
        elif predicted > 0.15:
            insights['spread_assessment'] = 'extreme'
        else:
            insights['spread_assessment'] = 'too_low'
        
        return insights
    
    def _calculate_performance_trends(self) -> Dict:
        """Calculate performance trends over time."""
        if not hasattr(state, 'formula_accuracy_history') or len(state.formula_accuracy_history) < 10:
            return {'status': 'insufficient_data'}
        
        history = state.formula_accuracy_history
        
        # Split into periods
        total_samples = len(history)
        period_size = max(5, total_samples // 4)
        
        periods = []
        for i in range(0, total_samples, period_size):
            period = history[i:i+period_size]
            if len(period) >= 3:
                accuracies = [acc for _, _, _, acc in period]
                if NUMPY_AVAILABLE:
                    avg_accuracy = np.mean(accuracies)
                else:
                    avg_accuracy = sum(accuracies) / len(accuracies)
                
                periods.append({
                    'start_time': datetime.fromtimestamp(period[0][0]).isoformat(),
                    'end_time': datetime.fromtimestamp(period[-1][0]).isoformat(),
                    'sample_count': len(period),
                    'average_accuracy': avg_accuracy
                })
        
        # Calculate trend
        if len(periods) >= 2:
            trend = periods[-1]['average_accuracy'] - periods[0]['average_accuracy']
            if trend > 0.05:
                trend_direction = 'improving'
            elif trend < -0.05:
                trend_direction = 'declining'
            else:
                trend_direction = 'stable'
        else:
            trend = 0
            trend_direction = 'unknown'
        
        return {
            'status': 'calculated',
            'trend_direction': trend_direction,
            'trend_magnitude': trend,
            'periods': periods,
            'total_samples': total_samples
        }
    
    def _calculate_risk_indicators(self, analysis: Dict) -> Dict:
        """Calculate various risk indicators."""
        risk_factors = []
        risk_score = 0
        
        if analysis:
            book_data = analysis.get('book_data', {})
            spread_analysis = analysis.get('spread_analysis', {})
            
            # Data quality risks
            if book_data.get('delay_ms', 0) > 2000:
                risk_factors.append('high_data_delay')
                risk_score += 0.3
            
            # Market structure risks
            total_orders = book_data.get('total_bids_filtered', 0) + book_data.get('total_asks_filtered', 0)
            if total_orders < 10:
                risk_factors.append('thin_market')
                risk_score += 0.4
            
            # Manipulation risks
            if spread_analysis.get('manipulation_detected', False):
                risk_factors.append('manipulation_detected')
                risk_score += 0.5
            
            # Model performance risks
            if state.formula_performance_score < 0.6:
                risk_factors.append('poor_model_performance')
                risk_score += 0.3
            
            # Time decay risks
            metrics = analysis.get('market_metrics', {})
            time_decay = metrics.get('time_decay', 0)
            if time_decay > 1.5:
                risk_factors.append('high_time_pressure')
                risk_score += 0.2
        
        # Overall risk level
        if risk_score >= 0.8:
            risk_level = 'critical'
        elif risk_score >= 0.5:
            risk_level = 'high'
        elif risk_score >= 0.3:
            risk_level = 'moderate'
        elif risk_score >= 0.1:
            risk_level = 'low'
        else:
            risk_level = 'minimal'
        
        return {
            'overall_risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'risk_details': {
                'data_quality_risk': 'high_data_delay' in risk_factors,
                'market_structure_risk': 'thin_market' in risk_factors,
                'manipulation_risk': 'manipulation_detected' in risk_factors,
                'model_risk': 'poor_model_performance' in risk_factors,
                'time_risk': 'high_time_pressure' in risk_factors
            }
        }
    
    def _get_trading_statistics(self) -> Dict:
        """Get trading-related statistics."""
        
        # Calculate current token value and capacity
        current_token_value = 0
        remaining_capacity = 0
        capacity_percentage = 0
        
        try:
            # This is a bit of a hack since we don't have direct access to trading executor here
            # But we can estimate from recent analysis if available
            if hasattr(state, 'current_analysis') and state.current_analysis:
                book_data = state.current_analysis.get('book_data', {})
                market_mid = book_data.get('market_mid', 0) or 0
                
                # We don't have direct balance access here, so we'll estimate or mark as unknown
                current_token_value = 0  # Would need executor access
                
        except:
            pass
        
        from collar_config import MAX_TOKEN_VALUE_USD, MAX_TRADE_AMOUNT
        
        return {
            'api_calls_count': state.api_calls_count,
            'formula_performance_score': state.formula_performance_score,
            'last_update_ago_seconds': time.time() - state.last_update_time if state.last_update_time > 0 else 0,
            'training_samples_total': len(self.training_data),
            'training_samples_last_hour': len([s for s in self.training_data if time.time() - s['timestamp'] < 3600]),
            'manipulation_alerts_count': len(state.manipulation_alerts) if hasattr(state, 'manipulation_alerts') else 0,
            'token_value_limit': {
                'max_token_value_usd': MAX_TOKEN_VALUE_USD,
                'max_trade_amount': MAX_TRADE_AMOUNT,
                'current_token_value_usd': current_token_value,
                'remaining_capacity_usd': max(0, MAX_TOKEN_VALUE_USD - current_token_value),
                'capacity_used_percentage': (current_token_value / MAX_TOKEN_VALUE_USD * 100) if MAX_TOKEN_VALUE_USD > 0 else 0
            }
        }
    
    def _assess_model_health(self) -> Dict:
        """Assess overall model health."""
        health_score = 1.0
        health_issues = []
        
        # Training data health
        if len(self.training_data) < 20:
            health_issues.append('insufficient_training_data')
            health_score -= 0.3
        
        # Performance health
        if state.formula_performance_score < 0.7:
            health_issues.append('low_accuracy')
            health_score -= 0.2
        
        # Calibration health
        if time.time() - self.last_calibration > 1800:  # 30 minutes
            health_issues.append('stale_calibration')
            health_score -= 0.1
        
        # Feature health
        if not SKLEARN_AVAILABLE:
            health_issues.append('limited_ml_capabilities')
            health_score -= 0.1
        
        health_score = max(0, health_score)
        
        if health_score >= 0.8:
            health_status = 'excellent'
        elif health_score >= 0.6:
            health_status = 'good'
        elif health_score >= 0.4:
            health_status = 'fair'
        elif health_score >= 0.2:
            health_status = 'poor'
        else:
            health_status = 'critical'
        
        return {
            'health_status': health_status,
            'health_score': health_score,
            'health_issues': health_issues,
            'recommendations': self._get_health_recommendations(health_issues)
        }
    
    def _analyze_manipulation_signals(self) -> Dict:
        """Analyze market manipulation signals."""
        if not hasattr(state, 'manipulation_alerts'):
            return {'status': 'no_alerts_system'}
        
        current_time = time.time()
        recent_alerts = [alert for alert in state.manipulation_alerts if current_time - alert['timestamp'] < 1800]  # Last 30 min
        
        return {
            'status': 'monitored',
            'recent_alerts_count': len(recent_alerts),
            'total_alerts_today': len(state.manipulation_alerts),
            'currently_manipulated': len(recent_alerts) > 0 and current_time - state.manipulation_alerts[-1]['timestamp'] < 300,
            'recent_alerts': recent_alerts[-5:] if recent_alerts else []
        }
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Model performance recommendations
        if state.formula_performance_score < 0.6:
            recommendations.append("Model accuracy below 60% - consider recalibration or data cleaning")
        
        # Data quality recommendations
        if analysis and analysis.get('book_data', {}).get('delay_ms', 0) > 2000:
            recommendations.append("High data delay detected - check data feed connection")
        
        # Market condition recommendations
        if analysis and analysis.get('spread_analysis', {}).get('manipulation_detected', False):
            recommendations.append("Market manipulation detected - using conservative base-level pricing")
        
        # Training data recommendations
        if len(self.training_data) < 30:
            recommendations.append("Low training data - model predictions may be unreliable")
        
        # Time-based recommendations
        now = datetime.now()
        minutes_to_expiry = (60 - now.minute)
        if minutes_to_expiry < 10:
            recommendations.append("Approaching expiry - expect increased volatility and manipulation")
        
        # Token value limit recommendations
        try:
            from collar_config import MAX_TOKEN_VALUE_USD
            if analysis:
                book_data = analysis.get('book_data', {})
                market_mid = book_data.get('market_mid', 0) or 0
                
                # We can't easily get balance here, so we'll provide general guidance
                recommendations.append(f"Token value limit set to ${MAX_TOKEN_VALUE_USD:.2f} - monitor capacity in trading status")
                
        except:
            pass
        
        # Library recommendations
        if not SKLEARN_AVAILABLE:
            recommendations.append("Install scikit-learn for enhanced ML capabilities: pip3 install scikit-learn")
        
        if not recommendations:
            recommendations.append("All systems operating normally")
        
        return recommendations
    
    def _get_health_recommendations(self, health_issues: List[str]) -> List[str]:
        """Get recommendations based on health issues."""
        recommendations = []
        
        for issue in health_issues:
            if issue == 'insufficient_training_data':
                recommendations.append("Allow more time for data collection or reduce minimum sample requirements")
            elif issue == 'low_accuracy':
                recommendations.append("Review feature engineering or consider simpler model")
            elif issue == 'stale_calibration':
                recommendations.append("Force model recalibration or reduce calibration interval")
            elif issue == 'limited_ml_capabilities':
                recommendations.append("Install additional ML libraries for enhanced performance")
        
        return recommendations
    
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
            X = []
            y = []
            for sample in self.training_data:
                X.append(sample['features'])
                y.append(sample['spread'])
            
            if NUMPY_AVAILABLE:
                X = np.array(X)
                y = np.array(y)
            
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
            
            if NUMPY_AVAILABLE:
                mae = np.mean(np.abs(predictions - y))
                r2_score = self.model.score(X, y)
            else:
                mae = sum(abs(p - y[i]) for i, p in enumerate(predictions)) / len(predictions)
                r2_score = 0.5  # Approximate
            
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
            
            # Create a simple linear model
            features = [sample['features'] for sample in recent_data]
            spreads = [sample['spread'] for sample in recent_data]
            
            if NUMPY_AVAILABLE:
                # Use numpy for more sophisticated calculation
                features_array = np.array(features)
                spreads_array = np.array(spreads)
                
                # Simple multiple linear regression: spread = w0*btc_vol + w1*token_vol + w2*momentum + ... + bias
                X = np.column_stack([features_array, np.ones(len(features_array))])  # Add bias term
                
                # Solve using least squares: weights = (X^T * X)^-1 * X^T * y
                try:
                    weights = np.linalg.solve(X.T @ X, X.T @ spreads_array)
                    self.model = {'weights': weights.tolist(), 'type': 'simple_linear'}
                    self.last_calibration = time.time()
                    
                    # Calculate simple accuracy
                    predictions = X @ weights
                    mae = np.mean(np.abs(predictions - spreads_array))
                    
                    print(f"🎯 Simple model calibrated: MAE={mae:.4f}, samples={len(recent_data)}")
                    self.save_formula_to_file()
                    return True
                    
                except np.linalg.LinAlgError:
                    print("❌ Matrix inversion failed - using average")
                    avg_spread = np.mean(spreads_array)
                    self.model = {'average': float(avg_spread), 'type': 'average'}
                    self.last_calibration = time.time()
                    return True
            else:
                # Fallback to simple average without numpy
                avg_spread = sum(spreads) / len(spreads)
                self.model = {'average': avg_spread, 'type': 'average'}
                self.last_calibration = time.time()
                print(f"🎯 Simple average model: ${avg_spread:.4f}, samples={len(recent_data)}")
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
                if NUMPY_AVAILABLE:
                    features = np.array([[btc_volatility, token_volatility, short_momentum, very_short_momentum, time_decay, latest_btc_change]])
                    predicted_spread = self.model.predict(features)[0]
                else:
                    # Fallback without numpy
                    features = [btc_volatility, token_volatility, short_momentum, very_short_momentum, time_decay, latest_btc_change]
                    predicted_spread = self.model.predict([features])[0]
                
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
                features = [btc_vol, token_vol, short_mom, very_short_mom, time_decay, latest_btc_change, 1.0]  # Include bias
                weights = self.model['weights']
                
                if NUMPY_AVAILABLE:
                    features_array = np.array(features)
                    weights_array = np.array(weights)
                    predicted_spread = np.dot(features_array, weights_array)
                else:
                    # Manual dot product without numpy
                    predicted_spread = sum(f * w for f, w in zip(features, weights))
                
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
                'manipulation_detected': manipulation_detected
            },
            'target_prices': {
                'bid': target_bid,
                'ask': target_ask
            }
        }
        
        # Store in global state
        state.current_analysis = analysis
        state.last_update_time = time.time()
        
        # Export comprehensive analysis every 5 minutes - MOVED TO AFTER analysis is created
        self.spread_calculator.export_comprehensive_analysis(analysis)
        
        # Add comprehensive export status to analysis
        analysis['comprehensive_export'] = {
            'last_export_ago_minutes': (time.time() - self.spread_calculator.last_comprehensive_export) / 60,
            'next_export_in_minutes': max(0, 5 - (time.time() - self.spread_calculator.last_comprehensive_export) / 60)
        }
        
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
            if NUMPY_AVAILABLE:
                avg_accuracy = np.mean(recent_accuracy)
            else:
                avg_accuracy = sum(recent_accuracy) / len(recent_accuracy)
            print(f"📊 Recent 10 predictions: {avg_accuracy:.1%} average accuracy")
    
    # Comprehensive Export Status
    if 'comprehensive_export' in analysis:
        export_info = analysis['comprehensive_export']
        last_export = export_info['last_export_ago_minutes']
        next_export = export_info['next_export_in_minutes']
        
        if last_export < 5:
            print(f"📋 Comprehensive Analysis: Last exported {last_export:.1f}min ago")
        else:
            print(f"📋 Comprehensive Analysis: Next export in {next_export:.1f}min")
    
    print("=" * 90)