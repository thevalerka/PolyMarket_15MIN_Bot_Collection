# improved_collar_strategy.py - Integration of Tree-Based Spread Models
import json
import time
import math
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Import our new tree-based calculator
from tree_based_spread_model import TreeBasedSpreadCalculator, MarketRegime

from collar_config import state, MIN_QUANTITY_FILTER, MIN_SPREAD, MAX_SPREAD, SPREAD_BUFFER, SPREAD_FORMULA_PATH

class ImprovedDynamicSpreadCalculator:
    """Enhanced spread calculator using tree-based models and better data collection."""
    
    def __init__(self):
        self.tree_calculator = TreeBasedSpreadCalculator()
        self.training_data = []
        self.last_calibration = 0
        self.calibration_interval = 300  # Recalibrate every 5 minutes
        self.min_training_samples = 15  # Minimum samples before ML training
        
        # Enhanced data collection
        self.market_stress_history = []
        self.volume_imbalance_history = []
        self.price_acceleration_history = []
        
    def load_formula_from_file(self) -> bool:
        """Load spread formula from JSON file with backward compatibility."""
        try:
            with open(SPREAD_FORMULA_PATH, 'r') as f:
                data = json.load(f)
            
            # Check if formula is from current hour
            current_hour = datetime.now().hour
            formula_hour = data.get('hour', -1)
            
            if formula_hour == current_hour:
                # Load training data
                self.training_data = data.get('training_data', [])
                self.last_calibration = data.get('last_calibration', 0)
                
                # Train models with loaded data
                if len(self.training_data) >= self.min_training_samples:
                    self.tree_calculator.train_models(self.training_data)
                    print(f"✅ Loaded tree models with {len(self.training_data)} samples")
                else:
                    print(f"⚠️ Insufficient training data ({len(self.training_data)} samples)")
                
                return True
            else:
                print(f"🔄 Formula from different hour ({formula_hour} vs {current_hour}) - starting fresh")
                return False
                
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"⚠️ Could not load spread formula: {e}")
            return False
    
    def save_formula_to_file(self):
        """Save current spread formula and model performance to JSON file."""
        try:
            data = {
                'hour': datetime.now().hour,
                'last_calibration': self.last_calibration,
                'model_type': 'tree_based_ensemble',
                'available_models': list(self.tree_calculator.models.keys()),
                'model_performance': self.tree_calculator.model_performance,
                'training_data': self.training_data[-200:],  # Keep last 200 samples
                'feature_importance': self.tree_calculator.get_feature_importance(),
                'decision_rules': self.tree_calculator.generate_decision_rules()[:10],
                'timestamp': datetime.now().isoformat()
            }
            
            with open(SPREAD_FORMULA_PATH, 'w') as f:
                json.dump(data, f, indent=2)
            
            avg_performance = np.mean(list(self.tree_calculator.model_performance.values())) if self.tree_calculator.model_performance else 0
            print(f"💾 Tree-based models saved (avg performance: {avg_performance:.1%})")
            
        except Exception as e:
            print(f"❌ Could not save spread formula: {e}")
    
    def add_training_sample(self, features: List[float], actual_spread: float, market_context: Dict = None):
        """Add training sample with enhanced market context."""
        if len(features) >= 6 and actual_spread > 0:
            
            # Enhanced feature engineering with market context
            if market_context:
                bid_count = market_context.get('bid_count', 10)
                ask_count = market_context.get('ask_count', 10)
                market_mid = market_context.get('market_mid', 0.5)
                
                # Add market stress indicators
                features = self.tree_calculator.engineer_features(
                    *features[:6], market_mid, bid_count, ask_count
                )
            
            sample = {
                'timestamp': time.time(),
                'features': features,
                'spread': actual_spread,
                'market_context': market_context or {}
            }
            
            self.training_data.append(sample)
            
            # Keep only recent data (last 500 samples or 2 hours)
            cutoff_time = time.time() - 7200
            self.training_data = [
                s for s in self.training_data 
                if s['timestamp'] > cutoff_time
            ][-500:]  # Also limit to 500 samples
    
    def calibrate_models(self) -> bool:
        """Calibrate tree-based models with current training data."""
        if len(self.training_data) < self.min_training_samples:
            print(f"⚠️ Insufficient data for calibration: {len(self.training_data)}/{self.min_training_samples}")
            return False
        
        try:
            print(f"🌳 Calibrating tree-based models with {len(self.training_data)} samples...")
            
            # Train all available models
            performance = self.tree_calculator.train_models(self.training_data)
            
            if performance:
                self.last_calibration = time.time()
                
                # Show best performing models
                best_models = sorted(performance.items(), key=lambda x: x[1], reverse=True)[:3]
                print("🏆 Top performing models:")
                for model_name, score in best_models:
                    print(f"   {model_name}: {score:.1%}")
                
                # Display feature importance from best model
                importance = self.tree_calculator.get_feature_importance()
                if importance:
                    best_model_name = best_models[0][0]
                    if best_model_name in importance:
                        print(f"\n📊 Feature importance ({best_model_name}):")
                        sorted_features = sorted(
                            importance[best_model_name].items(), 
                            key=lambda x: x[1], reverse=True
                        )
                        for feature, imp in sorted_features[:5]:
                            print(f"   {feature}: {imp:.3f}")
                
                # Save updated models
                self.save_formula_to_file()
                return True
            else:
                print("❌ Model training failed")
                return False
                
        except Exception as e:
            print(f"❌ Model calibration error: {e}")
            return False
    
    def predict_optimal_spread(self, btc_volatility: float, token_volatility: float, 
                             short_momentum: float, very_short_momentum: float, 
                             time_decay: float, latest_btc_change: float,
                             market_context: Dict = None) -> Tuple[float, Dict]:
        """Predict optimal spread using tree-based ensemble."""
        
        try:
            # Prepare base features
            base_features = [btc_volatility, token_volatility, short_momentum, 
                           very_short_momentum, time_decay, latest_btc_change]
            
            # Engineer enhanced features
            if market_context:
                bid_count = market_context.get('bid_count', 10)
                ask_count = market_context.get('ask_count', 10) 
                market_mid = market_context.get('market_mid', 0.5)
                
                features = self.tree_calculator.engineer_features(
                    *base_features, market_mid, bid_count, ask_count
                )
            else:
                features = self.tree_calculator.engineer_features(*base_features)
            
            # Get ensemble prediction
            predicted_spread, model_predictions = self.tree_calculator.predict_ensemble(features)
            
            # Ensure within bounds
            final_spread = max(MIN_SPREAD, min(MAX_SPREAD, predicted_spread))
            
            # Return prediction details
            prediction_info = {
                'ensemble_prediction': final_spread,
                'model_predictions': model_predictions,
                'model_performance': self.tree_calculator.model_performance,
                'features_used': len(features),
                'regime_detected': self._detect_market_regime(base_features)
            }
            
            return final_spread, prediction_info
            
        except Exception as e:
            print(f"❌ Tree-based prediction failed: {e}")
            # Fallback to simple calculation
            base_spread = (btc_volatility + token_volatility) * 0.5
            momentum_factor = abs(short_momentum) + abs(very_short_momentum)
            btc_impact = abs(latest_btc_change) * 2.0
            
            fallback_spread = base_spread * (1 + time_decay + momentum_factor + btc_impact)
            return max(MIN_SPREAD, min(MAX_SPREAD, fallback_spread)), {'fallback': True}
    
    def _detect_market_regime(self, features: List[float]) -> str:
        """Detect current market regime for interpretability."""
        if len(features) < 6:
            return "unknown"
        
        btc_vol, token_vol, short_mom, very_short_mom, time_decay, btc_change = features[:6]
        
        # Use the rule-based model to identify regime
        for regime in self.tree_calculator.market_regimes:
            if (regime.btc_vol_min <= btc_vol <= regime.btc_vol_max and
                regime.token_vol_min <= token_vol <= regime.token_vol_max and
                regime.time_decay_min <= time_decay <= regime.time_decay_max):
                return regime.name
        
        return "undefined"
    
    def should_recalibrate(self) -> bool:
        """Determine if models should be recalibrated."""
        # Time-based recalibration
        time_based = time.time() - self.last_calibration > self.calibration_interval
        
        # Performance-based recalibration
        avg_performance = 0
        if self.tree_calculator.model_performance:
            avg_performance = np.mean(list(self.tree_calculator.model_performance.values()))
        
        performance_based = avg_performance < 0.6  # Recalibrate if performance drops below 60%
        
        # Data-based recalibration (new data available)
        data_based = len(self.training_data) % 25 == 0 and len(self.training_data) >= self.min_training_samples
        
        return time_based or performance_based or data_based
    
    def get_model_insights(self) -> Dict:
        """Get insights about current model state."""
        insights = {
            'total_training_samples': len(self.training_data),
            'available_models': list(self.tree_calculator.models.keys()),
            'model_performance': self.tree_calculator.model_performance,
            'last_calibration_ago': time.time() - self.last_calibration,
            'next_calibration_in': max(0, self.calibration_interval - (time.time() - self.last_calibration))
        }
        
        # Recent prediction accuracy
        if hasattr(state, 'formula_accuracy_history') and state.formula_accuracy_history:
            recent_accuracy = [acc for _, _, _, acc in state.formula_accuracy_history[-20:]]
            if recent_accuracy:
                insights['recent_accuracy'] = np.mean(recent_accuracy)
                insights['accuracy_trend'] = 'improving' if len(recent_accuracy) > 10 and np.mean(recent_accuracy[-5:]) > np.mean(recent_accuracy[:5]) else 'stable'
        
        # Decision rules summary
        try:
            rules = self.tree_calculator.generate_decision_rules()
            insights['decision_rules_count'] = len(rules)
            insights['sample_rules'] = rules[:5]
        except:
            insights['decision_rules_count'] = 0
        
        return insights

# Example integration with the existing collar strategy
def integrate_tree_models_demo():
    """Demonstrate integration with existing collar strategy."""
    
    print("🔗 INTEGRATING TREE-BASED MODELS WITH COLLAR STRATEGY")
    print("=" * 70)
    
    # Initialize improved calculator
    calc = ImprovedDynamicSpreadCalculator()
    
    # Simulate market conditions over time
    market_scenarios = [
        {
            'name': 'Market Open (Stable)',
            'btc_vol': 0.02, 'token_vol': 0.04, 'short_mom': 0.01, 
            'very_short_mom': 0.0, 'time_decay': 0.1, 'btc_change': 0.001,
            'context': {'bid_count': 25, 'ask_count': 30, 'market_mid': 0.52}
        },
        {
            'name': 'Mid-Hour Volatility',
            'btc_vol': 0.08, 'token_vol': 0.12, 'short_mom': 0.06, 
            'very_short_mom': 0.03, 'time_decay': 0.5, 'btc_change': 0.015,
            'context': {'bid_count': 15, 'ask_count': 12, 'market_mid': 0.48}
        },
        {
            'name': 'Final Minutes Rush',
            'btc_vol': 0.03, 'token_vol': 0.06, 'short_mom': -0.04, 
            'very_short_mom': 0.08, 'time_decay': 2.0, 'btc_change': 0.005,
            'context': {'bid_count': 8, 'ask_count': 6, 'market_mid': 0.35}
        },
        {
            'name': 'Market Manipulation',
            'btc_vol': 0.01, 'token_vol': 0.25, 'short_mom': 0.15, 
            'very_short_mom': -0.12, 'time_decay': 1.0, 'btc_change': 0.0,
            'context': {'bid_count': 3, 'ask_count': 2, 'market_mid': 0.78}
        }
    ]
    
    print("🎯 Testing different market scenarios:")
    
    for scenario in market_scenarios:
        print(f"\n📊 {scenario['name']}:")
        
        # Predict spread
        spread, info = calc.predict_optimal_spread(
            scenario['btc_vol'], scenario['token_vol'], 
            scenario['short_mom'], scenario['very_short_mom'],
            scenario['time_decay'], scenario['btc_change'],
            scenario['context']
        )
        
        print(f"   Predicted Spread: ${spread:.3f}")
        print(f"   Market Regime: {info.get('regime_detected', 'unknown')}")
        print(f"   Models Used: {len(info.get('model_predictions', {}))}")
        
        # Simulate adding actual market spread as training data
        simulated_actual = spread + np.random.normal(0, 0.01)  # Add some noise
        calc.add_training_sample(
            [scenario['btc_vol'], scenario['token_vol'], scenario['short_mom'],
             scenario['very_short_mom'], scenario['time_decay'], scenario['btc_change']],
            simulated_actual,
            scenario['context']
        )
    
    # Calibrate models with new data
    print(f"\n🌳 Calibrating models with {len(calc.training_data)} samples...")
    success = calc.calibrate_models()
    
    if success:
        # Show insights
        insights = calc.get_model_insights()
        print(f"\n📈 Model Insights:")
        print(f"   Training Samples: {insights['total_training_samples']}")
        print(f"   Available Models: {insights['available_models']}")
        print(f"   Recent Accuracy: {insights.get('recent_accuracy', 0):.1%}")
        print(f"   Decision Rules: {insights['decision_rules_count']}")
        
        # Show sample decision rules
        if 'sample_rules' in insights:
            print(f"\n📋 Sample Decision Rules:")
            for rule in insights['sample_rules'][:3]:
                print(f"   {rule}")
    
    return calc

if __name__ == "__main__":
    calculator = integrate_tree_models_demo()
    print(f"\n✅ Tree-based spread calculator integration complete!")
    print(f"🎯 Ready for production use with enhanced interpretability and robustness.")