# tree_based_spread_model.py - Advanced Tree-Based Spread Prediction
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

# Try multiple tree-based libraries with fallbacks
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

@dataclass
class MarketRegime:
    """Define market regimes for rule-based decisions."""
    name: str
    btc_vol_min: float
    btc_vol_max: float
    token_vol_min: float
    token_vol_max: float
    time_decay_min: float
    time_decay_max: float
    base_spread: float
    volatility_multiplier: float
    momentum_impact: float

class TreeBasedSpreadCalculator:
    """Advanced tree-based spread prediction with multiple models."""
    
    def __init__(self):
        self.models = {}
        self.model_performance = {}
        self.training_data = []
        self.feature_names = [
            'btc_volatility', 'token_volatility', 'short_momentum', 
            'very_short_momentum', 'time_decay', 'latest_btc_change',
            'market_stress', 'volume_ratio', 'price_level'
        ]
        
        # Define market regimes for interpretable rules
        self.market_regimes = [
            MarketRegime("ultra_stable", 0.0, 0.02, 0.0, 0.03, 0.0, 0.4, 0.03, 1.0, 0.5),
            MarketRegime("stable", 0.02, 0.05, 0.03, 0.06, 0.0, 0.6, 0.04, 1.2, 0.7),
            MarketRegime("moderate", 0.05, 0.10, 0.06, 0.12, 0.0, 1.0, 0.06, 1.5, 1.0),
            MarketRegime("volatile", 0.10, 0.20, 0.12, 0.25, 0.6, 1.5, 0.08, 2.0, 1.5),
            MarketRegime("extreme", 0.20, 1.0, 0.25, 1.0, 1.0, 3.0, 0.12, 3.0, 2.0)
        ]
        
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize different tree-based models."""
        
        # 1. Rule-Based Decision Tree (always available)
        self.models['rule_based'] = RuleBasedSpreadModel(self.market_regimes)
        
        if SKLEARN_AVAILABLE:
            # 2. Random Forest - Great for stability and feature importance
            self.models['random_forest'] = RandomForestRegressor(
                n_estimators=50,        # Not too many for limited data
                max_depth=5,            # Prevent overfitting
                min_samples_split=5,    # Require minimum samples
                min_samples_leaf=3,     # Prevent tiny leaves
                random_state=42
            )
            
            # 3. Gradient Boosting - Sequential learning
            self.models['gradient_boost'] = GradientBoostingRegressor(
                n_estimators=30,        # Conservative for small data
                max_depth=3,            # Shallow trees
                learning_rate=0.1,      # Slow learning
                subsample=0.8,          # Bootstrap samples
                random_state=42
            )
            
            # 4. Simple Decision Tree - Most interpretable
            self.models['decision_tree'] = DecisionTreeRegressor(
                max_depth=4,            # Very shallow for interpretability
                min_samples_split=8,    # Conservative splitting
                min_samples_leaf=5,     # Minimum leaf size
                random_state=42
            )
        
        if XGBOOST_AVAILABLE:
            # 5. XGBoost - Often best performance
            self.models['xgboost'] = xgb.XGBRegressor(
                n_estimators=25,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        
        if LIGHTGBM_AVAILABLE:
            # 6. LightGBM - Fast and efficient
            self.models['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=25,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                feature_fraction=0.8,
                random_state=42,
                verbose=-1
            )
    
    def engineer_features(self, btc_vol: float, token_vol: float, short_mom: float, 
                         very_short_mom: float, time_decay: float, btc_change: float,
                         market_mid: float = 0.5, bid_count: int = 10, ask_count: int = 10) -> List[float]:
        """Engineer additional features for better prediction."""
        
        # Original features
        features = [btc_vol, token_vol, short_mom, very_short_mom, time_decay, btc_change]
        
        # Engineered features
        market_stress = abs(short_mom) + abs(very_short_mom) + btc_vol + abs(btc_change)
        volume_ratio = min(bid_count, ask_count) / max(bid_count, ask_count) if max(bid_count, ask_count) > 0 else 0.5
        price_level = abs(market_mid - 0.5)  # Distance from 50% (maximum uncertainty)
        
        features.extend([market_stress, volume_ratio, price_level])
        return features
    
    def predict_ensemble(self, features: List[float]) -> Tuple[float, Dict[str, float]]:
        """Get ensemble prediction from all available models."""
        predictions = {}
        weights = {}
        
        # Always available rule-based prediction
        rule_pred = self.models['rule_based'].predict(features)
        predictions['rule_based'] = rule_pred
        weights['rule_based'] = 0.3  # Base weight
        
        # ML model predictions (if available and trained)
        for model_name, model in self.models.items():
            if model_name == 'rule_based':
                continue
                
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict([features])[0]
                    predictions[model_name] = pred
                    
                    # Weight based on historical performance
                    performance = self.model_performance.get(model_name, 0.5)
                    weights[model_name] = performance
            except Exception as e:
                print(f"⚠️ {model_name} prediction failed: {e}")
        
        # Ensemble calculation
        if not predictions:
            return 0.05, {}  # Fallback
        
        total_weight = sum(weights.values())
        if total_weight == 0:
            ensemble_pred = np.mean(list(predictions.values()))
        else:
            ensemble_pred = sum(pred * weights[name] / total_weight 
                              for name, pred in predictions.items())
        
        return max(0.02, min(0.20, ensemble_pred)), predictions
    
    def train_models(self, training_data: List[Dict]) -> Dict[str, float]:
        """Train all available models and return performance metrics."""
        if len(training_data) < 10:
            return {}
        
        # Prepare data
        X = []
        y = []
        
        for sample in training_data:
            features = sample.get('features', [])
            if len(features) < 6:
                continue
                
            # Engineer additional features
            engineered = self.engineer_features(*features[:6])
            X.append(engineered)
            y.append(sample['spread'])
        
        if len(X) < 10:
            return {}
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data if we have enough
        if len(X) > 20:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        else:
            X_train, X_test, y_train, y_test = X, X, y, y  # Use all data for both
        
        performance = {}
        
        # Train sklearn models
        for model_name, model in self.models.items():
            if model_name == 'rule_based':
                continue
                
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred) if len(set(y_test)) > 1 else 0
                
                # Performance score (0-1, higher is better)
                accuracy = max(0, 1 - mae / np.mean(y_test)) if np.mean(y_test) > 0 else 0
                performance[model_name] = accuracy
                
                print(f"📊 {model_name}: MAE={mae:.4f}, R²={r2:.3f}, Accuracy={accuracy:.1%}")
                
            except Exception as e:
                print(f"❌ {model_name} training failed: {e}")
                performance[model_name] = 0.0
        
        self.model_performance.update(performance)
        return performance
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance from tree models."""
        importance_data = {}
        
        for model_name, model in self.models.items():
            if model_name == 'rule_based':
                continue
                
            try:
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    importance_dict = {
                        self.feature_names[i]: importances[i] 
                        for i in range(min(len(self.feature_names), len(importances)))
                    }
                    importance_data[model_name] = importance_dict
            except:
                continue
        
        return importance_data
    
    def generate_decision_rules(self) -> List[str]:
        """Generate human-readable decision rules from the best tree model."""
        rules = []
        
        # Rule-based logic
        rules.append("📋 RULE-BASED DECISION LOGIC:")
        for regime in self.market_regimes:
            rule = (f"IF BTC_vol({regime.btc_vol_min:.2f}-{regime.btc_vol_max:.2f}) "
                   f"AND token_vol({regime.token_vol_min:.2f}-{regime.token_vol_max:.2f}) "
                   f"AND time_decay({regime.time_decay_min:.1f}-{regime.time_decay_max:.1f}) "
                   f"→ Base: ${regime.base_spread:.3f}, Mult: {regime.volatility_multiplier:.1f}x")
            rules.append(f"  {regime.name.upper()}: {rule}")
        
        # Try to extract rules from decision tree if available
        if 'decision_tree' in self.models:
            try:
                tree = self.models['decision_tree']
                if hasattr(tree, 'tree_'):
                    rules.append("\n🌳 DECISION TREE RULES:")
                    rules.extend(self._extract_tree_rules(tree, self.feature_names))
            except:
                pass
        
        return rules
    
    def _extract_tree_rules(self, tree_model, feature_names: List[str], max_rules: int = 10) -> List[str]:
        """Extract interpretable rules from decision tree."""
        from sklearn.tree import _tree
        
        tree_ = tree_model.tree_
        rules = []
        
        def recurse(node, depth, condition=""):
            if len(rules) >= max_rules:
                return
                
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_names[tree_.feature[node]]
                threshold = tree_.threshold[node]
                
                left_condition = f"{condition} AND {name} <= {threshold:.3f}" if condition else f"{name} <= {threshold:.3f}"
                right_condition = f"{condition} AND {name} > {threshold:.3f}" if condition else f"{name} > {threshold:.3f}"
                
                recurse(tree_.children_left[node], depth + 1, left_condition)
                recurse(tree_.children_right[node], depth + 1, right_condition)
            else:
                # Leaf node
                spread_value = tree_.value[node][0][0]
                samples = tree_.n_node_samples[node]
                rule = f"  IF {condition} → Spread: ${spread_value:.3f} (n={samples})"
                rules.append(rule)
        
        recurse(0, 1)
        return rules

class RuleBasedSpreadModel:
    """Simple rule-based model using market regimes."""
    
    def __init__(self, regimes: List[MarketRegime]):
        self.regimes = regimes
    
    def predict(self, features: List[float]) -> float:
        """Predict spread using rule-based logic."""
        if len(features) < 6:
            return 0.05
        
        btc_vol, token_vol, short_mom, very_short_mom, time_decay, btc_change = features[:6]
        
        # Find matching regime
        for regime in self.regimes:
            if (regime.btc_vol_min <= btc_vol <= regime.btc_vol_max and
                regime.token_vol_min <= token_vol <= regime.token_vol_max and
                regime.time_decay_min <= time_decay <= regime.time_decay_max):
                
                # Calculate spread based on regime
                base_spread = regime.base_spread
                volatility_impact = (btc_vol + token_vol) * regime.volatility_multiplier
                momentum_impact = (abs(short_mom) + abs(very_short_mom) + abs(btc_change)) * regime.momentum_impact
                
                final_spread = base_spread + volatility_impact + momentum_impact
                return max(0.02, min(0.20, final_spread))
        
        # Fallback: conservative spread
        return 0.06

def demonstrate_tree_models():
    """Demonstrate different tree-based approaches with sample data."""
    
    print("🌳 TREE-BASED SPREAD PREDICTION DEMONSTRATION")
    print("=" * 60)
    
    # Initialize calculator
    calc = TreeBasedSpreadCalculator()
    
    # Sample training data (simulated diverse conditions)
    sample_training_data = [
        # Ultra stable conditions
        {'features': [0.01, 0.02, 0.0, 0.0, 0.1, 0.0], 'spread': 0.03},
        {'features': [0.015, 0.025, 0.01, 0.0, 0.2, 0.001], 'spread': 0.035},
        
        # Moderate volatility
        {'features': [0.05, 0.08, 0.05, 0.02, 0.5, 0.01], 'spread': 0.06},
        {'features': [0.07, 0.10, -0.03, 0.015, 0.6, -0.005], 'spread': 0.065},
        
        # High volatility
        {'features': [0.15, 0.20, 0.10, 0.05, 1.0, 0.02], 'spread': 0.10},
        {'features': [0.18, 0.25, -0.08, -0.03, 1.2, -0.015], 'spread': 0.12},
        
        # Extreme conditions
        {'features': [0.30, 0.40, 0.15, 0.08, 2.0, 0.03], 'spread': 0.15},
        {'features': [0.25, 0.35, -0.12, 0.06, 1.8, 0.025], 'spread': 0.14},
        
        # Time decay scenarios
        {'features': [0.03, 0.05, 0.02, 0.01, 0.1, 0.0], 'spread': 0.04},  # Early hour
        {'features': [0.03, 0.05, 0.02, 0.01, 2.5, 0.0], 'spread': 0.08},  # Final minutes
    ]
    
    # Train models
    print("🎯 Training tree-based models...")
    performance = calc.train_models(sample_training_data)
    
    # Test predictions
    print("\n🔮 Sample Predictions:")
    test_cases = [
        ([0.02, 0.03, 0.0, 0.0, 0.2, 0.0], "Stable conditions"),
        ([0.10, 0.15, 0.08, 0.03, 1.0, 0.015], "High volatility"),
        ([0.01, 0.02, 0.0, 0.0, 2.0, 0.0], "Time decay (final minutes)"),
        ([0.25, 0.30, -0.10, 0.05, 1.5, 0.02], "Extreme volatility")
    ]
    
    for features, description in test_cases:
        engineered = calc.engineer_features(*features)
        spread, predictions = calc.predict_ensemble(engineered)
        
        print(f"\n📊 {description}:")
        print(f"   Ensemble: ${spread:.3f}")
        for model_name, pred in predictions.items():
            print(f"   {model_name}: ${pred:.3f}")
    
    # Feature importance
    print("\n📈 Feature Importance:")
    importance = calc.get_feature_importance()
    for model_name, features in importance.items():
        print(f"\n{model_name}:")
        sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features[:5]:
            print(f"  {feature}: {importance:.3f}")
    
    # Decision rules
    print("\n📋 Generated Decision Rules:")
    rules = calc.generate_decision_rules()
    for rule in rules[:15]:  # Show first 15 rules
        print(rule)
    
    return calc

if __name__ == "__main__":
    # Run demonstration
    calculator = demonstrate_tree_models()
    
    print("\n✅ Tree-based spread calculator ready!")
    print("🎯 Available models:", list(calculator.models.keys()))
    print("📊 Model performance:", calculator.model_performance)