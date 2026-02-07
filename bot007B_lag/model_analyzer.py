#!/usr/bin/env python3
"""
ML Model Analysis Tool

Comprehensive analysis of trained ML arbitrage models from joblib files.
Analyzes model architecture, performance, feature importance, and training data.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

class MLModelAnalyzer:
    """Comprehensive ML model analysis tool"""
    
    def __init__(self, model_dir='/home/ubuntu/013_2025_polymarket/ml_models'):
        """Initialize the analyzer with model directory"""
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
        
        # Load models and data
        self.models_loaded = False
        self.call_model = None
        self.put_model = None
        self.anomaly_detector = None
        self.scaler = None
        self.training_data = None
        self.metadata = None
        self.recent_predictions = None
        
        self._load_models_and_data()
    
    def _load_models_and_data(self):
        """Load all models and associated data"""
        try:
            print("🔍 Loading ML models and data for analysis...")
            
            # Check if model files exist
            missing_files = []
            for name, path in self.model_files.items():
                if not os.path.exists(path):
                    missing_files.append(f"{name}: {path}")
            
            if missing_files:
                print("❌ Missing model files:")
                for file in missing_files:
                    print(f"   {file}")
                return False
            
            # Load sklearn models
            self.call_model = joblib.load(self.model_files['call_model'])
            self.put_model = joblib.load(self.model_files['put_model'])
            self.anomaly_detector = joblib.load(self.model_files['anomaly_detector'])
            self.scaler = joblib.load(self.model_files['scaler'])
            
            # Load metadata
            with open(self.model_files['model_metadata'], 'r') as f:
                self.metadata = json.load(f)
            
            # Load training data
            with open(self.model_files['training_data'], 'r') as f:
                training_data_list = json.load(f)
                self.training_data = pd.DataFrame(training_data_list)
            
            # Load recent predictions if available
            if os.path.exists(self.model_files['recent_predictions']):
                with open(self.model_files['recent_predictions'], 'r') as f:
                    predictions_list = json.load(f)
                    self.recent_predictions = pd.DataFrame(predictions_list)
            
            self.models_loaded = True
            print("✅ All models and data loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def analyze_model_architecture(self):
        """Analyze the architecture and parameters of the models"""
        if not self.models_loaded:
            return "❌ Models not loaded"
        
        print("\n" + "="*80)
        print("🏗️  MODEL ARCHITECTURE ANALYSIS")
        print("="*80)
        
        # CALL Model Analysis
        print("\n📈 CALL OPTION MODEL (Random Forest):")
        print(f"├─ Estimators: {self.call_model.n_estimators}")
        print(f"├─ Max Depth: {self.call_model.max_depth}")
        print(f"├─ Min Samples Split: {self.call_model.min_samples_split}")
        print(f"├─ Min Samples Leaf: {self.call_model.min_samples_leaf}")
        print(f"├─ Random State: {self.call_model.random_state}")
        print(f"├─ Number of Features: {self.call_model.n_features_in_}")
        print(f"└─ Number of Trees Trained: {len(self.call_model.estimators_)}")
        
        # PUT Model Analysis
        print("\n📉 PUT OPTION MODEL (Random Forest):")
        print(f"├─ Estimators: {self.put_model.n_estimators}")
        print(f"├─ Max Depth: {self.put_model.max_depth}")
        print(f"├─ Min Samples Split: {self.put_model.min_samples_split}")
        print(f"├─ Min Samples Leaf: {self.put_model.min_samples_leaf}")
        print(f"├─ Random State: {self.put_model.random_state}")
        print(f"├─ Number of Features: {self.put_model.n_features_in_}")
        print(f"└─ Number of Trees Trained: {len(self.put_model.estimators_)}")
        
        # Anomaly Detector Analysis
        print("\n🚨 ANOMALY DETECTOR (Isolation Forest):")
        print(f"├─ Contamination: {self.anomaly_detector.contamination}")
        print(f"├─ Max Samples: {self.anomaly_detector.max_samples}")
        print(f"├─ Number of Features: {self.anomaly_detector.n_features_in_}")
        print(f"└─ Random State: {self.anomaly_detector.random_state}")
        
        # Scaler Analysis
        print("\n⚖️  FEATURE SCALER (StandardScaler):")
        print(f"├─ Number of Features: {len(self.scaler.feature_names_in_) if hasattr(self.scaler, 'feature_names_in_') else 'N/A'}")
        print(f"├─ Mean Shape: {self.scaler.mean_.shape}")
        print(f"├─ Scale Shape: {self.scaler.scale_.shape}")
        print(f"└─ With Mean: {self.scaler.with_mean}")
    
    def analyze_feature_importance(self):
        """Analyze feature importance for both models"""
        if not self.models_loaded:
            return "❌ Models not loaded"
        
        print("\n" + "="*80)
        print("🎯 FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        # Get feature names from training data
        feature_cols = [col for col in self.training_data.columns 
                       if col not in ['call_price_change', 'put_price_change', 'timestamp']]
        
        # CALL Model Feature Importance
        call_importance = self.call_model.feature_importances_
        call_feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': call_importance
        }).sort_values('importance', ascending=False)
        
        print("\n📈 CALL MODEL - TOP 10 MOST IMPORTANT FEATURES:")
        for i, (_, row) in enumerate(call_feature_importance.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<25} {row['importance']:.4f}")
        
        # PUT Model Feature Importance
        put_importance = self.put_model.feature_importances_
        put_feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': put_importance
        }).sort_values('importance', ascending=False)
        
        print("\n📉 PUT MODEL - TOP 10 MOST IMPORTANT FEATURES:")
        for i, (_, row) in enumerate(put_feature_importance.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<25} {row['importance']:.4f}")
        
        # Feature importance comparison
        print("\n🔍 FEATURE IMPORTANCE COMPARISON:")
        importance_comparison = pd.merge(
            call_feature_importance[['feature', 'importance']].rename(columns={'importance': 'call_importance'}),
            put_feature_importance[['feature', 'importance']].rename(columns={'importance': 'put_importance'}),
            on='feature'
        )
        importance_comparison['importance_diff'] = importance_comparison['call_importance'] - importance_comparison['put_importance']
        importance_comparison['avg_importance'] = (importance_comparison['call_importance'] + importance_comparison['put_importance']) / 2
        
        print("\nTOP 5 FEATURES BY AVERAGE IMPORTANCE:")
        top_avg = importance_comparison.nlargest(5, 'avg_importance')
        for i, (_, row) in enumerate(top_avg.iterrows(), 1):
            print(f"{i}. {row['feature']:<25} Avg: {row['avg_importance']:.4f} (C: {row['call_importance']:.4f}, P: {row['put_importance']:.4f})")
        
        return call_feature_importance, put_feature_importance, importance_comparison
    
    def analyze_training_performance(self):
        """Analyze training performance and model quality"""
        if not self.models_loaded:
            return "❌ Models not loaded"
        
        print("\n" + "="*80)
        print("📊 TRAINING PERFORMANCE ANALYSIS")
        print("="*80)
        
        # Metadata analysis
        print("\n📋 TRAINING METADATA:")
        print(f"├─ Model Version: v{self.metadata.get('model_version', 'unknown')}")
        print(f"├─ Training Samples: {self.metadata.get('training_samples', 0):,}")
        print(f"├─ Training Attempts: {self.metadata.get('training_attempts', 0)}")
        print(f"├─ Successful Trainings: {self.metadata.get('successful_trainings', 0)}")
        
        success_rate = self.metadata.get('successful_trainings', 0) / max(self.metadata.get('training_attempts', 1), 1)
        print(f"├─ Training Success Rate: {success_rate:.1%}")
        
        if 'last_save_time' in self.metadata:
            last_save = datetime.fromisoformat(self.metadata['last_save_time'])
            time_since_save = (datetime.now() - last_save).total_seconds() / 3600
            print(f"└─ Last Save: {last_save.strftime('%Y-%m-%d %H:%M:%S')} ({time_since_save:.1f}h ago)")
        
        # Model performance metrics
        performance = self.metadata.get('model_performance', {})
        print(f"\n🎯 MODEL PERFORMANCE METRICS:")
        print(f"├─ CALL Model MAE: {performance.get('call_mae', 0):.6f}")
        print(f"├─ CALL Model R²: {performance.get('call_r2', 0):.4f}")
        print(f"├─ PUT Model MAE: {performance.get('put_mae', 0):.6f}")
        print(f"└─ PUT Model R²: {performance.get('put_r2', 0):.4f}")
        
        # Training quality history
        quality_history = self.metadata.get('training_quality_history', [])
        if quality_history:
            print(f"\n📈 TRAINING QUALITY HISTORY:")
            print(f"├─ Number of Quality Records: {len(quality_history)}")
            print(f"├─ Average Quality Score: {np.mean(quality_history):.4f}")
            print(f"├─ Best Quality Score: {np.max(quality_history):.4f}")
            print(f"├─ Latest Quality Score: {quality_history[-1]:.4f}")
            print(f"└─ Quality Trend: {self._calculate_trend(quality_history)}")
    
    def analyze_training_data(self):
        """Analyze the training data characteristics"""
        if not self.models_loaded or self.training_data is None:
            return "❌ Training data not available"
        
        print("\n" + "="*80)
        print("📈 TRAINING DATA ANALYSIS")
        print("="*80)
        
        print(f"\n📊 DATA OVERVIEW:")
        print(f"├─ Total Samples: {len(self.training_data):,}")
        print(f"├─ Total Features: {len(self.training_data.columns)}")
        print(f"├─ Memory Usage: {self.training_data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # Check for target variables
        has_call_target = 'call_price_change' in self.training_data.columns
        has_put_target = 'put_price_change' in self.training_data.columns
        
        print(f"├─ Has CALL targets: {'✅' if has_call_target else '❌'}")
        print(f"└─ Has PUT targets: {'✅' if has_put_target else '❌'}")
        
        if has_call_target:
            call_changes = self.training_data['call_price_change'].dropna()
            print(f"\n📈 CALL PRICE CHANGES:")
            print(f"├─ Count: {len(call_changes):,}")
            print(f"├─ Mean: {call_changes.mean():+.6f}")
            print(f"├─ Std: {call_changes.std():.6f}")
            print(f"├─ Min: {call_changes.min():+.6f}")
            print(f"├─ Max: {call_changes.max():+.6f}")
            print(f"├─ Variance: {call_changes.var():.8f}")
            print(f"└─ Zero changes: {(call_changes == 0).sum()} ({(call_changes == 0).mean():.1%})")
        
        if has_put_target:
            put_changes = self.training_data['put_price_change'].dropna()
            print(f"\n📉 PUT PRICE CHANGES:")
            print(f"├─ Count: {len(put_changes):,}")
            print(f"├─ Mean: {put_changes.mean():+.6f}")
            print(f"├─ Std: {put_changes.std():.6f}")
            print(f"├─ Min: {put_changes.min():+.6f}")
            print(f"├─ Max: {put_changes.max():+.6f}")
            print(f"├─ Variance: {put_changes.var():.8f}")
            print(f"└─ Zero changes: {(put_changes == 0).sum()} ({(put_changes == 0).mean():.1%})")
        
        # Feature analysis
        feature_cols = [col for col in self.training_data.columns 
                       if col not in ['call_price_change', 'put_price_change', 'timestamp']]
        
        print(f"\n🎯 FEATURE STATISTICS:")
        print(f"├─ Number of Features: {len(feature_cols)}")
        
        # Calculate feature statistics
        feature_stats = []
        for col in feature_cols[:10]:  # Show top 10 features
            if col in self.training_data.columns:
                values = self.training_data[col].dropna()
                if len(values) > 0:
                    feature_stats.append({
                        'feature': col,
                        'mean': values.mean(),
                        'std': values.std(),
                        'variance': values.var(),
                        'min': values.min(),
                        'max': values.max(),
                        'null_count': self.training_data[col].isnull().sum()
                    })
        
        if feature_stats:
            print("└─ Top 10 Features:")
            for i, stat in enumerate(feature_stats, 1):
                print(f"   {i:2d}. {stat['feature']:<20} μ={stat['mean']:+.4f}, σ={stat['std']:.4f}, var={stat['variance']:.6f}")
    
    def analyze_recent_predictions(self):
        """Analyze recent prediction accuracy"""
        if not self.models_loaded or self.recent_predictions is None:
            return "❌ Recent predictions not available"
        
        print("\n" + "="*80)
        print("🔮 RECENT PREDICTIONS ANALYSIS")
        print("="*80)
        
        print(f"\n📊 PREDICTION OVERVIEW:")
        print(f"├─ Total Predictions: {len(self.recent_predictions):,}")
        
        # Convert timestamp if needed
        if 'timestamp' in self.recent_predictions.columns:
            if self.recent_predictions['timestamp'].dtype == 'object':
                self.recent_predictions['timestamp'] = pd.to_datetime(self.recent_predictions['timestamp'])
            
            latest_prediction = self.recent_predictions['timestamp'].max()
            oldest_prediction = self.recent_predictions['timestamp'].min()
            time_span = (latest_prediction - oldest_prediction).total_seconds() / 3600
            
            print(f"├─ Time Span: {time_span:.1f} hours")
            print(f"├─ Latest Prediction: {latest_prediction}")
            print(f"└─ Oldest Prediction: {oldest_prediction}")
        
        # CALL prediction accuracy
        if all(col in self.recent_predictions.columns for col in ['call_actual', 'call_predicted']):
            call_actual = self.recent_predictions['call_actual']
            call_predicted = self.recent_predictions['call_predicted']
            
            call_mae = mean_absolute_error(call_actual, call_predicted)
            call_mse = mean_squared_error(call_actual, call_predicted)
            call_r2 = r2_score(call_actual, call_predicted)
            
            print(f"\n📈 CALL PREDICTION ACCURACY:")
            print(f"├─ MAE: {call_mae:.6f}")
            print(f"├─ MSE: {call_mse:.8f}")
            print(f"├─ RMSE: {np.sqrt(call_mse):.6f}")
            print(f"├─ R²: {call_r2:.4f}")
            
            # Prediction errors
            call_errors = call_actual - call_predicted
            print(f"├─ Mean Error: {call_errors.mean():+.6f}")
            print(f"├─ Error Std: {call_errors.std():.6f}")
            print(f"└─ Error Range: {call_errors.min():+.6f} to {call_errors.max():+.6f}")
        
        # PUT prediction accuracy
        if all(col in self.recent_predictions.columns for col in ['put_actual', 'put_predicted']):
            put_actual = self.recent_predictions['put_actual']
            put_predicted = self.recent_predictions['put_predicted']
            
            put_mae = mean_absolute_error(put_actual, put_predicted)
            put_mse = mean_squared_error(put_actual, put_predicted)
            put_r2 = r2_score(put_actual, put_predicted)
            
            print(f"\n📉 PUT PREDICTION ACCURACY:")
            print(f"├─ MAE: {put_mae:.6f}")
            print(f"├─ MSE: {put_mse:.8f}")
            print(f"├─ RMSE: {np.sqrt(put_mse):.6f}")
            print(f"├─ R²: {put_r2:.4f}")
            
            # Prediction errors
            put_errors = put_actual - put_predicted
            print(f"├─ Mean Error: {put_errors.mean():+.6f}")
            print(f"├─ Error Std: {put_errors.std():.6f}")
            print(f"└─ Error Range: {put_errors.min():+.6f} to {put_errors.max():+.6f}")
    
    def analyze_model_trees(self):
        """Analyze individual trees in the Random Forest models"""
        if not self.models_loaded:
            return "❌ Models not loaded"
        
        print("\n" + "="*80)
        print("🌳 RANDOM FOREST TREE ANALYSIS")
        print("="*80)
        
        # CALL Model Trees
        call_trees = self.call_model.estimators_
        call_depths = [tree.tree_.max_depth for tree in call_trees]
        call_nodes = [tree.tree_.node_count for tree in call_trees]
        call_leaves = [tree.tree_.n_leaves for tree in call_trees]
        
        print(f"\n📈 CALL MODEL TREES:")
        print(f"├─ Number of Trees: {len(call_trees)}")
        print(f"├─ Average Depth: {np.mean(call_depths):.1f}")
        print(f"├─ Max Depth: {np.max(call_depths)}")
        print(f"├─ Min Depth: {np.min(call_depths)}")
        print(f"├─ Average Nodes: {np.mean(call_nodes):.1f}")
        print(f"└─ Average Leaves: {np.mean(call_leaves):.1f}")
        
        # PUT Model Trees
        put_trees = self.put_model.estimators_
        put_depths = [tree.tree_.max_depth for tree in put_trees]
        put_nodes = [tree.tree_.node_count for tree in put_trees]
        put_leaves = [tree.tree_.n_leaves for tree in put_trees]
        
        print(f"\n📉 PUT MODEL TREES:")
        print(f"├─ Number of Trees: {len(put_trees)}")
        print(f"├─ Average Depth: {np.mean(put_depths):.1f}")
        print(f"├─ Max Depth: {np.max(put_depths)}")
        print(f"├─ Min Depth: {np.min(put_depths)}")
        print(f"├─ Average Nodes: {np.mean(put_nodes):.1f}")
        print(f"└─ Average Leaves: {np.mean(put_leaves):.1f}")
        
        # Tree complexity analysis
        print(f"\n🔍 TREE COMPLEXITY ANALYSIS:")
        print(f"├─ CALL trees with max depth: {sum(1 for d in call_depths if d == self.call_model.max_depth)}")
        print(f"├─ PUT trees with max depth: {sum(1 for d in put_depths if d == self.put_model.max_depth)}")
        print(f"├─ Average complexity ratio (CALL): {np.mean(call_leaves) / np.mean(call_nodes):.3f}")
        print(f"└─ Average complexity ratio (PUT): {np.mean(put_leaves) / np.mean(put_nodes):.3f}")
    
    def _calculate_trend(self, values):
        """Calculate trend direction from a list of values"""
        if len(values) < 2:
            return "Insufficient data"
        
        recent_avg = np.mean(values[-3:]) if len(values) >= 3 else values[-1]
        older_avg = np.mean(values[:3]) if len(values) >= 6 else np.mean(values[:-3]) if len(values) > 3 else values[0]
        
        diff = recent_avg - older_avg
        if diff > 0.01:
            return "📈 Improving"
        elif diff < -0.01:
            return "📉 Declining"
        else:
            return "➡️ Stable"
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report"""
        if not self.models_loaded:
            print("❌ Cannot generate report - models not loaded")
            return
        
        print("🧠 ML ARBITRAGE MODEL COMPREHENSIVE ANALYSIS")
        print("="*80)
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model Directory: {self.model_dir}")
        
        # Run all analyses
        self.analyze_model_architecture()
        feature_importance = self.analyze_feature_importance()
        self.analyze_training_performance()
        self.analyze_training_data()
        self.analyze_recent_predictions()
        self.analyze_model_trees()
        
        # Summary
        print("\n" + "="*80)
        print("📋 ANALYSIS SUMMARY")
        print("="*80)
        
        # Model readiness assessment
        performance = self.metadata.get('model_performance', {})
        call_r2 = performance.get('call_r2', 0)
        put_r2 = performance.get('put_r2', 0)
        
        training_samples = self.metadata.get('training_samples', 0)
        success_rate = self.metadata.get('successful_trainings', 0) / max(self.metadata.get('training_attempts', 1), 1)
        
        print(f"\n🎯 MODEL READINESS ASSESSMENT:")
        
        readiness_score = 0
        factors = []
        
        # Training samples
        if training_samples >= 1000:
            readiness_score += 25
            factors.append("✅ Sufficient training data")
        elif training_samples >= 500:
            readiness_score += 15
            factors.append("⚠️ Moderate training data")
        else:
            factors.append("❌ Insufficient training data")
        
        # Model performance
        avg_r2 = (call_r2 + put_r2) / 2
        if avg_r2 >= 0.3:
            readiness_score += 30
            factors.append("✅ Good model performance")
        elif avg_r2 >= 0.1:
            readiness_score += 20
            factors.append("⚠️ Moderate model performance")
        else:
            factors.append("❌ Poor model performance")
        
        # Training success rate
        if success_rate >= 0.7:
            readiness_score += 25
            factors.append("✅ High training success rate")
        elif success_rate >= 0.4:
            readiness_score += 15
            factors.append("⚠️ Moderate training success rate")
        else:
            factors.append("❌ Low training success rate")
        
        # Recent predictions
        if self.recent_predictions is not None and len(self.recent_predictions) >= 50:
            readiness_score += 20
            factors.append("✅ Good prediction history")
        elif self.recent_predictions is not None and len(self.recent_predictions) >= 20:
            readiness_score += 10
            factors.append("⚠️ Limited prediction history")
        else:
            factors.append("❌ No prediction history")
        
        print(f"\nREADINESS SCORE: {readiness_score}/100")
        
        if readiness_score >= 80:
            status = "🟢 READY FOR LIVE TRADING"
        elif readiness_score >= 60:
            status = "🟡 READY FOR PAPER TRADING"
        elif readiness_score >= 40:
            status = "🟠 NEEDS MORE TRAINING"
        else:
            status = "🔴 NOT READY - MAJOR ISSUES"
        
        print(f"STATUS: {status}")
        print("\nFACTORS:")
        for factor in factors:
            print(f"  {factor}")
        
        print("\n" + "="*80)
        print("✅ COMPREHENSIVE ANALYSIS COMPLETE")
        print("="*80)

def main():
    """Main function to run the model analysis"""
    print("🔍 ML Model Analysis Tool")
    print("="*50)
    
    # Initialize analyzer
    analyzer = MLModelAnalyzer()
    
    if not analyzer.models_loaded:
        print("❌ Could not load models. Please ensure:")
        print("   1. Models have been trained and saved")
        print("   2. Model directory path is correct")
        print(f"   3. Files exist in: {analyzer.model_dir}")
        return
    
    # Generate comprehensive report
    analyzer.generate_comprehensive_report()
    
    print(f"\n💡 TIP: You can also run individual analyses:")
    print(f"   analyzer.analyze_model_architecture()")
    print(f"   analyzer.analyze_feature_importance()")
    print(f"   analyzer.analyze_training_performance()")
    print(f"   analyzer.analyze_training_data()")
    print(f"   analyzer.analyze_recent_predictions()")
    print(f"   analyzer.analyze_model_trees()")

if __name__ == "__main__":
    main()
