#!/usr/bin/env python3
# analyze_comprehensive_export.py - Analyze the comprehensive real-time export

import json
import os
from datetime import datetime
from typing import Dict, Any

def load_comprehensive_analysis(file_path: str = "comprehensive_analysis.json") -> Dict:
    """Load the latest comprehensive analysis."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        print("💡 Make sure the bot is running and has exported at least one analysis")
        return {}
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {file_path}: {e}")
        return {}

def display_summary(data: Dict):
    """Display a summary of the comprehensive analysis."""
    if not data:
        return
    
    print("📊 COMPREHENSIVE ANALYSIS SUMMARY")
    print("=" * 60)
    
    # Basic info
    export_time = data.get('export_timestamp', 'Unknown')
    print(f"📅 Export Time: {export_time}")
    print(f"🕐 Hour: {data.get('hour', 'Unknown')}")
    print(f"⏰ Uptime: {data.get('uptime_minutes', 0):.1f} minutes")
    
    # Model performance
    performance = data.get('model_performance', {})
    print(f"\n🎯 Model Performance:")
    print(f"   Accuracy: {performance.get('formula_accuracy', 0):.1%}")
    print(f"   Training Samples: {performance.get('training_samples', 0)}")
    print(f"   Last Calibration: {performance.get('last_calibration_ago_minutes', 0):.1f}min ago")
    print(f"   Model Type: {performance.get('model_type', 'Unknown')}")
    
    # Market regime
    regime = data.get('market_regime', {})
    print(f"\n🌊 Market Regime:")
    print(f"   Current: {regime.get('current_regime', 'Unknown')}")
    print(f"   Confidence: {regime.get('confidence', 0):.1%}")
    print(f"   Minutes to Expiry: {regime.get('minutes_to_expiry', 0):.1f}")
    
    # Risk assessment
    risk = data.get('risk_indicators', {})
    print(f"\n⚠️ Risk Assessment:")
    print(f"   Level: {risk.get('overall_risk_level', 'Unknown').upper()}")
    print(f"   Score: {risk.get('risk_score', 0):.2f}")
    if risk.get('risk_factors'):
        print(f"   Factors: {', '.join(risk['risk_factors'])}")
    
    # Current market
    market = data.get('current_market', {})
    if market.get('status') == 'active':
        print(f"\n💹 Current Market:")
        print(f"   BTC Price: ${market.get('btc_price', 0):,.2f}")
        print(f"   Market Mid: ${market.get('market_mid', 0):.3f}")
        print(f"   Market Spread: ${market.get('market_spread', 0):.3f}")
        print(f"   Data Quality: {market.get('data_quality', 'Unknown')}")
    
    # Predictions
    predictions = data.get('prediction_insights', {})
    if predictions.get('status') == 'active':
        print(f"\n🔮 Current Predictions:")
        print(f"   Predicted Spread: ${predictions.get('predicted_spread', 0):.3f}")
        print(f"   Target Spread: ${predictions.get('target_spread', 0):.3f}")
        print(f"   Assessment: {predictions.get('spread_assessment', 'Unknown')}")
        
        targets = predictions.get('target_prices', {})
        if targets:
            print(f"   Target Prices: ${targets.get('bid', 0):.3f} x ${targets.get('ask', 0):.3f}")

def display_detailed_analysis(data: Dict):
    """Display detailed analysis sections."""
    if not data:
        return
    
    print("\n" + "=" * 60)
    print("📈 DETAILED ANALYSIS")
    print("=" * 60)
    
    # Feature analysis
    features = data.get('feature_analysis', {})
    if features.get('status') == 'analyzed':
        print("\n🔍 Feature Analysis:")
        feature_data = features.get('features', {})
        
        most_active = features.get('most_active_feature')
        least_active = features.get('least_active_feature')
        
        if most_active and least_active:
            print(f"   Most Active: {most_active}")
            print(f"   Least Active: {least_active}")
        
        for feature_name, stats in feature_data.items():
            print(f"   {feature_name}:")
            print(f"     Latest: {stats.get('latest', 0):.4f}")
            print(f"     Range: {stats.get('range', 0):.4f}")
            print(f"     Activity: {stats.get('activity_level', 'Unknown')}")
    
    # Performance trends
    trends = data.get('performance_trends', {})
    if trends.get('status') == 'calculated':
        print(f"\n📊 Performance Trends:")
        print(f"   Direction: {trends.get('trend_direction', 'Unknown')}")
        print(f"   Magnitude: {trends.get('trend_magnitude', 0):+.3f}")
        print(f"   Total Samples: {trends.get('total_samples', 0)}")
        
        periods = trends.get('periods', [])
        if periods:
            print("   Recent Periods:")
            for i, period in enumerate(periods[-3:]):  # Show last 3 periods
                print(f"     Period {i+1}: {period.get('average_accuracy', 0):.1%} accuracy ({period.get('sample_count', 0)} samples)")
    
    # Model health
    health = data.get('model_health', {})
    print(f"\n🏥 Model Health:")
    print(f"   Status: {health.get('health_status', 'Unknown').upper()}")
    print(f"   Score: {health.get('health_score', 0):.2f}")
    
    health_issues = health.get('health_issues', [])
    if health_issues:
        print(f"   Issues: {', '.join(health_issues)}")
    
    health_recs = health.get('recommendations', [])
    if health_recs:
        print("   Health Recommendations:")
        for rec in health_recs:
            print(f"     • {rec}")
    
    # Manipulation analysis
    manipulation = data.get('manipulation_analysis', {})
    if manipulation.get('status') == 'monitored':
        print(f"\n🚨 Manipulation Analysis:")
        print(f"   Recent Alerts: {manipulation.get('recent_alerts_count', 0)}")
        print(f"   Currently Manipulated: {manipulation.get('currently_manipulated', False)}")
        print(f"   Total Alerts Today: {manipulation.get('total_alerts_today', 0)}")

def display_recommendations(data: Dict):
    """Display actionable recommendations."""
    recommendations = data.get('recommendations', [])
    
    if recommendations:
        print("\n" + "=" * 60)
        print("💡 RECOMMENDATIONS")
        print("=" * 60)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("\n✅ No specific recommendations - system operating normally")

def monitor_file(file_path: str = "comprehensive_analysis.json"):
    """Monitor the file for changes and display updates."""
    print(f"👀 Monitoring {file_path} for updates...")
    print("Press Ctrl+C to stop")
    
    last_modified = 0
    
    try:
        while True:
            if os.path.exists(file_path):
                current_modified = os.path.getmtime(file_path)
                
                if current_modified > last_modified:
                    last_modified = current_modified
                    print(f"\n🔄 File updated at {datetime.fromtimestamp(current_modified)}")
                    
                    data = load_comprehensive_analysis(file_path)
                    if data:
                        display_summary(data)
                        print("\n" + "-" * 60)
                        print("Waiting for next update...")
            
            import time
            time.sleep(10)  # Check every 10 seconds
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")

def main():
    """Main function with menu options."""
    file_path = "comprehensive_analysis.json"
    
    print("📊 COMPREHENSIVE ANALYSIS READER")
    print("=" * 50)
    print("1. Show current analysis summary")
    print("2. Show detailed analysis") 
    print("3. Show recommendations")
    print("4. Monitor file for real-time updates")
    print("5. Show everything")
    
    try:
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice in ['1', '2', '3', '5']:
            data = load_comprehensive_analysis(file_path)
            
            if not data:
                return
            
            if choice in ['1', '5']:
                display_summary(data)
            
            if choice in ['2', '5']:
                display_detailed_analysis(data)
            
            if choice in ['3', '5']:
                display_recommendations(data)
                
        elif choice == '4':
            monitor_file(file_path)
            
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()