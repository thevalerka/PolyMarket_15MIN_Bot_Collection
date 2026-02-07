#!/usr/bin/env python3
# test_bot_import.py - Test all collar bot imports and dependencies

import sys
import traceback

def test_basic_imports():
    """Test basic Python imports."""
    print("🔍 Testing basic imports...")
    
    try:
        import json
        import time
        import math
        from datetime import datetime, timedelta
        from typing import Dict, List, Tuple, Optional
        print("✅ Basic Python modules: OK")
        return True
    except Exception as e:
        print(f"❌ Basic Python modules: {e}")
        return False

def test_numpy():
    """Test numpy import."""
    print("\n📊 Testing Numpy...")
    
    try:
        import numpy as np
        print(f"✅ Numpy {np.__version__}: Available")
        
        # Test basic operations
        arr = np.array([1, 2, 3, 4, 5])
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        print(f"   Basic operations: mean={mean_val}, std={std_val:.2f}")
        return True
    except ImportError:
        print("⚠️ Numpy: Not available (using fallback math)")
        return False
    except Exception as e:
        print(f"❌ Numpy error: {e}")
        return False

def test_sklearn():
    """Test scikit-learn import."""
    print("\n🧠 Testing Scikit-learn...")
    
    try:
        import sklearn
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        print(f"✅ Scikit-learn {sklearn.__version__}: Available")
        
        # Test basic model creation
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('regressor', LinearRegression())
        ])
        print("   Model creation: OK")
        return True
    except ImportError:
        print("⚠️ Scikit-learn: Not available (using simple calculations)")
        return False
    except Exception as e:
        print(f"❌ Scikit-learn error: {e}")
        return False

def test_collar_config():
    """Test collar config import."""
    print("\n⚙️ Testing collar_config...")
    
    try:
        from collar_config import (
            state, CLOB_API_URL, TOKEN_ID, MAX_TRADE_AMOUNT,
            MIN_QUANTITY_FILTER, MANDATORY_SELL_PRICE,
            MIN_SPREAD, MAX_SPREAD, SPREAD_BUFFER
        )
        print("✅ collar_config: All imports OK")
        print(f"   TOKEN_ID: {TOKEN_ID[:20]}...")
        print(f"   MAX_TRADE_AMOUNT: ${MAX_TRADE_AMOUNT}")
        print(f"   SPREAD_RANGE: ${MIN_SPREAD:.2f} - ${MAX_SPREAD:.2f}")
        return True
    except Exception as e:
        print(f"❌ collar_config error: {e}")
        traceback.print_exc()
        return False

def test_collar_strategy():
    """Test collar strategy import."""
    print("\n📈 Testing collar_strategy...")
    
    try:
        from collar_strategy import MarketAnalyzer, display_analysis
        print("✅ collar_strategy: MarketAnalyzer import OK")
        
        # Test basic initialization
        analyzer = MarketAnalyzer("test_book.json", "test_btc.json")
        print("   MarketAnalyzer initialization: OK")
        return True
    except Exception as e:
        print(f"❌ collar_strategy error: {e}")
        traceback.print_exc()
        return False

def test_collar_trading():
    """Test collar trading import."""
    print("\n💰 Testing collar_trading...")
    
    try:
        from collar_trading import TradingExecutor, TradingLogic, display_trading_status
        print("✅ collar_trading: All imports OK")
        return True
    except Exception as e:
        print(f"❌ collar_trading error: {e}")
        traceback.print_exc()
        return False

def test_advanced_libraries():
    """Test advanced ML libraries."""
    print("\n🚀 Testing advanced libraries...")
    
    libraries = [
        ("XGBoost", "xgboost", "xgb"),
        ("LightGBM", "lightgbm", "lgb"),
        ("Pandas", "pandas", "pd")
    ]
    
    available = []
    for name, module, alias in libraries:
        try:
            lib = __import__(module)
            version = getattr(lib, '__version__', 'unknown')
            print(f"✅ {name} {version}: Available")
            available.append(name)
        except ImportError:
            print(f"⚪ {name}: Not installed (optional)")
    
    return available

def main():
    """Run all tests."""
    print("🧪 COLLAR BOT DEPENDENCY TEST")
    print("=" * 50)
    
    results = {}
    
    # Test all components
    results['basic'] = test_basic_imports()
    results['numpy'] = test_numpy()
    results['sklearn'] = test_sklearn()
    results['config'] = test_collar_config()
    results['strategy'] = test_collar_strategy()
    results['trading'] = test_collar_trading()
    
    # Test advanced libraries
    advanced = test_advanced_libraries()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for component, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {component.capitalize()}")
    
    if advanced:
        print(f"🚀 Advanced libraries: {', '.join(advanced)}")
    
    print(f"\n📊 Overall: {passed}/{total} core components working")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Your collar bot is ready to run!")
        print("\n🚀 Start command: python3 run_collar_bot.py")
    else:
        print(f"\n⚠️ {total - passed} component(s) failed")
        
        if not results['config'] or not results['strategy'] or not results['trading']:
            print("❌ Critical components failed - bot cannot start")
            print("\n🔧 Try these fixes:")
            print("   1. Check file paths and permissions")
            print("   2. Ensure all .py files are in the same directory")
            print("   3. Run: chmod +x install_ml_deps.sh && ./install_ml_deps.sh")
        else:
            print("✅ Core components OK - bot will use fallback calculations")
            print("💡 Install ML dependencies for enhanced features:")
            print("   pip3 install numpy scikit-learn")

if __name__ == "__main__":
    main()