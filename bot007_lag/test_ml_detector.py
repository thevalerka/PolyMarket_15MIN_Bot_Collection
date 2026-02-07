#!/usr/bin/env python3
"""
Simple test script to verify ML Detector setup and find indentation issues
"""

import os
import json
from datetime import datetime

def test_directory_creation():
    """Test if we can create the model directory"""
    model_dir = '/home/ubuntu/013_2025_polymarket/ml_models'
    
    print("🧪 Testing directory creation...")
    print(f"📁 Target directory: {model_dir}")
    
    try:
        # Create directory
        os.makedirs(model_dir, exist_ok=True)
        print(f"✅ Directory created/exists: {os.path.exists(model_dir)}")
        
        # Test write permissions
        test_file = os.path.join(model_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write(f"Test file created at {datetime.now()}")
        
        print(f"✅ Write permissions: OK")
        
        # Clean up
        os.remove(test_file)
        print(f"✅ Cleanup: OK")
        
        # List directory contents
        contents = os.listdir(model_dir)
        print(f"📄 Directory contents: {contents}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_json_files():
    """Test if we can read the JSON files"""
    files = {
        'BTC': '/home/ubuntu/013_2025_polymarket/btc_price.json',
        'CALL': '/home/ubuntu/013_2025_polymarket/CALL.json', 
        'PUT': '/home/ubuntu/013_2025_polymarket/PUT.json'
    }
    
    print("\n🧪 Testing JSON file access...")
    
    for name, filepath in files.items():
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                print(f"✅ {name}: File exists and readable")
                
                # Show relevant fields
                if name == 'BTC':
                    print(f"   Price: ${data.get('price', 'N/A')}")
                else:
                    bid = data.get('best_bid', {}).get('price', 'N/A')
                    ask = data.get('best_ask', {}).get('price', 'N/A')
                    print(f"   {name}: {bid}/{ask}")
            else:
                print(f"❌ {name}: File not found at {filepath}")
                
        except Exception as e:
            print(f"❌ {name}: Error reading file - {e}")

def test_imports():
    """Test if all required packages can be imported"""
    print("\n🧪 Testing package imports...")
    
    packages = [
        'numpy', 'pandas', 'requests', 'json', 'sklearn', 'joblib'
    ]
    
    for package in packages:
        try:
            if package == 'sklearn':
                from sklearn.ensemble import RandomForestRegressor
                print(f"✅ sklearn: RandomForestRegressor imported")
            else:
                __import__(package)
                print(f"✅ {package}: OK")
        except ImportError as e:
            print(f"❌ {package}: Missing - {e}")
            print(f"   Install with: pip install {package}")

def find_indentation_error():
    """Try to identify the specific indentation error"""
    print("\n🔍 Trying to import ML detector to find the error...")
    
    try:
        # This will show us the exact line with the error
        from ml_arbitrage_detector import MLArbitrageDetector
        print("✅ Import successful!")
        
        # Try to create detector
        detector = MLArbitrageDetector()
        print("✅ Detector creation successful!")
        
        # Test basic methods
        print(f"📁 Model directory: {detector.model_dir}")
        print(f"📊 Models trained: {detector.models_trained}")
        
        return True
        
    except IndentationError as e:
        print(f"❌ Indentation Error: {e}")
        print("🔧 Fix needed in the source code")
        return False
    except SyntaxError as e:
        print(f"❌ Syntax Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Other Error: {e}")
        return False

def main():
    print("🧠 ML ARBITRAGE DETECTOR - SETUP TEST")
    print("=" * 50)
    
    # Run tests
    dir_ok = test_directory_creation()
    test_json_files() 
    test_imports()
    import_ok = find_indentation_error()
    
    print("\n" + "=" * 50)
    print("📋 SUMMARY:")
    print(f"├─ Directory setup: {'✅' if dir_ok else '❌'}")
    print(f"└─ Import/syntax: {'✅' if import_ok else '❌'}")
    
    if dir_ok and import_ok:
        print("🎉 All tests passed! Ready to run ML detector")
    else:
        print("⚠️  Fix the issues above before running the main detector")

if __name__ == "__main__":
    main()