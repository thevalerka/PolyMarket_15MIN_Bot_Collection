#!/usr/bin/env python3
# quick_fix.py - Quick fix for import issues
"""
Quick Fix for Spike Bot Import Issues

This script fixes common import problems when migrating from collar bot to spike bot.
"""

import os
import sys

def fix_imports():
    """Fix import statements in all files."""
    print("🔧 Fixing import statements...")
    
    # Files to check and fix
    files_to_fix = [
        'spike_strategy.py',
        'spike_trading.py', 
        'spike_trading_bot.py',
        'spike_logger.py',
        'spike_monitor.py',
        'spike_testing.py',
        'spike_bot_manager.py'
    ]
    
    # Import replacements
    replacements = [
        ('from collar_config import', 'from spike_config import'),
        ('import collar_config', 'import spike_config'),
        ('collar_config.', 'spike_config.'),
    ]
    
    fixed_files = []
    
    for filename in files_to_fix:
        if not os.path.exists(filename):
            print(f"   ⚠️ File not found: {filename}")
            continue
        
        try:
            # Read file
            with open(filename, 'r') as f:
                content = f.read()
            
            # Check if fixes needed
            needs_fix = any(old_text in content for old_text, new_text in replacements)
            
            if needs_fix:
                # Apply fixes
                original_content = content
                for old_text, new_text in replacements:
                    content = content.replace(old_text, new_text)
                
                # Write back only if changed
                if content != original_content:
                    with open(filename, 'w') as f:
                        f.write(content)
                    
                    fixed_files.append(filename)
                    print(f"   ✅ Fixed imports in: {filename}")
            else:
                print(f"   ✅ Already correct: {filename}")
                
        except Exception as e:
            print(f"   ❌ Error fixing {filename}: {e}")
    
    if fixed_files:
        print(f"\n🎉 Fixed imports in {len(fixed_files)} files!")
    else:
        print(f"\n✅ All import statements are already correct!")
    
    return fixed_files

def check_old_files():
    """Check for old collar files that might cause conflicts."""
    print("\n🔍 Checking for old collar bot files...")
    
    old_files = [
        'collar_config.py',
        'collar_strategy.py',
        'collar_trading.py', 
        'run_collar_bot.py'
    ]
    
    found_old_files = []
    for old_file in old_files:
        if os.path.exists(old_file):
            found_old_files.append(old_file)
            print(f"   ⚠️ Found old file: {old_file}")
    
    if found_old_files:
        print(f"\n🗑️ Recommendation: Backup and remove old files to avoid conflicts")
        
        response = input("   Remove old collar files now? (y/N): ").strip().lower()
        if response == 'y':
            for old_file in found_old_files:
                try:
                    # Backup first
                    backup_name = f"{old_file}.backup"
                    os.rename(old_file, backup_name)
                    print(f"   ✅ Moved {old_file} to {backup_name}")
                except Exception as e:
                    print(f"   ❌ Error moving {old_file}: {e}")
    else:
        print("   ✅ No old collar files found")

def test_imports():
    """Test if imports work correctly now."""
    print("\n🧪 Testing imports...")
    
    test_modules = [
        'spike_config',
        'spike_strategy', 
        'spike_trading',
        'spike_bot_manager'
    ]
    
    all_good = True
    
    for module in test_modules:
        try:
            __import__(module)
            print(f"   ✅ {module} imports successfully")
        except ImportError as e:
            print(f"   ❌ {module} import failed: {e}")
            all_good = False
        except Exception as e:
            print(f"   ⚠️ {module} other error: {e}")
    
    if all_good:
        print(f"\n🎉 All imports working correctly!")
        print(f"\n🚀 You can now run: python3 spike_bot_manager.py test quick")
    else:
        print(f"\n❌ Some imports still have issues")
        print(f"   Please check that all new spike bot files are present")
    
    return all_good

def main():
    """Main quick fix function."""
    print("🛠️ Quick Fix for Spike Bot Import Issues")
    print("=" * 50)
    
    # Fix import statements
    fixed_files = fix_imports()
    
    # Check for old files
    check_old_files()
    
    # Test imports
    success = test_imports()
    
    if success:
        print(f"\n✅ Quick fix completed successfully!")
        print(f"\nNext steps:")
        print(f"   1. Run: python3 spike_bot_manager.py test quick")
        print(f"   2. If tests pass: python3 spike_bot_manager.py start analysis")
    else:
        print(f"\n❌ Quick fix completed but issues remain")
        print(f"   Please ensure all spike bot files are in the current directory")

if __name__ == "__main__":
    main()
