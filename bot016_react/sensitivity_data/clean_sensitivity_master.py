#!/usr/bin/env python3
"""
Clean Sensitivity Master - Remove Zero Values
Removes all 0.0 and -0.0 values from put_sensitivity_raw and call_sensitivity_raw arrays
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def clean_sensitivity_file(input_file: str, output_file: str = None, backup: bool = True):
    """
    Clean sensitivity_master.json by removing zero values
    
    Args:
        input_file: Path to sensitivity_master.json
        output_file: Path to output file (default: overwrites input)
        backup: Whether to create backup before cleaning
    """
    
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"❌ Error: File not found: {input_file}")
        return False
    
    # Create backup
    if backup:
        backup_path = input_path.parent / f"{input_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        print(f"📦 Creating backup: {backup_path}")
        with open(input_path, 'r') as f:
            backup_data = f.read()
        with open(backup_path, 'w') as f:
            f.write(backup_data)
        print(f"✅ Backup created")
    
    # Load data
    print(f"\n📂 Loading: {input_file}")
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    total_bins = len(data.get('bins', {}))
    bins_cleaned = 0
    bins_removed = 0
    total_zeros_removed = 0
    
    print(f"📊 Found {total_bins} bins")
    print(f"\n🧹 Cleaning zeros...")
    
    cleaned_bins = {}
    
    for bin_key, bin_data in data.get('bins', {}).items():
        # Get raw arrays
        put_raw = bin_data.get('put_sensitivity_raw', [])
        call_raw = bin_data.get('call_sensitivity_raw', [])
        
        # Count zeros before
        put_zeros = sum(1 for x in put_raw if abs(x) < 1e-10)
        call_zeros = sum(1 for x in call_raw if abs(x) < 1e-10)
        
        # Filter out zeros
        put_raw_clean = [x for x in put_raw if abs(x) > 1e-10]
        call_raw_clean = [x for x in call_raw if abs(x) > 1e-10]
        
        # Skip bin if no data left
        if not put_raw_clean or not call_raw_clean:
            bins_removed += 1
            continue
        
        # Recalculate statistics if data changed
        if put_zeros > 0 or call_zeros > 0:
            bins_cleaned += 1
            total_zeros_removed += put_zeros + call_zeros
            
            # Recalculate stats
            import numpy as np
            
            bin_data['count'] = len(put_raw_clean)
            
            bin_data['put_sensitivity'] = {
                'avg': float(np.mean(put_raw_clean)),
                'median': float(np.median(put_raw_clean)),
                'std': float(np.std(put_raw_clean)),
                'min': float(np.min(put_raw_clean)),
                'max': float(np.max(put_raw_clean)),
                'p25': float(np.percentile(put_raw_clean, 25)),
                'p75': float(np.percentile(put_raw_clean, 75))
            }
            
            bin_data['call_sensitivity'] = {
                'avg': float(np.mean(call_raw_clean)),
                'median': float(np.median(call_raw_clean)),
                'std': float(np.std(call_raw_clean)),
                'min': float(np.min(call_raw_clean)),
                'max': float(np.max(call_raw_clean)),
                'p25': float(np.percentile(call_raw_clean, 25)),
                'p75': float(np.percentile(call_raw_clean, 75))
            }
            
            bin_data['put_sensitivity_raw'] = put_raw_clean
            bin_data['call_sensitivity_raw'] = call_raw_clean
        
        cleaned_bins[bin_key] = bin_data
    
    # Update data
    data['bins'] = cleaned_bins
    data['total_bins'] = len(cleaned_bins)
    data['last_cleaned'] = datetime.now().isoformat()
    
    # Save
    output_path = Path(output_file) if output_file else input_path
    print(f"\n💾 Saving to: {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ CLEANING COMPLETE")
    print(f"{'='*60}")
    print(f"Total bins (before):     {total_bins}")
    print(f"Bins cleaned:            {bins_cleaned}")
    print(f"Bins removed (empty):    {bins_removed}")
    print(f"Total bins (after):      {len(cleaned_bins)}")
    print(f"Total zeros removed:     {total_zeros_removed:,}")
    print(f"{'='*60}\n")
    
    return True


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python clean_sensitivity_master.py <sensitivity_master.json> [output_file]")
        print("\nExamples:")
        print("  # Clean in place (creates backup):")
        print("  python clean_sensitivity_master.py sensitivity_master.json")
        print("\n  # Clean to new file:")
        print("  python clean_sensitivity_master.py sensitivity_master.json sensitivity_master_clean.json")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("\n" + "="*60)
    print("🧹 SENSITIVITY MASTER CLEANER")
    print("="*60)
    print("Removes all zero values (0.0, -0.0) from sensitivity data")
    print("="*60 + "\n")
    
    success = clean_sensitivity_file(input_file, output_file)
    
    if success:
        print("✅ Done!")
    else:
        print("❌ Failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
