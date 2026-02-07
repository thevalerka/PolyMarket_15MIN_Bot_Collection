import json
import glob
from datetime import datetime
from collections import defaultdict
import os
import gc

# Configuration
data_folder = "/home/ubuntu/013_2025_polymarket/bot004_blackScholes/daily"
file_pattern = "option_probabilities_M15_*.json"

# Reversal thresholds to track
REVERSAL_THRESHOLDS = [0.01, 0.02, 0.05, 0.10,0.20,0.30]

# Volatility bands
VOLATILITY_BANDS = [
    (0.0, 0.1, "extremely_low"),
    (0.1, 0.22, "low"),
    (0.22, 0.36, "moderate"),
    (0.36, 0.54, "sustained"),
    (0.54, 0.80, "high"),
    (0.80, float('inf'), "extreme")
]

def get_volatility_band(volatility):
    """Classify volatility into bands"""
    for min_vol, max_vol, band_name in VOLATILITY_BANDS:
        if min_vol <= volatility < max_vol:
            return band_name
    return "extreme"

# Get all file paths
file_paths = glob.glob(os.path.join(data_folder, file_pattern))
file_paths.sort()

print(f"Found {len(file_paths)} files to process")

# Statistics tracking
reversal_stats = {
    'overall': {
        threshold: {
            'total_periods_hit': 0,
            'reversals': 0,
            'no_reversals': 0,
            'by_minute': {minute: {'hit': 0, 'reversed': 0} for minute in range(15)},
            'by_volatility': {band_name: {'hit': 0, 'reversed': 0} for _, _, band_name in VOLATILITY_BANDS}
        }
        for threshold in REVERSAL_THRESHOLDS
    }
}

total_periods_analyzed = 0
total_records_processed = 0
date_range_start = None
date_range_end = None
previous_last_period = None

def get_period_key(dt_str):
    """Get the period key for a datetime string (rounds down to :00, :15, :30, :45)"""
    dt = datetime.fromisoformat(dt_str)
    minute = (dt.minute // 15) * 15
    period_dt = dt.replace(minute=minute, second=0, microsecond=0)
    return period_dt.isoformat()

def get_period_start_datetime(period_key):
    """Convert period key back to datetime object"""
    return datetime.fromisoformat(period_key)

def get_minute_bucket(record_dt, period_start_dt):
    """Get the minute bucket (0-14) for a record within its period"""
    time_diff = (record_dt - period_start_dt).total_seconds()
    minute_bucket = int(time_diff // 60)
    return max(0, min(14, minute_bucket))

def process_file(file_path, previous_last_period):
    """Process a single file and return aggregated statistics"""
    print(f"Processing: {os.path.basename(file_path)}")

    file_records = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                file_records.append(record)
            except json.JSONDecodeError as e:
                print(f"  Error parsing line: {e}")
                continue

    print(f"  Loaded {len(file_records)} records")

    file_records.sort(key=lambda x: x['datetime'])

    periods = defaultdict(list)
    for record in file_records:
        period_key = get_period_key(record['datetime'])
        periods[period_key].append(record)

    sorted_periods = sorted(periods.keys())
    period_outcomes = []

    for i, period_key in enumerate(sorted_periods):
        period_records = periods[period_key]
        period_records.sort(key=lambda x: x['datetime'])

        last_record = period_records[-1]
        period_start_dt = get_period_start_datetime(period_key)

        # Track which thresholds were hit and when
        thresholds_hit = {
            threshold: {
                'hit': False,
                'first_minute': None,
                'volatility_at_hit': None
            }
            for threshold in REVERSAL_THRESHOLDS
        }

        for record in period_records:
            call_prob = record['call_probability']
            record_dt = datetime.fromisoformat(record['datetime'])
            minute_bucket = get_minute_bucket(record_dt, period_start_dt)
            volatility = record.get('volatility', 0)

            # Check each threshold
            for threshold in REVERSAL_THRESHOLDS:
                if call_prob <= threshold and not thresholds_hit[threshold]['hit']:
                    thresholds_hit[threshold]['hit'] = True
                    thresholds_hit[threshold]['first_minute'] = minute_bucket
                    thresholds_hit[threshold]['volatility_at_hit'] = volatility

        # Determine outcome
        call_won = None
        if i < len(sorted_periods) - 1:
            next_period_key = sorted_periods[i + 1]
            next_period_records = periods[next_period_key]
            next_period_records.sort(key=lambda x: x['datetime'])
            first_next = next_period_records[0]

            settlement_price = first_next['period_start_price']
            strike_price = last_record['strike_price']
            call_won = settlement_price > strike_price

        period_outcomes.append({
            'period': period_key,
            'thresholds_hit': thresholds_hit,
            'call_won': call_won,
            'last_record': last_record
        })

    # Resolve previous file's last period
    if previous_last_period and sorted_periods:
        first_period_records = periods[sorted_periods[0]]
        first_period_records.sort(key=lambda x: x['datetime'])
        first_next = first_period_records[0]

        settlement_price = first_next['period_start_price']
        strike_price = previous_last_period['last_record']['strike_price']
        previous_last_period['call_won'] = settlement_price > strike_price

        period_outcomes.insert(0, previous_last_period)

    # Calculate reversal statistics for this file
    file_reversal_stats = {
        threshold: {
            'total_periods_hit': 0,
            'reversals': 0,
            'no_reversals': 0,
            'by_minute': {minute: {'hit': 0, 'reversed': 0} for minute in range(15)},
            'by_volatility': {band_name: {'hit': 0, 'reversed': 0} for _, _, band_name in VOLATILITY_BANDS}
        }
        for threshold in REVERSAL_THRESHOLDS
    }

    periods_with_outcomes = 0
    for outcome in period_outcomes:
        if outcome['call_won'] is not None:
            periods_with_outcomes += 1

            # Check each threshold
            for threshold in REVERSAL_THRESHOLDS:
                threshold_info = outcome['thresholds_hit'][threshold]

                if threshold_info['hit']:
                    file_reversal_stats[threshold]['total_periods_hit'] += 1

                    minute_hit = threshold_info['first_minute']
                    volatility_at_hit = threshold_info['volatility_at_hit']
                    vol_band = get_volatility_band(volatility_at_hit)

                    # Record the hit
                    file_reversal_stats[threshold]['by_minute'][minute_hit]['hit'] += 1
                    file_reversal_stats[threshold]['by_volatility'][vol_band]['hit'] += 1

                    # Check if it reversed (won despite hitting low threshold)
                    if outcome['call_won']:
                        file_reversal_stats[threshold]['reversals'] += 1
                        file_reversal_stats[threshold]['by_minute'][minute_hit]['reversed'] += 1
                        file_reversal_stats[threshold]['by_volatility'][vol_band]['reversed'] += 1
                    else:
                        file_reversal_stats[threshold]['no_reversals'] += 1

    print(f"  Analyzed {periods_with_outcomes} periods with outcomes")

    last_period_for_next = period_outcomes[-1] if period_outcomes and period_outcomes[-1]['call_won'] is None else None

    return {
        'file_reversal_stats': file_reversal_stats,
        'last_period': last_period_for_next,
        'num_records': len(file_records),
        'num_periods': periods_with_outcomes,
        'date_range': (sorted_periods[0] if sorted_periods else None,
                      sorted_periods[-1] if sorted_periods else None)
    }

# Process each file
for file_path in file_paths:
    try:
        result = process_file(file_path, previous_last_period)

        # Aggregate statistics
        for threshold in REVERSAL_THRESHOLDS:
            reversal_stats['overall'][threshold]['total_periods_hit'] += \
                result['file_reversal_stats'][threshold]['total_periods_hit']
            reversal_stats['overall'][threshold]['reversals'] += \
                result['file_reversal_stats'][threshold]['reversals']
            reversal_stats['overall'][threshold]['no_reversals'] += \
                result['file_reversal_stats'][threshold]['no_reversals']

            for minute in range(15):
                reversal_stats['overall'][threshold]['by_minute'][minute]['hit'] += \
                    result['file_reversal_stats'][threshold]['by_minute'][minute]['hit']
                reversal_stats['overall'][threshold]['by_minute'][minute]['reversed'] += \
                    result['file_reversal_stats'][threshold]['by_minute'][minute]['reversed']

            for _, _, band_name in VOLATILITY_BANDS:
                reversal_stats['overall'][threshold]['by_volatility'][band_name]['hit'] += \
                    result['file_reversal_stats'][threshold]['by_volatility'][band_name]['hit']
                reversal_stats['overall'][threshold]['by_volatility'][band_name]['reversed'] += \
                    result['file_reversal_stats'][threshold]['by_volatility'][band_name]['reversed']

        total_records_processed += result['num_records']
        total_periods_analyzed += result['num_periods']

        if result['date_range'][0]:
            if date_range_start is None:
                date_range_start = result['date_range'][0]
            date_range_end = result['date_range'][1]

        previous_last_period = result['last_period']

        gc.collect()

    except Exception as e:
        print(f"  ERROR processing file: {e}")
        continue

print(f"\nTotal records processed: {total_records_processed}")
print(f"Total periods analyzed: {total_periods_analyzed}")

# Calculate reversal rates
for threshold in REVERSAL_THRESHOLDS:
    stats = reversal_stats['overall'][threshold]
    if stats['total_periods_hit'] > 0:
        stats['reversal_rate'] = stats['reversals'] / stats['total_periods_hit']
    else:
        stats['reversal_rate'] = 0

    # Calculate rates by minute
    for minute in range(15):
        minute_data = stats['by_minute'][minute]
        if minute_data['hit'] > 0:
            minute_data['reversal_rate'] = minute_data['reversed'] / minute_data['hit']
        else:
            minute_data['reversal_rate'] = 0

    # Calculate rates by volatility
    for _, _, band_name in VOLATILITY_BANDS:
        vol_data = stats['by_volatility'][band_name]
        if vol_data['hit'] > 0:
            vol_data['reversal_rate'] = vol_data['reversed'] / vol_data['hit']
        else:
            vol_data['reversal_rate'] = 0

# Prepare JSON output
json_output = {
    'summary': {
        'total_periods_analyzed': total_periods_analyzed,
        'total_records_processed': total_records_processed,
        'total_files_processed': len(file_paths),
        'date_range_start': date_range_start,
        'date_range_end': date_range_end,
        'thresholds_analyzed': [f"{t:.0%}" for t in REVERSAL_THRESHOLDS]
    },
    'reversal_analysis': {}
}

for threshold in REVERSAL_THRESHOLDS:
    threshold_key = f"{threshold:.0%}"
    stats = reversal_stats['overall'][threshold]

    json_output['reversal_analysis'][threshold_key] = {
        'threshold_value': threshold,
        'total_periods_hit': stats['total_periods_hit'],
        'reversals': stats['reversals'],
        'no_reversals': stats['no_reversals'],
        'reversal_rate': round(stats['reversal_rate'], 4),
        'by_minute': {},
        'by_volatility': {}
    }

    # Add minute breakdown
    for minute in range(15):
        minute_data = stats['by_minute'][minute]
        if minute_data['hit'] > 0:
            json_output['reversal_analysis'][threshold_key]['by_minute'][f'minute_{minute}'] = {
                'periods_hit': minute_data['hit'],
                'reversals': minute_data['reversed'],
                'reversal_rate': round(minute_data['reversal_rate'], 4)
            }

    # Add volatility breakdown
    for _, _, band_name in VOLATILITY_BANDS:
        vol_data = stats['by_volatility'][band_name]
        if vol_data['hit'] > 0:
            json_output['reversal_analysis'][threshold_key]['by_volatility'][band_name] = {
                'periods_hit': vol_data['hit'],
                'reversals': vol_data['reversed'],
                'reversal_rate': round(vol_data['reversal_rate'], 4)
            }

# Write JSON output
json_output_file = "binary_option_reversal_analysis.json"
with open(json_output_file, 'w') as f:
    json.dump(json_output, f, indent=2)

print(f"\nJSON results written to {json_output_file}")

# Write TXT report
txt_output_file = "binary_option_reversal_analysis.txt"
with open(txt_output_file, 'w') as f:
    f.write("=" * 100 + "\n")
    f.write("BINARY OPTION PROBABILITY REVERSAL ANALYSIS\n")
    f.write("=" * 100 + "\n\n")

    f.write("Analysis: How often do options reverse from low probabilities to win?\n")
    f.write("A 'reversal' occurs when the call probability drops to or below a threshold,\n")
    f.write("but the option still expires in-the-money (call wins).\n\n")

    f.write(f"Total periods analyzed: {total_periods_analyzed}\n")
    f.write(f"Total records processed: {total_records_processed}\n")
    f.write(f"Date range: {date_range_start} to {date_range_end}\n\n")

    f.write("=" * 100 + "\n\n")

    # Overall summary
    f.write("OVERALL REVERSAL STATISTICS:\n")
    f.write("-" * 100 + "\n")
    f.write(f"{'Threshold':<15} {'Periods Hit':<15} {'Reversals':<15} {'No Reversal':<15} {'Reversal Rate':<15}\n")
    f.write("-" * 100 + "\n")

    for threshold in REVERSAL_THRESHOLDS:
        stats = reversal_stats['overall'][threshold]
        threshold_str = f"{threshold:.0%}"
        f.write(f"{threshold_str:<15} {stats['total_periods_hit']:<15} {stats['reversals']:<15} "
               f"{stats['no_reversals']:<15} {stats['reversal_rate']:.2%}\n")

    f.write("\n" + "=" * 100 + "\n\n")

    # Detailed breakdown by threshold
    for threshold in REVERSAL_THRESHOLDS:
        stats = reversal_stats['overall'][threshold]

        f.write(f"\nDETAILED ANALYSIS FOR {threshold:.0%} THRESHOLD:\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Total periods that hit {threshold:.0%}: {stats['total_periods_hit']}\n")
        f.write(f"Reversals (won after hitting threshold): {stats['reversals']}\n")
        f.write(f"No reversals (lost after hitting threshold): {stats['no_reversals']}\n")
        f.write(f"Overall reversal rate: {stats['reversal_rate']:.2%}\n\n")

        # By minute
        f.write(f"REVERSALS BY MINUTE (when {threshold:.0%} was first hit):\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Minute':<10} {'Periods Hit':<15} {'Reversals':<15} {'Reversal Rate':<15}\n")
        f.write("-" * 100 + "\n")

        for minute in range(15):
            minute_data = stats['by_minute'][minute]
            if minute_data['hit'] > 0:
                f.write(f"{minute:<10} {minute_data['hit']:<15} {minute_data['reversed']:<15} "
                       f"{minute_data['reversal_rate']:.2%}\n")

        f.write("\n")

        # By volatility
        f.write(f"REVERSALS BY VOLATILITY (when {threshold:.0%} was first hit):\n")
        f.write("-" * 100 + "\n")
        f.write(f"{'Volatility Band':<20} {'Periods Hit':<15} {'Reversals':<15} {'Reversal Rate':<15}\n")
        f.write("-" * 100 + "\n")

        for _, _, band_name in VOLATILITY_BANDS:
            vol_data = stats['by_volatility'][band_name]
            if vol_data['hit'] > 0:
                f.write(f"{band_name.replace('_', ' ').title():<20} {vol_data['hit']:<15} "
                       f"{vol_data['reversed']:<15} {vol_data['reversal_rate']:.2%}\n")

        f.write("\n" + "=" * 100 + "\n")

    f.write("\n" + "=" * 100 + "\n")
    f.write("KEY INSIGHTS:\n")
    f.write("=" * 100 + "\n\n")

    f.write("1. Lower thresholds (1%, 2%) indicate more extreme situations\n")
    f.write("   - Higher reversal rates suggest strong comeback potential\n")
    f.write("   - Lower reversal rates indicate these are often 'dead' positions\n\n")

    f.write("2. Reversal rates by minute show:\n")
    f.write("   - Early minutes: More time to reverse\n")
    f.write("   - Late minutes: Less time, but more certainty\n\n")

    f.write("3. Volatility impact:\n")
    f.write("   - High volatility may enable more dramatic reversals\n")
    f.write("   - Low volatility reversals indicate fundamental mispricing\n\n")

    f.write("=" * 100 + "\n")
    f.write("END OF REPORT\n")
    f.write("=" * 100 + "\n")

print(f"TXT report written to {txt_output_file}")

print("\n" + "="*80)
print("REVERSAL ANALYSIS COMPLETE")
print("="*80)
print(f"\nOutput files:")
print(f"  1. {json_output_file} (JSON format)")
print(f"  2. {txt_output_file} (Text report)")
print("\nKey findings:")
for threshold in REVERSAL_THRESHOLDS:
    stats = reversal_stats['overall'][threshold]
    if stats['total_periods_hit'] > 0:
        print(f"  {threshold:.0%} threshold: {stats['reversals']}/{stats['total_periods_hit']} reversals "
              f"({stats['reversal_rate']:.1%} reversal rate)")
print("="*80)
