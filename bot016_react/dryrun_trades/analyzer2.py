import json
from datetime import datetime
import numpy as np

# === BIN DEFINITIONS ===
DISTANCE_BINS = [
    (0, 1, "0-1"), (1, 5, "1-5"), (5, 10, "5-10"), (10, 20, "10-20"),
    (20, 40, "20-40"), (40, 80, "40-80"), (80, 160, "80-160"),
    (160, 320, "160-320"), (320, 640, "320-640"), (640, 1280, "640-1280"),
    (1280, float('inf'), "1280+")
]

TIME_BINS = [
    (13*60, 15*60, "15m-13m"), (11*60, 13*60, "13m-11m"), (10*60, 11*60, "11m-10m"),
    (9*60, 10*60, "10m-9m"), (8*60, 9*60, "9m-8m"), (7*60, 8*60, "8m-7m"),
    (6*60, 7*60, "7m-6m"), (5*60, 6*60, "6m-5m"), (4*60, 5*60, "5m-4m"),
    (3*60, 4*60, "4m-3m"), (2*60, 3*60, "3m-2m"), (90, 120, "120s-90s"),
    (60, 90, "90s-60s"), (40, 60, "60s-40s"), (30, 40, "40s-30s"),
    (20, 30, "30s-20s"), (10, 20, "20s-10s"), (5, 10, "10s-5s"),
    (2, 5, "5s-2s"), (0, 2, "last-2s")
]

VOL_BINS = [
    (0, 10, "0-10"), (10, 20, "10-20"), (20, 30, "20-30"), (30, 40, "30-40"),
    (40, 60, "40-60"), (60, 90, "60-90"), (90, 120, "90-120"), (120, 240, "120-240"),
    (240, float('inf'), "240+")
]

# === HELPER FUNCTIONS ===
def get_bin_midpoint(bin_list, label):
    for low, high, name in bin_list:
        if name == label:
            return (low + high) / 2
    return None

def parse_open_bin(open_bin):
    parts = open_bin.split('|')
    distance_label = parts[0]
    time_label = parts[1]
    vol_label = parts[2]
    return distance_label, time_label, vol_label

def categorize_time(open_time_str):
    dt = datetime.fromisoformat(open_time_str.replace('Z', '+00:00'))
    hour = dt.hour
    weekday = dt.weekday()

    if 8 <= hour < 14 and weekday < 5:
        return 'Europe'
    elif 14 <= hour < 22 and weekday < 5:
        return 'USA'
    else:
        return 'Night/Asia/Weekend'

# === SAFE CORRELATION FUNCTION ===
def safe_correlation(x, y):
    try:
        if len(set(x)) < 2 or len(set(y)) < 2:
            return float('nan')
        return np.corrcoef(x, y)[0, 1]
    except Exception:
        return float('nan')

# === LOAD DATA ===
file_path = "/home/ubuntu/013_2025_polymarket/bot016_react/dryrun_trades/dryrun_trades_20260128.json"
with open(file_path, 'r') as f:
    data = json.load(f)

# === PREPROCESS TRADES ===
for trade in data['trades']:
    dist_label, time_label, vol_label = parse_open_bin(trade['open_bin'])
    trade['distance_label'] = dist_label
    trade['time_label'] = time_label
    trade['vol_label'] = vol_label

    # Map labels to midpoints
    trade['distance_numeric'] = get_bin_midpoint(DISTANCE_BINS, dist_label)
    trade['time_numeric'] = get_bin_midpoint(TIME_BINS, time_label)
    trade['vol_numeric'] = get_bin_midpoint(VOL_BINS, vol_label)

    trade['session'] = categorize_time(trade['open_time'])

# === GENERAL ANALYSIS ===
print("=== GENERAL ANALYSIS ===")
pnl_values = [t['pnl'] for t in data['trades']]
distances = [t['distance_numeric'] for t in data['trades']]
times = [t['time_numeric'] for t in data['trades']]
volatilities = [t['vol_numeric'] for t in data['trades']]

correlation_distance = safe_correlation(distances, pnl_values)
correlation_time = safe_correlation(times, pnl_values)
correlation_volatility = safe_correlation(volatilities, pnl_values)

print(f"PnL vs Distance Correlation: {correlation_distance:.4f}")
print(f"PnL vs Time to Expiry Correlation: {correlation_time:.4f}")
print(f"PnL vs Volatility Correlation: {correlation_volatility:.4f}")

# === CLUSTER ANALYSIS ===
num_clusters = min(5, len(data['trades']))
cluster_size = max(1, len(data['trades']) // num_clusters)
clusters = []
for i in range(num_clusters):
    start_index = i * cluster_size
    end_index = start_index + cluster_size if i < num_clusters - 1 else len(data['trades'])
    clusters.append(data['trades'][start_index:end_index])

print("\n=== CLUSTER ANALYSIS ===")
for idx, cluster in enumerate(clusters):
    print(f"\nCluster {idx+1}:")
    if not cluster:
        print("No trades in cluster.")
        continue
    avg_pnl = sum([t['pnl'] for t in cluster]) / len(cluster)
    print(f"Average PnL: {avg_pnl:.4f}")

    cluster_distances = [t['distance_numeric'] for t in cluster]
    cluster_pnls = [t['pnl'] for t in cluster]
    dist_corr = safe_correlation(cluster_distances, cluster_pnls)
    print(f"PnL vs Distance Correlation: {dist_corr:.4f}")

# === MARKET SESSION ANALYSIS ===
sessions = {'Europe': [], 'USA': [], 'Night/Asia/Weekend': []}
for trade in data['trades']:
    sessions[trade['session']].append(trade)

print("\n=== MARKET SESSION ANALYSIS ===")
for session, trades in sessions.items():
    if not trades:
        continue
    avg_pnl = sum([t['pnl'] for t in trades]) / len(trades)
    win_count = sum(1 for t in trades if t['pnl'] > 0)
    loss_count = len(trades) - win_count
    print(f"{session} - Avg PnL: {avg_pnl:.4f}, Wins: {win_count}, Losses: {loss_count}")
