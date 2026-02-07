# Advanced Market Maker for Polymarket Binary Options

A comprehensive market making engine implementing 10 advanced techniques to avoid adverse selection and optimize profitability.

## 🎯 Techniques Implemented

| # | Technique | Description | Module |
|---|-----------|-------------|--------|
| 1 | **Asymmetric Quote Skewing** | Skew quotes based on detected momentum | `mm_engine.py` |
| 2 | **Inventory-Based Skewing** | Adjust quotes to reduce position risk | `mm_engine.py` |
| 3 | **Quote Fading on Vol Spikes** | Pull/widen quotes during volatility | `VolatilityMonitor` |
| 4 | **Layered/Tiered Quoting** | Multiple quote layers with increasing size/spread | `generate_tiered_quotes()` |
| 5 | **Lead-Lag Arbitrage** | Use fastest price source as leading indicator | `LeadLagAnalyzer` |
| 6 | **Greeks-Aware Management** | Widen spreads as gamma increases near expiry | `GreeksCalculator` |
| 7 | **Order Flow Toxicity** | Detect informed flow and adjust accordingly | `OrderFlowAnalyzer` |
| 8 | **Time-of-Period Awareness** | Different strategies for each phase of 15min period | `PeriodPhaseManager` |
| 9 | **Correlation-Based Hedging** | Hedge with opposite option vs taking losses | `hedge_analyzer.py` |
| 10 | **Regime Detection** | Adapt strategy to market regime | `RegimeDetector` |

## 📁 File Structure

```
advanced_mm/
├── mm_engine.py         # Core MM engine with all techniques
├── mm_bot.py            # Production bot runner
├── hedge_analyzer.py    # Hedging strategy module
├── mm_backtest.py       # Strategy backtester
└── README.md            # This file
```

## 🚀 Quick Start

### 1. Run the Bot (Dry Run)

```bash
cd /home/ubuntu/013_2025_polymarket/advanced_mm
python3 mm_bot.py --dry-run
```

### 2. Run with Different Configs

```bash
# Aggressive (tighter spreads, more quoting)
python3 mm_bot.py --aggressive

# Conservative (wider spreads, more pulling)
python3 mm_bot.py --conservative

# Scalping (high frequency, small positions)
python3 mm_bot.py --scalping
```

### 3. Backtest Strategies

```bash
# Run with synthetic data
python3 mm_backtest.py

# Generate sample data
python3 mm_backtest.py --generate-sample --periods 200

# Run optimization
python3 mm_backtest.py --optimize
```

## ⚙️ Configuration

Edit `MMConfig` in `mm_engine.py` or use presets:

```python
config = MMConfig(
    # Base spreads
    base_spread=0.02,      # 2 cents base
    min_spread=0.01,       # Never tighter than 1 cent
    max_spread=0.10,       # Never wider than 10 cents
    
    # Position limits
    max_position_size=1000.0,
    max_inventory_imbalance=500.0,
    
    # Skewing factors
    inventory_skew_factor=0.001,   # Per unit of inventory
    momentum_skew_factor=0.005,    # Based on direction
    
    # Volatility
    vol_spike_threshold=0.002,     # 0.2% move triggers spike
    vol_pullback_seconds=5.0,      # Cooldown after spike
    
    # Tiered quoting
    tier_sizes=[50, 100, 200, 500],
    tier_spreads=[0.01, 0.02, 0.03, 0.05],
)
```

## 📊 Output Files

The bot writes to:

- `mm_quotes.json` - Current quote recommendations
- `mm_status.json` - MM status and metrics

### Quote Output Format

```json
{
  "timestamp": 1234567890123,
  "time_to_expiry_seconds": 542,
  "phase": "mid",
  "regime": "mean_reverting",
  "quotes": {
    "CALL": [
      {"side": "bid", "price": 0.42, "size": 50, "layer": 0},
      {"side": "ask", "price": 0.46, "size": 50, "layer": 0}
    ],
    "PUT": [...]
  }
}
```

## 🔧 Integration with Your Bot

```python
from mm_engine import AdvancedMarketMaker, MMConfig, create_market_state_from_files

# Initialize
config = MMConfig(base_spread=0.02)
mm = AdvancedMarketMaker(config)

# In your main loop
state = create_market_state_from_files(
    bybit_path='/path/to/bybit.json',
    chainlink_path='/path/to/chainlink.json',
    coinbase_path='/path/to/coinbase.json',
    call_path='/path/to/call.json',
    put_path='/path/to/put.json',
    period_start_timestamp=period_start_ms,
)

# Get quote recommendations
quotes = mm.generate_all_quotes(state)

# quotes['CALL'] and quotes['PUT'] contain Quote objects
for quote in quotes['CALL']:
    print(f"{quote.side} {quote.size} @ {quote.price}")
```

## 📈 Strategy Logic Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Market State Update                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  1. Update Analyzers                                         │
│     - Lead-Lag (price leadership)                           │
│     - Regime Detector (trending/mean-reverting/volatile)    │
│     - Volatility Monitor (spike detection)                  │
│     - Order Flow (toxicity)                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Determine Quote Action                                   │
│     - PULL_QUOTES: Vol spike, final phase, extreme toxicity │
│     - WIDEN_SPREADS: High vol regime, high toxicity         │
│     - QUOTE_BID_ONLY: Upward momentum + medium toxicity     │
│     - QUOTE_ASK_ONLY: Downward momentum + medium toxicity   │
│     - QUOTE_BOTH: Normal conditions                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Calculate Fair Value                                     │
│     - Use Chainlink price vs strike                         │
│     - Estimate implied volatility                           │
│     - Binary option probability                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Apply Skews                                              │
│     - Inventory skew (reduce position risk)                 │
│     - Momentum skew (avoid getting picked off)              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Calculate Dynamic Spread                                 │
│     - Base spread × time_factor × regime_factor             │
│       × toxicity_factor × vol_factor × phase_factor         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  6. Generate Tiered Quotes                                   │
│     - Layer 0: Small size, tight spread                     │
│     - Layer 1: Medium size, wider spread                    │
│     - Layer 2: Large size, even wider                       │
│     - Layer 3: Max size, widest spread                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  7. Output Quotes                                            │
│     - Write to mm_quotes.json                               │
│     - Your execution layer reads and places orders          │
└─────────────────────────────────────────────────────────────┘
```

## 🛡️ Risk Management

### Position Limits
- `max_position_size`: Hard cap on total position
- `max_inventory_imbalance`: Max difference between CALL and PUT exposure

### Automatic Quote Pulling
Quotes are automatically pulled when:
- Volatility spike detected (>2x normal)
- In final phase (<60s to expiry)
- Extreme order flow toxicity detected
- High gamma danger zone

### Hedging
Use `hedge_analyzer.py` to evaluate:
- Should you close a losing position OR hedge with opposite option?
- What's the max loss if hedged vs closed?

## 📝 PM2 Setup

```bash
# Start with PM2
pm2 start mm_bot.py --name mm-engine --interpreter python3 -- --interval 100

# Monitor
pm2 logs mm-engine

# Restart with different config
pm2 restart mm-engine -- --conservative
```

## ⚠️ Important Notes

1. **This generates RECOMMENDATIONS, not orders** - You need execution logic to actually place orders on Polymarket

2. **Backtest before live** - Always run backtester with your historical data first

3. **Monitor closely** - MM strategies can lose money quickly in adverse conditions

4. **Adjust parameters** - The defaults are starting points; optimize for your market

## 📞 Support

Questions? Check the code comments or modify parameters based on your observations.
