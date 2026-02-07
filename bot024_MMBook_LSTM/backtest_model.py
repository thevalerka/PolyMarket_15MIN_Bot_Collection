#!/usr/bin/env python3
"""
Backtest the Order Book Prediction Model
Simulates trading based on model predictions and calculates actual P&L
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/ubuntu/013_2025_polymarket')

from orderbook_ml_predictor import OrderBookPredictor, MAX_HOLDING_PERIOD, TRADE_SIZE


class Backtester:
    """Backtest trading strategies based on model predictions"""
    
    def __init__(self, predictor: OrderBookPredictor):
        self.predictor = predictor
        self.trades = []
        self.equity_curve = [TRADE_SIZE]  # Start with initial capital
    
    def run_backtest(self, threshold: float = 0.1):
        """
        Run backtest on collected data
        threshold: minimum prediction score to trigger a trade
        """
        print("\n" + "="*80)
        print("📈 RUNNING BACKTEST - BINARY OPTIONS")
        print("="*80)
        print(f"Total samples: {len(self.predictor.training_data)}")
        print(f"Signal threshold: {threshold}")
        print(f"Max holding period: {MAX_HOLDING_PERIOD} snapshots (or until expiry)")
        print(f"Trade size: ${TRADE_SIZE}")
        print("="*80 + "\n")
        
        if len(self.predictor.training_data) < SEQUENCE_LENGTH + 100:
            print("❌ Not enough data for backtest")
            return
        
        # Iterate through data
        num_samples = len(self.predictor.training_data)
        
        i = SEQUENCE_LENGTH
        while i < num_samples - 100:
            # Get sequence
            sequence = np.array([
                self.predictor.training_data[j]['features'] 
                for j in range(i - SEQUENCE_LENGTH, i)
            ])
            
            # Get prediction
            prediction_score = self.predictor.predict(sequence)
            
            # Entry snapshot
            entry_snapshot = self.predictor.training_data[i]
            entry_time = entry_snapshot['timestamp']
            entry_price = entry_snapshot['mid_price']
            seconds_to_expiry = entry_snapshot.get('seconds_to_expiry', 0)
            
            # Skip if too close to expiry
            if seconds_to_expiry < 30:
                i += 1
                continue
            
            # Determine signal
            if prediction_score > threshold:
                signal = 'BUY'
            elif prediction_score < -threshold:
                signal = 'SELL'
            else:
                i += 1
                continue  # No trade
            
            # Find exit point (respecting expiration)
            max_hold = min(MAX_HOLDING_PERIOD, max(seconds_to_expiry - 30, 10))
            exit_snapshot = None
            exit_idx = None
            
            for j in range(i + 1, num_samples):
                time_held = (self.predictor.training_data[j]['timestamp'] - entry_time).total_seconds()
                
                if time_held >= max_hold:
                    exit_snapshot = self.predictor.training_data[j]
                    exit_idx = j
                    break
                
                # Check if crossed expiration
                if self.predictor.training_data[j].get('seconds_to_expiry', 999) < 5:
                    exit_snapshot = self.predictor.training_data[j]
                    exit_idx = j
                    break
            
            if exit_snapshot is None:
                i += 1
                continue
            
            # Calculate P&L
            pnl = self.predictor.calculate_pnl(entry_snapshot, exit_snapshot, 
                                               'long' if signal == 'BUY' else 'short')
            
            # Record trade
            trade = {
                'entry_time': entry_time,
                'exit_time': exit_snapshot['timestamp'],
                'signal': signal,
                'prediction': prediction_score,
                'entry_price': entry_price,
                'exit_price': exit_snapshot['mid_price'],
                'seconds_held': (exit_snapshot['timestamp'] - entry_time).total_seconds(),
                'pnl': pnl,
                'cumulative_pnl': sum([t['pnl'] for t in self.trades]) + pnl
            }
            
            self.trades.append(trade)
            self.equity_curve.append(self.equity_curve[-1] + pnl)
            
            # Jump to exit point to avoid overlapping trades
            i = exit_idx + 1
        
        # Calculate metrics
        self.calculate_metrics()
        self.plot_results()
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        if not self.trades:
            print("❌ No trades executed")
            return
        
        df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(df)
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = df['pnl'].sum()
        avg_pnl = df['pnl'].mean()
        
        avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = df[df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Risk metrics
        sharpe_ratio = (df['pnl'].mean() / df['pnl'].std()) * np.sqrt(252) if df['pnl'].std() > 0 else 0
        
        max_drawdown = 0
        peak = self.equity_curve[0]
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Print results
        print("\n" + "="*80)
        print("📊 BACKTEST RESULTS")
        print("="*80)
        print(f"\nTRADE STATISTICS:")
        print(f"  Total trades:     {total_trades}")
        print(f"  Winning trades:   {winning_trades} ({win_rate*100:.1f}%)")
        print(f"  Losing trades:    {losing_trades} ({(1-win_rate)*100:.1f}%)")
        
        print(f"\nP&L METRICS:")
        print(f"  Total P&L:        ${total_pnl:+.2f}")
        print(f"  Average P&L:      ${avg_pnl:+.2f}")
        print(f"  Average Win:      ${avg_win:+.2f}")
        print(f"  Average Loss:     ${avg_loss:+.2f}")
        print(f"  Profit Factor:    {abs(avg_win/avg_loss) if avg_loss != 0 else 0:.2f}")
        
        print(f"\nRISK METRICS:")
        print(f"  Sharpe Ratio:     {sharpe_ratio:.2f}")
        print(f"  Max Drawdown:     {max_drawdown*100:.2f}%")
        print(f"  Return on Capital: {(total_pnl/TRADE_SIZE)*100:.2f}%")
        
        print("="*80 + "\n")
        
        # Show sample trades
        print("SAMPLE TRADES:")
        print(df.head(10).to_string(index=False))
    
    def plot_results(self):
        """Plot equity curve and other visualizations"""
        if not self.trades:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Backtest Results', fontsize=16)
        
        # Equity curve
        axes[0, 0].plot(self.equity_curve, linewidth=2)
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_xlabel('Trade Number')
        axes[0, 0].set_ylabel('Equity ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # P&L distribution
        df = pd.DataFrame(self.trades)
        axes[0, 1].hist(df['pnl'], bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_title('P&L Distribution')
        axes[0, 1].set_xlabel('P&L ($)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Cumulative P&L
        axes[1, 0].plot(df['cumulative_pnl'], linewidth=2, color='green')
        axes[1, 0].set_title('Cumulative P&L')
        axes[1, 0].set_xlabel('Trade Number')
        axes[1, 0].set_ylabel('Cumulative P&L ($)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Win/Loss by signal
        signal_pnl = df.groupby('signal')['pnl'].sum()
        axes[1, 1].bar(signal_pnl.index, signal_pnl.values, 
                       color=['green' if x > 0 else 'red' for x in signal_pnl.values])
        axes[1, 1].set_title('P&L by Signal Type')
        axes[1, 1].set_xlabel('Signal')
        axes[1, 1].set_ylabel('Total P&L ($)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_file = '/home/ubuntu/013_2025_polymarket/backtest_results.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"📊 Results plot saved to: {output_file}")


def main():
    """Main backtesting function"""
    predictor = OrderBookPredictor()
    
    # Load existing model
    if predictor.model is None:
        print("❌ No trained model found. Train a model first.")
        return
    
    # Check if we have training data
    if not predictor.training_data:
        print("❌ No training data available")
        return
    
    # Run backtest
    backtester = Backtester(predictor)
    backtester.run_backtest(threshold=0.1)


if __name__ == '__main__':
    main()
