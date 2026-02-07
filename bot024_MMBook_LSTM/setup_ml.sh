#!/bin/bash
# Setup ML Prediction System

echo "🔧 Setting up Order Book ML Prediction System"
echo "=============================================="

# Install dependencies
echo "📦 Installing Python dependencies..."
pip3 install --break-system-packages torch numpy pandas matplotlib scikit-learn

# Create directories
echo "📁 Creating directories..."
mkdir -p /home/ubuntu/013_2025_polymarket/bot024_MMBook_LSTM/models
mkdir -p /home/ubuntu/013_2025_polymarket/bot024_MMBook_LSTM/history

# Make scripts executable
echo "✅ Making scripts executable..."
chmod +x orderbook_ml_predictor.py
chmod +x live_predictor.py
chmod +x backtest_model.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Start data collection: python3 live_predictor.py"
echo "2. Wait for data (need 500+ samples)"
echo "3. Model will auto-train after 1 hour"
echo "4. Run backtest: python3 backtest_model.py"
echo ""
echo "Files will be saved to:"
echo "  - Models: /home/ubuntu/013_2025_polymarket/models/"
echo "  - History: /home/ubuntu/013_2025_polymarket/history/"
