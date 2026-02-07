module.exports = {
  apps: [{
    name: 'lstm-predictor',
    script: 'live_predictor_trading.py',
    interpreter: 'python3',
    cwd: '/home/ubuntu/013_2025_polymarket',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
    error_file: '/home/ubuntu/013_2025_polymarket/bot024_MMBook_LSTM/logs/error.log',
    out_file: '/home/ubuntu/013_2025_polymarket/bot024_MMBook_LSTM/logs/output.log',
    log_file: '/home/ubuntu/013_2025_polymarket/bot024_MMBook_LSTM/logs/combined.log',
    time: true,
    env: {
      PYTHONUNBUFFERED: '1'
    },
    kill_timeout: 10000,
    listen_timeout: 10000
  }]
}
