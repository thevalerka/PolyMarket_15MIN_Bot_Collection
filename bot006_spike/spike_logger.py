# spike_logger.py - Advanced Logging System for BTC Spike Trading Bot
"""
Comprehensive Logging System for Spike Trading Bot

Features:
- Multi-level logging (DEBUG, INFO, WARNING, ERROR)
- File rotation and management
- Trade execution logging
- Performance metrics logging
- Spike detection event logging
- Error tracking and alerting
- Real-time monitoring dashboard data
"""

import logging
import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler
import threading

# Import configuration (assuming we're using the enhanced config)
try:
    from spike_config import LOGGING_CONFIG, MONITORING, ALERTS, state
except ImportError:
    # Fallback configuration if spike_config not available
    LOGGING_CONFIG = {
        'log_level': 'INFO',
        'log_to_file': True,
        'log_file_path': './logs/spike_bot.log',
        'max_log_size_mb': 50,
        'keep_log_files': 7,
        'log_trades': True,
        'trade_log_path': './logs/trades.json',
    }
    MONITORING = {
        'save_performance_stats': True,
        'stats_file_path': './logs/performance.json',
        'stats_save_interval': 300,
    }
    ALERTS = {
        'enable_alerts': True,
        'alert_on_spike_detection': True,
        'alert_on_trade_execution': True,
        'alert_on_connection_issues': True,
        'alert_on_error_threshold': 5,
    }

class SpikeLogger:
    """Advanced logging system for spike trading bot."""
    
    def __init__(self):
        self.logger = None
        self.trade_logger = None
        self.performance_logger = None
        self.error_count = 0
        self.last_error_time = 0
        self.log_lock = threading.Lock()
        
        # Statistics
        self.log_stats = {
            'total_logs': 0,
            'error_logs': 0,
            'warning_logs': 0,
            'info_logs': 0,
            'debug_logs': 0,
            'trades_logged': 0,
            'spikes_logged': 0
        }
        
        self.setup_logging()
        
    def setup_logging(self):
        """Initialize all logging components."""
        # Create log directories
        self._create_log_directories()
        
        # Setup main logger
        self._setup_main_logger()
        
        # Setup trade logger
        if LOGGING_CONFIG.get('log_trades', True):
            self._setup_trade_logger()
        
        # Setup performance logger
        if MONITORING.get('save_performance_stats', True):
            self._setup_performance_logger()
        
        self.info("🚀 Spike Trading Logger initialized")
        
    def _create_log_directories(self):
        """Create necessary log directories."""
        for path_key in ['log_file_path', 'trade_log_path']:
            if path_key in LOGGING_CONFIG:
                log_dir = os.path.dirname(LOGGING_CONFIG[path_key])
                os.makedirs(log_dir, exist_ok=True)
        
        if 'stats_file_path' in MONITORING:
            stats_dir = os.path.dirname(MONITORING['stats_file_path'])
            os.makedirs(stats_dir, exist_ok=True)
    
    def _setup_main_logger(self):
        """Setup main application logger."""
        self.logger = logging.getLogger('spike_trading_bot')
        self.logger.setLevel(getattr(logging, LOGGING_CONFIG.get('log_level', 'INFO')))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if enabled)
        if LOGGING_CONFIG.get('log_to_file', True):
            max_bytes = LOGGING_CONFIG.get('max_log_size_mb', 50) * 1024 * 1024
            backup_count = LOGGING_CONFIG.get('keep_log_files', 7)
            
            file_handler = RotatingFileHandler(
                LOGGING_CONFIG['log_file_path'],
                maxBytes=max_bytes,
                backupCount=backup_count
            )
            
            file_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def _setup_trade_logger(self):
        """Setup dedicated trade logging."""
        self.trade_logger = logging.getLogger('spike_trades')
        self.trade_logger.setLevel(logging.INFO)
        self.trade_logger.handlers.clear()
        
        # Trade file handler
        trade_handler = RotatingFileHandler(
            LOGGING_CONFIG['trade_log_path'],
            maxBytes=10 * 1024 * 1024,  # 10MB for trade logs
            backupCount=5
        )
        
        trade_formatter = logging.Formatter('%(message)s')
        trade_handler.setFormatter(trade_formatter)
        self.trade_logger.addHandler(trade_handler)
    
    def _setup_performance_logger(self):
        """Setup performance metrics logger."""
        self.performance_logger = logging.getLogger('spike_performance')
        self.performance_logger.setLevel(logging.INFO)
        self.performance_logger.handlers.clear()
        
        # Performance file handler
        perf_path = MONITORING['stats_file_path'].replace('.json', '.log')
        perf_handler = RotatingFileHandler(
            perf_path,
            maxBytes=5 * 1024 * 1024,  # 5MB for performance logs
            backupCount=3
        )
        
        perf_formatter = logging.Formatter(
            '%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        perf_handler.setFormatter(perf_formatter)
        self.performance_logger.addHandler(perf_handler)
    
    # Standard logging methods
    def debug(self, message: str, extra_data: Dict = None):
        """Log debug message."""
        self._log(logging.DEBUG, message, extra_data)
        self.log_stats['debug_logs'] += 1
    
    def info(self, message: str, extra_data: Dict = None):
        """Log info message."""
        self._log(logging.INFO, message, extra_data)
        self.log_stats['info_logs'] += 1
    
    def warning(self, message: str, extra_data: Dict = None):
        """Log warning message."""
        self._log(logging.WARNING, message, extra_data)
        self.log_stats['warning_logs'] += 1
        
        # Check for alert threshold
        self._check_alert_threshold()
    
    def error(self, message: str, extra_data: Dict = None, exception: Exception = None):
        """Log error message."""
        if exception:
            message = f"{message} - Exception: {str(exception)}"
        
        self._log(logging.ERROR, message, extra_data)
        self.log_stats['error_logs'] += 1
        self.error_count += 1
        self.last_error_time = time.time()
        
        # Check for alert threshold
        self._check_alert_threshold()
    
    def _log(self, level: int, message: str, extra_data: Dict = None):
        """Internal logging method."""
        with self.log_lock:
            if self.logger:
                if extra_data:
                    message = f"{message} | Data: {json.dumps(extra_data)}"
                self.logger.log(level, message)
                self.log_stats['total_logs'] += 1
    
    # Specialized logging methods
    def log_spike_detection(self, spike_info: Dict):
        """Log spike detection event."""
        log_data = {
            'event_type': 'spike_detection',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'unix_timestamp': time.time(),
            'spike_strength': spike_info.get('strength', 0),
            'price_change': spike_info.get('price_change', 0),
            'threshold': spike_info.get('threshold', 0),
            'btc_price': spike_info.get('end_price', 0),
            'detection_method': 'candlestick_body_analysis'
        }
        
        self.info(f"🚨 SPIKE DETECTED! Strength: {spike_info.get('strength', 0):.2f}x, Change: {spike_info.get('price_change', 0):.3%}")
        
        if self.trade_logger:
            self.trade_logger.info(json.dumps(log_data))
        
        self.log_stats['spikes_logged'] += 1
        
        # Alert if enabled
        if ALERTS.get('alert_on_spike_detection', True):
            self._send_alert('spike_detection', log_data)
    
    def log_trade_execution(self, trade_type: str, success: bool, trade_data: Dict):
        """Log trade execution."""
        log_data = {
            'event_type': 'trade_execution',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'unix_timestamp': time.time(),
            'trade_type': trade_type,  # 'spike_buy', 'protected_sell', 'spread_sell', 'mandatory_sell'
            'success': success,
            'price': trade_data.get('price', 0),
            'amount': trade_data.get('amount', 0),
            'value_usd': trade_data.get('value_usd', 0),
            'order_id': trade_data.get('order_id', ''),
            'execution_time_ms': trade_data.get('execution_time_ms', 0)
        }
        
        status = "✅ SUCCESS" if success else "❌ FAILED"
        trade_emoji = {
            'spike_buy': '⚡',
            'protected_sell': '🛡️',
            'spread_sell': '📊',
            'mandatory_sell': '🚨'
        }.get(trade_type, '📈')
        
        self.info(f"{trade_emoji} {trade_type.upper()} {status}: ${trade_data.get('price', 0):.3f} x {trade_data.get('amount', 0):.3f}")
        
        if self.trade_logger:
            self.trade_logger.info(json.dumps(log_data))
        
        self.log_stats['trades_logged'] += 1
        
        # Update state
        if hasattr(state, 'update_trade_result'):
            profit_loss = trade_data.get('profit_loss', 0)
            state.update_trade_result(success, profit_loss)
        
        # Alert if enabled
        if ALERTS.get('alert_on_trade_execution', True):
            self._send_alert('trade_execution', log_data)
    
    def log_websocket_event(self, event_type: str, event_data: Dict):
        """Log WebSocket connection events."""
        log_data = {
            'event_type': f'websocket_{event_type}',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'unix_timestamp': time.time(),
            'connection_status': event_data.get('connected', False),
            'error_count': event_data.get('error_count', 0),
            'message_count': event_data.get('message_count', 0),
            'last_price': event_data.get('last_price', 0)
        }
        
        if event_type == 'connected':
            self.info("🟢 WebSocket connected successfully")
        elif event_type == 'disconnected':
            self.warning("🔴 WebSocket disconnected")
        elif event_type == 'error':
            self.error(f"❌ WebSocket error: {event_data.get('error', 'Unknown')}")
        elif event_type == 'reconnected':
            self.info(f"🔄 WebSocket reconnected after {event_data.get('attempts', 0)} attempts")
        
        # Update state
        if hasattr(state, 'update_websocket_status'):
            state.update_websocket_status(
                event_data.get('connected', False),
                event_data.get('error_count', 0)
            )
        
        # Alert on connection issues
        if ALERTS.get('alert_on_connection_issues', True) and event_type in ['disconnected', 'error']:
            self._send_alert('connection_issue', log_data)
    
    def log_performance_metrics(self, metrics: Dict):
        """Log performance metrics."""
        if not self.performance_logger:
            return
        
        log_data = {
            'event_type': 'performance_metrics',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'unix_timestamp': time.time(),
            'metrics': metrics
        }
        
        self.performance_logger.info(json.dumps(log_data))
        
        # Log summary to main logger
        trading_perf = metrics.get('trading_performance', {})
        success_rate = trading_perf.get('success_rate', 0)
        total_trades = trading_perf.get('total_trades', 0)
        
        self.info(f"📊 Performance: {total_trades} trades, {success_rate:.1%} success rate")
    
    def log_system_status(self, status_data: Dict):
        """Log system status update."""
        log_data = {
            'event_type': 'system_status',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'unix_timestamp': time.time(),
            'status': status_data
        }
        
        # Extract key metrics
        balance = status_data.get('balance_tokens', 0)
        token_value = status_data.get('token_value_usd', 0)
        capacity_remaining = status_data.get('buy_capacity_remaining', 0)
        
        self.debug(f"💰 Balance: {balance:.3f} tokens (${token_value:.2f}), Capacity: ${capacity_remaining:.2f}")
        
        if self.performance_logger:
            self.performance_logger.info(json.dumps(log_data))
    
    def _check_alert_threshold(self):
        """Check if error threshold reached for alerts."""
        if not ALERTS.get('enable_alerts', True):
            return
        
        threshold = ALERTS.get('alert_on_error_threshold', 5)
        time_window = 300  # 5 minutes
        
        current_time = time.time()
        if (self.error_count >= threshold and 
            current_time - self.last_error_time < time_window):
            
            alert_data = {
                'alert_type': 'error_threshold_reached',
                'error_count': self.error_count,
                'threshold': threshold,
                'time_window_minutes': time_window / 60
            }
            
            self._send_alert('error_threshold', alert_data)
            
            # Reset counter after alert
            self.error_count = 0
    
    def _send_alert(self, alert_type: str, alert_data: Dict):
        """Send alert through configured methods."""
        if not ALERTS.get('enable_alerts', True):
            return
        
        alert_methods = ALERTS.get('alert_methods', ['console'])
        
        alert_message = self._format_alert_message(alert_type, alert_data)
        
        for method in alert_methods:
            if method == 'console':
                self._send_console_alert(alert_message)
            elif method == 'file':
                self._send_file_alert(alert_message, alert_data)
            elif method == 'webhook':
                self._send_webhook_alert(alert_message, alert_data)
    
    def _format_alert_message(self, alert_type: str, alert_data: Dict) -> str:
        """Format alert message."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        if alert_type == 'spike_detection':
            return f"🚨 [{timestamp}] SPIKE DETECTED! Strength: {alert_data.get('spike_strength', 0):.2f}x"
        
        elif alert_type == 'trade_execution':
            trade_type = alert_data.get('trade_type', '').upper()
            status = "SUCCESS" if alert_data.get('success') else "FAILED"
            return f"📈 [{timestamp}] {trade_type} {status}: ${alert_data.get('price', 0):.3f}"
        
        elif alert_type == 'connection_issue':
            return f"⚠️ [{timestamp}] WebSocket connection issue detected"
        
        elif alert_type == 'error_threshold':
            count = alert_data.get('error_count', 0)
            return f"🚨 [{timestamp}] Error threshold reached: {count} errors"
        
        return f"🔔 [{timestamp}] Alert: {alert_type}"
    
    def _send_console_alert(self, message: str):
        """Send alert to console."""
        print(f"\n{'='*50}")
        print(f"ALERT: {message}")
        print(f"{'='*50}\n")
    
    def _send_file_alert(self, message: str, alert_data: Dict):
        """Send alert to file."""
        alert_file = LOGGING_CONFIG.get('log_file_path', './logs/spike_bot.log').replace('.log', '_alerts.log')
        
        try:
            with open(alert_file, 'a') as f:
                f.write(f"{datetime.now().isoformat()} - {message}\n")
                f.write(f"Data: {json.dumps(alert_data)}\n\n")
        except Exception as e:
            self.error(f"Failed to send file alert: {e}")
    
    def _send_webhook_alert(self, message: str, alert_data: Dict):
        """Send alert via webhook (placeholder for future implementation)."""
        # This would integrate with Discord, Slack, Teams, etc.
        self.debug(f"Webhook alert: {message}")
    
    def get_log_statistics(self) -> Dict:
        """Get logging statistics."""
        return {
            'log_stats': self.log_stats.copy(),
            'error_count': self.error_count,
            'last_error_ago': time.time() - self.last_error_time if self.last_error_time > 0 else None,
            'loggers_active': {
                'main_logger': self.logger is not None,
                'trade_logger': self.trade_logger is not None,
                'performance_logger': self.performance_logger is not None
            }
        }
    
    def flush_logs(self):
        """Flush all log handlers."""
        with self.log_lock:
            for handler in self.logger.handlers:
                handler.flush()
            
            if self.trade_logger:
                for handler in self.trade_logger.handlers:
                    handler.flush()
            
            if self.performance_logger:
                for handler in self.performance_logger.handlers:
                    handler.flush()

# Global logger instance
logger = SpikeLogger()

# Convenience functions for easy import
def log_spike(spike_info: Dict):
    """Log spike detection."""
    logger.log_spike_detection(spike_info)

def log_trade(trade_type: str, success: bool, trade_data: Dict):
    """Log trade execution."""
    logger.log_trade_execution(trade_type, success, trade_data)

def log_websocket(event_type: str, event_data: Dict):
    """Log WebSocket event."""
    logger.log_websocket_event(event_type, event_data)

def log_performance(metrics: Dict):
    """Log performance metrics."""
    logger.log_performance_metrics(metrics)

def log_system_status(status_data: Dict):
    """Log system status."""
    logger.log_system_status(status_data)

# Standard logging functions
def debug(message: str, extra_data: Dict = None):
    logger.debug(message, extra_data)

def info(message: str, extra_data: Dict = None):
    logger.info(message, extra_data)

def warning(message: str, extra_data: Dict = None):
    logger.warning(message, extra_data)

def error(message: str, extra_data: Dict = None, exception: Exception = None):
    logger.error(message, extra_data, exception)

# Usage example and test
if __name__ == "__main__":
    # Test the logging system
    info("🧪 Testing spike logger system")
    
    # Test spike detection logging
    spike_info = {
        'strength': 2.5,
        'price_change': 0.0087,
        'threshold': 0.0034,
        'end_price': 43567.89
    }
    log_spike(spike_info)
    
    # Test trade logging
    trade_data = {
        'price': 0.456,
        'amount': 9.87,
        'value_usd': 4.50,
        'order_id': 'test_123',
        'execution_time_ms': 234
    }
    log_trade('spike_buy', True, trade_data)
    
    # Test WebSocket logging
    websocket_data = {
        'connected': True,
        'message_count': 150,
        'last_price': 43567.89
    }
    log_websocket('connected', websocket_data)
    
    info("✅ Logging system test completed")
