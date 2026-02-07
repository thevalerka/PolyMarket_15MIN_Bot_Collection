# spike_monitor.py - Real-time Monitoring Dashboard for BTC Spike Trading Bot
"""
Real-time Monitoring Dashboard for Spike Trading Bot

Features:
- Live status dashboard
- Performance metrics tracking
- Spike detection visualization
- Trading statistics
- System health monitoring
- Alert management
- Export capabilities
"""

import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading
from collections import deque

# Import dependencies
try:
    from spike_config import state, SPIKE_DETECTION, PROTECTED_SELL, RISK_MANAGEMENT
    from spike_logger import logger
except ImportError:
    print("⚠️ Monitor running in standalone mode")
    state = None
    logger = None

class SpikeMonitor:
    """Real-time monitoring system for spike trading bot."""
    
    def __init__(self):
        self.running = False
        self.monitor_thread = None
        self.update_interval = 5  # Update every 5 seconds
        
        # Real-time metrics
        self.metrics_history = deque(maxlen=720)  # 1 hour at 5-second intervals
        self.spike_events = deque(maxlen=100)     # Last 100 spike events
        self.trade_events = deque(maxlen=50)      # Last 50 trades
        self.alerts = deque(maxlen=20)            # Last 20 alerts
        
        # Performance tracking
        self.session_start_time = time.time()
        self.last_metrics_update = 0
        self.dashboard_data = {}
        
        # Health monitoring
        self.health_checks = {
            'websocket_connection': False,
            'api_connectivity': False,
            'data_freshness': False,
            'trading_capacity': False,
            'spike_detection': False
        }
        
        print("📊 Spike Trading Monitor initialized")
    
    def start_monitoring(self):
        """Start the monitoring system."""
        if self.running:
            print("⚠️ Monitor already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        if logger:
            logger.info("📊 Real-time monitoring started")
        print("✅ Monitoring system started")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        self.running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        if logger:
            logger.info("📊 Real-time monitoring stopped")
        print("🛑 Monitoring system stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Update metrics
                self._update_metrics()
                
                # Perform health checks
                self._perform_health_checks()
                
                # Update dashboard data
                self._update_dashboard_data()
                
                # Check for alerts
                self._check_alerts()
                
                # Sleep until next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                if logger:
                    logger.error(f"Error in monitoring loop: {e}")
                else:
                    print(f"❌ Monitor error: {e}")
                time.sleep(self.update_interval)
    
    def _update_metrics(self):
        """Update real-time metrics."""
        current_time = time.time()
        
        # Get current metrics from state
        if state:
            metrics = {
                'timestamp': current_time,
                'datetime': datetime.now().isoformat(),
                
                # Trading metrics
                'balance_tokens': getattr(state, 'last_known_balance', 0),
                'api_calls_count': getattr(state, 'api_calls_count', 0),
                'trades_today': getattr(state, 'trades_today', 0),
                'successful_trades': getattr(state, 'successful_trades', 0),
                'failed_trades': getattr(state, 'failed_trades', 0),
                
                # Spike detection metrics
                'spike_count_today': getattr(state, 'spike_count_today', 0),
                'last_spike_time': getattr(state, 'last_spike_time', 0),
                'websocket_connected': getattr(state, 'websocket_connected', False),
                'price_update_count': getattr(state, 'price_update_count', 0),
                
                # System metrics
                'connection_errors': getattr(state, 'connection_errors', 0),
                'protected_sell_active': getattr(state, 'protected_sell_active', False),
                'uptime_hours': (current_time - self.session_start_time) / 3600,
            }
            
            # Calculate derived metrics
            total_trades = metrics['successful_trades'] + metrics['failed_trades']
            metrics['success_rate'] = metrics['successful_trades'] / total_trades if total_trades > 0 else 0
            metrics['last_spike_ago_minutes'] = (current_time - metrics['last_spike_time']) / 60 if metrics['last_spike_time'] > 0 else None
            
            # BTC price history stats
            if hasattr(state, 'btc_price_history') and state.btc_price_history:
                recent_prices = [price for _, price in state.btc_price_history[-10:]]
                if len(recent_prices) > 1:
                    price_changes = [abs(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1] 
                                   for i in range(1, len(recent_prices))]
                    metrics['btc_volatility'] = sum(price_changes) / len(price_changes) if price_changes else 0
                    metrics['latest_btc_price'] = recent_prices[-1]
                else:
                    metrics['btc_volatility'] = 0
                    metrics['latest_btc_price'] = recent_prices[0] if recent_prices else 0
            else:
                metrics['btc_volatility'] = 0
                metrics['latest_btc_price'] = 0
            
        else:
            # Fallback metrics when state not available
            metrics = {
                'timestamp': current_time,
                'datetime': datetime.now().isoformat(),
                'status': 'state_unavailable',
                'uptime_hours': (current_time - self.session_start_time) / 3600,
            }
        
        # Add to history
        self.metrics_history.append(metrics)
        self.last_metrics_update = current_time
    
    def _perform_health_checks(self):
        """Perform system health checks."""
        current_time = time.time()
        
        # WebSocket connection check
        if state and hasattr(state, 'websocket_connected'):
            self.health_checks['websocket_connection'] = state.websocket_connected
        
        # Data freshness check (last price update within 30 seconds)
        if state and hasattr(state, 'last_price_update'):
            data_age = current_time - state.last_price_update
            self.health_checks['data_freshness'] = data_age < 30
        
        # API connectivity check (recent API calls)
        if state and hasattr(state, 'api_calls_count'):
            self.health_checks['api_connectivity'] = state.api_calls_count > 0
        
        # Trading capacity check
        if state and hasattr(state, 'can_place_spike_buy'):
            self.health_checks['trading_capacity'] = state.can_place_spike_buy()
        
        # Spike detection check (system active)
        if state and hasattr(state, 'spike_detection_active'):
            self.health_checks['spike_detection'] = state.spike_detection_active
    
    def _update_dashboard_data(self):
        """Update dashboard data structure."""
        current_metrics = self.metrics_history[-1] if self.metrics_history else {}
        
        self.dashboard_data = {
            'status': {
                'system_status': 'running' if self.running else 'stopped',
                'health_score': sum(self.health_checks.values()) / len(self.health_checks),
                'uptime_hours': current_metrics.get('uptime_hours', 0),
                'last_update': datetime.now().isoformat()
            },
            
            'trading': {
                'balance_tokens': current_metrics.get('balance_tokens', 0),
                'trades_today': current_metrics.get('trades_today', 0),
                'success_rate': current_metrics.get('success_rate', 0),
                'api_calls': current_metrics.get('api_calls_count', 0),
                'protected_sell_active': current_metrics.get('protected_sell_active', False)
            },
            
            'spike_detection': {
                'spikes_today': current_metrics.get('spike_count_today', 0),
                'last_spike_ago_minutes': current_metrics.get('last_spike_ago_minutes'),
                'btc_price': current_metrics.get('latest_btc_price', 0),
                'btc_volatility': current_metrics.get('btc_volatility', 0),
                'websocket_connected': current_metrics.get('websocket_connected', False)
            },
            
            'health_checks': self.health_checks.copy(),
            
            'alerts': {
                'total_alerts': len(self.alerts),
                'recent_alerts': list(self.alerts)[-5:] if self.alerts else []
            },
            
            'performance': self._calculate_performance_metrics(),
            
            'configuration': {
                'spike_threshold': SPIKE_DETECTION.get('min_spike_threshold', 0) if 'SPIKE_DETECTION' in globals() else 0,
                'lookback_minutes': SPIKE_DETECTION.get('lookback_minutes', 0) if 'SPIKE_DETECTION' in globals() else 0,
                'protected_sell_duration': PROTECTED_SELL.get('duration_seconds', 0) if 'PROTECTED_SELL' in globals() else 0,
                'max_spike_buys_per_hour': RISK_MANAGEMENT.get('max_spike_buys_per_hour', 0) if 'RISK_MANAGEMENT' in globals() else 0
            }
        }
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics from history."""
        if len(self.metrics_history) < 2:
            return {'status': 'insufficient_data'}
        
        # Get metrics from last hour
        one_hour_ago = time.time() - 3600
        recent_metrics = [m for m in self.metrics_history if m['timestamp'] > one_hour_ago]
        
        if not recent_metrics:
            return {'status': 'no_recent_data'}
        
        # Calculate trends
        first_metric = recent_metrics[0]
        last_metric = recent_metrics[-1]
        
        api_calls_per_hour = last_metric.get('api_calls_count', 0) - first_metric.get('api_calls_count', 0)
        trades_per_hour = last_metric.get('trades_today', 0) - first_metric.get('trades_today', 0)
        
        return {
            'api_calls_per_hour': api_calls_per_hour,
            'trades_per_hour': trades_per_hour,
            'avg_success_rate': sum(m.get('success_rate', 0) for m in recent_metrics) / len(recent_metrics),
            'data_points': len(recent_metrics),
            'time_range_hours': (last_metric['timestamp'] - first_metric['timestamp']) / 3600
        }
    
    def _check_alerts(self):
        """Check for new alerts and system issues."""
        current_time = time.time()
        
        # Check for stale data
        if state and hasattr(state, 'last_price_update'):
            data_age = current_time - state.last_price_update
            if data_age > 60:  # More than 1 minute old
                self._add_alert('stale_data', f"Price data is {data_age:.0f} seconds old")
        
        # Check for connection issues
        if not self.health_checks.get('websocket_connection', False):
            self._add_alert('connection_issue', "WebSocket disconnected")
        
        # Check for trading capacity
        if not self.health_checks.get('trading_capacity', False):
            self._add_alert('trading_capacity', "Trading capacity limited")
        
        # Check for high error rate
        current_metrics = self.metrics_history[-1] if self.metrics_history else {}
        connection_errors = current_metrics.get('connection_errors', 0)
        if connection_errors > 10:
            self._add_alert('high_error_rate', f"High error count: {connection_errors}")
    
    def _add_alert(self, alert_type: str, message: str):
        """Add alert to the queue."""
        alert = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'severity': self._get_alert_severity(alert_type)
        }
        
        # Avoid duplicate alerts (same type within 5 minutes)
        five_minutes_ago = time.time() - 300
        recent_alerts = [a for a in self.alerts if a['timestamp'] > five_minutes_ago and a['type'] == alert_type]
        
        if not recent_alerts:
            self.alerts.append(alert)
            
            if logger:
                severity = alert['severity']
                if severity == 'critical':
                    logger.error(f"🚨 CRITICAL: {message}")
                elif severity == 'warning':
                    logger.warning(f"⚠️ WARNING: {message}")
                else:
                    logger.info(f"ℹ️ INFO: {message}")
    
    def _get_alert_severity(self, alert_type: str) -> str:
        """Determine alert severity."""
        critical_alerts = ['connection_issue', 'api_failure', 'trading_failure']
        warning_alerts = ['stale_data', 'high_error_rate', 'trading_capacity']
        
        if alert_type in critical_alerts:
            return 'critical'
        elif alert_type in warning_alerts:
            return 'warning'
        else:
            return 'info'
    
    def add_spike_event(self, spike_info: Dict):
        """Add spike detection event."""
        event = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'strength': spike_info.get('strength', 0),
            'price_change': spike_info.get('price_change', 0),
            'threshold': spike_info.get('threshold', 0),
            'btc_price': spike_info.get('end_price', 0)
        }
        
        self.spike_events.append(event)
        
        if logger:
            logger.debug(f"📊 Monitor recorded spike event: {spike_info.get('strength', 0):.2f}x")
    
    def add_trade_event(self, trade_type: str, success: bool, trade_data: Dict):
        """Add trade execution event."""
        event = {
            'timestamp': time.time(),
            'datetime': datetime.now().isoformat(),
            'trade_type': trade_type,
            'success': success,
            'price': trade_data.get('price', 0),
            'amount': trade_data.get('amount', 0),
            'value_usd': trade_data.get('value_usd', 0)
        }
        
        self.trade_events.append(event)
        
        if logger:
            status = "SUCCESS" if success else "FAILED"
            logger.debug(f"📊 Monitor recorded trade: {trade_type} {status}")
    
    def get_dashboard_data(self) -> Dict:
        """Get current dashboard data."""
        return self.dashboard_data.copy()
    
    def get_historical_data(self, hours: int = 1) -> List[Dict]:
        """Get historical metrics data."""
        cutoff_time = time.time() - (hours * 3600)
        return [m for m in self.metrics_history if m['timestamp'] > cutoff_time]
    
    def get_spike_history(self, count: int = 20) -> List[Dict]:
        """Get recent spike detection history."""
        return list(self.spike_events)[-count:] if self.spike_events else []
    
    def get_trade_history(self, count: int = 10) -> List[Dict]:
        """Get recent trade history."""
        return list(self.trade_events)[-count:] if self.trade_events else []
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file."""
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'session_start': datetime.fromtimestamp(self.session_start_time).isoformat(),
                'dashboard_data': self.dashboard_data,
                'metrics_history': list(self.metrics_history),
                'spike_events': list(self.spike_events),
                'trade_events': list(self.trade_events),
                'alerts': list(self.alerts)
            }
            
            # Create directory if needed
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            if logger:
                logger.info(f"📊 Metrics exported to {filepath}")
            print(f"✅ Metrics exported to {filepath}")
            
        except Exception as e:
            if logger:
                logger.error(f"Failed to export metrics: {e}")
            else:
                print(f"❌ Export failed: {e}")
    
    def display_live_status(self):
        """Display live status in console."""
        if not self.dashboard_data:
            print("⚠️ No dashboard data available")
            return
        
        status = self.dashboard_data['status']
        trading = self.dashboard_data['trading']
        spike_detection = self.dashboard_data['spike_detection']
        health = self.dashboard_data['health_checks']
        
        print("\n" + "="*80)
        print("📊 LIVE SPIKE TRADING MONITOR")
        print("="*80)
        
        # System Status
        print(f"🟢 System Status: {status['system_status'].upper()}")
        print(f"💚 Health Score: {status['health_score']:.1%}")
        print(f"⏱️ Uptime: {status['uptime_hours']:.1f} hours")
        print(f"🔄 Last Update: {datetime.fromisoformat(status['last_update']).strftime('%H:%M:%S')}")
        
        print("\n📈 TRADING METRICS")
        print(f"💰 Balance: {trading['balance_tokens']:.3f} tokens")
        print(f"📊 Trades Today: {trading['trades_today']}")
        print(f"✅ Success Rate: {trading['success_rate']:.1%}")
        print(f"📞 API Calls: {trading['api_calls']}")
        print(f"🛡️ Protected Sell: {'ACTIVE' if trading['protected_sell_active'] else 'INACTIVE'}")
        
        print("\n⚡ SPIKE DETECTION")
        print(f"🚨 Spikes Today: {spike_detection['spikes_today']}")
        if spike_detection['last_spike_ago_minutes']:
            print(f"⏳ Last Spike: {spike_detection['last_spike_ago_minutes']:.1f} minutes ago")
        else:
            print("⏳ Last Spike: Never")
        print(f"💰 BTC Price: ${spike_detection['btc_price']:,.2f}")
        print(f"📊 BTC Volatility: {spike_detection['btc_volatility']:.3%}")
        print(f"🔗 WebSocket: {'CONNECTED' if spike_detection['websocket_connected'] else 'DISCONNECTED'}")
        
        print("\n🏥 HEALTH CHECKS")
        for check_name, status in health.items():
            status_emoji = "✅" if status else "❌"
            check_display = check_name.replace('_', ' ').title()
            print(f"{status_emoji} {check_display}")
        
        # Recent alerts
        alerts = self.dashboard_data['alerts']
        if alerts['recent_alerts']:
            print("\n🚨 RECENT ALERTS")
            for alert in alerts['recent_alerts']:
                severity_emoji = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}.get(alert['severity'], "🔔")
                alert_time = datetime.fromisoformat(alert['datetime']).strftime('%H:%M:%S')
                print(f"{severity_emoji} [{alert_time}] {alert['message']}")
        
        print("="*80)

# Global monitor instance
monitor = SpikeMonitor()

# Convenience functions
def start_monitoring():
    """Start the monitoring system."""
    monitor.start_monitoring()

def stop_monitoring():
    """Stop the monitoring system."""
    monitor.stop_monitoring()

def get_dashboard():
    """Get current dashboard data."""
    return monitor.get_dashboard_data()

def display_status():
    """Display live status."""
    monitor.display_live_status()

def export_data(filepath: str = None):
    """Export monitoring data."""
    if not filepath:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = f"./logs/monitor_export_{timestamp}.json"
    
    monitor.export_metrics(filepath)

# Integration functions for the trading bot
def record_spike(spike_info: Dict):
    """Record spike detection event."""
    monitor.add_spike_event(spike_info)

def record_trade(trade_type: str, success: bool, trade_data: Dict):
    """Record trade execution event."""
    monitor.add_trade_event(trade_type, success, trade_data)

# Command-line interface for standalone monitoring
if __name__ == "__main__":
    import sys
    import signal
    
    def signal_handler(sig, frame):
        print("\n🛑 Stopping monitor...")
        stop_monitoring()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🚀 Starting Spike Trading Monitor in standalone mode")
    start_monitoring()
    
    try:
        while True:
            time.sleep(10)
            display_status()
            time.sleep(20)  # Display every 30 seconds total
    except KeyboardInterrupt:
        stop_monitoring()
