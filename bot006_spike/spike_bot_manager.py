#!/usr/bin/env python3
# spike_bot_manager.py - Complete Bot Manager and CLI Interface
"""
Complete BTC Spike Trading Bot Manager

Features:
- CLI interface for bot management
- Integrated monitoring and logging
- Safety checks and validation
- Performance optimization
- Emergency controls
- Configuration management
- Testing integration
"""

import sys
import time
import json
import signal
import argparse
import threading
from datetime import datetime
from typing import Dict, Optional

# Import all bot components
try:
    from spike_trading_bot import BTCSpikeBot
    from spike_strategy import SpikeDetector
    from spike_config import state, get_config_summary, validate_configuration
    from spike_logger import logger, log_spike, log_trade, log_websocket
    from spike_monitor import monitor, start_monitoring, stop_monitoring, display_status
    from spike_testing import SpikeTestSuite, run_quick_test, run_full_test
    from binance_websocket import EnhancedBinanceWebSocket
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Critical Error: Missing components - {e}")
    print("Please ensure all bot files are in the correct directory")
    sys.exit(1)

class SpikeBotManager:
    """Complete bot manager with CLI interface."""
    
    def __init__(self):
        self.bot = None
        self.monitoring_active = False
        self.emergency_stop = False
        self.startup_time = time.time()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print("🎯 BTC Spike Trading Bot Manager initialized")
    
    def _signal_handler(self, sig, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n🛑 Received signal {sig} - Initiating graceful shutdown...")
        self.emergency_stop = True
        self.stop_all_systems()
        sys.exit(0)
    
    def validate_system(self) -> bool:
        """Validate system before starting."""
        print("🔍 Validating system configuration...")
        
        # Configuration validation
        config_errors = validate_configuration()
        if config_errors:
            print("❌ Configuration validation failed:")
            for error in config_errors:
                print(f"   • {error}")
            return False
        
        print("✅ Configuration validation passed")
        
        # Run quick tests
        print("🧪 Running system tests...")
        suite = SpikeTestSuite()
        suite.verbose = False
        results = suite.run_all_tests(include_live_tests=False)
        
        failed_tests = [name for name, result in results.items() 
                       if result['status'] in ['failed', 'error']]
        
        if failed_tests:
            print("❌ System tests failed:")
            for test in failed_tests:
                print(f"   • {test}: {results[test]['message']}")
            return False
        
        print("✅ System tests passed")
        return True
    
    def start_bot(self, mode: str = 'live') -> bool:
        """Start the trading bot."""
        try:
            if not self.validate_system():
                print("❌ System validation failed - cannot start bot")
                return False
            
            print(f"🚀 Starting BTC Spike Trading Bot in {mode.upper()} mode...")
            
            # Initialize bot
            self.bot = BTCSpikeBot()
            
            # Start monitoring
            if not self.monitoring_active:
                start_monitoring()
                self.monitoring_active = True
                print("✅ Monitoring system started")
            
            # Start bot based on mode
            if mode == 'live':
                print("⚡ LIVE TRADING MODE - Real money at risk!")
                print("🔄 Starting in 3 seconds... (Ctrl+C to cancel)")
                time.sleep(3)
                
                if not self.emergency_stop:
                    self.bot.run_live_trading()
            
            elif mode == 'analysis':
                print("📊 ANALYSIS MODE - No trading, monitoring only")
                self.bot.run_analysis_only()
            
            elif mode == 'test':
                print("🧪 TEST MODE - Running with test data")
                self._run_test_mode()
            
            else:
                print(f"❌ Unknown mode: {mode}")
                return False
            
            return True
            
        except KeyboardInterrupt:
            print("\n👋 Bot stopped by user")
            return True
        except Exception as e:
            logger.error(f"Bot startup failed: {e}")
            print(f"❌ Bot startup failed: {e}")
            return False
        finally:
            self.stop_all_systems()
    
    def stop_all_systems(self):
        """Stop all bot systems gracefully."""
        print("🛑 Stopping all systems...")
        
        # Stop bot
        if self.bot:
            try:
                # Cancel all orders
                self.bot.trading_executor.cancel_all_orders("manager shutdown")
                print("✅ Orders cancelled")
            except:
                print("⚠️ Could not cancel orders")
        
        # Stop monitoring
        if self.monitoring_active:
            stop_monitoring()
            self.monitoring_active = False
            print("✅ Monitoring stopped")
        
        # Stop WebSocket if running
        try:
            if self.bot and hasattr(self.bot, 'binance_ws'):
                self.bot.binance_ws.stop()
                print("✅ WebSocket stopped")
        except:
            pass
        
        print("✅ All systems stopped")
    
    def _run_test_mode(self):
        """Run bot in test mode with simulated data."""
        print("🧪 Test mode not fully implemented - running analysis mode")
        self.bot.run_analysis_only()
    
    def show_status(self):
        """Show current bot status."""
        print("\n📊 CURRENT BOT STATUS")
        print("="*50)
        
        # System status
        uptime = time.time() - self.startup_time
        print(f"⏱️ Manager Uptime: {uptime/3600:.1f} hours")
        print(f"🤖 Bot Active: {'Yes' if self.bot else 'No'}")
        print(f"📊 Monitoring: {'Active' if self.monitoring_active else 'Inactive'}")
        
        # State information
        if state:
            print(f"💰 Balance: {getattr(state, 'last_known_balance', 0):.3f} tokens")
            print(f"📞 API Calls: {getattr(state, 'api_calls_count', 0)}")
            print(f"🚨 Spikes Today: {getattr(state, 'spike_count_today', 0)}")
            print(f"📈 Trades Today: {getattr(state, 'trades_today', 0)}")
            print(f"🔗 WebSocket: {'Connected' if getattr(state, 'websocket_connected', False) else 'Disconnected'}")
        
        # Live monitoring display
        if self.monitoring_active:
            print("\n📈 LIVE MONITOR DATA:")
            display_status()
        
        print("="*50)
    
    def run_tests(self, test_type: str = 'quick'):
        """Run bot tests."""
        print(f"🧪 Running {test_type} tests...")
        
        if test_type == 'quick':
            results = run_quick_test()
        elif test_type == 'full':
            results = run_full_test()
        else:
            print(f"❌ Unknown test type: {test_type}")
            return
        
        # Display summary
        passed = sum(1 for r in results.values() if r['status'] == 'passed')
        total = len(results)
        
        print(f"\n✅ Tests completed: {passed}/{total} passed")
    
    def emergency_stop_trading(self):
        """Emergency stop all trading operations."""
        print("🚨 EMERGENCY STOP ACTIVATED!")
        
        try:
            if self.bot:
                self.bot.trading_executor.cancel_all_orders("EMERGENCY STOP")
                print("✅ All orders cancelled")
            
            self.emergency_stop = True
            print("✅ Emergency stop completed")
            
        except Exception as e:
            print(f"❌ Emergency stop failed: {e}")
    
    def show_configuration(self):
        """Display current configuration."""
        print("\n⚙️ CURRENT CONFIGURATION")
        print("="*50)
        
        config = get_config_summary()
        
        for section, settings in config.items():
            print(f"\n📋 {section.replace('_', ' ').title()}:")
            if isinstance(settings, dict):
                for key, value in settings.items():
                    print(f"   {key}: {value}")
            else:
                print(f"   {settings}")
        
        print("="*50)
    
    def export_data(self, filepath: str = None):
        """Export monitoring and performance data."""
        if not filepath:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = f"./logs/export_{timestamp}.json"
        
        try:
            # Export monitoring data
            monitor.export_metrics(filepath)
            
            # Add manager data
            manager_data = {
                'manager_uptime_hours': (time.time() - self.startup_time) / 3600,
                'bot_active': self.bot is not None,
                'monitoring_active': self.monitoring_active,
                'emergency_stop': self.emergency_stop
            }
            
            # Load existing export and add manager data
            with open(filepath, 'r') as f:
                export_data = json.load(f)
            
            export_data['manager_data'] = manager_data
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"✅ Data exported to {filepath}")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")

def create_cli_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description='BTC Spike Trading Bot Manager',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s start live          # Start live trading
  %(prog)s start analysis      # Start analysis mode
  %(prog)s test quick          # Run quick tests
  %(prog)s status              # Show current status
  %(prog)s stop                # Emergency stop
  %(prog)s config              # Show configuration
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start the bot')
    start_parser.add_argument('mode', choices=['live', 'analysis', 'test'], 
                             help='Bot operating mode')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run tests')
    test_parser.add_argument('type', choices=['quick', 'full'], 
                            default='quick', nargs='?',
                            help='Test type to run')
    
    # Status command
    subparsers.add_parser('status', help='Show bot status')
    
    # Stop command
    subparsers.add_parser('stop', help='Emergency stop all operations')
    
    # Config command
    subparsers.add_parser('config', help='Show configuration')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export data')
    export_parser.add_argument('--file', help='Export file path')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Live monitoring')
    monitor_parser.add_argument('--interval', type=int, default=30,
                               help='Update interval in seconds')
    
    return parser

def run_live_monitor(interval: int = 30):
    """Run live monitoring display."""
    manager = SpikeBotManager()
    
    try:
        # Start monitoring system
        start_monitoring()
        print(f"📊 Live monitoring started (updates every {interval}s)")
        print("Press Ctrl+C to stop")
        
        while True:
            display_status()
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")
    finally:
        stop_monitoring()

def main():
    """Main CLI entry point."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = SpikeBotManager()
    
    try:
        if args.command == 'start':
            success = manager.start_bot(args.mode)
            sys.exit(0 if success else 1)
            
        elif args.command == 'test':
            manager.run_tests(args.type)
            
        elif args.command == 'status':
            manager.show_status()
            
        elif args.command == 'stop':
            manager.emergency_stop_trading()
            
        elif args.command == 'config':
            manager.show_configuration()
            
        elif args.command == 'export':
            manager.export_data(args.file)
            
        elif args.command == 'monitor':
            run_live_monitor(args.interval)
            
        else:
            print(f"❌ Unknown command: {args.command}")
            parser.print_help()
            
    except Exception as e:
        logger.error(f"CLI command failed: {e}")
        print(f"❌ Command failed: {e}")
        sys.exit(1)

# Interactive mode functions
def interactive_mode():
    """Run interactive mode for bot management."""
    manager = SpikeBotManager()
    
    print("\n🎮 BTC Spike Trading Bot - Interactive Mode")
    print("=" * 50)
    print("Available commands:")
    print("  start [live|analysis|test] - Start the bot")
    print("  status                     - Show status")
    print("  test [quick|full]         - Run tests")
    print("  stop                      - Emergency stop")
    print("  config                    - Show configuration")
    print("  export [filename]         - Export data")
    print("  monitor                   - Live monitoring")
    print("  help                      - Show this help")
    print("  quit                      - Exit")
    print("=" * 50)
    
    while True:
        try:
            command = input("\n🤖 spike-bot> ").strip().split()
            
            if not command:
                continue
                
            cmd = command[0].lower()
            
            if cmd == 'quit' or cmd == 'exit':
                break
                
            elif cmd == 'help':
                print("Available commands: start, status, test, stop, config, export, monitor, help, quit")
                
            elif cmd == 'start':
                mode = command[1] if len(command) > 1 else 'analysis'
                if mode in ['live', 'analysis', 'test']:
                    manager.start_bot(mode)
                else:
                    print("❌ Invalid mode. Use: live, analysis, or test")
                    
            elif cmd == 'status':
                manager.show_status()
                
            elif cmd == 'test':
                test_type = command[1] if len(command) > 1 else 'quick'
                manager.run_tests(test_type)
                
            elif cmd == 'stop':
                manager.emergency_stop_trading()
                
            elif cmd == 'config':
                manager.show_configuration()
                
            elif cmd == 'export':
                filepath = command[1] if len(command) > 1 else None
                manager.export_data(filepath)
                
            elif cmd == 'monitor':
                print("Starting live monitoring... (Ctrl+C to return)")
                try:
                    run_live_monitor(10)  # 10 second updates
                except KeyboardInterrupt:
                    print("Returned to interactive mode")
                    
            else:
                print(f"❌ Unknown command: {cmd}")
                
        except KeyboardInterrupt:
            print("\nUse 'quit' to exit properly")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("👋 Goodbye!")
    manager.stop_all_systems()

# Quick launcher functions
def quick_start_live():
    """Quick start in live mode."""
    print("⚡ QUICK START - LIVE TRADING")
    manager = SpikeBotManager()
    return manager.start_bot('live')

def quick_start_analysis():
    """Quick start in analysis mode."""
    print("📊 QUICK START - ANALYSIS MODE")
    manager = SpikeBotManager()
    return manager.start_bot('analysis')

def quick_test():
    """Quick test run."""
    print("🧪 QUICK TEST")
    return run_quick_test()

if __name__ == "__main__":
    # Check if running interactively
    if len(sys.argv) == 1:
        interactive_mode()
    else:
        main()
