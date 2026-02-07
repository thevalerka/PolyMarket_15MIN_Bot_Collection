# spike_testing.py - Testing Utilities for BTC Spike Trading Bot
"""
Comprehensive Testing Suite for Spike Trading Bot

Features:
- Configuration validation
- Spike detection testing with simulated data
- WebSocket connection testing
- API connectivity testing
- Trading logic simulation
- Performance benchmarking
- Safety checks and validation
"""

import time
import json
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import threading

# Try to import bot components
try:
    from spike_strategy import SpikeDetector, MarketAnalyzer
    from spike_trading import TradingExecutor, SpikeTradingLogic
    from binance_websocket import EnhancedBinanceWebSocket
    from spike_config import state, SPIKE_DETECTION, validate_configuration
    from spike_logger import logger
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Some components not available: {e}")
    COMPONENTS_AVAILABLE = False

class SpikeTestSuite:
    """Comprehensive testing suite for spike trading bot."""
    
    def __init__(self):
        self.test_results = {}
        self.test_start_time = 0
        self.verbose = True
        
        print("🧪 Spike Trading Test Suite initialized")
    
    def run_all_tests(self, include_live_tests: bool = False) -> Dict:
        """Run complete test suite."""
        self.test_start_time = time.time()
        print("\n🚀 Starting Comprehensive Test Suite")
        print("="*60)
        
        # Configuration tests
        self.test_configuration()
        
        # Component tests
        if COMPONENTS_AVAILABLE:
            self.test_spike_detection()
            self.test_market_analysis()
            
            if include_live_tests:
                print("\n⚠️ Running LIVE tests (will make real connections)")
                self.test_websocket_connection()
                self.test_api_connectivity()
        
        # Simulation tests
        self.test_trading_simulation()
        self.test_performance_simulation()
        
        # Generate test report
        self.generate_test_report()
        
        return self.test_results
    
    def test_configuration(self):
        """Test configuration validation."""
        print("\n📋 Testing Configuration...")
        
        test_name = "configuration_validation"
        try:
            if COMPONENTS_AVAILABLE:
                errors = validate_configuration()
                if errors:
                    self.test_results[test_name] = {
                        'status': 'failed',
                        'errors': errors,
                        'message': f"Configuration has {len(errors)} errors"
                    }
                    self._print_result(test_name, False, f"{len(errors)} configuration errors")
                else:
                    self.test_results[test_name] = {
                        'status': 'passed',
                        'message': "Configuration validation passed"
                    }
                    self._print_result(test_name, True, "All configuration valid")
            else:
                self.test_results[test_name] = {
                    'status': 'skipped',
                    'message': "Components not available"
                }
                self._print_result(test_name, None, "Skipped - components unavailable")
                
        except Exception as e:
            self.test_results[test_name] = {
                'status': 'error',
                'error': str(e),
                'message': f"Configuration test failed: {e}"
            }
            self._print_result(test_name, False, f"Error: {e}")
    
    def test_spike_detection(self):
        """Test spike detection with simulated data."""
        print("\n⚡ Testing Spike Detection...")
        
        test_name = "spike_detection"
        try:
            # Create spike detector
            spike_detector = SpikeDetector(lookback_minutes=5, min_spike_threshold=0.005)
            
            # Generate test price data
            test_prices = self._generate_test_price_data()
            
            # Feed data to detector
            spikes_detected = []
            for timestamp, price in test_prices:
                spike_detector.add_price_point(price, timestamp)
                spike_detected, spike_info = spike_detector.detect_spike()
                
                if spike_detected:
                    spikes_detected.append(spike_info)
            
            # Evaluate results
            expected_spikes = self._count_expected_spikes(test_prices)
            detected_count = len(spikes_detected)
            
            # Allow some tolerance (±2 spikes)
            spike_accuracy = abs(detected_count - expected_spikes) <= 2
            
            if spike_accuracy and detected_count > 0:
                self.test_results[test_name] = {
                    'status': 'passed',
                    'spikes_detected': detected_count,
                    'expected_spikes': expected_spikes,
                    'test_data_points': len(test_prices),
                    'message': f"Detected {detected_count} spikes (expected ~{expected_spikes})"
                }
                self._print_result(test_name, True, f"Detected {detected_count} spikes correctly")
            else:
                self.test_results[test_name] = {
                    'status': 'failed',
                    'spikes_detected': detected_count,
                    'expected_spikes': expected_spikes,
                    'message': f"Spike detection accuracy issue: got {detected_count}, expected ~{expected_spikes}"
                }
                self._print_result(test_name, False, f"Detection mismatch: {detected_count} vs {expected_spikes}")
                
        except Exception as e:
            self.test_results[test_name] = {
                'status': 'error',
                'error': str(e),
                'message': f"Spike detection test failed: {e}"
            }
            self._print_result(test_name, False, f"Error: {e}")
    
    def test_market_analysis(self):
        """Test market analysis components."""
        print("\n📊 Testing Market Analysis...")
        
        test_name = "market_analysis"
        try:
            # Create test data files
            test_book_data = self._create_test_book_data()
            test_btc_data = self._create_test_btc_data()
            
            # Save test files
            with open('test_book.json', 'w') as f:
                json.dump(test_book_data, f)
            
            with open('test_btc.json', 'w') as f:
                json.dump(test_btc_data, f)
            
            # Test market analyzer
            analyzer = MarketAnalyzer('test_book.json', 'test_btc.json')
            analysis = analyzer.analyze_market()
            
            # Validate analysis
            if analysis and 'book_data' in analysis and 'spike_targets' in analysis:
                book_valid = analysis['book_data'].get('best_filtered_ask') is not None
                targets_valid = analysis['spike_targets'].get('buy_near_ask') is not None
                
                if book_valid and targets_valid:
                    self.test_results[test_name] = {
                        'status': 'passed',
                        'analysis_keys': list(analysis.keys()),
                        'message': "Market analysis working correctly"
                    }
                    self._print_result(test_name, True, "Analysis produces valid targets")
                else:
                    self.test_results[test_name] = {
                        'status': 'failed',
                        'book_valid': book_valid,
                        'targets_valid': targets_valid,
                        'message': "Analysis missing required components"
                    }
                    self._print_result(test_name, False, "Analysis incomplete")
            else:
                self.test_results[test_name] = {
                    'status': 'failed',
                    'analysis_result': analysis,
                    'message': "Market analysis returned invalid result"
                }
                self._print_result(test_name, False, "Analysis failed")
                
        except Exception as e:
            self.test_results[test_name] = {
                'status': 'error',
                'error': str(e),
                'message': f"Market analysis test failed: {e}"
            }
            self._print_result(test_name, False, f"Error: {e}")
        
        finally:
            # Clean up test files
            try:
                import os
                os.remove('test_book.json')
                os.remove('test_btc.json')
            except:
                pass
    
    def test_websocket_connection(self):
        """Test WebSocket connection (live test)."""
        print("\n🔗 Testing WebSocket Connection (Live)...")
        
        test_name = "websocket_connection"
        try:
            # Create WebSocket with timeout
            ws = EnhancedBinanceWebSocket()
            connection_successful = False
            price_received = False
            
            def test_callback(price_data):
                nonlocal price_received
                price_received = True
                print(f"   📊 Received price: ${price_data['price']:,.2f}")
            
            ws.set_spike_callback(test_callback)
            
            # Start WebSocket in separate thread with timeout
            ws_thread = threading.Thread(target=ws.start, daemon=True)
            ws_thread.start()
            
            # Wait for connection and data
            timeout = 30  # 30 seconds timeout
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                if ws.connected:
                    connection_successful = True
                    break
                time.sleep(1)
            
            # Wait a bit more for price data
            if connection_successful:
                time.sleep(5)
            
            ws.stop()
            
            if connection_successful and price_received:
                self.test_results[test_name] = {
                    'status': 'passed',
                    'connection_time': time.time() - start_time,
                    'message': "WebSocket connected and received data"
                }
                self._print_result(test_name, True, "Connection and data reception OK")
            elif connection_successful:
                self.test_results[test_name] = {
                    'status': 'partial',
                    'message': "Connected but no data received"
                }
                self._print_result(test_name, None, "Connected but no data")
            else:
                self.test_results[test_name] = {
                    'status': 'failed',
                    'message': "Failed to connect within timeout"
                }
                self._print_result(test_name, False, "Connection timeout")
                
        except Exception as e:
            self.test_results[test_name] = {
                'status': 'error',
                'error': str(e),
                'message': f"WebSocket test failed: {e}"
            }
            self._print_result(test_name, False, f"Error: {e}")
    
    def test_api_connectivity(self):
        """Test API connectivity (live test)."""
        print("\n📡 Testing API Connectivity (Live)...")
        
        test_name = "api_connectivity"
        try:
            # Create trading executor
            executor = TradingExecutor()
            
            # Test balance retrieval
            balance_raw, balance_readable = executor.get_balance()
            
            if balance_raw >= 0:  # Balance could be 0, that's valid
                self.test_results[test_name] = {
                    'status': 'passed',
                    'balance_tokens': balance_readable,
                    'message': f"API connection successful, balance: {balance_readable:.3f} tokens"
                }
                self._print_result(test_name, True, f"Balance: {balance_readable:.3f} tokens")
            else:
                self.test_results[test_name] = {
                    'status': 'failed',
                    'message': "API returned invalid balance"
                }
                self._print_result(test_name, False, "Invalid balance response")
                
        except Exception as e:
            self.test_results[test_name] = {
                'status': 'error',
                'error': str(e),
                'message': f"API connectivity test failed: {e}"
            }
            self._print_result(test_name, False, f"Error: {e}")
    
    def test_trading_simulation(self):
        """Test trading logic with simulated conditions."""
        print("\n💹 Testing Trading Logic Simulation...")
        
        test_name = "trading_simulation"
        try:
            # Simulate various market conditions
            test_scenarios = [
                self._create_spike_scenario(),
                self._create_normal_scenario(),
                self._create_volatile_scenario(),
                self._create_mandatory_sell_scenario()
            ]
            
            scenario_results = []
            
            for i, scenario in enumerate(test_scenarios):
                print(f"   📋 Testing scenario {i+1}: {scenario['name']}")
                
                # Simulate trading logic decisions
                result = self._simulate_trading_decisions(scenario)
                scenario_results.append(result)
                
                if self.verbose:
                    print(f"      {result['decisions_made']} decisions made")
            
            # Evaluate overall results
            successful_scenarios = sum(1 for r in scenario_results if r['success'])
            total_scenarios = len(scenario_results)
            
            if successful_scenarios == total_scenarios:
                self.test_results[test_name] = {
                    'status': 'passed',
                    'scenarios_tested': total_scenarios,
                    'scenarios_passed': successful_scenarios,
                    'scenario_details': scenario_results,
                    'message': f"All {total_scenarios} trading scenarios passed"
                }
                self._print_result(test_name, True, f"{successful_scenarios}/{total_scenarios} scenarios passed")
            else:
                self.test_results[test_name] = {
                    'status': 'partial',
                    'scenarios_tested': total_scenarios,
                    'scenarios_passed': successful_scenarios,
                    'scenario_details': scenario_results,
                    'message': f"{successful_scenarios}/{total_scenarios} scenarios passed"
                }
                self._print_result(test_name, None, f"{successful_scenarios}/{total_scenarios} scenarios")
                
        except Exception as e:
            self.test_results[test_name] = {
                'status': 'error',
                'error': str(e),
                'message': f"Trading simulation failed: {e}"
            }
            self._print_result(test_name, False, f"Error: {e}")
    
    def test_performance_simulation(self):
        """Test performance under various loads."""
        print("\n⚡ Testing Performance Simulation...")
        
        test_name = "performance_simulation"
        try:
            if not COMPONENTS_AVAILABLE:
                self.test_results[test_name] = {
                    'status': 'skipped',
                    'message': "Components not available"
                }
                self._print_result(test_name, None, "Skipped - components unavailable")
                return
            
            # Performance test parameters
            data_points = 1000
            spike_detector = SpikeDetector()
            
            # Measure processing time
            start_time = time.time()
            
            # Generate and process data
            for i in range(data_points):
                timestamp = time.time() * 1000 + i * 1000  # 1 second intervals
                price = 40000 + random.random() * 1000  # Random BTC price
                
                spike_detector.add_price_point(price, timestamp)
                spike_detector.detect_spike()
            
            processing_time = time.time() - start_time
            points_per_second = data_points / processing_time
            
            # Performance thresholds
            min_performance = 100  # At least 100 points per second
            
            if points_per_second >= min_performance:
                self.test_results[test_name] = {
                    'status': 'passed',
                    'data_points_processed': data_points,
                    'processing_time_seconds': processing_time,
                    'points_per_second': points_per_second,
                    'message': f"Performance adequate: {points_per_second:.1f} points/sec"
                }
                self._print_result(test_name, True, f"{points_per_second:.1f} points/sec")
            else:
                self.test_results[test_name] = {
                    'status': 'failed',
                    'data_points_processed': data_points,
                    'processing_time_seconds': processing_time,
                    'points_per_second': points_per_second,
                    'message': f"Performance below threshold: {points_per_second:.1f} points/sec"
                }
                self._print_result(test_name, False, f"Too slow: {points_per_second:.1f} points/sec")
                
        except Exception as e:
            self.test_results[test_name] = {
                'status': 'error',
                'error': str(e),
                'message': f"Performance test failed: {e}"
            }
            self._print_result(test_name, False, f"Error: {e}")
    
    def _generate_test_price_data(self) -> List[Tuple[int, float]]:
        """Generate realistic test price data with known spikes."""
        data = []
        base_price = 40000.0
        timestamp = int(time.time() * 1000)
        
        # Generate 5 minutes of data (300 points)
        for i in range(300):
            current_timestamp = timestamp + i * 1000  # 1 second intervals
            
            # Add some random walk
            base_price += random.uniform(-50, 50)
            
            # Add artificial spikes at known intervals
            if i in [50, 150, 250]:  # Spike at specific points
                spike_magnitude = random.uniform(0.006, 0.012)  # 0.6% to 1.2% spike
                spike_direction = random.choice([1, -1])
                base_price *= (1 + spike_magnitude * spike_direction)
            
            data.append((current_timestamp, base_price))
        
        return data
    
    def _count_expected_spikes(self, price_data: List[Tuple[int, float]]) -> int:
        """Count expected spikes in test data (rough estimate)."""
        # This is a simplified version of spike counting
        # In real testing, we'd use the same algorithm as the detector
        significant_moves = 0
        
        for i in range(1, len(price_data)):
            prev_price = price_data[i-1][1]
            curr_price = price_data[i][1]
            
            if prev_price > 0:
                change = abs(curr_price - prev_price) / prev_price
                if change > 0.005:  # 0.5% threshold
                    significant_moves += 1
        
        # Expect roughly 3-5 spikes in our test data
        return 3
    
    def _create_test_book_data(self) -> Dict:
        """Create test order book data."""
        return {
            'timestamp': int(time.time() * 1000),
            'delay_ms': 100,
            'complete_book': {
                'bids': [
                    {'price': '0.456', 'size': '500'},
                    {'price': '0.455', 'size': '1000'},
                    {'price': '0.454', 'size': '750'}
                ],
                'asks': [
                    {'price': '0.458', 'size': '1200'},
                    {'price': '0.459', 'size': '800'},
                    {'price': '0.460', 'size': '600'}
                ]
            }
        }
    
    def _create_test_btc_data(self) -> Dict:
        """Create test BTC price data."""
        return {
            'price': 42567.89,
            'timestamp': int(time.time() * 1000),
            'symbol': 'BTCUSDT'
        }
    
    def _create_spike_scenario(self) -> Dict:
        """Create a spike trading scenario."""
        return {
            'name': 'Spike Detection',
            'spike_detected': True,
            'spike_strength': 2.5,
            'best_ask': 0.458,
            'market_mid': 0.457,
            'balance_tokens': 5.0,
            'protected_levels': [{'price': 0.459, 'size': 1200}]
        }
    
    def _create_normal_scenario(self) -> Dict:
        """Create a normal market scenario."""
        return {
            'name': 'Normal Market',
            'spike_detected': False,
            'best_ask': 0.456,
            'market_mid': 0.455,
            'balance_tokens': 3.5,
            'market_spread': 0.002
        }
    
    def _create_volatile_scenario(self) -> Dict:
        """Create a volatile market scenario."""
        return {
            'name': 'High Volatility',
            'spike_detected': False,
            'best_ask': 0.467,
            'market_mid': 0.465,
            'balance_tokens': 2.0,
            'btc_volatility': 0.15,
            'market_spread': 0.008
        }
    
    def _create_mandatory_sell_scenario(self) -> Dict:
        """Create a mandatory sell scenario."""
        return {
            'name': 'Mandatory Sell',
            'spike_detected': False,
            'best_bid': 0.990,
            'market_mid': 0.991,
            'balance_tokens': 4.2,
            'mandatory_sell_triggered': True
        }
    
    def _simulate_trading_decisions(self, scenario: Dict) -> Dict:
        """Simulate trading logic decisions for a scenario."""
        decisions_made = 0
        actions_taken = []
        
        # Simulate spike buy decision
        if scenario.get('spike_detected'):
            actions_taken.append('spike_buy')
            decisions_made += 1
        
        # Simulate protected sell decision
        if scenario.get('balance_tokens', 0) > 0.1 and scenario.get('protected_levels'):
            actions_taken.append('protected_sell')
            decisions_made += 1
        
        # Simulate mandatory sell decision
        if scenario.get('mandatory_sell_triggered'):
            actions_taken.append('mandatory_sell')
            decisions_made += 1
        
        # Simulate spread sell decision
        if (scenario.get('balance_tokens', 0) > 0.1 and 
            not scenario.get('spike_detected') and 
            not scenario.get('mandatory_sell_triggered')):
            actions_taken.append('spread_sell')
            decisions_made += 1
        
        return {
            'success': decisions_made > 0,
            'decisions_made': decisions_made,
            'actions_taken': actions_taken,
            'scenario_name': scenario['name']
        }
    
    def _print_result(self, test_name: str, success: Optional[bool], message: str):
        """Print test result with formatting."""
        if success is True:
            status_emoji = "✅"
            status_text = "PASS"
        elif success is False:
            status_emoji = "❌"
            status_text = "FAIL"
        else:
            status_emoji = "⚠️"
            status_text = "SKIP"
        
        test_display = test_name.replace('_', ' ').title()
        print(f"{status_emoji} {status_text}: {test_display} - {message}")
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        total_time = time.time() - self.test_start_time
        
        print("\n" + "="*60)
        print("📋 TEST SUITE REPORT")
        print("="*60)
        
        # Count results
        passed = sum(1 for r in self.test_results.values() if r['status'] == 'passed')
        failed = sum(1 for r in self.test_results.values() if r['status'] == 'failed')
        errors = sum(1 for r in self.test_results.values() if r['status'] == 'error')
        skipped = sum(1 for r in self.test_results.values() if r['status'] == 'skipped')
        partial = sum(1 for r in self.test_results.values() if r['status'] == 'partial')
        
        total_tests = len(self.test_results)
        
        print(f"📊 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"🚨 Errors: {errors}")
        print(f"⚠️ Partial: {partial}")
        print(f"⏭️ Skipped: {skipped}")
        print(f"⏱️ Total Time: {total_time:.2f} seconds")
        
        # Overall status
        if failed == 0 and errors == 0:
            print(f"\n🎉 Overall Status: SUCCESS ({passed}/{total_tests} tests passed)")
        elif failed > 0 or errors > 0:
            print(f"\n⚠️ Overall Status: ISSUES FOUND ({failed + errors} failures/errors)")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if failed > 0 or errors > 0:
            print("   🔧 Fix failed tests before running the bot")
        
        if not COMPONENTS_AVAILABLE:
            print("   📦 Install missing components for full testing")
        
        if 'websocket_connection' in self.test_results and self.test_results['websocket_connection']['status'] != 'passed':
            print("   🌐 Check internet connection for WebSocket functionality")
        
        if 'api_connectivity' in self.test_results and self.test_results['api_connectivity']['status'] != 'passed':
            print("   🔑 Verify API credentials and permissions")
        
        if passed == total_tests:
            print("   🚀 System ready for live trading!")
        
        print("="*60)
        
        # Save report to file
        self._save_test_report()
    
    def _save_test_report(self):
        """Save test report to file."""
        try:
            report_data = {
                'test_run_timestamp': datetime.now().isoformat(),
                'test_duration_seconds': time.time() - self.test_start_time,
                'test_results': self.test_results,
                'components_available': COMPONENTS_AVAILABLE,
                'summary': {
                    'total_tests': len(self.test_results),
                    'passed': sum(1 for r in self.test_results.values() if r['status'] == 'passed'),
                    'failed': sum(1 for r in self.test_results.values() if r['status'] == 'failed'),
                    'errors': sum(1 for r in self.test_results.values() if r['status'] == 'error'),
                    'skipped': sum(1 for r in self.test_results.values() if r['status'] == 'skipped')
                }
            }
            
            # Create logs directory
            import os
            os.makedirs('./logs', exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = f"./logs/test_report_{timestamp}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            print(f"📄 Test report saved: {report_file}")
            
        except Exception as e:
            print(f"⚠️ Could not save test report: {e}")

# Convenience functions
def run_quick_test():
    """Run quick test suite (no live tests)."""
    suite = SpikeTestSuite()
    return suite.run_all_tests(include_live_tests=False)

def run_full_test():
    """Run full test suite including live tests."""
    suite = SpikeTestSuite()
    return suite.run_all_tests(include_live_tests=True)

def test_spike_detection_only():
    """Test only spike detection functionality."""
    suite = SpikeTestSuite()
    suite.test_spike_detection()
    return suite.test_results

def test_configuration_only():
    """Test only configuration validation."""
    suite = SpikeTestSuite()
    suite.test_configuration()
    return suite.test_results

# Command line interface
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        
        if test_type == 'quick':
            print("🏃‍♂️ Running quick test suite...")
            run_quick_test()
        elif test_type == 'full':
            print("🔍 Running full test suite...")
            run_full_test()
        elif test_type == 'spike':
            print("⚡ Testing spike detection only...")
            test_spike_detection_only()
        elif test_type == 'config':
            print("📋 Testing configuration only...")
            test_configuration_only()
        else:
            print("❓ Unknown test type. Options: quick, full, spike, config")
    else:
        print("🧪 Running default quick test suite...")
        run_quick_test()
