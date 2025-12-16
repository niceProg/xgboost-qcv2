#!/usr/bin/env python3
"""
End-to-End System Test for XGBoost Real-time Trading System
Tests all components: Monitor, Trainer, API, and QuantConnect Integration
"""

import os
import sys
import json
import time
import logging
import requests
import pymysql
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SystemTester:
    """Test the complete XGBoost real-time trading system."""

    def __init__(self):
        self.api_base_url = "http://localhost:8000"
        self.test_results = []

    def run_test(self, test_name: str, test_func):
        """Run a test and record results."""
        logger.info(f"Running test: {test_name}")
        try:
            result = test_func()
            if result:
                logger.info(f"✅ {test_name} - PASSED")
                self.test_results.append((test_name, "PASSED", None))
            else:
                logger.error(f"❌ {test_name} - FAILED")
                self.test_results.append((test_name, "FAILED", "Test returned False"))
        except Exception as e:
            logger.error(f"❌ {test_name} - ERROR: {e}")
            self.test_results.append((test_name, "ERROR", str(e)))

    def test_api_health(self) -> bool:
        """Test API health endpoint."""
        response = requests.get(f"{self.api_base_url}/health", timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get('status') == 'healthy' and data.get('model_available')

    def test_database_connection(self) -> bool:
        """Test database connection."""
        try:
            conn = pymysql.connect(
                host=os.getenv('TRADING_DB_HOST', 'localhost'),
                port=int(os.getenv('TRADING_DB_PORT', 3306)),
                user=os.getenv('TRADING_DB_USER'),
                password=os.getenv('TRADING_DB_PASSWORD'),
                database=os.getenv('TRADING_DB_NAME'),
                connect_timeout=5
            )

            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            conn.close()
            return result[0] == 1
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            return False

    def test_api_prediction(self) -> bool:
        """Test API prediction endpoint."""
        # Prepare test features
        test_features = {
            "price_close": 42000.0,
            "price_high": 42500.0,
            "price_low": 41500.0,
            "price_open": 41800.0,
            "volume_usd": 1000000.0,
            "price_range": 1000.0,
            "body_size": 200.0,
            "upper_wick": 700.0,
            "lower_wick": 300.0,
            "close_position": 0.5,
            "price_change_pct": 0.48,
            "volume_price_ratio": 23.81
        }

        response = requests.post(
            f"{self.api_base_url}/predict",
            json=test_features,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        # Check response structure
        required_fields = ['prediction', 'confidence', 'probability', 'timestamp']
        return all(field in data for field in required_fields)

    def test_api_signal(self) -> bool:
        """Test API signal generation endpoint."""
        request_data = {
            "exchange": "binance",
            "symbol": "BTCUSDT",
            "interval": "1h"
        }

        response = requests.post(
            f"{self.api_base_url}/signal",
            json=request_data,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        # Check response structure
        required_fields = ['signal', 'confidence', 'recommendation', 'timestamp']
        signal_valid = data.get('signal') in ['BUY', 'SELL', 'HOLD']

        return all(field in data for field in required_fields) and signal_valid

    def test_quantconnect_integration(self) -> bool:
        """Test QuantConnect algorithm integration."""
        # Read the QuantConnect algorithm file
        algo_file = Path("XGBoostTradingAlgorithm_Final.py")
        if not algo_file.exists():
            return False

        # Check if it contains API integration code
        content = algo_file.read_text()

        # Key integration checks
        checks = [
            "def CallAPI" in content,
            "def GetTradingSignal" in content,
            "/signal" in content,
            "api_base_url" in content,
            "test.dragonfortune.ai" in content
        ]

        return all(checks)

    def test_file_structure(self) -> bool:
        """Test if all required files are present."""
        required_files = [
            "quantconnect_api.py",
            "realtime_monitor.py",
            "realtime_trainer.py",
            "XGBoostTradingAlgorithm_Final.py",
            "docker-compose.yml",
            "requirements.txt"
        ]

        return all(Path(f).exists() for f in required_files)

    def test_monitor_tables(self) -> bool:
        """Test if monitor is configured for correct tables."""
        monitor_file = Path("realtime_monitor.py")
        if not monitor_file.exists():
            return False

        content = monitor_file.read_text()

        # Check for required tables
        required_tables = [
            'cg_spot_price_history',
            'cg_funding_rate_history',
            'cg_futures_basis_history'
        ]

        return all(table in content for table in required_tables)

    def test_training_pipeline(self) -> bool:
        """Test if training pipeline is properly configured."""
        trainer_file = Path("realtime_trainer.py")
        if not trainer_file.exists():
            return False

        content = trainer_file.read_text()

        # Check for key components
        required_components = [
            "def train_model_incremental",
            "def prepare_training_data",
            "def save_model",
            "xgb.XGBClassifier"
        ]

        return all(comp in content for comp in required_components)

    def generate_report(self):
        """Generate test report."""
        logger.info("\n" + "="*50)
        logger.info("SYSTEM TEST REPORT")
        logger.info("="*50)

        passed = sum(1 for _, status, _ in self.test_results if status == "PASSED")
        failed = sum(1 for _, status, _ in self.test_results if status in ["FAILED", "ERROR"])
        total = len(self.test_results)

        logger.info(f"\nTotal Tests: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success Rate: {(passed/total*100):.1f}%")

        logger.info("\nDetailed Results:")
        logger.info("-"*50)

        for test_name, status, error in self.test_results:
            status_symbol = "✅" if status == "PASSED" else "❌"
            logger.info(f"{status_symbol} {test_name}: {status}")
            if error:
                logger.info(f"   Error: {error}")

        # Save report to file
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "success_rate": passed/total*100,
            "results": [
                {
                    "test": name,
                    "status": status,
                    "error": error
                }
                for name, status, error in self.test_results
            ]
        }

        report_file = Path("../logs/system_test_report.json")
        report_file.parent.mkdir(exist_ok=True)

        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"\n📄 Report saved to: {report_file}")

    def run_all_tests(self):
        """Run all system tests."""
        logger.info("Starting XGBoost Trading System Tests")
        logger.info("=====================================")

        # Check if API is running
        try:
            response = requests.get(f"{self.api_base_url}/", timeout=5)
            logger.info("✅ API is accessible")
        except:
            logger.error("❌ API is not running. Please start the system first with ./deploy.sh")
            return False

        # Run all tests
        tests = [
            ("API Health Check", self.test_api_health),
            ("Database Connection", self.test_database_connection),
            ("API Prediction Endpoint", self.test_api_prediction),
            ("API Signal Generation", self.test_api_signal),
            ("File Structure", self.test_file_structure),
            ("QuantConnect Integration", self.test_quantconnect_integration),
            ("Monitor Configuration", self.test_monitor_tables),
            ("Training Pipeline", self.test_training_pipeline)
        ]

        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
            time.sleep(1)  # Small delay between tests

        # Generate report
        self.generate_report()

        # Return overall success
        return all(status == "PASSED" for _, status, _ in self.test_results)

def main():
    """Main function."""
    print("🧪 XGBoost Trading System - End-to-End Test")
    print("=" * 50)

    # Check environment variables
    if not os.getenv('TRADING_DB_HOST'):
        print("⚠️ Warning: TRADING_DB_HOST not set. Some tests may fail.")

    # Run tests
    tester = SystemTester()
    success = tester.run_all_tests()

    if success:
        print("\n🎉 All tests passed! System is ready for production.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the report and fix issues.")
        sys.exit(1)

if __name__ == "__main__":
    main()