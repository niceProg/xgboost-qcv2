#!/usr/bin/env python3
"""
Daily pipeline runner for XGBoost trading model.
This script runs the complete pipeline with daily mode to update the model.
"""

import os
import sys
import subprocess
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/xgboost_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DailyRunner:
    """Handle daily pipeline execution."""

    def __init__(self, config_file: str = None):
        self.config_file = config_file or os.getenv('CONFIG_FILE', './daily_config.json')
        self.config = self.load_config()
        self.pipeline_scripts = [
            'load_database.py',
            'merge_7_tables.py',
            'feature_engineering.py',
            'label_builder.py',
            'xgboost_trainer.py',
            'model_evaluation_with_leverage.py'
        ]

    def load_config(self) -> Dict:
        """Load configuration from JSON file."""
        default_config = {
            "exchange": "binance",
            "pairs": ["BTCUSDT", "ETHUSDT"],
            "intervals": ["1h", "4h"],
            "timezone": "WIB",
            "trading_hours": "7:00-16:00",
            "output_dir": "./output_train",
            "mode": "daily",
            "notification": {
                "enabled": False,
                "webhook_url": None
            },
            "cleanup": {
                "enabled": True,
                "keep_days": 30
            }
        }

        if Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)
                # Merge with defaults
                for key in default_config:
                    if key not in user_config:
                        user_config[key] = default_config[key]
                logger.info(f"Loaded config from {self.config_file}")
                return user_config
            except Exception as e:
                logger.error(f"Error loading config: {e}. Using defaults.")
                return default_config
        else:
            # Create default config file
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Created default config at {self.config_file}")
            return default_config

    def run_script(self, script: str, exchange: str, pair: str, interval: str) -> Tuple[bool, str]:
        """Run a single script with given parameters."""
        cmd = [
            'python', script,
            '--exchange', exchange,
            '--pair', pair,
            '--interval', interval,
            '--mode', self.config['mode'],
            '--timezone', self.config['timezone'],
            '--trading-hours', self.config['trading_hours'],
            '--output-dir', self.config['output_dir']
        ]

        try:
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout per script
            )

            if result.returncode == 0:
                logger.info(f"✅ {script} completed successfully")
                return True, result.stdout
            else:
                logger.error(f"❌ {script} failed with code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                return False, result.stderr

        except subprocess.TimeoutExpired:
            logger.error(f"❌ {script} timed out")
            return False, "Script timed out after 1 hour"
        except Exception as e:
            logger.error(f"❌ Error running {script}: {str(e)}")
            return False, str(e)

    def run_pipeline(self) -> Dict:
        """Run complete pipeline for all configured pairs and intervals."""
        start_time = datetime.now()
        logger.info("=" * 50)
        logger.info(f"Starting daily pipeline at {start_time}")
        logger.info("=" * 50)

        results = {
            'start_time': start_time.isoformat(),
            'success': True,
            'pairs_processed': [],
            'errors': []
        }

        # Run pipeline for each pair and interval combination
        for pair in self.config['pairs']:
            for interval in self.config['intervals']:
                pair_result = {
                    'pair': pair,
                    'interval': interval,
                    'scripts': {}
                }

                logger.info(f"\nProcessing {pair} - {interval}")
                logger.info("-" * 40)

                # Run each script in sequence
                for script in self.pipeline_scripts:
                    success, output = self.run_script(
                        script,
                        self.config['exchange'],
                        pair,
                        interval
                    )

                    pair_result['scripts'][script] = {
                        'success': success,
                        'output': output[-500:] if output else ""  # Keep last 500 chars
                    }

                    if not success:
                        results['success'] = False
                        results['errors'].append(f"{script} failed for {pair}-{interval}: {output[-200:]}")
                        logger.error(f"Stopping pipeline for {pair}-{interval} due to {script} failure")
                        break

                results['pairs_processed'].append(pair_result)

        # Cleanup old files if enabled
        if self.config['cleanup']['enabled']:
            self.cleanup_old_files()

        # Calculate total time
        end_time = datetime.now()
        duration = end_time - start_time
        results['end_time'] = end_time.isoformat()
        results['duration_seconds'] = duration.total_seconds()

        logger.info("=" * 50)
        logger.info(f"Pipeline completed in {duration}")
        logger.info(f"Success: {results['success']}")
        logger.info("=" * 50)

        # Send notification if enabled
        if self.config['notification']['enabled']:
            self.send_notification(results)

        return results

    def cleanup_old_files(self):
        """Clean up old output files to save disk space."""
        logger.info("Cleaning up old files...")
        output_dir = Path(self.config['output_dir'])
        keep_days = self.config['cleanup']['keep_days']

        try:
            cutoff_time = datetime.now().timestamp() - (keep_days * 24 * 3600)

            # Clean up model files
            for model_file in output_dir.glob("xgboost_trading_model_*.joblib"):
                if model_file.stat().st_mtime < cutoff_time:
                    model_file.unlink()
                    logger.info(f"Deleted old model: {model_file}")

            # Clean up result JSON files
            for json_file in output_dir.glob("*results_*.json"):
                if json_file.stat().st_mtime < cutoff_time:
                    json_file.unlink()
                    logger.info(f"Deleted old result: {json_file}")

            # Clean up old parquet files except the latest
            for pattern in ["*.parquet", "*.csv"]:
                files = list(output_dir.glob(pattern))
                files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                for old_file in files[5:]:  # Keep latest 5 files
                    old_file.unlink()
                    logger.info(f"Deleted old file: {old_file}")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def send_notification(self, results: Dict):
        """Send notification about pipeline completion."""
        webhook_url = self.config['notification'].get('webhook_url')
        if not webhook_url:
            return

        try:
            import requests

            message = f"""
🤖 XGBoost Daily Pipeline Report
{'✅ Success' if results['success'] else '❌ Failed'}
Duration: {results['duration_seconds'] / 60:.1f} minutes
Pairs processed: {len(results['pairs_processed'])}
            """

            if results['errors']:
                message += "\n\nErrors:\n" + "\n".join(results['errors'][:3])

            payload = {"text": message}
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            logger.info("Notification sent successfully")

        except Exception as e:
            logger.error(f"Failed to send notification: {e}")

    def save_results(self, results: Dict):
        """Save pipeline results to file."""
        results_dir = Path(self.config['output_dir']) / 'daily_runs'
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"daily_run_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {results_file}")


def main():
    """Main function to run daily pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='Run daily XGBoost pipeline')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be run without executing')
    args = parser.parse_args()

    runner = DailyRunner(config_file=args.config)

    if args.dry_run:
        print("Dry run mode - would execute:")
        for pair in runner.config['pairs']:
            for interval in runner.config['intervals']:
                print(f"  Pipeline for {pair}-{interval}")
        return

    # Run pipeline
    results = runner.run_pipeline()

    # Save results
    runner.save_results(results)

    # Exit with appropriate code
    sys.exit(0 if results['success'] else 1)


if __name__ == "__main__":
    main()