#!/usr/bin/env python3
"""
Multi-day pipeline runner for XGBoost trading model.
Loads and processes data for specified date ranges with daily updates.
"""

import os
import sys
import subprocess
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Dict
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/xgboost_multi_day_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MultiDayRunner:
    """Handle multi-day pipeline execution for data range processing."""

    def __init__(self, config_file: str = None):
        self.config_file = config_file or os.getenv('CONFIG_FILE', './multi_day_config.json')
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
            "pairs": ["BTCUSDT"],
            "intervals": ["1h"],
            "output_dir": "./output_train",
            "mode": "initial",  # Use initial mode for date range
            "date_range": {
                "start_date": None,  # Format: YYYY-MM-DD or None for auto
                "end_date": None,    # Format: YYYY-MM-DD or None for today
                "days_back": 7       # Default: last 7 days
            },
            "processing": {
                "incremental": True,    # Process incremental (add to existing)
                "update_model": True,   # Retrain model after each day
                "evaluation": True     # Run evaluation after training
            },
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

    def get_date_range(self) -> List[str]:
        """Get list of dates to process in YYYY-MM-DD format."""
        date_config = self.config.get('date_range', {})

        # Determine end date
        if date_config.get('end_date'):
            end_date = datetime.strptime(date_config['end_date'], '%Y-%m-%d')
        else:
            end_date = datetime.now().date()

        # Determine start date
        if date_config.get('start_date'):
            start_date = datetime.strptime(date_config['start_date'], '%Y-%m-%d').date()
        else:
            # Use days_back
            days_back = date_config.get('days_back', 7)
            start_date = end_date - timedelta(days=days_back)

        # Generate date list
        dates = []
        current_date = start_date

        while current_date <= end_date:
            dates.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)

        return dates

    def run_script(self, script: str, exchange: str, pair: str, interval: str,
                  date_str: str, output_subdir: str) -> Tuple[bool, str]:
        """Run a single script with given parameters and date."""

        # Convert date string to timestamp range for the specific day
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        day_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = target_date.replace(hour=23, minute=59, second=59, microsecond=999999)

        # Convert to UTC timestamp (milliseconds)
        start_timestamp = int(day_start.timestamp() * 1000)
        end_timestamp = int(day_end.timestamp() * 1000)

        cmd = [
            'python', script,
            '--exchange', exchange,
            '--pair', pair,
            '--interval', interval,
            '--time', f'{start_timestamp},{end_timestamp}',
            '--mode', self.config['mode'],
            '--output-dir', output_subdir
        ]

        try:
            logger.info(f"Running: {' '.join(cmd)}")
            logger.info(f"Processing date: {date_str} ({day_start.strftime('%Y-%m-%d %H:%M')} to {day_end.strftime('%Y-%m-%d %H:%M')})")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout per script
            )

            if result.returncode == 0:
                logger.info(f"✅ {script} completed successfully for {date_str}")
                return True, result.stdout
            else:
                logger.error(f"❌ {script} failed for {date_str} with code {result.returncode}")
                logger.error(f"Error output: {result.stderr[-500:]}")  # Last 500 chars
                return False, result.stderr

        except subprocess.TimeoutExpired:
            logger.error(f"❌ {script} timed out for {date_str}")
            return False, "Script timed out after 1 hour"
        except Exception as e:
            logger.error(f"❌ Error running {script} for {date_str}: {str(e)}")
            return False, str(e)

    def process_date(self, date_str: str):
        """Process all data for a single date."""
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing date: {date_str}")
        logger.info(f"{'='*50}")

        # Create subdirectory for this date
        date_subdir = Path(self.config['output_dir']) / f"date_{date_str.replace('-', '')}"
        date_subdir.mkdir(exist_ok=True)

        # Process each pair and interval combination
        for pair in self.config['pairs']:
            for interval in self.config['intervals']:
                logger.info(f"\nProcessing {pair}-{interval} for {date_str}")
                logger.info("-" * 40)

                # Step 1-3: Load, merge, and feature engineering
                data_loaded = True
                for script in self.pipeline_scripts[:3]:  # Only load, merge, feature
                    success, output = self.run_script(
                        script,
                        self.config['exchange'],
                        pair,
                        interval,
                        date_str,
                        str(date_subdir)
                    )

                    if not success:
                        data_loaded = False
                        logger.error(f"Failed at {script} for {pair}-{interval} on {date_str}")
                        break

                if not data_loaded:
                    continue

                # Step 4: Build labels with existing features
                if self.config['processing']['update_model']:
                    # Load all processed data up to this date for training
                    self.build_labels_and_train(date_subdir, pair, interval, date_str)

    def build_labels_and_train(self, date_subdir: Path, pair: str, interval: str, current_date: str):
        """Build labels and train model with all available data."""
        logger.info(f"Building labels and training model for {pair}-{interval}")

        # Get all processed data directories
        base_dir = Path(self.config['output_dir'])
        all_date_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('date_')])

        if len(all_date_dirs) < 2:
            logger.warning("Not enough data directories for training (need at least 2 days)")
            return

        # Combine data from all date directories
        combined_data = []
        for date_dir in all_date_dirs:
            merged_file = date_dir / 'merged_7_tables.parquet'
            if merged_file.exists():
                logger.info(f"Loading data from {date_dir.name}")
                # Load and append to combined dataset
                # Implementation would depend on your data structure

        # Run remaining pipeline steps
        output_dir = base_dir / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(exist_ok=True)

        # Copy latest merged data
        latest_merged = max(all_date_dirs, key=lambda x: x.stat().st_mtime) / 'merged_7_tables.parquet'
        import shutil
        shutil.copy2(latest_merged, output_dir / 'merged_7_tables.parquet')

        # Run remaining scripts
        for script in self.pipeline_scripts[3:]:  # label_builder, trainer, evaluation
            cmd = [
                'python', script,
                '--exchange', self.config['exchange'],
                '--pair', pair,
                '--interval', interval,
                '--output-dir', str(output_dir)
            ]

            try:
                logger.info(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

                if result.returncode == 0:
                    logger.info(f"✅ {script} completed")
                else:
                    logger.error(f"❌ {script} failed")
                    logger.error(result.stderr)
            except Exception as e:
                logger.error(f"Error running {script}: {e}")

    def run_pipeline(self) -> Dict:
        """Run complete pipeline for all configured dates."""
        start_time = datetime.now()
        logger.info("=" * 50)
        logger.info(f"Starting multi-day pipeline at {start_time}")
        logger.info("=" * 50)

        # Get dates to process
        dates = self.get_date_range()
        logger.info(f"Will process {len(dates)} dates: {dates[0]} to {dates[-1]}")

        results = {
            'start_time': start_time.isoformat(),
            'success': True,
            'dates_processed': [],
            'errors': []
        }

        # Process each date
        for date_str in dates:
            try:
                self.process_date(date_str)
                results['dates_processed'].append(date_str)
                logger.info(f"✅ Completed processing {date_str}")
            except Exception as e:
                logger.error(f"❌ Failed to process {date_str}: {e}")
                results['success'] = False
                results['errors'].append(f"{date_str}: {str(e)}")

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
        logger.info(f"Processed {len(results['dates_processed'])}/{len(dates)} dates")
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

            # Clean up date directories
            for date_dir in output_dir.glob('date_*'):
                if date_dir.stat().st_mtime < cutoff_time:
                    # Check if this is an old date directory
                    try:
                        date_str = date_dir.name.replace('date_', '')
                        dir_date = datetime.strptime(date_str, '%Y%m%d')
                        if (datetime.now() - dir_date).days > keep_days:
                            import shutil
                            shutil.rmtree(date_dir)
                            logger.info(f"Deleted old date directory: {date_dir}")
                    except:
                        pass

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

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def send_notification(self, results: Dict):
        """Send notification about pipeline completion."""
        webhook_url = self.config['notification'].get('webhook_url')
        if not webhook_url:
            logger.info("No webhook URL configured, skipping notification")
            return

        try:
            import requests

            message = f"""
🤖 Multi-Day XGBoost Pipeline Report
{'✅ Success' if results['success'] else '❌ Failed'}
Duration: {results['duration_seconds'] / 60:.1f} minutes
Dates processed: {len(results['dates_processed'])}/{len(self.get_date_range())}
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
        results_dir = Path(self.config['output_dir']) / 'multi_day_runs'
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"multi_day_run_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {results_file}")


def main():
    """Main function to run multi-day pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='Run multi-day XGBoost pipeline')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, help='Number of days to process')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be run without executing')
    args = parser.parse_args()

    runner = MultiDayRunner(config_file=args.config)

    # Override config with command line args
    if args.start_date or args.end_date or args.days:
        runner.config['date_range'] = {}
        if args.start_date:
            runner.config['date_range']['start_date'] = args.start_date
        if args.end_date:
            runner.config['date_range']['end_date'] = args.end_date
        if args.days:
            runner.config['date_range']['days_back'] = args.days

    if args.dry_run:
        dates = runner.get_date_range()
        print("Dry run mode - would execute:")
        print(f"Dates to process: {len(dates)}")
        print(f"Date range: {dates[0]} to {dates[-1]}")
        for pair in runner.config['pairs']:
            for interval in runner.config['intervals']:
                print(f"  - Process {pair}-{interval}")
        return

    # Run pipeline
    results = runner.run_pipeline()

    # Save results
    runner.save_results(results)

    # Exit with appropriate code
    sys.exit(0 if results['success'] else 1)


if __name__ == "__main__":
    main()