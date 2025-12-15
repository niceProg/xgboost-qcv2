#!/usr/bin/env python3
"""
Daily pipeline runner for XGBoost trading model.
Runs the complete training pipeline for current trading day (7:00 - 16:00).
"""

import os
import sys
import logging
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
import json

# Import our modules
from command_line_options import parse_arguments, validate_arguments, DataFilter
from database_storage import DatabaseStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DailyPipelineRunner:
    """Runner for daily XGBoost training pipeline."""

    def __init__(self):
        self.exchange = os.getenv('EXCHANGE', 'binance')
        self.pair = os.getenv('PAIR', 'BTCUSDT')
        self.interval = os.getenv('INTERVAL', '1h')
        self.trading_hours = os.getenv('TRADING_HOURS', '00:00-09:00')  # 7:00-16:00 WIB in UTC
        self.timezone = os.getenv('TIMEZONE', 'UTC')
        self.output_dir = os.getenv('OUTPUT_DIR', './output_train')

        # Initialize database storage
        self.db_storage = None
        if os.getenv('ENABLE_DB_STORAGE', 'true').lower() == 'true':
            try:
                self.db_storage = DatabaseStorage(storage_path=self.output_dir)
                logger.info("Database storage initialized")
            except Exception as e:
                logger.error(f"Failed to initialize database storage: {e}")

    def run_command(self, cmd, name):
        """Run a command and log the result."""
        logger.info(f"Running {name}...")
        logger.info(f"Command: {' '.join(cmd)}")

        start_time = time.time()
        try:
            # Set environment variables
            env = os.environ.copy()
            env['PYTHONPATH'] = str(Path(__file__).parent)

            # Run the command
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout per step
            )

            execution_time = time.time() - start_time

            if result.returncode == 0:
                logger.info(f"{name} completed successfully in {execution_time:.2f} seconds")
                if result.stdout:
                    logger.debug(f"Output: {result.stdout}")
                return True
            else:
                logger.error(f"{name} failed after {execution_time:.2f} seconds")
                logger.error(f"Error: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"{name} timed out after 1 hour")
            return False
        except Exception as e:
            logger.error(f"Error running {name}: {e}")
            return False

    def check_trading_hours(self):
        """Check if current time is within trading hours."""
        import pytz

        # Get timezone
        if self.timezone == 'WIB':
            tz = pytz.timezone('Asia/Jakarta')  # WIB = Asia/Jakarta
        elif self.timezone == 'UTC':
            tz = datetime.timezone.utc
        else:
            tz = None

        # Get current time
        now = datetime.now(tz) if tz else datetime.now()

        # Parse trading hours
        start_str, end_str = self.trading_hours.split('-')
        start_hour, start_min = map(int, start_str.strip().split(':'))
        end_hour, end_min = map(int, end_str.strip().split(':'))

        # Create time objects
        start_time = now.replace(hour=start_hour, minute=start_min, second=0, microsecond=0)
        end_time = now.replace(hour=end_hour, minute=end_min, second=0, microsecond=0)

        # Check if within trading hours
        if start_time <= now <= end_time:
            logger.info(f"Current time {now} is within trading hours ({self.trading_hours})")
            return True
        else:
            logger.info(f"Current time {now} is outside trading hours ({self.trading_hours})")
            return False

    def run_initial_setup(self):
        """Run initial setup for training from 2024."""
        logger.info("Running initial setup (training from 2024)...")

        # Create session for initial setup
        if self.db_storage:
            try:
                session_id = self.db_storage.create_training_session(
                    exchange_filter=[self.exchange],
                    symbol_filter=[self.pair],
                    interval_filter=[self.interval],
                    notes=f"Initial setup - historical data from 2024 for {self.pair}"
                )
                logger.info(f"Created initial setup session: {session_id}")
            except Exception as e:
                logger.error(f"Failed to create session: {e}")

        # Build commands with --mode initial
        base_cmd = [
            'python',
            '--exchange', self.exchange,
            '--pair', self.pair,
            '--interval', self.interval,
            '--mode', 'initial',
            '--timezone', self.timezone,
            '--output-dir', self.output_dir
        ]

        # Run each step
        steps = [
            (['load_database.py'] + base_cmd, 'Load Database'),
            (['merge_7_tables.py'] + base_cmd, 'Merge Tables'),
            (['feature_engineering.py'] + base_cmd, 'Feature Engineering'),
            (['label_builder.py'] + base_cmd, 'Label Building'),
            (['xgboost_trainer.py'] + base_cmd, 'Model Training'),
            (['model_evaluation_with_leverage.py'] + base_cmd, 'Model Evaluation')
        ]

        for cmd, name in steps:
            if not self.run_command(cmd, name):
                logger.error(f"Pipeline failed at {name}")
                if self.db_storage:
                    self.db_storage.update_session_status(
                        status='failed',
                        notes=f"Failed at {name}"
                    )
                return False

        # Update session status
        if self.db_storage:
            self.db_storage.update_session_status(
                status='completed',
                notes='Initial setup completed'
            )

        logger.info("Initial setup completed successfully!")
        return True

    def run_daily_pipeline(self):
        """Run daily pipeline for current trading day."""
        logger.info("Running daily pipeline...")

        # Check if within trading hours
        if not self.check_trading_hours():
            logger.warning("Not within trading hours. Skipping daily run.")
            return False

        # Create session for daily run
        if self.db_storage:
            try:
                # Get today's date for session naming
                today = datetime.now().strftime('%Y-%m-%d')
                session_id = self.db_storage.create_training_session(
                    exchange_filter=[self.exchange],
                    symbol_filter=[self.pair],
                    interval_filter=[self.interval],
                    trading_hours=self.trading_hours,
                    timezone=self.timezone,
                    notes=f"Daily run - {today} for {self.pair}"
                )
                logger.info(f"Created daily session: {session_id}")
            except Exception as e:
                logger.error(f"Failed to create session: {e}")

        # Build commands with --mode daily and trading hours
        base_cmd = [
            'python',
            '--exchange', self.exchange,
            '--pair', self.pair,
            '--interval', self.interval,
            '--mode', 'daily',
            '--trading-hours', self.trading_hours,
            '--timezone', self.timezone,
            '--output-dir', self.output_dir
        ]

        # Run each step
        steps = [
            (['load_database.py'] + base_cmd, 'Load Database'),
            (['merge_7_tables.py'] + base_cmd, 'Merge Tables'),
            (['feature_engineering.py'] + base_cmd, 'Feature Engineering'),
            (['label_builder.py'] + base_cmd, 'Label Building'),
            (['xgboost_trainer.py'] + base_cmd, 'Model Training'),
            (['model_evaluation_with_leverage.py'] + base_cmd, 'Model Evaluation')
        ]

        total_start_time = time.time()
        all_success = True

        for cmd, name in steps:
            if not self.run_command(cmd, name):
                logger.error(f"Pipeline failed at {name}")
                all_success = False
                break

        total_time = time.time() - total_start_time

        # Update session status
        if self.db_storage:
            if all_success:
                self.db_storage.update_session_status(
                    status='completed',
                    notes=f"Daily run completed in {total_time:.2f} seconds"
                )
            else:
                self.db_storage.update_session_status(
                    status='failed',
                    notes=f"Pipeline failed"
                )

        if all_success:
            logger.info(f"Daily pipeline completed successfully in {total_time:.2f} seconds!")
        else:
            logger.error("Daily pipeline failed")

        return all_success

    def run(self, mode=None):
        """Run the pipeline."""
        # Determine mode
        if mode is None:
            mode = os.getenv('PIPELINE_MODE', 'daily')  # default to daily

        logger.info(f"Starting XGBoost trading pipeline in {mode} mode")
        logger.info(f"Parameters: {self.exchange}/{self.pair}/{self.interval}")
        logger.info(f"Trading Hours: {self.trading_hours} ({self.timezone})")

        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True)

        # Run based on mode
        if mode == 'initial':
            return self.run_initial_setup()
        elif mode == 'daily':
            return self.run_daily_pipeline()
        else:
            logger.error(f"Unknown mode: {mode}")
            return False

def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Run XGBoost trading pipeline')
    parser.add_argument(
        '--mode',
        choices=['initial', 'daily'],
        help='Pipeline mode: initial (train from 2024) or daily (current day)'
    )
    parser.add_argument(
        '--check-hours',
        action='store_true',
        help='Only check if current time is within trading hours'
    )

    args = parser.parse_args()

    # Create runner
    runner = DailyPipelineRunner()

    # Check trading hours if requested
    if args.check_hours:
        if runner.check_trading_hours():
            print("Within trading hours")
            sys.exit(0)
        else:
            print("Outside trading hours")
            sys.exit(1)

    # Run pipeline
    success = runner.run(mode=args.mode)

    if success:
        logger.info("Pipeline completed successfully")
        sys.exit(0)
    else:
        logger.error("Pipeline failed")
        sys.exit(1)

if __name__ == "__main__":
    main()