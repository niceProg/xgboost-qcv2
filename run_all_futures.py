#!/usr/bin/env python3
"""
Run complete XGBoost futures training pipeline.
This script executes all steps: load data, merge tables, feature engineering, label building, and training.
"""

import argparse
import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_step(script_name: str, args: list, step_name: str) -> bool:
    """Run a single step in the pipeline."""
    logger.info(f"\n{'='*50}")
    logger.info(f"Starting: {step_name}")
    logger.info(f"{'='*50}")

    cmd = [sys.executable, script_name] + args
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        logger.info(f"✅ {step_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {step_name} failed with error code {e.returncode}")
        return False
    except FileNotFoundError:
        logger.error(f"❌ Script not found: {script_name}")
        return False


def main():
    """Main function to run the complete pipeline."""
    parser = argparse.ArgumentParser(
        description="XGBoost Futures Training Pipeline - Run All Steps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults
  python run_all_futures.py

  # Run with filters
  python run_all_futures.py --exchange binance --symbol BTC --interval 1h --days 30

  # Run with specific time range
  python run_all_futures.py --symbol BTC,ETH --interval 1h --time 1704067200000,1706745600000
        """
    )

    # Filtering options (passed to load_database_futures.py)
    parser.add_argument('--exchange', type=str, help='Exchange name(s): binance,okx,bybit')
    parser.add_argument('--pair', type=str, help='Trading pair(s): BTCUSDT,ETHUSDT')
    parser.add_argument('--symbol', type=str, help='Symbol(s): BTC,ETH')
    parser.add_argument('--interval', type=str, help='Time interval(s): 1m,5m,1h,4h')
    parser.add_argument('--time', type=str, help='Time range (ms): start_time,end_time')
    parser.add_argument('--days', type=int, help='Number of recent days')

    # Output options
    parser.add_argument('--output-dir', type=str, default='./output_train_futures',
                       help='Output directory (default: ./output_train_futures)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    # Step selection
    parser.add_argument('--start-from', type=str,
                       choices=['load', 'merge', 'features', 'labels', 'train'],
                       default='load',
                       help='Start from specific step (default: load)')
    parser.add_argument('--skip-to', type=str,
                       choices=['load', 'merge', 'features', 'labels', 'train'],
                       help='Skip to specific step')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Build common arguments for all scripts
    common_args = ['--output-dir', args.output_dir]
    if args.verbose:
        common_args.append('--verbose')

    # Build filter arguments for load_database_futures.py
    filter_args = []
    if args.exchange:
        filter_args.extend(['--exchange', args.exchange])
    if args.pair:
        filter_args.extend(['--pair', args.pair])
    if args.symbol:
        filter_args.extend(['--symbol', args.symbol])
    if args.interval:
        filter_args.extend(['--interval', args.interval])
    if args.time:
        filter_args.extend(['--time', args.time])
    if args.days:
        filter_args.extend(['--days', args.days])

    # Print summary
    logger.info("\n" + "="*50)
    logger.info("XGBoost Futures Training Pipeline")
    logger.info("="*50)
    logger.info(f"Output directory: {args.output_dir}")
    if args.exchange:
        logger.info(f"Exchange: {args.exchange}")
    if args.symbol:
        logger.info(f"Symbol: {args.symbol}")
    if args.pair:
        logger.info(f"Pair: {args.pair}")
    if args.interval:
        logger.info(f"Interval: {args.interval}")
    if args.days:
        logger.info(f"Days: {args.days}")
    logger.info("="*50 + "\n")

    # Define pipeline steps
    pipeline_steps = [
        ('load', 'load_database_futures.py', filter_args + common_args, 'Load Data from Database'),
        ('merge', 'merge_futures_9_tables.py', common_args, 'Merge 9 Futures Tables'),
        ('features', 'feature_engineering_futures.py', common_args, 'Feature Engineering'),
        ('labels', 'label_builder_futures.py', common_args, 'Build Labels'),
        ('train', 'xgboost_trainer_futures.py', common_args, 'Train XGBoost Model'),
    ]

    # Find start index
    start_index = 0
    if args.skip_to:
        for i, (step, _, _, _) in enumerate(pipeline_steps):
            if step == args.skip_to:
                start_index = i
                break

    # Run pipeline
    start_time = datetime.now()
    failed_step = None

    for i, (step, script, script_args, step_name) in enumerate(pipeline_steps):
        if i < start_index:
            continue
        if step == args.start_from or i > start_index:
            success = run_step(script, script_args, step_name)
            if not success:
                failed_step = step_name
                break

    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time

    logger.info("\n" + "="*50)
    if failed_step:
        logger.error(f"❌ Pipeline failed at: {failed_step}")
    else:
        logger.info("✅ Pipeline completed successfully!")
    logger.info(f"Total duration: {duration}")
    logger.info("="*50)

    if not failed_step:
        logger.info("\n📦 Output files:")
        logger.info(f"  Model: {args.output_dir}/latest_futures_model.joblib")
        logger.info(f"  Features: {args.output_dir}/model_features_futures.txt")
        logger.info(f"  Results: {args.output_dir}/datasets/summary/")
        logger.info("\n🚀 Ready for QuantConnect backtest!")

    return 0 if not failed_step else 1


if __name__ == "__main__":
    sys.exit(main())
