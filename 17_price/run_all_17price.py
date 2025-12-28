#!/usr/bin/env python3
"""
Run complete XGBoost 17-Price Features training pipeline.

SIMPLE PIPELINE - Price Only:
1. Load price data from database (1 table only!)
2. Train XGBoost model with 17 price features
3. Save results

No 9-table merge, no complex feature engineering.
Just OHLCV data → 17 price features → XGBoost model.
"""

import argparse
import subprocess
import sys
import logging
import os
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
    """Main function to run the complete 17-price pipeline."""
    parser = argparse.ArgumentParser(
        description="XGBoost 17-Price Features Training Pipeline - Simple & Clean",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults
  python run_all_17price.py

  # Run with filters
  python run_all_17price.py --symbol BTC --interval 1h --days 30

  # Run with custom model parameters
  python run_all_17price.py --estimators 200 --depth 6 --lr 0.05 --label threshold

Pipeline:
  1. Load price data (1 table only: cg_futures_price_history)
  2. Train XGBoost model (17 price features)
  3. Save to database (optional)
        """
    )

    # Filtering options (passed to load_price_only.py)
    parser.add_argument('--exchange', type=str, help='Exchange: binance,okx,bybit')
    parser.add_argument('--pair', type=str, help='Trading pair: BTCUSDT,ETHUSDT')
    parser.add_argument('--symbol', type=str, help='Symbol (base asset): BTC,ETH')
    parser.add_argument('--interval', type=str, help='Interval: 1m,5m,1h,4h')
    parser.add_argument('--time', type=str, help='Time range (ms): start_time,end_time')
    parser.add_argument('--days', type=int, help='Number of recent days')

    # Output options
    parser.add_argument('--output-dir', type=str, default='./output_train_17price',
                       help='Output directory (default: ./output_train_17price)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    # Model options
    parser.add_argument('--label', type=str, default='next_bar',
                       choices=['next_bar', 'threshold', '3bar'],
                       help='Label type (default: next_bar)')
    parser.add_argument('--estimators', type=int, default=200,
                       help='Number of estimators (default: 200)')
    parser.add_argument('--depth', type=int, default=6,
                       help='Max depth (default: 6)')
    parser.add_argument('--lr', type=float, default=0.05,
                       help='Learning rate (default: 0.05)')
    parser.add_argument('--no-balance', action='store_true',
                       help='Disable data balancing')

    # Step selection
    parser.add_argument('--start-from', type=str,
                       choices=['load', 'train'],
                       default='load',
                       help='Start from specific step (default: load)')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Build common arguments
    common_args = ['--output-dir', args.output_dir]
    if args.verbose:
        common_args.append('--verbose')

    # Build filter arguments for load_price_only.py
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

    # Build model training arguments
    model_args = [
        '--label', args.label,
        '--estimators', str(args.estimators),
        '--depth', str(args.depth),
        '--lr', str(args.lr),
        '--output-dir', args.output_dir,
    ]
    if args.no_balance:
        model_args.append('--no-balance')

    # Print summary
    logger.info("\n" + "="*50)
    logger.info("XGBoost 17-Price Features Training Pipeline")
    logger.info("="*50)
    logger.info(f"Output directory: {args.output_dir}")
    if args.symbol:
        logger.info(f"Symbol: {args.symbol}")
    if args.exchange:
        logger.info(f"Exchange: {args.exchange}")
    if args.pair:
        logger.info(f"Pair: {args.pair}")
    if args.interval:
        logger.info(f"Interval: {args.interval}")
    if args.days:
        logger.info(f"Days: {args.days}")
    logger.info("\n⚡ PIPELINE:")
    logger.info("  1. Load price data (1 table: cg_futures_price_history)")
    logger.info("  2. Train XGBoost model (17 price features)")
    logger.info(f"\n📊 Model Config:")
    logger.info(f"  Label Type: {args.label}")
    logger.info(f"  Estimators: {args.estimators}")
    logger.info(f"  Max Depth: {args.depth}")
    logger.info(f"  Learning Rate: {args.lr}")
    logger.info(f"  Balance Data: {not args.no_balance}")
    logger.info("="*50 + "\n")

    # Define pipeline steps
    # SIMPLE: Only 2 steps - Load price data, then train model
    pipeline_steps = [
        ('load', 'load_price_only.py', filter_args + common_args, 'Load Price Data (1 Table Only)'),
        ('train', 'train_model_17price.py', model_args, 'Train XGBoost Model (17 Price)'),
    ]

    # Find start index
    start_index = 0
    if args.start_from == 'train':
        start_index = 1

    # Run pipeline
    start_time = datetime.now()
    failed_step = None
    training_metadata = None

    for i, (step, script, script_args, step_name) in enumerate(pipeline_steps):
        if i < start_index:
            continue
        if step == args.start_from or i > start_index:
            success = run_step(script, script_args, step_name)
            if not success:
                failed_step = step_name
                break

    # Load training metadata if training was successful
    if not failed_step:
        try:
            import json
            import glob

            # Find latest metadata file
            metadata_files = glob.glob(os.path.join(args.output_dir, "metadata_17price_*.json"))
            if metadata_files:
                latest_metadata = max(metadata_files, key=os.path.getctime)
                with open(latest_metadata, 'r') as f:
                    training_metadata = json.load(f)

                logger.info("\n" + "="*50)
                logger.info("TRAINING RESULTS")
                logger.info("="*50)
                logger.info(f"Model Version: {training_metadata.get('model_version', 'N/A')}")
                logger.info(f"Feature Hash: {training_metadata.get('feature_hash', 'N/A')}")

                metrics = training_metadata.get('metrics', {})
                logger.info(f"\nTest Metrics:")
                logger.info(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
                logger.info(f"  Precision: {metrics.get('precision', 0):.4f}")
                logger.info(f"  Recall:    {metrics.get('recall', 0):.4f}")
                logger.info(f"  F1 Score:  {metrics.get('f1', 0):.4f}")
                logger.info(f"  ROC AUC:   {metrics.get('roc_auc', 0):.4f}")

                # Database saving is handled in train_model_17price.py

        except Exception as e:
            logger.warning(f"⚠️ Could not load/save metadata: {e}")

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
        logger.info(f"  Model:     {args.output_dir}/latest_model.joblib")
        logger.info(f"  Features:  {args.output_dir}/model_features_17price.txt")
        logger.info(f"  Metadata:  {args.output_dir}/metadata_17price_*.json")
        logger.info(f"  API Format:{args.output_dir}/api_format_17price_*.json")
        logger.info("\n🚀 Ready for QuantConnect backtest!")
        logger.info("\n📊 Next steps:")
        logger.info("  1. Upload api_format_*.json to dragonfortune.ai API")
        logger.info("  2. Update qc_futures.py with 17 price features")
        logger.info("  3. Run backtest on QuantConnect")

    return 0 if not failed_step else 1


if __name__ == "__main__":
    sys.exit(main())
