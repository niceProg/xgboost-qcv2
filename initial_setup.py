#!/usr/bin/env python3
"""
Initial setup script for XGBoost trading model.
Runs the complete pipeline with historical data from 2024.
This creates the baseline model and populates the database.
"""

import os
import sys
import logging
import subprocess
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from database_storage import DatabaseStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(cmd, name, env_vars=None):
    """Run a command and return success status."""
    logger.info(f"Running {name}...")
    logger.info(f"Command: {' '.join(cmd)}")

    # Set environment
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    env['PYTHONPATH'] = str(Path(__file__).parent)

    try:
        result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
        logger.info(f"{name} completed successfully")
        if result.stdout:
            logger.debug(f"Output: {result.stdout[-500:]}")  # Last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"{name} failed with exit code {e.returncode}")
        logger.error(f"Error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error running {name}: {e}")
        return False

def initial_setup():
    """Run initial setup for training from 2024."""
    logger.info("=" * 60)
    logger.info("INITIAL SETUP - Training from 2024")
    logger.info("=" * 60)

    # Get parameters from environment or use defaults
    exchange = os.getenv('EXCHANGE', 'binance')
    pair = os.getenv('PAIR', 'BTCUSDT')
    interval = os.getenv('INTERVAL', '1h')
    output_dir = os.getenv('OUTPUT_DIR', './output_train')
    trading_hours = os.getenv('TRADING_HOURS', '00:00-09:00')  # 7:00-16:00 WIB in UTC
    timezone = os.getenv('TIMEZONE', 'UTC')

    logger.info(f"Parameters: {exchange}/{pair}/{interval}")
    logger.info(f"Trading Hours: {trading_hours} ({timezone})")
    logger.info(f"Output Directory: {output_dir}")

    # Initialize database storage
    db_storage = None
    session_id = None
    try:
        db_storage = DatabaseStorage(storage_path=output_dir)
        session_id = db_storage.create_training_session(
            exchange_filter=[exchange],
            symbol_filter=[pair],
            interval_filter=[interval],
            trading_hours=trading_hours,
            timezone=timezone,
            notes=f"Initial setup - historical data from 2024 for {pair}"
        )
        logger.info(f"Created training session: {session_id}")
    except Exception as e:
        logger.warning(f"Database storage not available: {e}")
        db_storage = None

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Step 1: Setup database if needed
    logger.info("\n=== Step 1: Database Setup ===")
    if db_storage:
        logger.info("Database already initialized via DatabaseStorage")
    else:
        # Try to run setup script
        if Path('setup_database.py').exists():
            if not run_command(['python', 'setup_database.py'], 'Database Setup'):
                logger.error("Database setup failed")
                return False
        else:
            logger.warning("setup_database.py not found - skipping database setup")

    # Step 2: Load historical data
    logger.info("\n=== Step 2: Loading Historical Data ===")
    load_cmd = [
        'python', 'load_database.py',
        '--exchange', exchange,
        '--pair', pair,
        '--interval', interval,
        '--mode', 'initial',
        '--timezone', timezone,
        '--output-dir', output_dir,
        '--verbose'
    ]
    if not run_command(load_cmd, 'Load Historical Data'):
        logger.error("Failed to load historical data")
        if db_storage:
            db_storage.update_session_status(
                status='failed',
                notes='Failed to load historical data'
            )
        return False

    # Step 3: Merge tables
    logger.info("\n=== Step 3: Merging Tables ===")
    merge_cmd = [
        'python', 'merge_7_tables.py',
        '--exchange', exchange,
        '--pair', pair,
        '--interval', interval,
        '--mode', 'initial',
        '--timezone', timezone,
        '--output-dir', output_dir
    ]
    if not run_command(merge_cmd, 'Merge Tables'):
        logger.error("Failed to merge tables")
        if db_storage:
            db_storage.update_session_status(
                status='failed',
                notes='Failed to merge tables'
            )
        return False

    # Step 4: Feature engineering
    logger.info("\n=== Step 4: Feature Engineering ===")
    feature_cmd = [
        'python', 'feature_engineering.py',
        '--exchange', exchange,
        '--pair', pair,
        '--interval', interval,
        '--mode', 'initial',
        '--timezone', timezone,
        '--output-dir', output_dir
    ]
    if not run_command(feature_cmd, 'Feature Engineering'):
        logger.error("Failed to engineer features")
        if db_storage:
            db_storage.update_session_status(
                status='failed',
                notes='Failed to engineer features'
            )
        return False

    # Step 5: Label building
    logger.info("\n=== Step 5: Label Building ===")
    label_cmd = [
        'python', 'label_builder.py',
        '--exchange', exchange,
        '--pair', pair,
        '--interval', interval,
        '--mode', 'initial',
        '--timezone', timezone,
        '--output-dir', output_dir
    ]
    if not run_command(label_cmd, 'Label Building'):
        logger.error("Failed to build labels")
        if db_storage:
            db_storage.update_session_status(
                status='failed',
                notes='Failed to build labels'
            )
        return False

    # Step 6: Model training
    logger.info("\n=== Step 6: Model Training ===")
    train_cmd = [
        'python', 'xgboost_trainer.py',
        '--exchange', exchange,
        '--pair', pair,
        '--interval', interval,
        '--mode', 'initial',
        '--timezone', timezone,
        '--output-dir', output_dir
    ]
    if not run_command(train_cmd, 'Model Training'):
        logger.error("Failed to train model")
        if db_storage:
            db_storage.update_session_status(
                status='failed',
                notes='Failed to train model'
            )
        return False

    # Step 7: Model evaluation
    logger.info("\n=== Step 7: Model Evaluation ===")
    eval_cmd = [
        'python', 'model_evaluation_with_leverage.py',
        '--exchange', exchange,
        '--pair', pair,
        '--interval', interval,
        '--mode', 'initial',
        '--timezone', timezone,
        '--output-dir', output_dir
    ]
    if not run_command(eval_cmd, 'Model Evaluation'):
        logger.error("Failed to evaluate model")
        if db_storage:
            db_storage.update_session_status(
                status='failed',
                notes='Failed to evaluate model'
            )
        return False

    # Success!
    logger.info("\n" + "=" * 60)
    logger.info("INITIAL SETUP COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)

    if db_storage:
        try:
            # Load evaluation results to update session
            eval_file = Path(output_dir) / 'performance_metrics_*.json'
            import glob
            eval_files = glob.glob(str(eval_file))
            if eval_files:
                import json
                with open(eval_files[-1], 'r') as f:
                    metrics = json.load(f)

                db_storage.update_session_status(
                    status='completed',
                    notes='Initial setup completed - baseline model ready',
                    metrics=metrics
                )
        except Exception as e:
            logger.warning(f"Failed to update final session status: {e}")

    logger.info("\nNext steps:")
    logger.info("1. Start the scheduler: python scheduler.py")
    logger.info("2. Or run Docker: docker-compose up -d")
    logger.info("3. Access API: http://localhost:8000/api/v1/")

    return True

def check_prerequisites():
    """Check if prerequisites are met."""
    logger.info("Checking prerequisites...")

    # Check if required files exist
    required_files = [
        'load_database.py',
        'merge_7_tables.py',
        'feature_engineering.py',
        'label_builder.py',
        'xgboost_trainer.py',
        'model_evaluation_with_leverage.py',
        'command_line_options.py'
    ]

    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)

    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False

    # Check Python packages
    try:
        import pandas
        import numpy
        import xgboost
        import sqlalchemy
        import pymysql
        logger.info("All required Python packages are installed")
    except ImportError as e:
        logger.error(f"Missing Python package: {e}")
        logger.info("Run: pip install -r requirements.txt")
        return False

    logger.info("Prerequisites check passed")
    return True

def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Initial setup for XGBoost trading model')
    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Skip prerequisite checks'
    )
    parser.add_argument(
        '--exchange',
        type=str,
        default='binance',
        help='Exchange name'
    )
    parser.add_argument(
        '--pair',
        type=str,
        default='BTCUSDT',
        help='Trading pair'
    )
    parser.add_argument(
        '--interval',
        type=str,
        default='1h',
        help='Time interval'
    )

    args = parser.parse_args()

    # Set environment variables
    os.environ['EXCHANGE'] = args.exchange
    os.environ['PAIR'] = args.pair
    os.environ['INTERVAL'] = args.interval

    # Check prerequisites
    if not args.skip_checks:
        if not check_prerequisites():
            sys.exit(1)

    # Run initial setup
    success = initial_setup()

    if success:
        logger.info("\n✅ Initial setup completed successfully!")
        logger.info("You can now start the daily scheduler.")
        sys.exit(0)
    else:
        logger.error("\n❌ Initial setup failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()