#!/usr/bin/env python3
"""
Monitor and debug pipeline run to ensure CSV saving works
"""

import os
import sys
import time
import subprocess
import pymysql
from pathlib import Path
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

db_config = {
    'host': os.getenv('DB_HOST', '103.150.81.86'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'database': os.getenv('DB_NAME', 'xgboostqc'),
    'user': os.getenv('DB_USER', 'xgboostqc'),
    'password': os.getenv('DB_PASSWORD', '6SPxBDwXH6WyxpfT')
}

def get_db_count(table_name):
    """Get record count from table"""
    try:
        conn = pymysql.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except:
        return None

def monitor_pipeline():
    """Monitor pipeline execution"""

    print("="*80)
    print("MONITORING XGBOOST PIPELINE RUN")
    print("="*80)

    # Check initial state
    print("\n📊 Initial Database State:")
    tables_to_monitor = [
        'xgboost_training_sessions',
        'xgboost_models',
        'xgboost_features',
        'xgboost_evaluations',
        'xgboost_trades',
        'xgboost_account_statements'
    ]

    for table in tables_to_monitor:
        count = get_db_count(table)
        status = "N/A" if count is None else str(count)
        print(f"  {table:30}: {status}")

    print("\n🚀 Starting pipeline...")

    # Start the pipeline
    process = subprocess.Popen(
        ["bash", "simple_run.sh"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy()
    )

    # Monitor output
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

            # Check for specific step completion
            if "Step 1: Load Database completed" in output:
                logger.info("✅ Load Database completed - Checking records...")
                time.sleep(1)
                count = get_db_count('xgboost_training_sessions')
                logger.info(f"  Sessions: {count}")

            elif "Step 5: Model Training completed" in output:
                logger.info("✅ Model Training completed - Checking records...")
                time.sleep(1)
                model_count = get_db_count('xgboost_models')
                logger.info(f"  Models: {model_count}")

            elif "Step 6: Model Evaluation completed" in output:
                logger.info("✅ Model Evaluation completed - Checking records...")
                time.sleep(2)  # Give time for DB operations

                eval_count = get_db_count('xgboost_evaluations')
                trades_count = get_db_count('xgboost_trades')
                stmt_count = get_db_count('xgboost_account_statements')

                logger.info(f"  Evaluations: {eval_count}")
                logger.info(f"  Trades: {trades_count}")
                logger.info(f"  Account Statements: {stmt_count}")

                # Check CSV files
                time.sleep(1)
                output_dir = Path('./output_train')
                csv_files = ['trades.csv', 'rekening_koran.csv', 'rekening_koran_cash.csv']
                for csv_file in csv_files:
                    file_path = output_dir / csv_file
                    if file_path.exists():
                        size_mb = file_path.stat().st_size / 1024 / 1024
                        logger.info(f"  Created {csv_file}: {size_mb:.2f}MB")
                    else:
                        logger.warning(f"  Missing: {csv_file}")

    # Get final results
    return_code = process.poll()
    stderr = process.stderr.read()

    if return_code == 0:
        print("\n✅ Pipeline completed successfully!")

        # Check final database state
        print("\n📊 Final Database State:")
        for table in tables_to_monitor:
            count = get_db_count(table)
            status = "N/A" if count is None else str(count)
            print(f"  {table:30}: {status}")

        # Detailed check for trades and statements
        print("\n💰 Detailed Trade Records:")
        if get_db_count('xgboost_trades') > 0:
            conn = pymysql.connect(**db_config)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT trade_count, total_pnl, winning_trades, losing_trades
                FROM xgboost_trades
                ORDER BY created_at DESC LIMIT 1
            """)
            trade_data = cursor.fetchone()
            if trade_data:
                print(f"  Trades: {trade_data[0]}")
                print(f"  Total P&L: {trade_data[1]:.2f}")
                print(f"  Win Rate: {(trade_data[2]/trade_data[0]*100):.1f}%")
            conn.close()

    else:
        print("\n❌ Pipeline failed!")
        print("Error output:")
        print(stderr)

    return return_code

if __name__ == "__main__":
    return_code = monitor_pipeline()
    sys.exit(return_code)