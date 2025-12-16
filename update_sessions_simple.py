#!/usr/bin/env python3
"""
Simple update for xgboost_training_sessions table
"""

import os
import pymysql
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Database config
db_config = {
    'host': os.getenv('DB_HOST', '103.150.81.86'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'database': os.getenv('DB_NAME', 'xgboostqc'),
    'user': os.getenv('DB_USER', 'xgboostqc'),
    'password': os.getenv('DB_PASSWORD', '6SPxBDwXH6WyxpfT')
}

def main():
    conn = pymysql.connect(**db_config)
    cursor = conn.cursor()

    # Get the existing session
    cursor.execute(
        "SELECT session_id FROM xgboost_training_sessions "
        "WHERE status = 'data_loaded' LIMIT 1"
    )
    result = cursor.fetchone()

    if not result:
        print("No data_loaded session found")
        return

    session_id = result[0]
    print(f"Updating session: {session_id}")

    # Load latest training and performance data
    output_dir = Path('./output_train')

    # Get latest training file
    training_files = list(output_dir.glob("training_results_*.json"))
    if training_files:
        latest_training = max(training_files, key=lambda x: x.stat().st_mtime)
        with open(latest_training, 'r') as f:
            training_data = json.load(f)

    # Get latest performance file
    perf_files = list(output_dir.glob("performance_report_*.json"))
    if perf_files:
        latest_perf = max(perf_files, key=lambda x: x.stat().st_mtime)
        with open(latest_perf, 'r') as f:
            perf_data = json.load(f)

    # Prepare metrics
    metrics = training_data.get('metrics', {})

    # Build notes
    notes = "Training completed successfully.\n"
    notes += f"Model: {training_data.get('model_name', 'unknown')}\n"
    notes += f"Features: {training_data.get('feature_count', 0)}\n"
    notes += f"Samples: {training_data.get('sample_count', 0)}\n\n"

    if perf_files:
        strategy = perf_data.get('strategy_summary', {})
        perf = strategy.get('performance_metrics', {})
        trading = strategy.get('trading_activity', {})

        notes += "Performance:\n"
        notes += f"Return: {perf.get('total_return', 0):.2%}\n"
        notes += f"Sharpe: {perf.get('sharpe_ratio', 0):.2f}\n"
        notes += f"Win Rate: {perf.get('win_rate', 0):.2%}\n"
        notes += f"Trades: {trading.get('total_trades', 0)}"

        total_return = perf.get('total_return', 0.0)
        sharpe = perf.get('sharpe_ratio', 0.0)
        drawdown = perf.get('max_drawdown', 0.0)
    else:
        total_return = 0.0
        sharpe = 0.0
        drawdown = 0.0

    # Update the session
    sql = """UPDATE xgboost_training_sessions SET
        status = 'completed',
        total_samples = %s,
        feature_count = %s,
        model_version = %s,
        best_params = %s,
        test_auc = %s,
        test_accuracy = %s,
        sharpe_ratio = %s,
        total_return = %s,
        max_drawdown = %s,
        notes = %s,
        completed_at = %s
        WHERE session_id = %s"""

    params = [
        training_data.get('sample_count', 0),
        training_data.get('feature_count', 0),
        training_data.get('model_name', ''),
        json.dumps(training_data.get('parameters', {})),
        metrics.get('roc_auc', 0.0),
        metrics.get('accuracy', 0.0),
        sharpe,
        total_return,
        drawdown,
        notes,
        datetime.now(),
        session_id
    ]

    cursor.execute(sql, params)
    conn.commit()

    # Verify update
    cursor.execute(
        "SELECT status, test_auc, total_return, sharpe_ratio "
        "FROM xgboost_training_sessions WHERE session_id = %s",
        (session_id,)
    )
    updated = cursor.fetchone()

    print(f"\nUpdated successfully!")
    print(f"Status: {updated[0]}")
    print(f"Test AUC: {updated[1]}")
    print(f"Total Return: {updated[2]}")
    print(f"Sharpe Ratio: {updated[3]}")

    conn.close()

if __name__ == "__main__":
    main()