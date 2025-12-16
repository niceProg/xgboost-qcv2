#!/usr/bin/env python3
"""
Simplified script to fix xgboost_training_sessions table
"""

import os
import pymysql
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database config
db_config = {
    'host': os.getenv('DB_HOST', '103.150.81.86'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'database': os.getenv('DB_NAME', 'xgboostqc'),
    'user': os.getenv('DB_USER', 'xgboostqc'),
    'password': os.getenv('DB_PASSWORD', '6SPxBDwXH6WyxpfT')
}

def update_session(session_id, training_file=None, perf_file=None):
    """Update single session with training and performance data"""
    conn = pymysql.connect(**db_config)
    cursor = conn.cursor()

    print(f"\n=== Updating Session: {session_id} ===")

    # Check if session exists
    cursor.execute("SELECT id FROM xgboost_training_sessions WHERE session_id = %s", (session_id,))
    exists = cursor.fetchone()

    if not exists:
        print(f"⚠️ Session {session_id} not found in database")
        conn.close()
        return

    # Load training results if provided
    if training_file and Path(training_file).exists():
        with open(training_file, 'r') as f:
            training_data = json.load(f)

        metrics = training_data.get('metrics', {})
        cv_results = training_data.get('cross_validation', {})

        # Update with training metrics
        sql = """
        UPDATE xgboost_training_sessions SET
            status = 'completed',
            total_samples = %s,
            feature_count = %s,
            model_version = %s,
            best_params = %s,
            model_path = %s,
            train_auc = %s,
            val_auc = %s,
            test_auc = %s,
            test_accuracy = %s,
            test_precision = %s,
            test_recall = %s,
            test_f1 = %s,
            completed_at = %s
        WHERE session_id = %s
        """

        params = [
            training_data.get('sample_count', 0),
            training_data.get('feature_count', 0),
            training_data.get('model_name', ''),
            json.dumps(training_data.get('parameters', {})),
            f"./output_train/{training_data.get('model_name', '')}",
            metrics.get('train_auc', 0.0),
            metrics.get('val_auc', 0.0),
            metrics.get('roc_auc', 0.0),
            metrics.get('accuracy', 0.0),
            metrics.get('precision', 0.0),
            metrics.get('recall', 0.0),
            metrics.get('f1', 0.0),
            datetime.now(),
            session_id
        ]

        cursor.execute(sql, params)
        print(f"✅ Updated training metrics for {session_id}")

    # Load performance results if provided
    if perf_file and Path(perf_file).exists():
        with open(perf_file, 'r') as f:
            perf_data = json.load(f)

        strategy_summary = perf_data.get('strategy_summary', {})
        performance_metrics = strategy_summary.get('performance_metrics', {})
        trading_activity = strategy_summary.get('trading_activity', {})

        # Create comprehensive notes
        notes = f"""Training completed successfully.
Performance Metrics:
- Total Return: {performance_metrics.get('total_return', 0):.2%}
- Sharpe Ratio: {performance_metrics.get('sharpe_ratio', 0):.2f}
- Max Drawdown: {performance_metrics.get('max_drawdown', 0):.2%}
- Win Rate: {performance_metrics.get('win_rate', 0):.2%}
- Total Trades: {trading_activity.get('total_trades', 0)}
Trading Period: {strategy_summary.get('trading_period', {}).get('start_date', '')} to {strategy_summary.get('trading_period', {}).get('end_date', '')}"""

        # Update with performance metrics
        sql = """
        UPDATE xgboost_training_sessions SET
            total_return = %s,
            sharpe_ratio = %s,
            max_drawdown = %s,
            notes = %s
        WHERE session_id = %s
        """

        params = [
            performance_metrics.get('total_return', 0.0),
            performance_metrics.get('sharpe_ratio', 0.0),
            performance_metrics.get('max_drawdown', 0.0),
            notes,
            session_id
        ]

        cursor.execute(sql, params)
        print(f"✅ Updated performance metrics for {session_id}")

    conn.commit()
    conn.close()

def main():
    """Main function - find and update all sessions"""
    output_dir = Path('./output_train')

    # Find all training result files
    training_files = sorted(output_dir.glob("training_results_*.json"), reverse=True)
    perf_files = sorted(output_dir.glob("performance_report_*.json"), reverse=True)

    print(f"Found {len(training_files)} training files and {len(perf_files)} performance files")

    # Update each session
    for training_file in training_files:
        # Extract session ID from filename
        session_id = training_file.stem.replace('training_results_', '')

        # Find corresponding performance file
        perf_file = None
        for pf in perf_files:
            if session_id in pf.name:
                perf_file = pf
                break

        update_session(session_id, training_file, perf_file)

    print("\n=== Done! ===")

    # Show final result
    conn = pymysql.connect(**db_config)
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    cursor.execute("SELECT * FROM xgboost_training_sessions ORDER BY created_at DESC")
    records = cursor.fetchall()

    print("\n=== Updated Sessions ===")
    for record in records:
        print(f"\nSession: {record['session_id']}")
        print(f"  Status: {record['status']}")
        print(f"  Total Samples: {record['total_samples']}")
        print(f"  Feature Count: {record['feature_count']}")
        print(f"  Model Version: {record['model_version']}")
        print(f"  Test AUC: {record['test_auc']:.4f}")
        print(f"  Total Return: {record['total_return']:.4f}")
        print(f"  Sharpe Ratio: {record['sharpe_ratio']:.4f}")
        print(f"  Max Drawdown: {record['max_drawdown']:.4f}")
        print(f"  Completed At: {record['completed_at']}")

    conn.close()

if __name__ == "__main__":
    main()