#!/usr/bin/env python3
"""
Check and update xgboost_training_sessions table with complete data
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

def check_table_structure():
    """Check table structure"""
    conn = pymysql.connect(**db_config)
    cursor = conn.cursor()

    cursor.execute("DESCRIBE xgboost_training_sessions")
    columns = cursor.fetchall()

    print("=== xgboost_training_sessions Table Structure ===")
    for col in columns:
        print(f"{col[0]}: {col[1]}")

    conn.close()

def check_current_data():
    """Check current data in table"""
    conn = pymysql.connect(**db_config)
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    cursor.execute("SELECT * FROM xgboost_training_sessions")
    records = cursor.fetchall()

    print("\n=== Current Data in xgboost_training_sessions ===")
    if records:
        for record in records:
            print("\nRecord ID:", record['id'])
            for key, value in record.items():
                print(f"  {key}: {value}")
    else:
        print("No records found")

    conn.close()

def update_training_sessions():
    """Update training sessions with complete data from result files"""
    conn = pymysql.connect(**db_config)
    cursor = conn.cursor()

    # Find all training result files
    output_dir = Path('./output_train')
    result_files = list(output_dir.glob("training_results_*.json"))

    print(f"\n=== Processing {len(result_files)} training result files ===")

    for result_file in sorted(result_files, key=lambda x: x.stat().st_mtime, reverse=True):
        print(f"\nProcessing: {result_file.name}")

        with open(result_file, 'r') as f:
            results = json.load(f)

        # Extract session_id from filename
        if 'training_results_' in result_file.name:
            session_id = result_file.name.replace('training_results_', '').replace('.json', '')
        else:
            session_id = results.get('model_name', f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        # Prepare update data
        metrics = results.get('metrics', {})
        cv_results = results.get('cross_validation', {})

        # Update SQL with all fields
        sql = """
        UPDATE xgboost_training_sessions SET
            status = %s,
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
            sharpe_ratio = %s,
            total_return = %s,
            max_drawdown = %s,
            notes = %s,
            completed_at = %s
        WHERE session_id = %s
        """

        params = [
            'completed',  # status
            results.get('sample_count', 0),
            results.get('feature_count', 0),
            results.get('model_name', 'unknown'),
            json.dumps(results.get('parameters', {})),
            str(output_dir / results.get('model_name', '')),
            metrics.get('train_auc', 0.0),
            metrics.get('val_auc', 0.0),
            metrics.get('roc_auc', 0.0),
            metrics.get('accuracy', 0.0),
            metrics.get('precision', 0.0),
            metrics.get('recall', 0.0),
            metrics.get('f1', 0.0),
            metrics.get('sharpe_ratio', 0.0),
            metrics.get('total_return', 0.0),
            metrics.get('max_drawdown', 0.0),
            "Training completed successfully",
            datetime.now(),
            session_id
        ]

        try:
            cursor.execute(sql, params)
            if cursor.rowcount > 0:
                print(f"✅ Updated session: {session_id}")
            else:
                print(f"⚠️ Session {session_id} not found, inserting new...")

                # Insert if not exists
                insert_sql = """
                INSERT INTO xgboost_training_sessions (
                    session_id, status, total_samples, feature_count,
                    model_version, best_params, model_path,
                    train_auc, val_auc, test_auc, test_accuracy,
                    test_precision, test_recall, test_f1,
                    sharpe_ratio, total_return, max_drawdown,
                    notes, completed_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """

                insert_params = params[:-1]  # Remove session_id from end
                cursor.execute(insert_sql, insert_params)
                print(f"✅ Inserted new session: {session_id}")

        except Exception as e:
            print(f"❌ Error updating {session_id}: {e}")

    conn.commit()
    conn.close()

def load_performance_metrics():
    """Load performance metrics from evaluation files"""
    conn = pymysql.connect(**db_config)
    cursor = conn.cursor()

    output_dir = Path('./output_train')

    # Find latest performance report
    report_files = list(output_dir.glob("performance_report_*.json"))
    if not report_files:
        print("\n⚠️ No performance report files found")
        return

    latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
    print(f"\n=== Loading performance from: {latest_report.name} ===")

    with open(latest_report, 'r') as f:
        report = json.load(f)

    # Get session_id to update
    cursor.execute("SELECT session_id FROM xgboost_training_sessions ORDER BY created_at DESC LIMIT 1")
    result = cursor.fetchone()

    if result:
        session_id = result[0]

        # Extract performance metrics
        strategy_summary = report.get('strategy_summary', {})
        performance_metrics = strategy_summary.get('performance_metrics', {})

        # Update with performance metrics (only columns that exist in table)
        sql = """
        UPDATE xgboost_training_sessions SET
            total_return = %s,
            sharpe_ratio = %s,
            max_drawdown = %s,
            notes = %s
        WHERE session_id = %s
        """

        trading_activity = strategy_summary.get('trading_activity', {})
        notes = f"Trading metrics: Total trades={trading_activity.get('total_trades', 0)}, Win rate={performance_metrics.get('win_rate', 0):.2%}"

        params = [
            performance_metrics.get('total_return', 0.0),
            performance_metrics.get('sharpe_ratio', 0.0),
            performance_metrics.get('max_drawdown', 0.0),
            notes,
            session_id
        ]

        cursor.execute(sql, params)
        conn.commit()
        print(f"✅ Updated performance metrics for session: {session_id}")

    conn.close()

def main():
    """Main function"""
    check_table_structure()
    check_current_data()
    update_training_sessions()
    load_performance_metrics()

    # Final check
    print("\n=== Final Check ===")
    check_current_data()

if __name__ == "__main__":
    main()