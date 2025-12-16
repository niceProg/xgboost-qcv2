#!/usr/bin/env python3
"""
Rebuild xgboost_training_sessions table with complete data
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
)

def get_latest_session_id():
    """Get the latest session from load_database.py output"""
    conn = pymysql.connect(**db_config)
    cursor = conn.cursor()

    # Get the latest session that was created by load_database
    cursor.execute(
        "SELECT session_id, created_at FROM xgboost_training_sessions "
        "WHERE status = 'data_loaded' "
        "ORDER BY created_at DESC LIMIT 1"
    )
    result = cursor.fetchone()

    conn.close()
    return result[0] if result else None

def update_existing_session():
    """Update the existing data_loaded session with latest training data"""
    conn = pymysql.connect(**db_config)
    cursor = conn.cursor()

    # Get latest training results
    output_dir = Path('./output_train')
    training_files = list(output_dir.glob("training_results_*.json"))
    perf_files = list(output_dir.glob("performance_report_*.json"))

    if not training_files:
        print("❌ No training files found")
        return

    # Get latest training file
    latest_training = max(training_files, key=lambda x: x.stat().st_mtime)
    with open(latest_training, 'r') as f:
        training_data = json.load(f)

    # Get latest performance file
    if perf_files:
        latest_perf = max(perf_files, key=lambda x: x.stat().st_mtime)
        with open(latest_perf, 'r') as f:
            perf_data = json.load(f)

    # Get existing session
    session_id = get_latest_session_id()
    if not session_id:
        print("❌ No existing session found")
        return

    print(f"Updating existing session: {session_id}")

    # Prepare training metrics
    metrics = training_data.get('metrics', {})

    # Build comprehensive notes
    notes = "Training completed successfully.\n"
    notes += f"Model: {training_data.get('model_name', 'unknown')}\n"
    notes += f"Features: {training_data.get('feature_count', 0)}\n"
    notes += f"Samples: {training_data.get('sample_count', 0)}\n\n"

    if perf_files:
        strategy_summary = perf_data.get('strategy_summary', {})
        performance_metrics = strategy_summary.get('performance_metrics', {})
        trading_activity = strategy_summary.get('trading_activity', {})

        notes += "Performance Metrics:\n"
        notes += f"- Total Return: {performance_metrics.get('total_return', 0):.2%}\n"
        notes += f"- CAGR: {performance_metrics.get('cagr', 0):.2%}\n"
        notes += f"- Sharpe Ratio: {performance_metrics.get('sharpe_ratio', 0):.2f}\n"
        notes += f"- Max Drawdown: {performance_metrics.get('max_drawdown', 0):.2%}\n"
        notes += f"- Win Rate: {performance_metrics.get('win_rate', 0):.2%}\n"
        notes += f"- Profit Factor: {performance_metrics.get('profit_factor', 0):.2f}\n"
        notes += f"- Total Trades: {trading_activity.get('total_trades', 0)}\n"
        notes += f"- Winning Trades: {trading_activity.get('winning_trades', 0)}\n"
        notes += f"- Losing Trades: {trading_activity.get('losing_trades', 0)}\n"
        notes += f"\nTrading Period: {strategy_summary.get('trading_period', {}).get('start_date', '')} to {strategy_summary.get('trading_period', {}).get('end_date', '')}\n"

        # Get performance values
        total_return = performance_metrics.get('total_return', 0.0)
        sharpe_ratio = performance_metrics.get('sharpe_ratio', 0.0)
        max_drawdown = performance_metrics.get('max_drawdown', 0.0)
    else:
        notes += "No performance evaluation available."
        total_return = 0.0
        sharpe_ratio = 0.0
        max_drawdown = 0.0

    # Update the session
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
        sharpe_ratio = %s,
        total_return = %s,
        max_drawdown = %s,
        notes = %s,
        completed_at = %s
    WHERE session_id = %s
    """

    params = [
        training_data.get('sample_count', 0),
        training_data.get('feature_count', 0),
        training_data.get('model_name', 'unknown'),
        json.dumps(training_data.get('parameters', {})),
        f"./output_train/{training_data.get('model_name', '')}",
        metrics.get('train_auc', 0.0),
        metrics.get('val_auc', 0.0),
        metrics.get('roc_auc', 0.0),
        metrics.get('accuracy', 0.0),
        metrics.get('precision', 0.0),
        metrics.get('recall', 0.0),
        metrics.get('f1', 0.0),
        sharpe_ratio,
        total_return,
        max_drawdown,
        notes,
        datetime.now(),
        session_id
    ]

    cursor.execute(sql, params)
    conn.commit()
    conn.close()

    print(f"✅ Updated session {session_id} with complete data")
    return session_id

def insert_new_sessions():
    """Insert new sessions for other training runs"""
    conn = pymysql.connect(**db_config)
    cursor = conn.cursor()

    output_dir = Path('./output_train')
    training_files = sorted(output_dir.glob("training_results_*.json"), reverse=True)
    perf_files = sorted(output_dir.glob("performance_report_*.json"), reverse=True)

    # Skip the latest one (already updated)
    if training_files:
        training_files = training_files[1:]

    for training_file in training_files:
        # Extract session ID from filename
        session_id = training_file.stem.replace('training_results_', '')

        # Check if already exists
        cursor.execute("SELECT id FROM xgboost_training_sessions WHERE session_id = %s", (session_id,))
        if cursor.fetchone():
            print(f"⚠️ Session {session_id} already exists, skipping...")
            continue

        print(f"Inserting new session: {session_id}")

        with open(training_file, 'r') as f:
            training_data = json.load(f)

        # Find corresponding performance file
        perf_data = None
        for pf in perf_files:
            if session_id in pf.name:
                with open(pf, 'r') as f:
                    perf_data = json.load(f)
                break

        # Prepare data
        metrics = training_data.get('metrics', {})

        # Build notes
        notes = f"Model: {training_data.get('model_name', 'unknown')}\n"
        notes += f"Features: {training_data.get('feature_count', 0)}\n"
        notes += f"Samples: {training_data.get('sample_count', 0)}\n"

        total_return = 0.0
        sharpe_ratio = 0.0
        max_drawdown = 0.0

        if perf_data:
            strategy_summary = perf_data.get('strategy_summary', {})
            performance_metrics = strategy_summary.get('performance_metrics', {})
            trading_activity = strategy_summary.get('trading_activity', {})

            notes += f"\nTotal Return: {performance_metrics.get('total_return', 0):.2%}\n"
            notes += f"Sharpe Ratio: {performance_metrics.get('sharpe_ratio', 0):.2f}\n"
            notes += f"Win Rate: {performance_metrics.get('win_rate', 0):.2%}\n"
            notes += f"Total Trades: {trading_activity.get('total_trades', 0)}"

            total_return = performance_metrics.get('total_return', 0.0)
            sharpe_ratio = performance_metrics.get('sharpe_ratio', 0.0)
            max_drawdown = performance_metrics.get('max_drawdown', 0.0)

        # Insert new session
        sql = """
        INSERT INTO xgboost_training_sessions (
            session_id, status, total_samples, feature_count,
            model_version, best_params, model_path,
            train_auc, val_auc, test_auc, test_accuracy,
            test_precision, test_recall, test_f1,
            sharpe_ratio, total_return, max_drawdown,
            notes, completed_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        params = [
            session_id,
            'completed',
            training_data.get('sample_count', 0),
            training_data.get('feature_count', 0),
            training_data.get('model_name', 'unknown'),
            json.dumps(training_data.get('parameters', {})),
            f"./output_train/{training_data.get('model_name', '')}",
            metrics.get('train_auc', 0.0),
            metrics.get('val_auc', 0.0),
            metrics.get('roc_auc', 0.0),
            metrics.get('accuracy', 0.0),
            metrics.get('precision', 0.0),
            metrics.get('recall', 0.0),
            metrics.get('f1', 0.0),
            sharpe_ratio,
            total_return,
            max_drawdown,
            notes,
            datetime.fromtimestamp(training_file.stat().st_mtime)
        ]

        cursor.execute(sql, params)
        print(f"✅ Inserted session {session_id}")

    conn.commit()
    conn.close()

def show_final_result():
    """Show the final state of the table"""
    conn = pymysql.connect(**db_config)
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    cursor.execute("""
        SELECT session_id, status, total_samples, feature_count,
               test_auc, total_return, sharpe_ratio, max_drawdown,
               created_at, completed_at
        FROM xgboost_training_sessions
        ORDER BY created_at DESC
    """)
    records = cursor.fetchall()

    print("\n" + "="*80)
    print("FINAL STATE OF xgboost_training_sessions TABLE")
    print("="*80)

    for i, record in enumerate(records, 1):
        print(f"\n{i}. Session: {record['session_id']}")
        print(f"   Status: {record['status']}")
        print(f"   Created: {record['created_at']}")
        print(f"   Completed: {record['completed_at'] or 'N/A'}")
        print(f"   Samples: {record['total_samples'] or 'N/A'}")
        print(f"   Features: {record['feature_count'] or 'N/A'}")
        print(f"   Test AUC: {record['test_auc'] or 'N/A'}")
        print(f"   Total Return: {record['total_return'] or 'N/A'}")
        print(f"   Sharpe Ratio: {record['sharpe_ratio'] or 'N/A'}")
        print(f"   Max Drawdown: {record['max_drawdown'] or 'N/A'}")

    print("\n" + "="*80)
    print(f"Total Sessions: {len(records)}")
    print("="*80)

    conn.close()

def main():
    """Main function"""
    print("Rebuilding xgboost_training_sessions table...\n")

    # First update the existing session
    updated_session = update_existing_session()

    # Then insert other sessions
    insert_new_sessions()

    # Show final result
    show_final_result()

if __name__ == "__main__":
    main()