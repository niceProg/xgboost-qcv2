#!/usr/bin/env python3
"""
Fix ALL fields in xgboost_training_sessions table
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

def get_table_columns():
    """Get all columns from the table"""
    conn = pymysql.connect(**db_config)
    cursor = conn.cursor()

    cursor.execute("DESCRIBE xgboost_training_sessions")
    columns = [row[0] for row in cursor.fetchall()]
    conn.close()
    return columns

def get_current_data():
    """Get current data from the table"""
    conn = pymysql.connect(**db_config)
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    cursor.execute("SELECT * FROM xgboost_training_sessions")
    records = cursor.fetchall()
    conn.close()
    return records

def update_all_fields():
    """Update ALL fields in the table"""
    conn = pymysql.connect(**db_config)
    cursor = conn.cursor()

    # Get current data
    cursor.execute("SELECT * FROM xgboost_training_sessions")
    record = cursor.fetchone()
    session_id = record[1]  # session_id is at index 1

    print(f"Updating session: {session_id}")

    # Load training data
    output_dir = Path('./output_train')

    # Get latest training results
    training_files = list(output_dir.glob("training_results_*.json"))
    if training_files:
        latest_training = max(training_files, key=lambda x: x.stat().st_mtime)
        with open(latest_training, 'r') as f:
            training_data = json.load(f)

    # Get performance data
    perf_files = list(output_dir.glob("performance_report_*.json"))
    if perf_files:
        latest_perf = max(perf_files, key=lambda x: x.stat().st_mtime)
        with open(latest_perf, 'r') as f:
            perf_data = json.load(f)

    # Extract metrics
    metrics = training_data.get('metrics', {})
    cv_results = training_data.get('cross_validation', {})

    # Build comprehensive notes
    notes = "XGBoost Trading Model Training Results\n"
    notes += "=" * 40 + "\n\n"
    notes += f"Model: {training_data.get('model_name', 'unknown')}\n"
    notes += f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    notes += f"Exchange: Binance\n"
    notes += f"Symbol: BTCUSDT\n"
    notes += f"Interval: 1h\n\n"

    notes += "Model Metrics:\n"
    notes += f"- Test AUC: {metrics.get('roc_auc', 0):.4f}\n"
    notes += f"- Test Accuracy: {metrics.get('accuracy', 0):.4f}\n"
    notes += f"- Test Precision: {metrics.get('precision', 0):.4f}\n"
    notes += f"- Test Recall: {metrics.get('recall', 0):.4f}\n"
    notes += f"- Test F1-Score: {metrics.get('f1', 0):.4f}\n"
    notes += f"- CV Mean AUC: {cv_results.get('mean_auc', 0):.4f}\n"
    notes += f"- CV Std AUC: {cv_results.get('std_auc', 0):.4f}\n\n"

    if perf_files:
        strategy = perf_data.get('strategy_summary', {})
        performance = strategy.get('performance_metrics', {})
        trading = strategy.get('trading_activity', {})
        trading_period = strategy.get('trading_period', {})

        notes += "Trading Performance:\n"
        notes += f"- Total Return: {performance.get('total_return', 0):.2%}\n"
        notes += f"- CAGR: {performance.get('cagr', 0):.2%}\n"
        notes += f"- Benchmark Return: {performance.get('benchmark_return', 0):.2%}\n"
        notes += f"- Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}\n"
        notes += f"- Win Rate: {performance.get('win_rate', 0):.2%}\n"
        notes += f"- Profit Factor: {performance.get('profit_factor', 0):.2f}\n"
        notes += f"- Max Drawdown: {performance.get('max_drawdown', 0):.2%}\n"
        notes += f"- Total Trades: {trading.get('total_trades', 0)}\n"
        notes += f"- Winning Trades: {trading.get('winning_trades', 0)}\n"
        notes += f"- Losing Trades: {trading.get('losing_trades', 0)}\n"
        notes += f"- Average Win: {trading.get('avg_win', 0):.4f}\n"
        notes += f"- Average Loss: {trading.get('avg_loss', 0):.4f}\n\n"

        notes += "Trading Period:\n"
        notes += f"- Start Date: {trading_period.get('start_date', 'N/A')}\n"
        notes += f"- End Date: {trading_period.get('end_date', 'N/A')}\n"
        notes += f"- Trading Days: {trading_period.get('total_days', 0)}\n\n"

    notes += "Hyperparameters:\n"
    params = training_data.get('parameters', {})
    for key, value in params.items():
        notes += f"- {key}: {value}\n"

    notes += "\nFeatures:\n"
    notes += f"- Feature Count: {training_data.get('feature_count', 0)}\n"
    notes += f"- Sample Count: {training_data.get('sample_count', 0)}\n"
    notes += f"- Input Data: 7 tables merged (spot_price, funding_rate, futures_basis, ls_global_ratio, ls_top_ratio)\n"

    # Get trading performance values
    if perf_files:
        total_return = performance.get('total_return', 0.0)
        sharpe_ratio = performance.get('sharpe_ratio', 0.0)
        max_drawdown = performance.get('max_drawdown', 0.0)
    else:
        total_return = 0.0
        sharpe_ratio = 0.0
        max_drawdown = 0.0

    # Update ALL fields
    sql = """
    UPDATE xgboost_training_sessions SET
        status = %s,
        exchange_filter = %s,
        symbol_filter = %s,
        interval_filter = %s,
        time_range = %s,
        days_filter = %s,
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
        json.dumps(['binance']),  # exchange_filter
        json.dumps(['BTCUSDT']),  # symbol_filter
        json.dumps(['1h']),  # interval_filter
        json.dumps({
            'start': '2024-01-01 00:00:00',
            'end': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'timezone': 'UTC'
        }),  # time_range
        724,  # days_filter (approx days from 2024-01-01 to now)
        training_data.get('sample_count', 0),  # total_samples
        training_data.get('feature_count', 0),  # feature_count
        training_data.get('model_name', 'unknown'),  # model_version
        json.dumps(training_data.get('parameters', {})),  # best_params
        f"./output_train/{training_data.get('model_name', '')}",  # model_path
        metrics.get('train_auc', 0.0),  # train_auc
        metrics.get('val_auc', 0.0),  # val_auc
        metrics.get('roc_auc', 0.0),  # test_auc
        metrics.get('accuracy', 0.0),  # test_accuracy
        metrics.get('precision', 0.0),  # test_precision
        metrics.get('recall', 0.0),  # test_recall
        metrics.get('f1', 0.0),  # test_f1
        sharpe_ratio,  # sharpe_ratio
        total_return,  # total_return
        max_drawdown,  # max_drawdown
        notes,  # notes
        datetime.now(),  # completed_at
        session_id  # WHERE
    ]

    cursor.execute(sql, params)
    conn.commit()
    conn.close()

    print("✅ All fields updated successfully!")

def show_result():
    """Show the updated record"""
    conn = pymysql.connect(**db_config)
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    cursor.execute("SELECT * FROM xgboost_training_sessions")
    record = cursor.fetchone()

    print("\n" + "="*80)
    print("UPDATED xgboost_training_sessions RECORD")
    print("="*80)

    # Group fields by category
    basic_fields = ['id', 'session_id', 'created_at', 'status', 'completed_at']
    filter_fields = ['exchange_filter', 'symbol_filter', 'interval_filter', 'time_range', 'days_filter']
    data_fields = ['total_samples', 'feature_count']
    model_fields = ['model_version', 'model_path', 'best_params']
    metric_fields = ['train_auc', 'val_auc', 'test_auc', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1']
    trading_fields = ['total_return', 'sharpe_ratio', 'max_drawdown']

    print("\n📅 Basic Info:")
    for field in basic_fields:
        value = record[field]
        if value and isinstance(value, datetime):
            value = value.strftime('%Y-%m-%d %H:%M:%S')
        print(f"  {field:15}: {value}")

    print("\n🔍 Filters:")
    for field in filter_fields:
        value = record[field]
        if value and isinstance(value, str):
            try:
                parsed = json.loads(value)
                value = f"{parsed} (JSON)"
            except:
                pass
        print(f"  {field:15}: {value}")

    print("\n📊 Data Statistics:")
    for field in data_fields:
        value = record[field]
        if value:
            if field == 'total_samples':
                value = f"{value:,}"
        print(f"  {field:15}: {value}")

    print("\n🤖 Model Info:")
    for field in model_fields:
        value = record[field]
        if field == 'best_params' and value:
            # Show just a few params
            try:
                params = json.loads(value)
                if len(params) > 3:
                    value = str(dict(list(params.items())[:3])) + "..."
            except:
                pass
        print(f"  {field:15}: {value[:50]}..." if value and len(str(value)) > 50 else f"  {field:15}: {value}")

    print("\n📈 Model Metrics:")
    for field in metric_fields:
        value = record[field]
        if value is not None:
            print(f"  {field:15}: {value:.4f}")
        else:
            print(f"  {field:15}: N/A")

    print("\n💰 Trading Performance:")
    for field in trading_fields:
        value = record[field]
        if value is not None:
            if field == 'total_return':
                print(f"  {field:15}: {value:.2%}")
            else:
                print(f"  {field:15}: {value:.4f}")
        else:
            print(f"  {field:15}: N/A")

    print("\n📝 Notes (first 500 chars):")
    notes = record.get('notes', '')[:500]
    print(f"  {notes}..." if len(notes) == 500 else f"  {notes}")

    print("\n" + "="*80)

    # Check for any NULL values
    null_fields = []
    for key, value in record.items():
        if value is None:
            null_fields.append(key)

    if null_fields:
        print(f"\n⚠️ Still NULL fields: {null_fields}")
    else:
        print(f"\n✅ No NULL fields - All fields populated!")

    conn.close()

def main():
    """Main function"""
    print("Fixing ALL fields in xgboost_training_sessions table...")
    update_all_fields()
    show_result()

if __name__ == "__main__":
    main()