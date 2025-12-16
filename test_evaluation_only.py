#!/usr/bin/env python3
"""
Test only model evaluation step
"""

import os
import sys
from pathlib import Path

print("=== Testing Model Evaluation Only ===")

# Check if required files exist
output_dir = Path('./output_train')

required_files = [
    'latest_model.joblib',
    'X_train_features.parquet',
    'y_train_labels.parquet',
    'labeled_data.parquet'
]

print("\n📁 Checking required files...")
missing_files = []
for file in required_files:
    file_path = output_dir / file
    if file_path.exists():
        print(f"✅ {file}")
    else:
        print(f"❌ {file} - MISSING")
        missing_files.append(file)

if missing_files:
    print(f"\n❌ Missing files: {missing_files}")
    print("Please run the full pipeline first with: ./simple_run.sh")
    sys.exit(1)

# Run evaluation
print("\n🚀 Running model evaluation...")
cmd = [
    'python', 'model_evaluation_with_leverage.py',
    '--exchange', 'binance',
    '--pair', 'BTCUSDT',
    '--interval', '1h'
]

import subprocess
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    print("✅ Evaluation completed successfully!")
else:
    print("❌ Evaluation failed!")
    print("\nError output:")
    print(result.stderr[-1000:])

# Check if CSV files were created
print("\n📊 Checking CSV outputs...")
csv_files = ['trades.csv', 'rekening_koran.csv', 'rekening_koran_cash.csv']
created_files = []
for csv_file in csv_files:
    file_path = output_dir / csv_file
    if file_path.exists():
        size_mb = file_path.stat().st_size / 1024 / 1024
        print(f"✅ {csv_file} ({size_mb:.2f}MB)")
        created_files.append(csv_file)
    else:
        print(f"❌ {csv_file} - NOT CREATED")

if created_files:
    # Check database
    print("\n🔍 Checking database...")
    import pymysql
    from dotenv import load_dotenv

    load_dotenv()
    db_config = {
        'host': os.getenv('DB_HOST', '103.150.81.86'),
        'port': int(os.getenv('DB_PORT', 3306)),
        'database': os.getenv('DB_NAME', 'xgboostqc'),
        'user': os.getenv('DB_USER', 'xgboostqc'),
        'password': os.getenv('DB_PASSWORD', '6SPxBDwXH6WyxpfT')
    }

    conn = pymysql.connect(**db_config)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM xgboost_trades")
    trades_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM xgboost_account_statements")
    stmt_count = cursor.fetchone()[0]

    print(f"xgboost_trades records: {trades_count}")
    print(f"xgboost_account_statements records: {stmt_count}")

    if trades_count > 0 and stmt_count > 0:
        print("\n✅ SUCCESS! CSV data saved to database")
    else:
        print("\n⚠️ WARNING: CSV files created but NOT saved to database")

    conn.close()
else:
    print("\n❌ No CSV files were created")