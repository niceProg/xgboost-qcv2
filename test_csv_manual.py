#!/usr/bin/env python3
"""
Manual test of CSV storage functionality
"""

import os
import pymysql
from dotenv import load_dotenv
from csv_storage import CSVStorage
from pathlib import Path

load_dotenv()

db_config = {
    'host': os.getenv('DB_HOST', '103.150.81.86'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'database': os.getenv('DB_NAME', 'xgboostqc'),
    'user': os.getenv('DB_USER', 'xgboostqc'),
    'password': os.getenv('DB_PASSWORD', '6SPxBDwXH6WyxpfT')
}

# Get session ID
conn = pymysql.connect(**db_config)
cursor = conn.cursor()

cursor.execute(
    "SELECT session_id FROM xgboost_training_sessions "
    "WHERE status = 'completed' ORDER BY created_at DESC LIMIT 1"
)
result = cursor.fetchone()

if not result:
    print("No completed session found")
    exit(1)

session_id = result[0]
print(f"Testing CSV storage for session: {session_id}")

# Get latest model
output_dir = Path('./output_train')
model_files = list(output_dir.glob("*.joblib"))
model_name = model_files[0].name if model_files else "test_model"
print(f"Model: {model_name}")

# Test CSV storage
csv_storage = CSVStorage()

# Clear existing data
cursor.execute("DELETE FROM xgboost_feature_importance WHERE session_id = %s", (session_id,))
cursor.execute("DELETE FROM xgboost_trades WHERE session_id = %s", (session_id,))
cursor.execute("DELETE FROM xgboost_account_statements WHERE session_id = %s", (session_id,))
cursor.execute("DELETE FROM xgboost_csv_files WHERE session_id = %s", (session_id,))
conn.commit()

# Save feature importance
feature_file = output_dir / "feature_importance_20251216_184037.csv"
if feature_file.exists():
    csv_storage.save_feature_importance(feature_file, session_id, model_name)
    print("✅ Saved feature importance")

# Save trades
trades_file = output_dir / "trades.csv"
if trades_file.exists():
    csv_storage.save_trades_summary(trades_file, session_id)
    print("✅ Saved trades summary")

# Save account statements
equity_file = output_dir / "rekening_koran.csv"
cash_file = output_dir / "rekening_koran_cash.csv"

if equity_file.exists():
    csv_storage.save_account_statement_sample(equity_file, session_id, 'equity')
    print("✅ Saved equity account statement")

if cash_file.exists():
    csv_storage.save_account_statement_sample(cash_file, session_id, 'cash')
    print("✅ Saved cash account statement")

# Save metadata
for csv_file in output_dir.glob("*.csv"):
    file_type = "output"
    if "feature" in csv_file.name.lower():
        file_type = "feature_importance"
    elif "trades" in csv_file.name.lower():
        file_type = "trades"
    elif "rekening" in csv_file.name.lower() or "koran" in csv_file.name.lower():
        file_type = "account_statement"

    csv_storage.save_csv_metadata(
        csv_file,
        session_id,
        file_type,
        f"Output file from XGBoost trading pipeline - {file_type}"
    )

print("✅ Saved CSV metadata")

# Verify
print("\n=== Verification ===")
tables = [
    ('xgboost_feature_importance', 'Feature Importance'),
    ('xgboost_trades', 'Trades Summary'),
    ('xgboost_account_statements', 'Account Statements'),
    ('xgboost_csv_files', 'CSV Files Metadata')
]

for table_name, display_name in tables:
    cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE session_id = %s", (session_id,))
    count = cursor.fetchone()[0]
    print(f"{display_name:25}: {count} records")

    if count > 0 and table_name == 'xgboost_feature_importance':
        cursor.execute(f"SELECT feature, importance FROM {table_name} WHERE session_id = %s LIMIT 3", (session_id,))
        for row in cursor.fetchall():
            print(f"  - {row[0]:30}: {row[1]:.4f}")

conn.close()