#!/usr/bin/env python3
"""
Debug CSV saving during pipeline run
"""

import os
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

print("="*80)
print("DEBUGGING CSV SAVING")
print("="*80)

# Check if ENABLE_DB_STORAGE is set
print(f"\n🔧 Environment Variables:")
print(f"ENABLE_DB_STORAGE: {os.getenv('ENABLE_DB_STORAGE', 'not set')}")
print(f"DB_HOST: {os.getenv('DB_HOST', 'not set')}")
print(f"DB_NAME: {os.getenv('DB_NAME', 'not set')}")

# Check tables exist
conn = pymysql.connect(**db_config)
cursor = conn.cursor()

cursor.execute("SHOW TABLES")
tables = [row[0] for row in cursor.fetchall()]

print(f"\n📋 Available Tables: {len(tables)}")
for table in tables:
    if 'trades' in table or 'account' in table:
        print(f"  ✅ {table}")
    else:
        print(f"  {table}")

# Check current records
print(f"\n📊 Current Records:")
cursor.execute("SELECT COUNT(*) FROM xgboost_trades")
trades_count = cursor.fetchone()[0]
print(f"xgboost_trades: {trades_count} records")

cursor.execute("SELECT COUNT(*) FROM xgboost_account_statements")
stmt_count = cursor.fetchone()[0]
print(f"xgboost_account_statements: {stmt_count} records")

# Check CSV files in output
from pathlib import Path
output_dir = Path('./output_train')

print(f"\n📁 CSV Files in output directory:")
csv_files = list(output_dir.glob("*.csv"))
for csv_file in csv_files:
    size_mb = csv_file.stat().st_size / 1024 / 1024
    print(f"  • {csv_file.name} ({size_mb:.2f}MB)")

# Test CSV storage manually
print(f"\n🧪 Testing CSV Storage Manually...")
from csv_storage import CSVStorage

csv_storage = CSVStorage()

# Get latest session
cursor.execute(
    "SELECT session_id FROM xgboost_training_sessions "
    "ORDER BY created_at DESC LIMIT 1"
)
result = cursor.fetchone()

if result:
    session_id = result[0]
    print(f"Latest session: {session_id}")

    # Test saving trades
    trades_file = output_dir / "trades.csv"
    if trades_file.exists():
        print(f"\n💾 Testing trades saving...")
        try:
            csv_storage.save_trades_summary(trades_file, session_id)
            print("✅ Trades saved successfully")
        except Exception as e:
            print(f"❌ Error saving trades: {e}")
            import traceback
            traceback.print_exc()

    # Test saving account statements
    equity_file = output_dir / "rekening_koran.csv"
    cash_file = output_dir / "rekening_koran_cash.csv"

    if equity_file.exists():
        print(f"\n💾 Testing equity statement saving...")
        try:
            csv_storage.save_account_statement_sample(equity_file, session_id, 'equity')
            print("✅ Equity statement saved successfully")
        except Exception as e:
            print(f"❌ Error saving equity statement: {e}")

    if cash_file.exists():
        print(f"\n💾 Testing cash statement saving...")
        try:
            csv_storage.save_account_statement_sample(cash_file, session_id, 'cash')
            print("✅ Cash statement saved successfully")
        except Exception as e:
            print(f"❌ Error saving cash statement: {e}")
else:
    print("\n⚠️ No training session found in database")

conn.close()