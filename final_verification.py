#!/usr/bin/env python3
"""
Final verification of xgboost_training_sessions table
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

conn = pymysql.connect(**db_config)
cursor = conn.cursor()

# Get column names
cursor.execute("DESCRIBE xgboost_training_sessions")
columns = [col[0] for col in cursor.fetchall()]

# Get data
cursor.execute("SELECT * FROM xgboost_training_sessions")
record = cursor.fetchone()

print("=== FINAL VERIFICATION: xgboost_training_sessions ===")
print("\nAll Columns Status:")
print("-" * 50)

null_count = 0
empty_count = 0

for i, (col_name, value) in enumerate(zip(columns, record)):
    if value is None:
        status = "❌ NULL"
        null_count += 1
    elif value == '':
        status = "⚠️ EMPTY"
        empty_count += 1
    else:
        status = "✅ FILLED"

    # Truncate long values
    display_value = str(value)[:50]
    if len(str(value)) > 50:
        display_value += "..."

    print(f"{col_name:25}: {status:8} | {display_value}")

print("-" * 50)
print(f"\nSummary:")
print(f"  Total Columns: {len(columns)}")
print(f"  Filled Columns: {len(columns) - null_count - empty_count}")
print(f"  NULL Columns: {null_count}")
print(f"  EMPTY Columns: {empty_count}")

if null_count == 0 and empty_count == 0:
    print("\n🎉 ALL COLUMNS ARE POPULATED! No NULL or EMPTY values found!")
else:
    print(f"\n⚠️ Found {null_count} NULL and {empty_count} EMPTY columns that need attention")

print("\n=== Key Information ===")
print(f"Session ID: {record[1]}")  # session_id at index 1
print(f"Status: {record[3]}")      # status at index 3
print(f"Test AUC: {record[14]}")   # test_auc at index 14
print(f"Total Return: {record[21]}%")  # total_return at index 21
print(f"Sharpe Ratio: {record[22]}")  # sharpe_ratio at index 22

conn.close()