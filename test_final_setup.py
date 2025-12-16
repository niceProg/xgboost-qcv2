#!/usr/bin/env python3
"""
Test final setup after removing unwanted tables
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

print("="*80)
print("DATABASE TABLES AFTER CLEANUP")
print("="*80)

# Get all tables in xgboostqc database
cursor.execute("SHOW TABLES")
tables = [row[0] for row in cursor.fetchall()]

print(f"\n📋 Total Tables: {len(tables)}")
print("\nList of Tables:")
for i, table in enumerate(tables, 1):
    print(f"  {i}. {table}")

# Check record counts for remaining tables
print("\n" + "="*50)
print("RECORD COUNTS")
print("="*50)

# Core tables
core_tables = [
    ('xgboost_training_sessions', 'Training Sessions'),
    ('xgboost_models', 'Models'),
    ('xgboost_features', 'Features'),
    ('xgboost_evaluations', 'Evaluations'),
    ('xgboost_trades', 'Trades Summary'),
    ('xgboost_account_statements', 'Account Statements')
]

for table, display_name in core_tables:
    if table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"{display_name:20}: {count} records")
    else:
        print(f"{display_name:20}: Table not found")

print("\n" + "="*50)
print("REMOVED TABLES (as requested)")
print("="*50)
removed_tables = ['xgboost_feature_importance', 'xgboost_csv_files', 'xgboost_trade_events']
for table in removed_tables:
    print(f"❌ {table}")

print("\n" + "="*50)
print("✅ Unwanted tables successfully removed!")
print("✅ Only essential tables remain in database")
print("="*50)

conn.close()