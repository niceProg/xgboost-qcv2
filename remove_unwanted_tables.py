#!/usr/bin/env python3
"""
Remove unwanted tables from database
"""

import os
import pymysql
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

tables_to_remove = [
    'xgboost_trade_events',
    'xgboost_csv_files',
    'xgboost_feature_importance'
]

conn = pymysql.connect(**db_config)
cursor = conn.cursor()

print("Removing unwanted tables...")
for table in tables_to_remove:
    try:
        cursor.execute(f"DROP TABLE IF EXISTS {table}")
        print(f"✅ Dropped table: {table}")
    except Exception as e:
        print(f"⚠️ Error dropping {table}: {e}")

conn.commit()
conn.close()

print("\n✅ All unwanted tables removed successfully!")