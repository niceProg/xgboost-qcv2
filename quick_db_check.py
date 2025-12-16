#!/usr/bin/env python3
"""
Quick database fix script - just verify what's in the database
"""

import os
import pymysql
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

try:
    conn = pymysql.connect(**db_config)
    cursor = conn.cursor()

    print("=== Database Contents ===")

    tables = [
        ('xgboost_training_sessions', 'Training Sessions'),
        ('xgboost_models', 'Models'),
        ('xgboost_features', 'Features'),
        ('xgboost_evaluations', 'Evaluations')
    ]

    for table_name, display_name in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"{display_name}: {count} records")

        if count > 0:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
            columns = [desc[0] for desc in cursor.description]
            row = cursor.fetchone()
            print(f"  Columns: {', '.join(columns[:5])}...")

    conn.close()
    print("\nDatabase connection successful!")

except Exception as e:
    print(f"Error: {e}")