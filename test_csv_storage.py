#!/usr/bin/env python3
"""
Test CSV storage functionality
"""

import os
import pymysql
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

db_config = {
    'host': os.getenv('DB_HOST', '103.150.81.86'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'database': os.getenv('DB_NAME', 'xgboostqc'),
    'user': os.getenv('DB_USER', 'xgboostqc'),
    'password': os.getenv('DB_PASSWORD', '6SPxBDwXH6WyxpfT')
}

def check_csv_tables():
    """Check all CSV-related tables"""
    conn = pymysql.connect(**db_config)
    cursor = conn.cursor()

    tables = [
        ('xgboost_feature_importance', 'Feature Importance'),
        ('xgboost_trades', 'Trades Summary'),
        ('xgboost_account_statements', 'Account Statements'),
        ('xgboost_csv_files', 'CSV Files Metadata')
    ]

    print("=== CSV Storage Tables Status ===")
    for table_name, display_name in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"{display_name:25}: {count} records")

        if count > 0:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
            sample = cursor.fetchone()
            print(f"  Sample: {sample[:3]}...")

    conn.close()

def test_csv_storage():
    """Test saving current CSV files"""
    from csv_storage import CSVStorage
    from pathlib import Path

    output_dir = Path('./output_train')
    csv_storage = CSVStorage()

    # Get latest session
    conn = pymysql.connect(**db_config)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT session_id FROM xgboost_training_sessions "
        "WHERE status = 'completed' ORDER BY created_at DESC LIMIT 1"
    )
    result = cursor.fetchone()

    if result:
        session_id = result[0]
        print(f"\n=== Testing CSV Storage for Session: {session_id} ===")

        # Get latest model
        model_files = list(output_dir.glob("*.joblib"))
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime) if model_files else None
        model_name = latest_model.name if latest_model else "unknown_model"

        print(f"Model: {model_name}")

        # Save all CSV outputs
        csv_storage.save_all_csv_outputs(output_dir, session_id, model_name)

        print("\n✅ CSV outputs saved successfully!")
    else:
        print("\n⚠️ No completed session found")

    conn.close()

if __name__ == "__main__":
    print("Checking current CSV storage...")
    check_csv_tables()
    test_csv_storage()