#!/usr/bin/env python3
"""
Debug feature importance saving
"""

import os
import pymysql
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

db_config = {
    'host': os.getenv('DB_HOST', '103.150.81.86'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'database': os.getenv('DB_NAME', 'xgboostqc'),
    'user': os.getenv('DB_USER', 'xgboostqc'),
    'password': os.getenv('DB_PASSWORD', '6SPxBDwXH6WyxpfT')
}

# Get latest feature importance file
output_dir = Path('./output_train')
feature_files = list(output_dir.glob("feature_importance_*.csv"))
latest_file = max(feature_files, key=lambda x: x.stat().st_mtime)

print(f"Processing: {latest_file}")
print(f"File size: {latest_file.stat().st_size} bytes")

# Read CSV
df = pd.read_csv(latest_file)
print(f"Rows in CSV: {len(df)}")
print("Sample data:")
print(df.head())

# Save to database manually
conn = pymysql.connect(**db_config)
cursor = conn.cursor()

# Get session ID
cursor.execute(
    "SELECT session_id FROM xgboost_training_sessions "
    "WHERE status = 'completed' ORDER BY created_at DESC LIMIT 1"
)
result = cursor.fetchone()
session_id = result[0] if result else "test_session"
model_name = latest_file.name

print(f"\nSession ID: {session_id}")
print(f"Model Name: {model_name}")

# Clear existing
cursor.execute(
    "DELETE FROM xgboost_feature_importance WHERE session_id = %s AND model_name = %s",
    (session_id, model_name)
)

# Insert data
inserted = 0
for _, row in df.iterrows():
    try:
        cursor.execute("""
            INSERT INTO xgboost_feature_importance (
                session_id, model_name, feature, importance
            ) VALUES (%s, %s, %s, %s)
        """, (session_id, model_name, row['feature'], row['importance']))
        inserted += 1
    except Exception as e:
        print(f"Error inserting {row['feature']}: {e}")

conn.commit()
print(f"\nInserted {inserted} records")

# Verify
cursor.execute(
    "SELECT COUNT(*) FROM xgboost_feature_importance WHERE session_id = %s",
    (session_id,)
)
count = cursor.fetchone()[0]
print(f"Total records in DB: {count}")

conn.close()