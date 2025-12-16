#!/usr/bin/env python3
"""
Test pipeline with clean database state
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

# Clean the table for fresh test
conn = pymysql.connect(**db_config)
cursor = conn.cursor()

# Delete existing data
cursor.execute("DELETE FROM xgboost_evaluations")
cursor.execute("DELETE FROM xgboost_models")
cursor.execute("DELETE FROM xgboost_features")
cursor.execute("DELETE FROM xgboost_training_sessions")
conn.commit()

print("✅ Cleared all tables for fresh test")

# Run the pipeline
print("\n🚀 Running pipeline: ./simple_run.sh")
import subprocess

result = subprocess.run(
    ["bash", "simple_run.sh"],
    capture_output=True,
    text=True,
    env=os.environ.copy()
)

if result.returncode == 0:
    print("✅ Pipeline completed successfully!")
else:
    print("❌ Pipeline failed!")
    print("Error output:")
    print(result.stderr[-1000:])  # Last 1000 chars of error

# Check the database
print("\n📊 Checking database contents...")

cursor.execute("SELECT COUNT(*) FROM xgboost_training_sessions")
session_count = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM xgboost_models")
model_count = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM xgboost_evaluations")
eval_count = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM xgboost_features")
feature_count = cursor.fetchone()[0]

print(f"\nDatabase Records:")
print(f"  Training Sessions: {session_count}")
print(f"  Models: {model_count}")
print(f"  Evaluations: {eval_count}")
print(f"  Features: {feature_count}")

# Check training session details
if session_count > 0:
    cursor.execute("""
        SELECT session_id, status, total_samples, feature_count,
               test_auc, total_return, sharpe_ratio
        FROM xgboost_training_sessions
    """)
    session = cursor.fetchone()

    print(f"\n📈 Training Session Details:")
    print(f"  Session ID: {session[0]}")
    print(f"  Status: {session[1]}")
    print(f"  Total Samples: {session[2] or 'NULL'}")
    print(f"  Feature Count: {session[3] or 'NULL'}")
    print(f"  Test AUC: {session[4] or 'NULL'}")
    print(f"  Total Return: {session[5] or 'NULL'}")
    print(f"  Sharpe Ratio: {session[6] or 'NULL'}")

conn.close()

# Check for NULL values in critical fields
conn = pymysql.connect(**db_config)
cursor = conn.cursor()

cursor.execute("""
    SELECT COUNT(*) as null_count
    FROM xgboost_training_sessions
    WHERE total_samples IS NULL
       OR feature_count IS NULL
       OR test_auc IS NULL
""")
nulls = cursor.fetchone()[0]

if nulls > 0:
    print(f"\n⚠️ WARNING: Found {nulls} records with NULL values in critical fields!")
else:
    print("\n✅ All critical fields are populated!")

conn.close()