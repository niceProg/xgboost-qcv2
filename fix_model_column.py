#!/usr/bin/env python3
"""
Update model_data column to LONGBLOB to support larger models
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

    # Alter table to change model_data column to LONGBLOB
    sql = """
    ALTER TABLE xgboost_models
    MODIFY COLUMN model_data LONGBLOB
    """

    cursor.execute(sql)
    conn.commit()

    print("✅ Successfully updated model_data column to LONGBLOB")

    # Check table structure
    cursor.execute("DESCRIBE xgboost_models")
    columns = cursor.fetchall()
    for col in columns:
        if col[0] == 'model_data':
            print(f"Column {col[0]} type: {col[1]}")

    conn.close()

except Exception as e:
    print(f"❌ Error: {e}")