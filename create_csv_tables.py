#!/usr/bin/env python3
"""
Create CSV storage tables in database
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

def create_csv_tables():
    """Create all CSV storage tables"""
    conn = pymysql.connect(**db_config)
    cursor = conn.cursor()

    # 1. Feature Importance Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS xgboost_feature_importance (
            id INT AUTO_INCREMENT PRIMARY KEY,
            session_id VARCHAR(100),
            model_name VARCHAR(255),
            feature VARCHAR(255),
            importance FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_session_model (session_id, model_name),
            INDEX idx_feature (feature(50))
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)

    # 2. Trades Summary Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS xgboost_trades (
            id INT AUTO_INCREMENT PRIMARY KEY,
            session_id VARCHAR(100),
            trade_count INT,
            total_pnl DECIMAL(15,4),
            winning_trades INT,
            losing_trades INT,
            avg_win DECIMAL(10,4),
            avg_loss DECIMAL(10,4),
            best_trade DECIMAL(10,4),
            worst_trade DECIMAL(10,4),
            file_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_session (session_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)

    # 3. Account Statements Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS xgboost_account_statements (
            id INT AUTO_INCREMENT PRIMARY KEY,
            session_id VARCHAR(100),
            statement_type ENUM('equity', 'cash'),
            row_count INT,
            columns_info JSON,
            sample_data JSON,
            file_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_session (session_id, statement_type)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)

    # 4. CSV Files Metadata Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS xgboost_csv_files (
            id INT AUTO_INCREMENT PRIMARY KEY,
            session_id VARCHAR(100),
            file_name VARCHAR(255),
            file_path TEXT,
            file_size BIGINT,
            file_type VARCHAR(100),
            row_count INT,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_session (session_id),
            INDEX idx_file_type (file_type)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)

    conn.commit()
    conn.close()

    print("✅ All CSV storage tables created successfully!")

def test_csv_functionality():
    """Test if CSV storage works with actual data"""
    from csv_storage import CSVStorage
    from pathlib import Path

    print("\n=== Testing CSV Storage Functionality ===")

    # Check for existing CSV files
    output_dir = Path('./output_train')
    csv_files = list(output_dir.glob("*.csv"))

    print(f"\nFound {len(csv_files)} CSV files:")
    for csv_file in csv_files:
        print(f"   - {csv_file.name}")

    if not csv_files:
        print("\n⚠️ No CSV files found. Need to run pipeline first to generate CSV files.")
        print("   Run: ./simple_run.sh")
        return

    # Get latest session
    conn = pymysql.connect(**db_config)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT session_id FROM xgboost_training_sessions "
        "WHERE status = 'completed' ORDER BY created_at DESC LIMIT 1"
    )
    result = cursor.fetchone()

    if not result:
        print("\n⚠️ No completed session found")
        return

    session_id = result[0]
    model_files = list(output_dir.glob("*.joblib"))
    model_name = model_files[0].name if model_files else "test_model"

    print(f"\nTesting CSV storage for session: {session_id}")
    print(f"Model: {model_name}")

    # Use CSVStorage to save all outputs
    csv_storage = CSVStorage()
    csv_storage.save_all_csv_outputs(output_dir, session_id, model_name)

    print("\n✅ CSV files saved to database!")

    # Verify data was saved
    tables = [
        ('xgboost_feature_importance', 'Feature Importance'),
        ('xgboost_trades', 'Trades Summary'),
        ('xgboost_account_statements', 'Account Statements'),
        ('xgboost_csv_files', 'CSV Files Metadata')
    ]

    for table_name, display_name in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"   {display_name:25}: {count} records")

    conn.close()

if __name__ == "__main__":
    print("Creating CSV storage tables...")
    create_csv_tables()
    test_csv_functionality()
