#!/usr/bin/env python3
"""
Create tables to store CSV outputs
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

conn = pymysql.connect(**db_config)
cursor = conn.cursor()

# Create table for feature importance
cursor.execute("""
CREATE TABLE IF NOT EXISTS xgboost_feature_importance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    model_name VARCHAR(200) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    feature VARCHAR(100) NOT NULL,
    importance FLOAT NOT NULL,
    INDEX idx_session (session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
""")

# Create table for merged data (sample only due to size)
cursor.execute("""
CREATE TABLE IF NOT EXISTS xgboost_merged_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    row_count INT NOT NULL,
    columns_info TEXT,
    sample_data LONGTEXT,
    file_path VARCHAR(500),
    INDEX idx_session (session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
""")

# Create table for trade events (sample due to size)
cursor.execute("""
CREATE TABLE IF NOT EXISTS xgboost_trade_events (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    event_count INT NOT NULL,
    columns_info TEXT,
    sample_data LONGTEXT,
    file_path VARCHAR(500),
    INDEX idx_session (session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
""")

# Create table for trades (summary)
cursor.execute("""
CREATE TABLE IF NOT EXISTS xgboost_trades (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    trade_count INT NOT NULL,
    total_pnl FLOAT DEFAULT 0,
    winning_trades INT DEFAULT 0,
    losing_trades INT DEFAULT 0,
    avg_win FLOAT DEFAULT 0,
    avg_loss FLOAT DEFAULT 0,
    best_trade FLOAT DEFAULT 0,
    worst_trade FLOAT DEFAULT 0,
    file_path VARCHAR(500),
    INDEX idx_session (session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
""")

# Create table for account statements (sample due to size)
cursor.execute("""
CREATE TABLE IF NOT EXISTS xgboost_account_statements (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    statement_type ENUM('equity', 'cash') NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    row_count INT NOT NULL,
    columns_info TEXT,
    sample_data LONGTEXT,
    file_path VARCHAR(500),
    INDEX idx_session (session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
""")

# Create table for all CSV files metadata
cursor.execute("""
CREATE TABLE IF NOT EXISTS xgboost_csv_files (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    file_name VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size INT NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    row_count INT DEFAULT 0,
    description TEXT,
    INDEX idx_session (session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
""")

conn.commit()
conn.close()

print("✅ Created all CSV storage tables successfully!")