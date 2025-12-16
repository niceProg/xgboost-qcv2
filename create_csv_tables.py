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

# Create table for trades (summary) - Keep this one
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

# Create table for account statements (sample due to size) - Keep this one
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

conn.commit()
conn.close()

print("✅ Created all CSV storage tables successfully!")