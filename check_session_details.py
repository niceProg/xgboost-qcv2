#!/usr/bin/env python3
"""
Check detailed xgboost_training_sessions data
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
cursor = conn.cursor(pymysql.cursors.DictCursor)

cursor.execute("SELECT * FROM xgboost_training_sessions")
record = cursor.fetchone()

print("=== xgboost_training_sessions Details ===")
print(f"Session ID: {record['session_id']}")
print(f"Status: {record['status']}")
print(f"Created: {record['created_at']}")
print(f"Completed: {record['completed_at']}")
print(f"\nData Info:")
print(f"  Total Samples: {record['total_samples']:,}")
print(f"  Feature Count: {record['feature_count']}")
print(f"\nModel Info:")
print(f"  Model Version: {record['model_version']}")
print(f"  Model Path: {record['model_path']}")
print(f"\nPerformance Metrics:")
print(f"  Train AUC: {record['train_auc']:.4f if record['train_auc'] else 'N/A'}")
print(f"  Val AUC: {record['val_auc']:.4f if record['val_auc'] else 'N/A'}")
print(f"  Test AUC: {record['test_auc']:.4f if record['test_auc'] else 'N/A'}")
print(f"  Test Accuracy: {record['test_accuracy']:.4f if record['test_accuracy'] else 'N/A'}")
print(f"  Test Precision: {record['test_precision']:.4f if record['test_precision'] else 'N/A'}")
print(f"  Test Recall: {record['test_recall']:.4f if record['test_recall'] else 'N/A'}")
print(f"  Test F1: {record['test_f1']:.4f if record['test_f1'] else 'N/A'}")
print(f"\nTrading Performance:")
print(f"  Total Return: {record['total_return']:.2% if record['total_return'] else 'N/A'}")
print(f"  Sharpe Ratio: {record['sharpe_ratio']:.2f if record['sharpe_ratio'] else 'N/A'}")
print(f"  Max Drawdown: {record['max_drawdown']:.2% if record['max_drawdown'] else 'N/A'}")
print(f"\nFilters:")
print(f"  Exchange: {record['exchange_filter']}")
print(f"  Symbol: {record['symbol_filter']}")
print(f"  Interval: {record['interval_filter']}")
print(f"\nNotes:")
print(record['notes'][:500] + "..." if len(record['notes']) > 500 else record['notes'])

conn.close()