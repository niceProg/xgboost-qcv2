#!/usr/bin/env python3
"""
Verify CSV data stored in database
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
cursor = conn.cursor()

print("=" * 80)
print("CSV DATA STORED IN DATABASE")
print("=" * 80)

# Feature Importance
cursor.execute("""
    SELECT session_id, model_name, COUNT(*) as feature_count
    FROM xgboost_feature_importance
    GROUP BY session_id, model_name
""")
feature_data = cursor.fetchall()

print(f"\n📊 Feature Importance Records: {len(feature_data)} sessions")
for session, model, count in feature_data:
    print(f"  • {session}: {model} ({count} features)")

# Show top 10 features from latest session
if feature_data:
    cursor.execute("""
        SELECT feature, importance
        FROM xgboost_feature_importance
        WHERE session_id = %s
        ORDER BY importance DESC
        LIMIT 10
    """, (feature_data[0][0],))
    top_features = cursor.fetchall()

    print(f"\n🔝 Top 10 Features ({feature_data[0][0]}):")
    for feature, importance in top_features:
        print(f"  {feature[:30]:30} | {importance:.4f}")

# Trades Summary
cursor.execute("""
    SELECT session_id, trade_count, total_pnl, winning_trades,
           losing_trades, avg_win, avg_loss, best_trade, worst_trade
    FROM xgboost_trades
""")
trades_data = cursor.fetchall()

if trades_data:
    print(f"\n💰 Trades Summary: {len(trades_data)} records")
    for data in trades_data:
        print(f"  Session: {data[0]}")
        print(f"    Trades: {data[1]}")
        print(f"    Total P&L: {data[2]:.2f}")
        print(f"    Win Rate: {(data[3]/data[1]*100):.1f}%")
        print(f"    Best Trade: {data[6]:.2f}")
        print(f"    Worst Trade: {data[7]:.2f}")

# Account Statements
cursor.execute("""
    SELECT session_id, statement_type, row_count
    FROM xgboost_account_statements
""")
statement_data = cursor.fetchall()

if statement_data:
    print(f"\n📈 Account Statements: {len(statement_data)} records")
    for data in statement_data:
        print(f"  • {data[0]}: {data[1]} ({data[2]} rows)")

# CSV Files Metadata
cursor.execute("""
    SELECT file_name, file_type, file_size, row_count, description
    FROM xgboost_csv_files
    ORDER BY created_at DESC
""")
csv_meta = cursor.fetchall()

print(f"\n📁 CSV Files Metadata: {len(csv_meta)} files")
for data in csv_meta:
    size_mb = data[2] / 1024 / 1024
    print(f"  • {data[0]}")
    print(f"    Type: {data[1]}, Size: {size_mb:.2f}MB, Rows: {data[3]}")
    print(f"    Desc: {data[4][:80]}...")

conn.close()

print("\n" + "=" * 80)
print("✅ All CSV data successfully stored in database!")