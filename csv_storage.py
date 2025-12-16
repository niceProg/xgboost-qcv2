#!/usr/bin/env python3
"""
CSV Storage module for saving pipeline outputs to database
"""

import os
import pymysql
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()

class CSVStorage:
    """Handle saving CSV outputs to database"""

    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', '103.150.81.86'),
            'port': int(os.getenv('DB_PORT', 3306)),
            'database': os.getenv('DB_NAME', 'xgboostqc'),
            'user': os.getenv('DB_USER', 'xgboostqc'),
            'password': os.getenv('DB_PASSWORD', '6SPxBDwXH6WyxpfT')
        }

    def get_connection(self):
        """Get database connection"""
        return pymysql.connect(**self.db_config)

    # Feature importance saving removed as requested

    def save_trades_summary(self, csv_path: Path, session_id: str):
        """Save trades summary CSV to database"""
        try:
            df = pd.read_csv(csv_path)

            conn = self.get_connection()
            cursor = conn.cursor()

            # Calculate summary statistics
            total_pnl = df['pnl'].sum()
            winning_trades = len(df[df['pnl'] > 0])
            losing_trades = len(df[df['pnl'] < 0])
            avg_win = df[df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = df[df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
            best_trade = df['pnl'].max()
            worst_trade = df['pnl'].min()

            # Delete existing
            cursor.execute(
                "DELETE FROM xgboost_trades WHERE session_id = %s",
                (session_id,)
            )

            # Insert summary
            cursor.execute("""
                INSERT INTO xgboost_trades (
                    session_id, trade_count, total_pnl, winning_trades,
                    losing_trades, avg_win, avg_loss, best_trade, worst_trade, file_path
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                session_id, len(df), total_pnl, winning_trades,
                losing_trades, avg_win, avg_loss, best_trade, worst_trade, str(csv_path)
            ))

            conn.commit()
            conn.close()
            logger.info(f"Saved trades summary: {len(df)} trades, PnL: {total_pnl:.2f}")

        except Exception as e:
            logger.error(f"Error saving trades summary: {e}")

    # CSV metadata saving removed as requested

    def save_account_statement_sample(self, csv_path: Path, session_id: str, statement_type: str):
        """Save sample of account statement"""
        try:
            df = pd.read_csv(csv_path)

            conn = self.get_connection()
            cursor = conn.cursor()

            # Get sample data (first 10 rows)
            sample_df = df.head(10)
            sample_json = sample_df.to_json(orient='records')
            columns_info = json.dumps(list(df.columns))

            # Delete existing
            cursor.execute("""
                DELETE FROM xgboost_account_statements
                WHERE session_id = %s AND statement_type = %s
            """, (session_id, statement_type))

            # Insert sample
            cursor.execute("""
                INSERT INTO xgboost_account_statements (
                    session_id, statement_type, row_count, columns_info,
                    sample_data, file_path
                ) VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                session_id, statement_type, len(df), columns_info,
                sample_json, str(csv_path)
            ))

            conn.commit()
            conn.close()
            logger.info(f"Saved account statement sample: {statement_type}")

        except Exception as e:
            logger.error(f"Error saving account statement: {e}")

    def save_all_csv_outputs(self, output_dir: Path, session_id: str, model_name: str):
        """Save selected CSV outputs from a training session"""
        logger.info("Saving CSV outputs to database...")

        # Save trades summary
        trades_file = output_dir / "trades.csv"
        if trades_file.exists():
            self.save_trades_summary(trades_file, session_id)

        # Save account statements
        equity_file = output_dir / "rekening_koran.csv"
        cash_file = output_dir / "rekening_koran_cash.csv"

        if equity_file.exists():
            self.save_account_statement_sample(equity_file, session_id, 'equity')

        if cash_file.exists():
            self.save_account_statement_sample(cash_file, session_id, 'cash')

        logger.info("CSV outputs saved to database successfully!")