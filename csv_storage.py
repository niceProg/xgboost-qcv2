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

    def save_feature_importance(self, csv_path: Path, session_id: str, model_name: str):
        """Save feature importance CSV to database"""
        try:
            df = pd.read_csv(csv_path)

            conn = self.get_connection()
            cursor = conn.cursor()

            # Delete existing entries for this model
            cursor.execute(
                "DELETE FROM xgboost_feature_importance WHERE session_id = %s AND model_name = %s",
                (session_id, model_name)
            )

            # Insert feature importance data
            for _, row in df.iterrows():
                cursor.execute("""
                    INSERT INTO xgboost_feature_importance (
                        session_id, model_name, feature, importance
                    ) VALUES (%s, %s, %s, %s)
                """, (session_id, model_name, row['feature'], row['importance']))

            conn.commit()
            conn.close()
            logger.info(f"Saved {len(df)} feature importance records")

        except Exception as e:
            logger.error(f"Error saving feature importance: {e}")

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

    def save_csv_metadata(self, csv_path: Path, session_id: str, file_type: str, description: str = None):
        """Save CSV file metadata to database"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Get row count
            df = pd.read_csv(csv_path, nrows=10)
            row_count = len(df) if len(df) > 0 else 0

            # Get file size
            file_size = csv_path.stat().st_size

            cursor.execute("""
                INSERT INTO xgboost_csv_files (
                    session_id, file_name, file_path, file_size,
                    file_type, row_count, description
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                session_id, csv_path.name, str(csv_path), file_size,
                file_type, row_count, description
            ))

            conn.commit()
            conn.close()
            logger.info(f"Saved CSV metadata: {csv_path.name}")

        except Exception as e:
            logger.error(f"Error saving CSV metadata: {e}")

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
        """Save all CSV outputs from a training session"""
        logger.info(f"Saving CSV outputs to database for session {session_id}...")

        # Save feature importance (any feature importance file)
        feature_files = list(output_dir.glob("feature_importance_*.csv"))
        if feature_files:
            # Use the latest feature importance file
            latest_feature_file = max(feature_files, key=lambda x: x.stat().st_mtime)
            self.save_feature_importance(latest_feature_file, session_id, model_name)
            logger.info(f"Saved feature importance from {latest_feature_file.name}")

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

        # Save metadata for all CSV files
        csv_files = list(output_dir.glob("*.csv"))
        for csv_file in csv_files:
            file_type = csv_file.name.split('_')[0] if '_' in csv_file.name else 'unknown'
            description = f"CSV output from XGBoost pipeline - {csv_file.name}"

            if 'feature_importance' in csv_file.name:
                description = "Feature importance scores from trained XGBoost model"
            elif 'trades' in csv_file.name:
                description = "Summary of all completed trades with P&L"
            elif 'trade_events' in csv_file.name:
                description = "Detailed log of all trading events"
            elif 'rekening_koran' in csv_file.name:
                description = "Account statement showing equity/cash balance over time"
            elif 'merged_7_tables' in csv_file.name:
                description = "Merged data from all 7 source tables"

            self.save_csv_metadata(csv_file, session_id, file_type, description)

        logger.info("CSV outputs saved to database successfully!")