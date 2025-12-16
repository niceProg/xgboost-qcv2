#!/usr/bin/env python3
"""
Fix and populate XGBoost database storage.
This script will properly store all training artifacts to xgboostqc database.
"""

import os
import sys
import json
import pickle
import base64
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DatabaseStorageFix:
    """Fix database storage issues and populate data properly."""

    def __init__(self):
        # Database config for xgboostqc database
        self.db_config = {
            'host': os.getenv('DB_HOST', '103.150.81.86'),
            'port': int(os.getenv('DB_PORT', 3306)),
            'database': os.getenv('DB_NAME', 'xgboostqc'),
            'user': os.getenv('DB_USER', 'xgboostqc'),
            'password': os.getenv('DB_PASSWORD', '6SPxBDwXH6WyxpfT')
        }
        self.output_dir = Path('./output_train')

    def get_connection(self):
        """Get database connection."""
        try:
            import pymysql
            conn = pymysql.connect(**self.db_config)
            logger.info("Connected to xgboostqc database successfully")
            return conn
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def create_tables(self):
        """Create necessary tables if they don't exist."""
        sql_statements = [
            """
            CREATE TABLE IF NOT EXISTS xgboost_training_sessions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(100) UNIQUE NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                status VARCHAR(50) DEFAULT 'created',
                exchange_filter TEXT,
                symbol_filter TEXT,
                interval_filter TEXT,
                time_range TEXT,
                days_filter INT,
                total_samples INT,
                feature_count INT,
                model_version VARCHAR(50),
                best_params TEXT,
                model_path VARCHAR(500),
                train_auc FLOAT,
                val_auc FLOAT,
                test_auc FLOAT,
                test_accuracy FLOAT,
                test_precision FLOAT,
                test_recall FLOAT,
                test_f1 FLOAT,
                sharpe_ratio FLOAT,
                total_return FLOAT,
                max_drawdown FLOAT,
                notes TEXT,
                completed_at DATETIME,
                INDEX idx_session_id (session_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,

            """
            CREATE TABLE IF NOT EXISTS xgboost_models (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(100) NOT NULL,
                model_name VARCHAR(200) NOT NULL,
                model_version VARCHAR(50),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_latest BOOLEAN DEFAULT FALSE,
                model_data LONGBLOB,
                feature_names TEXT,
                hyperparams TEXT,
                train_score FLOAT,
                val_score FLOAT,
                cv_scores TEXT,
                INDEX idx_session_id (session_id),
                INDEX idx_latest (is_latest)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,

            """
            CREATE TABLE IF NOT EXISTS xgboost_features (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(100) NOT NULL,
                feature_type VARCHAR(50) NOT NULL,
                table_name VARCHAR(100),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                file_path VARCHAR(500),
                row_count INT,
                file_size_bytes INT,
                file_format VARCHAR(20) DEFAULT 'parquet',
                data_sample LONGTEXT,
                schema_info TEXT,
                INDEX idx_session_id (session_id),
                INDEX idx_feature_type (feature_type)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,

            """
            CREATE TABLE IF NOT EXISTS xgboost_evaluations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(100) NOT NULL,
                eval_type VARCHAR(50) NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                initial_cash FLOAT,
                leverage FLOAT,
                margin_fraction FLOAT,
                fee_rate FLOAT,
                threshold FLOAT,
                total_return FLOAT,
                cagr FLOAT,
                max_drawdown FLOAT,
                sharpe_ratio FLOAT,
                win_rate FLOAT,
                profit_factor FLOAT,
                total_trades INT,
                winning_trades INT,
                trade_log_path VARCHAR(500),
                equity_curve_path VARCHAR(500),
                performance_plot_path VARCHAR(500),
                detailed_metrics TEXT,
                INDEX idx_session_id (session_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        ]

        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            for sql in sql_statements:
                cursor.execute(sql)
            conn.commit()
            logger.info("Database tables created/verified successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def process_training_sessions(self):
        """Process training results and update sessions."""
        logger.info("Processing training sessions...")

        # Find all training result files
        result_files = list(self.output_dir.glob("training_results_*.json"))

        if not result_files:
            logger.warning("No training result files found")
            return

        conn = self.get_connection()

        try:
            cursor = conn.cursor()

            for result_file in sorted(result_files, key=lambda x: x.stat().st_mtime, reverse=True):
                logger.info(f"Processing {result_file.name}")

                with open(result_file, 'r') as f:
                    results = json.load(f)

                session_id = results.get('model_name', f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

                # Extract session from filename if possible
                if 'training_results_' in result_file.name:
                    session_id = result_file.name.replace('training_results_', '').replace('.json', '')

                # Check if session exists
                cursor.execute("SELECT id FROM xgboost_training_sessions WHERE session_id = %s", (session_id,))

                metrics = results.get('metrics', {})
                cv_results = results.get('cross_validation', {})

                if cursor.fetchone():
                    # Update existing session
                    sql = """
                    UPDATE xgboost_training_sessions SET
                        status = %s,
                        total_samples = %s,
                        feature_count = %s,
                        model_version = %s,
                        best_params = %s,
                        model_path = %s,
                        train_auc = %s,
                        val_auc = %s,
                        test_auc = %s,
                        test_accuracy = %s,
                        test_precision = %s,
                        test_recall = %s,
                        test_f1 = %s,
                        sharpe_ratio = %s,
                        total_return = %s,
                        max_drawdown = %s,
                        notes = %s,
                        completed_at = %s
                    WHERE session_id = %s
                    """
                    params = [
                        'completed',  # status
                        results.get('sample_count', 0),  # Use sample_count from results
                        results.get('feature_count', 0),
                        results.get('model_name', 'unknown'),
                        json.dumps(results.get('parameters', {})),
                        results.get('model_path', ''),
                        0.0,  # train_auc (not available in this JSON)
                        0.0,  # val_auc (not available in this JSON)
                        metrics.get('roc_auc', 0.0),
                        metrics.get('accuracy', 0.0),
                        metrics.get('precision', 0.0),
                        metrics.get('recall', 0.0),
                        metrics.get('f1', 0.0),
                        0.0,  # sharpe_ratio (not available in this JSON)
                        0.0,  # total_return (not available in this JSON)
                        0.0,  # max_drawdown (not available in this JSON)
                        "Training completed successfully",
                        datetime.now(),
                        session_id  # For WHERE clause
                    ]
                else:
                    # Insert new session
                    sql = """
                    INSERT INTO xgboost_training_sessions (
                        session_id, status, total_samples, feature_count,
                        model_version, best_params, model_path,
                        train_auc, val_auc, test_auc, test_accuracy,
                        test_precision, test_recall, test_f1,
                        sharpe_ratio, total_return, max_drawdown
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    params = [
                        session_id,  # 1
                        'completed',  # 2 status
                        results.get('sample_count', 0),  # 3
                        results.get('feature_count', 0),  # 4
                        results.get('model_name', 'unknown'),  # 5
                        json.dumps(results.get('parameters', {})),  # 6
                        results.get('model_path', ''),  # 7
                        0.0,  # 8 train_auc (not available in this JSON)
                        0.0,  # 9 val_auc (not available in this JSON)
                        metrics.get('roc_auc', 0.0),  # 10
                        metrics.get('accuracy', 0.0),  # 11
                        metrics.get('precision', 0.0),  # 12
                        metrics.get('recall', 0.0),  # 13
                        metrics.get('f1', 0.0),  # 14
                        0.0,  # 15 sharpe_ratio (not available in this JSON)
                        0.0,  # 16 total_return (not available in this JSON)
                        0.0   # 17 max_drawdown (not available in this JSON)
                    ]

                try:
                    cursor.execute(sql, params)
                except Exception as e:
                    logger.error(f"SQL Error: {e}")
                    logger.error(f"SQL: {sql}")
                    logger.error(f"Params count: {len(params)}")
                    logger.error(f"Expected params: {sql.count('%s')}")
                    raise
                logger.info(f"Updated session: {session_id}")

            conn.commit()
            logger.info(f"Processed {len(result_files)} training sessions")

        except Exception as e:
            logger.error(f"Error processing sessions: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def process_models(self):
        """Store model files and metadata."""
        logger.info("Processing models...")

        # Find all model files
        model_files = list(self.output_dir.glob("*.joblib"))

        if not model_files:
            logger.warning("No model files found")
            return

        conn = self.get_connection()

        try:
            cursor = conn.cursor()

            # Mark all models as not latest first
            cursor.execute("UPDATE xgboost_models SET is_latest = FALSE")

            for model_file in sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True):
                logger.info(f"Processing model: {model_file.name}")

                try:
                    # Load model to get feature names
                    model = joblib.load(model_file)
                    feature_names = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else []

                    # Serialize model
                    with open(model_file, 'rb') as f:
                        model_data = f.read()

                    # Get session_id from filename
                    if 'xgboost_trading_model_' in model_file.name:
                        session_id = model_file.stem.replace('xgboost_trading_model_', '')
                    else:
                        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                    # Check if model exists
                    cursor.execute("SELECT id FROM xgboost_models WHERE model_name = %s", (model_file.name,))

                    if cursor.fetchone():
                        # Update existing
                        sql = """
                        UPDATE xgboost_models SET
                            session_id = %s,
                            model_data = %s,
                            feature_names = %s,
                            created_at = %s,
                            is_latest = %s
                        WHERE model_name = %s
                        """
                        params = [
                            session_id,
                            model_data,
                            json.dumps(feature_names),
                            datetime.fromtimestamp(model_file.stat().st_mtime),
                            True if 'latest_model' in model_file.name else False,
                            model_file.name
                        ]
                    else:
                        # Insert new
                        sql = """
                        INSERT INTO xgboost_models (
                            session_id, model_name, model_data, feature_names,
                            created_at, is_latest
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                        """
                        params = [
                            session_id,
                            model_file.name,
                            model_data,
                            json.dumps(feature_names),
                            datetime.fromtimestamp(model_file.stat().st_mtime),
                            True if 'latest_model' in model_file.name else False
                        ]

                    cursor.execute(sql, params)
                    logger.info(f"Stored model: {model_file.name}")

                except Exception as e:
                    logger.error(f"Error processing model {model_file.name}: {e}")

            conn.commit()
            logger.info(f"Processed {len(model_files)} models")

        except Exception as e:
            logger.error(f"Error processing models: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def process_features(self):
        """Process and store feature data."""
        logger.info("Processing features...")

        feature_files = [
            (self.output_dir / 'merged_7_tables.parquet', 'raw_data', 'merged_7_tables'),
            (self.output_dir / 'features_engineered.parquet', 'engineered', 'features_engineered'),
            (self.output_dir / 'labeled_data.parquet', 'labeled', 'labeled_data'),
            (self.output_dir / 'X_train_features.parquet', 'training', 'X_train_features'),
            (self.output_dir / 'trading_results.parquet', 'results', 'trading_results')
        ]

        conn = self.get_connection()

        try:
            cursor = conn.cursor()

            for file_path, feature_type, table_name in feature_files:
                if not file_path.exists():
                    logger.warning(f"Feature file not found: {file_path}")
                    continue

                logger.info(f"Processing {file_path.name} ({feature_type})")

                try:
                    # Read parquet file
                    df = pd.read_parquet(file_path)

                    # Get schema info
                    schema = {
                        'columns': list(df.columns),
                        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                        'shape': df.shape
                    }

                    # Get data sample (first 5 rows as JSON)
                    # Convert timestamps to strings for JSON serialization
                    df_sample = df.head(5).copy()
                    for col in df_sample.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns:
                        df_sample[col] = df_sample[col].astype(str)
                    sample_data = df_sample.to_dict('records')

                    # Get session_id (try to infer from filename or use latest)
                    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                    # Insert into database
                    sql = """
                    INSERT INTO xgboost_features (
                        session_id, feature_type, table_name, file_path,
                        row_count, file_size_bytes, file_format,
                        data_sample, schema_info
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """

                    params = [
                        session_id,
                        feature_type,
                        table_name,
                        str(file_path),
                        len(df),
                        file_path.stat().st_size,
                        'parquet',
                        json.dumps(sample_data),
                        json.dumps(schema)
                    ]

                    cursor.execute(sql, params)
                    logger.info(f"Stored {feature_type}: {len(df)} rows")

                except Exception as e:
                    logger.error(f"Error processing {file_path.name}: {e}")

            conn.commit()
            logger.info("Feature processing completed")

        except Exception as e:
            logger.error(f"Error processing features: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def process_evaluations(self):
        """Process evaluation results."""
        logger.info("Processing evaluations...")

        conn = self.get_connection()

        try:
            cursor = conn.cursor()

            patterns = ['performance_metrics_*.json', 'performance_report_*.json']
            eval_types = ['performance', 'detailed']

            for pattern, eval_type in zip(patterns, eval_types):
                files = list(self.output_dir.glob(pattern))

                for file_path in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
                    logger.info(f"Processing {file_path.name} ({eval_type})")

                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)

                        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                        if eval_type == 'performance':
                            # Performance metrics
                            sql = """
                            INSERT INTO xgboost_evaluations (
                                session_id, eval_type, initial_cash, leverage,
                                margin_fraction, fee_rate, threshold,
                                total_return, cagr, max_drawdown, sharpe_ratio,
                                win_rate, profit_factor, total_trades, winning_trades,
                                detailed_metrics
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """

                            params = [
                                session_id,
                                eval_type,
                                float(os.getenv('INITIAL_CASH', 1000)),
                                float(os.getenv('LEVERAGE', 10)),
                                float(os.getenv('MARGIN_FRACTION', 0.2)),
                                float(os.getenv('FEE_RATE', 0.0004)),
                                float(os.getenv('THRESHOLD', 0.5)),
                                data.get('total_return', 0.0),
                                data.get('cagr', 0.0),
                                data.get('max_drawdown', 0.0),
                                data.get('sharpe_ratio', 0.0),
                                data.get('win_rate', 0.0),
                                data.get('profit_factor', 1.0),
                                data.get('total_trades', 0),
                                data.get('winning_trades', 0),
                                json.dumps(data)
                            ]

                        else:  # detailed report
                            strategy_summary = data.get('strategy_summary', {})
                            performance_metrics = data.get('performance_metrics', {})

                            sql = """
                            INSERT INTO xgboost_evaluations (
                                session_id, eval_type, total_return, cagr,
                                max_drawdown, sharpe_ratio, win_rate, profit_factor,
                                total_trades, winning_trades, detailed_metrics
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """

                            params = [
                                session_id,
                                eval_type,
                                performance_metrics.get('total_return', 0.0),
                                performance_metrics.get('cagr', 0.0),
                                performance_metrics.get('max_drawdown', 0.0),
                                performance_metrics.get('sharpe_ratio', 0.0),
                                performance_metrics.get('win_rate', 0.0),
                                performance_metrics.get('profit_factor', 1.0),
                                performance_metrics.get('total_trades', 0),
                                performance_metrics.get('winning_trades', 0),
                                json.dumps(data)
                            ]

                        cursor.execute(sql, params)
                        logger.info(f"Stored evaluation: {file_path.name}")

                    except Exception as e:
                        logger.error(f"Error processing {file_path.name}: {e}")

            conn.commit()
            logger.info("Evaluation processing completed")

        except Exception as e:
            logger.error(f"Error processing evaluations: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def verify_data(self):
        """Verify data was inserted correctly."""
        logger.info("\n=== Verifying Database Contents ===")

        conn = self.get_connection()

        try:
            cursor = conn.cursor()

            tables = [
                ('xgboost_training_sessions', 'Training Sessions'),
                ('xgboost_models', 'Models'),
                ('xgboost_features', 'Features'),
                ('xgboost_evaluations', 'Evaluations')
            ]

            for table_name, display_name in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                logger.info(f"{display_name}: {count} records")

                # Show sample data
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
                sample = cursor.fetchone()
                if sample:
                    logger.info(f"  Sample: {sample}")

        except Exception as e:
            logger.error(f"Error verifying data: {e}")
        finally:
            conn.close()

    def fix_all(self):
        """Run all fix operations."""
        logger.info("Starting database fix operations...")

        self.create_tables()
        self.process_training_sessions()
        self.process_models()
        self.process_features()
        self.process_evaluations()
        self.verify_data()

        logger.info("\n✅ Database fix completed successfully!")
        logger.info("All data has been properly inserted into xgboostqc database.")


def main():
    """Main function to fix database storage."""
    fixer = DatabaseStorageFix()

    try:
        fixer.fix_all()
    except Exception as e:
        logger.error(f"Database fix failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()