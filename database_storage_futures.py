#!/usr/bin/env python3
"""
Database storage manager for XGBoost futures training pipeline.
Stores all outputs (features, models, evaluation results) to database server.
This enables easy API access for QuantConnect integration.
Uses pymysql for database operations.
"""

import os
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
import numpy as np
import pymysql
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class DatabaseStorageFutures:
    """Manages database storage for XGBoost futures training pipeline outputs."""

    def __init__(self, db_config: Optional[Dict] = None, storage_path: str = './output_train_futures'):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)

        # Database configuration from .env
        if db_config is None:
            db_config = {
                'host': os.getenv('DB_HOST', '103.150.81.86'),
                'port': int(os.getenv('DB_PORT', 3306)),
                'database': os.getenv('DB_NAME', 'xgboostqc'),
                'user': os.getenv('DB_USER', 'xgboostqc'),
                'password': os.getenv('DB_PASSWORD', '6SPxBDwXH6WyxpfT')
            }

        self.db_config = db_config

    def _get_connection(self):
        """Get database connection."""
        try:
            conn = pymysql.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                database=self.db_config['database'],
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def _create_tables_if_not_exists(self):
        """Create tables if they don't exist."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                # Create xgboost_models table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS xgboost_models (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        model_name VARCHAR(200) NOT NULL,
                        model_version VARCHAR(50) DEFAULT 'futures',
                        model_file VARCHAR(500),
                        model_data LONGBLOB,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_model_version_created (model_version, created_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)

                # Create xgboost_dataset_summary table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS xgboost_dataset_summary (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        model_version VARCHAR(50) DEFAULT 'futures',
                        summary_file VARCHAR(500),
                        summary_data LONGBLOB,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_model_version_created (model_version, created_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)

                conn.commit()
                logger.info("Created xgboost_dataset_summary and xgboost_models tables")

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to create tables: {e}")
            raise
        finally:
            conn.close()

    def store_model(self,
                   model: Any,
                   model_name: str,
                   feature_names: List[str],
                   hyperparams: Dict,
                   train_score: float = None,
                   val_score: float = None,
                   cv_scores: Optional[List[float]] = None,
                   model_version: str = 'futures') -> int:
        """Store trained model in database. Returns model ID."""
        # Ensure tables exist
        self._create_tables_if_not_exists()

        conn = self._get_connection()
        try:
            # Serialize model
            model_data = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)

            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            created_at = datetime.now()

            # Insert model with explicit created_at
            with conn.cursor() as cursor:
                sql = """
                    INSERT INTO xgboost_models (model_name, model_version, model_file, model_data, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (
                    model_name,
                    model_version,
                    f"{model_name}_{timestamp}.pkl",
                    model_data,
                    created_at
                ))
                model_id = cursor.lastrowid
                conn.commit()

            logger.info(f"Stored model: {model_name} (version: {model_version}, ID: {model_id})")
            return model_id

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to store model: {e}")
            raise
        finally:
            conn.close()

    def store_dataset_summary(self,
                            summary_file: str,
                            summary_data: Optional[bytes] = None,
                            model_version: str = 'futures') -> int:
        """Store dataset summary to xgboost_dataset_summary table. Returns summary ID."""
        # Ensure tables exist
        self._create_tables_if_not_exists()

        conn = self._get_connection()
        try:
            created_at = datetime.now()

            # Insert dataset summary with explicit created_at
            with conn.cursor() as cursor:
                sql = """
                    INSERT INTO xgboost_dataset_summary (model_version, summary_file, summary_data, created_at)
                    VALUES (%s, %s, %s, %s)
                """
                cursor.execute(sql, (
                    model_version,
                    summary_file,
                    summary_data,
                    created_at
                ))
                summary_id = cursor.lastrowid
                conn.commit()

            logger.info(f"Stored dataset summary (version: {model_version}, ID: {summary_id})")
            return summary_id

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to store dataset summary: {e}")
            raise
        finally:
            conn.close()

    def load_latest_model(self, model_version: str = 'futures') -> tuple:
        """Load latest model from database using created_at ordering."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                sql = """
                    SELECT id, model_name, model_data
                    FROM xgboost_models
                    WHERE model_version = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                """
                cursor.execute(sql, (model_version,))
                result = cursor.fetchone()

                if result:
                    model = pickle.loads(result['model_data'])
                    return model, [], result['id']
                else:
                    raise ValueError(f"No model found in database for version: {model_version}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        finally:
            conn.close()

    def load_latest_dataset_summary(self, model_version: str = 'futures') -> tuple:
        """Load latest dataset summary from database."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                sql = """
                    SELECT id, summary_file, summary_data
                    FROM xgboost_dataset_summary
                    WHERE model_version = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                """
                cursor.execute(sql, (model_version,))
                result = cursor.fetchone()

                if result:
                    summary_data = result['summary_data']
                    if summary_data:
                        summary_text = summary_data.decode('utf-8')
                    else:
                        summary_text = ""
                    return summary_text, result['id'], result['summary_file']
                else:
                    raise ValueError(f"No dataset summary found in database for version: {model_version}")

        except Exception as e:
            logger.error(f"Failed to load dataset summary: {e}")
            raise
        finally:
            conn.close()
