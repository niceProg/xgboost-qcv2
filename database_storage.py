#!/usr/bin/env python3
"""
Database storage manager for XGBoost training pipeline.
Stores all outputs (features, models, evaluation results) to database server.
This enables easy API access for QuantConnect integration.
"""

import os
import json
import pickle
import base64
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import pandas as pd
import numpy as np
import pymysql
from sqlalchemy import create_engine, text, Column, String, DateTime, Integer, Float, Text, LargeBinary, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.mysql import MEDIUMBLOB, LONGTEXT, LONGBLOB
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)
Base = declarative_base()

# Database tables for storing training artifacts
class TrainingSession(Base):
    __tablename__ = 'xgboost_training_sessions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), unique=True, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(50), default='created')  # created, training, completed, failed

    # Training parameters
    exchange_filter = Column(Text)
    symbol_filter = Column(Text)
    interval_filter = Column(Text)
    time_range = Column(Text)
    days_filter = Column(Integer)

    # Data statistics
    total_samples = Column(Integer)
    feature_count = Column(Integer)

    # Model metadata
    model_version = Column(String(50))
    best_params = Column(Text)  # JSON
    model_path = Column(String(500))

    # Performance metrics
    train_auc = Column(Float)
    val_auc = Column(Float)
    test_auc = Column(Float)
    test_accuracy = Column(Float)
    test_precision = Column(Float)
    test_recall = Column(Float)
    test_f1 = Column(Float)
    sharpe_ratio = Column(Float)
    total_return = Column(Float)
    max_drawdown = Column(Float)

    # Additional metadata
    notes = Column(Text)
    completed_at = Column(DateTime)

class ModelStorage(Base):
    __tablename__ = 'xgboost_models'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    model_name = Column(String(200), nullable=False)
    model_version = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    is_latest = Column(Boolean, default=False)

    # Model serialization (using BLOB)
    model_data = Column(LONGBLOB)  # pickled model

    # Model metadata
    feature_names = Column(Text)  # JSON list
    hyperparams = Column(Text)  # JSON

    # Training metrics
    train_score = Column(Float)
    val_score = Column(Float)
    cv_scores = Column(Text)  # JSON

class FeatureStorage(Base):
    __tablename__ = 'xgboost_features'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    feature_type = Column(String(50), nullable=False)  # raw, engineered, labeled
    table_name = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)

    # Data storage
    file_path = Column(String(500))
    row_count = Column(Integer)
    file_size_bytes = Column(Integer)
    file_format = Column(String(20), default='parquet')  # parquet, csv

    # For direct database storage (optional)
    data_sample = Column(LONGTEXT)  # JSON sample of first 10 rows
    schema_info = Column(Text)  # JSON schema

class EvaluationResult(Base):
    __tablename__ = 'xgboost_evaluations'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    eval_type = Column(String(50), nullable=False)  # backtest, walk_forward, etc.
    created_at = Column(DateTime, default=datetime.utcnow)

    # Evaluation parameters
    initial_cash = Column(Float)
    leverage = Column(Float)
    margin_fraction = Column(Float)
    fee_rate = Column(Float)
    threshold = Column(Float)

    # Performance metrics
    total_return = Column(Float)
    cagr = Column(Float)
    max_drawdown = Column(Float)
    sharpe_ratio = Column(Float)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    total_trades = Column(Integer)
    winning_trades = Column(Integer)

    # Detailed results
    trade_log_path = Column(String(500))
    equity_curve_path = Column(String(500))
    performance_plot_path = Column(String(500))

    # Full results as JSON
    detailed_metrics = Column(Text)

class DatabaseStorage:
    """Manages database storage for XGBoost training pipeline outputs."""

    def __init__(self, db_config: Optional[Dict] = None, storage_path: str = './output_train'):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)

        # Database configuration
        if db_config is None:
            db_config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': int(os.getenv('DB_PORT', 3306)),
                'database': os.getenv('DB_NAME', 'xgboost_training'),
                'user': os.getenv('DB_USER', 'root'),
                'password': os.getenv('DB_PASSWORD', '')
            }

        self.db_config = db_config
        self.engine = None
        self.SessionLocal = None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._initialize_database()

    def _initialize_database(self):
        """Initialize database connection and create tables."""
        try:
            # Create connection string
            connection_string = (
                f"mysql+pymysql://{self.db_config['user']}:{self.db_config['password']}"
                f"@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
                f"?charset=utf8mb4"
            )

            # Create engine
            self.engine = create_engine(
                connection_string,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                echo=False  # Set to True for SQL logging
            )

            # Create tables
            Base.metadata.create_all(bind=self.engine)

            # Create session factory
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

            logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def get_session(self) -> Session:
        """Get a database session."""
        if self.SessionLocal is None:
            raise RuntimeError("Database not initialized")
        return self.SessionLocal()

    def create_training_session(self,
                             exchange_filter: Optional[List[str]] = None,
                             symbol_filter: Optional[List[str]] = None,
                             interval_filter: Optional[List[str]] = None,
                             time_range: Optional[tuple] = None,
                             days_filter: Optional[int] = None,
                             notes: Optional[str] = None) -> str:
        """Create a new training session record."""
        db = self.get_session()
        try:
            # Serialize filters to JSON
            filters_json = json.dumps({
                'exchange': exchange_filter or [],
                'symbol': symbol_filter or [],
                'interval': interval_filter or [],
                'time_range': time_range,
                'days': days_filter
            })

            session = TrainingSession(
                session_id=self.session_id,
                exchange_filter=json.dumps(exchange_filter) if exchange_filter else None,
                symbol_filter=json.dumps(symbol_filter) if symbol_filter else None,
                interval_filter=json.dumps(interval_filter) if interval_filter else None,
                time_range=json.dumps(time_range) if time_range else None,
                days_filter=days_filter,
                notes=notes
            )

            db.add(session)
            db.commit()

            logger.info(f"Created training session: {self.session_id}")
            return self.session_id

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create training session: {e}")
            raise
        finally:
            db.close()

    def store_file(self,
                  file_path: Union[str, Path],
                  table_name: str,
                  feature_type: str,
                  description: Optional[str] = None) -> str:
        """Store file information in database and return the stored path."""
        file_path = Path(file_path)
        db = self.get_session()

        try:
            # Get file stats
            stat = file_path.stat()
            row_count = 0

            # Try to get row count for data files
            try:
                if file_path.suffix == '.parquet':
                    df = pd.read_parquet(file_path)
                    row_count = len(df)
                elif file_path.suffix == '.csv':
                    df = pd.read_csv(file_path)
                    row_count = len(df)
            except:
                pass

            # Store in database
            feature = FeatureStorage(
                session_id=self.session_id,
                feature_type=feature_type,
                table_name=table_name,
                file_path=str(file_path),
                row_count=row_count,
                file_size_bytes=stat.st_size,
                file_format=file_path.suffix.lstrip('.')
            )

            db.add(feature)
            db.commit()

            # Copy file to organized storage path
            organized_path = self.storage_path / self.session_id / feature_type
            organized_path.mkdir(parents=True, exist_ok=True)

            dest_path = organized_path / file_path.name
            if file_path != dest_path:
                import shutil
                shutil.copy2(file_path, dest_path)
                logger.info(f"Copied file to: {dest_path}")

            return str(dest_path)

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to store file {file_path}: {e}")
            raise
        finally:
            db.close()

    def store_model(self,
                   model: Any,
                   model_name: str,
                   feature_names: List[str],
                   hyperparams: Dict,
                   train_score: float,
                   val_score: float,
                   cv_scores: Optional[List[float]] = None,
                   is_latest: bool = True) -> str:
        """Store trained model in database."""
        db = self.get_session()

        try:
            # Serialize model
            model_data = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)

            # If this is latest, unset previous latest
            if is_latest:
                db.query(ModelStorage).filter(
                    ModelStorage.session_id == self.session_id,
                    ModelStorage.is_latest == True
                ).update({'is_latest': False})

            # Store model
            model_storage = ModelStorage(
                session_id=self.session_id,
                model_name=model_name,
                model_version=self.session_id,
                is_latest=is_latest,
                model_data=model_data,
                feature_names=json.dumps(feature_names),
                hyperparams=json.dumps(hyperparams),
                train_score=train_score,
                val_score=val_score,
                cv_scores=json.dumps(cv_scores) if cv_scores else None
            )

            db.add(model_storage)
            db.commit()

            logger.info(f"Stored model: {model_name}")
            return self.session_id

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to store model: {e}")
            raise
        finally:
            db.close()

    def update_session_status(self,
                            status: str,
                            metrics: Optional[Dict] = None,
                            model_version: Optional[str] = None,
                            notes: Optional[str] = None):
        """Update training session status and metrics."""
        db = self.get_session()

        try:
            session = db.query(TrainingSession).filter(
                TrainingSession.session_id == self.session_id
            ).first()

            if session:
                session.status = status
                if metrics:
                    for key, value in metrics.items():
                        if hasattr(session, key):
                            setattr(session, key, value)

                if model_version:
                    session.model_version = model_version

                if notes:
                    session.notes = notes

                if status == 'completed':
                    session.completed_at = datetime.utcnow()

                db.commit()
                logger.info(f"Updated session status to: {status}")

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to update session: {e}")
            raise
        finally:
            db.close()

    def store_evaluation_results(self,
                               eval_type: str,
                               metrics: Dict,
                               eval_params: Optional[Dict] = None,
                               detailed_results: Optional[Dict] = None) -> str:
        """Store evaluation results in database."""
        db = self.get_session()

        try:
            eval_result = EvaluationResult(
                session_id=self.session_id,
                eval_type=eval_type,
                detailed_metrics=json.dumps(metrics, default=str),
                **{k: v for k, v in metrics.items() if hasattr(EvaluationResult, k)}
            )

            if eval_params:
                for key, value in eval_params.items():
                    if hasattr(eval_result, key):
                        setattr(eval_result, key, value)

            db.add(eval_result)
            db.commit()

            logger.info(f"Stored evaluation results: {eval_type}")
            return self.session_id

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to store evaluation results: {e}")
            raise
        finally:
            db.close()

    def load_latest_model(self, session_id: Optional[str] = None) -> tuple:
        """Load latest model from database."""
        db = self.get_session()

        try:
            query = db.query(ModelStorage).filter(ModelStorage.is_latest == True)
            if session_id:
                query = query.filter(ModelStorage.session_id == session_id)

            model_record = query.order_by(ModelStorage.created_at.desc()).first()

            if model_record:
                model = pickle.loads(model_record.model_data)
                feature_names = json.loads(model_record.feature_names) if model_record.feature_names else []
                return model, feature_names, model_record.session_id
            else:
                raise ValueError("No model found in database")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        finally:
            db.close()

    def get_session_history(self, limit: int = 10) -> List[Dict]:
        """Get history of training sessions."""
        db = self.get_session()

        try:
            sessions = db.query(TrainingSession).order_by(
                TrainingSession.created_at.desc()
            ).limit(limit).all()

            results = []
            for session in sessions:
                results.append({
                    'session_id': session.session_id,
                    'created_at': session.created_at,
                    'status': session.status,
                    'total_samples': session.total_samples,
                    'feature_count': session.feature_count,
                    'test_auc': session.test_auc,
                    'sharpe_ratio': session.sharpe_ratio,
                    'total_return': session.total_return
                })

            return results

        except Exception as e:
            logger.error(f"Failed to get session history: {e}")
            raise
        finally:
            db.close()

    def cleanup_old_data(self, days: int = 30):
        """Clean up old data from database."""
        db = self.get_session()

        try:
            cutoff_date = datetime.utcnow() - pd.Timedelta(days=days)

            # Delete old training sessions and related data
            old_sessions = db.query(TrainingSession).filter(
                TrainingSession.created_at < cutoff_date
            ).all()

            count = 0
            for session in old_sessions:
                # Delete related records
                db.query(ModelStorage).filter(
                    ModelStorage.session_id == session.session_id
                ).delete()
                db.query(FeatureStorage).filter(
                    FeatureStorage.session_id == session.session_id
                ).delete()
                db.query(EvaluationResult).filter(
                    EvaluationResult.session_id == session.session_id
                ).delete()

                db.delete(session)
                count += 1

            db.commit()
            logger.info(f"Cleaned up {count} old training sessions")

        except Exception as e:
            db.rollback()
            logger.error(f"Failed to cleanup old data: {e}")
            raise
        finally:
            db.close()