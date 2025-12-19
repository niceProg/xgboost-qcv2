#!/usr/bin/env python3
"""
XGBoost Model API for QuantConnect Integration
Provides endpoints to fetch latest trained models and dataset summaries
Returns data in base64 format for easy consumption by QuantConnect
"""

import os
import base64
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, Optional, Any
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

# Database imports
from database_storage import DatabaseStorage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for QuantConnect

# Initialize database storage
try:
    db_storage = DatabaseStorage()
    logger.info("✅ Database connected successfully")
except Exception as e:
    logger.error(f"❌ Failed to connect to database: {e}")
    db_storage = None

def encode_to_base64(data: bytes) -> str:
    """Encode bytes to base64 string."""
    return base64.b64encode(data).decode('utf-8')

def read_file_as_base64(file_path: str) -> Optional[str]:
    """Read file and return as base64 string."""
    try:
        with open(file_path, 'rb') as f:
            return encode_to_base64(f.read())
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'database_connected': db_storage is not None
    })

@app.route('/api/v1/latest/model', methods=['GET'])
def get_latest_model():
    """Get latest trained model with base64 encoding."""

    if not db_storage:
        return jsonify({
            'error': 'Database not connected',
            'message': 'Unable to connect to database'
        }), 500

    try:
        # Get latest model from database
        model, feature_names, session_id = db_storage.load_latest_model()

        # Serialize model to bytes
        model_bytes = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)

        # Encode to base64
        model_base64 = encode_to_base64(model_bytes)

        # Get model metadata
        db = db_storage.get_session()
        model_record = db.query(db_storage.db_model).filter(
            db_storage.db_model.session_id == session_id,
            db_storage.db_model.is_latest == True
        ).first()

        response = {
            'success': True,
            'session_id': session_id,
            'model_name': model_record.model_name if model_record else 'unknown',
            'model_version': model_record.model_version if model_record else session_id,
            'model_file': model_record.model_file if model_record else 'latest_model.joblib',
            'created_at': model_record.created_at.isoformat() if model_record else None,
            'model_data_base64': model_base64,
            'feature_names': feature_names,
            'content_type': 'application/octet-stream',
            'file_extension': '.joblib'
        }

        db.close()
        logger.info(f"✅ Retrieved latest model for session: {session_id}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"❌ Failed to get latest model: {e}")
        return jsonify({
            'error': 'Failed to retrieve model',
            'message': str(e)
        }), 500

@app.route('/api/v1/latest/dataset-summary', methods=['GET'])
def get_latest_dataset_summary():
    """Get latest dataset summary with base64 encoding."""

    if not db_storage:
        return jsonify({
            'error': 'Database not connected',
            'message': 'Unable to connect to database'
        }), 500

    try:
        # Get latest dataset summary from database
        db = db_storage.get_session()

        # Get latest session from xgboost_models first
        latest_model = db.query(db_storage.db_model).filter(
            db_storage.db_model.is_latest == True
        ).order_by(db_storage.db_model.created_at.desc()).first()

        if not latest_model:
            return jsonify({
                'error': 'No trained models found',
                'message': 'No models available in database'
            }), 404

        # Get corresponding dataset summary
        summary_record = db.query(db_storage.db_dataset_summary).filter(
            db_storage.db_dataset_summary.session_id == latest_model.session_id
        ).first()

        if not summary_record:
            return jsonify({
                'error': 'Dataset summary not found',
                'message': f'No dataset summary found for session: {latest_model.session_id}'
            }), 404

        # Try to read the actual summary file
        summary_base64 = None
        summary_file_path = summary_record.summary_file

        # Try to find the file in common locations
        possible_paths = [
            f"./output_train/datasets/summary/{summary_file_path}",
            f"./output_train/{summary_file_path}",
            summary_file_path
        ]

        for path in possible_paths:
            if os.path.exists(path):
                summary_base64 = read_file_as_base64(path)
                break

        response = {
            'success': True,
            'session_id': summary_record.session_id,
            'summary_file': summary_record.summary_file,
            'created_at': summary_record.created_at.isoformat(),
            'summary_data_base64': summary_base64,
            'content_type': 'text/plain',
            'file_extension': '.txt'
        }

        db.close()
        logger.info(f"✅ Retrieved dataset summary for session: {summary_record.session_id}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"❌ Failed to get dataset summary: {e}")
        return jsonify({
            'error': 'Failed to retrieve dataset summary',
            'message': str(e)
        }), 500

@app.route('/api/v1/model/<session_id>', methods=['GET'])
def get_model_by_session(session_id: str):
    """Get model by specific session ID."""

    if not db_storage:
        return jsonify({
            'error': 'Database not connected',
            'message': 'Unable to connect to database'
        }), 500

    try:
        # Load specific model by session_id
        model, feature_names, _ = db_storage.load_latest_model(session_id=session_id)

        # Serialize model to bytes
        model_bytes = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)

        # Encode to base64
        model_base64 = encode_to_base64(model_bytes)

        # Get model metadata
        db = db_storage.get_session()
        model_record = db.query(db_storage.db_model).filter(
            db_storage.db_model.session_id == session_id
        ).first()

        response = {
            'success': True,
            'session_id': session_id,
            'model_name': model_record.model_name if model_record else 'unknown',
            'model_version': model_record.model_version if model_record else session_id,
            'model_file': model_record.model_file if model_record else f'model_{session_id}.joblib',
            'created_at': model_record.created_at.isoformat() if model_record else None,
            'model_data_base64': model_base64,
            'feature_names': feature_names,
            'content_type': 'application/octet-stream',
            'file_extension': '.joblib'
        }

        db.close()
        logger.info(f"✅ Retrieved model for session: {session_id}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"❌ Failed to get model for session {session_id}: {e}")
        return jsonify({
            'error': 'Failed to retrieve model',
            'message': str(e)
        }), 500

@app.route('/api/v1/dataset-summary/<session_id>', methods=['GET'])
def get_dataset_summary_by_session(session_id: str):
    """Get dataset summary by specific session ID."""

    if not db_storage:
        return jsonify({
            'error': 'Database not connected',
            'message': 'Unable to connect to database'
        }), 500

    try:
        # Get dataset summary by session_id
        db = db_storage.get_session()

        summary_record = db.query(db_storage.db_dataset_summary).filter(
            db_storage.db_dataset_summary.session_id == session_id
        ).first()

        if not summary_record:
            return jsonify({
                'error': 'Dataset summary not found',
                'message': f'No dataset summary found for session: {session_id}'
            }), 404

        # Try to read the actual summary file
        summary_base64 = None
        summary_file_path = summary_record.summary_file

        # Try to find the file in common locations
        possible_paths = [
            f"./output_train/datasets/summary/{summary_file_path}",
            f"./output_train/{summary_file_path}",
            summary_file_path
        ]

        for path in possible_paths:
            if os.path.exists(path):
                summary_base64 = read_file_as_base64(path)
                break

        response = {
            'success': True,
            'session_id': session_id,
            'summary_file': summary_record.summary_file,
            'created_at': summary_record.created_at.isoformat(),
            'summary_data_base64': summary_base64,
            'content_type': 'text/plain',
            'file_extension': '.txt'
        }

        db.close()
        logger.info(f"✅ Retrieved dataset summary for session: {session_id}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"❌ Failed to get dataset summary for session {session_id}: {e}")
        return jsonify({
            'error': 'Failed to retrieve dataset summary',
            'message': str(e)
        }), 500

@app.route('/api/v1/sessions', methods=['GET'])
def list_sessions():
    """List all available training sessions."""

    if not db_storage:
        return jsonify({
            'error': 'Database not connected',
            'message': 'Unable to connect to database'
        }), 500

    try:
        db = db_storage.get_session()

        # Get all models with their session info
        models = db.query(db_storage.db_model).order_by(
            db_storage.db_model.created_at.desc()
        ).all()

        sessions = []
        for model in models:
            # Check if dataset summary exists
            summary_exists = db.query(db_storage.db_dataset_summary).filter(
                db_storage.db_dataset_summary.session_id == model.session_id
            ).first() is not None

            sessions.append({
                'session_id': model.session_id,
                'model_name': model.model_name,
                'model_version': model.model_version,
                'is_latest': model.is_latest,
                'created_at': model.created_at.isoformat(),
                'has_dataset_summary': summary_exists
            })

        db.close()

        return jsonify({
            'success': True,
            'total_sessions': len(sessions),
            'sessions': sessions
        })

    except Exception as e:
        logger.error(f"❌ Failed to list sessions: {e}")
        return jsonify({
            'error': 'Failed to retrieve sessions',
            'message': str(e)
        }), 500

# Database model references (to avoid circular imports)
def init_db_models():
    """Initialize database model references."""
    if db_storage:
        from database_storage import ModelStorage, DatasetSummary
        db_storage.db_model = ModelStorage
        db_storage.db_dataset_summary = DatasetSummary

# Initialize database models
init_db_models()

if __name__ == '__main__':
    port = int(os.getenv('API_PORT', 5000))
    host = os.getenv('API_HOST', '0.0.0.0')
    debug = os.getenv('API_DEBUG', 'False').lower() == 'true'

    logger.info(f"🚀 Starting XGBoost API server on {host}:{port}")
    logger.info("📊 Available endpoints:")
    logger.info("   GET /health - Health check")
    logger.info("   GET /api/v1/latest/model - Get latest model")
    logger.info("   GET /api/v1/latest/dataset-summary - Get latest dataset summary")
    logger.info("   GET /api/v1/model/<session_id> - Get model by session")
    logger.info("   GET /api/v1/dataset-summary/<session_id> - Get dataset summary by session")
    logger.info("   GET /api/v1/sessions - List all sessions")

    app.run(host=host, port=port, debug=debug)