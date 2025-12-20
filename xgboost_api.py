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
from typing import Dict, Optional, Any, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

# FastAPI app
app = FastAPI(
    title="XGBoost Trading Model API",
    description="API for XGBoost trading models and dataset summaries for QuantConnect",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database storage
try:
    db_storage = DatabaseStorage()
    logger.info("✅ Database connected successfully")
except Exception as e:
    logger.error(f"❌ Failed to connect to database: {e}")
    db_storage = None

# Pydantic models for response
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    database_connected: bool

class ModelResponse(BaseModel):
    success: bool
    session_id: str
    model_name: str
    model_version: str
    model_file: str
    created_at: Optional[str]
    model_data_base64: str
    feature_names: List[str]
    content_type: str
    file_extension: str

class DatasetSummaryResponse(BaseModel):
    success: bool
    session_id: str
    summary_file: str
    created_at: str
    summary_data_base64: Optional[str]
    content_type: str
    file_extension: str

class SessionInfo(BaseModel):
    session_id: str
    model_name: str
    model_version: Optional[str]
    is_latest: bool
    created_at: str
    has_dataset_summary: bool

class SessionsResponse(BaseModel):
    success: bool
    total_sessions: int
    sessions: List[SessionInfo]

class ErrorResponse(BaseModel):
    error: str
    message: str

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

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        database_connected=db_storage is not None
    )

@app.get("/api/v1/latest/model", response_model=ModelResponse)
async def get_latest_model():
    """Get latest trained model with base64 encoding."""

    if not db_storage:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not connected"
        )

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

        response = ModelResponse(
            success=True,
            session_id=session_id,
            model_name=model_record.model_name if model_record else 'unknown',
            model_version=model_record.model_version if model_record else session_id,
            model_file=model_record.model_file if model_record else 'latest_model.joblib',
            created_at=model_record.created_at.isoformat() if model_record else None,
            model_data_base64=model_base64,
            feature_names=feature_names,
            content_type='application/octet-stream',
            file_extension='.joblib'
        )

        db.close()
        logger.info(f"✅ Retrieved latest model for session: {session_id}")
        return response

    except Exception as e:
        logger.error(f"❌ Failed to get latest model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model: {str(e)}"
        )

@app.get("/api/v1/latest/dataset-summary", response_model=DatasetSummaryResponse)
async def get_latest_dataset_summary():
    """Get latest dataset summary with base64 encoding."""

    if not db_storage:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not connected"
        )

    try:
        # Get latest dataset summary from database
        db = db_storage.get_session()

        # Get latest session from xgboost_models first
        latest_model = db.query(db_storage.db_model).filter(
            db_storage.db_model.is_latest == True
        ).order_by(db_storage.db_model.created_at.desc()).first()

        if not latest_model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No trained models found"
            )

        # Get corresponding dataset summary
        summary_record = db.query(db_storage.db_dataset_summary).filter(
            db_storage.db_dataset_summary.session_id == latest_model.session_id
        ).first()

        if not summary_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No dataset summary found for session: {latest_model.session_id}"
            )

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

        response = DatasetSummaryResponse(
            success=True,
            session_id=summary_record.session_id,
            summary_file=summary_record.summary_file,
            created_at=summary_record.created_at.isoformat(),
            summary_data_base64=summary_base64,
            content_type='text/plain',
            file_extension='.txt'
        )

        db.close()
        logger.info(f"✅ Retrieved dataset summary for session: {summary_record.session_id}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get dataset summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve dataset summary: {str(e)}"
        )

@app.get("/api/v1/model/{session_id}", response_model=ModelResponse)
async def get_model_by_session(session_id: str):
    """Get model by specific session ID."""

    if not db_storage:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not connected"
        )

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

        response = ModelResponse(
            success=True,
            session_id=session_id,
            model_name=model_record.model_name if model_record else 'unknown',
            model_version=model_record.model_version if model_record else session_id,
            model_file=model_record.model_file if model_record else f'model_{session_id}.joblib',
            created_at=model_record.created_at.isoformat() if model_record else None,
            model_data_base64=model_base64,
            feature_names=feature_names,
            content_type='application/octet-stream',
            file_extension='.joblib'
        )

        db.close()
        logger.info(f"✅ Retrieved model for session: {session_id}")
        return response

    except Exception as e:
        logger.error(f"❌ Failed to get model for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model: {str(e)}"
        )

@app.get("/api/v1/dataset-summary/{session_id}", response_model=DatasetSummaryResponse)
async def get_dataset_summary_by_session(session_id: str):
    """Get dataset summary by specific session ID."""

    if not db_storage:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not connected"
        )

    try:
        # Get dataset summary by session_id
        db = db_storage.get_session()

        summary_record = db.query(db_storage.db_dataset_summary).filter(
            db_storage.db_dataset_summary.session_id == session_id
        ).first()

        if not summary_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No dataset summary found for session: {session_id}"
            )

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

        response = DatasetSummaryResponse(
            success=True,
            session_id=session_id,
            summary_file=summary_record.summary_file,
            created_at=summary_record.created_at.isoformat(),
            summary_data_base64=summary_base64,
            content_type='text/plain',
            file_extension='.txt'
        )

        db.close()
        logger.info(f"✅ Retrieved dataset summary for session: {session_id}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get dataset summary for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve dataset summary: {str(e)}"
        )

@app.get("/api/v1/sessions", response_model=SessionsResponse)
async def list_sessions():
    """List all available training sessions."""

    if not db_storage:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not connected"
        )

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

            sessions.append(SessionInfo(
                session_id=model.session_id,
                model_name=model.model_name,
                model_version=model.model_version,
                is_latest=model.is_latest,
                created_at=model.created_at.isoformat(),
                has_dataset_summary=summary_exists
            ))

        db.close()

        return SessionsResponse(
            success=True,
            total_sessions=len(sessions),
            sessions=sessions
        )

    except Exception as e:
        logger.error(f"❌ Failed to list sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve sessions: {str(e)}"
        )

# Database model references (to avoid circular imports)
def init_db_models():
    """Initialize database model references."""
    if db_storage:
        from database_storage import ModelStorage, DatasetSummary
        db_storage.db_model = ModelStorage
        db_storage.db_dataset_summary = DatasetSummary

# Initialize database models
init_db_models()

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv('API_PORT', 5000))
    host = os.getenv('API_HOST', '0.0.0.0')

    logger.info(f"🚀 Starting XGBoost FastAPI server on {host}:{port}")
    logger.info("📊 Available endpoints:")
    logger.info("   GET /health - Health check")
    logger.info("   GET /api/v1/latest/model - Get latest model")
    logger.info("   GET /api/v1/latest/dataset-summary - Get latest dataset summary")
    logger.info("   GET /api/v1/model/<session_id> - Get model by session")
    logger.info("   GET /api/v1/dataset-summary/<session_id> - Get dataset summary by session")
    logger.info("   GET /api/v1/sessions - List all sessions")
    logger.info("📖 API docs available at: http://localhost:5000/docs")

    uvicorn.run(app, host=host, port=port)