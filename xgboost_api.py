#!/usr/bin/env python3
"""
XGBoost Model API for QuantConnect Integration
Provides separate endpoints for spot and futures models
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
    version="2.0.0"
)

# Enable CORS - Load from environment
from_env_origins = os.getenv('ALLOWED_ORIGINS', '*').split(',')

app.add_middleware(
    CORSMiddleware,
    allow_origins=from_env_origins if from_env_origins else ["*"],
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
    model_version: str
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

class ModelInsertRequest(BaseModel):
    model_name: str
    model_data_base64: str
    is_latest: bool = True
    feature_names: Optional[List[str]] = []
    hyperparams: Optional[Dict] = {}
    train_score: Optional[float] = None
    val_score: Optional[float] = None
    cv_scores: Optional[List[float]] = None

class ModelInsertResponse(BaseModel):
    success: bool
    message: str
    session_id: str
    model_id: Optional[int] = None

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

# =========================================================
# SHARED HELPER FUNCTIONS
# =========================================================

def get_latest_model_by_version(model_version: str):
    """Helper function to get latest model by version (spot/futures)."""
    if not db_storage:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not connected"
        )

    try:
        # Get latest model from database with specific version
        model, feature_names, session_id = db_storage.load_latest_model()

        # Verify the model version matches
        db = db_storage.get_session()
        model_record = db.query(db_storage.db_model).filter(
            db_storage.db_model.session_id == session_id,
            db_storage.db_model.is_latest == True,
            db_storage.db_model.model_version == model_version
        ).first()

        if not model_record:
            db.close()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No {model_version} model found"
            )

        # Serialize model to bytes
        model_bytes = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)

        # Encode to base64
        model_base64 = encode_to_base64(model_bytes)

        response = ModelResponse(
            success=True,
            session_id=session_id,
            model_name=model_record.model_name,
            model_version=model_record.model_version,
            model_file=model_record.model_file,
            created_at=model_record.created_at.isoformat(),
            model_data_base64=model_base64,
            feature_names=feature_names,
            content_type='application/octet-stream',
            file_extension='.joblib'
        )

        db.close()
        logger.info(f"✅ Retrieved latest {model_version} model for session: {session_id}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get latest {model_version} model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model: {str(e)}"
        )

def get_dataset_summary_by_version(model_version: str):
    """Helper function to get latest dataset summary by version (spot/futures)."""
    if not db_storage:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not connected"
        )

    try:
        # Get latest dataset summary from database with specific version
        db = db_storage.get_session()

        # Get the latest dataset summary record with specific model_version
        summary_record = db.query(db_storage.db_dataset_summary).filter(
            db_storage.db_dataset_summary.model_version == model_version
        ).order_by(
            db_storage.db_dataset_summary.created_at.desc()
        ).first()

        if not summary_record:
            # No dataset summary found for this version
            db.close()
            return DatasetSummaryResponse(
                success=False,
                session_id="",
                model_version=model_version,
                summary_file="",
                created_at=datetime.utcnow().isoformat(),
                summary_data_base64=None,
                content_type='text/plain',
                file_extension='.txt'
            )

        # Try to get summary from database first (new behavior)
        summary_base64 = None

        if summary_record.summary_data:
            # Summary data exists in database as blob - encode to base64
            summary_base64 = encode_to_base64(summary_record.summary_data)
            logger.info(f"✅ Retrieved {model_version} dataset summary blob from database")
        else:
            # Fallback: Try to read from file (backward compatibility)
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
                    logger.info(f"✅ Retrieved {model_version} dataset summary from file: {path}")
                    break

        response = DatasetSummaryResponse(
            success=True,
            session_id=summary_record.session_id,
            model_version=summary_record.model_version,
            summary_file=summary_record.summary_file,
            created_at=summary_record.created_at.isoformat(),
            summary_data_base64=summary_base64,
            content_type='text/plain',
            file_extension='.txt'
        )

        db.close()
        logger.info(f"✅ Retrieved {model_version} dataset summary for session: {summary_record.session_id}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get {model_version} dataset summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve dataset summary: {str(e)}"
        )

def get_model_by_session_version(session_id: str, model_version: str):
    """Helper function to get model by session ID and version."""
    if not db_storage:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not connected"
        )

    try:
        # Load specific model by session_id
        model, feature_names, _ = db_storage.load_latest_model(session_id=session_id)

        # Get model metadata
        db = db_storage.get_session()
        model_record = db.query(db_storage.db_model).filter(
            db_storage.db_model.session_id == session_id,
            db_storage.db_model.model_version == model_version
        ).first()

        if not model_record:
            db.close()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No {model_version} model found with session_id: {session_id}"
            )

        # Serialize model to bytes
        model_bytes = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)

        # Encode to base64
        model_base64 = encode_to_base64(model_bytes)

        response = ModelResponse(
            success=True,
            session_id=session_id,
            model_name=model_record.model_name,
            model_version=model_record.model_version,
            model_file=model_record.model_file,
            created_at=model_record.created_at.isoformat(),
            model_data_base64=model_base64,
            feature_names=feature_names,
            content_type='application/octet-stream',
            file_extension='.joblib'
        )

        db.close()
        logger.info(f"✅ Retrieved {model_version} model for session: {session_id}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get {model_version} model for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model: {str(e)}"
        )

def get_sessions_by_version(model_version: str):
    """Helper function to list sessions by version."""
    if not db_storage:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not connected"
        )

    try:
        db = db_storage.get_session()

        # Get all models with specific version
        models = db.query(db_storage.db_model).filter(
            db_storage.db_model.model_version == model_version
        ).order_by(
            db_storage.db_model.created_at.desc()
        ).all()

        sessions = []
        for model in models:
            # Check if dataset summary exists
            summary_exists = db.query(db_storage.db_dataset_summary).filter(
                db_storage.db_dataset_summary.session_id == model.session_id,
                db_storage.db_dataset_summary.model_version == model_version
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
        logger.error(f"❌ Failed to list {model_version} sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve sessions: {str(e)}"
        )

def insert_model_by_version(model_request: ModelInsertRequest, model_version: str):
    """Helper function to insert model with specific version."""
    if not db_storage:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not connected"
        )

    try:
        # Decode base64 model data
        model_bytes = base64.b64decode(model_request.model_data_base64)
        model = pickle.loads(model_bytes)

        # Generate new session_id for this model
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        db_storage.session_id = session_id

        # Store the model using the store_model method
        returned_session_id = db_storage.store_model(
            model=model,
            model_name=model_request.model_name,
            feature_names=model_request.feature_names,
            hyperparams=model_request.hyperparams,
            train_score=model_request.train_score,
            val_score=model_request.val_score,
            cv_scores=model_request.cv_scores,
            is_latest=model_request.is_latest,
            model_version=model_version  # 'spot' or 'futures'
        )

        # Get the inserted model record ID
        db = db_storage.get_session()
        model_record = db.query(db_storage.db_model).filter(
            db_storage.db_model.session_id == returned_session_id
        ).first()
        model_id = model_record.id if model_record else None

        db.close()

        logger.info(f"✅ Successfully inserted {model_version} model: {model_request.model_name} (Session: {returned_session_id})")

        return ModelInsertResponse(
            success=True,
            message=f"Model inserted successfully (version: {model_version})",
            session_id=returned_session_id,
            model_id=model_id
        )

    except Exception as e:
        logger.error(f"❌ Failed to insert {model_version} model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to insert model: {str(e)}"
        )

# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        database_connected=db_storage is not None
    )

# =========================================================
# SPOT ENDPOINTS (/api/v1/spot/*)
# =========================================================

@app.get("/api/v1/spot/latest/model", response_model=ModelResponse)
async def get_spot_latest_model():
    """Get latest spot trained model with base64 encoding."""
    return get_latest_model_by_version('spot')

@app.get("/api/v1/spot/latest/dataset-summary", response_model=DatasetSummaryResponse)
async def get_spot_latest_dataset_summary():
    """Get latest spot dataset summary with base64 encoding."""
    return get_dataset_summary_by_version('spot')

@app.get("/api/v1/spot/model/{session_id}", response_model=ModelResponse)
async def get_spot_model_by_session(session_id: str):
    """Get spot model by specific session ID."""
    return get_model_by_session_version(session_id, 'spot')

@app.get("/api/v1/spot/sessions", response_model=SessionsResponse)
async def list_spot_sessions():
    """List all available spot training sessions."""
    return get_sessions_by_version('spot')

@app.post("/api/v1/spot/model", response_model=ModelInsertResponse)
async def insert_spot_model(model_request: ModelInsertRequest):
    """Insert a new spot trained model to the database."""
    return insert_model_by_version(model_request, 'spot')

# =========================================================
# FUTURES ENDPOINTS (/api/v1/futures/*)
# =========================================================

@app.get("/api/v1/futures/latest/model", response_model=ModelResponse)
async def get_futures_latest_model():
    """Get latest futures trained model with base64 encoding."""
    return get_latest_model_by_version('futures')

@app.get("/api/v1/futures/latest/dataset-summary", response_model=DatasetSummaryResponse)
async def get_futures_latest_dataset_summary():
    """Get latest futures dataset summary with base64 encoding."""
    return get_dataset_summary_by_version('futures')

@app.get("/api/v1/futures/model/{session_id}", response_model=ModelResponse)
async def get_futures_model_by_session(session_id: str):
    """Get futures model by specific session ID."""
    return get_model_by_session_version(session_id, 'futures')

@app.get("/api/v1/futures/sessions", response_model=SessionsResponse)
async def list_futures_sessions():
    """List all available futures training sessions."""
    return get_sessions_by_version('futures')

@app.post("/api/v1/futures/model", response_model=ModelInsertResponse)
async def insert_futures_model(model_request: ModelInsertRequest):
    """Insert a new futures trained model to the database."""
    return insert_model_by_version(model_request, 'futures')

# =========================================================
# DATABASE MODEL REFERENCES
# =========================================================

def init_db_models():
    """Initialize database model references."""
    if db_storage:
        from database_storage import ModelStorage, DatasetSummary
        db_storage.db_model = ModelStorage
        db_storage.db_dataset_summary = DatasetSummary

# Initialize database models
init_db_models()

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv('API_PORT', 5000))
    host = os.getenv('API_HOST', '0.0.0.0')

    logger.info(f"🚀 Starting XGBoost FastAPI server on {host}:{port}")
    logger.info("📊 Available endpoints:")
    logger.info("   GET /health - Health check")
    logger.info("")
    logger.info("   SPOT Endpoints:")
    logger.info("   GET /api/v1/spot/latest/model - Get latest spot model")
    logger.info("   GET /api/v1/spot/latest/dataset-summary - Get latest spot dataset summary")
    logger.info("   GET /api/v1/spot/model/<session_id> - Get spot model by session")
    logger.info("   GET /api/v1/spot/sessions - List all spot sessions")
    logger.info("   POST /api/v1/spot/model - Insert new spot model")
    logger.info("")
    logger.info("   FUTURES Endpoints:")
    logger.info("   GET /api/v1/futures/latest/model - Get latest futures model")
    logger.info("   GET /api/v1/futures/latest/dataset-summary - Get latest futures dataset summary")
    logger.info("   GET /api/v1/futures/model/<session_id> - Get futures model by session")
    logger.info("   GET /api/v1/futures/sessions - List all futures sessions")
    logger.info("   POST /api/v1/futures/model - Insert new futures model")
    logger.info("")
    logger.info("📖 API docs available at: http://localhost:5000/docs")

    uvicorn.run(app, host=host, port=port)
