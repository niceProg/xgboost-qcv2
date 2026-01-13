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
    id: int
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
    id: int
    model_version: str
    summary_file: str
    created_at: str
    summary_data_base64: Optional[str]
    content_type: str
    file_extension: str

class ModelInfo(BaseModel):
    id: int
    model_name: str
    model_version: Optional[str]
    created_at: str
    has_dataset_summary: bool

class ModelsResponse(BaseModel):
    success: bool
    total_models: int
    models: List[ModelInfo]

class ErrorResponse(BaseModel):
    error: str
    message: str

class ModelInsertRequest(BaseModel):
    model_name: str
    model_data_base64: str
    feature_names: Optional[List[str]] = []
    hyperparams: Optional[Dict] = {}
    train_score: Optional[float] = None
    val_score: Optional[float] = None
    cv_scores: Optional[List[float]] = None

class ModelInsertResponse(BaseModel):
    success: bool
    message: str
    model_id: int

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
    """Helper function to get latest model by version (spot/futures) using created_at."""
    if not db_storage:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not connected"
        )

    try:
        db = db_storage.get_session()

        # Get latest model from database with specific version using created_at DESC
        model_record = db.query(db_storage.db_model).filter(
            db_storage.db_model.model_version == model_version
        ).order_by(
            db_storage.db_model.created_at.desc()
        ).first()

        if not model_record:
            db.close()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No {model_version} model found"
            )

        # Load model from binary data
        model = pickle.loads(model_record.model_data)
        feature_names = []  # feature_names tidak disimpan per client requirement

        # Serialize model to bytes
        model_bytes = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)

        # Encode to base64
        model_base64 = encode_to_base64(model_bytes)

        response = ModelResponse(
            success=True,
            id=model_record.id,
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
        logger.info(f"✅ Retrieved latest {model_version} model (ID: {model_record.id})")
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
                id=0,
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
            id=summary_record.id,
            model_version=summary_record.model_version,
            summary_file=summary_record.summary_file,
            created_at=summary_record.created_at.isoformat(),
            summary_data_base64=summary_base64,
            content_type='text/plain',
            file_extension='.txt'
        )

        db.close()
        logger.info(f"✅ Retrieved {model_version} dataset summary (ID: {summary_record.id})")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get {model_version} dataset summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve dataset summary: {str(e)}"
        )

def get_summary_by_id_version(summary_id: int, model_version: str):
    """Helper function to get dataset summary by ID and version."""
    if not db_storage:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not connected"
        )

    try:
        db = db_storage.get_session()

        # Get dataset summary by ID and version
        summary_record = db.query(db_storage.db_dataset_summary).filter(
            db_storage.db_dataset_summary.id == summary_id,
            db_storage.db_dataset_summary.model_version == model_version
        ).first()

        if not summary_record:
            db.close()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No {model_version} dataset summary found with id: {summary_id}"
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
            id=summary_record.id,
            model_version=summary_record.model_version,
            summary_file=summary_record.summary_file,
            created_at=summary_record.created_at.isoformat(),
            summary_data_base64=summary_base64,
            content_type='text/plain',
            file_extension='.txt'
        )

        db.close()
        logger.info(f"✅ Retrieved {model_version} dataset summary (ID: {summary_id})")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get {model_version} dataset summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve dataset summary: {str(e)}"
        )

def get_model_by_id_version(model_id: int, model_version: str):
    """Helper function to get model by ID and version."""
    if not db_storage:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database not connected"
        )

    try:
        # Get model metadata from database
        db = db_storage.get_session()
        model_record = db.query(db_storage.db_model).filter(
            db_storage.db_model.id == model_id,
            db_storage.db_model.model_version == model_version
        ).first()

        if not model_record:
            db.close()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No {model_version} model found with id: {model_id}"
            )

        # Load model from binary data
        model = pickle.loads(model_record.model_data)
        feature_names = []  # feature_names tidak disimpan per client requirement

        # Serialize model to bytes
        model_bytes = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)

        # Encode to base64
        model_base64 = encode_to_base64(model_bytes)

        response = ModelResponse(
            success=True,
            id=model_id,
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
        logger.info(f"✅ Retrieved {model_version} model (ID: {model_id})")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get {model_version} model for id {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model: {str(e)}"
        )

def list_models_by_version(model_version: str):
    """Helper function to list models by version."""
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

        model_list = []
        for model in models:
            # Check if dataset summary exists (by version and closest created_at)
            summary_exists = db.query(db_storage.db_dataset_summary).filter(
                db_storage.db_dataset_summary.model_version == model_version
            ).first() is not None

            model_list.append(ModelInfo(
                id=model.id,
                model_name=model.model_name,
                model_version=model.model_version,
                created_at=model.created_at.isoformat(),
                has_dataset_summary=summary_exists
            ))

        db.close()

        return ModelsResponse(
            success=True,
            total_models=len(model_list),
            models=model_list
        )

    except Exception as e:
        logger.error(f"❌ Failed to list {model_version} models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve models: {str(e)}"
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

        # Store the model using the store_model method (returns model_id)
        model_id = db_storage.store_model(
            model=model,
            model_name=model_request.model_name,
            feature_names=model_request.feature_names,
            hyperparams=model_request.hyperparams,
            train_score=model_request.train_score,
            val_score=model_request.val_score,
            cv_scores=model_request.cv_scores,
            model_version=model_version  # 'spot' or 'futures'
        )

        logger.info(f"✅ Successfully inserted {model_version} model: {model_request.model_name} (ID: {model_id})")

        return ModelInsertResponse(
            success=True,
            message=f"Model inserted successfully (version: {model_version})",
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

@app.get("/api/v1/spot/summary/{summary_id}", response_model=DatasetSummaryResponse)
async def get_spot_summary_by_id(summary_id: int):
    """Get spot dataset summary by specific ID."""
    return get_summary_by_id_version(summary_id, 'spot')

@app.get("/api/v1/spot/model/{model_id}", response_model=ModelResponse)
async def get_spot_model_by_id(model_id: int):
    """Get spot model by specific ID."""
    return get_model_by_id_version(model_id, 'spot')

@app.get("/api/v1/spot/models", response_model=ModelsResponse)
async def list_spot_models():
    """List all available spot models."""
    return list_models_by_version('spot')

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

@app.get("/api/v1/futures/summary/{summary_id}", response_model=DatasetSummaryResponse)
async def get_futures_summary_by_id(summary_id: int):
    """Get futures dataset summary by specific ID."""
    return get_summary_by_id_version(summary_id, 'futures')

@app.get("/api/v1/futures/model/{model_id}", response_model=ModelResponse)
async def get_futures_model_by_id(model_id: int):
    """Get futures model by specific ID."""
    return get_model_by_id_version(model_id, 'futures')

@app.get("/api/v1/futures/models", response_model=ModelsResponse)
async def list_futures_models():
    """List all available futures models."""
    return list_models_by_version('futures')

@app.post("/api/v1/futures/model", response_model=ModelInsertResponse)
async def insert_futures_model(model_request: ModelInsertRequest):
    """Insert a new futures trained model to the database."""
    return insert_model_by_version(model_request, 'futures')

# =========================================================
# FUTURES17 ENDPOINTS (/api/v1/futures17/*)
# =========================================================

@app.get("/api/v1/futures17/latest/model", response_model=ModelResponse)
async def get_futures17_latest_model():
    """Get latest futures17 trained model with base64 encoding."""
    return get_latest_model_by_version('futures17')

@app.get("/api/v1/futures17/latest/dataset-summary", response_model=DatasetSummaryResponse)
async def get_futures17_latest_dataset_summary():
    """Get latest futures17 dataset summary with base64 encoding."""
    return get_dataset_summary_by_version('futures17')

@app.get("/api/v1/futures17/summary/{summary_id}", response_model=DatasetSummaryResponse)
async def get_futures17_summary_by_id(summary_id: int):
    """Get futures17 dataset summary by specific ID."""
    return get_summary_by_id_version(summary_id, 'futures17')

@app.get("/api/v1/futures17/model/{model_id}", response_model=ModelResponse)
async def get_futures17_model_by_id(model_id: int):
    """Get futures17 model by specific ID."""
    return get_model_by_id_version(model_id, 'futures17')

@app.get("/api/v1/futures17/models", response_model=ModelsResponse)
async def list_futures17_models():
    """List all available futures17 models."""
    return list_models_by_version('futures17')

@app.post("/api/v1/futures17/model", response_model=ModelInsertResponse)
async def insert_futures17_model(model_request: ModelInsertRequest):
    """Insert a new futures17 trained model to the database."""
    return insert_model_by_version(model_request, 'futures17')

# =========================================================
# FUTURES NEW GEN ENDPOINTS (/api/v1/futures_new_gen/*)
# =========================================================

@app.get("/api/v1/futures_new_gen/latest/model", response_model=ModelResponse)
async def get_futures_new_gen_latest_model():
    """Get latest futures_new_gen trained model with base64 encoding."""
    return get_latest_model_by_version('futures_new_gen')

@app.get("/api/v1/futures_new_gen/latest/dataset-summary", response_model=DatasetSummaryResponse)
async def get_futures_new_gen_latest_dataset_summary():
    """Get latest futures_new_gen dataset summary with base64 encoding."""
    return get_dataset_summary_by_version('futures_new_gen')

@app.get("/api/v1/futures_new_gen/summary/{summary_id}", response_model=DatasetSummaryResponse)
async def get_futures_new_gen_summary_by_id(summary_id: int):
    """Get futures_new_gen dataset summary by specific ID."""
    return get_summary_by_id_version(summary_id, 'futures_new_gen')

@app.get("/api/v1/futures_new_gen/model/{model_id}", response_model=ModelResponse)
async def get_futures_new_gen_model_by_id(model_id: int):
    """Get futures_new_gen model by specific ID."""
    return get_model_by_id_version(model_id, 'futures_new_gen')

@app.get("/api/v1/futures_new_gen/models", response_model=ModelsResponse)
async def list_futures_new_gen_models():
    """List all available futures_new_gen models."""
    return list_models_by_version('futures_new_gen')

@app.post("/api/v1/futures_new_gen/model", response_model=ModelInsertResponse)
async def insert_futures_new_gen_model(model_request: ModelInsertRequest):
    """Insert a new futures_new_gen trained model to the database."""
    return insert_model_by_version(model_request, 'futures_new_gen')

# =========================================================
# FUTURES NEW GEN ETH ENDPOINTS (/api/v1/futures_new_gen_eth/*)
# =========================================================

@app.get("/api/v1/futures_new_gen_eth/latest/model", response_model=ModelResponse)
async def get_futures_new_gen_eth_latest_model():
    """Get latest futures_new_gen_eth trained model with base64 encoding."""
    return get_latest_model_by_version('futures_new_gen_eth')

@app.get("/api/v1/futures_new_gen_eth/latest/dataset-summary", response_model=DatasetSummaryResponse)
async def get_futures_new_gen_eth_latest_dataset_summary():
    """Get latest futures_new_gen_eth dataset summary with base64 encoding."""
    return get_dataset_summary_by_version('futures_new_gen_eth')

@app.get("/api/v1/futures_new_gen_eth/summary/{summary_id}", response_model=DatasetSummaryResponse)
async def get_futures_new_gen_eth_summary_by_id(summary_id: int):
    """Get futures_new_gen_eth dataset summary by specific ID."""
    return get_summary_by_id_version(summary_id, 'futures_new_gen_eth')

@app.get("/api/v1/futures_new_gen_eth/model/{model_id}", response_model=ModelResponse)
async def get_futures_new_gen_eth_model_by_id(model_id: int):
    """Get futures_new_gen_eth model by specific ID."""
    return get_model_by_id_version(model_id, 'futures_new_gen_eth')

@app.get("/api/v1/futures_new_gen_eth/models", response_model=ModelsResponse)
async def list_futures_new_gen_eth_models():
    """List all available futures_new_gen_eth models."""
    return list_models_by_version('futures_new_gen_eth')

@app.post("/api/v1/futures_new_gen_eth/model", response_model=ModelInsertResponse)
async def insert_futures_new_gen_eth_model(model_request: ModelInsertRequest):
    """Insert a new futures_new_gen_eth trained model to the database."""
    return insert_model_by_version(model_request, 'futures_new_gen_eth')

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
    logger.info("   GET /api/v1/spot/model/{model_id} - Get spot model by ID")
    logger.info("   GET /api/v1/spot/summary/{summary_id} - Get spot dataset summary by ID")
    logger.info("   GET /api/v1/spot/models - List all spot models")
    logger.info("   POST /api/v1/spot/model - Insert new spot model")
    logger.info("")
    logger.info("   FUTURES Endpoints:")
    logger.info("   GET /api/v1/futures/latest/model - Get latest futures model")
    logger.info("   GET /api/v1/futures/latest/dataset-summary - Get latest futures dataset summary")
    logger.info("   GET /api/v1/futures/model/{model_id} - Get futures model by ID")
    logger.info("   GET /api/v1/futures/summary/{summary_id} - Get futures dataset summary by ID")
    logger.info("   GET /api/v1/futures/models - List all futures models")
    logger.info("   POST /api/v1/futures/model - Insert new futures model")
    logger.info("")
    logger.info("   FUTURES17 Endpoints:")
    logger.info("   GET /api/v1/futures17/latest/model - Get latest futures17 model")
    logger.info("   GET /api/v1/futures17/latest/dataset-summary - Get latest futures17 dataset summary")
    logger.info("   GET /api/v1/futures17/model/{model_id} - Get futures17 model by ID")
    logger.info("   GET /api/v1/futures17/summary/{summary_id} - Get futures17 dataset summary by ID")
    logger.info("   GET /api/v1/futures17/models - List all futures17 models")
    logger.info("   POST /api/v1/futures17/model - Insert new futures17 model")
    logger.info("")
    logger.info("   FUTURES NEW GEN Endpoints:")
    logger.info("   GET /api/v1/futures_new_gen/latest/model - Get latest futures_new_gen model")
    logger.info("   GET /api/v1/futures_new_gen/latest/dataset-summary - Get latest futures_new_gen dataset summary")
    logger.info("   GET /api/v1/futures_new_gen/model/{model_id} - Get futures_new_gen model by ID")
    logger.info("   GET /api/v1/futures_new_gen/summary/{summary_id} - Get futures_new_gen dataset summary by ID")
    logger.info("   GET /api/v1/futures_new_gen/models - List all futures_new_gen models")
    logger.info("   POST /api/v1/futures_new_gen/model - Insert new futures_new_gen model")
    logger.info("")
    logger.info("   FUTURES NEW GEN ETH Endpoints:")
    logger.info("   GET /api/v1/futures_new_gen_eth/latest/model - Get latest futures_new_gen_eth model")
    logger.info("   GET /api/v1/futures_new_gen_eth/latest/dataset-summary - Get latest futures_new_gen_eth dataset summary")
    logger.info("   GET /api/v1/futures_new_gen_eth/model/{model_id} - Get futures_new_gen_eth model by ID")
    logger.info("   GET /api/v1/futures_new_gen_eth/summary/{summary_id} - Get futures_new_gen_eth dataset summary by ID")
    logger.info("   GET /api/v1/futures_new_gen_eth/models - List all futures_new_gen_eth models")
    logger.info("   POST /api/v1/futures_new_gen_eth/model - Insert new futures_new_gen_eth model")
    logger.info("")
    logger.info("📖 API docs available at: http://localhost:5000/docs")

    uvicorn.run(app, host=host, port=port)
