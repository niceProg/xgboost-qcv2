#!/usr/bin/env python3
"""
Read-Only XGBoost Trading API Server
Provides GET endpoints only for model status and trading information
No modifications allowed - purely informational API
"""

import os
import json
import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import uvicorn
from fastapi.responses import JSONResponse

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app with comprehensive documentation
app = FastAPI(
    title="Read-Only XGBoost Trading API",
    description="""
    ## 🔒 Read-Only Trading API for XGBoost Model

    This API provides read-only access to XGBoost trading model information:

    ### 📊 **Core Features:**
    - **Model Status**: Current model information and availability
    - **System Health**: API and system status monitoring
    - **Training Status**: Monitor training progress and history
    - **Performance Metrics**: Model performance data

    ### 🔒 **Security & Safety:**
    - **Read-Only**: No modifications allowed
    - **GET Only**: All endpoints use GET method
    - **No Data Changes**: Cannot trigger training or modify data
    - **Informational**: Pure status and information access

    ### 🌐 **Universal Access:**
    - **Web Dashboards**: Perfect for monitoring dashboards
    - **Mobile Apps**: Safe for client applications
    - **External Systems**: Read-only integrations
    - **Analytics**: Data collection and monitoring

    ---

    **Base URL**: `https://your-domain.com:8000`
    **Documentation**: `/docs` (Swagger UI)
    **Alternative Docs**: `/redoc`

    **Version**: 2.0.0
    **Mode**: Read-Only
    """,
    version="2.0.0",
    contact={
        "name": "API Support",
        "email": "support@your-domain.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=[
        {
            "name": "status",
            "description": "📊 System status and health check endpoints"
        },
        {
            "name": "models",
            "description": "🤖 Model information and status endpoints"
        },
        {
            "name": "training",
            "description": "🧠 Training status and history endpoints"
        },
        {
            "name": "monitoring",
            "description": "📈 Monitoring and metrics endpoints"
        }
    ]
)

# CORS middleware - allow more origins for universal access
from fastapi.middleware.cors import CORSMiddleware
allowed_origins = [
    os.getenv('QUANTCONNECT_CORS_ORIGIN', 'https://www.quantconnect.com'),
    "https://your-domain.com",  # Your web app
    "http://localhost:3000",   # Local development
    "http://localhost:8080",   # Local development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "OPTIONS"],  # GET only
    allow_headers=["*"],
)

# Configuration
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', '../output_train'))
STATE_DIR = Path(os.getenv('STATE_DIR', '../state'))

# Pydantic Models for read-only responses
class ModelInfo(BaseModel):
    """Model information response"""
    model_available: bool = Field(..., description="Whether a trained model is available")
    model_version: str = Field(..., description="Current model version")
    model_file: str = Field(..., description="Model filename")
    model_size_mb: Optional[float] = Field(None, description="Model file size in MB")
    last_training: Optional[str] = Field(None, description="Last training timestamp")
    training_accuracy: Optional[float] = Field(None, description="Last training accuracy")
    feature_count: int = Field(..., description="Number of features in model")
    supported_symbols: List[str] = Field(..., description="Supported trading symbols")

class TrainingStatus(BaseModel):
    """Training status response"""
    last_training: Optional[str] = Field(None, description="Last training timestamp")
    last_training_success: Optional[bool] = Field(None, description="Whether last training was successful")
    training_active: bool = Field(..., description="Whether training is currently active")
    pipeline_type: Optional[str] = Field(None, description="Training pipeline type")
    incremental_mode: bool = Field(..., description="Whether incremental training is enabled")
    next_training_check: Optional[str] = Field(None, description="Next training check time")

class SystemHealth(BaseModel):
    """System health response"""
    api_status: str = Field(..., description="API server status")
    api_version: str = Field(..., description="API version")
    model_available: bool = Field(..., description="Model availability status")
    database_connection: str = Field(..., description="Database connection status")
    monitor_active: bool = Field(..., description="Database monitor status")
    timestamp: str = Field(..., description="Health check timestamp")

class TrainingHistory(BaseModel):
    """Training history entry"""
    timestamp: str = Field(..., description="Training timestamp")
    training_successful: bool = Field(..., description="Whether training was successful")
    pipeline_type: str = Field(..., description="Type of training pipeline")
    trigger_reason: str = Field(..., description="Reason for training trigger")
    models_created: int = Field(..., description="Number of models created")
    tables_with_new_data: List[str] = Field(..., description="Tables with new data")

# Core API Functions
class ReadOnlyModelManager:
    """Read-only model information manager"""

    def __init__(self):
        self.model_info = {}
        self.refresh_model_info()

    def refresh_model_info(self):
        """Refresh model information"""
        try:
            model_files = list(OUTPUT_DIR.glob("*.joblib"))

            if not model_files:
                self.model_info = {
                    "available": False,
                    "file": None,
                    "size_mb": None,
                    "last_modified": None
                }
                return

            # Get latest model
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)

            self.model_info = {
                "available": True,
                "file": str(latest_model.name),
                "size_mb": latest_model.stat().st_size / (1024 * 1024),
                "last_modified": datetime.fromtimestamp(latest_model.stat().st_mtime).isoformat(),
            }

        except Exception as e:
            logger.error(f"❌ Error refreshing model info: {e}")
            self.model_info = {"available": False}

# Initialize model manager
model_manager = ReadOnlyModelManager()

# Helper functions
def get_training_status() -> Dict[str, Any]:
    """Get current training status"""
    status_file = STATE_DIR / 'last_training_status.json'

    if status_file.exists():
        try:
            with open(status_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error reading training status: {e}")

    return {}

def get_system_status() -> Dict[str, Any]:
    """Get comprehensive system status"""
    return {
        "api": {
            "status": "running",
            "version": "2.0.0",
            "mode": "read-only"
        },
        "model": model_manager.model_info,
        "paths": {
            "output_dir": str(OUTPUT_DIR),
            "state_dir": str(STATE_DIR),
            "output_dir_exists": OUTPUT_DIR.exists(),
            "state_dir_exists": STATE_DIR.exists()
        }
    }

# API Endpoints - GET Only

@app.get("/",
         response_model=Dict[str, Any],
         tags=["status"],
         summary="API Root Information",
         description="Get basic API information and available endpoints")
async def root():
    """API root endpoint with basic information"""
    return {
        "name": "Read-Only XGBoost Trading API",
        "version": "2.0.0",
        "mode": "read-only",
        "description": "Read-only API for XGBoost trading model information",
        "endpoints": {
            "health": "/health",
            "model_info": "/model/info",
            "status": "/status",
            "training_status": "/training/status",
            "training_history": "/training/history",
            "docs": "/docs"
        },
        "model_available": model_manager.model_info.get("available", False),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health",
         response_model=SystemHealth,
         tags=["status"],
         summary="System Health Check",
         description="Comprehensive health check of the read-only API system")
async def health_check():
    """Detailed health check endpoint"""
    system_status = get_system_status()

    return SystemHealth(
        api_status="healthy",
        api_version="2.0.0",
        model_available=model_manager.model_info.get("available", False),
        database_connection="assumed_active",  # Read-only - assume active
        monitor_active=True,  # Read-only - assume active
        timestamp=datetime.now().isoformat()
    )

@app.get("/model/info",
         response_model=ModelInfo,
         tags=["models"],
         summary="Get Model Information",
         description="Get detailed information about the loaded XGBoost model")
async def get_model_info():
    """Get comprehensive model information"""

    # Load training status
    training_status = get_training_status()

    return ModelInfo(
        model_available=model_manager.model_info.get("available", False),
        model_version=training_status.get("model_version", "unknown"),
        model_file=model_manager.model_info.get("file", ""),
        model_size_mb=model_manager.model_info.get("size_mb"),
        last_training=training_status.get("last_training_time"),
        training_accuracy=training_status.get("training_accuracy"),
        feature_count=50,  # Update based on actual feature count
        supported_symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT"]  # Update based on actual supported symbols
    )

@app.get("/model/list",
         response_model=List[Dict[str, Any]],
         tags=["models"],
         summary="List Available Models",
         description="Get list of all available model files")
async def list_models():
    """List all available model files with details"""
    try:
        model_files = list(OUTPUT_DIR.glob("*.joblib"))

        models = []
        for model_file in model_files:
            models.append({
                "filename": model_file.name,
                "size_mb": model_file.stat().st_size / (1024 * 1024),
                "last_modified": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat(),
                "path": str(model_file)
            })

        # Sort by last modified (most recent first)
        models.sort(key=lambda x: x["last_modified"], reverse=True)

        return models

    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list models: {str(e)}"
        )

@app.get("/status",
         response_model=Dict[str, Any],
         tags=["status"],
         summary="Get System Status",
         description="Get comprehensive system and training status")
async def get_status():
    """Get system status including training information"""

    # Check training status
    training_status = get_training_status()
    system_status = get_system_status()

    return {
        "api": system_status["api"],
        "model": {
            "available": model_manager.model_info.get("available", False),
            "file": model_manager.model_info.get("file"),
            "last_loaded": model_manager.model_info.get("last_modified")
        },
        "training": {
            "last_training": training_status.get("last_training_time"),
            "last_successful": training_status.get("training_successful", False),
            "pipeline_type": training_status.get("pipeline_type", "unknown"),
            "incremental_mode": True,  # Always true for this system
            "status": "monitoring_active"  # Always true for read-only
        },
        "directories": system_status["paths"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/training/status",
         response_model=TrainingStatus,
         tags=["training"],
         summary="Get Training Status",
         description="Get current training system status")
async def get_training_status_info():
    """Get detailed training status"""

    training_status = get_training_status()

    return TrainingStatus(
        last_training=training_status.get("last_training_time"),
        last_training_success=training_status.get("training_successful", False),
        training_active=False,  # Read-only - always false
        pipeline_type=training_status.get("pipeline_type", "CORE_6_STEP"),
        incremental_mode=True,
        next_training_check="automatic_on_new_data"
    )

@app.get("/training/history",
         response_model=List[TrainingHistory],
         tags=["training"],
         summary="Get Training History",
         description="Get history of training runs and results")
async def get_training_history(
    limit: int = Query(10, ge=1, le=100, description="Number of records to return"),
    successful_only: bool = Query(False, description="Filter successful training runs only"),
    days: int = Query(30, ge=1, le=365, description="Filter history to last N days")
):
    """Get training history from log file"""

    try:
        log_file = STATE_DIR / 'training_results.log'

        if not log_file.exists():
            return []

        training_history = []
        cutoff_date = datetime.now() - timedelta(days=days)

        with open(log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())

                    # Filter by date
                    if "timestamp" in entry:
                        entry_date = datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00"))
                        if entry_date < cutoff_date:
                            continue

                    # Filter by success if requested
                    if successful_only and not entry.get('training_successful', False):
                        continue

                    # Convert to response model
                    training_history.append(TrainingHistory(**entry))

                except (json.JSONDecodeError, ValueError):
                    continue

        # Sort by timestamp (most recent first) and limit
        training_history.sort(key=lambda x: x.timestamp, reverse=True)

        return training_history[:limit]

    except Exception as e:
        logger.error(f"Error reading training history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read training history: {str(e)}"
        )

@app.get("/monitoring/metrics",
         response_model=Dict[str, Any],
         tags=["monitoring"],
         summary="Get Monitoring Metrics",
         description="Get system monitoring and performance metrics")
async def get_monitoring_metrics():
    """Get monitoring metrics for dashboard"""

    try:
        # Count model files
        model_files = list(OUTPUT_DIR.glob("*.joblib"))

        # Get recent training activity
        training_history = []
        log_file = STATE_DIR / 'training_results.log'

        if log_file.exists():
            recent_date = datetime.now() - timedelta(days=7)

            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if "timestamp" in entry:
                            entry_date = datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00"))
                            if entry_date >= recent_date:
                                training_history.append(entry)
                    except (json.JSONDecodeError, ValueError):
                        continue

        return {
            "models": {
                "total_count": len(model_files),
                "total_size_mb": sum(f.stat().st_size for f in model_files) / (1024 * 1024),
                "latest_model": model_manager.model_info.get("file"),
                "model_available": model_manager.model_info.get("available", False)
            },
            "training": {
                "last_week_runs": len(training_history),
                "last_week_success": sum(1 for t in training_history if t.get('training_successful', False)),
                "last_training": get_training_status().get("last_training_time"),
                "pipeline_active": True
            },
            "system": {
                "api_mode": "read-only",
                "uptime": "N/A",  # Could implement uptime tracking
                "memory_usage": "N/A"  # Could add memory monitoring
            },
            "directories": {
                "output_dir_exists": OUTPUT_DIR.exists(),
                "state_dir_exists": STATE_DIR.exists(),
                "log_files": len(list(STATE_DIR.glob("*.log")))
            }
        }

    except Exception as e:
        logger.error(f"Error getting monitoring metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get monitoring metrics: {str(e)}"
        )

@app.get("/config/info",
         response_model=Dict[str, Any],
         tags=["status"],
         summary="Get Configuration Information",
         description="Get current API configuration and settings")
async def get_config_info():
    """Get configuration information"""

    return {
        "api": {
            "title": "Read-Only XGBoost Trading API",
            "version": "2.0.0",
            "mode": "read-only",
            "allowed_methods": ["GET", "OPTIONS"],
            "cors_origins": allowed_origins
        },
        "paths": {
            "output_dir": str(OUTPUT_DIR),
            "state_dir": str(STATE_DIR),
            "model_dir": str(OUTPUT_DIR)
        },
        "environment": {
            "api_host": os.getenv('API_HOST', '0.0.0.0'),
            "api_port": os.getenv('API_PORT', '8000'),
            "domain": os.getenv('DOMAIN', 'localhost')
        },
        "features": {
            "model_predictions": "disabled",
            "trading_signals": "disabled",
            "training_trigger": "disabled",
            "status_monitoring": "enabled",
            "model_information": "enabled",
            "training_history": "enabled"
        },
        "security": {
            "authentication": "not_required",
            "rate_limiting": "not_implemented",
            "data_modification": "disabled"
        }
    }

# Run server
if __name__ == "__main__":
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 8000))

    logger.info(f"🚀 Starting Read-Only XGBoost Trading API")
    logger.info(f"📍 Server: http://{host}:{port}")
    logger.info(f"📚 Docs: http://{host}:{port}/docs")
    logger.info(f"🔍 Alternative Docs: http://{host}:{port}/redoc")
    logger.info(f"🔒 Mode: Read-Only (GET methods only)")

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )