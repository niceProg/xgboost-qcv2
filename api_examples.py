#!/usr/bin/env python3
"""
Example API endpoints for XGBoost trading model server.
This demonstrates how to expose the trained models and results through REST API.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from sqlalchemy.orm import Session

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from database_storage import DatabaseStorage, get_session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="XGBoost Trading Model API",
    description="API for accessing trained XGBoost models and trading results",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database dependency
def get_db_session():
    db_storage = DatabaseStorage()
    return db_storage.get_session()

# Pydantic models
class TrainingSession(BaseModel):
    session_id: str
    created_at: datetime
    status: str
    total_samples: Optional[int] = None
    feature_count: Optional[int] = None
    test_auc: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    total_return: Optional[float] = None

class PredictionRequest(BaseModel):
    features: Dict[str, float]
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    session_id: str
    timestamp: datetime

class PerformanceMetrics(BaseModel):
    total_return: float
    cagr: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    winning_trades: int

# Initialize database storage
db_storage = DatabaseStorage()

@app.get("/api/v1/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "XGBoost Trading Model API",
        "version": "1.0.0",
        "endpoints": {
            "sessions": "/api/v1/sessions",
            "models": "/api/v1/models",
            "predict": "/api/v1/predict",
            "performance": "/api/v1/performance",
            "latest_model": "/api/v1/models/latest"
        }
    }

@app.get("/")
async def root_legacy():
    """Legacy root endpoint - redirect to v1."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/api/v1/")

@app.get("/api/v1/sessions", response_model=List[TrainingSession])
async def get_training_sessions(
    limit: int = Field(default=10, ge=1, le=100),
    status: Optional[str] = None
):
    """Get list of training sessions."""
    try:
        sessions = db_storage.get_session_history(limit=limit)

        # Filter by status if provided
        if status:
            sessions = [s for s in sessions if s.get('status') == status]

        return [TrainingSession(**session) for session in sessions]
    except Exception as e:
        logger.error(f"Failed to get sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sessions")

@app.get("/api/v1/sessions/{session_id}")
async def get_session_details(session_id: str):
    """Get detailed information about a specific training session."""
    try:
        db = db_storage.get_session()
        # Query for session details with related data
        from database_storage import TrainingSession, ModelStorage, FeatureStorage, EvaluationResult

        session = db.query(TrainingSession).filter(
            TrainingSession.session_id == session_id
        ).first()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Get related models
        models = db.query(ModelStorage).filter(
            ModelStorage.session_id == session_id
        ).all()

        # Get feature files
        features = db.query(FeatureStorage).filter(
            FeatureStorage.session_id == session_id
        ).all()

        # Get evaluations
        evaluations = db.query(EvaluationResult).filter(
            EvaluationResult.session_id == session_id
        ).all()

        return {
            "session": {
                "session_id": session.session_id,
                "created_at": session.created_at,
                "status": session.status,
                "notes": session.notes,
                "total_samples": session.total_samples,
                "feature_count": session.feature_count,
                "metrics": {
                    "test_auc": session.test_auc,
                    "test_accuracy": session.test_accuracy,
                    "test_f1": session.test_f1,
                    "sharpe_ratio": session.sharpe_ratio,
                    "total_return": session.total_return,
                    "max_drawdown": session.max_drawdown
                }
            },
            "models": [{
                "model_name": m.model_name,
                "created_at": m.created_at,
                "is_latest": m.is_latest,
                "train_score": m.train_score,
                "val_score": m.val_score
            } for m in models],
            "features": [{
                "feature_type": f.feature_type,
                "table_name": f.table_name,
                "row_count": f.row_count,
                "file_size": f.file_size_bytes
            } for f in features],
            "evaluations": [{
                "eval_type": e.eval_type,
                "created_at": e.created_at,
                "total_return": e.total_return,
                "sharpe_ratio": e.sharpe_ratio,
                "max_drawdown": e.max_drawdown
            } for e in evaluations]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session details: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve session details")

@app.get("/api/v1/models/latest")
async def get_latest_model():
    """Get information about the latest trained model."""
    try:
        model, feature_names, session_id = db_storage.load_latest_model()

        return {
            "model": {
                "session_id": session_id,
                "feature_count": len(feature_names),
                "feature_names": feature_names,
                "model_type": "XGBoostClassifier"
            }
        }
    except Exception as e:
        logger.error(f"Failed to load latest model: {e}")
        raise HTTPException(status_code=500, detail="Failed to load latest model")

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    session_id: Optional[str] = None
):
    """Make prediction using the latest trained model."""
    try:
        # Load model
        model, feature_names, model_session_id = db_storage.load_latest_model(session_id)

        # Prepare features
        features_df = pd.DataFrame([request.features])

        # Ensure all required features are present
        missing_features = set(feature_names) - set(request.features.keys())
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing features: {missing_features}"
            )

        # Select and order features
        X = features_df[feature_names]

        # Make prediction
        probability = float(model.predict_proba(X)[0, 1])
        prediction = int(probability > request.threshold)

        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            session_id=model_session_id,
            timestamp=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.get("/api/v1/performance/{session_id}", response_model=PerformanceMetrics)
async def get_performance_metrics(session_id: str):
    """Get performance metrics for a specific training session."""
    try:
        db = db_storage.get_session()
        from database_storage import TrainingSession, EvaluationResult

        # Get session metrics
        session = db.query(TrainingSession).filter(
            TrainingSession.session_id == session_id
        ).first()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Get evaluation results
        eval_results = db.query(EvaluationResult).filter(
            EvaluationResult.session_id == session_id,
            EvaluationResult.eval_type == 'backtest'
        ).first()

        if eval_results:
            return PerformanceMetrics(
                total_return=eval_results.total_return or 0.0,
                cagr=eval_results.total_return or 0.0,  # Using total_return as placeholder
                max_drawdown=eval_results.max_drawdown or 0.0,
                sharpe_ratio=eval_results.sharpe_ratio or 0.0,
                win_rate=eval_results.win_rate or 0.0,
                total_trades=eval_results.total_trades or 0,
                winning_trades=eval_results.winning_trades or 0
            )
        else:
            # Return session metrics if no evaluation results
            return PerformanceMetrics(
                total_return=session.total_return or 0.0,
                cagr=session.total_return or 0.0,
                max_drawdown=session.max_drawdown or 0.0,
                sharpe_ratio=session.sharpe_ratio or 0.0,
                win_rate=0.0,  # Not available at session level
                total_trades=0,
                winning_trades=0
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve performance metrics")

@app.get("/api/v1/export/{session_id}")
async def export_session_data(session_id: str, format: str = "json"):
    """Export all data for a training session."""
    try:
        db = db_storage.get_session()
        from database_storage import TrainingSession, ModelStorage, EvaluationResult

        # Get all session data
        session = db.query(TrainingSession).filter(
            TrainingSession.session_id == session_id
        ).first()

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        export_data = {
            "session": {
                "session_id": session.session_id,
                "created_at": session.created_at.isoformat(),
                "status": session.status,
                "filters": {
                    "exchange": json.loads(session.exchange_filter) if session.exchange_filter else None,
                    "symbol": json.loads(session.symbol_filter) if session.symbol_filter else None,
                    "interval": json.loads(session.interval_filter) if session.interval_filter else None
                }
            },
            "model": None,
            "evaluation": None
        }

        # Get model info
        model = db.query(ModelStorage).filter(
            ModelStorage.session_id == session_id,
            ModelStorage.is_latest == True
        ).first()

        if model:
            export_data["model"] = {
                "feature_names": json.loads(model.feature_names),
                "hyperparams": json.loads(model.hyperparams),
                "metrics": {
                    "train_score": model.train_score,
                    "val_score": model.val_score
                }
            }

        # Get evaluation
        eval_result = db.query(EvaluationResult).filter(
            EvaluationResult.session_id == session_id
        ).first()

        if eval_result:
            export_data["evaluation"] = json.loads(eval_result.detailed_metrics)

        return export_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export session: {e}")
        raise HTTPException(status_code=500, detail="Export failed")

@app.delete("/api/v1/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a training session and all related data."""
    try:
        # This would require implementing delete functionality in DatabaseStorage
        # For now, return not implemented
        raise HTTPException(
            status_code=501,
            detail="Delete functionality not implemented yet"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail="Delete failed")

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api_examples:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("ENV", "development") == "development"
    )