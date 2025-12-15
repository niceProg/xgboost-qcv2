#!/usr/bin/env python3
"""
API Server untuk XGBoost Trading Model.
Menyediakan endpoints untuk mengakses model dan hasil yang tersimpan di output_train.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import pickle
import joblib
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="XGBoost Trading Model API",
    description="API untuk mengakses model XGBoost trading dan hasil evaluasi",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Output directory
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', './output_train'))

# Pydantic models
class ModelInfo(BaseModel):
    model_name: str
    session_id: str
    created_at: datetime
    file_path: str
    feature_count: Optional[int] = None
    metrics: Optional[Dict] = None

class PredictionRequest(BaseModel):
    features: Dict[str, float]
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    timestamp: datetime
    model_info: ModelInfo

class FileListResponse(BaseModel):
    file_type: str
    files: List[Dict]

class PerformanceMetrics(BaseModel):
    session_id: str
    total_return: float
    cagr: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    trading_days: int

def get_output_files() -> Dict[str, List[Dict]]:
    """Get all files in output directory organized by type."""
    files = {
        'raw_data': [],
        'features': [],
        'labels': [],
        'models': [],
        'evaluations': [],
        'trades': [],
        'logs': []
    }

    if not OUTPUT_DIR.exists():
        return files

    for file_path in OUTPUT_DIR.rglob('*'):
        if file_path.is_file():
            stat = file_path.stat()
            file_info = {
                'name': file_path.name,
                'path': str(file_path),
                'size': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'extension': file_path.suffix.lower()
            }

            # Categorize by file type
            if 'table' in file_path.name or file_path.suffix == '.parquet':
                if 'merged' in file_path.name or 'labeled' in file_path.name:
                    files['features'].append(file_info)
                elif 'features' in file_path.name:
                    files['features'].append(file_info)
                else:
                    files['raw_data'].append(file_info)
            elif 'model' in file_path.name or file_path.suffix in ['.joblib', '.pkl']:
                files['models'].append(file_info)
            elif 'trade' in file_path.name or 'rekening' in file_path.name:
                files['trades'].append(file_info)
            elif 'performance' in file_path.name or 'metrics' in file_path.name:
                files['evaluations'].append(file_info)
            elif file_path.suffix == '.log':
                files['logs'].append(file_info)
            else:
                files['labels'].append(file_info)

    return files

def load_latest_model() -> tuple:
    """Load the latest trained model."""
    # Cek file latest_model.joblib
    latest_model_path = OUTPUT_DIR / 'latest_model.joblib'

    if not latest_model_path.exists():
        # Jika tidak ada, cari model terbaru
        model_files = list(OUTPUT_DIR.glob('xgboost_trading_model_*.joblib'))
        if not model_files:
            raise HTTPException(
                status_code=404,
                detail="No trained model found"
            )

        # Get the most recent model
        latest_model_path = max(model_files, key=lambda x: x.stat().st_mtime)

    try:
        model = joblib.load(latest_model_path)

        # Load feature list if available
        feature_file = OUTPUT_DIR / 'model_features.txt'
        if feature_file.exists():
            with open(feature_file, 'r') as f:
                features = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        else:
            features = []

        return model, features, latest_model_path
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {str(e)}"
        )

def load_performance_metrics(session_id: str = None) -> Dict:
    """Load performance metrics from JSON files."""
    metrics_files = list(OUTPUT_DIR.glob('performance_metrics_*.json'))

    if session_id:
        # Look for specific session
        metrics_files = [f for f in metrics_files if session_id in f.name]

    if not metrics_files:
        raise HTTPException(
            status_code=404,
            detail="No performance metrics found"
        )

    # Get the most recent metrics file
    latest_metrics = max(metrics_files, key=lambda x: x.stat().st_mtime)

    try:
        with open(latest_metrics, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load metrics: {str(e)}"
        )

@app.get("/")
async def root():
    """Root endpoint dengan API informasi."""
    return {
        "message": "XGBoost Trading Model API",
        "version": "1.0.0",
        "status": "running",
        "output_directory": str(OUTPUT_DIR),
        "endpoints": {
            "models": "/api/v1/models",
            "latest_model": "/api/v1/models/latest",
            "model_history": "/api/v1/models/history",
            "predict": "/api/v1/predict",
            "performance": "/api/v1/performance",
            "files": "/api/v1/files",
            "download": "/api/v1/download/{filename}"
        }
    }

@app.get("/api/v1/")
async def api_v1_root():
    """API v1 root endpoint."""
    return {
        "message": "XGBoost Trading Model API v1",
        "endpoints": [
            "/api/v1/models/latest",
            "/api/v1/models/history",
            "/api/v1/predict",
            "/api/v1/performance/{session_id}",
            "/api/v1/files",
            "/api/v1/download/{filename}"
        ]
    }

@app.get("/api/v1/models/latest", response_model=ModelInfo)
async def get_latest_model():
    """Get informasi model terbaru."""
    try:
        model, features, model_path = load_latest_model()

        # Untuk latest, nama file selalu "latest_model.joblib"
        model_name = "latest_model.joblib"

        # Extract session info if the actual file has timestamp
        if "xgboost_trading_model_" in model_path.name:
            session_id = model_path.stem.replace('xgboost_trading_model_', '')
        else:
            session_id = "latest"

        # Try to load performance metrics
        try:
            metrics = load_performance_metrics(session_id)
        except:
            metrics = None

        return ModelInfo(
            model_name=model_name,  # Selalu "latest_model.joblib"
            session_id=session_id,
            created_at=datetime.fromtimestamp(model_path.stat().st_mtime),
            file_path=str(model_path),  # Path ke file aktual
            feature_count=len(features),
            metrics=metrics
        )
    except Exception as e:
        logger.error(f"Error getting latest model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/models/history", response_model=List[ModelInfo])
async def get_model_history():
    """Get semua model yang tersedia."""
    model_files = []

    # Only include timestamped models in history, not "latest_model.joblib"
    for file_path in OUTPUT_DIR.glob('xgboost_trading_model_*.joblib'):
        stat = file_path.stat()
        session_id = file_path.stem.replace('xgboost_trading_model_', '')

        # Try to load performance metrics
        try:
            metrics = load_performance_metrics(session_id)
        except:
            metrics = None

        model_files.append(ModelInfo(
            model_name=file_path.name,  # e.g., "xgboost_trading_model_20241215_143000.joblib"
            session_id=session_id,
            created_at=datetime.fromtimestamp(stat.st_mtime),
            file_path=str(file_path),
            metrics=metrics
        ))

    # Sort by creation date (newest first)
    model_files.sort(key=lambda x: x.created_at, reverse=True)

    return model_files

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Buat prediksi menggunakan model terbaru."""
    try:
        # Load model
        model, feature_names, model_path = load_latest_model()

        # Prepare features
        features_df = pd.DataFrame([request.features])

        # Check if all required features are present
        missing_features = set(feature_names) - set(request.features.keys())
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing features: {list(missing_features)}"
            )

        # Select and order features
        X = features_df[feature_names]

        # Make prediction
        probability = float(model.predict_proba(X)[0, 1])
        prediction = int(probability > request.threshold)

        # Model info
        # Untuk response, gunakan nama "latest_model.joblib" untuk konsistensi
        model_info = ModelInfo(
            model_name="latest_model.joblib",
            session_id="latest",
            created_at=datetime.fromtimestamp(model_path.stat().st_mtime),
            file_path=str(model_path),  # Path ke file aktual
            feature_count=len(feature_names)
        )

        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            timestamp=datetime.utcnow(),
            model_info=model_info
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/v1/performance/{session_id}", response_model=PerformanceMetrics)
async def get_performance(session_id: str):
    """Get performance metrics untuk session tertentu."""
    try:
        metrics = load_performance_metrics(session_id)

        return PerformanceMetrics(
            session_id=session_id,
            total_return=metrics.get('total_return', 0.0),
            cagr=metrics.get('cagr', 0.0),
            max_drawdown=metrics.get('max_drawdown', 0.0),
            sharpe_ratio=metrics.get('sharpe_ratio', 0.0),
            win_rate=metrics.get('win_rate', 0.0),
            total_trades=metrics.get('total_trades', 0),
            winning_trades=metrics.get('winning_trades', 0),
            losing_trades=metrics.get('total_trades', 0) - metrics.get('winning_trades', 0),
            avg_win=metrics.get('avg_win', 0.0),
            avg_loss=metrics.get('avg_loss', 0.0),
            trading_days=metrics.get('trading_days', 0)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get performance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load performance metrics")

@app.get("/api/v1/files", response_model=FileListResponse)
async def list_files(file_type: Optional[str] = None):
    """List semua file di output directory."""
    files = get_output_files()

    if file_type and file_type in files:
        return FileListResponse(file_type=file_type, files=files[file_type])
    elif not file_type:
        # Return all files with their types
        all_files = []
        for ftype, flist in files.items():
            for f in flist:
                f['type'] = ftype
                all_files.append(f)
        return FileListResponse(file_type='all', files=all_files)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Available types: {list(files.keys())}"
        )

@app.get("/api/v1/download/{filename}")
async def download_file(filename: str):
    """Download file dari output directory."""
    # Find the file
    file_path = None
    for path in OUTPUT_DIR.rglob('*'):
        if path.name == filename:
            file_path = path
            break

    if not file_path or not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File {filename} not found"
        )

    # Check file size (limit to 100MB)
    if file_path.stat().st_size > 100 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="File too large"
        )

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type='application/octet-stream'
    )

@app.get("/api/v1/file/view/{filename}")
async def view_file_content(filename: str, limit: int = 100):
    """View content dari file (text/csv/json)."""
    # Find the file
    file_path = None
    for path in OUTPUT_DIR.rglob('*'):
        if path.name == filename:
            file_path = path
            break

    if not file_path or not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File {filename} not found"
        )

    try:
        if file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                content = json.load(f)
            return {"filename": filename, "content": content}
        elif file_path.suffix in ['.csv', '.txt']:
            lines = []
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    if i >= limit:
                        break
                    lines.append(line.rstrip('\n'))
            return {"filename": filename, "lines": lines}
        elif file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
            return {
                "filename": filename,
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "head": df.head(limit).to_dict(orient='records')
            }
        else:
            return {"filename": filename, "message": "Binary file - cannot view content"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read file: {str(e)}"
        )

@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint."""
    # Count latest model and timestamped models separately
    latest_model = OUTPUT_DIR / 'latest_model.joblib'
    timestamped_models = list(OUTPUT_DIR.glob('xgboost_trading_model_*.joblib'))

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "output_dir_exists": OUTPUT_DIR.exists(),
        "has_latest_model": latest_model.exists(),
        "timestamped_model_count": len(timestamped_models),
        "total_models": (1 if latest_model.exists() else 0) + len(timestamped_models)
    }

if __name__ == "__main__":
    # Run the API server
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("ENV", "production") == "development"
    )