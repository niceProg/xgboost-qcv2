#!/usr/bin/env python3
"""
Enhanced API Server untuk XGBoost Trading Model.
Menampilkan semua output_train, models terbaru, dan dataset_summary.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="XGBoost Trading Model API v2",
    description="Enhanced API untuk mengakses semua output XGBoost trading",
    version="2.0.0"
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
class FileInfo(BaseModel):
    name: str
    path: str
    size: int
    modified: datetime
    extension: str
    type: str  # Additional field for file type

class ModelInfo(BaseModel):
    name: str
    path: str
    size: int
    modified: datetime
    is_latest: bool
    feature_count: Optional[int] = None

class DatasetSummary(BaseModel):
    total_samples: int
    feature_count: int
    target_distribution: Dict[str, int]
    feature_list: List[str]
    last_updated: datetime

class OutputDirectoryResponse(BaseModel):
    path: str
    total_files: int
    file_types: Dict[str, int]
    files: List[FileInfo]

def get_all_files_with_type() -> List[FileInfo]:
    """Get all files in output_train with type classification."""
    files = []

    if not OUTPUT_DIR.exists():
        return files

    # Define file type mappings
    type_mappings = {
        # Models
        '.joblib': 'model',
        'model': 'model',
        'latest_model': 'model',

        # Data files
        '.parquet': 'data',
        '.csv': 'data',
        'merged_7_tables': 'merged_data',
        'labeled_data': 'labeled_data',
        'features_engineered': 'features',
        'X_train': 'training_data',
        'y_train': 'training_data',

        # Results and metrics
        'performance_metrics': 'metrics',
        'performance_report': 'report',
        'training_results': 'training_results',
        'feature_importance': 'importance',
        'trading_results': 'trading_results',

        # Account statements
        'rekening_koran': 'account_statement',
        'trade_events': 'trade_events',
        'trades': 'trades',

        # Configuration
        'column_mapping': 'config',
        'dataset_summary': 'summary',
        'feature_list': 'config',
        'model_features': 'config',
    }

    for file_path in OUTPUT_DIR.rglob('*'):
        if file_path.is_file():
            stat = file_path.stat()

            # Determine file type
            file_type = 'other'
            for pattern, type_name in type_mappings.items():
                if pattern in file_path.name.lower() or file_path.name.lower().endswith(pattern):
                    file_type = type_name
                    break

            file_info = FileInfo(
                name=file_path.name,
                path=str(file_path.relative_to(OUTPUT_DIR)),
                size=stat.st_size,
                modified=datetime.fromtimestamp(stat.st_mtime),
                extension=file_path.suffix.lower(),
                type=file_type
            )
            files.append(file_info)

    # Sort by type and then by modified time
    files.sort(key=lambda x: (x.type, x.modified), reverse=True)
    return files

def get_model_files() -> List[ModelInfo]:
    """Get all model files with additional info."""
    models = []

    if not OUTPUT_DIR.exists():
        return models

    # Get all joblib files
    model_files = list(OUTPUT_DIR.glob('*.joblib'))
    latest_model_path = OUTPUT_DIR / 'latest_model.joblib'

    # Load feature list if available
    feature_count = None
    feature_list_path = OUTPUT_DIR / 'model_features.txt'
    if feature_list_path.exists():
        with open(feature_list_path, 'r') as f:
            features = f.readlines()
            # Count actual feature lines (skip headers)
            feature_count = len([l for l in features if l.strip() and not l.startswith('#')])

    for model_path in model_files:
        stat = model_path.stat()
        model_info = ModelInfo(
            name=model_path.name,
            path=str(model_path.relative_to(OUTPUT_DIR)),
            size=stat.st_size,
            modified=datetime.fromtimestamp(stat.st_mtime),
            is_latest=(model_path == latest_model_path),
            feature_count=feature_count
        )
        models.append(model_info)

    # Sort by modified time (newest first)
    models.sort(key=lambda x: x.modified, reverse=True)
    return models

def get_dataset_summary() -> DatasetSummary:
    """Read dataset_summary.txt and parse it."""
    summary_path = OUTPUT_DIR / 'dataset_summary.txt'

    if not summary_path.exists():
        raise HTTPException(
            status_code=404,
            detail="dataset_summary.txt not found"
        )

    # Read the file
    with open(summary_path, 'r') as f:
        content = f.read()

    # Parse the content
    total_samples = 0
    feature_count = 0
    target_distribution = {}
    feature_list = []

    for line in content.split('\n'):
        line = line.strip()

        if 'Total samples:' in line:
            total_samples = int(line.split(':')[-1].strip())
        elif 'Feature count:' in line:
            feature_count = int(line.split(':')[-1].strip())
        elif 'Bullish (1):' in line:
            target_distribution['bullish'] = int(line.split(':')[-1].strip())
        elif 'Bearish (0):' in line:
            target_distribution['bearish'] = int(line.split(':')[-1].strip())

    # Get feature list from model_features.txt
    feature_list_path = OUTPUT_DIR / 'model_features.txt'
    if feature_list_path.exists():
        with open(feature_list_path, 'r') as f:
            features = f.readlines()
            feature_list = [f.strip() for f in features if f.strip() and not f.startswith('#')]

    stat = summary_path.stat()

    return DatasetSummary(
        total_samples=total_samples,
        feature_count=feature_count,
        target_distribution=target_distribution,
        feature_list=feature_list,
        last_updated=datetime.fromtimestamp(stat.st_mtime)
    )

def get_directory_stats() -> Dict[str, Any]:
    """Get statistics about the output directory."""
    stats = {
        'path': str(OUTPUT_DIR),
        'exists': OUTPUT_DIR.exists(),
        'total_size': 0,
        'file_count': 0,
        'directory_count': 0,
        'last_modified': None
    }

    if OUTPUT_DIR.exists():
        for item in OUTPUT_DIR.rglob('*'):
            if item.is_file():
                stats['total_size'] += item.stat().st_size
                stats['file_count'] += 1
                if not stats['last_modified'] or item.stat().st_mtime > stats['last_modified']:
                    stats['last_modified'] = item.stat().st_mtime
            elif item.is_dir():
                stats['directory_count'] += 1

        if stats['last_modified']:
            stats['last_modified'] = datetime.fromtimestamp(stats['last_modified'])

    return stats

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "message": "XGBoost Trading Model API v2",
        "version": "2.0.0",
        "description": "API for accessing XGBoost trading model outputs",
        "endpoints": {
            "output_directory": "/output_train - Root folder contents",
            "models": "/models - All model files",
            "latest_model": "/models/latest - Latest model info",
            "dataset_summary": "/dataset/summary - Dataset information",
            "stats": "/stats - Directory statistics"
        }
    }

@app.get("/output_train", response_model=OutputDirectoryResponse)
async def get_output_directory():
    """Get complete contents of output_train directory."""
    files = get_all_files_with_type()

    # Calculate file type counts
    file_types = {}
    for file in files:
        file_types[file.type] = file_types.get(file.type, 0) + 1

    # Get directory stats
    stats = get_directory_stats()

    return OutputDirectoryResponse(
        path=stats['path'],
        total_files=stats['file_count'],
        file_types=file_types,
        files=files
    )

@app.get("/output_train/browse/{path:path}")
async def browse_output_path(path: str):
    """Browse specific path in output_train."""
    full_path = OUTPUT_DIR / path

    # Security check - ensure path is within OUTPUT_DIR
    try:
        full_path.resolve().relative_to(OUTPUT_DIR.resolve())
    except ValueError:
        raise HTTPException(
            status_code=403,
            detail="Access denied - path outside output directory"
        )

    if not full_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Path not found: {path}"
        )

    if full_path.is_file():
        # Return file info
        stat = full_path.stat()
        return {
            "name": full_path.name,
            "type": "file",
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime),
            "path": str(full_path.relative_to(OUTPUT_DIR))
        }
    elif full_path.is_dir():
        # List directory contents
        contents = []
        for item in full_path.iterdir():
            stat = item.stat()
            contents.append({
                "name": item.name,
                "type": "directory" if item.is_dir() else "file",
                "size": stat.st_size if item.is_file() else 0,
                "modified": datetime.fromtimestamp(stat.st_mtime)
            })

        return {
            "name": full_path.name,
            "type": "directory",
            "path": str(full_path.relative_to(OUTPUT_DIR)),
            "contents": sorted(contents, key=lambda x: (x['type'], x['name']))
        }

@app.get("/models", response_model=List[ModelInfo])
async def get_all_models():
    """Get all model files with detailed information."""
    return get_model_files()

@app.get("/models/latest")
async def get_latest_model():
    """Get information about the latest model."""
    models = get_model_files()

    if not models:
        raise HTTPException(
            status_code=404,
            detail="No models found"
        )

    # Find the latest model (is_latest=True)
    latest = None
    for model in models:
        if model.is_latest:
            latest = model
            break

    if not latest:
        # Fallback to most recently modified
        latest = models[0]

    # Load additional model info if possible
    model_path = OUTPUT_DIR / latest.path

    try:
        model = joblib.load(model_path)

        # Get feature names if available
        feature_names = getattr(model, 'feature_names_in_', None)
        if feature_names is None and hasattr(model, 'get_booster'):
            booster = model.get_booster()
            if booster:
                feature_names = booster.feature_names

        return {
            **latest.dict(),
            "feature_names": list(feature_names) if feature_names else None,
            "n_features": len(feature_names) if feature_names else None,
            "n_classes": getattr(model, 'n_classes_', None)
        }
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return latest.dict()

@app.get("/models/download/{model_name}")
async def download_model(model_name: str):
    """Download a specific model file."""
    model_path = OUTPUT_DIR / model_name

    # Security check
    if not model_path.exists() or not model_path.suffix == '.joblib':
        raise HTTPException(
            status_code=404,
            detail="Model file not found"
        )

    return FileResponse(
        model_path,
        media_type='application/octet-stream',
        filename=model_name
    )

@app.get("/dataset/summary", response_model=DatasetSummary)
async def get_dataset_summary():
    """Get dataset summary information."""
    return get_dataset_summary()

@app.get("/dataset/features")
async def get_dataset_features():
    """Get list of all features."""
    feature_list_path = OUTPUT_DIR / 'model_features.txt'

    if not feature_list_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Feature list not found"
        )

    with open(feature_list_path, 'r') as f:
        features = f.readlines()

    feature_list = [f.strip() for f in features if f.strip() and not f.startswith('#')]

    return {
        "total_features": len(feature_list),
        "features": feature_list
    }

@app.get("/stats")
async def get_directory_statistics():
    """Get statistics about the output directory."""
    stats = get_directory_stats()

    # Get additional statistics
    files_by_type = {}
    files = get_all_files_with_type()
    for file in files:
        files_by_type[file.type] = files_by_type.get(file.type, 0) + 1

    # Get latest activity
    latest_files = {}
    for file in files[:10]:  # Top 10 most recent
        file_type = file.type
        if file_type not in latest_files:
            latest_files[file_type] = file.dict()

    return {
        "directory": stats,
        "files_by_type": files_by_type,
        "latest_activity": latest_files,
        "summary": {
            "total_files": stats['file_count'],
            "total_size_mb": round(stats['total_size'] / (1024 * 1024), 2),
            "last_activity": stats['last_modified']
        }
    }

@app.get("/preview/{file_name}")
async def preview_file(file_name: str, limit: int = 10):
    """Preview contents of a CSV or JSON file."""
    file_path = OUTPUT_DIR / file_name

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="File not found"
        )

    try:
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path, nrows=limit)
            return {
                "file": file_name,
                "type": "csv",
                "columns": list(df.columns),
                "preview": df.to_dict('records'),
                "total_rows": len(pd.read_csv(file_path))
            }
        elif file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
            return {
                "file": file_name,
                "type": "json",
                "preview": data
            }
        else:
            return {
                "file": file_name,
                "type": file_path.suffix,
                "message": "Preview not available for this file type"
            }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading file: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "api_server_v2:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )