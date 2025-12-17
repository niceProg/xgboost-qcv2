#!/usr/bin/env python3
"""
Structured XGBoost Trading API
Clean API endpoints for organized output_train structure
"""

import os
import json
import logging
import joblib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Structured XGBoost Trading API",
    description="""
    ## 📁 Structured Trading API

    Clean API for accessing organized XGBoost model outputs:

    ### 📊 Directory Structure:
    ```
    output_train/
    ├── models/          # Trained model files
    ├── datasets/         # Training datasets
    │   ├── raw/         # Raw market data
    │   └── *.parquet    # Processed datasets
    ├── reports/         # Performance reports
    ├── logs/           # Training logs
    └── features/       # Feature files
    ```

    ### 🎯 API Endpoints:
    - `/output_train/` - Directory listing
    - `/output_train/models/` - All model files
    - `/output_train/models/latest` - Latest model
    - `/output_train/datasets/summary` - Dataset summary

    ---

    **Base URL**: `https://your-domain.com:8000`
    **Documentation**: `/docs`

    **Version**: 2.0.0
    **Mode**: Read-Only
    """,
    version="2.0.0"
)

# CORS middleware
allowed_origins = [
    os.getenv('QUANTCONNECT_CORS_ORIGIN', 'https://www.quantconnect.com'),
    "https://your-domain.com",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

# Configuration
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', '../output_train'))

class DirectoryListing(BaseModel):
    """Directory listing response"""
    path: str
    directories: List[str]
    files: List[str]
    total_items: int

class FileInfo(BaseModel):
    """File information response"""
    name: str
    path: str
    size_bytes: int
    size_mb: float
    modified_time: str
    type: str

def get_directory_listing(dir_path: Path) -> DirectoryListing:
    """Get directory listing"""
    try:
        directories = [d.name for d in dir_path.iterdir() if d.is_dir()]
        files = [f.name for f in dir_path.iterdir() if f.is_file()]

        return DirectoryListing(
            path=str(dir_path.relative_to(OUTPUT_DIR.parent)),
            directories=sorted(directories),
            files=sorted(files),
            total_items=len(directories) + len(files)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading directory: {str(e)}")

def get_file_info(file_path: Path) -> FileInfo:
    """Get file information"""
    try:
        stat = file_path.stat()

        return FileInfo(
            name=file_path.name,
            path=str(file_path.relative_to(OUTPUT_DIR.parent)),
            size_bytes=stat.st_size,
            size_mb=stat.st_size / (1024 * 1024),
            modified_time=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            type=file_path.suffix
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

# API Endpoints

@app.get("/",
         summary="API Root",
         description="Get API information and available endpoints")
async def root():
    """API root endpoint"""
    return {
        "name": "Structured XGBoost Trading API",
        "version": "2.0.0",
        "mode": "read-only",
        "description": "Clean API for organized XGBoost model outputs",
        "endpoints": {
            "directory": "/output_train/",
            "models": "/output_train/models/",
            "latest_model": "/output_train/models/latest",
            "dataset_summary": "/output_train/datasets/summary",
            "datasets": "/output_train/datasets/",
            "docs": "/docs"
        },
        "output_directory": str(OUTPUT_DIR),
        "structure_ready": OUTPUT_DIR.exists()
    }

@app.get("/output_train/",
         response_model=DirectoryListing,
         summary="Output Train Directory",
         description="Get listing of output_train directory contents")
async def get_output_train():
    """Get output_train directory listing"""
    if not OUTPUT_DIR.exists():
        raise HTTPException(status_code=404, detail="output_train directory not found")

    return get_directory_listing(OUTPUT_DIR)

@app.get("/output_train/models/",
         response_model=List[FileInfo],
         summary="All Model Files",
         description="Get list of all trained model files")
async def get_models():
    """Get all model files"""
    models_dir = OUTPUT_DIR / "models"

    if not models_dir.exists():
        # Check if models are in root directory (old structure)
        model_files = list(OUTPUT_DIR.glob("*.joblib"))
        if not model_files:
            return []

        return [get_file_info(f) for f in model_files]

    model_files = list(models_dir.glob("*.joblib"))
    return [get_file_info(f) for f in model_files]

@app.get("/output_train/models/latest",
         summary="Latest Model File",
         description="Get latest trained model file")
async def get_latest_model():
    """Get latest model file"""
    models_dir = OUTPUT_DIR / "models"

    # Check for latest_model symlink first
    latest_symlink = models_dir / "latest_model.joblib"
    if latest_symlink.exists() and latest_symlink.is_symlink():
        return FileResponse(
            path=str(latest_symlink),
            filename="latest_model.joblib",
            media_type="application/octet-stream"
        )

    # Fallback to finding latest by modification time
    model_files = []

    # Check models directory
    if models_dir.exists():
        model_files.extend(models_dir.glob("*.joblib"))

    # Check root directory
    model_files.extend(OUTPUT_DIR.glob("*.joblib"))

    if not model_files:
        raise HTTPException(status_code=404, detail="No model files found")

    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)

    return FileResponse(
        path=str(latest_model),
        filename="latest_model.joblib",
        media_type="application/octet-stream"
    )

@app.get("/output_train/datasets/summary",
         response_class=PlainTextResponse,
         summary="Dataset Summary",
         description="Get current dataset summary information")
async def get_dataset_summary():
    """Get current dataset summary"""
    dataset_summary_file = OUTPUT_DIR / "datasets" / "dataset_summary.txt"

    if not dataset_summary_file.exists():
        # Fallback to root directory
        dataset_summary_file = OUTPUT_DIR / "dataset_summary.txt"

    if not dataset_summary_file.exists():
        raise HTTPException(status_code=404, detail="Dataset summary file not found")

    return PlainTextResponse(
        content=dataset_summary_file.read_text(),
        media_type="text/plain"
    )

@app.get("/output_train/datasets/",
         response_model=DirectoryListing,
         summary="Datasets Directory",
         description="Get datasets directory contents")
async def get_datasets():
    """Get datasets directory listing"""
    datasets_dir = OUTPUT_DIR / "datasets"

    if not datasets_dir.exists():
        # Fallback to listing dataset files in root
        dataset_files = []
        if OUTPUT_DIR.exists():
            dataset_files = [f.name for f in OUTPUT_DIR.glob("*.parquet")]

        return DirectoryListing(
            path="output_train",
            directories=[],
            files=sorted(dataset_files),
            total_items=len(dataset_files)
        )

    return get_directory_listing(datasets_dir)

@app.get("/output_train/datasets/raw/",
         response_model=List[FileInfo],
         summary="Raw Dataset Files",
         description="Get raw dataset files from database")
async def get_raw_datasets():
    """Get raw dataset files"""
    raw_dir = OUTPUT_DIR / "datasets" / "raw"

    if not raw_dir.exists():
        return []

    raw_files = list(raw_dir.glob("*.parquet"))
    return [get_file_info(f) for f in raw_files]

@app.get("/output_train/datasets/{filename:path}",
         summary="Get Dataset File",
         description="Get specific dataset file")
async def get_dataset_file(filename: str):
    """Get specific dataset file"""
    # Try datasets directory first
    dataset_file = OUTPUT_DIR / "datasets" / filename

    if not dataset_file.exists():
        # Try raw subdirectory
        dataset_file = OUTPUT_DIR / "datasets" / "raw" / filename

    if not dataset_file.exists():
        # Try root directory
        dataset_file = OUTPUT_DIR / filename

    if not dataset_file.exists():
        raise HTTPException(status_code=404, detail=f"Dataset file not found: {filename}")

    return FileResponse(
        path=str(dataset_file),
        filename=filename,
        media_type="application/octet-stream"
    )

@app.get("/output_train/models/{filename}",
         summary="Get Model File",
         description="Get specific model file")
async def get_model_file(filename: str):
    """Get specific model file"""
    model_file = OUTPUT_DIR / "models" / filename

    if not model_file.exists():
        model_file = OUTPUT_DIR / filename

    if not model_file.exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {filename}")

    return FileResponse(
        path=str(model_file),
        filename=filename,
        media_type="application/octet-stream"
    )

@app.get("/output_train/{subdirectory:path}",
         summary="Get Subdirectory",
         description="Get contents of any subdirectory in output_train")
async def get_subdirectory(subdirectory: str):
    """Get subdirectory contents"""
    sub_path = OUTPUT_DIR / subdirectory

    if not sub_path.exists():
        raise HTTPException(status_code=404, detail=f"Subdirectory not found: {subdirectory}")

    if not sub_path.is_dir():
        # Return file if it's a file
        return FileResponse(
            path=str(sub_path),
            filename=sub_path.name,
            media_type="application/octet-stream"
        )

    return get_directory_listing(sub_path)

@app.get("/structure",
         summary="Output Structure Info",
         description="Get information about output_train structure")
async def get_structure_info():
    """Get structure information"""
    structure_file = OUTPUT_DIR / "structure_metadata.json"

    if structure_file.exists():
        return json.loads(structure_file.read_text())

    # Fallback basic structure
    return {
        "directory": str(OUTPUT_DIR),
        "exists": OUTPUT_DIR.exists(),
        "structure": {
            "models": "Trained model files",
            "datasets": "Training and raw data",
            "reports": "Performance reports",
            "logs": "Training logs",
            "features": "Feature files"
        }
    }

# Health check
@app.get("/health",
         summary="Health Check",
         description="API health check")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "output_directory": str(OUTPUT_DIR),
        "directory_exists": OUTPUT_DIR.exists(),
        "timestamp": datetime.now().isoformat()
    }

# Run server
if __name__ == "__main__":
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 8000))

    logger.info(f"🚀 Starting Structured XGBoost Trading API")
    logger.info(f"📍 Server: http://{host}:{port}")
    logger.info(f"📚 Docs: http://{host}:{port}/docs")
    logger.info(f"📁 Output directory: {OUTPUT_DIR}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )