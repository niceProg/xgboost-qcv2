#!/usr/bin/env python3
"""
QuantConnect API Server - Clean and Simple Integration
Provides endpoints for QuantConnect trading algorithms.
"""

import os
import json
import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="XGBoost QuantConnect API",
    description="Clean API for QuantConnect trading algorithms",
    version="1.0.0"
)

# CORS middleware for QuantConnect
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv('QUANTCONNECT_CORS_ORIGIN', 'https://www.quantconnect.com')],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Configuration
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', '../output_train'))
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', 8000))

# Pydantic models
class PredictionRequest(BaseModel):
    features: Dict[str, float]

class PredictionResponse(BaseModel):
    prediction: int  # 0 or 1
    confidence: float
    probability: float
    timestamp: str

class SignalRequest(BaseModel):
    exchange: str = "binance"
    symbol: str = "BTCUSDT"
    interval: str = "1h"

class SignalResponse(BaseModel):
    signal: str  # "BUY", "SELL", "HOLD"
    confidence: float
    price: Optional[float] = None
    recommendation: Dict
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_available: bool
    last_update: Optional[str]
    uptime: str

# Global variables
latest_model = None
model_metadata = {}
start_time = datetime.now()

def load_latest_model():
    """Load the latest XGBoost model."""
    global latest_model, model_metadata

    try:
        models_dir = OUTPUT_DIR / 'models'
        if not models_dir.exists():
            models_dir = OUTPUT_DIR

        # Find latest model file
        model_files = list(models_dir.glob("xgboost_trading_model_*.joblib"))
        if not model_files:
            return False

        model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_file = model_files[0]

        # Load model
        latest_model = joblib.load(latest_file)

        # Load metadata
        perf_file = OUTPUT_DIR / 'model_performance.json'
        if perf_file.exists():
            with open(perf_file, 'r') as f:
                state_data = json.load(f)
                model_metadata = {
                    'file_name': latest_file.name,
                    'last_training': state_data.get('last_training'),
                    'version': state_data.get('model_version'),
                    'total_updates': state_data.get('total_updates'),
                    'performance': state_data.get('latest_performance', {})
                }
        else:
            model_metadata = {
                'file_name': latest_file.name,
                'last_training': None,
                'version': 1
            }

        # Get feature names
        if hasattr(latest_model, 'feature_names_in_'):
            model_metadata['n_features'] = len(latest_model.feature_names_in_)
            model_metadata['feature_names'] = latest_model.feature_names_in_.tolist()
        else:
            booster = latest_model.get_booster()
            if booster and hasattr(booster, 'feature_names'):
                model_metadata['n_features'] = len(booster.feature_names)
                model_metadata['feature_names'] = booster.feature_names

        logger.info(f"✅ Loaded model: {model_metadata['file_name']}")
        return True

    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        latest_model = None
        model_metadata = {}
        return False

# Load model at startup
load_latest_model()

@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "service": "XGBoost QuantConnect API",
        "status": "running",
        "version": "1.0.0",
        "model_loaded": latest_model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = datetime.now() - start_time

    return HealthResponse(
        status="healthy" if latest_model else "unhealthy",
        model_available=latest_model is not None,
        last_update=model_metadata.get('last_training'),
        uptime=str(uptime)
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction using latest model."""
    if latest_model is None:
        raise HTTPException(status_code=503, detail="Model not available")

    try:
        # Get expected features
        expected_features = model_metadata.get('feature_names', [])
        if not expected_features:
            raise HTTPException(status_code=500, detail="Model feature names not available")

        # Prepare features
        features_vector = []
        missing_features = []

        for feature in expected_features:
            if feature in request.features:
                value = request.features[feature]
                if not np.isfinite(value):
                    value = 0.0
                features_vector.append(value)
            else:
                missing_features.append(feature)
                features_vector.append(0.0)  # Default value for missing feature

        # Make prediction
        X = np.array(features_vector).reshape(1, -1)
        prediction = latest_model.predict(X)[0]
        probability = float(latest_model.predict_proba(X)[0, 1])
        confidence = max(probability, 1 - probability)

        return PredictionResponse(
            prediction=int(prediction),
            confidence=float(confidence),
            probability=float(probability),
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/signal", response_model=SignalResponse)
async def generate_signal(request: SignalRequest):
    """Generate trading signal."""
    try:
        # Get current market data (simulate for now)
        market_data = get_market_data(request.exchange, request.symbol, request.interval)

        # Engineer features for current data
        features = engineer_features(market_data)
        if not features:
            raise HTTPException(status_code=500, detail="Failed to engineer features")

        # Make prediction
        pred_request = PredictionRequest(features=features)
        prediction = await predict(pred_request)

        # Determine signal
        probability = prediction.probability
        confidence = prediction.confidence

        # Signal thresholds
        buy_threshold = 0.55
        sell_threshold = 0.45

        if probability > buy_threshold:
            signal = "BUY"
        elif probability < sell_threshold:
            signal = "SELL"
        else:
            signal = "HOLD"

        # Generate recommendation
        recommendation = generate_recommendation(
            signal,
            confidence,
            market_data,
            probability
        )

        return SignalResponse(
            signal=signal,
            confidence=confidence,
            price=market_data.get('close'),
            recommendation=recommendation,
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signal generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Signal generation failed: {str(e)}")

@app.get("/models", response_model=Dict)
async def get_models():
    """Get model information."""
    return {
        "model_available": latest_model is not None,
        "metadata": model_metadata,
        "current_time": datetime.now().isoformat()
    }

@app.get("/status", response_model=Dict)
async def get_status():
    """Get system status."""
    return {
        "api_status": "running",
        "model_loaded": latest_model is not None,
        "model_info": model_metadata,
        "uptime": str(datetime.now() - start_time),
        "current_time": datetime.now().isoformat(),
        "database_status": "Connected" if check_database_connection() else "Disconnected"
    }

# Helper functions
def get_market_data(exchange: str, symbol: str, interval: str) -> Dict:
    """Get current market data (simulated)."""
    # In production, this would query database or exchange API
    # For now, simulate realistic data

    base_price = 42000.0 if symbol == "BTCUSDT" else 3000.0
    import random

    # Simulate realistic price movement
    current_time = datetime.now()
    price_variation = random.uniform(-0.02, 0.02)  # ±2%

    close = base_price * (1 + price_variation)
    open_price = close * random.uniform(0.998, 1.002)
    high = max(open_price, close) * random.uniform(1.0, 1.01)
    low = min(open_price, close) * random.uniform(0.99, 1.0)
    volume = random.uniform(1000000, 5000000)

    return {
        "exchange": exchange,
        "symbol": symbol,
        "interval": interval,
        "time": current_time.isoformat(),
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    }

def engineer_features(market_data: Dict) -> Dict[str, float]:
    """Engineer features from market data."""
    try:
        close = market_data.get('close', 0)
        volume = market_data.get('volume', 0)
        high = market_data.get('high', 0)
        low = market_data.get('low', 0)
        open_price = market_data.get('open', close)

        # Basic features
        features = {
            "price_close": close,
            "price_high": high,
            "price_low": low,
            "price_open": open_price,
            "volume_usd": close * volume,
            "price_range": high - low,
            "body_size": abs(close - open_price),
            "upper_wick": high - max(open_price, close),
            "lower_wick": min(open_price, close) - low,
        }

        # Calculate derived features
        if high != low:
            features['close_position'] = (close - low) / (high - low)

        if open_price > 0:
            features['price_change_pct'] = ((close - open_price) / open_price) * 100

        if close > 0:
            features['volume_price_ratio'] = volume / close

        return features

    except Exception as e:
        logger.error(f"Error engineering features: {e}")
        return {}

def generate_recommendation(signal: str, confidence: float, market_data: Dict, probability: float) -> Dict:
    """Generate trading recommendation."""
    current_price = market_data.get('close', 0)

    recommendation = {
        "action": signal,
        "confidence_level": "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low",
        "reasoning": f"Model confidence: {confidence:.2%}",
        "risk_level": "Medium",
        "position_size": 0.0,
        "stop_loss": 0.0,
        "take_profit": 0.0
    }

    if signal == "BUY":
        recommendation["reasoning"] = f"Model predicts upward movement with {confidence:.1%} confidence (probability: {probability:.3f})"
        recommendation["position_size"] = min(confidence, 0.95)
        recommendation["stop_loss"] = current_price * 0.98  # 2% SL
        recommendation["take_profit"] = current_price * 1.05  # 5% TP
        recommendation["risk_level"] = "Low" if confidence > 0.8 else "Medium"
    elif signal == "SELL":
        recommendation["reasoning"] = f"Model predicts downward movement with {confidence:.1%} confidence (probability: {probability:.3f})"
        recommendation["position_size"] = -min(confidence, 0.95)  # Short
        recommendation["stop_loss"] = current_price * 1.02  # 2% SL
        recommendation["take_profit"] = current_price * 0.95  # 5% TP
        recommendation["risk_level"] = "Low" if confidence > 0.8 else "Medium"
    else:  # HOLD
        recommendation["reasoning"] = f"No clear signal detected (confidence: {confidence:.1%})"
        recommendation["risk_level"] = "Very Low"

    return recommendation

def check_database_connection() -> bool:
    """Check database connection."""
    try:
        import pymysql
        from dotenv import load_dotenv
        load_dotenv()

        db_config = {
            'host': os.getenv('TRADING_DB_HOST', 'localhost'),
            'port': int(os.getenv('TRADING_DB_PORT', 3306)),
            'user': os.getenv('TRADING_DB_USER'),
            'password': os.getenv('TRADING_DB_PASSWORD'),
            'database': os.getenv('TRADING_DB_NAME', 'newera'),
            'connect_timeout': 2
        }

        conn = pymysql.connect(**db_config)
        conn.close()
        return True
    except:
        return False

if __name__ == "__main__":
    uvicorn.run(
        "quantconnect_api:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info"
    )