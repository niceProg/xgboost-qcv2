#!/usr/bin/env python3
"""
Real-time API Extensions untuk XGBoost Trading Model.
Endpoints untuk real-time prediction dan signal generation.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Output directory
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', './output_train'))

# FastAPI app
app = FastAPI(
    title="XGBoost Real-time Trading API",
    description="Real-time prediction and signal generation API",
    version="1.0.0"
)

# Pydantic models
class PredictionRequest(BaseModel):
    features: Dict[str, float]
    model_name: Optional[str] = None

class PredictionResponse(BaseModel):
    prediction: int  # 0 or 1
    confidence: float
    prediction_probability: float
    model_info: Dict
    timestamp: str

class SignalRequest(BaseModel):
    exchange: str = "binance"
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    threshold: float = 0.5

class SignalResponse(BaseModel):
    signal: str  # "BUY", "SELL", "HOLD"
    confidence: float
    price: Optional[float] = None
    prediction_probability: float
    features_used: List[str]
    timestamp: str
    recommendation: Dict

class MarketDataResponse(BaseModel):
    exchange: str
    symbol: str
    interval: str
    price: float
    volume: float
    change_24h: float
    timestamp: str

def get_latest_model_path() -> Optional[Path]:
    """Get path to latest model."""
    models_dir = OUTPUT_DIR / 'models'
    if not models_dir.exists():
        models_dir = OUTPUT_DIR

    model_files = list(models_dir.glob("xgboost_trading_model_*.joblib"))
    if not model_files:
        return None

    # Sort by modification time
    model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return model_files[0]

def load_latest_model():
    """Load latest trained model."""
    model_path = get_latest_model_path()
    if not model_path:
        raise HTTPException(status_code=404, detail="No model found")

    try:
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Failed to load model")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "XGBoost Real-time Trading API",
        "status": "running",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check if model exists
        model_path = get_latest_model_path()
        model_available = model_path is not None

        return {
            "status": "healthy",
            "model_available": model_available,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction using latest model."""
    try:
        # Load latest model
        model = load_latest_model()

        # Get feature names
        feature_names = getattr(model, 'feature_names_in_', None)
        if not feature_names:
            booster = model.get_booster()
            feature_names = booster.feature_names if booster else None

        if not feature_names:
            raise HTTPException(status_code=500, detail="Model feature names not available")

        # Prepare features
        X = []
        missing_features = []
        for feature in feature_names:
            if feature in request.features:
                X.append(request.features[feature])
            else:
                X.append(0.0)
                missing_features.append(feature)

        X = np.array(X).reshape(1, -1)

        # Make prediction
        prediction = model.predict(X)[0]
        prediction_proba = model.predict_proba(X)[0]
        confidence = float(max(prediction_proba))

        # Log missing features
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")

        return PredictionResponse(
            prediction=int(prediction),
            confidence=confidence,
            prediction_probability=float(prediction_proba[1]),
            model_info={
                "features": len(feature_names),
                "missing_features": missing_features,
                "model_type": "XGBoost"
            },
            timestamp=datetime.now().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/signal", response_model=SignalResponse)
async def generate_signal(request: SignalRequest):
    """Generate trading signal from latest data."""
    try:
        # Get current market data
        current_data = await get_current_market_data(
            request.exchange,
            request.symbol,
            request.interval
        )

        if not current_data:
            raise HTTPException(status_code=404, detail="No current data available")

        # Engineer features
        features = await engineer_realtime_features(current_data)

        if not features:
            raise HTTPException(status_code=500, detail="Feature engineering failed")

        # Make prediction
        pred_request = PredictionRequest(features=features)
        prediction = await predict(pred_request)

        # Determine signal
        signal_probability = prediction.prediction_probability

        if signal_probability > (0.5 + request.threshold/2):
            signal = "BUY"
        elif signal_probability < (0.5 - request.threshold/2):
            signal = "SELL"
        else:
            signal = "HOLD"

        # Generate recommendation
        recommendation = generate_recommendation(
            signal,
            prediction.confidence,
            current_data
        )

        return SignalResponse(
            signal=signal,
            confidence=prediction.confidence,
            price=current_data.get('close'),
            prediction_probability=signal_probability,
            features_used=list(features.keys()),
            timestamp=datetime.now().isoformat(),
            recommendation=recommendation
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signal generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Signal generation failed: {str(e)}")

@app.get("/market/{exchange}/{symbol}/{interval}", response_model=MarketDataResponse)
async def get_market_data(exchange: str, symbol: str, interval: str):
    """Get current market data for symbol."""
    try:
        current_data = await get_current_market_data(exchange, symbol, interval)

        if not current_data:
            raise HTTPException(status_code=404, detail="Market data not found")

        # Calculate 24h change
        close_price = current_data.get('close', 0)
        open_price = current_data.get('open', close_price)
        change_24h = ((close_price - open_price) / open_price) * 100 if open_price > 0 else 0

        return MarketDataResponse(
            exchange=exchange,
            symbol=symbol,
            interval=interval,
            price=close_price,
            volume=current_data.get('volume', 0),
            change_24h=change_24h,
            timestamp=current_data.get('time', datetime.now().isoformat())
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get market data: {str(e)}")

async def get_current_market_data(exchange: str, symbol: str, interval: str) -> Optional[Dict]:
    """Get current market data for signal generation."""
    try:
        # In production, this would query database or exchange API
        # For demo, simulate with realistic data
        import random

        base_price = 42000.0 if symbol == "BTCUSDT" else 3000.0
        variation = random.uniform(-0.02, 0.02)  # ±2% variation

        close = base_price * (1 + variation)
        open_price = close * random.uniform(0.98, 1.02)
        high = max(open_price, close) * random.uniform(1.0, 1.01)
        low = min(open_price, close) * random.uniform(0.99, 1.0)
        volume = random.uniform(1000000, 5000000)

        return {
            "exchange": exchange,
            "symbol": symbol,
            "interval": interval,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "time": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting market data: {e}")
        return None

async def engineer_realtime_features(data: Dict) -> Optional[Dict[str, float]]:
    """Engineer features from current market data."""
    try:
        # Feature engineering - simplified version
        close = data.get('close', 0)
        volume = data.get('volume', 0)
        high = data.get('high', 0)
        low = data.get('low', 0)
        open_price = data.get('open', close)

        features = {
            "price_close": close,
            "price_high": high,
            "price_low": low,
            "volume_usd": volume,
            "price_range": high - low,
            "price_change_pct": ((close - open_price) / open_price) * 100 if open_price > 0 else 0,
            "volume_price_ratio": volume / close if close > 0 else 0,
            "high_low_ratio": high / low if low > 0 else 0,
            "close_position": (close - low) / (high - low) if high != low else 0.5,
            "price_volume_trend": ((close - open_price) / open_price) * volume / 1000000 if open_price > 0 else 0
        }

        return features

    except Exception as e:
        logger.error(f"Error engineering features: {e}")
        return None

def generate_recommendation(signal: str, confidence: float, market_data: Dict) -> Dict:
    """Generate trading recommendation based on signal."""
    current_price = market_data.get('close', 0)

    recommendation = {
        "action": signal,
        "confidence_level": "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low",
        "reasoning": "",
        "risk_level": "Medium",
        "suggested_position_size": 0.0,
        "stop_loss": 0.0,
        "take_profit": 0.0,
        "holding_period": "1-4 hours"
    }

    if signal == "BUY":
        recommendation["reasoning"] = f"Model predicts upward movement with {confidence:.1%} confidence"
        recommendation["suggested_position_size"] = min(1.0, confidence * 1.5)  # Scale position size
        recommendation["stop_loss"] = current_price * 0.985  # 1.5% stop loss
        recommendation["take_profit"] = current_price * 1.03  # 3% take profit
        recommendation["risk_level"] = "Low" if confidence > 0.8 else "Medium"
        recommendation["holding_period"] = "2-6 hours"
    elif signal == "SELL":
        recommendation["reasoning"] = f"Model predicts downward movement with {confidence:.1%} confidence"
        recommendation["suggested_position_size"] = -min(1.0, confidence * 1.5)  # Short position
        recommendation["stop_loss"] = current_price * 1.015  # 1.5% stop loss
        recommendation["take_profit"] = current_price * 0.97  # 3% take profit
        recommendation["risk_level"] = "Low" if confidence > 0.8 else "Medium"
        recommendation["holding_period"] = "2-6 hours"
    else:  # HOLD
        recommendation["reasoning"] = f"No clear signal detected (confidence: {confidence:.1%})"
        recommendation["suggested_position_size"] = 0.0
        recommendation["risk_level"] = "Very Low"
        recommendation["holding_period"] = "Wait for clear signal"

    return recommendation

@app.get("/status")
async def get_realtime_status():
    """Get real-time system status."""
    try:
        # Check if model exists
        model_path = get_latest_model_path()
        model_available = model_path is not None

        # Get latest model update
        latest_update = None
        if model_path:
            latest_update = datetime.fromtimestamp(model_path.stat().st_mtime).isoformat()

        # Count realtime data files
        realtime_files = 0
        try:
            realtime_dir = Path('./realtime_data')
            if realtime_dir.exists():
                realtime_files = len(list(realtime_dir.glob("*.parquet")))
        except:
            pass

        # Get monitor status
        monitor_running = False
        try:
            with open('./state/monitor_status.json', 'r') as f:
                status = json.load(f)
                monitor_running = status.get('running', False)
        except:
            pass

        return {
            "monitor_running": monitor_running,
            "model_available": model_available,
            "latest_model_update": latest_update,
            "pending_data_files": realtime_files,
            "api_status": "running",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return {
            "monitor_running": False,
            "model_available": False,
            "api_status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/models")
async def get_models():
    """Get available models."""
    try:
        models_dir = OUTPUT_DIR / 'models'
        if not models_dir.exists():
            models_dir = OUTPUT_DIR

        model_files = list(models_dir.glob("xgboost_trading_model_*.joblib"))
        models = []

        for model_file in model_files:
            stat = model_file.stat()
            models.append({
                "name": model_file.name,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "path": str(model_file.relative_to(OUTPUT_DIR))
            })

        # Sort by modification time (newest first)
        models.sort(key=lambda x: x['modified'], reverse=True)

        return {
            "total_models": len(models),
            "models": models
        }

    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "realtime_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )