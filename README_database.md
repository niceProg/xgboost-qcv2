# XGBoost Trading Model - Database Storage Integration

This document explains how to set up and use the database storage system for the XGBoost trading model pipeline.

## Overview

The database storage system allows you to:
- Store all training artifacts (data, features, models, results) in a central database
- Access trained models via REST API
- Easily integrate with QuantConnect
- Track training history and performance metrics
- Organize outputs by training sessions

## Architecture

```
┌─────────────────────┐     ┌──────────────────────┐     ┌────────────────────┐
│  Trading Pipeline   │ --> │    Database          │ --> │   REST API         │
│                     │     │                      │     │                    │
│ • load_database.py  │     │ • Training Sessions  │     │ • /api/v1/models  │
│ • merge_7_tables.py │     │ • Models             │     │ • /api/v1/predict │
│ • feature_engineering│     │ • Features           │     │ • /api/v1/perf    │
│ • label_builder.py  │     │ • Evaluations        │     │                    │
│ • xgboost_trainer.py│     │                      │     └────────────────────┘
│ • model_evaluation  │     └──────────────────────┘
└─────────────────────┘
```

## Quick Start

### 1. Setup Database

First, create the database and tables:

```bash
# Copy environment configuration
cp .env.example .env
# Edit .env with your database credentials

# Install additional requirements
pip install -r requirements_db.txt

# Setup database
python setup_database.py
```

### 2. Run Training Pipeline

Run the training pipeline with database storage enabled:

```bash
# Enable database storage (set in .env)
ENABLE_DB_STORAGE=true

# Run the full pipeline
python load_database.py --exchange binance --symbol BTCUSDT --interval 1h
python merge_7_tables.py --exchange binance --symbol BTCUSDT --interval 1h
python feature_engineering.py --exchange binance --symbol BTCUSDT --interval 1h
python label_builder.py --exchange binance --symbol BTCUSDT --interval 1h
python xgboost_trainer.py --exchange binance --symbol BTCUSDT --interval 1h
python model_evaluation_with_leverage.py --exchange binance --symbol BTCUSDT --interval 1h
```

### 3. Start API Server

```bash
python api_examples.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Get Training Sessions
```http
GET /api/v1/sessions
```

### Get Latest Model Info
```http
GET /api/v1/models/latest
```

### Make Prediction
```http
POST /api/v1/predict
Content-Type: application/json

{
  "features": {
    "price_close_return_1": 0.02,
    "funding_close": 0.0001,
    "basis_close_basis": 0.01
  },
  "threshold": 0.5
}
```

### Get Performance Metrics
```http
GET /api/v1/performance/{session_id}
```

### Export Session Data
```http
GET /api/v1/export/{session_id}
```

## Database Schema

### Training Sessions (`xgboost_training_sessions`)
- Stores metadata for each training run
- Links to all related artifacts

### Models (`xgboost_models`)
- Serialized XGBoost models
- Feature names and hyperparameters
- Training and validation scores

### Features (`xgboost_features`)
- Information about stored feature files
- File paths, row counts, metadata

### Evaluations (`xgboost_evaluations`)
- Backtest and evaluation results
- Performance metrics
- Trading statistics

## QuantConnect Integration

### 1. API Client

Create a Python client to connect to the API:

```python
import requests
import pandas as pd

class XGBoostModelClient:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url

    def predict(self, features):
        response = requests.post(
            f"{self.api_url}/api/v1/predict",
            json={"features": features}
        )
        return response.json()

    def get_latest_model_info(self):
        response = requests.get(
            f"{self.api_url}/api/v1/models/latest"
        )
        return response.json()

# Use in QuantConnect
model_client = XGBoostModelClient("http://your-api-server:8000")
```

### 2. QuantConnect Algorithm

```python
from QuantConnect import *
from QuantConnect.Algorithm import *
from QuantConnect.Data import *

class XGBoostTradingAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetCash(100000)

        # Add crypto
        self.AddCrypto("BTCUSDT", Resolution.Hour)

        # Initialize model client
        self.model_client = XGBoostModelClient()

        # Get model info
        model_info = self.model_client.get_latest_model_info()
        self.required_features = model_info["model"]["feature_names"]

    def OnData(self, data):
        if not data.ContainsKey("BTCUSDT"):
            return

        # Calculate features (same as in pipeline)
        features = self.calculate_features(data)

        # Get prediction
        try:
            result = self.model_client.predict(features)
            if result["prediction"] == 1:
                self.SetHoldings("BTCUSDT", 1.0)
            else:
                self.Liquidate("BTCUSDT")
        except Exception as e:
            self.Error(f"Prediction failed: {e}")
```

## Configuration

### Environment Variables

```bash
# Database for storing results
DB_HOST=localhost
DB_PORT=3306
DB_NAME=xgboost_training
DB_USER=your_username
DB_PASSWORD=your_password

# Enable/disable database storage
ENABLE_DB_STORAGE=true

# API server
API_PORT=8000
```

### Custom Database Configuration

You can use a custom database configuration by modifying the `DatabaseStorage` initialization:

```python
from database_storage import DatabaseStorage

# Custom config
custom_db_config = {
    'host': 'your-db-server.com',
    'port': 5432,
    'database': 'your_db',
    'user': 'your_user',
    'password': 'your_password'
}

storage = DatabaseStorage(db_config=custom_db_config)
```

## Backup and Recovery

### Backup Database

```bash
# Full backup
mysqldump -h localhost -u user -p xgboost_training > backup_$(date +%Y%m%d).sql

# Just models
mysqldump -h localhost -u user -p xgboost_training xgboost_models > models_backup_$(date +%Y%m%d).sql
```

### Recovery

```bash
# Restore full database
mysql -h localhost -u user -p xgboost_training < backup_20231215.sql
```

## Monitoring

### Check Training Sessions

```python
from database_storage import DatabaseStorage

storage = DatabaseStorage()
sessions = storage.get_session_history(limit=10)

for session in sessions:
    print(f"Session: {session['session_id']}")
    print(f"Status: {session['status']}")
    print(f"AUC: {session['test_auc']}")
    print(f"Sharpe: {session['sharpe_ratio']}")
```

### Cleanup Old Data

```python
# Clean up data older than 30 days
storage.cleanup_old_data(days=30)
```

## Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Check DB_HOST, DB_PORT, DB_USER, DB_PASSWORD
   - Ensure database exists
   - Check firewall settings

2. **Model Loading Error**
   - Ensure model was saved properly
   - Check model compatibility
   - Verify feature names match

3. **API Connection Error**
   - Check API server is running
   - Verify CORS settings
   - Check network connectivity

### Logs

Check the logs for detailed error information:
- Pipeline logs: Standard output
- API logs: Console when running API server
- Database logs: MySQL error log

## Best Practices

1. **Version Control**: Track model versions with sessions
2. **Backups**: Regularly backup database
3. **Monitoring**: Set up alerts for failed training sessions
4. **Security**: Use environment variables for credentials
5. **Documentation**: Document model versions and parameters

## Future Enhancements

1. **Model Registry**: Advanced versioning and tagging
2. **A/B Testing**: Deploy multiple models
3. **Auto-scaling**: Cloud deployment options
4. **Monitoring Dashboard**: Real-time metrics
5. **Model Drift Detection**: Automatic retraining triggers