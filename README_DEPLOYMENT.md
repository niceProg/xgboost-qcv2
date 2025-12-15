# XGBoost Trading Model - Deployment Guide

This guide explains how to deploy the XGBoost trading model for production use.

## Deployment Architecture

### Phase 1: Initial Setup
- Run initial training with historical data from 2024
- Store baseline model and all artifacts in database
- Setup database tables and API endpoints

### Phase 2: Daily Operations
- Run daily pipeline during trading hours (7:00 - 16:00)
- Update model with new data
- Track performance and maintain logs

## Quick Start

### 1. Initial Setup (One-time)

```bash
# Clone repository
git clone <repository_url>
cd xgboost-qc

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_db.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your database credentials

# Run initial setup (trains from 2024)
python initial_setup.py

# Or with Docker
docker-compose up --build
# In separate terminal:
docker-compose exec xgboost_pipeline python initial_setup.py
```

### 2. Start Daily Operations

```bash
# Option 1: Run scheduler (recommended)
python scheduler.py

# Option 2: Use Docker
docker-compose up -d  # Starts all services

# Option 3: Manual daily run
python run_daily_pipeline.py --mode daily
```

### 3. Access API

The API will be available at: `http://localhost:8000/api/v1/`

## Docker Deployment

### Prerequisites
- Docker and Docker Compose installed
- At least 4GB RAM available
- Ports 3307, 3308, and 8000 available

### Docker Compose Services

1. **trading_db** (Port 3307)
   - MySQL database for market data
   - Persists trading history

2. **results_db** (Port 3308)
   - MySQL database for training results
   - Stores models, features, evaluations

3. **xgboost_pipeline**
   - Runs the training pipeline
   - Daily updates during trading hours

4. **api_server** (Port 8000)
   - REST API for model access
   - Separate for production scaling

### Running with Docker

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Run initial setup
docker-compose run --rm xgboost_pipeline python initial_setup.py

# Access container
docker-compose exec xgboost_pipeline bash
```

### Docker Environment Variables

```bash
# Database Configuration
DB_HOST=trading_db
DB_PORT=3306
DB_NAME=trading_data
DB_USER=trading_user
DB_PASSWORD=your_password

RESULTS_DB_HOST=results_db
RESULTS_DB_PORT=3306
RESULTS_DB_NAME=xgboost_training
RESULTS_DB_USER=xgboost_user
RESULTS_DB_PASSWORD=your_password

# Pipeline Configuration
ENABLE_DB_STORAGE=true
PIPELINE_MODE=daily
EXCHANGE=binance
PAIR=BTCUSDT
INTERVAL=1h
TRADING_HOURS=7:00-16:00
TIMEZONE=UTC

# API Configuration
API_PORT=8000
ENV=production
```

## Scheduler Configuration

### Trading Hours
- Default: 7:00 AM to 4:00 PM (UTC)
- Configurable via `TRADING_HOURS` environment variable
- Format: "HH:MM-HH:MM"

### Scheduler Modes

```bash
# Run continuously (auto-starts at 7:00)
python scheduler.py

# Run once immediately
python scheduler.py --run-once

# Check if within trading hours
python scheduler.py --check-hours

# Show scheduler status
python scheduler.py --status
```

### Production Scheduler

For production, use a process manager like systemd:

```bash
# Create service file
sudo nano /etc/systemd/system/xgboost-scheduler.service

# Content:
[Unit]
Description=XGBoost Trading Pipeline Scheduler
After=network.target mysql.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/xgboost-qc
Environment=PATH=/opt/xgboost-qc/venv/bin
ExecStart=/opt/xgboost-qc/venv/bin/python scheduler.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable xgboost-scheduler
sudo systemctl start xgboost-scheduler

# Check status
sudo systemctl status xgboost-scheduler

# View logs
sudo journalctl -u xgboost-scheduler -f
```

## API Usage

### Base URL
```
http://localhost:8000/api/v1/
```

### Key Endpoints

```bash
# Get latest model info
curl http://localhost:8000/api/v1/models/latest

# Get training sessions
curl http://localhost:8000/api/v1/sessions

# Make prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "price_close_return_1": 0.02,
      "funding_close": 0.0001,
      "basis_close_basis": 0.01
    },
    "threshold": 0.5
  }'

# Get performance metrics
curl http://localhost:8000/api/v1/performance/<session_id>
```

## Monitoring and Maintenance

### Logs

```bash
# Pipeline logs
tail -f /var/log/xgboost/pipeline.log

# Scheduler logs
tail -f scheduler.log

# Docker logs
docker-compose logs -f xgboost_pipeline
```

### Database Maintenance

```bash
# Check database size
mysql -u root -p -e "SELECT table_schema AS 'Database', ROUND(SUM(data_length + index_length) / 1024 / 1024, 2) AS 'Size (MB)' FROM information_schema.tables WHERE table_schema='xgboost_training' GROUP BY table_schema;"

# Cleanup old data (from Python)
python -c "
from database_storage import DatabaseStorage
storage = DatabaseStorage()
storage.cleanup_old_data(days=30)
"

# Backup database
mysqldump -h localhost -u xgboost_user -p xgboost_training > backup_$(date +%Y%m%d).sql
```

### Performance Monitoring

```bash
# Check recent sessions
python -c "
from database_storage import DatabaseStorage
storage = DatabaseStorage()
sessions = storage.get_session_history(limit=5)
for s in sessions:
    print(f'{s[\"session_id\"]}: {s[\"status\"]} - AUC: {s[\"test_auc\"]:.4f}')
"
```

## QuantConnect Integration

### API Client Example

```python
import requests

class XGBoostModelClient:
    def __init__(self, api_url="http://your-server:8000"):
        self.api_url = api_url
        self.headers = {"Content-Type": "application/json"}

    def predict(self, features):
        response = requests.post(
            f"{self.api_url}/api/v1/predict",
            json={"features": features},
            headers=self.headers
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Prediction failed: {response.text}")

    def get_latest_model(self):
        response = requests.get(f"{self.api_url}/api/v1/models/latest")
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to get model: {response.text}")
```

### QuantConnect Algorithm

```python
from QuantConnect import *
from QuantConnect.Algorithm import *
from QuantConnect.Data import *
import requests

class XGBoostTradingAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2023, 1, 1)
        self.SetCash(100000)
        self.AddCrypto("BTCUSDT", Resolution.Hour)

        # Initialize model client
        self.model_client = XGBoostModelClient("http://your-api-server:8000")

        # Check if model is available
        try:
            model_info = self.model_client.get_latest_model()
            self.Debug(f"Model loaded: {model_info['model']['session_id']}")
            self.required_features = model_info['model']['feature_names']
        except Exception as e:
            self.Error(f"Failed to load model: {e}")
            self.Quit()

    def OnData(self, data):
        if not data.ContainsKey("BTCUSDT"):
            return

        # Calculate features (same as pipeline)
        features = self.calculate_features(data)

        # Get prediction
        try:
            result = self.model_client.predict(features)
            if result["prediction"] == 1:
                self.SetHoldings("BTCUSDT", 0.95)
            else:
                self.Liquidate("BTCUSDT")
        except Exception as e:
            self.Error(f"Prediction failed: {e}")

    def calculate_features(self, data):
        # Implement same feature calculations as pipeline
        # This is simplified - you need to match exactly
        return {
            "price_close_return_1": 0.02,
            "funding_close": 0.0001,
            "basis_close_basis": 0.01,
            # ... all required features
        }
```

## Troubleshooting

### Common Issues

1. **Database Connection Error**
   ```bash
   # Check if database is running
   docker-compose ps

   # Check logs
   docker-compose logs trading_db
   ```

2. **Pipeline Timeout**
   - Increase timeout in run_daily_pipeline.py
   - Check if market data is available

3. **Memory Issues**
   - Increase Docker memory limit
   - Reduce data size in initial setup

4. **API Not Responding**
   - Check if API server is running
   - Verify port 8000 is accessible

### Debug Mode

```bash
# Enable debug logging
export PYTHONPATH=/path/to/xgboost-qc
python load_database.py --verbose --mode daily
```

## Security Considerations

1. **Database Security**
   - Use strong passwords
   - Limit database access to application
   - Enable SSL for production

2. **API Security**
   - Add authentication middleware
   - Use HTTPS in production
   - Implement rate limiting

3. **Network Security**
   - Use firewall to restrict access
   - Run services in isolated network
   - Monitor for unauthorized access

## Scaling

### Horizontal Scaling

1. **API Scaling**
   ```yaml
   # docker-compose.yml
   api_server:
     deploy:
       replicas: 3
   ```

2. **Load Balancing**
   - Use Nginx or HAProxy
   - Implement health checks

### Vertical Scaling

1. **Increase Resources**
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 4G
   ```

2. **Database Optimization**
   - Add indexes
   - Partition large tables
   - Use read replicas for API