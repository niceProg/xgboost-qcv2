# XGBoost Trading Model - Deployment Guide

This guide explains how to deploy the XGBoost trading model for production use with WIB timezone.

## Deployment Architecture

### Phase 1: Initial Setup (Manual)
- Run initial training with historical data from 2024
- All data and models stored in output_train directory
- Setup API server for model access

### Phase 2: Daily Operations (Manual/Scheduler)
- Run daily pipeline during trading hours (7:00 - 16:00 WIB)
- Update model with new data
- API reads directly from output_train directory

## Quick Start

### 1. Initial Setup (One-time)

```bash
# Clone repository
git clone <repository_url>
cd xgboost-qc

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_db.txt

# Configure environment
cp .env.example .env
# Edit .env with your database credentials:
```

Edit `.env`:
```bash
# Trading Database (market data)
TRADING_DB_HOST=103.150.81.86
TRADING_DB_PORT=3306
TRADING_DB_USER=newera
TRADING_DB_PASSWORD=WCXscc8twHi3kxDW
TRADING_DB_NAME=newera

# Results Database (training results)
DB_HOST=103.150.81.86
DB_PORT=3306
DB_USER=xgboostqc
DB_PASSWORD=6SPxBDwXH6WyxpfT
DB_NAME=xgboostqc

# Pipeline Configuration
ENABLE_DB_STORAGE=true
EXCHANGE=binance
PAIR=BTCUSDT
INTERVAL=1h
# Trading hours in UTC (7:00-16:00 WIB = 00:00-09:00 UTC)
TRADING_HOURS=00:00-09:00
TIMEZONE=WIB  # Waktu Indonesia Barat (UTC+7)
OUTPUT_DIR=./output_train

# Model Evaluation
LEVERAGE=10
MARGIN_FRACTION=0.2
INITIAL_CASH=1000
THRESHOLD=0.5
FEE_RATE=0.0004
```

### 2. Run Initial Training (Step by Step)

```bash
# Step 1: Load data from 2024
python load_database.py --exchange binance --pair BTCUSDT --interval 1h --mode initial --timezone WIB

# Step 2: Merge tables
python merge_7_tables.py --exchange binance --pair BTCUSDT --interval 1h --mode initial --timezone WIB

# Step 3: Feature engineering
python feature_engineering.py --exchange binance --pair BTCUSDT --interval 1h --mode initial --timezone WIB

# Step 4: Label building
python label_builder.py --exchange binance --pair BTCUSDT --interval 1h --mode initial --timezone WIB

# Step 5: Train model
python xgboost_trainer.py --exchange binance --pair BTCUSDT --interval 1h --mode initial --timezone WIB

# Step 6: Evaluate model
python model_evaluation_with_leverage.py --exchange binance --pair BTCUSDT --interval 1h --mode initial --timezone WIB
```

### 3. Start Daily Operations

```bash
# Option 1: Manual daily run
python run_daily_pipeline.py --mode daily --trading-hours 00:00-09:00 --timezone WIB

# Option 2: Use scheduler (auto-run at 7 AM WIB)
python scheduler.py

# Option 3: Use helper script
./simple_run.sh

# Option 4: Use Docker for daily pipeline (secure)
docker run --rm \
  --env-file ./.env \
  -v /opt/xgboost-qc/output_train:/app/output_train \
  xgboost-qc python run_daily_pipeline.py --mode daily

# Option 5: Use Docker Compose (recommended)
docker-compose --profile pipeline up -d
```

### 4. Start API Server

```bash
# Option 1: Start API server directly
python api_server.py

# Option 2: Start API server with Docker (secure)
docker run -d \
  --name xgboost-api \
  --env-file ./.env \
  -p 8000:8000 \
  -v /opt/xgboost-qc/output_train:/app/output_train:ro \
  xgboost-qc python api_server.py

# Option 3: Use Docker Compose for API (recommended)
docker-compose --profile api up -d

# Option 4: Start scheduler with Docker
docker-compose --profile scheduler up -d
```

## Docker Deployment

### Prerequisites
- Docker installed
- Docker Compose installed
- At least 2GB RAM available
- Port 8000 available
- Database server accessible from Docker host

### Docker Image Build

```bash
# Build the Docker image
docker build -t xgboost-qc .

# Or pull from registry (if available)
docker pull your-registry/xgboost-qc:latest
```

### Docker Compose for API Server

Create `docker-compose-api.yml`:

```yaml
version: '3.8'

services:
  api_server:
    image: xgboost-qc:latest
    container_name: xgboost_api
    command: ["python", "api_server.py"]
    ports:
      - "8000:8000"
    environment:
      # Database Configuration
      DB_HOST: 103.150.81.86
      DB_PORT: 3306
      DB_USER: xgboostqc
      DB_PASSWORD: 6SPxBDwXH6WyxpfT
      DB_NAME: xgboostqc

      # API Configuration
      API_PORT: 8000
      ENV: production
      OUTPUT_DIR: /app/output_train
    volumes:
      - ./output_train:/app/output_train:ro  # Read-only for API
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### Running with Docker

```bash
# Build and run API server
docker-compose -f docker-compose-api.yml up -d

# View logs
docker-compose -f docker-compose-api.yml logs -f

# Stop API server
docker-compose -f docker-compose-api.yml down

# Scale API server (load balancing)
docker-compose -f docker-compose-api.yml up -d --scale api_server=3
```

### Docker Environment Variables

Using `.env` file (recommended and more secure):

```bash
# All environment variables are loaded from .env file
# No need to pass them manually

# Build and run with environment file
docker-compose --profile api up -d

# Or with docker run (using env file)
docker run --rm \
  --env-file ./.env \
  -v /opt/xgboost-qc/output_train:/app/output_train \
  xgboost-qc python run_daily_pipeline.py
```

Alternative: Direct environment variables (less secure):
```bash
# Only use for testing or CI/CD environments
docker run --rm \
  -e DB_HOST=103.150.81.86 \
  -e DB_USER=xgboostqc \
  -e DB_PASSWORD=6SPxBDwXH6WyxpfT \
  -e DB_NAME=xgboostqc \
  -v /opt/xgboost-qc/output_train:/app/output_train \
  xgboost-qc python api_server.py
```

### Server Directory Structure

```
/opt/xgboost-qc/
├── .env                    # Environment configuration (keep secure!)
├── output_train/           # Training outputs (server volume)
├── logs/                  # Application logs (server volume)
├── docker-compose.yml       # Docker Compose configuration
├── Dockerfile              # Docker image definition
└── src/                   # Source code
```

### Secure Volume Mounting

All important directories use server volumes, not Docker volumes:

```yaml
# In docker-compose.yml
volumes:
  - /opt/xgboost-qc/output_train:/app/output_train:ro  # Persistent data
  - /opt/xgboost-qc/.env:/app/.env:ro                # Config (read-only)
  - /opt/xgboost-qc/logs:/app/logs                      # Logs
```

## Manual Execution

### Environment Variables

```bash
# For single command
EXCHANGE=binance PAIR=BTCUSDT INTERVAL=1h MODE=initial TIMEZONE=WIB python load_database.py

# Or export first
export EXCHANGE=binance
export PAIR=BTCUSDT
export INTERVAL=1h
export TIMEZONE=WIB
export MODE=initial
python load_database.py
```

### Helper Script

Use `simple_run.sh` to run all steps automatically:
```bash
# Run with defaults
./simple_run.sh

# Or with custom parameters
EXCHANGE=binance PAIR=ETHUSDT INTERVAL=1h ./simple_run.sh
```

## API Server

### Starting the API

```bash
# With default port (8000)
python api_server.py

# With custom port
API_PORT=8080 python api_server.py

# Development mode with auto-reload
ENV=development python api_server.py
```

### API Endpoints

```bash
# Get API info
curl http://localhost:8000/api/v1/

# Get latest model
curl http://localhost:8000/api/v1/models/latest

# Get model history
curl http://localhost:8000/api/v1/models/history

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

# List all files
curl http://localhost:8000/api/v1/files

# Download file
curl http://localhost:8000/api/v1/download/latest_model.joblib -o model.joblib

# Health check
curl http://localhost:8000/api/v1/health
```

### API Response Format

**Latest Model** (`/api/v1/models/latest`):
```json
{
  "model_name": "latest_model.joblib",
  "session_id": "latest",
  "created_at": "2024-12-15T14:30:00",
  "file_path": "./output_train/latest_model.joblib",
  "feature_count": 45,
  "metrics": {
    "test_auc": 0.7542,
    "sharpe_ratio": 1.23
  }
}
```

**Model History** (`/api/v1/models/history`):
```json
[
  {
    "model_name": "xgboost_trading_model_20241215_143000.joblib",
    "session_id": "20241215_143000",
    "created_at": "2024-12-15T14:30:00",
    "file_path": "./output_train/xgboost_trading_model_20241215_143000.joblib"
  }
]
```

## Scheduler Configuration

### Trading Hours
- Default: 7:00 AM to 4:00 PM WIB (UTC+7)
- In UTC this is: 12:00 AM to 9:00 AM (00:00-09:00 UTC)
- Configurable via `TRADING_HOURS` environment variable (always in UTC)
- Format: "HH:MM-HH:MM" (24-hour format in UTC)

### Scheduler Modes

```bash
# Run continuously (auto-starts at 7 AM WIB)
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

## Output Directory Structure

```
output_train/
├── merged_7_tables.parquet              # Merged trading data
├── features_engineered.parquet          # Engineered features
├── labeled_data.parquet                 # Data with labels
├── X_train_features.parquet             # Training features
├── y_train_labels.parquet               # Training labels
├── latest_model.joblib                  # Latest model (always this name)
├── xgboost_trading_model_*.joblib       # Timestamped models
├── performance_metrics_*.json           # Performance metrics
├── performance_report_*.json            # Detailed reports
├── feature_importance_*.csv             # Feature importance
├── rekening_koran.csv                   # Equity statement
├── trades.csv                           # Trade log
├── trade_events.csv                     # Trade events
└── performance_analysis.png             # Performance plots
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
            self.Debug(f"Model loaded: {model_info['model_name']}")
            self.required_features = model_info['feature_count']
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
   # Check database connection
   python -c "
   import pymysql
   conn = pymysql.connect(
       host='103.150.81.86',
       port=3306,
       user='newera',
       password='WCXscc8twHi3kxDW',
       database='newera'
   )
   print('Trading DB connection successful!')

   conn2 = pymysql.connect(
       host='103.150.81.86',
       port=3306,
       user='xgboostqc',
       password='6SPxBDwXH6WyxpfT',
       database='xgboostqc'
   )
   print('Results DB connection successful!')
   "
   ```

2. **Timezone Issues**
   ```bash
   # Check WIB timezone
   python -c "
   import pytz
   from datetime import datetime
   tz = pytz.timezone('Asia/Jakarta')
   now = datetime.now(tz)
   print(f'Current WIB time: {now}')
   print(f'Trading hours: 7:00-16:00 WIB')
   "
   ```

3. **Model Loading Error**
   ```bash
   # Check model files
   ls -la output_train/*.joblib
   ls -la output_train/latest_model.joblib

   # Verify model
   python -c "
   import joblib
   model = joblib.load('output_train/latest_model.joblib')
   print(f'Model loaded: {type(model)}')
   "
   ```

4. **API Not Responding**
   ```bash
   # Check if API server is running
   ps aux | grep api_server.py

   # Check port
   netstat -tlnp | grep 8000
   ```

## Production Deployment

### Option 1: Docker Deployment (Recommended)

```bash
# Build Docker image
docker build -t xgboost-qc .

# Save to registry (optional)
docker tag xgboost-qc:latest your-registry/xgboost-qc:latest
docker push your-registry/xgboost-qc:latest
```

#### Production Docker Compose

Create `docker-compose-prod.yml`:

```yaml
version: '3.8'

services:
  api_server:
    image: xgboost-qc:latest
    container_name: xgboost_api_prod
    command: ["python", "api_server.py"]
    ports:
      - "8000:8000"
    environment:
      # Database Configuration
      DB_HOST: 103.150.81.86
      DB_PORT: 3306
      DB_USER: xgboostqc
      DB_PASSWORD: 6SPxBDwXH6WyxpfT
      DB_NAME: xgboostqc

      # API Configuration
      API_PORT: 8000
      ENV: production
      OUTPUT_DIR: /app/output_train

      # Logging
      PYTHONUNBUFFERED: 1
    volumes:
      - ./output_train:/app/output_train:ro
      - ./logs:/app/logs
    restart: unless-stopped
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '1'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - xgboost_network
```

#### Deploy to Production Server

```bash
# On production server
git clone <repository>
cd xgboost-qc

# Copy .env file (do not commit to git)
scp .env.production .env

# Build and run
docker-compose -f docker-compose-prod.yml up -d

# Scale if needed
docker-compose -f docker-compose-prod.yml up -d --scale api_server=3

# Update without downtime
docker-compose -f docker-compose-prod.yml pull
docker-compose -f docker-compose-prod.yml up -d --no-deps api_server
```

### Option 2: Manual Deployment

#### Using Supervisor

```bash
# Install supervisor
sudo apt-get install supervisor

# Create config
sudo nano /etc/supervisor/conf.d/xgboost.conf

# Content:
[program:xgboost-api]
command=/opt/xgboost-qc/venv/bin/python api_server.py
directory=/opt/xgboost-qc
user=ubuntu
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/xgboost-api.log
environment=PATH="/opt/xgboost-qc/venv/bin",DB_HOST="103.150.81.86"

[program:xgboost-scheduler]
command=/opt/xgboost-qc/venv/bin/python scheduler.py
directory=/opt/xgboost-qc
user=ubuntu
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/xgboost-scheduler.log
environment=PATH="/opt/xgboost-qc/venv/bin",TIMEZONE="WIB"

# Reload supervisor
sudo supervisorctl reread
sudo supervisorctl update

# Start services
sudo supervisorctl start xgboost-api
sudo supervisorctl start xgboost-scheduler
```

#### Using Nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/xgboost-api
server {
    listen 80;
    server_name your-domain.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL Configuration
    ssl_certificate /etc/ssl/certs/your-domain.crt;
    ssl_certificate_key /etc/ssl/private/your-domain.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Security Headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";

    # API Routes
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # File Downloads (limit size)
    location /api/v1/download/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_max_body_size 100M;
        proxy_set_header Host $host;
    }

    # Health Check
    location /api/v1/health {
        proxy_pass http://127.0.0.1:8000;
        access_log off;
    }

    # Default
    location / {
        return 404;
    }
}
```

## Security Considerations

1. **Database Security**
   - Use strong passwords
   - Enable SSL for production
   - Limit database access to application IP

2. **API Security**
   - Add API key authentication
   - Use HTTPS in production
   - Implement rate limiting

3. **Network Security**
   - Configure firewall
   - Use VPN for remote access
   - Monitor logs regularly

## Maintenance

### Daily Checks
- Check scheduler logs
- Verify API health
- Monitor model performance
- Check database disk space

### Weekly Tasks
- Backup database
- Archive old models
- Review trading performance
- Update system packages

### Monthly Tasks
- Rotate logs
- Update dependencies
- Performance optimization
- Security audit