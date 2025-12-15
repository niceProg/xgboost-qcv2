# XGBoost Trading Pipeline - Deployment Guide

## Overview

This guide covers deployment of the XGBoost Trading Pipeline with:
- Daily automated runs using systemd timer
- API server deployment using Docker

## 1. Environment Setup

### 1.1 System Requirements

- Ubuntu 20.04+ or Debian 10+
- Python 3.9+
- MySQL/PostgreSQL database (remote or local)
- Docker & Docker Compose (for API deployment)
- At least 2GB RAM, 2 CPU cores

### 1.2 Clone Repository

```bash
git clone <repository-url>
cd xgboost-qc
```

### 1.3 Install Dependencies

```bash
# System dependencies
sudo apt update
sudo apt install -y python3 python3-pip python3-venv

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt
pip install -r requirements_db.txt
```

### 1.4 Configure Environment

Create `.env` file:

```bash
cp .env.example .env
nano .env
```

```env
# Database Configuration (for training data)
TRADING_DB_HOST=your-mysql-host
TRADING_DB_PORT=3306
TRADING_DB_NAME=trading_data
TRADING_DB_USER=your-username
TRADING_DB_PASSWORD=your-password

# Training Configuration
OUTPUT_DIR=./output_train
ENABLE_DB_STORAGE=true

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

## 2. Daily Pipeline Setup (Systemd)

### 2.1 Install Systemd Service

```bash
# Copy service files
sudo cp xgboost-daily.service /etc/systemd/system/
sudo cp xgboost-daily.timer /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable and start timer
sudo systemctl enable xgboost-daily.timer
sudo systemctl start xgboost-daily.timer
```

### 2.2 Check Timer Status

```bash
# Check timer status
sudo systemctl status xgboost-daily.timer

# List timers
systemctl list-timers --all

# Check last run
journalctl -u xgboost-daily.service
```

### 2.3 Manual Test Run

```bash
# Test daily runner manually
python3 daily_runner.py --config daily_config.json

# Test with dry-run (no execution)
python3 daily_runner.py --dry-run
```

### 2.4 Configure Daily Schedule

Edit the timer file to change schedule:

```bash
sudo nano /etc/systemd/system/xgboost-daily.timer
```

Default: Runs weekdays at 09:30 UTC (16:30 WIB)

```ini
[Timer]
# Run every weekday at 16:30 WIB (09:30 UTC)
OnCalendar=*-*-* 09:30:00 UTC
```

To run multiple times per day:

```ini
[Timer]
# Run every 4 hours
OnCalendar=*-*-* 0/4:00:00
```

After changes:

```bash
sudo systemctl daemon-reload
sudo systemctl restart xgboost-daily.timer
```

## 3. API Deployment (Docker)

### 3.1 Build Docker Image

```bash
# Build image
docker build -t xgboost-trading:latest .

# Or use docker-compose
docker-compose -f docker-compose.api.yml build
```

### 3.2 Run API Server

Option A: Direct Docker

```bash
docker run -d \
  --name xgboost-api \
  -p 8000:8000 \
  -v $(pwd)/output_train:/app/output_train \
  -v $(pwd)/logs:/app/logs \
  --env-file ./.env \
  xgboost-trading:latest
```

Option B: Docker Compose (Recommended)

```bash
# Start API server
docker-compose -f docker-compose.api.yml up -d

# Check status
docker-compose -f docker-compose.api.yml ps

# View logs
docker-compose -f docker-compose.api.yml logs -f
```

### 3.3 Production Deployment with Nginx

```bash
# Start with nginx reverse proxy
docker-compose -f docker-compose.api.yml --profile production up -d
```

Create `nginx.conf` for reverse proxy:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream xgboost_api {
        server xgboost-api:8000;
    }

    server {
        listen 80;
        server_name your-domain.com;

        location / {
            proxy_pass http://xgboost_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Health check endpoint
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
    }
}
```

### 3.4 SSL Configuration (Optional)

```bash
# Add SSL certificates to ./ssl directory
mkdir -p ssl
cp your-cert.pem ssl/
cp your-key.pem ssl/

# Update nginx.conf for SSL
# ... add HTTPS block ...
```

## 4. Monitoring & Maintenance

### 4.1 Check API Health

```bash
# Local check
curl http://localhost:8000/api/v1/health

# Remote check
curl http://your-domain.com/api/v1/health
```

### 4.2 View Logs

```bash
# Systemd logs
sudo journalctl -u xgboost-daily.service -f

# Docker logs
docker logs -f xgboost-api

# Application logs
tail -f logs/xgboost_pipeline.log
```

### 4.3 Backup Data

```bash
# Backup output directory
tar -czf backup_$(date +%Y%m%d).tar.gz output_train/

# Backup to cloud storage (optional)
aws s3 cp backup_$(date +%Y%m%d).tar.gz s3://your-backup-bucket/
```

### 4.4 Update Pipeline

```bash
# Stop services
docker-compose -f docker-compose.api.yml down
sudo systemctl stop xgboost-daily.timer

# Pull updates
git pull origin main

# Rebuild and restart
docker-compose -f docker-compose.api.yml build
docker-compose -f docker-compose.api.yml up -d
sudo systemctl start xgboost-daily.timer
```

## 5. API Usage Examples

### 5.1 Health Check

```bash
curl http://localhost:8000/api/v1/health
```

### 5.2 Model Status

```bash
curl http://localhost:8000/api/v1/model/status
```

### 5.3 Make Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "price_close_return_1": 0.01,
      "price_rolling_vol_5": 0.02,
      "funding_zscore": 0.5
    },
    "threshold": 0.5
  }'
```

### 5.4 Get Performance Metrics

```bash
curl http://localhost:8000/api/v1/performance/latest
```

## 6. Troubleshooting

### 6.1 Common Issues

1. **Pipeline fails to load data**
   - Check database connection in `.env`
   - Verify database credentials
   - Check network connectivity

2. **API returns 404 for model**
   - Run training pipeline first
   - Check `output_train/latest_model.joblib` exists
   - Verify model file permissions

3. **Docker container fails to start**
   - Check Docker logs: `docker logs xgboost-api`
   - Verify volume mounts
   - Check environment variables

### 6.2 Debug Mode

Enable debug logging:

```bash
# Set environment variable
export LOG_LEVEL=DEBUG

# Or in .env
LOG_LEVEL=DEBUG
```

## 7. Security Considerations

1. **API Security**
   - Add authentication middleware
   - Use HTTPS in production
   - Implement rate limiting
   - Validate all inputs

2. **Database Security**
   - Use read-only user for API
   - Encrypt database connections
   - Regular password rotation

3. **System Security**
   - Create dedicated user for service
   - Use firewall rules
   - Regular security updates
   - Monitor access logs

## 8. Performance Optimization

1. **Database Optimization**
   - Add indexes on time columns
   - Partition large tables by date
   - Use connection pooling

2. **API Optimization**
   - Add response caching
   - Use async endpoints
   - Implement pagination
   - Add CDN for static files

3. **Resource Monitoring**
   - Monitor CPU/Memory usage
   - Set up alerts for failures
   - Track API response times
   - Monitor disk space usage

## 9. Scale-Out Options

For high-availability deployment:

1. **Load Balancer Setup**
   ```yaml
   # docker-compose.scale.yml
   version: '3.8'
   services:
     xgboost-api:
       deploy:
         replicas: 3
       # ... other config
     nginx:
       # Load balancer config
   ```

2. **Database Replication**
   - Set up read replicas
   - Implement failover
   - Use connection strings

3. **Message Queue**
   - Add Redis/RabbitMQ for async tasks
   - Queue heavy operations
   - Background job processing