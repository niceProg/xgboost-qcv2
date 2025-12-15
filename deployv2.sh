#!/bin/bash

# Deploy tanpa build Docker image
echo "=========================================="
echo "Quick Deploy - No Build"
echo "=========================================="

# Create directories
mkdir -p ./output_train ./logs

# Set permissions
chmod 755 ./output_train ./logs

# Deploy pipeline
echo "Deploying XGBoost Pipeline..."
docker run -d \
  --name xgboost_pipeline \
  --env-file ./.env \
  -v $(pwd)/output_train:/app/output_train \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd):/app/source \
  python:3.10-slim \
  sh -c "
    pip install --quiet \
      pymysql sqlalchemy pandas numpy joblib fastapi uvicorn pydantic \
      pytz python-multipart aiofiles requests mysqlclient &&
    cd /app/source &&
    python run_daily_pipeline.py
  "

# Wait a bit
sleep 5

# Show logs
echo ""
echo "Container status:"
docker ps --filter "name=xgboost_pipeline"

echo ""
echo "Logs (Ctrl+C to stop):"
docker logs -f xgboost_pipeline