#!/bin/bash

# XGBoost Real-time Trading System - Production Deployment Script
# One-click deployment for the complete real-time training and API system

set -e  # Exit on any error

echo "🚀 XGBoost Real-time Trading System - Production Deployment"
echo "=========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

print_status "Working directory: $SCRIPT_DIR"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p ../output_train/models
mkdir -p ../state
mkdir -p ../logs
mkdir -p ../config

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    print_status "Creating .env file from template..."
    cp .env.example .env
    print_warning "Please edit .env file with your database credentials and API settings"
    print_warning "Then run this script again"
    exit 1
fi

# Source environment variables
source .env

# Validate required environment variables
print_status "Validating environment variables..."
required_vars=("TRADING_DB_HOST" "TRADING_DB_USER" "TRADING_DB_PASSWORD" "TRADING_DB_NAME")
missing_vars=()

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -ne 0 ]; then
    print_error "Missing required environment variables: ${missing_vars[*]}"
    print_error "Please set these in your .env file"
    exit 1
fi

# Test database connection
print_status "Testing database connection..."
python3 - <<EOF
import pymysql
import os

try:
    conn = pymysql.connect(
        host=os.getenv('TRADING_DB_HOST'),
        port=int(os.getenv('TRADING_DB_PORT', 3306)),
        user=os.getenv('TRADING_DB_USER'),
        password=os.getenv('TRADING_DB_PASSWORD'),
        database=os.getenv('TRADING_DB_NAME'),
        connect_timeout=5
    )
    print("✅ Database connection successful")
    conn.close()
except Exception as e:
    print(f"❌ Database connection failed: {e}")
    exit(1)
EOF

if [ $? -ne 0 ]; then
    print_error "Database connection test failed. Please check your credentials."
    exit 1
fi

# Setup database for smart monitoring
print_status "Setting up database for smart monitoring..."
python3 setup_database.py

if [ $? -ne 0 ]; then
    print_error "Database setup failed. Please check the error messages."
    exit 1
fi

# Create Docker Compose file
print_status "Creating Docker Compose configuration..."
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  # FastAPI Server for QuantConnect Integration
  quantconnect-api:
    build:
      context: .
      dockerfile: Dockerfile.api
    container_name: xgboost-quantconnect-api
    ports:
      - "${API_PORT:-8000}:8000"
    environment:
      - TRADING_DB_HOST=${TRADING_DB_HOST}
      - TRADING_DB_PORT=${TRADING_DB_PORT}
      - TRADING_DB_USER=${TRADING_DB_USER}
      - TRADING_DB_PASSWORD=${TRADING_DB_PASSWORD}
      - TRADING_DB_NAME=${TRADING_DB_NAME}
      - API_HOST=${API_HOST:-0.0.0.0}
      - API_PORT=${API_PORT:-8000}
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}
      - QUANTCONNECT_CORS_ORIGIN=${QUANTCONNECT_CORS_ORIGIN:-https://www.quantconnect.com}
    volumes:
      - ../output_train:/app/output_train
      - ../state:/app/state
      - ../logs:/app/logs
    restart: unless-stopped
    networks:
      - xgboost-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Real-time Monitor for New Data
  realtime-monitor:
    build:
      context: .
      dockerfile: Dockerfile.monitor
    container_name: xgboost-realtime-monitor
    environment:
      - TRADING_DB_HOST=${TRADING_DB_HOST}
      - TRADING_DB_PORT=${TRADING_DB_PORT}
      - TRADING_DB_USER=${TRADING_DB_USER}
      - TRADING_DB_PASSWORD=${TRADING_DB_PASSWORD}
      - TRADING_DB_NAME=${TRADING_DB_NAME}
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}
    volumes:
      - ../state:/app/state
      - ../logs:/app/logs
    restart: unless-stopped
    networks:
      - xgboost-network
    depends_on:
      - quantconnect-api

  # Real-time Trainer
  realtime-trainer:
    build:
      context: .
      dockerfile: Dockerfile.trainer
    container_name: xgboost-realtime-trainer
    environment:
      - TRADING_DB_HOST=${TRADING_DB_HOST}
      - TRADING_DB_PORT=${TRADING_DB_PORT}
      - TRADING_DB_USER=${TRADING_DB_USER}
      - TRADING_DB_PASSWORD=${TRADING_DB_PASSWORD}
      - TRADING_DB_NAME=${TRADING_DB_NAME}
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}
    volumes:
      - ../output_train:/app/output_train
      - ../state:/app/state
      - ../logs:/app/logs
    restart: unless-stopped
    networks:
      - xgboost-network
    depends_on:
      - quantconnect-api

networks:
  xgboost-network:
    driver: bridge

volumes:
  output_train:
    driver: local
  state:
    driver: local
  logs:
    driver: local
EOF

# Create Dockerfiles
print_status "Creating Dockerfiles..."

# Dockerfile for API
cat > Dockerfile.api << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code
COPY quantconnect_api.py .

# Create necessary directories
RUN mkdir -p /app/output_train /app/state /app/logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run API server
CMD ["python", "quantconnect_api.py"]
EOF

# Dockerfile for Monitor
cat > Dockerfile.monitor << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy monitor code
COPY realtime_monitor.py .

# Create necessary directories
RUN mkdir -p /app/state /app/logs

# Run monitor
CMD ["python", "realtime_monitor.py"]
EOF

# Dockerfile for Trainer
cat > Dockerfile.trainer << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy trainer code
COPY realtime_trainer.py .

# Create necessary directories
RUN mkdir -p /app/output_train /app/state /app/logs

# Run trainer (will be triggered by monitor)
CMD ["python", "realtime_trainer.py", "--mode", "incremental"]
EOF

# Create requirements.txt
print_status "Creating requirements.txt..."
cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.4.2
python-dotenv==1.0.0
requests==2.31.0
pandas==2.1.3
numpy==1.24.3
scikit-learn==1.3.2
xgboost==2.0.2
joblib==1.3.2
pymysql==1.1.0
schedule==1.2.0
python-multipart==0.0.6
aiofiles==23.2.1
EOF

# Build and start containers
print_status "Building Docker containers..."
docker-compose build

print_status "Starting containers..."
docker-compose up -d

# Wait for services to be ready
print_status "Waiting for services to be ready..."
sleep 10

# Check service health
print_status "Checking service health..."

# Check API health
API_URL="http://localhost:${API_PORT:-8000}"
if curl -f -s "$API_URL/health" > /dev/null; then
    print_status "✅ API service is healthy"
else
    print_warning "⚠️ API service might still be starting up"
fi

# Check if containers are running
if docker-compose ps | grep -q "Up"; then
    print_status "✅ All containers are running"
    docker-compose ps
else
    print_error "❌ Some containers failed to start"
    docker-compose ps
    exit 1
fi

# Create management scripts
print_status "Creating management scripts..."

# Status script
cat > status.sh << 'EOF'
#!/bin/bash

echo "📊 XGBoost Trading System - Service Status"
echo "========================================"

docker-compose ps

echo ""
echo "🔍 Container Logs:"
echo "------------------"

echo "📡 API Logs:"
docker-compose logs --tail=20 quantconnect-api

echo ""
echo "👀 Monitor Logs:"
docker-compose logs --tail=10 realtime-monitor

echo ""
echo "🏋 Trainer Logs:"
docker-compose logs --tail=10 realtime-trainer
EOF

# Manual training trigger script
cat > trigger_training.sh << 'EOF'
#!/bin/bash

echo "🚀 Triggering Manual Training"
echo "============================"

# Create trigger file
cat > ../state/realtime_trigger.json << EOF
{
  "timestamp": "$(date -Iseconds)",
  "trigger_reason": "manual_trigger",
  "triggered_by": "$(whoami)",
  "tables_with_new_data": ["cg_spot_price_history"]
}
EOF

echo "✅ Training trigger created"
echo "📋 Check trainer logs for progress:"
echo "   docker-compose logs -f realtime-trainer"
EOF

# Stop script
cat > stop.sh << 'EOF'
#!/bin/bash

echo "🛑 Stopping XGBoost Trading System"
echo "=================================="

docker-compose down

echo "✅ All services stopped"
EOF

# Update script
cat > update.sh << 'EOF'
#!/bin/bash

echo "🔄 Updating XGBoost Trading System"
echo "=================================="

# Pull latest code (if using git)
if [ -d .git ]; then
    git pull
fi

# Rebuild and restart
docker-compose down
docker-compose build
docker-compose up -d

echo "✅ System updated and restarted"
EOF

# Make scripts executable
chmod +x status.sh trigger_training.sh stop.sh update.sh

# Success message
echo ""
echo "✅ Deployment completed successfully!"
echo ""
echo "🎉 Your XGBoost Real-time Trading System is now running!"
echo ""
echo "📊 Service URLs:"
echo "   • API Health: $API_URL/health"
echo "   • API Docs: $API_URL/docs"
echo "   • API Status: $API_URL/status"
echo ""
echo "🔧 Management Commands:"
echo "   • Check status: ./status.sh"
echo "   • View logs: docker-compose logs -f [service-name]"
echo "   • Trigger training: ./trigger_training.sh"
echo "   • Stop services: ./stop.sh"
echo "   • Update system: ./update.sh"
echo ""
echo "📱 Don't forget to configure your Telegram notifications in .env"
echo "💡 Your QuantConnect algorithm can now use the API at test.dragonfortune.ai:8000"
echo ""
echo "⚠️ IMPORTANT: Configure your firewall and SSL certificates for production use"