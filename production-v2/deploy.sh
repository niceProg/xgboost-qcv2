#!/bin/bash

# XGBoost Real-time System Deployment - API & Monitoring Only
# Run this AFTER historical training with simple_run.sh

set -e  # Exit on any error

echo "🚀 XGBoost Real-time System Deployment"
echo "===================================="
echo "🤖 Smart monitoring & API server for 2025 data"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

print_status "Working directory: $SCRIPT_DIR"

# Check if historical training was completed
print_header "Step 1: Checking Prerequisites"
echo ""

# Check .env file
if [ ! -f .env ]; then
    print_error "❌ .env file not found!"
    echo ""
    print_header "🔧 Setup Required:"
    echo ""
    echo "1. Copy environment template:"
    echo "   cp .env.example .env"
    echo ""
    echo "2. Edit with your database credentials:"
    echo "   nano .env"
    echo ""
    print_warning "⚠️  Never commit .env file to version control!"
    echo ""
    exit 1
fi

# Load environment variables
source .env
print_status "✅ Environment file loaded"

# Check if model files exist from historical training
print_status "Checking for trained models..."
MODEL_DIR="../output_train"

if [ ! -d "$MODEL_DIR" ]; then
    print_warning "⚠️ Model directory not found: $MODEL_DIR"
    print_warning "   Have you run historical training first?"
    echo ""
    print_header "📋 Required Steps:"
    echo "1. cd .. (go to parent folder)"
    echo "2. ./simple_run.sh (run historical training)"
    echo "3. cd production-v2 (back to this folder)"
    echo "4. ./deploy.sh (run this script again)"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    model_count=$(ls -1 "$MODEL_DIR"/*.joblib 2>/dev/null | wc -l)
    if [ "$model_count" -gt 0 ]; then
        print_status "✅ Found $model_count trained model(s)"
        echo "   Models:"
        ls -la "$MODEL_DIR"/*.joblib 2>/dev/null | head -3
    else
        print_warning "⚠️ No model files found in $MODEL_DIR"
        print_warning "   System will create models when data arrives"
    fi
fi

echo ""

# Check Docker
print_header "Step 2: Checking Docker Installation"
echo ""

if ! command -v docker &> /dev/null; then
    print_error "❌ Docker is not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_error "❌ Docker Compose is not installed"
    exit 1
fi

print_status "✅ Docker and Docker Compose are available"
echo ""

# Test database connection (quick test)
print_header "Step 3: Testing Database Connection"
echo ""

python3 -c "
import pymysql
import os
from dotenv import load_dotenv

load_dotenv('.env')

try:
    conn = pymysql.connect(
        host=os.getenv('TRADING_DB_HOST'),
        port=int(os.getenv('TRADING_DB_PORT', 3306)),
        user=os.getenv('TRADING_DB_USER'),
        password=os.getenv('TRADING_DB_PASSWORD'),
        database=os.getenv('TRADING_DB_NAME'),
        connect_timeout=5
    )
    print('✅ Database connection successful')
    conn.close()
except Exception as e:
    print(f'❌ Database connection failed: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    print_error "Database connection test failed!"
    print_error "Please check your .env configuration"
    exit 1
fi

print_status "✅ Database connection validated"
echo ""

# Restructure output_train directory
print_header "Step 4: Restructuring Output Train Directory"
echo ""

print_status "Running directory restructuring..."
python3 ../restructure_output_train.py

if [ $? -eq 0 ]; then
    print_status "✅ Directory restructuring completed"
else
    print_warning "⚠️ Directory restructuring had issues, continuing..."
fi

# Setup database for monitoring
print_header "Step 5: Setting up Database for Smart Monitoring"
echo ""

if [ -f "setup_database.py" ]; then
    print_status "Running database setup..."
    python3 setup_database.py

    if [ $? -eq 0 ]; then
        print_status "✅ Database setup completed"
    else
        print_error "❌ Database setup failed!"
        exit 1
    fi
else
    print_warning "⚠️ setup_database.py not found, skipping"
fi

echo ""

# Create Docker Compose file
print_header "Step 6: Creating Docker Configuration"
echo ""

cat > docker-compose.yml << EOF
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
      # Trading Database (for monitoring)
      TRADING_DB_HOST: \${TRADING_DB_HOST}
      TRADING_DB_PORT: \${TRADING_DB_PORT}
      TRADING_DB_USER: \${TRADING_DB_USER}
      TRADING_DB_PASSWORD: \${TRADING_DB_PASSWORD}
      TRADING_DB_NAME: \${TRADING_DB_NAME}

      # Results Database (for storing results)
      DB_HOST: \${DB_HOST}
      DB_PORT: \${DB_PORT}
      DB_USER: \${DB_USER}
      DB_PASSWORD: \${DB_PASSWORD}
      DB_NAME: \${DB_NAME}

      # API Configuration
      API_HOST: \${API_HOST:-0.0.0.0}
      API_PORT: \${API_PORT:-8000}
      DOMAIN: \${DOMAIN}
      OUTPUT_DIR: /app/output_train

      # QuantConnect Integration
      QUANTCONNECT_CORS_ORIGIN: \${QUANTCONNECT_CORS_ORIGIN:-https://www.quantconnect.com}
      TELEGRAM_BOT_TOKEN: \${TELEGRAM_BOT_TOKEN}
      TELEGRAM_CHAT_ID: \${TELEGRAM_CHAT_ID}

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
      start_period: 60s

  # Real-time Monitor for New Data
  realtime-monitor:
    build:
      context: .
      dockerfile: Dockerfile.monitor
    container_name: xgboost-realtime-monitor
    environment:
      TRADING_DB_HOST: \${TRADING_DB_HOST}
      TRADING_DB_PORT: \${TRADING_DB_PORT}
      TRADING_DB_USER: \${TRADING_DB_USER}
      TRADING_DB_PASSWORD: \${TRADING_DB_PASSWORD}
      TRADING_DB_NAME: \${TRADING_DB_NAME}
      OUTPUT_DIR: /app/output_train

      # Focus on 2025 data
      focus_year: 2025

    volumes:
      - ../output_train:/app/output_train
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
      TRADING_DB_HOST: \${TRADING_DB_HOST}
      TRADING_DB_PORT: \${TRADING_DB_PORT}
      TRADING_DB_USER: \${TRADING_DB_USER}
      TRADING_DB_PASSWORD: \${TRADING_DB_PASSWORD}
      TRADING_DB_NAME: \${TRADING_DB_NAME}
      OUTPUT_DIR: /app/output_train

      # Training Configuration
      PERFORMANCE_THRESHOLD: \${PERFORMANCE_THRESHOLD:-0.6}

    volumes:
      - ../output_train:/app/output_train
      - ../state:/app/state
      - ../logs:/app/logs
    restart: "no"  # Controlled by monitor
    networks:
      - xgboost-network

networks:
  xgboost-network:
    driver: bridge
EOF

print_status "✅ Docker Compose configuration created"

# Create Dockerfiles
print_status "Creating Dockerfiles..."

# API Dockerfile
cat > Dockerfile.api << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code
COPY structured_api.py .

# Also copy original for compatibility
COPY quantconnect_api.py .

# Copy restructuring script
COPY ../restructure_output_train.py ../

# Create necessary directories
RUN mkdir -p /app/output_train /app/state /app/logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run API server
CMD ["python", "structured_api.py"]
EOF

# Monitor Dockerfile
cat > Dockerfile.monitor << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

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

# Trainer Dockerfile
cat > Dockerfile.trainer << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy trainer code
COPY realtime_trainer_pipeline.py .

# Create necessary directories
RUN mkdir -p /app/output_train /app/state /app/logs

# Create log directory
RUN mkdir -p /var/log

# Run trainer (will be triggered by monitor)
CMD ["python", "realtime_trainer_pipeline.py", "--mode", "incremental"]
EOF

# Requirements
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

# Core Pipeline Dependencies
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.17.0
tqdm==4.66.1
yfinance==0.2.28
ta==0.10.2  # Alternative to ta-lib, pure Python
EOF

print_status "✅ Dockerfiles and requirements created"
echo ""

# Build and start containers
print_header "Step 7: Building and Starting Services"
echo ""

print_status "Building Docker containers..."
docker-compose build

print_status "Starting containers..."
docker-compose up -d

# Wait for services
print_status "Waiting for services to start..."
sleep 30

# Check service health
print_header "Step 8: Verifying Deployment"
echo ""

# Check API health
API_URL="http://localhost:${API_PORT:-8000}"
if curl -f -s "$API_URL/health" > /dev/null; then
    print_status "✅ API service is healthy"
else
    print_warning "⚠️ API service might still be starting up"
fi

# Check containers
print_status "Checking container status..."
if docker-compose ps | grep -q "Up"; then
    print_status "✅ Containers are running"
    docker-compose ps
else
    print_error "❌ Some containers failed to start"
    docker-compose ps
    exit 1
fi

echo ""

# Create management scripts
print_header "Step 9: Creating Management Scripts"
echo ""

# Status script
cat > status.sh << 'EOF'
#!/bin/bash

echo "📊 XGBoost Real-time System Status"
echo "=================================="

docker-compose ps

echo ""
echo "🔍 Service Health:"
echo "------------------"

API_URL="http://localhost:8000"
echo "📡 API Health:"
curl -s "$API_URL/health" | python3 -m json.tool 2>/dev/null || echo "   API not responding"

echo ""
echo "📚 API Documentation:"
echo "   Swagger UI: $API_URL/docs"
echo "   ReDoc: $API_URL/redoc"

echo ""
echo "📈 API Status:"
curl -s "$API_URL/status" | python3 -m json.tool 2>/dev/null || echo "   Status endpoint not responding"

echo ""
echo "🔍 Container Logs (last 10 lines each):"
echo "----------------------------------------"

echo "📡 API Logs:"
docker-compose logs --tail=10 quantconnect-api

echo ""
echo "👀 Monitor Logs:"
docker-compose logs --tail=10 realtime-monitor

echo ""
echo "🏋 Trainer Logs:"
docker-compose logs --tail=10 realtime-trainer
EOF

# Stop script
cat > stop.sh << 'EOF'
#!/bin/bash

echo "🛑 Stopping XGBoost Real-time System"
echo "=================================="

docker-compose down

echo "✅ All services stopped"
EOF

# Restart script
cat > restart.sh << 'EOF'
#!/bin/bash

echo "🔄 Restarting XGBoost Real-time System"
echo "===================================="

docker-compose restart

echo "✅ Services restarted"
echo ""
echo "Check status with: ./status.sh"
EOF

# Test API script
cat > test_api.sh << 'EOF'
#!/bin/bash

API_URL="http://localhost:8000"

echo "🧪 Testing Read-Only XGBoost API Endpoints"
echo "========================================"

echo ""
echo "1. Testing API Root..."
response=$(curl -s "$API_URL/")
echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"

echo ""
echo "2. Testing Health Check..."
response=$(curl -s "$API_URL/health")
echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"

echo ""
echo "3. Testing Model Info..."
response=$(curl -s "$API_URL/model/info")
echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"

echo ""
echo "4. Testing System Status..."
response=$(curl -s "$API_URL/status")
echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"

echo ""
echo "5. Testing Training Status..."
response=$(curl -s "$API_URL/training/status")
echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"

echo ""
echo "6. Testing Model List..."
response=$(curl -s "$API_URL/model/list")
echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"

echo ""
echo "7. Testing Monitoring Metrics..."
response=$(curl -s "$API_URL/monitoring/metrics")
echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"

echo ""
echo "8. Testing Config Info..."
response=$(curl -s "$API_URL/config/info")
echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"

echo ""
echo "📚 Full Documentation: $API_URL/docs"
echo "🔒 Mode: Read-Only (GET methods only)"
EOF

# Trigger training script
cat > trigger_training.sh << 'EOF'
#!/bin/bash

echo "🚀 Triggering Manual Model Training"
echo "=================================="

# Create trigger file
mkdir -p ../state
cat > ../state/realtime_trigger.json << EOF
{
  "timestamp": "$(date -Iseconds)",
  "trigger_reason": "manual_trigger",
  "triggered_by": "$(whoami)",
  "tables_with_new_data": ["cg_spot_price_history"]
}
EOF

echo "✅ Training trigger created"
echo "📋 Monitor trainer logs:"
echo "   docker-compose logs -f realtime-trainer"
EOF

# Make scripts executable
chmod +x status.sh stop.sh restart.sh test_api.sh trigger_training.sh

print_status "✅ Management scripts created"
echo ""

# Success message
print_header "🎉 Real-time System Deployment Complete!"
echo ""
echo "${GREEN}What's been deployed:${NC}"
echo "  ✅ FastAPI server for QuantConnect (Port ${API_PORT:-8000})"
echo "  ✅ Real-time database monitor (2025 data)"
echo "  ✅ Incremental model trainer"
echo "  ✅ Management scripts"
echo ""
echo "${YELLOW}Structured API Endpoints:${NC}"
echo "  📁 Directory Listing: $API_URL/output_train/"
echo "  🤖 Model Files: $API_URL/output_train/models/"
echo "  🎯 Latest Model: $API_URL/output_train/models/latest"
echo "  📋 Dataset Summary: $API_URL/output_train/datasets/summary"
echo "  📊 All Datasets: $API_URL/output_train/datasets/"
echo "  🔧 Structure Info: $API_URL/structure"
echo "  ❤️ Health Check: $API_URL/health"
echo ""
echo "📚 Documentation: $API_URL/docs"
echo "📖 Alternative Docs: $API_URL/redoc"
echo ""
echo "🔒 Mode: Read-Only (GET methods only - file access only)"
echo ""
echo "${YELLOW}Management Commands:${NC}"
echo "  📊 System status: ./status.sh"
echo "  🧪 Test API: ./test_api.sh"
echo "  🚀 Trigger training: ./trigger_training.sh"
echo "  🔄 Restart system: ./restart.sh"
echo "  🛑 Stop system: ./stop.sh"
echo ""
echo "${BLUE}QuantConnect Integration:${NC}"
echo "  📄 Algorithm: XGBoostTradingAlgorithm_Final.py"
echo "  🌐 API Base URL: https://${DOMAIN}:8000"
echo "  🔗 Replace ObjectStore with API calls"
echo ""
echo "${GREEN}🚀 Your automated real-time system is now running!${NC}"
echo "   It will detect new 2025 data and update models automatically"
echo ""
echo "${BLUE}Next Steps:${NC}"
echo "  1. Test API: ./test_api.sh"
echo "  2. Upload XGBoostTradingAlgorithm_Final.py to QuantConnect"
echo "  3. Configure algorithm with your API endpoint"
echo "  4. Monitor: ./status.sh"