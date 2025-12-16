#!/bin/bash

echo "🚀 Starting XGBoost Real-time Trading System"
echo "==========================================="

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p ./state
mkdir -p ./realtime_data
mkdir -p ./logs
mkdir -p ./grafana/dashboards
mkdir -p ./grafana/datasources

# Make scripts executable
chmod +x realtime_monitor.py
chmod +x realtime_trainer.py
chmod +x realtime_api.py

# Create default config files
echo "📝 Creating configuration files..."

# Realtime config
if [ ! -f realtime_config.json ]; then
cat > realtime_config.json << EOF
{
  "monitor_pairs": ["BTCUSDT", "ETHUSDT"],
  "monitor_intervals": ["1h", "4h"],
  "exchanges": ["binance"],
  "processing": {
    "batch_delay": 30,
    "max_batch_size": 5000,
    "min_new_records": 10
  },
  "notifications": {
    "enabled": true,
    "webhook_url": "",
    "telegram_bot_token": "",
    "telegram_chat_id": ""
  },
  "model_update": {
    "enabled": true,
    "update_frequency": "hourly",
    "min_samples_for_update": 100,
    "performance_threshold": 0.6
  }
}
EOF
fi

# Monitor status file
if [ ! -f ./state/monitor_status.json ]; then
cat > ./state/monitor_status.json << EOF
{
  "running": false,
  "last_check": null,
  "start_time": null
}
EOF
fi

# Check environment variables
echo "🔍 Checking environment variables..."
required_vars=("TRADING_DB_HOST" "TRADING_DB_USER" "TRADING_DB_PASSWORD" "TRADING_DB_NAME")
missing_vars=()

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -gt 0 ]; then
    echo "⚠️  Warning: Missing environment variables:"
    for var in "${missing_vars[@]}"; do
        echo "   - $var"
    done
    echo ""
    echo "Please set these variables in your .env file or environment."
    echo "Continuing anyway (system may not work correctly)..."
    echo ""
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker-compose -f docker-compose-realtime.yml down --remove-orphans

# Build and start services
echo "🔨 Building and starting services..."
docker-compose -f docker-compose-realtime.yml up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🏥 Checking service health..."

# Check main API
if curl -f http://localhost:8000/ > /dev/null 2>&1; then
    echo "✅ Main API (port 8000) is healthy"
else
    echo "❌ Main API (port 8000) is not responding"
fi

# Check real-time API
if curl -f http://localhost:8001/health > /dev/null 2>&1; then
    echo "✅ Real-time API (port 8001) is healthy"
else
    echo "❌ Real-time API (port 8001) is not responding"
fi

# Check Redis (if available)
if docker exec xgboost_redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is healthy"
else
    echo "⚠️  Redis is not responding (optional)"
fi

# Show system status
echo ""
echo "📊 System Status:"
echo "=================="

# Show running containers
echo "Running containers:"
docker-compose -f docker-compose-realtime.yml ps

echo ""
echo "🌐 API Endpoints:"
echo "=================="
echo "Main API:         http://localhost:8001"
echo "Real-time API:    http://localhost:8000"
echo "Health Check:     http://localhost:8000/health"
echo "System Status:    http://localhost:8000/status"

echo ""
echo "📈 Quick Test Commands:"
echo "======================="
echo "# Test real-time prediction:"
echo "curl -X POST http://localhost:8000/predict \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"features\": {\"price_close\": 42000, \"volume_usd\": 1000000}}'"
echo ""
echo "# Generate trading signal:"
echo "curl -X POST http://localhost:8000/signal \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"symbol\": \"BTCUSDT\", \"threshold\": 0.5}'"
echo ""
echo "# Get market data:"
echo "curl http://localhost:8000/market/binance/BTCUSDT/1h"

echo ""
echo "📋 Logs:"
echo "========="
echo "Main API:        docker logs -f xgboost_api"
echo "Real-time API:   docker logs -f realtime_api"
echo "Monitor:         docker logs -f realtime_monitor"
echo "All logs:        docker-compose -f docker-compose-realtime.yml logs -f"

echo ""
echo "🛠️  Management:"
echo "==============="
echo "Stop system:     docker-compose -f docker-compose-realtime.yml down"
echo "Restart:         docker-compose -f docker-compose-realtime.yml restart"
echo "Update:          docker-compose -f docker-compose-realtime.yml pull && docker-compose -f docker-compose-realtime.yml up -d"

echo ""
echo "🎉 Real-time Trading System is ready!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Configure your webhook URLs in .env for notifications"
echo "2. Set up Telegram bot for alerts (optional)"
echo "3. Monitor the system via the status endpoint"
echo "4. Check logs regularly for any issues"