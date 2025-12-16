#!/bin/bash

echo "🌐 Starting XGBoost Public API Server"
echo "=================================="

# Check requirements
echo "🔍 Checking requirements..."

# Check domain
if [ -z "$DOMAIN" ]; then
    echo "❌ DOMAIN environment variable not set"
    echo "Set: export DOMAIN=your-domain.com"
    exit 1
fi

# Check SSL certificate
if [ ! -f "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ]; then
    echo "⚠️ SSL certificate not found"
    echo "Run: certbot --nginx -d $DOMAIN"
    echo "Or start without SSL: docker-compose -f docker-compose-public.yml up --scale nginx=0 xgboost_realtime_api realtime_monitor"
    read -p "Continue without SSL? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
    USE_SSL=false
else
    USE_SSL=true
fi

# Update nginx config with domain
sed -i "s/your-domain.com/$DOMAIN/g" nginx.conf

# Create directories
mkdir -p ./state
mkdir -p ./realtime_data
mkdir -p ./logs

# Make scripts executable
chmod +x realtime_monitor.py
chmod +x realtime_api.py

# Start services
echo "🚀 Starting services..."

if [ "$USE_SSL" = true ]; then
    # Start with SSL
    docker-compose -f docker-compose-public.yml up -d
else
    # Start without SSL
    docker-compose -f docker-compose-public.yml up -d --scale nginx=0 xgboost_realtime_api realtime_monitor
    echo "⚠️ Started without SSL - HTTP only"
    echo "📡 API available at: http://localhost:8001"
fi

# Wait for services
echo "⏳ Waiting for API to start..."
sleep 10

# Test API
echo "🧪 Testing API endpoints..."

# Test health
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed"
fi

# Test signal endpoint
echo "📊 Testing signal generation..."
response=$(curl -s -X POST http://localhost:8000/signal \
    -H 'Content-Type: application/json' \
    -d '{"symbol": "BTCUSDT", "threshold": 0.5}')

if echo "$response" | grep -q "signal"; then
    echo "✅ Signal endpoint working"
    echo "📈 Sample signal: $response"
else
    echo "❌ Signal endpoint failed"
fi

# Display information
echo ""
echo "🌐 Public API Server Ready!"
echo "============================"
echo ""
echo "📡 API Endpoints:"
if [ "$USE_SSL" = true ]; then
    echo "HTTPS: https://$DOMAIN"
    echo "Health: https://$DOMAIN/health"
    echo "Signal: https://$DOMAIN/signal"
else
    echo "HTTP: http://localhost:8000"
    echo "Health: http://localhost:8000/health"
    echo "Signal: http://localhost:8000/signal"
fi

echo ""
echo "📋 QuantConnect Setup:"
echo "======================"
echo "1. Copy XGBoostTradingAlgorithm_API.py to your QuantConnect project"
echo "2. Update api_base_url in the algorithm:"
if [ "$USE_SSL" = true ]; then
    echo "   self.api_base_url = \"https://$DOMAIN\""
else
    echo "   self.api_base_url = \"http://YOUR_PUBLIC_IP:8001\""
fi
echo "3. Set your Telegram token in the algorithm"
echo "4. Backtest to verify"
echo "5. Deploy to live trading"

echo ""
echo "🔧 Management Commands:"
echo "======================"
echo "View logs:      docker logs -f xgboost_realtime_api"
echo "View monitor:   docker logs -f realtime_monitor"
echo "Stop:           docker-compose -f docker-compose-public.yml down"
echo "Restart:        docker-compose -f docker-compose-public.yml restart"

echo ""
echo "✅ Public API server is running!"
echo "🚀 Your QuantConnect can now call it for real-time signals!"