#!/bin/bash

# XGBoost API Deployment Script
# This script deploys the XGBoost API for QuantConnect integration

set -e

echo "🚀 XGBoost API Deployment Script"
echo "================================"

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Load production environment
if [ -f .env.production ]; then
    export $(cat .env.production | grep -v '^#' | xargs)
    echo "✅ Production environment loaded from .env.production"
else
    echo "⚠️ .env.production not found, using defaults"
    export API_HOST=${API_HOST:-0.0.0.0}
    export API_PORT=${API_PORT:-5000}
    export API_DEBUG=${API_DEBUG:-false}

    # Database configuration (can be overridden)
    export DB_HOST=${DB_HOST:-103.150.81.86}
    export DB_PORT=${DB_PORT:-3306}
    export DB_NAME=${DB_NAME:-xgboostqc}
    export DB_USER=${DB_USER:-xgboostqc}
    export DB_PASSWORD=${DB_PASSWORD:-6SPxBDwXH6WyxpfT}
fi

# Set CORS origins for production
if [ -z "$ALLOWED_ORIGINS" ]; then
    export ALLOWED_ORIGINS="*"
    echo "🌐 Setting CORS to allow all origins: $ALLOWED_ORIGINS"
fi

echo "📋 Configuration:"
echo "   API Host: ${API_HOST}"
echo "   API Port: ${API_PORT}"
echo "   API Debug: ${API_DEBUG}"
echo "   Database Host: ${DB_HOST}"
echo "   Database Port: ${DB_PORT}"
echo "   Database Name: ${DB_NAME}"
echo ""

# Build and start the API container
echo "🔨 Building XGBoost API container..."
docker-compose -f docker-compose.api.yml build

echo "🚀 Starting XGBoost API container..."
docker-compose -f docker-compose.api.yml up -d

echo "⏳ Waiting for API to be healthy..."
sleep 10

# Check if API is running (internal container check)
if docker exec xgboost-api curl -f http://localhost:5000/health > /dev/null 2>&1; then
    echo "✅ XGBoost API is running successfully!"
    echo ""
    echo "🌐 API Only accessible through domain:"
    echo "   Health Check: https://api.dragonfortune.ai/health"
    echo "   Latest Model: https://api.dragonfortune.ai/api/v1/latest/model"
    echo "   Latest Summary: https://api.dragonfortune.ai/api/v1/latest/dataset-summary"
    echo "   List Sessions: https://api.dragonfortune.ai/api/v1/sessions"
    echo ""
    echo "📖 API Documentation: https://api.dragonfortune.ai/docs"
    echo "📖 Alternative Docs: https://api.dragonfortune.ai/redoc"
    echo ""
    echo "📖 For QuantConnect integration, see quantconnect_integration_example.py"
    echo ""
    echo "🧪 To test the API (through domain):"
    echo "   curl https://api.dragonfortune.ai/health"
    echo "   curl https://api.dragonfortune.ai/api/v1/latest/model"
    echo "   curl https://api.dragonfortune.ai/api/v1/latest/dataset-summary"
    echo ""
    echo "⚠️  Direct IP/port access is disabled for security"
else
    echo "❌ XGBoost API failed to start properly!"
    echo "📋 Check logs with: docker-compose -f docker-compose.api.yml logs"
    exit 1
fi

echo ""
echo "✅ Deployment completed successfully!"
echo "🔧 To stop the API: docker-compose -f docker-compose.api.yml down"
echo "📋 To view logs: docker-compose -f docker-compose.api.yml logs -f"
echo "📖 To view API docs: open https://api.dragonfortune.ai/docs"