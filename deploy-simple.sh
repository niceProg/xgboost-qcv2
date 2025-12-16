#!/bin/bash

# Simple FastAPI Deployment Script
#==================================

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "${GREEN}=== FastAPI Simple Deployment ===${NC}"

# Config
IMAGE_NAME="xgboost-api:prod"
CONTAINER_NAME="xgboost_api"
API_PORT=${API_PORT:-8000}

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Check if .env exists
if [ ! -f .env ]; then
    print_error ".env file not found!"
    exit 1
fi

# Stop and remove existing container
if docker ps -a -q -f name=$CONTAINER_NAME | grep -q .; then
    print_status "Stopping existing container..."
    docker stop $CONTAINER_NAME || true
    docker rm $CONTAINER_NAME || true
fi

# Build image
print_status "Building Docker image..."
docker build -f Dockerfile.prod -t $IMAGE_NAME .

# Run container
print_status "Starting FastAPI server..."
docker run -d \
    --name $CONTAINER_NAME \
    --env-file .env \
    -p $API_PORT:8000 \
    -v $(pwd)/output_train:/app/output_train:ro \
    -v $(pwd)/logs:/app/logs \
    --restart unless-stopped \
    $IMAGE_NAME

# Wait for container to start
print_status "Waiting for API to start..."
sleep 10

# Check if container is running
if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
    print_status "✅ Container is running!"

    # Show logs
    print_status "Recent logs:"
    docker logs $CONTAINER_NAME 2>&1 | tail -10

    echo ""
    print_status "📡 API Server: http://localhost:$API_PORT"
    print_status "📚 API Docs: http://localhost:$API_PORT/docs"
    echo ""
    print_status "To view logs: docker logs -f $CONTAINER_NAME"
    print_status "To stop: docker stop $CONTAINER_NAME"
else
    print_error "❌ Failed to start container!"
    docker logs $CONTAINER_NAME 2>&1
    exit 1
fi