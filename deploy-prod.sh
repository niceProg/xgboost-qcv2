#!/bin/bash

# Deploy XGBoost FastAPI to Production
# Uses existing external databases

set -e

echo "=========================================="
echo "XGBoost FastAPI - Production Deployment"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose-prod.yml"
DOCKERFILE="Dockerfile.prod"
IMAGE_NAME="xgboost-api:prod"
API_PORT=${API_PORT:-8000}

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check dependencies
check_dependencies() {
    log_step "Checking dependencies..."

    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi

    # Check .env file
    if [ ! -f .env ]; then
        log_error ".env file not found!"
        log_info "Please create .env file with database configuration"
        exit 1
    fi

    log_info "✓ Dependencies OK"
}

# Validate environment
validate_env() {
    log_step "Validating environment variables..."

    # Source .env file
    source .env

    # Required variables
    required_vars=("TRADING_DB_HOST" "TRADING_DB_USER" "TRADING_DB_PASSWORD" "TRADING_DB_NAME"
                   "DB_HOST" "DB_USER" "DB_PASSWORD" "DB_NAME")

    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            log_error "Required variable $var is not set in .env"
            exit 1
        fi
    done

    log_info "✓ Environment variables validated"
}

# Build Docker image
build_image() {
    log_step "Building Docker image..."

    # Build using production Dockerfile
    if docker build -f $DOCKERFILE -t $IMAGE_NAME .; then
        log_info "✓ Docker image built successfully"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
}

# Create directories
create_directories() {
    log_step "Creating directories..."

    mkdir -p output_train logs
    chmod 755 output_train logs

    log_info "✓ Directories created"
}

# Stop existing API
stop_existing_api() {
    log_step "Stopping existing API container..."

    if docker ps -q -f name=xgboost_api | grep -q .; then
        log_info "Stopping existing API container..."
        docker stop xgboost_api || true
        docker rm xgboost_api || true
    fi

    log_info "✓ Existing API stopped"
}

# Deploy API
deploy_api() {
    log_step "Deploying FastAPI server..."

    # Source .env for environment variables
    source .env

    # Run API container
    docker run -d \
        --name xgboost_api \
        --env-file .env \
        -p $API_PORT:8000 \
        -v $(pwd)/output_train:/app/output_train:ro \
        -v $(pwd)/logs:/app/logs \
        --restart unless-stopped \
        $IMAGE_NAME

    log_info "✓ API server deployed"
}

# Wait for API to be ready
wait_for_api() {
    log_step "Waiting for API to be ready..."

    for i in {1..60}; do
        if curl -s http://localhost:$API_PORT/api/v1/health > /dev/null 2>&1; then
            log_info "✓ API server is ready!"
            break
        fi

        if [ $i -eq 60 ]; then
            log_error "API server failed to start within 120 seconds"
            echo ""
            log_info "Checking container logs..."
            docker logs xgboost_api 2>&1 | tail -20
            exit 1
        fi

        echo -n "."
        sleep 2
    done
}

# Show status
show_status() {
    log_step "Deployment Status"
    echo ""

    # Container status
    echo "Container Status:"
    docker ps -f name=xgboost_api
    echo ""

    # URLs
    log_info "📡 API Server: http://localhost:$API_PORT"
    log_info "📚 API Documentation: http://localhost:$API_PORT/docs"
    log_info "📖 Alternative Docs: http://localhost:$API_PORT/redoc"
    echo ""

    # Management commands
    log_info "Management Commands:"
    echo "  View logs:     docker logs -f xgboost_api"
    echo "  Stop API:      docker stop xgboost_api"
    echo "  Restart API:   docker restart xgboost_api"
    echo "  Update image:  docker build -f $DOCKERFILE -t $IMAGE_NAME . && docker restart xgboost_api"
    echo ""

    # Health check
    if curl -s http://localhost:$API_PORT/api/v1/health | grep -q "healthy"; then
        log_info "✅ API is healthy and ready to use!"
    else
        log_warn "⚠️  API might not be fully ready yet"
    fi
}

# Main execution
main() {
    log_info "Starting FastAPI production deployment..."
    echo ""

    check_dependencies
    validate_env
    build_image
    create_directories
    stop_existing_api
    deploy_api
    wait_for_api
    show_status

    echo ""
    log_info "🎉 Deployment completed successfully!"
}

# Run main function
main "$@"