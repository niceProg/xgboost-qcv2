#!/bin/bash

# Deploy XGBoost Trading Pipeline
# Quick deployment script

set -e

echo "=========================================="
echo "XGBoost Trading Pipeline - Quick Deploy"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="xgboost-qc"
SERVER_DIR="/opt/${PROJECT_NAME}"
IMAGE_NAME="${PROJECT_NAME}:latest"
COMPOSE_FILE="docker-compose.yml"

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

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_error "Jangan run sebagai root!"
        log_info "Run sebagai user biasa dengan sudo privilege"
        exit 1
    fi
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."

    if ! command -v docker &> /dev/null; then
        log_error "Docker belum terinstall"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose belum terinstall"
        exit 1
    fi

    log_info "Dependencies OK!"
}

# Prepare server directories
prepare_directories() {
    log_info "Preparing server directories..."

    # Create directories with sudo
    sudo mkdir -p ${SERVER_DIR}/{output_train,logs,backups}

    # Copy .env file if not exists
    if [ ! -f ${SERVER_DIR}/.env ]; then
        if [ -f .env ]; then
            log_info "Copying .env file to server directory..."
            sudo cp .env ${SERVER_DIR}/.env
        else
            log_warn ".env file tidak ditemukan, membuat default..."
            sudo tee ${SERVER_DIR}/.env > /dev/null <<EOF
# Trading Database
TRADING_DB_HOST=103.150.81.86
TRADING_DB_PORT=3306
TRADING_DB_USER=newera
TRADING_DB_PASSWORD=WCXscc8twHi3kxDW
TRADING_DB_NAME=newera

# Results Database
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
TRADING_HOURS=00:00-09:00
TIMEZONE=WIB
OUTPUT_DIR=/app/output_train

# API Configuration
API_PORT=8000
ENV=production

# Model Evaluation
LEVERAGE=10
MARGIN_FRACTION=0.2
INITIAL_CASH=1000
THRESHOLD=0.5
FEE_RATE=0.0004
SLIPPAGE_RATE=0
EOF
        fi
    fi

    # Set permissions
    sudo chown -R $USER:$USER ${SERVER_DIR}
    sudo chmod 600 ${SERVER_DIR}/.env

    log_info "Directories prepared at ${SERVER_DIR}"
}

# Build Docker image
build_image() {
    log_info "Building Docker image..."

    # Build image
    if docker build -t ${IMAGE_NAME} .; then
        log_info "Docker image built successfully!"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
}

# Deploy services
deploy_services() {
    log_info "Deploying services..."

    # Create symlink to server directory
    if [ ! -L ./output_train ]; then
        ln -sf ${SERVER_DIR}/output_train ./output_train
    fi

    if [ ! -L ./logs ]; then
        ln -sf ${SERVER_DIR}/logs ./logs
    fi

    # Read deployment choice
    echo ""
    echo "Pilih deployment mode:"
    echo "1) Daily Pipeline Only"
    echo "2) API Server Only"
    echo "3) Daily Pipeline + API Server (Recommended)"
    echo "4) All services + Scheduler"
    read -p "Pilih (1-4): " choice

    case $choice in
        1)
            log_info "Deploying Daily Pipeline..."
            docker-compose --profile pipeline up -d
            ;;
        2)
            log_info "Deploying API Server..."
            docker-compose --profile api up -d
            ;;
        3)
            log_info "Deploying Daily Pipeline + API Server..."
            docker-compose --profile pipeline --profile api up -d
            ;;
        4)
            log_info "Deploying All Services..."
            docker-compose --profile pipeline --profile api --profile scheduler up -d
            ;;
        *)
            log_error "Invalid choice"
            exit 1
            ;;
    esac
}

# Wait for services to be ready
wait_for_services() {
    log_info "Waiting for services to be ready..."

    # Wait for API server
    if docker ps --filter "name=xgboost_api" --format "table {{.Names}}" | grep -q xgboost_api; then
        log_info "Waiting for API server to be healthy..."
        for i in {1..30}; do
            if curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
                log_info "API server is healthy!"
                break
            fi
            sleep 2
        done
    fi
}

# Show status
show_status() {
    log_info "Deployment Status:"
    echo ""
    docker-compose ps
    echo ""

    # Show URLs
    if docker ps --filter "name=xgboost_api" --format "table {{.Names}}" | grep -q xgboost_api; then
        log_info "API Server: http://localhost:8000"
        log_info "API Documentation: http://localhost:8000/docs"
    fi

    log_info "Data Directory: ${SERVER_DIR}/output_train"
    log_info "Logs Directory: ${SERVER_DIR}/logs"

    # Show how to check logs
    echo ""
    log_info "Check logs with:"
    echo "  docker-compose logs -f xgboost_pipeline"
    echo "  docker-compose logs -f xgboost_api"
    echo "  docker-compose logs -f"
}

# Main execution
main() {
    log_info "Starting deployment..."

    check_root
    check_dependencies
    prepare_directories
    build_image
    deploy_services
    wait_for_services
    show_status

    echo ""
    log_info "Deployment completed successfully!"
    log_info "Use './stop.sh' to stop all services"
    log_info "Use './deploy.sh' again to update"
}

# Run main function
main