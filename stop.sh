#!/bin/bash

# Stop XGBoost Trading Pipeline
# Quick stop script

set -e

echo "=========================================="
echo "XGBoost Trading Pipeline - Stop Services"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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

# Stop all services
stop_all() {
    log_info "Stopping all XGBoost services..."

    # Use sudo if needed
    DOCKER_CMD="docker"
    if ! docker ps &>/dev/null; then
        DOCKER_CMD="sudo docker"
    fi

    # Stop all docker-compose services
    if [ -f docker-compose.yml ]; then
        $DOCKER_CMD-compose --profile pipeline --profile api --profile scheduler down 2>/dev/null || true

        # Force remove any running containers
        $DOCKER_CMD stop $($DOCKER_CMD ps -q --filter "name=xgboost") 2>/dev/null || true
        $DOCKER_CMD rm $($DOCKER_CMD ps -aq --filter "name=xgboost") 2>/dev/null || true

        log_info "All services stopped!"
    else
        log_error "docker-compose.yml not found!"
    fi
}

# Clean up (optional)
cleanup() {
    read -p "Remove Docker images? (y/N): " remove_images

    if [[ $remove_images =~ ^[Yy]$ ]]; then
        log_info "Removing Docker images..."
        $DOCKER_CMD rmi xgboost-qc:latest 2>/dev/null || true
        $DOCKER_CMD image prune -f
        log_info "Images removed!"
    fi

    read -p "Clean up old logs and data? (y/N): " cleanup_data

    if [[ $cleanup_data =~ ^[Yy]$ ]]; then
        SERVER_DIR="/opt/xgboost-qc"
        if [ -d "$SERVER_DIR/logs" ]; then
            sudo find $SERVER_DIR/logs -name "*.log" -mtime +7 -delete
            log_info "Old logs cleaned!"
        fi

        if [ -d "$SERVER_DIR/output_train" ]; then
            sudo find $SERVER_DIR/output_train -name "*.csv" -mtime +30 -delete
            sudo find $SERVER_DIR/output_train -name "xgboost_trading_model_*.joblib" -mtime +30 -delete
            log_info "Old models cleaned!"
        fi
    fi
}

# Show status
show_status() {
    log_info "Checking status..."

    # Use sudo if needed
    DOCKER_CMD="docker"
    if ! docker ps &>/dev/null; then
        DOCKER_CMD="sudo docker"
    fi

    # Check containers
    running_containers=$($DOCKER_CMD ps --filter "name=xgboost" --format "table {{.Names}}")
    if [ -n "$running_containers" ]; then
        log_warn "Still running containers:"
        echo "$running_containers"
    else
        log_info "All containers stopped!"
    fi

    # Show disk usage
    if [ -d "/opt/xgboost-qc" ]; then
        echo ""
        log_info "Disk usage:"
        du -sh /opt/xgboost-qc/* 2>/dev/null | sort -rh
    fi
}

# Main execution
main() {
    stop_all

    echo ""
    read -p "Run cleanup? (y/N): " do_cleanup

    if [[ $do_cleanup =~ ^[Yy]$ ]]; then
        cleanup
    fi

    show_status

    echo ""
    log_info "All XGBoost services stopped!"
    log_info "Use './deploy.sh' to restart services"
}

# Run main function
main