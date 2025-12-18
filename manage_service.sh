#!/bin/bash

# XGBoost Real-time Service Management Script
# Easy service management commands

SERVICE_NAME="xgboost-realtime"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored messages
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Show service status
show_status() {
    print_status $BLUE "📊 XGBoost Real-time Service Status"
    echo "=================================="

    if sudo systemctl is-active --quiet $SERVICE_NAME; then
        print_status $GREEN "🟢 Service Status: RUNNING"

        # Get uptime
        UPTIME=$(sudo systemctl show $SERVICE_NAME --property=ActiveEnterTimestamp | cut -d= -f2)
        echo "   Started: $UPTIME"

        # Get memory usage
        PID=$(sudo systemctl show $SERVICE_NAME --property=MainPID | cut -d= -f2)
        if [ "$PID" -gt 0 ]; then
            MEMORY=$(ps -p $PID -o rss= 2>/dev/null | awk '{print int($1/1024)"MB"}')
            echo "   Memory Usage: $MEMORY"
        fi

    else
        print_status $RED "🔴 Service Status: NOT RUNNING"
    fi

    echo ""
    echo "📋 Last 5 log lines:"
    journalctl -u $SERVICE_NAME -n 5 --no-pager
}

# Start service
start_service() {
    print_status $BLUE "🚀 Starting $SERVICE_NAME..."
    sudo systemctl start $SERVICE_NAME

    sleep 2
    if sudo systemctl is-active --quiet $SERVICE_NAME; then
        print_status $GREEN "✅ Service started successfully!"
        show_status
    else
        print_status $RED "❌ Failed to start service!"
        sudo systemctl status $SERVICE_NAME --no-pager -l
    fi
}

# Stop service
stop_service() {
    print_status $BLUE "🛑 Stopping $SERVICE_NAME..."
    sudo systemctl stop $SERVICE_NAME

    sleep 2
    if sudo systemctl is-active --quiet $SERVICE_NAME; then
        print_status $RED "❌ Service is still running!"
    else
        print_status $GREEN "✅ Service stopped successfully!"
    fi
}

# Restart service
restart_service() {
    print_status $BLUE "🔄 Restarting $SERVICE_NAME..."
    sudo systemctl restart $SERVICE_NAME

    sleep 3
    if sudo systemctl is-active --quiet $SERVICE_NAME; then
        print_status $GREEN "✅ Service restarted successfully!"
        show_status
    else
        print_status $RED "❌ Failed to restart service!"
        sudo systemctl status $SERVICE_NAME --no-pager -l
    fi
}

# Follow logs
follow_logs() {
    print_status $BLUE "📋 Following $SERVICE_NAME logs (Ctrl+C to exit)"
    echo "================================================================"
    journalctl -u $SERVICE_NAME -f
}

# Show detailed logs
show_logs() {
    local lines=${1:-20}
    print_status $BLUE "📋 Last $lines lines of $SERVICE_NAME logs"
    echo "======================================================"
    journalctl -u $SERVICE_NAME -n $lines --no-pager
}

# Monitor application logs
monitor_app_logs() {
    print_status $BLUE "📋 Following application logs (Ctrl+C to exit)"
    echo "======================================================"

    # Check if logs directory exists
    if [ -d "./logs" ]; then
        tail -f ./logs/realtime_monitor.log ./logs/realtime_trainer.log
    else
        print_status $RED "❌ Logs directory not found!"
    fi
}

# Test service
test_service() {
    print_status $BLUE "🧪 Testing $SERVICE_NAME service"
    echo "=================================="

    # Check if service file exists
    if [ ! -f "/etc/systemd/system/$SERVICE_NAME.service" ]; then
        print_status $RED "❌ Service not installed! Run ./install_systemd.sh first"
        return 1
    fi

    # Show service configuration
    echo "📄 Service Configuration:"
    sudo systemctl cat $SERVICE_NAME | head -20
    echo "..."

    # Show status
    echo ""
    show_status

    # Test database connection
    echo ""
    print_status $BLUE "🔍 Testing database connection..."
    source .xgboost-qc/bin/activate
    python3 -c "
import os
from dotenv import load_dotenv
import pymysql

load_dotenv()

try:
    conn = pymysql.connect(
        host=os.getenv('TRADING_DB_HOST', 'localhost'),
        port=int(os.getenv('TRADING_DB_PORT', 3306)),
        user=os.getenv('TRADING_DB_USER'),
        password=os.getenv('TRADING_DB_PASSWORD'),
        database=os.getenv('TRADING_DB_NAME', 'newera')
    )
    print('✅ Database connection successful!')
    conn.close()
except Exception as e:
    print(f'❌ Database connection failed: {e}')
"
}

# Enable/disable service
toggle_service() {
    if sudo systemctl is-enabled --quiet $SERVICE_NAME; then
        print_status $BLUE "🔧 Disabling $SERVICE_NAME..."
        sudo systemctl disable $SERVICE_NAME
        print_status $GREEN "✅ Service disabled (won't start on boot)"
    else
        print_status $BLUE "🔧 Enabling $SERVICE_NAME..."
        sudo systemctl enable $SERVICE_NAME
        print_status $GREEN "✅ Service enabled (will start on boot)"
    fi
}

# Show help
show_help() {
    echo "🔧 XGBoost Real-time Service Management"
    echo "======================================"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  status    Show service status and recent logs"
    echo "  start     Start the service"
    echo "  stop      Stop the service"
    echo "  restart   Restart the service"
    echo "  logs      Show last 20 lines of logs"
    echo "  follow    Follow service logs in real-time"
    echo "  monitor   Follow application logs"
    echo "  test      Test service configuration"
    echo "  toggle    Enable/disable service on boot"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 status     # Check service status"
    echo "  $0 start      # Start service"
    echo "  $0 logs 50    # Show last 50 log lines"
    echo "  $0 monitor    # Monitor application logs"
    echo ""
    echo "Service Name: $SERVICE_NAME"
    echo "Working Directory: $(pwd)"
}

# Main script logic
case "${1:-help}" in
    "status")
        show_status
        ;;
    "start")
        start_service
        ;;
    "stop")
        stop_service
        ;;
    "restart")
        restart_service
        ;;
    "logs")
        show_logs "${2:-20}"
        ;;
    "follow")
        follow_logs
        ;;
    "monitor")
        monitor_app_logs
        ;;
    "test")
        test_service
        ;;
    "toggle")
        toggle_service
        ;;
    "help"|*)
        show_help
        ;;
esac