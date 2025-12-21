#!/bin/bash

# XGBoost Automation Setup Script
# Sets up automatic training and Telegram notifications

set -e

echo "🤖 XGBoost Automation Setup"
echo "============================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if required files exist
check_prerequisites() {
    print_info "Checking prerequisites..."

    if [ ! -f "realtime_monitor.py" ]; then
        print_error "realtime_monitor.py not found"
        exit 1
    fi

    if [ ! -f "realtime_trainer_pipeline.py" ]; then
        print_error "realtime_trainer_pipeline.py not found"
        exit 1
    fi

    # notification_manager.py is no longer required - self-contained in realtime_monitor.py

    print_success "All required files found"
}

# Setup directories
setup_directories() {
    print_info "Setting up directories..."

    mkdir -p logs
    mkdir -p state
    mkdir -p config
    mkdir -p output_train

    print_success "Directories created"
}

# Setup environment variables
setup_environment() {
    print_info "Setting up environment variables..."

    # Create .env file if not exists
    if [ ! -f ".env" ]; then
        cp .env.production .env
        print_info "Created .env from .env.production"
    fi

    # Check if Telegram token is configured
    if grep -q "YOUR_BOT_TOKEN_HERE" .env; then
        print_error "Please configure Telegram Bot Token in .env file"
        echo ""
        echo "📝 Steps to get Telegram Bot Token:"
        echo "1. Message @BotFather on Telegram"
        echo "2. Send: /newbot"
        echo "3. Give bot name: XGBoost Trading Bot"
        echo "4. Give bot username: xgboost_trading_bot"
        echo "5. Copy the token and update TELEGRAM_BOT_TOKEN in .env"
        echo ""
        echo "📝 Steps to get Chat ID:"
        echo "1. Message your bot to start chat"
        echo "2. Send: /start"
        echo "3. Get chat ID: https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates"
        echo "4. Update TELEGRAM_CHAT_ID in .env"
        exit 1
    fi

    print_success "Environment variables configured"
}

# Create systemd service for monitoring
create_monitor_service() {
    print_info "Creating systemd service for monitoring..."

    cat > /tmp/xgboost-monitor.service << EOF
[Unit]
Description=XGBoost Real-time Monitor
After=network.target mysql.service

[Service]
Type=simple
User=root
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/.xgboost-qc/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=$(pwd)/.xgboost-qc/bin/python realtime_monitor.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    print_info "Systemd service file created at /tmp/xgboost-monitor.service"
    print_info "To install: sudo cp /tmp/xgboost-monitor.service /etc/systemd/system/"
    print_info "To enable: sudo systemctl enable xgboost-monitor"
    print_info "To start: sudo systemctl start xgboost-monitor"
}

# Create automation scheduler
create_scheduler() {
    print_info "Creating automation scheduler..."

    cat > automation_scheduler.py << 'EOF'
#!/usr/bin/env python3
"""
Automation Scheduler for XGBoost Training
Runs training based on schedule and data availability
"""

import os
import json
import logging
import schedule
import time
from datetime import datetime
from pathlib import Path

# Notifications now handled by realtime_monitor.py directly
from realtime_trainer_pipeline import RealtimeTrainerPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/automation_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutomationScheduler:
    """Handles scheduled training and notifications."""

    def __init__(self):
        self.notifier = ModelUpdateNotifier()
        self.trainer = RealtimeTrainerPipeline()
        self.auto_train_enabled = os.getenv('AUTO_TRAIN_ENABLED', 'true').lower() == 'true'

    def check_and_run_training(self):
        """Check if training should run and execute."""
        if not self.auto_train_enabled:
            logger.info("Auto training is disabled")
            return

        try:
            logger.info("🚀 Starting scheduled training run...")

            # Run the training pipeline
            result = self.trainer.run_training_pipeline()

            if result['success']:
                logger.info(f"✅ Training completed: {result['model_path']}")

                # Send notification
                model_info = {
                    'model_path': result['model_path'],
                    'accuracy': result.get('accuracy', 0),
                    'created_at': datetime.now().isoformat(),
                    'session_id': result.get('session_id', 'unknown')
                }

                self.notifier.send_model_ready_notification(model_info)
                logger.info("📱 Notification sent successfully")

            else:
                logger.error(f"❌ Training failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            logger.error(f"💥 Scheduler error: {e}")

    def start_scheduler(self):
        """Start the automation scheduler."""
        schedule_str = os.getenv('TRAINING_SCHEDULE', '0 */6 * * *')

        logger.info(f"📅 Scheduler started with schedule: {schedule_str}")

        # Parse and set schedule (default: every 6 hours)
        schedule.every(6).hours.do(self.check_and_run_training)

        logger.info("⏰ Automation scheduler is running...")
        logger.info("Press Ctrl+C to stop")

        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("🛑 Scheduler stopped by user")

if __name__ == "__main__":
    scheduler = AutomationScheduler()
    scheduler.start_scheduler()
EOF

    chmod +x automation_scheduler.py
    print_success "Automation scheduler created"
}

# Test Telegram connection
test_telegram() {
    print_info "Testing Telegram connection..."

    python3 -c "
# Notifications now handled by realtime_monitor.py directly
import os

try:
    notifier = ModelUpdateNotifier()
    if notifier.telegram_token and notifier.telegram_chat_id:
        message = '🤖 XGBoost Automation System Online!\n\n✅ Monitoring started\n📊 Ready for training'
        notifier.send_telegram_message(message)
        print('✅ Telegram test successful')
    else:
        print('❌ Telegram not configured')
except Exception as e:
    print(f'❌ Telegram test failed: {e}')
"
}

# Main setup process
main() {
    check_prerequisites
    setup_directories
    setup_environment
    create_monitor_service
    create_scheduler
    test_telegram

    echo ""
    print_success "🎉 Automation setup completed!"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Install systemd service:"
    echo "   sudo cp /tmp/xgboost-monitor.service /etc/systemd/system/"
    echo "   sudo systemctl enable xgboost-monitor"
    echo "   sudo systemctl start xgboost-monitor"
    echo ""
    echo "2. Start automation scheduler:"
    echo "   python3 automation_scheduler.py"
    echo ""
    echo "3. Or run both with cron:"
    echo "   crontab -e"
    echo "   Add: */5 * * * * cd $(pwd) && python3 automation_scheduler.py"
    echo ""
    echo "📱 Test Telegram notifications should arrive shortly!"
}

# Run main function
main "$@"