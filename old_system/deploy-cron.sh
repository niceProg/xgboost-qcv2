#!/bin/bash

echo "🚀 Setting up XGBoost Cron Pipeline"

# Make scripts executable
chmod +x scripts/cron_monitor.sh
chmod +x daily_runner.py

# Create log directories
sudo mkdir -p /var/log/xgboost
sudo chown $USER:$USER /var/log/xgboost

# Setup logrotate
sudo cp logrotate.conf /etc/logrotate.d/xgboost-cron

# Choose deployment method
echo "Choose deployment method:"
echo "1) System crontab (simple)"
echo "2) Docker cron container"
echo "3) Manual trigger only"
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo "Setting up system crontab..."
        # Add to system crontab
        (crontab -l 2>/dev/null; cat crontab) | crontab -
        echo "✅ Crontab updated. Run 'crontab -l' to verify."
        ;;
    2)
        echo "Starting Docker cron container..."
        docker-compose -f docker-compose.cron.yml up -d xgboost_cron
        echo "✅ Cron container started."
        ;;
    3)
        echo "Manual mode - use: docker-compose run --rm xgboost_pipeline"
        ;;
esac

echo "🎉 Setup complete!"
echo "Check logs: tail -f /var/log/xgboost_cron.log"