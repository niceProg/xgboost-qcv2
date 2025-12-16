#!/bin/bash

echo "⏰ Deploying XGBoost Daily Training Cron System"
echo "==============================================="

# Check requirements
echo "🔍 Checking requirements..."

# Check database credentials
required_vars=("TRADING_DB_HOST" "TRADING_DB_USER" "TRADING_DB_PASSWORD" "TRADING_DB_NAME")
missing_vars=()

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -gt 0 ]; then
    echo "❌ Missing environment variables: ${missing_vars[*]}"
    exit 1
fi

# Create directories
echo "📁 Creating directories..."
mkdir -p logs/cron logs/api logs/cron backups state/cron
mkdir -p state/monitoring
mkdir -p output_train

# Setup timezone
echo "🌍 Setting timezone..."
sudo timedatectl set-timezone Asia/Jakarta
sudo ln -sf /usr/share/zoneinfo/Asia/Jakarta /etc/localtime

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose -f docker-compose-prod-cron.yml down

# Build images
echo "🔨 Building Docker images..."
docker-compose -f docker-compose-prod-cron.yml build

# Start cron services
echo "⏰ Starting cron services..."
docker-compose -f docker-compose-prod-cron.yml up -d xgboost_cron backup_service log_monitor

# Wait for cron to initialize
echo "⏳ Waiting for cron to initialize..."
sleep 10

# Show cron jobs
echo "📋 Active cron jobs:"
docker exec xgboost_cron crontab -l

# Manual run for first time
echo "🚀 Running initial daily training..."
docker-compose -f docker-compose-prod-cron.yml run --rm xgboost_daily_runner

# Wait for initial training to complete
echo "⏳ Waiting for initial training to complete..."
echo "(This may take 10-30 minutes depending on data size)"

# Monitor logs
echo "📊 Monitoring training progress..."
docker logs -f xgboost_daily_runner &
MONITOR_PID=$!

# Check if training completed
while true; do
    if docker exec xgboost_daily_runner test -f /app/output_train/xgboost_trading_model_*.joblib > /dev/null 2>&1; then
        echo "✅ Initial training completed successfully!"
        break
    fi

    # Check if there are any errors
    if docker exec xgboost_daily_runner test -f /app/logs/daily_runner.log && grep -q "ERROR" /app/logs/daily_runner.log; then
        echo "❌ Training completed with errors. Check logs:"
        docker logs xgboost_daily_runner
        break
    fi

    sleep 30
done

# Kill monitor
kill $MONITOR_PID 2>/dev/null

# Run evaluation
echo "📊 Running model evaluation..."
docker-compose -f docker-compose-prod-cron.yml run --rm xgboost_evaluator

# Setup log rotation
echo "📋 Setting up log rotation..."
cat > /etc/cron.d/xgboost-logrotate << EOF
# XGBoost Log Rotation
0 1 * * * root find /app/logs/cron -name "*.log" -mtime +7 -delete
0 2 * * 1 root tar -czf /app/backups/logs_$(date +\%Y\%m\%d).tar.gz /app/logs/cron/* && find /app/logs/cron -name "*.log" -delete
EOF

# Create status dashboard
echo "📊 Creating status dashboard..."
cat > status-cron.sh << 'EOF'
#!/bin/bash

echo "📊 XGBoost Cron System Status"
echo "==========================="
echo "Time: $(date)"
echo ""

echo "🏥 Container Status:"
docker-compose -f docker-compose-prod-cron.yml ps
echo ""

echo "📋 Recent Cron Jobs:"
docker exec xgboost_cron tail -n 5 /var/log/cron/cron
echo ""

echo "📈 Latest Model:"
ls -la ./output_train/xgboost_trading_model_*.joblib | tail -1
echo ""

echo "📊 Model Performance:"
if [ -f "./output_train/model_performance.json" ]; then
    python -c "
import json
with open('./output_train/model_performance.json', 'r') as f:
    data = json.load(f)
    print(f\"Latest AUC: {data.get('latest_auc', 'N/A')}\")
    print(f\"Latest Accuracy: {data.get('latest_accuracy', 'N/A')}\")
    print(f\"Total Updates: {data.get('updates', 'N/A')}\")
"
else
    echo "No performance data available"
fi
echo ""

echo "📋 Recent Logs:"
echo "Daily Runner:"
docker logs --tail 5 xgboost_daily_runner 2>/dev/null || echo "No logs available"
echo ""
echo "Cron:"
docker exec xgboost_cron tail -n 3 /var/log/cron/daily_runner.log 2>/dev/null || echo "No cron logs"
EOF

chmod +x status-cron.sh

# Create management scripts
echo "🔧 Creating management scripts..."

# Manual runner script
cat > run-daily.sh << 'EOF'
#!/bin/bash
echo "🚀 Running daily training manually..."
docker-compose -f docker-compose-prod-cron.yml run --rm xgboost_daily_runner
echo "✅ Daily training completed"
EOF

chmod +x run-daily.sh

# View logs script
cat > view-logs.sh << 'EOF'
#!/bin/bash
echo "📋 XGBoost Cron Logs"
echo "==================="

echo "Choose logs to view:"
echo "1) Daily Runner"
echo "2) Real-time Trainer"
echo "3) Evaluator"
echo "4) System Logs"

read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo "📊 Daily Runner Logs:"
        docker logs -f xgboost_daily_runner
        ;;
    2)
        echo "⚡ Real-time Trainer Logs:"
        docker logs -f xgboost_realtime_trainer
        ;;
    3)
        echo "📈 Evaluator Logs:"
        docker logs -f xgboost_evaluator
        ;;
    4)
        echo "🔧 System Logs:"
        docker logs -f xgboost_cron
        ;;
    *)
        echo "Invalid choice"
        ;;
esac
EOF

chmod +x view-logs.sh

# Show final status
echo ""
echo "🎉 XGBoost Cron System Deployed!"
echo "================================"
echo ""
echo "⏰ Cron Schedule (Asia/Jakarta Time):"
echo "- Daily Training: 02:00 AM (2 AM)"
echo "- Real-time Updates: 06:00 AM, 12:00 PM, 06:00 PM"
echo "- Upload Reminder: 03:00 AM"
echo "- Backup: 04:00 AM"
echo ""
echo "📁 Important Files:"
echo "- Daily Runner: daily_runner.py"
echo "- Config: daily_config.json"
echo "- Output: ./output_train/"
echo "- Logs: ./logs/cron/"
echo "- Backups: ./backups/"
echo ""
echo "🔧 Management Commands:"
echo "- Run manually: ./run-daily.sh"
echo "- View status: ./status-cron.sh"
echo "- View logs: ./view-logs.sh"
echo "- Check jobs: docker exec xgboost_cron crontab -l"
echo "- Stop system: docker-compose -f docker-compose-prod-cron.yml down"
echo "- Restart: docker-compose -f docker-compose-prod-cron.yml restart"
echo ""
echo "📊 Monitoring:"
echo "- Model files: ls -la ./output_train/*.joblib"
echo "- Performance: cat ./output_train/model_performance.json"
echo "- Recent trades: tail ./output_train/trade_events.csv"
echo ""
echo "✅ Daily training system is ready!"

# Display next run time
echo ""
echo "⏰ Next scheduled runs:"
echo "- Daily training: $(date -d 'today 02:00' '+%Y-%m-%d %H:%M:%S')"
echo "- First realtime update: $(date -d 'today 06:00' '+%Y-%m-%d %H:%M:%S')"