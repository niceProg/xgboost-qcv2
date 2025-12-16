#!/bin/bash

# Monitor XGBoost cron jobs
LOG_FILE="/var/log/xgboost_cron.log"
ERROR_THRESHOLD=5

# Check recent errors
error_count=$(grep -c "ERROR\|Failed\|failed" $LOG_FILE)
echo "Recent errors: $error_count"

if [ $error_count -gt $ERROR_THRESHOLD ]; then
    echo "⚠️ Error threshold exceeded!"
    # Send alert ( webhook/email )
    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"🚨 XGBoost pipeline errors detected!"}' \
        YOUR_WEBHOOK_URL
fi

# Check last successful run
last_success=$(grep "✅" $LOG_FILE | tail -1 | awk '{print $1, $2}')
echo "Last successful run: $last_success"

# Check API health
api_health=$(curl -s http://localhost:8000/api/v1/health | jq -r '.status')
echo "API status: $api_health"