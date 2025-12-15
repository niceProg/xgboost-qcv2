#!/bin/bash

# Docker entrypoint script for XGBoost trading pipeline

set -e

echo "======================================"
echo "XGBoost Trading Pipeline Docker"
echo "======================================"

# Check if database is ready
echo "Checking database connection..."

# Wait for database to be ready
if [ -n "$DB_HOST" ]; then
    echo "Waiting for trading database at $DB_HOST:$DB_PORT..."
    while ! python -c "import pymysql; pymysql.connect(host='$DB_HOST', port=$DB_PORT, user='$DB_USER', password='$DB_PASSWORD', database='$DB_NAME')" 2>/dev/null; do
        echo "Database not ready, waiting..."
        sleep 2
    done
    echo "Trading database is ready!"
fi

if [ -n "$RESULTS_DB_HOST" ]; then
    echo "Waiting for results database at $RESULTS_DB_HOST:$RESULTS_DB_PORT..."
    while ! python -c "import pymysql; pymysql.connect(host='$RESULTS_DB_HOST', port=$RESULTS_DB_PORT, user='$RESULTS_DB_USER', password='$RESULTS_DB_PASSWORD', database='$RESULTS_DB_NAME')" 2>/dev/null; do
        echo "Results database not ready, waiting..."
        sleep 2
    done
    echo "Results database is ready!"
fi

# Create output directory
mkdir -p /app/output_train

# Check mode
MODE=${PIPELINE_MODE:-daily}

echo "Running in $MODE mode"

if [ "$MODE" = "initial" ]; then
    echo "Running initial setup (training from 2024)..."
    python initial_setup.py
elif [ "$MODE" = "daily" ]; then
    echo "Running daily pipeline..."
    python run_daily_pipeline.py --mode daily
elif [ "$MODE" = "api" ]; then
    echo "Starting API server..."
    python api_examples.py
elif [ "$MODE" = "scheduler" ]; then
    echo "Starting scheduler..."
    python scheduler.py
else
    echo "Unknown mode: $MODE"
    echo "Available modes: initial, daily, api, scheduler"
    exit 1
fi