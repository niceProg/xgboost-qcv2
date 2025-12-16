#!/bin/bash

# Quick script to run pipeline for today only
# Usage: ./run_today.sh [exchange] [pair] [interval]

EXCHANGE=${1:-binance}
PAIR=${2:-BTCUSDT}
INTERVAL=${3:-1h}

echo "========================================"
echo "XGBoost Pipeline - Today Only"
echo "========================================"
echo "Exchange: $EXCHANGE"
echo "Pair: $PAIR"
echo "Interval: $INTERVAL"
echo "Date: $(date +%Y-%m-%d)"
echo ""

# Run multi-day runner for today only
python multi_day_runner.py \
    --exchange $EXCHANGE \
    --pairs ["$PAIR"] \
    --intervals ["$INTERVAL"] \
    --start-date $(date +%Y-%m-%d) \
    --end-date $(date +%Y-%m-%d)