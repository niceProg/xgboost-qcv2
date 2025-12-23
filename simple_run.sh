#!/bin/bash

# Simple script untuk menjalankan pipeline XGBoost step by step

echo "========================================"
echo "XGBoost Trading Pipeline - Manual Run"
echo "========================================"

# Default parameters
EXCHANGE=${EXCHANGE:-binance}
PAIR=${PAIR:-BTCUSDT}
INTERVAL=${INTERVAL:-1h}
OUTPUT_DIR=${OUTPUT_DIR:-./output_train}

echo "Configuration:"
echo "  Exchange: $EXCHANGE"
echo "  Pair: $PAIR"
echo "  Interval: $INTERVAL"
echo "  Output Directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to run a step
run_step() {
    local script=$1
    local step_name=$2
    shift 2  # Remove first two arguments, remaining are flags
    local extra_flags="$@"  # Capture all remaining flags

    echo "=========================================="
    echo "Running $step_name..."
    echo "Command: python $script $extra_flags --exchange $EXCHANGE --pair $PAIR --interval $INTERVAL --output-dir $OUTPUT_DIR"
    echo "=========================================="

    if python "$script" $extra_flags \
        --exchange "$EXCHANGE" \
        --pair "$PAIR" \
        --interval "$INTERVAL" \
        --output-dir "$OUTPUT_DIR"; then
        echo "✅ $step_name completed successfully"
        echo ""
    else
        echo "❌ $step_name failed!"
        echo "Pipeline stopped."
        exit 1
    fi
}

# Check flags and set mode
EXTRA_FLAGS=""
MODE_SPECIFIED=""

# Check for mode flags
if [[ "$*" == *"--initial"* ]]; then
    echo "🔄 Running with --initial mode (historical data from 2024)"
    EXTRA_FLAGS="--initial"
    MODE_SPECIFIED="yes"
elif [[ "$*" == *"--daily"* ]]; then
    echo "🔄 Running with --daily mode (current day data only)"
    EXTRA_FLAGS="--daily"
    MODE_SPECIFIED="yes"
elif [[ "$*" == *"--days"* ]]; then
    # Extract days value
    DAYS_VALUE=$(echo "$@" | grep -o -- '--days [0-9]*' | awk '{print $2}')
    if [ -n "$DAYS_VALUE" ]; then
        echo "🔄 Running with --days $DAYS_VALUE mode"
        EXTRA_FLAGS="--days $DAYS_VALUE"
        MODE_SPECIFIED="yes"
    fi
elif [[ "$*" == *"--time"* ]]; then
    # Extract time value
    TIME_VALUE=$(echo "$@" | grep -o -- '--time [^ ]*' | awk '{print $2}')
    if [ -n "$TIME_VALUE" ]; then
        echo "🔄 Running with --time $TIME_VALUE mode"
        EXTRA_FLAGS="--time $TIME_VALUE"
        MODE_SPECIFIED="yes"
    fi
fi

if [ "$MODE_SPECIFIED" != "yes" ]; then
    echo "⚠️  No mode specified. Please specify either:"
    echo "   --initial  (for historical data from 2024)"
    echo "   --daily    (for current day data only)"
    echo "   --days N   (for last N days)"
    echo "   --time start,end (for custom time range)"
    echo ""
    echo "Example usage:"
    echo "  $0 --initial"
    echo "  $0 --daily"
    echo "  $0 --days 30"
    echo "  $0 --time 1609459200000,1640995199000"
    echo ""
    exit 1
fi

# Run each step with the appropriate flags
run_step "load_database.py" "Step 1: Load Database" $EXTRA_FLAGS
run_step "merge_7_tables.py" "Step 2: Merge Tables" $EXTRA_FLAGS
run_step "feature_engineering.py" "Step 3: Feature Engineering" $EXTRA_FLAGS
run_step "label_builder.py" "Step 4: Label Building" $EXTRA_FLAGS
run_step "xgboost_trainer.py" "Step 5: Model Training" $EXTRA_FLAGS
run_step "model_evaluation_with_leverage.py" "Step 6: Model Evaluation" $EXTRA_FLAGS


echo "=========================================="
echo "✅ Pipeline completed successfully!"
echo "=========================================="

# Show structured output directory contents
echo ""
echo "Structured Output Directory Contents:"
echo "📁 $OUTPUT_DIR/"
ls -la "$OUTPUT_DIR"

echo ""
echo "📁 Model files:"
if [ -d "$OUTPUT_DIR/models" ]; then
    ls -la "$OUTPUT_DIR/models/"
else
    ls -la "$OUTPUT_DIR"/*.joblib 2>/dev/null || echo "No model files found"
fi

echo ""
echo "📁 Dataset files:"
if [ -d "$OUTPUT_DIR/datasets" ]; then
    echo "  📄 Dataset summary:"
    ls -la "$OUTPUT_DIR/datasets/"*summary*.txt 2>/dev/null || echo "  No dataset summary found"
    echo ""
    echo "  📄 Datasets:"
    ls -la "$OUTPUT_DIR/datasets/"*.parquet 2>/dev/null || echo "  No dataset files found"
fi

echo ""
echo "📁 Feature files:"
if [ -d "$OUTPUT_DIR/features" ]; then
    ls -la "$OUTPUT_DIR/features/" || echo "No feature files found"
fi

echo ""
echo "Next steps:"
echo "1. Deploy real-time system: cd production-v2 && ./deploy.sh"
echo "2. Access structured API: http://localhost:8000/output_train/"
echo "3. View latest model: http://localhost:8000/output_train/models/latest"
echo "4. View dataset summary: http://localhost:8000/output_train/datasets/summary"
echo ""
echo "✅ Pipeline completed with structured output!"
echo "📁 New structure ready for production-v2 API access"