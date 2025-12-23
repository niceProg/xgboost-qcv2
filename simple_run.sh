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
echo "Usage Options:"
echo "1. Default (all data in database):"
echo "   ./simple_run.sh"
echo ""
echo "2. Filter by exchange, pair, interval:"
echo "   EXCHANGE=okx PAIR=ETHUSDT INTERVAL=4h ./simple_run.sh"
echo ""
echo "3. Load 30 hari terakhir:"
echo "   ./simple_run.sh --days 30"
echo ""
echo "4. Load data dalam time range tertentu (milliseconds timestamp):"
echo "   ./simple_run.sh --time 1700000000000,1701000000000"
echo ""
echo "5. Combine semua:"
echo "   EXCHANGE=binance PAIR=ETHUSDT INTERVAL=1h ./simple_run.sh --days 60"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to run a step
run_step() {
    local script=$1
    local mode_flags=$2
    local step_name=$3

    echo "=========================================="
    echo "Running $step_name..."
    echo "Command: python $script $mode_flags --exchange $EXCHANGE --pair $PAIR --interval $INTERVAL --output-dir $OUTPUT_DIR"
    echo "=========================================="

    if python $script $mode_flags \
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

# Parse mode flags (optional)
MODE_FLAG=""
if [[ "$*" == *"--days"* ]]; then
    # Extract days value from arguments
    for arg in "$@"; do
        if [[ "$arg" == "--days" ]]; then
            # Next arg is the value
            MODE_FLAG="--days $2"
            echo "🔄 Running with --days mode (last $2 days)"
            break
        elif [[ "$arg" == "--days="* ]]; then
            # Value is in the same arg
            MODE_FLAG="$arg"
            DAYS_VALUE="${arg#--days=}"
            echo "🔄 Running with --days mode (last ${DAYS_VALUE} days)"
            break
        fi
    done
elif [[ "$*" == *"--time"* ]]; then
    # Extract time value from arguments
    for arg in "$@"; do
        if [[ "$arg" == "--time" ]]; then
            # Next arg is the value
            MODE_FLAG="--time $2"
            echo "🔄 Running with --time mode (custom time range in milliseconds)"
            break
        elif [[ "$arg" == "--time="* ]]; then
            # Value is in the same arg
            MODE_FLAG="$arg"
            echo "🔄 Running with --time mode (custom time range in milliseconds)"
            break
        fi
    done
else
    # Default: no time filter (load all data from database)
    echo "🔄 No time filter specified, loading all data from database"
    echo "   (You can specify: --days N or --time start,end)"
fi

# Run each step with the appropriate flags
run_step "load_database.py" "$MODE_FLAG" "Step 1: Load Database"
run_step "merge_7_tables.py" "$MODE_FLAG" "Step 2: Merge Tables"
run_step "feature_engineering.py" "$MODE_FLAG" "Step 3: Feature Engineering"
run_step "label_builder.py" "$MODE_FLAG" "Step 4: Label Building"
run_step "xgboost_trainer.py" "$MODE_FLAG" "Step 5: Model Training"
run_step "model_evaluation_with_leverage.py" "$MODE_FLAG" "Step 6: Model Evaluation"


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
echo "1. Check model in database"
echo "2. Access API for model predictions"
echo ""
echo "✅ Pipeline completed!"
