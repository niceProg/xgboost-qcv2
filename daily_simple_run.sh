#!/bin/bash

# Simple script untuk menjalankan daily pipeline XGBoost
# Mode: Daily (hanya hari ini)
# Timezone: WIB
# Trading Hours: 7:00-16:00

echo "========================================"
echo "XGBoost Daily Trading Pipeline"
echo "========================================"

# Default parameters
EXCHANGE=${EXCHANGE:-binance}
PAIR=${PAIR:-BTCUSDT}
INTERVAL=${INTERVAL:-1h}
MODE=${MODE:-daily}
OUTPUT_DIR=${OUTPUT_DIR:-./output_train}

echo "Configuration:"
echo "  Exchange: $EXCHANGE"
echo "  Pair: $PAIR"
echo "  Interval: $INTERVAL"
echo "  Mode: $MODE"
echo "  Output Directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to run a step
run_step() {
    local script=$1
    local step_name=$2

    echo "=========================================="
    echo "Running $step_name..."
    echo "Command: python $script --exchange $EXCHANGE --pair $PAIR --interval $INTERVAL --mode $MODE"
    echo "=========================================="

    if python "$script" \
        --exchange "$EXCHANGE" \
        --pair "$PAIR" \
        --interval "$INTERVAL" \
        --mode "$MODE" \
        --output-dir "$OUTPUT_DIR"; then
        echo "✅ $step_name completed successfully"
        echo ""
    else
        echo "❌ $step_name failed!"
        echo "Pipeline stopped."
        exit 1
    fi
}

# Run each step
run_step "load_database.py" "Step 1: Load Database"
run_step "merge_7_tables.py" "Step 2: Merge Tables"
run_step "feature_engineering.py" "Step 3: Feature Engineering"
run_step "label_builder.py" "Step 4: Label Building"
run_step "xgboost_trainer.py" "Step 5: Model Training"
run_step "model_evaluation_with_leverage.py" "Step 6: Model Evaluation"

echo "=========================================="
echo "✅ Daily Pipeline completed successfully!"
echo "=========================================="

# Show output directory contents
echo ""
echo "Output directory contents:"
ls -la "$OUTPUT_DIR"

# Show model files
echo ""
echo "Trained models:"
ls -la "$OUTPUT_DIR"/*.joblib 2>/dev/null || echo "No model files found"

echo ""
echo "Daily run completed at: $(date)"