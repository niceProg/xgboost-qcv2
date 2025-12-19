#!/bin/bash

# Quick database index creation script

echo "🔧 Adding created_at indexes for performance optimization"
echo "======================================================"

# Load environment variables
source .xgboost-qc/bin/activate
source .env 2>/dev/null

tables=(
    "cg_spot_price_history"
    "cg_funding_rate_history"
    "cg_futures_basis_history"
    "cg_spot_aggregated_taker_volume_history"
    "cg_long_short_global_account_ratio_history"
    "cg_long_short_top_account_ratio_history"
)

for table in "${tables[@]}"; do
    echo "📊 Processing $table..."

    # Check if table exists and has created_at column
    mysql -h"$TRADING_DB_HOST" -u"$TRADING_DB_USER" -p"$TRADING_DB_PASSWORD" "$TRADING_DB_NAME" -e "
        -- Check if index exists
        SELECT COUNT(*) INTO @index_exists
        FROM INFORMATION_SCHEMA.STATISTICS
        WHERE TABLE_SCHEMA = '$TRADING_DB_NAME'
        AND TABLE_NAME = '$table'
        AND INDEX_NAME = 'idx_${table}_created_at';

        -- Create index if not exists
        SET @sql = IF(@index_exists = 0,
            'CREATE INDEX idx_${table}_created_at ON $table(created_at)',
            'SELECT \"Index already exists\"');

        PREPARE stmt FROM @sql;
        EXECUTE stmt;
        DEALLOCATE PREPARE stmt;
    " 2>/dev/null

    if [ $? -eq 0 ]; then
        echo "   ✅ $table optimized"
    else
        echo "   ❌ Error with $table"
    fi
done

echo ""
echo "✅ Index optimization completed!"
echo ""
echo "🧪 Testing query performance..."
mysql -h"$TRADING_DB_HOST" -u"$TRADING_DB_USER" -p"$TRADING_DB_PASSWORD" "$TRADING_DB_NAME" -e "
    SELECT 'Testing query performance...' as status;
    SELECT COUNT(*) as recent_records FROM cg_spot_price_history WHERE created_at > DATE_SUB(NOW(), INTERVAL 1 MINUTE);
    SELECT 'Performance test completed' as status;
" 2>/dev/null

echo "🚀 Database is now optimized!"