# XGBoost Trading Model - Manual Execution Guide

## Overview
Ini adalah panduan untuk menjalankan pipeline XGBoost secara manual tanpa Docker.

## Konfigurasi Environment

### 1. Setup .env file
```bash
# Copy template
cp .env.example .env

# Edit .env dengan konfigurasi Anda:
```

### 2. Konfigurasi Database
Edit file `.env` dengan konfigurasi database Anda:

```bash
# Trading Database (untuk load market data)
TRADING_DB_HOST=103.150.81.86
TRADING_DB_PORT=3306
TRADING_DB_USER=newera
TRADING_DB_PASSWORD=WCXscc8twHi3kxDW
TRADING_DB_NAME=newera

# XGBoost Results Database (untuk menyimpan hasil)
DB_HOST=103.150.81.86
DB_PORT=3306
DB_USER=xgboostqc
DB_PASSWORD=6SPxBDwXH6WyxpfT
DB_NAME=xgboostqc

# Pipeline Configuration
ENABLE_DB_STORAGE=true
EXCHANGE=binance
PAIR=BTCUSDT
INTERVAL=1h
TRADING_HOURS=7:00-16:00
TIMEZONE=WIB  # Waktu Indonesia Barat

# Model Evaluation
LEVERAGE=10
MARGIN_FRACTION=0.2
INITIAL_CASH=1000
THRESHOLD=0.5

# Output
OUTPUT_DIR=./output_train
```

## Phase 1: Initial Setup (Training dari 2024)

Jalankan setiap step secara berurutan:

### Option 1: Manual step-by-step
```bash
# Step 1: Load data dari database
python load_database.py --exchange binance --pair BTCUSDT --interval 1h --mode initial --timezone WIB

# Step 2: Merge tabel
python merge_7_tables.py --exchange binance --pair BTCUSDT --interval 1h --mode initial --timezone WIB

# Step 3: Feature engineering
python feature_engineering.py --exchange binance --pair BTCUSDT --interval 1h --mode initial --timezone WIB

# Step 4: Build labels
python label_builder.py --exchange binance --pair BTCUSDT --interval 1h --mode initial --timezone WIB

# Step 5: Train model
python xgboost_trainer.py --exchange binance --pair BTCUSDT --interval 1h --mode initial --timezone WIB

# Step 6: Evaluate model
python model_evaluation_with_leverage.py --exchange binance --pair BTCUSDT --interval 1h --mode initial --timezone WIB
```

### Option 2: Gunakan script
```bash
# Run dengan script helper
./simple_run.sh

# Atau dengan environment variables:
EXCHANGE=binance PAIR=BTCUSDT INTERVAL=1h MODE=initial TIMEZONE=WIB ./simple_run.sh
```

## Phase 2: Daily Operations

Setelah initial setup selesai, jalankan daily pipeline:

### Manual Daily Run
```bash
# Jalankan untuk data hari ini (jam 7-4 sore)
python run_daily_pipeline.py --mode daily --trading-hours 7:00-16:00 --timezone WIB

# Atau dengan parameters:
python run_daily_pipeline.py --mode daily --exchange binance --pair BTCUSDT --interval 1h
```

### Auto-Scheduler (Optional)
```bash
# Start scheduler yang auto-run jam 7 pagi
python scheduler.py

# Check scheduler status
python scheduler.py --status
```

## API Server

Setelah training selesai, jalankan API server untuk mengakses hasil:

```bash
# Start API server
python api_server.py

# API akan tersedia di:
# http://localhost:8000/api/v1/
```

## API Endpoints

### Model Access
- `GET /api/v1/models/latest` - Model terbaru
- `GET /api/v1/models/history` - Semua model
- `POST /api/v1/predict` - Buat prediksi

### Example Predict Request
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "price_close_return_1": 0.02,
      "funding_close": 0.0001,
      "basis_close_basis": 0.01
    },
    "threshold": 0.5
  }'
```

### File Access
- `GET /api/v1/files` - List semua file di output_train
- `GET /api/v1/download/{filename}` - Download file
- `GET /api/v1/file/view/{filename}` - View file content

### Performance
- `GET /api/v1/performance/{session_id}` - Metrics performa

## Output Directory Structure

```
output_train/
├── merged_7_tables.parquet          # Data merged
├── features_engineered.parquet      # Features
├── labeled_data.parquet             # Data dengan labels
├── X_train_features.parquet         # Training features
├── y_train_labels.parquet           # Training labels
├── xgboost_trading_model_*.joblib   # Trained models
├── latest_model.joblib              # Latest model symlink
├── performance_metrics_*.json       # Performance metrics
├── performance_report_*.json        # Detailed reports
├── feature_importance_*.csv         # Feature importance
├── rekening_koran.csv               # Equity statement
├── trades.csv                       # Trade log
├── trade_events.csv                 # Trade events
└── performance_analysis.png         # Performance plots
```

## Troubleshooting

### Database Connection Error
```bash
# Check database connection
python -c "
import pymysql
conn = pymysql.connect(
    host='103.150.81.86',
    port=3306,
    user='newera',
    password='WCXscc8twHi3kxDW',
    database='newera'
)
print('Connection successful!')
"
```

### Missing Dependencies
```bash
# Install required packages
pip install -r requirements.txt
pip install -r requirements_db.txt
```

### Timezone Issues
```bash
# Check WIB timezone
python -c "
import pytz
from datetime import datetime
tz = pytz.timezone('Asia/Jakarta')
now = datetime.now(tz)
print(f'Current WIB time: {now}')
"
```

### Model Loading Error
```bash
# Check model files
ls -la output_train/*.joblib

# Verify model
python -c "
import joblib
model = joblib.load('output_train/latest_model.joblib')
print(f'Model loaded: {type(model)}')
"
```

## Next Steps for QuantConnect Integration

1. Deploy API server di server yang dapat diakses
2. Gunakan API client untuk mengakses model:
   ```python
   import requests

   response = requests.get('http://your-server:8000/api/v1/models/latest')
   model_info = response.json()
   ```

3. Implementasikan di QuantConnect algorithm untuk trading real-time

## Notes

- Timezone menggunakan WIB (UTC+7)
- Trading hours default: 07:00 - 16:00 WIB
- Semua hasil tersimpan di database `xgboostqc`
- Data market diambil dari database `newera`
- Output files tersimpan di `./output_train`