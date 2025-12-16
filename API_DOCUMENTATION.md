# XGBoost Trading Model API v2 Documentation

## Overview
API v2 menyediakan endpoints lengkap untuk mengakses semua output dari XGBoost trading pipeline yang tersimpan di `output_train`.

## Installation & Running
```bash
# Activate virtual environment
source .xgboost-qc/bin/activate

# Install dependencies
pip install fastapi uvicorn

# Run API server
python api_server_v2.py

# API akan berjalan di http://localhost:8000
```

## Endpoints

### 1. Root Endpoint
```
GET /
```
Menampilkan informasi dasar API dan daftar endpoints yang tersedia.

### 2. Output Directory Contents
```
GET /output_train
```
Response:
```json
{
  "path": "./output_train",
  "total_files": 15,
  "file_types": {
    "model": 3,
    "data": 5,
    "metrics": 2,
    "report": 2,
    "config": 2,
    "account_statement": 1
  },
  "files": [
    {
      "name": "latest_model.joblib",
      "path": "latest_model.joblib",
      "size": 78403,
      "modified": "2025-12-16T18:40:37",
      "extension": ".joblib",
      "type": "model"
    },
    ...
  ]
}
```

### 3. Browse Specific Path
```
GET /output_train/browse/{path}
```
Browse direktori atau file spesifik dalam output_train.

### 4. Models Information

#### Get All Models
```
GET /models
```
Menampilkan semua model files dengan detail informasi.

Response:
```json
[
  {
    "name": "latest_model.joblib",
    "path": "latest_model.joblib",
    "size": 78403,
    "modified": "2025-12-16T18:40:37",
    "is_latest": true,
    "feature_count": 54
  },
  {
    "name": "xgboost_trading_model_20251216_184037.joblib",
    "path": "xgboost_trading_model_20251216_184037.joblib",
    "size": 78403,
    "modified": "2025-12-16T18:40:37",
    "is_latest": false,
    "feature_count": 54
  }
]
```

#### Get Latest Model
```
GET /models/latest
```
Menampilkan informasi detail tentang model terbaru.

Response:
```json
{
  "name": "latest_model.joblib",
  "path": "latest_model.joblib",
  "size": 78403,
  "modified": "2025-12-16T18:40:37",
  "is_latest": true,
  "feature_count": 54,
  "feature_names": ["price_open", "price_high", "price_low", ...],
  "n_features": 54,
  "n_classes": 2
}
```

#### Download Model
```
GET /models/download/{model_name}
```
Download file model spesifik.

### 5. Dataset Information

#### Dataset Summary
```
GET /dataset/summary
```
Menampilkan informasi lengkap dataset.

Response:
```json
{
  "total_samples": 17138,
  "feature_count": 54,
  "target_distribution": {
    "bullish": 8691,
    "bearish": 8447
  },
  "feature_list": ["price_open", "price_high", "price_low", ...],
  "last_updated": "2025-12-16T18:40:37"
}
```

#### Dataset Features
```
GET /dataset/features
```
Menampilkan daftar lengkap semua features.

Response:
```json
{
  "total_features": 54,
  "features": [
    "price_open",
    "price_high",
    "price_low",
    "price_close",
    "price_volume_usd",
    "funding_open",
    "funding_high",
    "funding_low",
    ...
  ]
}
```

### 6. Directory Statistics
```
GET /stats
```
Menampilkan statistik lengkap direktori output_train.

Response:
```json
{
  "directory": {
    "path": "./output_train",
    "exists": true,
    "total_size": 50000000,
    "file_count": 15,
    "directory_count": 2
  },
  "files_by_type": {
    "model": 3,
    "data": 5,
    "metrics": 2,
    "report": 2
  },
  "latest_activity": {
    "model": {
      "name": "latest_model.joblib",
      "type": "model",
      "modified": "2025-12-16T18:40:37"
    }
  },
  "summary": {
    "total_files": 15,
    "total_size_mb": 47.68,
    "last_activity": "2025-12-16T18:40:37"
  }
}
```

### 7. File Preview
```
GET /preview/{file_name}?limit=10
```
Preview isi file CSV atau JSON.

Response for CSV:
```json
{
  "file": "trades.csv",
  "type": "csv",
  "columns": ["exchange", "symbol", "interval", "side", "entry_time", ...],
  "preview": [
    {
      "exchange": "Binance",
      "symbol": "BTCUSDT",
      "interval": "1h",
      "side": "Buy",
      "entry_time": "2024-01-01 02:00:00",
      ...
    }
  ],
  "total_rows": 3785
}
```

## Usage Examples

### 1. Get All Output Files
```python
import requests

response = requests.get('http://localhost:8000/output_train')
data = response.json()

print(f"Total files: {data['total_files']}")
for file in data['files']:
    if file['type'] == 'model':
        print(f"Model: {file['name']} ({file['size']} bytes)")
```

### 2. Get Latest Model Info
```python
response = requests.get('http://localhost:8000/models/latest')
latest = response.json()

print(f"Latest model: {latest['name']}")
print(f"Features: {latest['n_features']}")
if latest['feature_names']:
    print(f"First 5 features: {latest['feature_names'][:5]}")
```

### 3. Download Model
```python
import requests

model_name = "latest_model.joblib"
response = requests.get(f'http://localhost:8000/models/download/{model_name}')

if response.status_code == 200:
    with open(model_name, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded {model_name}")
```

### 4. Get Dataset Statistics
```python
response = requests.get('http://localhost:8000/dataset/summary')
summary = response.json()

print(f"Dataset Summary:")
print(f"  Samples: {summary['total_samples']:,}")
print(f"  Features: {summary['feature_count']}")
print(f"  Bullish: {summary['target_distribution']['bullish']} ({summary['target_distribution']['bullish']/summary['total_samples']*100:.1f}%)")
print(f"  Bearish: {summary['target_distribution']['bearish']} ({summary['target_distribution']['bearish']/summary['total_samples']*100:.1f}%)")
```

## File Type Classifications

API mengkategorikan file berdasarkan nama dan ekstensi:

- **model**: `.joblib` files dengan 'model' atau 'latest_model' dalam nama
- **data**: `.parquet`, `.csv` files
- **merged_data**: 'merged_7_tables' files
- **labeled_data**: 'labeled_data' files
- **features**: 'features_engineered', 'X_train', 'y_train' files
- **metrics**: 'performance_metrics', 'training_results' files
- **report**: 'performance_report' files
- **importance**: 'feature_importance' files
- **trading_results**: 'trading_results' files
- **account_statement**: 'rekening_koran' files
- **trade_events**: 'trade_events' files
- **trades**: 'trades' files
- **config**: 'column_mapping', 'feature_list', 'model_features' files
- **summary**: 'dataset_summary' files

## Security Features

1. **Path Validation**: API memvalidasi bahwa semua path akses berada dalam output_train
2. **File Type Check**: Hanya file dengan ekstensi aman yang dapat di-download
3. **Input Sanitization**: Semua input divalidasi sebelum diproses

## Error Handling

API mengembalikan HTTP status codes yang sesuai:
- `200`: Success
- `404`: File/Path not found
- `403`: Access denied (path di luar output_train)
- `500`: Internal server error

## Integration Tips

1. **Real-time Updates**: Use `/stats` endpoint untuk monitoring aktivitas terbaru
2. **Model Management**: Gunakan `/models/latest` untuk selalu mendapatkan model terkini
3. **Data Validation**: Gunakan `/dataset/summary` untuk validasi dataset sebelum training
4. **File Browsing**: Gunakan `/output_train/browse/{path}` untuk navigasi file structure

## Development Notes

- API berjalan pada port 8000 secara default
- Auto-reload enabled untuk development
- CORS enabled untuk cross-origin requests
- Logging aktif untuk debugging
- Error handling komprehensif