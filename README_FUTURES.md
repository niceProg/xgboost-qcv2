# XGBoost Futures Training Pipeline

Pipeline training XGBoost untuk futures trading dengan data dari Coinglass.

## Overview

Pipeline ini mengambil data dari 9 tabel futures di database, melakukan feature engineering, dan melatih model XGBoost untuk prediksi tren harga.

## Data Sources (9 Tables)

| Tabel | Prefix | Deskripsi |
|-------|--------|-----------|
| `cg_futures_price_history` | `price` | OHLCV price data (Base Table - WAJIB) |
| `cg_funding_rate_history` | `funding` | Funding rate |
| `cg_futures_basis_history` | `basis` | Basis futures-spot |
| `cg_open_interest_aggregated_history` | `oi` | Open Interest |
| `cg_liquidation_aggregated_history` | `liq` | Liquidation data |
| `cg_futures_aggregated_taker_buy_sell_volume_history` | `taker` | Taker volume |
| `cg_futures_aggregated_ask_bids_history` | `ob` | Orderbook depth |
| `cg_long_short_global_account_ratio_history` | `ls_global` | L/S ratio global |
| `cg_long_short_top_account_ratio_history` | `ls_top` | L/S ratio top trader |

## Pipeline Steps

```
1. load_database_futures.py     → Load data dari DB ke parquet
2. merge_futures_9_tables.py    → Merge 9 tabel menjadi 1
3. feature_engineering_futures.py  → Buat fitur ML
4. label_builder_futures.py     → Buat label binary (0/1)
5. xgboost_trainer_futures.py   → Train model XGBoost
```

## Quick Start

### Run All Steps (Recommended)

```bash
# Run dengan default settings (semua data)
python run_all_futures.py

# Run dengan filter
python run_all_futures.py --exchange binance --symbol BTC --interval 1h --days 30

# Run dengan time range
python run_all_futures.py --symbol BTC,ETH --interval 1h --time 1704067200000,1706745600000
```

### Run Individual Steps

```bash
# Step 1: Load data dari database
python load_database_futures.py --exchange binance --symbol BTC --interval 1h --days 30

# Step 2: Merge tables
python merge_futures_9_tables.py

# Step 3: Feature engineering
python feature_engineering_futures.py

# Step 4: Build labels
python label_builder_futures.py

# Step 5: Train model
python xgboost_trainer_futures.py
```

## Command Line Options

| Option | Deskripsi | Contoh |
|--------|-----------|--------|
| `--exchange` | Filter exchange | `--exchange binance` |
| `--pair` | Filter trading pair | `--pair BTCUSDT` |
| `--symbol` | Filter symbol | `--symbol BTC` |
| `--interval` | Filter timeframe | `--interval 1h` |
| `--days` | Jumlah hari ke belakang | `--days 30` |
| `--time` | Time range (ms) | `--time 1704067200000,1706745600000` |
| `--output-dir` | Output directory | `--output-dir ./output_train_futures` |
| `--verbose` | Verbose logging | `--verbose` |

## Output Files

Setelah pipeline selesai, output files akan ada di `{output_dir}/`:

```
output_train_futures/
├── datasets/
│   ├── raw/                          # Raw data dari DB
│   │   ├── cg_futures_price_history.parquet
│   │   ├── cg_funding_rate_history.parquet
│   │   └── ...
│   ├── merged_futures_9_tables.parquet
│   ├── features_engineered_futures.parquet
│   ├── summary/
│   │   └── qc_summary_futures_*.txt   # Summary untuk QuantConnect
│   ├── X_train_features_futures.parquet
│   └── y_train_labels_futures.parquet
├── models/
│   ├── xgboost_futures_model_*.joblib
│   └── latest_futures_model.joblib    # Model terbaru
├── features/
│   └── feature_list_futures.txt
├── model_features_futures.txt
├── training_results_futures_*.json
└── feature_importance_futures_*.csv
```

## Features

Pipeline ini membuat fitur-fitur berikut:

### Price Features (14 fitur)
- Returns (1, 5), Log return, Volatility
- True range, Rolling mean/std, Volume z-score
- Candlestick: wick upper/lower, body size

### Funding Rate Features (6 fitur)
- Normalized funding, Rolling mean/std, Z-score
- Extreme positive/negative indicators

### Basis Features (5 fitur)
- Delta, drift, Rolling mean/std
- Z-score, Volatility

### Open Interest Features (5 fitur) - *NEW*
- Change (abs & pct), Price correlation
- Momentum, Z-score

### Liquidation Features (9 fitur) - *NEW*
- Total, Ratio, Imbalance
- Rolling means, Spike detection
- Long/Short z-scores

### Taker Volume Features (9 fitur)
- Buy ratio, Imbalance, Rolling stats
- Z-scores, Momentum

### Orderbook Features (8 fitur) - *NEW*
- Bid/Ask ratio, Imbalance
- Pressure indicators, Depth change

### Long/Short Ratio Features (7 fitur)
- Global & Top ratios, Z-scores
- Delta, Extreme indicators, Top vs Global

### Cross-Table Features (7 fitur)
- Funding * OI, Funding * Price
- Liquidation * Price, OI * Taker
- Orderbook * Price, LS * Price, Liquidation * Funding

**Total: ~70+ fitur**

## Requirements

```bash
pip install pandas numpy xgboost scikit-learn pyarrow pymysql sqlalchemy python-dotenv
```

## Environment Variables

Buat file `.env` dengan konfigurasi database:

```bash
TRADING_DB_HOST=your_host
TRADING_DB_PORT=3306
TRADING_DB_NAME=your_database
TRADING_DB_USER=your_user
TRADING_DB_PASSWORD=your_password
```

## QuantConnect Integration

Setelah training selesai, gunakan file-file berikut untuk backtest di QuantConnect:

1. **Model**: `latest_futures_model.joblib`
2. **Summary**: `datasets/summary/qc_summary_futures_*.txt`
3. **Features**: `model_features_futures.txt`

## Troubleshooting

### Error: Required table not found
Pastikan data sudah ada di database. Cek tabel yang tersedia di `data.sql`.

### Error: No data loaded
Pastikan filter yang digunakan sesuai dengan data di database. Coba tanpa filter dulu.

### Error: Memory issues
Kurangi `--days` atau gunakan filter symbol/interval yang lebih spesifik.

## License

MIT License
