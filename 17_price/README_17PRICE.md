# 17 Price Features Training

XGBoost model training dengan **hanya 17 price features** yang tersedia di QuantConnect.

**SIMPLE & CLEAN - No 9-table merge!**

---

## FILES

| File | Description |
|------|-------------|
| `load_price_only.py` | Load price data dari database (1 table only!) |
| `feature_engineering_17price.py` | Feature engineering untuk 17 price features |
| `label_builder_17price.py` | Label builder untuk binary classification |
| `train_model_17price.py` | Main training script |
| `run_all_17price.py` | Run complete pipeline (2 steps only!) |
| `requirements_17price.txt` | Python dependencies |

---

## PIPELINE (Simple!)

```
┌─────────────────────────────────────────────────────────┐
│  STEP 1: Load Price Data                                │
│  Script: load_price_only.py                             │
│  Table: cg_futures_price_history (1 table only!)        │
│  Output: data.csv + parquet                             │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  STEP 2: Train XGBoost Model                            │
│  Script: train_model_17price.py                         │
│  Features: 17 price features (OHLCV derived)            │
│  Output: Model + Metadata + API Format                  │
└─────────────────────────────────────────────────────────┘
```

**TIDAK ADA:**
- No 9-table merge
- No complex feature engineering
- No external dependencies (except database)

---

## 17 PRICE FEATURES

```
1.  price_open
2.  price_high
3.  price_low
4.  price_close
5.  price_volume_usd
6.  price_close_return_1
7.  price_close_return_5
8.  price_log_return
9.  price_rolling_vol_5
10. price_true_range
11. price_close_mean_5
12. price_close_std_5
13. price_volume_mean_10
14. price_volume_zscore
15. price_volume_change
16. price_wick_upper
17. price_wick_lower
18. price_body_size
```

**Note:** Semua features ini bisa dihitung dari OHLCV data yang tersedia di QuantConnect!

---

## CARA PAKAI

### 1. Install Dependencies

```bash
cd 17_price
pip install -r requirements_17price.txt
```

### 2. Run Complete Pipeline

```bash
# Default settings
python run_all_17price.py

# With filters
python run_all_17price.py --symbol BTC --interval 1h --days 30

# With custom model parameters
python run_all_17price.py --symbol BTC --interval 1h --estimators 200 --depth 6 --lr 0.05
```

### 3. Run Individual Steps

```bash
# Step 1: Load data only
python load_price_only.py --symbol BTC --interval 1h --days 30

# Step 2: Train model only (requires data.csv)
python train_model_17price.py --label next_bar --estimators 200
```

### 4. Output Files

Training akan menghasilkan:

```
output_train_17price/
├── data.csv                                          # Raw OHLCV data
├── datasets/raw/cg_futures_price_history.parquet    # Parquet format
├── xgboost_model_17price_YYYYMMDD_HHMMSS.joblib    # Model file
├── latest_model.joblib                               # Latest model
├── model_features_17price.txt                        # Feature names
├── metadata_17price_YYYYMMDD_HHMMSS.json           # Training metadata
├── api_format_17price_YYYYMMDD_HHMMSS.json         # API format (for upload)
└── dataset_summary_17price_YYYYMMDD_HHMMSS.json    # Dataset summary
```

---

## CONFIGURATION

### Label Types

| Type | Description |
|------|-------------|
| `next_bar` | Next bar direction (1 = up, 0 = down) |
| `threshold` | Threshold-based (e.g., >0.5% = buy, <-0.5% = sell) |
| `3bar` | Multi-bar dengan profit target |

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 200 | Number of trees |
| `max_depth` | 6 | Maximum tree depth |
| `learning_rate` | 0.05 | Learning rate |
| `subsample` | 0.8 | Subsample ratio |
| `colsample_bytree` | 0.8 | Feature sampling |

---

## INTEGRASI DENGAN QUANTCONNECT

Setelah training selesai:

1. **Upload model ke ObjectStore** QuantConnect
2. **Update `qc_futures.py`** untuk pakai model baru:

```python
# Di qc_futures.py, update feature list:
self.model_features = FEATURES_17  # Dari feature_engineering_17price.py
```

---

## METRICS YANG DI-OPTIMISASI

| Metric | Target | Description |
|--------|--------|-------------|
| Accuracy | > 55% | Overall correctness |
| Precision | > 55% | True positives / Predicted positives |
| Recall | > 50% | True positives / Actual positives |
| F1 Score | > 0.50 | Harmonic mean of precision & recall |
| ROC AUC | > 0.55 | Area under ROC curve |

---

## RETRAINING

Retrain model secara berkala (misalnya weekly/bulanan):

```bash
# Weekly retrain
python run_all_17price.py --label threshold --estimators 300 --days 30

# Save dengan timestamp baru untuk versioning
```

---

## TROUBLESHOOTING

### Error: "Data file not found"

```bash
# Pastikan data.csv ada (dari load_price_only.py)
ls -la data.csv

# Atau run complete pipeline
python run_all_17price.py
```

### Error: "Module not found"

```bash
pip install -r requirements_17price.txt
```

### Poor Performance

- Coba label type yang berbeda: `--label threshold`
- Tambah estimators: `--estimators 500`
- Adjust learning rate: `--lr 0.03`
- Add more data: `--days 60`

---

## NOTES

- **Price Only:** Pipeline ini hanya menggunakan 1 table (price history)
- **QuantConnect Ready:** 17 features fully compatible dengan QuantConnect data
- **No Merge:** Tidak ada merging 9 tables seperti futures full training
- **Simple:** Langsung dari database → training → model

---

Created: 2024-12-28
For: DragonFortune Futures Trading
