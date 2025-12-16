(.xgboost-qc) ➜  xgboost-qc git:(docker) ✗ ./simple_run.sh
========================================
XGBoost Trading Pipeline - Manual Run
========================================
Configuration:
  Exchange: binance
  Pair: BTCUSDT
  Interval: 1h
  Mode: initial
  Timezone: WIB
  Trading Hours: 7:00-16:00
  Output Directory: ./output_train

==========================================
Running Step 1: Load Database...
Command: python load_database.py --exchange binance --pair BTCUSDT --interval 1h --mode initial --timezone WIB
==========================================
=== Active Filters ===
Mode: initial
Exchange(s): binance
Pair(s): BTCUSDT
Interval(s): 1h
Trading Hours: 07:00 - 16:00
Timezone: WIB
Time Range: 2024-01-01 00:00:00 to 2025-12-16 07:29:17.922000
====================
2025-12-16 07:29:18,389 - INFO - Database initialized successfully
2025-12-16 07:29:18,550 - INFO - Created training session: 20251216_072917
2025-12-16 07:29:18,551 - INFO - Created training session: 20251216_072917
2025-12-16 07:29:18,551 - INFO - Attempting to load data from database...
2025-12-16 07:29:18,551 - INFO - Loading data from cg_spot_price_history...
2025-12-16 07:29:18,552 - INFO - Successfully connected to MySQL database
2025-12-16 07:29:19,603 - INFO - Loaded 17139 rows from cg_spot_price_history
2025-12-16 07:29:19,603 - INFO - Loading data from cg_funding_rate_history...
2025-12-16 07:29:19,604 - INFO - Successfully connected to MySQL database
2025-12-16 07:29:20,582 - INFO - Loaded 17146 rows from cg_funding_rate_history
2025-12-16 07:29:20,582 - INFO - Loading data from cg_futures_basis_history...
2025-12-16 07:29:20,583 - INFO - Successfully connected to MySQL database
2025-12-16 07:29:21,561 - INFO - Loaded 17139 rows from cg_futures_basis_history
2025-12-16 07:29:21,562 - INFO - Loading data from cg_long_short_global_account_ratio_history...
2025-12-16 07:29:21,562 - INFO - Successfully connected to MySQL database
2025-12-16 07:29:22,435 - INFO - Loaded 17146 rows from cg_long_short_global_account_ratio_history
2025-12-16 07:29:22,435 - INFO - Loading data from cg_long_short_top_account_ratio_history...
2025-12-16 07:29:22,436 - INFO - Successfully connected to MySQL database
2025-12-16 07:29:23,399 - INFO - Loaded 17146 rows from cg_long_short_top_account_ratio_history
2025-12-16 07:29:23,399 - INFO - 
=== Summary ===
2025-12-16 07:29:23,399 - INFO - Successfully loaded data from 5 tables:
2025-12-16 07:29:23,399 - INFO - 
=== Data Quality Report for cg_spot_price_history ===
2025-12-16 07:29:23,399 - INFO - Total rows: 17139
2025-12-16 07:29:23,400 - INFO - Date range: 1704067200000 to 1765764000000
2025-12-16 07:29:23,406 - INFO - No missing values found
2025-12-16 07:29:23,412 - INFO - No duplicate rows found
2025-12-16 07:29:23,412 - INFO - ==================================================
2025-12-16 07:29:23,452 - INFO - Saved 17139 rows to output_train/cg_spot_price_history.parquet
2025-12-16 07:29:23,573 - INFO - Copied file to: output_train/20251216_072917/raw_data/cg_spot_price_history.parquet
2025-12-16 07:29:23,573 - INFO - File stored to database: output_train/20251216_072917/raw_data/cg_spot_price_history.parquet
2025-12-16 07:29:23,574 - INFO - 
=== Data Quality Report for cg_funding_rate_history ===
2025-12-16 07:29:23,574 - INFO - Total rows: 17146
2025-12-16 07:29:23,574 - INFO - Date range: 1704067200000 to 1765789200000
2025-12-16 07:29:23,584 - INFO - No missing values found
2025-12-16 07:29:23,596 - INFO - No duplicate rows found
2025-12-16 07:29:23,596 - INFO - ==================================================
2025-12-16 07:29:23,622 - INFO - Saved 17146 rows to output_train/cg_funding_rate_history.parquet
2025-12-16 07:29:23,728 - INFO - Copied file to: output_train/20251216_072917/raw_data/cg_funding_rate_history.parquet
2025-12-16 07:29:23,729 - INFO - File stored to database: output_train/20251216_072917/raw_data/cg_funding_rate_history.parquet
2025-12-16 07:29:23,729 - INFO - 
=== Data Quality Report for cg_futures_basis_history ===
2025-12-16 07:29:23,729 - INFO - Total rows: 17139
2025-12-16 07:29:23,729 - INFO - Date range: 1704067200000 to 1765764000000
2025-12-16 07:29:23,741 - INFO - No missing values found
2025-12-16 07:29:23,757 - INFO - No duplicate rows found
2025-12-16 07:29:23,757 - INFO - ==================================================
2025-12-16 07:29:23,780 - INFO - Saved 17139 rows to output_train/cg_futures_basis_history.parquet
2025-12-16 07:29:23,887 - INFO - Copied file to: output_train/20251216_072917/raw_data/cg_futures_basis_history.parquet
2025-12-16 07:29:23,888 - INFO - File stored to database: output_train/20251216_072917/raw_data/cg_futures_basis_history.parquet
2025-12-16 07:29:23,888 - INFO - 
=== Data Quality Report for cg_long_short_global_account_ratio_history ===
2025-12-16 07:29:23,888 - INFO - Total rows: 17146
2025-12-16 07:29:23,888 - INFO - Date range: 1704067200000 to 1765789200000
2025-12-16 07:29:23,897 - INFO - No missing values found
2025-12-16 07:29:23,907 - INFO - No duplicate rows found
2025-12-16 07:29:23,907 - INFO - ==================================================
2025-12-16 07:29:23,933 - INFO - Saved 17146 rows to output_train/cg_long_short_global_account_ratio_history.parquet
2025-12-16 07:29:24,039 - INFO - Copied file to: output_train/20251216_072917/raw_data/cg_long_short_global_account_ratio_history.parquet
2025-12-16 07:29:24,039 - INFO - File stored to database: output_train/20251216_072917/raw_data/cg_long_short_global_account_ratio_history.parquet
2025-12-16 07:29:24,040 - INFO - 
=== Data Quality Report for cg_long_short_top_account_ratio_history ===
2025-12-16 07:29:24,040 - INFO - Total rows: 17146
2025-12-16 07:29:24,040 - INFO - Date range: 1704067200000 to 1765789200000
2025-12-16 07:29:24,050 - INFO - No missing values found
2025-12-16 07:29:24,062 - INFO - No duplicate rows found
2025-12-16 07:29:24,063 - INFO - ==================================================
2025-12-16 07:29:24,083 - INFO - Saved 17146 rows to output_train/cg_long_short_top_account_ratio_history.parquet
2025-12-16 07:29:24,190 - INFO - Copied file to: output_train/20251216_072917/raw_data/cg_long_short_top_account_ratio_history.parquet
2025-12-16 07:29:24,190 - INFO - File stored to database: output_train/20251216_072917/raw_data/cg_long_short_top_account_ratio_history.parquet
2025-12-16 07:29:24,190 - INFO - 
All data saved to output_train
2025-12-16 07:29:24,332 - INFO - Updated session status to: data_loaded
2025-12-16 07:29:24,332 - INFO - Ready for merge_7_tables.py
✅ Step 1: Load Database completed successfully

==========================================
Running Step 2: Merge Tables...
Command: python merge_7_tables.py --exchange binance --pair BTCUSDT --interval 1h --mode initial --timezone WIB
==========================================
2025-12-16 07:29:25,295 - INFO - Loading table data...
2025-12-16 07:29:25,295 - INFO - Loading table data from parquet files...
2025-12-16 07:29:25,366 - INFO - Loaded 17139 rows from cg_spot_price_history
2025-12-16 07:29:25,373 - INFO - Loaded 17146 rows from cg_funding_rate_history
2025-12-16 07:29:25,383 - INFO - Loaded 17139 rows from cg_futures_basis_history
2025-12-16 07:29:25,390 - INFO - Loaded 17146 rows from cg_long_short_global_account_ratio_history
2025-12-16 07:29:25,396 - INFO - Loaded 17146 rows from cg_long_short_top_account_ratio_history
2025-12-16 07:29:25,397 - INFO - Merging tables...
2025-12-16 07:29:25,397 - INFO - Starting table merge process...
2025-12-16 07:29:25,397 - INFO - Preparing base DataFrame...
2025-12-16 07:29:25,421 - INFO - Base DataFrame prepared with 17139 rows
2025-12-16 07:29:25,421 - INFO - Merging cg_funding_rate_history...
2025-12-16 07:29:25,449 - INFO - Merged cg_funding_rate_history, result: 17139 rows
2025-12-16 07:29:25,450 - INFO - Merging cg_futures_basis_history...
2025-12-16 07:29:25,477 - INFO - Merged cg_futures_basis_history, result: 17139 rows
2025-12-16 07:29:25,477 - INFO - Merging cg_long_short_global_account_ratio_history...
2025-12-16 07:29:25,518 - INFO - Merged cg_long_short_global_account_ratio_history, result: 17139 rows
2025-12-16 07:29:25,519 - INFO - Merging cg_long_short_top_account_ratio_history...
2025-12-16 07:29:25,549 - INFO - Merged cg_long_short_top_account_ratio_history, result: 17139 rows
2025-12-16 07:29:25,562 - INFO - Merge completed. Final DataFrame: 17139 rows, 23 columns
2025-12-16 07:29:25,562 - INFO - Cleaning merged data...
2025-12-16 07:29:25,562 - INFO - Cleaning merged data...
2025-12-16 07:29:25,667 - INFO - Data cleaning: 17139 -> 17139 rows (100.0% retained)
2025-12-16 07:29:25,672 - INFO - No missing values after cleaning
2025-12-16 07:29:25,672 - INFO - 
=== Merged Data Validation ===
2025-12-16 07:29:25,672 - INFO - Total rows: 17139
2025-12-16 07:29:25,672 - INFO - Total columns: 23
2025-12-16 07:29:25,673 - INFO - Time range: 2024-01-01 00:00:00 to 2025-12-15 02:00:00
2025-12-16 07:29:25,675 - INFO - Exchanges: 1 - ['Binance']
2025-12-16 07:29:25,677 - INFO - Symbols: 1 - ['BTCUSDT']
2025-12-16 07:29:25,678 - INFO - Intervals: 1 - ['1h']
2025-12-16 07:29:25,680 - INFO - Median time difference: 0 days 01:00:00
2025-12-16 07:29:25,682 - INFO - Rows with complete price data: 17139/17139 (100.0%)
2025-12-16 07:29:25,682 - INFO - ========================================
2025-12-16 07:29:25,740 - INFO - Merged data saved to output_train/merged_7_tables.parquet
2025-12-16 07:29:26,148 - INFO - Merged data saved to output_train/merged_7_tables.csv
2025-12-16 07:29:26,148 - INFO - Column mapping saved to output_train/column_mapping.txt
2025-12-16 07:29:26,148 - INFO - 
=== Merge Complete ===
2025-12-16 07:29:26,148 - INFO - Ready for feature_engineering.py
✅ Step 2: Merge Tables completed successfully

==========================================
Running Step 3: Feature Engineering...
Command: python feature_engineering.py --exchange binance --pair BTCUSDT --interval 1h --mode initial --timezone WIB
==========================================
2025-12-16 07:29:26,858 - INFO - Loading merged data for feature engineering...
2025-12-16 07:29:26,858 - INFO - Loading merged data...
2025-12-16 07:29:26,986 - INFO - Loaded 17139 rows with 23 columns
2025-12-16 07:29:26,986 - INFO - Adding price features...
2025-12-16 07:29:27,056 - INFO - Adding funding rate features...
2025-12-16 07:29:27,072 - INFO - Adding basis features...
2025-12-16 07:29:27,091 - INFO - Skipping taker volume features (spot data not used)
2025-12-16 07:29:27,091 - INFO - Skipping orderbook features (spot data not used)
2025-12-16 07:29:27,091 - INFO - Adding long/short ratio features...
2025-12-16 07:29:27,133 - INFO - Adding cross-table features...
2025-12-16 07:29:27,134 - INFO - Added 2 cross-table features
2025-12-16 07:29:27,134 - INFO - Cleaning features...
2025-12-16 07:29:27,361 - INFO - Dropped 0 rows with missing essential features
2025-12-16 07:29:27,364 - INFO - Final feature set: 35 features
2025-12-16 07:29:27,364 - INFO - 
=== Feature Validation ===
2025-12-16 07:29:27,364 - INFO - Total engineered features: 35
2025-12-16 07:29:27,460 - INFO - Feature statistics summary:
2025-12-16 07:29:27,461 - INFO -       price_close_return_1  price_close_return_5  ...  cross_funding_price  cross_ls_price
mean                0.0001                0.0003  ...               0.0002          0.0000
std                 0.0052                0.0115  ...               0.0064          0.0010
min                -0.0490               -0.1062  ...              -0.0689         -0.0188
max                 0.0503                0.0989  ...               0.1299          0.0175

[4 rows x 35 columns]
2025-12-16 07:29:27,531 - INFO - No constant features found
2025-12-16 07:29:27,531 - INFO - ==============================
2025-12-16 07:29:27,632 - INFO - Features saved to output_train/features_engineered.parquet
2025-12-16 07:29:27,632 - INFO - Feature list saved to output_train/feature_list.txt
2025-12-16 07:29:27,690 - INFO - Features-only dataset saved to output_train/features_only.parquet
2025-12-16 07:29:27,690 - INFO - 
=== Feature Engineering Complete ===
2025-12-16 07:29:27,690 - INFO - Ready for label_builder.py
✅ Step 3: Feature Engineering completed successfully

==========================================
Running Step 4: Label Building...
Command: python label_builder.py --exchange binance --pair BTCUSDT --interval 1h --mode initial --timezone WIB
==========================================
2025-12-16 07:29:28,383 - INFO - Loading feature data for label building...
2025-12-16 07:29:28,383 - INFO - Loading engineered features data...
2025-12-16 07:29:28,441 - INFO - Loaded 17139 rows with 58 columns
2025-12-16 07:29:28,441 - INFO - Creating binary labels for trend prediction...
2025-12-16 07:29:28,473 - INFO - Label Distribution:
2025-12-16 07:29:28,473 - INFO - Bullish (1): 8691 (50.7%)
2025-12-16 07:29:28,474 - INFO - Bearish (0): 8447 (49.3%)
2025-12-16 07:29:28,479 - INFO - Adding label statistics...
2025-12-16 07:29:28,480 - INFO - Label Statistics:
2025-12-16 07:29:28,480 - INFO - Total samples: 17138
2025-12-16 07:29:28,480 - INFO - Imbalance ratio (bullish/bearish): 1.029
2025-12-16 07:29:28,487 - INFO - Label distribution by group (top 10):
2025-12-16 07:29:28,489 - INFO -   Binance/BTCUSDT/1h: 17138 samples, 50.7% bullish
2025-12-16 07:29:28,490 - INFO - 
=== Label Validation ===
2025-12-16 07:29:28,490 - INFO - Total samples: 17138
2025-12-16 07:29:28,491 - INFO - Unique labels: 2
2025-12-16 07:29:28,491 - INFO - Label counts:
target
1    8691
0    8447
Name: count, dtype: int64
2025-12-16 07:29:28,492 - INFO - Reasonably balanced label distribution
2025-12-16 07:29:28,541 - INFO - Time validation passed - no future leakage detected
2025-12-16 07:29:28,700 - INFO - Top 10 feature-label correlations:
2025-12-16 07:29:28,700 - INFO -   price_log_return: 0.055
2025-12-16 07:29:28,700 - INFO -   price_close_return_1: 0.055
2025-12-16 07:29:28,700 - INFO -   price_close_return_5: 0.045
2025-12-16 07:29:28,700 - INFO -   price_wick_upper: 0.031
2025-12-16 07:29:28,700 - INFO -   cross_ls_price: 0.027
2025-12-16 07:29:28,700 - INFO -   price_volume_change: 0.016
2025-12-16 07:29:28,700 - INFO -   ls_top_zscore: 0.016
2025-12-16 07:29:28,700 - INFO -   price_rolling_vol_5: 0.014
2025-12-16 07:29:28,700 - INFO -   price_true_range: 0.014
2025-12-16 07:29:28,700 - INFO -   basis_zscore: 0.013
2025-12-16 07:29:28,700 - INFO - ==============================
2025-12-16 07:29:28,700 - INFO - Preparing training data...
2025-12-16 07:29:28,701 - INFO - Using 54 features for training
2025-12-16 07:29:28,742 - INFO - Final training set: 17138 samples, 54 features
2025-12-16 07:29:28,856 - INFO - Labeled dataset saved to output_train/labeled_data.parquet
2025-12-16 07:29:28,939 - INFO - Features saved to output_train/X_train_features.parquet
2025-12-16 07:29:28,941 - INFO - Labels saved to output_train/y_train_labels.parquet
2025-12-16 07:29:28,941 - INFO - Training feature list saved to output_train/training_features.txt
2025-12-16 07:29:28,945 - INFO - Dataset summary saved to output_train/dataset_summary.txt
2025-12-16 07:29:28,945 - INFO - 
=== Label Building Complete ===
2025-12-16 07:29:28,945 - INFO - Ready for xgboost_trainer.py
✅ Step 4: Label Building completed successfully

==========================================
Running Step 5: Model Training...
Command: python xgboost_trainer.py --exchange binance --pair BTCUSDT --interval 1h --mode initial --timezone WIB
==========================================
2025-12-16 07:29:30,328 - INFO - Loading training data...
2025-12-16 07:29:30,328 - INFO - Loading training data...
2025-12-16 07:29:30,396 - INFO - Loaded features: (17138, 54)
2025-12-16 07:29:30,396 - INFO - Loaded labels: (17138,)
2025-12-16 07:29:30,397 - INFO - Label distribution: {1: 8691, 0: 8447}
2025-12-16 07:29:30,398 - INFO - Preparing data splits...
2025-12-16 07:29:30,434 - INFO - Train set: 10968 samples (0.507 bullish)
2025-12-16 07:29:30,434 - INFO - Validation set: 2742 samples (0.507 bullish)
2025-12-16 07:29:30,434 - INFO - Test set: 3428 samples (0.507 bullish)
2025-12-16 07:29:30,435 - INFO - Performing hyperparameter tuning...
2025-12-16 07:29:30,435 - INFO - Performing hyperparameter tuning...
2025-12-16 07:31:28,022 - INFO - Best validation AUC: 0.5542
2025-12-16 07:31:28,022 - INFO - Best parameters: {'objective': 'binary:logistic', 'eval_metric': ['logloss', 'auc'], 'learning_rate': 0.1, 'max_depth': 6, 'min_child_weight': 5, 'subsample': 1.0, 'colsample_bytree': 0.8, 'colsample_bylevel': 0.8, 'alpha': 0, 'lambda': 1, 'random_state': 42, 'n_estimators': 100, 'early_stopping_rounds': 10, 'verbosity': 1}
2025-12-16 07:31:28,023 - INFO - Training final model...
2025-12-16 07:31:28,023 - INFO - Training XGBoost model...
2025-12-16 07:31:28,023 - INFO - Using parameters: {'objective': 'binary:logistic', 'eval_metric': ['logloss', 'auc'], 'learning_rate': 0.1, 'max_depth': 6, 'min_child_weight': 5, 'subsample': 1.0, 'colsample_bytree': 0.8, 'colsample_bylevel': 0.8, 'alpha': 0, 'lambda': 1, 'random_state': 42, 'n_estimators': 100, 'early_stopping_rounds': 10, 'verbosity': 1}
2025-12-16 07:31:28,432 - INFO - Training completed:
2025-12-16 07:31:28,432 - INFO -   Train AUC: 0.7698, Val AUC: 0.5438
2025-12-16 07:31:28,432 - INFO -   Train LogLoss: 0.6444, Val LogLoss: 0.6915
2025-12-16 07:31:28,433 - INFO -   Best iteration: 11
2025-12-16 07:31:28,433 - INFO - Evaluating model performance...
2025-12-16 07:31:28,495 - INFO - Test Set Performance:
2025-12-16 07:31:28,495 - INFO -   accuracy: 0.5216
2025-12-16 07:31:28,495 - INFO -   precision: 0.5254
2025-12-16 07:31:28,495 - INFO -   recall: 0.5834
2025-12-16 07:31:28,495 - INFO -   f1: 0.5529
2025-12-16 07:31:28,496 - INFO -   roc_auc: 0.5265
2025-12-16 07:31:28,499 - INFO - Confusion Matrix:
2025-12-16 07:31:28,499 - INFO -   True Negatives: 774
2025-12-16 07:31:28,500 - INFO -   False Positives: 916
2025-12-16 07:31:28,500 - INFO -   False Negatives: 724
2025-12-16 07:31:28,500 - INFO -   True Positives: 1014
2025-12-16 07:31:28,500 - INFO - Analyzing feature importance...
2025-12-16 07:31:28,502 - INFO - Top 15 Most Important Features:
2025-12-16 07:31:28,502 - INFO -   price_close_return_5: 0.0304
2025-12-16 07:31:28,503 - INFO -   ls_global_global_account_long_short_ratio: 0.0254
2025-12-16 07:31:28,503 - INFO -   price_close_return_1: 0.0254
2025-12-16 07:31:28,503 - INFO -   funding_open: 0.0235
2025-12-16 07:31:28,503 - INFO -   price_close: 0.0230
2025-12-16 07:31:28,504 - INFO -   funding_mean_24: 0.0230
2025-12-16 07:31:28,504 - INFO -   price_wick_upper: 0.0227
2025-12-16 07:31:28,504 - INFO -   price_volume_usd: 0.0227
2025-12-16 07:31:28,504 - INFO -   ls_global_global_account_long_percent: 0.0226
2025-12-16 07:31:28,505 - INFO -   basis_close_change: 0.0225
2025-12-16 07:31:28,505 - INFO -   price_close_mean_5: 0.0224
2025-12-16 07:31:28,505 - INFO -   price_log_return: 0.0224
2025-12-16 07:31:28,506 - INFO -   price_close_std_5: 0.0224
2025-12-16 07:31:28,506 - INFO -   basis_open_change: 0.0221
2025-12-16 07:31:28,507 - INFO -   price_volume_mean_10: 0.0220
2025-12-16 07:31:28,507 - INFO - Performing cross validation...
2025-12-16 07:31:33,089 - INFO - Cross Validation Results (5 folds):
2025-12-16 07:31:33,090 - INFO -   Mean AUC: 0.5166 ± 0.0049
2025-12-16 07:31:33,090 - INFO -   Scores: ['0.5183', '0.5234', '0.5191', '0.5095', '0.5128']
2025-12-16 07:31:33,090 - INFO - Saving model and results...
2025-12-16 07:31:33,094 - INFO - Model saved to output_train/xgboost_trading_model_20251216_073133.joblib
2025-12-16 07:31:33,097 - INFO - Latest model saved to output_train/latest_model.joblib
2025-12-16 07:31:33,098 - INFO - Results saved to output_train/training_results_20251216_073133.json
2025-12-16 07:31:33,102 - INFO - Feature importance saved to output_train/feature_importance_20251216_073133.csv
2025-12-16 07:31:33,102 - INFO - Feature list saved to output_train/model_features.txt
2025-12-16 07:31:33,102 - INFO - 
=== Training Complete ===
2025-12-16 07:31:33,102 - INFO - Ready for model_evaluation.py
✅ Step 5: Model Training completed successfully

==========================================
Running Step 6: Model Evaluation...
Command: python model_evaluation_with_leverage.py --exchange binance --pair BTCUSDT --interval 1h --mode initial --timezone WIB
==========================================
2025-12-16 07:31:35,393 - INFO - Loading model and data for evaluation...
2025-12-16 07:31:35,394 - INFO - Loading model and test data...
2025-12-16 07:31:36,141 - INFO - Model loaded from output_train/latest_model.joblib
2025-12-16 07:31:36,192 - INFO - Loaded 17138 samples from labeled data
2025-12-16 07:31:36,192 - INFO - Creating trading signals...
2025-12-16 07:31:36,309 - INFO - Trading Signal Distribution:
2025-12-16 07:31:36,310 - INFO - Buy signals (1): 9818 (57.3%)
2025-12-16 07:31:36,310 - INFO - Neutral/Sell signals (0): 7320 (42.7%)
2025-12-16 07:31:36,359 - INFO - [Binance BTCUSDT 1h] transitions: buy=3785 sell=3785 rows=17138
2025-12-16 07:31:36,359 - INFO - [Binance BTCUSDT 1h] leverage=10.0 margin_fraction=0.2 initial_cash=1000.0
2025-12-16 07:31:37,352 - INFO - Saved trade events to: output_train/trade_events.csv (rows=7570)
2025-12-16 07:31:37,353 - INFO - Saved paired trades to: output_train/trades.csv (rows=3785)
2025-12-16 07:31:37,353 - INFO - Saved account statement (EQUITY) to: output_train/rekening_koran.csv (rows=7571)
2025-12-16 07:31:37,353 - INFO - Saved account statement (CASH) to: output_train/rekening_koran_cash.csv (rows=7571)
2025-12-16 07:31:37,354 - INFO - Calculating trading returns...
2025-12-16 07:31:37,386 - INFO - Calculated returns for 17137 trading periods
2025-12-16 07:31:37,386 - INFO - Calculating performance metrics...
2025-12-16 07:31:37,397 - INFO - Performance Summary:
2025-12-16 07:31:37,397 - INFO -   total_return: 897600.5788
2025-12-16 07:31:37,397 - INFO -   benchmark_return: 1.1010
2025-12-16 07:31:37,397 - INFO -   cagr: 1103.6966
2025-12-16 07:31:37,397 - INFO -   benchmark_cagr: 0.4616
2025-12-16 07:31:37,398 - INFO -   max_drawdown: -0.3937
2025-12-16 07:31:37,398 - INFO -   benchmark_max_drawdown: -0.3476
2025-12-16 07:31:37,398 - INFO -   sharpe_ratio: 1.6498
2025-12-16 07:31:37,398 - INFO -   win_rate: 0.6062
2025-12-16 07:31:37,398 - INFO -   profit_factor: 0.9958
2025-12-16 07:31:37,398 - INFO -   total_trades: 9818
2025-12-16 07:31:37,398 - INFO -   winning_trades: 5952
2025-12-16 07:31:37,398 - INFO -   losing_trades: 3866
2025-12-16 07:31:37,398 - INFO -   avg_win: 0.0069
2025-12-16 07:31:37,398 - INFO -   avg_loss: -0.0069
2025-12-16 07:31:37,398 - INFO -   trading_days: 714
2025-12-16 07:31:37,398 - INFO - Creating performance plots...
2025-12-16 07:31:39,876 - INFO - Performance plots saved to output_train/performance_analysis.png
2025-12-16 07:31:39,877 - INFO - Generating detailed report...
2025-12-16 07:31:39,881 - INFO - Saving evaluation results...
2025-12-16 07:31:40,012 - INFO - Saved metrics to output_train/performance_metrics_20251216_073139.json
2025-12-16 07:31:40,013 - INFO - Saved report to output_train/performance_report_20251216_073139.json
2025-12-16 07:31:40,013 - INFO - Saved trading results to output_train/trading_results.parquet
2025-12-16 07:31:40,013 - INFO - 
=== Model Evaluation Complete ===
2025-12-16 07:31:40,013 - INFO - Generated files:
2025-12-16 07:31:40,013 - INFO -   - output_train/rekening_koran.csv (EQUITY)
2025-12-16 07:31:40,013 - INFO -   - output_train/rekening_koran_cash.csv (CASH)
2025-12-16 07:31:40,013 - INFO -   - output_train/trade_events.csv
2025-12-16 07:31:40,013 - INFO -   - output_train/trades.csv
2025-12-16 07:31:40,013 - INFO - 
Leverage settings used:
2025-12-16 07:31:40,013 - INFO -   LEVERAGE=10.0
2025-12-16 07:31:40,013 - INFO -   MARGIN_FRACTION=0.2
2025-12-16 07:31:40,013 - INFO -   INITIAL_CASH=1000.0
2025-12-16 07:31:40,013 - INFO -   FEE_RATE=0.0004
2025-12-16 07:31:40,014 - INFO -   THRESHOLD=0.5
✅ Step 6: Model Evaluation completed successfully

==========================================
✅ Pipeline completed successfully!
==========================================

Output directory contents:
total 35644
drwxr-xr-x 4 yumna yumna    4096 Dec 16 07:31 .
drwxr-xr-x 7 yumna yumna    4096 Dec 16 07:22 ..
drwxr-xr-x 3 yumna yumna    4096 Dec 15 21:45 20251215_214515
drwxr-xr-x 3 yumna yumna    4096 Dec 16 07:29 20251216_072917
-rw-r--r-- 1 yumna yumna  515261 Dec 16 07:29 cg_funding_rate_history.parquet
-rw-r--r-- 1 yumna yumna  335801 Dec 16 07:29 cg_futures_basis_history.parquet
-rw-r--r-- 1 yumna yumna  256001 Dec 16 07:29 cg_long_short_global_account_ratio_history.parquet
-rw-r--r-- 1 yumna yumna  248865 Dec 16 07:29 cg_long_short_top_account_ratio_history.parquet
-rw-r--r-- 1 yumna yumna  780115 Dec 16 07:29 cg_spot_price_history.parquet
-rw-r--r-- 1 yumna yumna     472 Dec 16 07:29 column_mapping.txt
-rw-r--r-- 1 yumna yumna     265 Dec 16 07:29 dataset_summary.txt
-rw-r--r-- 1 yumna yumna    1636 Dec 15 21:46 feature_importance_20251215_214645.csv
-rw-r--r-- 1 yumna yumna    1636 Dec 16 07:31 feature_importance_20251216_073133.csv
-rw-r--r-- 1 yumna yumna     652 Dec 16 07:29 feature_list.txt
-rw-r--r-- 1 yumna yumna 5347801 Dec 16 07:29 features_engineered.parquet
-rw-r--r-- 1 yumna yumna 3754029 Dec 16 07:29 features_only.parquet
-rw-r--r-- 1 yumna yumna 5350255 Dec 16 07:29 labeled_data.parquet
-rw-r--r-- 1 yumna yumna   78403 Dec 16 07:31 latest_model.joblib
-rw-r--r-- 1 yumna yumna 3082664 Dec 16 07:29 merged_7_tables.csv
-rw-r--r-- 1 yumna yumna 1594374 Dec 16 07:29 merged_7_tables.parquet
-rw-r--r-- 1 yumna yumna    1052 Dec 16 07:31 model_features.txt
-rw-r--r-- 1 yumna yumna  429650 Dec 16 07:31 performance_analysis.png
-rw-r--r-- 1 yumna yumna     517 Dec 15 21:46 performance_metrics_20251215_214649.json
-rw-r--r-- 1 yumna yumna     517 Dec 16 07:31 performance_metrics_20251216_073139.json
-rw-r--r-- 1 yumna yumna    1037 Dec 15 21:46 performance_report_20251215_214649.json
-rw-r--r-- 1 yumna yumna    1037 Dec 16 07:31 performance_report_20251216_073139.json
-rw-r--r-- 1 yumna yumna  536839 Dec 16 07:31 rekening_koran_cash.csv
-rw-r--r-- 1 yumna yumna  536892 Dec 16 07:31 rekening_koran.csv
-rw-r--r-- 1 yumna yumna 1274964 Dec 16 07:31 trade_events.csv
-rw-r--r-- 1 yumna yumna  621001 Dec 16 07:31 trades.csv
-rw-r--r-- 1 yumna yumna 6292104 Dec 16 07:31 trading_results.parquet
-rw-r--r-- 1 yumna yumna    1055 Dec 16 07:29 training_features.txt
-rw-r--r-- 1 yumna yumna     994 Dec 15 21:46 training_results_20251215_214645.json
-rw-r--r-- 1 yumna yumna     994 Dec 16 07:31 training_results_20251216_073133.json
-rw-r--r-- 1 yumna yumna   78403 Dec 15 21:46 xgboost_trading_model_20251215_214645.joblib
-rw-r--r-- 1 yumna yumna   78403 Dec 16 07:31 xgboost_trading_model_20251216_073133.joblib
-rw-r--r-- 1 yumna yumna 5189282 Dec 16 07:29 X_train_features.parquet
-rw-r--r-- 1 yumna yumna    3377 Dec 16 07:29 y_train_labels.parquet

Trained models:
-rw-r--r-- 1 yumna yumna 78403 Dec 16 07:31 ./output_train/latest_model.joblib
-rw-r--r-- 1 yumna yumna 78403 Dec 15 21:46 ./output_train/xgboost_trading_model_20251215_214645.joblib
-rw-r--r-- 1 yumna yumna 78403 Dec 16 07:31 ./output_train/xgboost_trading_model_20251216_073133.joblib

Next steps:
1. Start API server: python api_server.py
2. Access API at: http://localhost:8000/api/v1/
3. View results: ls -la ./output_train