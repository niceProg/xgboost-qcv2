#!/usr/bin/env python3
"""
XGBoost Model Trainer for 17 Price Features (QuantConnect Compatible)

This script trains an XGBoost model using only the 17 price features
that are available in QuantConnect.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
import base64
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from parent directory
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# XGBoost
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Import feature engineering and label builder
from feature_engineering_17price import FeatureEngineer17, FEATURES_17
from label_builder_17price import LabelBuilder17


# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    # Data paths
    DATA_PATH = "./output_train_17price/data.csv"  # Path to price data CSV
    OUTPUT_DIR = "./output_train_17price"

    # Feature settings
    FEATURES = FEATURES_17

    # Label settings
    FORWARD_BARS = 1  # Predict next bar
    LABEL_TYPE = "next_bar"  # 'next_bar', 'threshold', '3bar'
    BUY_THRESHOLD = 0.005  # 0.5% for threshold-based
    SELL_THRESHOLD = -0.005

    # Model settings
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    N_ESTIMATORS = 200
    MAX_DEPTH = 6
    LEARNING_RATE = 0.05
    SUBSAMPLE = 0.8
    COLSAMPLE_BYTREE = 0.8

    # Training settings
    BALANCE_DATA = True  # Balance buy/sell samples
    BALANCE_METHOD = "undersample"  # 'undersample' or 'oversample'

    # API output (for integration with dragonfortune.ai API)
    SAVE_TO_API_FORMAT = True


# =============================================================================
# MAIN TRAINER
# =============================================================================
class XGBoostTrainer17:
    """Train XGBoost model with 17 price features."""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.fe = FeatureEngineer17()
        self.lb = LabelBuilder17(
            forward_bars=self.config.FORWARD_BARS,
            profit_threshold=self.config.BUY_THRESHOLD,
            loss_threshold=self.config.SELL_THRESHOLD
        )

        # Create output directory
        Path(self.config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Load and prepare data."""
        print(f"Loading data from: {self.config.DATA_PATH}")

        if not os.path.exists(self.config.DATA_PATH):
            print(f"Warning: Data file not found: {self.config.DATA_PATH}")
            print(f"Current directory: {os.getcwd()}")
            print(f"Looking in: {os.path.abspath(self.config.DATA_PATH)}")
            raise FileNotFoundError(f"Data file not found: {self.config.DATA_PATH}")

        df = pd.read_csv(self.config.DATA_PATH)
        print(f"Loaded {len(df)} rows")

        # Handle column naming: 'time' from database, convert to 'timestamp'
        if 'time' in df.columns and 'timestamp' not in df.columns:
            df['timestamp'] = df['time']
            df = df.sort_values('time')

        # Ensure datetime column
        if 'timestamp' in df.columns:
            # If timestamp is in milliseconds (from database), convert to datetime
            if df['timestamp'].dtype in ['int64', 'float64']:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        elif 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
            df = df.sort_values('timestamp')

        return df

    def prepare_features_labels(self, df: pd.DataFrame) -> tuple:
        """Prepare features and labels."""
        print("\n" + "="*60)
        print("PREPARING FEATURES AND LABELS")
        print("="*60)

        # Add features
        print("Adding 17 price features...")
        df = self.fe.add_features(df)
        print(f"Features added: {len(self.config.FEATURES)}")

        # Add labels based on type
        print(f"\nCreating labels (type: {self.config.LABEL_TYPE})...")

        if self.config.LABEL_TYPE == "next_bar":
            from label_builder_17price import create_labels_next_bar
            df = create_labels_next_bar(df)

        elif self.config.LABEL_TYPE == "threshold":
            from label_builder_17price import create_labels_threshold
            df = create_labels_threshold(df, self.config.BUY_THRESHOLD, self.config.SELL_THRESHOLD)

        elif self.config.LABEL_TYPE == "3bar":
            from label_builder_17price import create_labels_3bar
            df = create_labels_3bar(df, profit_pct=self.config.BUY_THRESHOLD)

        else:
            # Default
            df = self.lb.add_labels(df)

        # Drop rows with NaN labels
        df = df.dropna(subset=['label', 'forward_return'] if 'forward_return' in df.columns else ['label'])

        print(f"Labels created:")
        label_counts = df['label'].value_counts()
        print(f"  Buy (1):  {label_counts.get(1, 0)}")
        print(f"  Sell (0): {label_counts.get(0, 0)}")

        return df

    def split_data(self, df: pd.DataFrame) -> tuple:
        """Split data into train/test sets."""
        print("\n" + "="*60)
        print("SPLITTING DATA")
        print("="*60)

        # Get feature columns
        feature_cols = self.config.FEATURES

        # Balance data if enabled
        if self.config.BALANCE_DATA:
            print(f"\nBalancing data ({self.config.BALANCE_METHOD})...")
            df = self.lb.create_balanced_dataset(df, method=self.config.BALANCE_METHOD)

        # Split (time-series aware - no shuffle!)
        split_idx = int(len(df) * (1 - self.config.TEST_SIZE))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        print(f"Train: {len(train_df)} rows ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
        print(f"Test:  {len(test_df)} rows ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")

        X_train = train_df[feature_cols].values
        y_train = train_df['label'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['label'].values

        # Handle NaN/Inf
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        return X_train, X_test, y_train, y_test, train_df, test_df

    def train_model(self, X_train, y_train) -> xgb.XGBClassifier:
        """Train XGBoost model."""
        print("\n" + "="*60)
        print("TRAINING MODEL")
        print("="*60)

        model = xgb.XGBClassifier(
            n_estimators=self.config.N_ESTIMATORS,
            max_depth=self.config.MAX_DEPTH,
            learning_rate=self.config.LEARNING_RATE,
            subsample=self.config.SUBSAMPLE,
            colsample_bytree=self.config.COLSAMPLE_BYTREE,
            random_state=self.config.RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )

        print(f"Model parameters:")
        print(f"  n_estimators:     {self.config.N_ESTIMATORS}")
        print(f"  max_depth:        {self.config.MAX_DEPTH}")
        print(f"  learning_rate:    {self.config.LEARNING_RATE}")
        print(f"  subsample:        {self.config.SUBSAMPLE}")
        print(f"  colsample_bytree: {self.config.COLSAMPLE_BYTREE}")

        model.fit(X_train, y_train)

        print("\nModel trained successfully!")
        return model

    def evaluate_model(self, model, X_test, y_test) -> dict:
        """Evaluate model performance."""
        print("\n" + "="*60)
        print("EVALUATING MODEL")
        print("="*60)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5,
        }

        print(f"\nTest Set Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")

        # Feature importance
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.config.FEATURES,
            'importance': importance
        }).sort_values('importance', ascending=False)

        print(f"\nTop 10 Feature Importance:")
        for _, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        return metrics

    def save_model(self, model, train_df, test_df, metrics):
        """Save model and metadata."""
        print("\n" + "="*60)
        print("SAVING MODEL")
        print("="*60)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save model as joblib
        model_path = os.path.join(self.config.OUTPUT_DIR, f"xgboost_model_17price_{timestamp}.joblib")
        joblib.dump(model, model_path)
        print(f"Model saved: {model_path}")

        # Save latest model copy (for easy access)
        latest_model_path = os.path.join(self.config.OUTPUT_DIR, "latest_model.joblib")
        joblib.dump(model, latest_model_path)
        print(f"Latest model: {latest_model_path}")

        # Save feature names
        feature_path = os.path.join(self.config.OUTPUT_DIR, f"features_17price_{timestamp}.txt")
        with open(feature_path, 'w') as f:
            f.write('\n'.join(self.config.FEATURES))
        print(f"Features saved: {feature_path}")

        # Save latest feature names (for easy access)
        latest_feature_path = os.path.join(self.config.OUTPUT_DIR, "model_features_17price.txt")
        with open(latest_feature_path, 'w') as f:
            f.write('\n'.join(self.config.FEATURES))
        print(f"Latest features: {latest_feature_path}")

        # Save metadata
        metadata = {
            'model_type': 'XGBoostClassifier',
            'model_type_code': '17price',
            'model_version': f"17price_{timestamp}",
            'features': self.config.FEATURES,
            'n_features': len(self.config.FEATURES),
            'feature_hash': str(hash(tuple(self.config.FEATURES))),
            'trained_at': datetime.now().isoformat(),
            'train_start': str(train_df['timestamp'].min()),
            'train_end': str(train_df['timestamp'].max()),
            'test_start': str(test_df['timestamp'].min()),
            'test_end': str(test_df['timestamp'].max()),
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'metrics': metrics,
            'config': {
                'n_estimators': self.config.N_ESTIMATORS,
                'max_depth': self.config.MAX_DEPTH,
                'learning_rate': self.config.LEARNING_RATE,
                'forward_bars': self.config.FORWARD_BARS,
                'label_type': self.config.LABEL_TYPE,
            }
        }

        metadata_path = os.path.join(self.config.OUTPUT_DIR, f"metadata_17price_{timestamp}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"Metadata saved: {metadata_path}")

        # Save in API format (base64 encoded model)
        if self.config.SAVE_TO_API_FORMAT:
            self.save_api_format(model, metadata, timestamp)

        return metadata

    def save_api_format(self, model, metadata, timestamp):
        """Save model in format compatible with dragonfortune.ai API."""
        # Save model as base64 using BytesIO (joblib doesn't have dumps())
        from io import BytesIO
        buffer = BytesIO()
        joblib.dump(model, buffer)
        model_bytes = buffer.getvalue()
        model_b64 = base64.b64encode(model_bytes).decode('utf-8')

        api_data = {
            'success': True,
            'model_version': f"17price_{timestamp}",
            'model_hash': metadata['feature_hash'],
            'model_name': 'XGBoost 17-Price Model',
            'created_at': metadata['trained_at'],
            'feature_names': self.config.FEATURES,
            'n_features': len(self.config.FEATURES),
            'model_data_base64': model_b64,
            'metrics': metadata['metrics'],
            'config': metadata['config'],
            'train_info': {
                'train_start': metadata['train_start'],
                'train_end': metadata['train_end'],
                'train_samples': metadata['train_samples'],
            }
        }

        api_path = os.path.join(self.config.OUTPUT_DIR, f"api_format_17price_{timestamp}.json")
        with open(api_path, 'w') as f:
            json.dump(api_data, f, indent=2)
        print(f"API format saved: {api_path}")

        # Save dataset summary for API
        summary = f"""17-Price Model Training Summary
================================

Model Type: XGBoost Classifier
Features: {len(self.config.FEATURES)} (Price only)

Training Period:
- Start: {metadata['train_start']}
- End:   {metadata['train_end']}

Label Strategy: {self.config.LABEL_TYPE.upper()}
- Forward Bars: {self.config.FORWARD_BARS}

Performance Metrics:
- Accuracy:  {metadata['metrics']['accuracy']:.4f}
- Precision: {metadata['metrics']['precision']:.4f}
- Recall:    {metadata['metrics']['recall']:.4f}
- F1 Score:  {metadata['metrics']['f1']:.4f}
- ROC AUC:   {metadata['metrics']['roc_auc']:.4f}

Model Hash: {metadata['feature_hash']}
Created:    {metadata['trained_at']}
"""

        summary_b64 = base64.b64encode(summary.encode('utf-8')).decode('utf-8')

        summary_data = {
            'success': True,
            'session_id': timestamp,
            'created_at': metadata['trained_at'],
            'summary_data_base64': summary_b64
        }

        summary_path = os.path.join(self.config.OUTPUT_DIR, f"dataset_summary_17price_{timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"Dataset summary saved: {summary_path}")

        # Save to database
        if self.config.SAVE_TO_API_FORMAT:
            self.save_to_database(model, metadata, timestamp, summary)

    def save_to_database(self, model, metadata, timestamp, summary):
        """Save model and dataset summary to database."""
        try:
            # Import from parent directory
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from database_storage_futures import DatabaseStorageFutures

            db_storage = DatabaseStorageFutures()

            # Store model with model_version='futures17'
            model_name = f"xgboost_model_17price_{timestamp}.joblib"
            model_id = db_storage.store_model(
                model=model,
                model_name=model_name,
                feature_names=self.config.FEATURES,
                hyperparams=metadata.get('config', {}),
                train_score=metadata['metrics'].get('accuracy', 0),
                val_score=metadata['metrics'].get('roc_auc', 0),
                cv_scores=[],
                model_version='futures17'
            )

            print(f"Model saved to database with ID: {model_id}")

            # Store dataset summary
            summary_bytes = summary.encode('utf-8')
            summary_id = db_storage.store_dataset_summary(
                summary_file=f"dataset_summary_17price_{timestamp}.txt",
                summary_data=summary_bytes,
                model_version='futures17'
            )

            print(f"Dataset summary saved to database with ID: {summary_id}")

        except ImportError as e:
            print(f"Warning: Could not import database_storage_futures: {e}")
        except Exception as e:
            print(f"Warning: Failed to save to database: {e}")

    def run(self):
        """Run full training pipeline."""
        print("="*60)
        print("XGBOOST TRAINER - 17 PRICE FEATURES")
        print("="*60)
        print(f"Features: {len(self.config.FEATURES)}")
        print(f"Label Type: {self.config.LABEL_TYPE}")
        print(f"Forward Bars: {self.config.FORWARD_BARS}")
        print(f"Balance Data: {self.config.BALANCE_DATA}")

        try:
            # Load data
            df = self.load_data()

            # Prepare features and labels
            df = self.prepare_features_labels(df)

            # Split data
            X_train, X_test, y_train, y_test, train_df, test_df = self.split_data(df)

            # Train model
            model = self.train_model(X_train, y_train)

            # Evaluate
            metrics = self.evaluate_model(model, X_test, y_test)

            # Save
            metadata = self.save_model(model, train_df, test_df, metrics)

            print("\n" + "="*60)
            print("TRAINING COMPLETE!")
            print("="*60)

            return model, metadata

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None, None


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train XGBoost model with 17 price features')
    parser.add_argument('--data', type=str, default='./output_train_17price/data.csv', help='Path to data CSV')
    parser.add_argument('--output', type=str, default='./output_train_17price', help='Output directory (deprecated, use --output-dir)')
    parser.add_argument('--output-dir', type=str, default='./output_train_17price', help='Output directory')
    parser.add_argument('--label', type=str, default='next_bar', choices=['next_bar', 'threshold', '3bar'],
                       help='Label type')
    parser.add_argument('--estimators', type=int, default=200, help='Number of estimators')
    parser.add_argument('--depth', type=int, default=6, help='Max depth')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate')
    parser.add_argument('--no-balance', action='store_true', help='Disable data balancing')

    args = parser.parse_args()

    # Update config
    config = Config()
    config.DATA_PATH = args.data
    # Prefer output-dir, fallback to output for backward compatibility
    config.OUTPUT_DIR = args.output_dir if args.output_dir != '../output_train_17price' else args.output
    config.LABEL_TYPE = args.label
    config.N_ESTIMATORS = args.estimators
    config.MAX_DEPTH = args.depth
    config.LEARNING_RATE = args.lr
    config.BALANCE_DATA = not args.no_balance

    # Run training
    trainer = XGBoostTrainer17(config)
    trainer.run()
