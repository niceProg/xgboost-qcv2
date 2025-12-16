#!/usr/bin/env python3
"""
Real-time XGBoost trainer untuk incremental learning.
Update model dengan data baru yang masuk secara real-time.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb

# Import existing modules
from load_database import DatabaseLoader
from merge_7_tables import TableMerger
from feature_engineering import FeatureEngineer
from label_builder import LabelBuilder
from command_line_options import DataFilter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/realtime_trainer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RealtimeTrainer:
    """Handle real-time incremental training."""

    def __init__(self, output_dir: str = './output_train'):
        self.output_dir = Path(output_dir)
        self.realtime_dir = Path('./realtime_data')
        self.models_dir = self.output_dir / 'models'
        self.models_dir.mkdir(exist_ok=True)

        # Model performance tracking
        self.performance_history = self.load_performance_history()

    def load_performance_history(self) -> Dict:
        """Load model performance history."""
        file_path = self.output_dir / 'model_performance.json'
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return {}

    def save_performance_history(self, performance: Dict):
        """Save model performance history."""
        file_path = self.output_dir / 'model_performance.json'
        with open(file_path, 'w') as f:
            json.dump(performance, f, indent=2)

    def get_latest_model_path(self) -> Optional[Path]:
        """Get path to latest model."""
        model_files = list(self.models_dir.glob("xgboost_trading_model_*.joblib"))
        if not model_files:
            return None

        # Sort by modification time
        model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return model_files[0]

    def load_latest_model(self) -> Optional[xgb.XGBClassifier]:
        """Load latest trained model."""
        model_path = self.get_latest_model_path()
        if not model_path:
            logger.error("No existing model found")
            return None

        try:
            model = joblib.load(model_path)
            logger.info(f"Loaded model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

    def get_new_realtime_data(self) -> Optional[pd.DataFrame]:
        """Get new realtime data untuk processing."""
        if not self.realtime_dir.exists():
            return None

        # Get all new parquet files
        files = list(self.realtime_dir.glob("cg_spot_price_history_*.parquet"))
        if not files:
            return None

        # Combine all new data
        dfs = []
        for file in files:
            try:
                df = pd.read_parquet(file)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error reading {file}: {e}")

        if not dfs:
            return None

        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(combined_df)} new records from {len(files)} files")
        return combined_df

    def process_new_data(self, trigger_file: Optional[str] = None) -> bool:
        """Process new data dan update model."""
        logger.info("🚀 Starting real-time data processing")

        try:
            # Get new data
            new_data = self.get_new_realtime_data()
            if new_data is None or new_data.empty:
                logger.info("No new data to process")
                return False

            # Convert new data to expected format
            # This is simplified - in production, you'd need proper data integration
            processed_data = self.prepare_new_data(new_data)
            if processed_data is None:
                logger.error("Failed to prepare new data")
                return False

            # Load existing model
            existing_model = self.load_latest_model()
            if existing_model is None:
                logger.warning("No existing model - skipping incremental update")
                return False

            # Update model incrementally
            updated_model = self.update_model_incremental(existing_model, processed_data)
            if updated_model is None:
                logger.error("Failed to update model")
                return False

            # Evaluate updated model
            performance = self.evaluate_model(updated_model, processed_data)
            if performance is None:
                logger.error("Failed to evaluate model")
                return False

            # Save updated model if performance is good
            if self.should_save_model(performance):
                self.save_updated_model(updated_model, performance)
                self.cleanup_processed_data()
            else:
                logger.warning("Model performance below threshold - not saving")

            return True

        except Exception as e:
            logger.error(f"Error in real-time processing: {e}")
            return False

    def prepare_new_data(self, new_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepare new data untuk training."""
        try:
            # Create DataFilter for processing
            # Get latest timestamp from new data
            max_time = new_data['time'].max()
            min_time = new_data['time'].min()

            data_filter = DataFilter(
                exchange=new_data['exchange'].iloc[0] if 'exchange' in new_data.columns else 'binance',
                pair=new_data['symbol'].iloc[0] if 'symbol' in new_data.columns else 'BTCUSDT',
                interval='1h',
                timezone='WIB',
                trading_hours='7:00-16:00',
                mode='realtime',
                days=1  # Only recent data
            )

            # Process data through pipeline (simplified version)
            # 1. Load additional data for context (last few days)
            loader = DatabaseLoader(data_filter, output_dir=str(self.output_dir))
            historical_data = loader.load_all_tables()

            # Combine new data with historical context
            if historical_data is not None:
                # Add new data to historical
                combined = pd.concat([historical_data, new_data], ignore_index=True)
                # Remove duplicates
                combined = combined.drop_duplicates(subset=['time', 'exchange', 'symbol'], keep='last')
                # Sort by time
                combined = combined.sort_values('time')
            else:
                combined = new_data

            # 2. Merge tables (simplified - normally you'd use TableMerger)
            merged_data = combined.copy()  # Placeholder

            # 3. Feature engineering
            engineer = FeatureEngineer(data_filter, output_dir=str(self.output_dir))
            features_data = engineer.engineer_features(merged_data)

            if features_data is None:
                logger.error("Feature engineering failed")
                return None

            # 4. Label building (only for historical data, not newest)
            builder = LabelBuilder(data_filter, output_dir=str(self.output_dir))
            labeled_data = builder.build_labels(features_data)

            if labeled_data is None:
                logger.error("Label building failed")
                return None

            # Keep only records with labels (exclude newest data)
            labeled_data = labeled_data.dropna(subset=['target'])

            logger.info(f"Prepared {len(labeled_data)} labeled samples")
            return labeled_data

        except Exception as e:
            logger.error(f"Error preparing new data: {e}")
            return None

    def update_model_incremental(self, model: xgb.XGBClassifier, new_data: pd.DataFrame) -> Optional[xgb.XGBClassifier]:
        """Update model dengan incremental learning."""
        try:
            # Load feature list
            features_file = self.output_dir / "training_features.txt"
            if not features_file.exists():
                logger.error("Training features file not found")
                return None

            with open(features_file, 'r') as f:
                feature_list = [line.strip() for line in f if line.strip()]

            # Prepare features and target
            X = new_data[feature_list]
            y = new_data['target']

            # Remove any rows with missing values
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]

            if len(X) == 0:
                logger.warning("No valid samples for training")
                return None

            logger.info(f"Updating model with {len(X)} samples")

            # Create DMatrix for efficiency
            dtrain = xgb.DMatrix(X, label=y, feature_names=feature_list)

            # Get current model parameters
            params = model.get_params()
            xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'learning_rate': params.get('learning_rate', 0.1),
                'max_depth': params.get('max_depth', 6),
                'min_child_weight': params.get('min_child_weight', 1),
                'subsample': params.get('subsample', 0.8),
                'colsample_bytree': params.get('colsample_bytree', 0.8),
                'alpha': params.get('alpha', 0),
                'lambda': params.get('lambda', 1)
            }

            # Incremental training with early stopping
            evals_result = {}
            model_new = xgb.train(
                xgb_params,
                dtrain,
                xgb_model=model.get_booster(),
                num_boost_round=50,  # Limited rounds for incremental update
                evals=[(dtrain, 'train')],
                evals_result=evals_result,
                early_stopping_rounds=10,
                verbose_eval=10
            )

            # Create new classifier
            updated_model = xgb.XGBClassifier()
            updated_model._Booster = model_new
            updated_model.n_features_in_ = len(feature_list)
            updated_model.feature_names_in_ = feature_list

            logger.info("✅ Model updated successfully")
            return updated_model

        except Exception as e:
            logger.error(f"Error updating model incrementally: {e}")
            return None

    def evaluate_model(self, model: xgb.XGBClassifier, data: pd.DataFrame) -> Optional[Dict]:
        """Evaluate model performance on new data."""
        try:
            # Load feature list
            features_file = self.output_dir / "training_features.txt"
            if not features_file.exists():
                return None

            with open(features_file, 'r') as f:
                feature_list = [line.strip() for line in f if line.strip()]

            # Prepare features and target
            X = data[feature_list]
            y = data['target']

            # Remove missing values
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]

            if len(X) < 20:  # Need minimum samples
                logger.warning("Insufficient samples for evaluation")
                return None

            # Make predictions
            y_pred_proba = model.predict_proba(X)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)

            # Calculate metrics
            accuracy = accuracy_score(y, y_pred)
            auc = roc_auc_score(y, y_pred_proba)

            # Additional metrics
            buy_accuracy = accuracy_score(y[y == 1], y_pred[y == 1]) if sum(y == 1) > 0 else 0
            hold_accuracy = accuracy_score(y[y == 0], y_pred[y == 0]) if sum(y == 0) > 0 else 0

            performance = {
                'timestamp': datetime.now().isoformat(),
                'samples': len(X),
                'accuracy': float(accuracy),
                'auc': float(auc),
                'buy_accuracy': float(buy_accuracy),
                'hold_accuracy': float(hold_accuracy),
                'buy_signals': int(sum(y_pred)),
                'buy_signal_rate': float(sum(y_pred) / len(y_pred)),
                'actual_buys': int(sum(y)),
                'actual_buy_rate': float(sum(y) / len(y))
            }

            logger.info(f"Model performance: AUC={auc:.3f}, Accuracy={accuracy:.3f}")
            return performance

        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return None

    def should_save_model(self, performance: Dict) -> bool:
        """Check if model should be saved based on performance."""
        # Minimum thresholds
        min_auc = 0.6
        min_accuracy = 0.6

        # Check if performance meets thresholds
        if performance['auc'] < min_auc or performance['accuracy'] < min_accuracy:
            logger.warning(f"Performance below threshold: AUC={performance['auc']:.3f}, Acc={performance['accuracy']:.3f}")
            return False

        # Check against previous performance
        if self.performance_history:
            latest_auc = self.performance_history.get('latest_auc', 0)
            if performance['auc'] < latest_auc * 0.95:  # Allow 5% degradation
                logger.warning(f"Performance degraded: {performance['auc']:.3f} vs {latest_auc:.3f}")
                return False

        return True

    def save_updated_model(self, model: xgb.XGBClassifier, performance: Dict):
        """Save updated model dengan timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"xgboost_trading_model_realtime_{timestamp}.joblib"
        model_path = self.models_dir / model_filename

        # Save model
        joblib.dump(model, model_path)
        logger.info(f"Saved updated model to {model_path}")

        # Update latest model symlink
        latest_path = self.models_dir / "latest_model.joblib"
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(model_path)

        # Update performance history
        self.performance_history['latest_update'] = performance['timestamp']
        self.performance_history['latest_auc'] = performance['auc']
        self.performance_history['latest_accuracy'] = performance['accuracy']
        self.performance_history['updates'] = self.performance_history.get('updates', 0) + 1

        self.save_performance_history(self.performance_history)

        # Send notification
        self.send_update_notification(performance)

    def cleanup_processed_data(self):
        """Clean up processed realtime data files."""
        try:
            # Remove processed files
            for file in self.realtime_dir.glob("cg_spot_price_history_*.parquet"):
                file.unlink()
                logger.info(f"Cleaned up {file.name}")

        except Exception as e:
            logger.error(f"Error cleaning up data: {e}")

    def send_update_notification(self, performance: Dict):
        """Send model update notification."""
        message = f"""
🤖 **Model Updated Successfully!**

📊 Performance Metrics:
• AUC: {performance['auc']:.3f}
• Accuracy: {performance['accuracy']:.3f}
• Samples: {performance['samples']}

🎯 Signal Statistics:
• Buy Signals: {performance['buy_signals']} ({performance['buy_signal_rate']:.1%})
• Actual Buys: {performance['actual_buys']} ({performance['actual_buy_rate']:.1%})

⏰ Updated: {performance['timestamp']}
        """

        # Send to webhook if configured
        webhook_url = os.getenv('WEBHOOK_URL')
        if webhook_url:
            try:
                import requests
                requests.post(webhook_url, json={"text": message})
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")

        logger.info("✅ Model update notification sent")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Real-time XGBoost trainer')
    parser.add_argument('--trigger-file', type=str, help='Trigger file path')
    parser.add_argument('--mode', type=str, choices=['incremental', 'full'], default='incremental',
                      help='Training mode')
    parser.add_argument('--output-dir', type=str, default='./output_train',
                      help='Output directory')

    args = parser.parse_args()

    trainer = RealtimeTrainer(output_dir=args.output_dir)

    if args.mode == 'incremental':
        success = trainer.process_new_data(args.trigger_file)
        sys.exit(0 if success else 1)
    else:
        logger.error("Full retraining not implemented in realtime trainer")
        sys.exit(1)


if __name__ == "__main__":
    main()