#!/usr/bin/env python3
"""
Real-time XGBoost trainer for new data processing.
Trains model incrementally with new 2025 data.
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

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

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

class RealtimeXGBoostTrainer:
    """Handle real-time incremental training with new 2025 data."""

    def __init__(self, output_dir: str = './output_train'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Configuration
        self.config = self.load_config()

        # Model paths
        self.models_dir = self.output_dir / 'models'
        self.models_dir.mkdir(exist_ok=True)

        # State
        self.state_file = './state/model_status.json'
        self.state = self.load_state()

    def load_config(self) -> Dict:
        """Load training configuration."""
        default_config = {
            "min_samples_for_update": 100,
            "performance_threshold": 0.6,
            "max_features": 100,
            "batch_size": 1000,
            "early_stopping_rounds": 20,
            "training": {
                "learning_rate": 0.1,
                "max_depth": 6,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "alpha": 0,
                "lambda": 1
            },
            "notification": {
                "enabled": True,
                "telegram_token": os.getenv('TELEGRAM_BOT_TOKEN'),
                "telegram_chat_id": os.getenv('TELEGRAM_CHAT_ID')
            }
        }

        config_path = Path('./config/realtime_config.json')
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                # Merge with defaults
                for key in default_config:
                    if key not in user_config:
                        user_config[key] = default_config[key]
                    elif isinstance(default_config[key], dict):
                        default_config[key].update(user_config.get(key, {}))
                return user_config
            except Exception as e:
                logger.error(f"Error loading config: {e}. Using defaults.")

        return default_config

    def load_state(self) -> Dict:
        """Load training state."""
        state_path = Path(self.state_file)
        state_path.parent.mkdir(exist_ok=True)

        if state_path.exists():
            try:
                with open(state_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading state: {e}. Using defaults.")

        # Default state
        return {
            'last_training': None,
            'model_version': 1,
            'total_updates': 0,
            'latest_performance': {},
            'training_history': []
        }

    def save_state(self):
        """Save training state."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def get_latest_model(self) -> Optional[Tuple[xgb.XGBClassifier, Dict]]:
        """Get latest trained model."""
        try:
            model_files = list(self.models_dir.glob("xgboost_trading_model_*.joblib"))
            if not model_files:
                logger.warning("No existing model found. Will train from scratch.")
                return None

            # Sort by modification time
            model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest_model_file = model_files[0]

            # Load model
            model = joblib.load(latest_model_file)

            # Get model metadata
            model_info = {
                'file_name': latest_model_file.name,
                'modified': datetime.fromtimestamp(latest_model_file.stat().st_mtime).isoformat(),
                'version': self.state['model_version']
            }

            logger.info(f"✅ Loaded latest model: {latest_model_file.name}")
            return model, model_info

        except Exception as e:
            logger.error(f"Error loading latest model: {e}")
            return None

    def prepare_training_data(self) -> Optional[pd.DataFrame]:
        """Prepare training data with all available sources."""
        try:
            logger.info("📊 Preparing training data...")

            # Check if we have labeled data
            labeled_file = self.output_dir / 'labeled_data.parquet'
            if labeled_file.exists():
                df = pd.read_parquet(labeled_file)
                logger.info(f"📋 Loaded {len(df)} samples from labeled_data.parquet")
            else:
                # Try to load from merged data and create labels
                merged_file = self.output_dir / 'merged_7_tables.parquet'
                if not merged_file.exists():
                    logger.error("❌ No training data found. Run full pipeline first.")
                    return None

                df = pd.read_parquet(merged_file)
                logger.info(f"📋 Loaded {len(df)} samples from merged_7_tables.parquet")

                # Create labels if not present
                if 'target' not in df.columns:
                    df = self.create_labels(df)
                    # Save labeled data for future use
                    df.to_parquet(labeled_file, index=False)
                    logger.info("📋 Created and saved labels")

            # Filter for recent data (2025)
            df['time'] = pd.to_datetime(df['time'])
            df_2025 = df[df['time'].dt.year == 2025].copy()

            if len(df_2025) < self.config['min_samples_for_update']:
                logger.warning(f"⚠️ Insufficient 2025 data: {len(df_2025)} samples (need {self.config['min_samples_for_update']})")
                return None

            logger.info(f"📊 Prepared {len(df_2025)} 2025 samples for training")
            return df_2025

        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None

    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary labels for trend prediction."""
        try:
            # Sort by symbol, exchange, interval, time
            df = df.sort_values(['symbol', 'exchange', 'interval', 'time'])

            df['next_close'] = df.groupby(['symbol', 'exchange', 'interval'])['price_close'].shift(-1)

            # Calculate next period return
            df['target'] = (df['next_close'] > df['price_close']).astype(int)

            # Remove rows without next close (last records per group)
            df = df.dropna(subset=['target'])

            # Convert target to int
            df['target'] = df['target'].astype(int)

            logger.info(f"📊 Created labels: {df['target'].value_counts().to_dict()}")
            return df

        except Exception as e:
            logger.error(f"Error creating labels: {e}")
            return df

    def get_feature_list(self) -> List[str]:
        """Get list of features for training."""
        try:
            feature_file = self.output_dir / 'training_features.txt'
            if feature_file.exists():
                with open(feature_file, 'r') as f:
                    features = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                logger.info(f"📋 Loaded {len(features)} features from training_features.txt")
                return features
            else:
                # Create feature list from all numeric columns
                df = self.prepare_training_data()
                if df is not None:
                    # Exclude target and time columns
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    features = [col for col in numeric_cols if col not in ['target', 'next_close']]

                    # Save feature list
                    with open(feature_file, 'w') as f:
                        f.write('\n'.join(features))

                    logger.info(f"📋 Created feature list with {len(features)} features")
                    return features
                else:
                    return []

        except Exception as e:
            logger.error(f"Error getting feature list: {e}")
            return []

    def train_model_incremental(self, data: pd.DataFrame, existing_model: Optional[xgb.XGBClassifier] = None) -> Optional[xgb.XGBClassifier]:
        """Train model incrementally with new data."""
        try:
            features = self.get_feature_list()
            if not features:
                logger.error("❌ No features found")
                return None

            # Prepare features and target
            X = data[features]
            y = data['target']

            # Remove rows with missing values
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]

            if len(X) < self.config['min_samples_for_update']:
                logger.warning(f"⚠️ Insufficient valid samples: {len(X)}")
                return None

            logger.info(f"🏋 Training with {len(X)} samples and {len(features)} features")

            # Create DMatrix
            dtrain = xgb.DMatrix(X, label=y, feature_names=features)

            # Training parameters
            params = self.config['training']
            params['objective'] = 'binary:logistic'
            params['eval_metric'] = 'auc'
            params['verbosity'] = 1

            # Get current model parameters if exists
            if existing_model:
                params['learning_rate'] = 0.05  # Lower learning rate for incremental
                num_boost_round = 50  # Fewer rounds for incremental
            else:
                num_boost_round = 200  # More rounds for initial training

            # Train model
            evals_result = {}
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=num_boost_round,
                evals=[(dtrain, 'train')],
                evals_result=evals_result,
                early_stopping_rounds=self.config['early_stopping_rounds'],
                verbose_eval=False,
                xgb_model=existing_model  # Continue training if model exists
            )

            # Create classifier object
            classifier = xgb.XGBClassifier()
            classifier._Booster = model
            classifier.n_features_in_ = len(features)
            classifier.feature_names_in_ = features

            # Evaluate model
            predictions = classifier.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, predictions)
            y_pred = (predictions > 0.5).astype(int)
            accuracy = accuracy_score(y, y_pred)

            performance = {
                'auc': float(auc),
                'accuracy': float(accuracy),
                'samples': len(X),
                'features': len(features),
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"✅ Model trained successfully:")
            logger.info(f"   AUC: {auc:.4f}")
            logger.info(f"   Accuracy: {accuracy:.4f}")
            logger.info(f"   Samples: {len(X)}")

            return classifier

        except Exception as e:
            logger.error(f"Error training model: {e}")
            return None

    def save_model(self, model: xgb.XGBClassifier, performance: Dict):
        """Save trained model and update state."""
        try:
            # Generate model filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"xgboost_trading_model_{timestamp}.joblib"
            model_path = self.models_dir / model_filename

            # Save model
            joblib.dump(model, model_path)
            logger.info(f"💾 Saved model: {model_filename}")

            # Update state
            self.state['last_training'] = datetime.now().isoformat()
            self.state['model_version'] += 1
            self.state['total_updates'] += 1
            self.state['latest_performance'] = performance
            self.state['training_history'].append(performance)

            # Keep only last 50 training history entries
            if len(self.state['training_history']) > 50:
                self.state['training_history'] = self.state['training_history'][-50:]

            self.save_state()

            # Create symlink to latest
            latest_path = self.models_dir / 'latest_model.joblib'
            if latest_path.exists():
                latest_path.unlink()
            latest_path.symlink_to(model_path)

            # Save performance summary
            perf_file = self.output_dir / 'model_performance.json'
            with open(perf_file, 'w') as f:
                json.dump(self.state, f, indent=2)

            # Send notification
            self.send_notification(performance)

            logger.info("✅ Model and state saved successfully")

        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def send_notification(self, performance: Dict):
        """Send notification about model update."""
        if not self.config['notification']['enabled']:
            return

        try:
            message = f"""
🤖 **Model Updated Successfully!**

📊 **Performance Metrics:**
• AUC: {performance['auc']:.3f}
• Accuracy: {performance['accuracy']:.3f}
• Samples: {performance['samples']:,}
• Features: {performance['features']}

📈 **Training Info:**
• Version: {self.state['model_version']}
• Total Updates: {self.state['total_updates']}
• Last Update: {performance['timestamp']}

⏰ **Status:** Ready for QuantConnect
🤖 *XGBoost Real-time Trainer*
            """

            # Send Telegram notification
            telegram_token = self.config['notification']['telegram_token']
            telegram_chat_id = self.config['notification']['telegram_chat_id']

            if telegram_token and telegram_chat_id:
                import requests
                url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
                payload = {
                    'chat_id': telegram_chat_id,
                    'text': message,
                    'parse_mode': 'Markdown'
                }
                requests.post(url, json=payload)
                logger.info("📱 Telegram notification sent")

        except Exception as e:
            logger.error(f"Error sending notification: {e}")

    def run_training(self, trigger_file: Optional[str] = None) -> bool:
        """Run the real-time training process."""
        try:
            logger.info("🚀 Starting real-time XGBoost training")

            # Check if triggered by monitor
            if trigger_file:
                trigger_path = Path(trigger_file)
                if trigger_path.exists():
                    with open(trigger_path, 'r') as f:
                        trigger_info = json.load(f)
                    logger.info(f"📋 Trigger: {trigger_info.get('trigger_reason', 'Unknown')}")
                    logger.info(f"📊 Tables: {trigger_info.get('tables_with_new_data', [])}")

            # Load existing model
            existing_model, model_info = self.get_latest_model()

            # Prepare training data
            training_data = self.prepare_training_data()
            if training_data is None:
                logger.error("❌ Failed to prepare training data")
                return False

            # Train model
            model = self.train_model_incremental(training_data, existing_model)
            if model is None:
                logger.error("❌ Model training failed")
                return False

            # Evaluate on training data
            features = self.get_feature_list()
            X = training_data[features]
            y = training_data['target']
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X_eval = X[mask]
            y_eval = y[mask]

            predictions = model.predict_proba(X_eval)[:, 1]
            auc = roc_auc_score(y_eval, predictions)
            accuracy = accuracy_score(y_eval, (predictions > 0.5).astype(int))

            performance = {
                'auc': float(auc),
                'accuracy': float(accuracy),
                'samples': len(X_eval),
                'features': len(features),
                'timestamp': datetime.now().isoformat()
            }

            # Check performance threshold
            if auc < self.config['performance_threshold']:
                logger.warning(f"⚠️ Performance below threshold: {auc:.3f} < {self.config['performance_threshold']}")
                # Still save but with warning

            # Save model
            self.save_model(model, performance)

            logger.info("✅ Real-time training completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error in run_training: {e}")
            return False

    def run_full_training(self) -> bool:
        """Run full training from scratch."""
        logger.info("🏋 Running full training from scratch")
        return self.run_training()


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Real-time XGBoost trainer')
    parser.add_argument('--trigger-file', type=str, help='Trigger file path')
    parser.add_argument('--mode', type=str, choices=['incremental', 'full'], default='incremental',
                      help='Training mode')
    parser.add_argument('--output-dir', type=str, default='./output_train',
                      help='Output directory')
    parser.add_argument('--test', action='store_true', help='Test mode')

    args = parser.parse_args()

    trainer = RealtimeXGBoostTrainer(output_dir=args.output_dir)

    if args.test:
        logger.info("🧪 Running in test mode")
        data = trainer.prepare_training_data()
        if data is not None:
            print(f"✅ Test successful - found {len(data)} samples")
        else:
            print("❌ Test failed - no data found")
        return

    success = False
    if args.mode == 'full':
        success = trainer.run_full_training()
    else:
        success = trainer.run_training(args.trigger_file)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()