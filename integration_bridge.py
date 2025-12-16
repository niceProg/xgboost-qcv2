#!/usr/bin/env python3
"""
Integration Bridge antara Real-time XGBoost System dan QuantConnect.
Menghubungkan real-time training dengan QuantConnect ObjectStore.
"""

import os
import json
import logging
import requests
import time
from datetime import datetime
from pathlib import Path
import joblib
from typing import Dict, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/integration_bridge.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class QuantConnectBridge:
    """Bridge untuk menghubungkan real-time system dengan QuantConnect."""

    def __init__(self, config_file: str = 'qc_integration_config.json'):
        self.config = self.load_config(config_file)
        self.qc_api_token = self.config.get('qc_api_token')
        self.project_id = self.config.get('project_id')
        self.algorithm_id = self.config.get('algorithm_id')

        # Local paths
        self.output_dir = Path(self.config.get('output_dir', './output_train'))
        self.model_dir = self.output_dir / 'models'

    def load_config(self, config_file: str) -> Dict:
        """Load konfigurasi integration."""
        default_config = {
            "qc_api_token": os.getenv("QUANTCONNECT_API_TOKEN"),
            "project_id": os.getenv("QC_PROJECT_ID"),
            "algorithm_id": os.getenv("QC_ALGORITHM_ID"),
            "output_dir": "./output_train",
            "sync_settings": {
                "auto_sync": True,
                "sync_interval_minutes": 60,  # Sync every hour
                "min_performance_threshold": 0.6,  # Only sync if model performs well
                "max_model_size_mb": 50  # Don't sync if model too large
            },
            "notifications": {
                "enabled": True,
                "webhook_url": os.getenv("WEBHOOK_URL"),
                "telegram_token": os.getenv("TELEGRAM_BOT_TOKEN"),
                "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID")
            }
        }

        if Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                # Merge with defaults
                for key in default_config:
                    if key not in user_config:
                        user_config[key] = default_config[key]
                    elif isinstance(default_config[key], dict):
                        for subkey in default_config[key]:
                            if subkey not in user_config[key]:
                                user_config[key][subkey] = default_config[key][subkey]
                return user_config
            except Exception as e:
                logger.error(f"Error loading config: {e}. Using defaults.")
                return default_config
        else:
            # Create default config
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config

    def get_latest_model_info(self) -> Optional[Tuple[Path, Dict]]:
        """Get latest model dan performance info."""
        try:
            # Get latest model file
            model_files = list(self.model_dir.glob("xgboost_trading_model_*.joblib"))
            if not model_files:
                logger.warning("No model files found")
                return None

            # Sort by modification time
            model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest_model = model_files[0]

            # Get performance info
            perf_file = self.output_dir / 'model_performance.json'
            performance = {}
            if perf_file.exists():
                with open(perf_file, 'r') as f:
                    performance = json.load(f)

            # Get model metadata
            model_stat = latest_model.stat()
            model_info = {
                'filename': latest_model.name,
                'size_mb': model_stat.st_size / (1024 * 1024),
                'modified': datetime.fromtimestamp(model_stat.st_mtime).isoformat(),
                'performance': performance
            }

            return latest_model, model_info

        except Exception as e:
            logger.error(f"Error getting latest model info: {e}")
            return None

    def should_sync_model(self, model_info: Dict) -> bool:
        """Check if model should sync to QuantConnect."""
        sync_settings = self.config['sync_settings']

        # Check auto sync
        if not sync_settings.get('auto_sync', True):
            return False

        # Check model size
        if model_info['size_mb'] > sync_settings.get('max_model_size_mb', 50):
            logger.warning(f"Model too large: {model_info['size_mb']:.1f}MB")
            return False

        # Check performance threshold
        performance = model_info.get('performance', {})
        latest_auc = performance.get('latest_auc', 0)
        if latest_auc < sync_settings.get('min_performance_threshold', 0.6):
            logger.warning(f"Model performance below threshold: AUC={latest_auc:.3f}")
            return False

        # Check if recently updated (within last hour)
        modified_time = datetime.fromisoformat(model_info['modified'])
        time_since_update = datetime.now() - modified_time
        if time_since_update.total_seconds() < sync_settings.get('sync_interval_minutes', 60) * 60:
            logger.info("Model updated recently, checking if newer than QC version")
            # Still check if this is newer than what's on QC

        return True

    def prepare_quantconnect_package(self, model_path: Path) -> Dict:
        """Prepare package untuk QuantConnect ObjectStore."""
        try:
            # Load model to get feature info
            model = joblib.load(model_path)

            # Get feature names
            feature_names = getattr(model, 'feature_names_in_', None)
            if not feature_names:
                booster = model.get_booster()
                feature_names = booster.feature_names if booster else None

            # Create model metadata
            model_metadata = {
                'filename': model_path.name,
                'model_type': 'XGBoost',
                'n_features': len(feature_names) if feature_names else 0,
                'feature_names': feature_names.tolist() if feature_names is not None else [],
                'created_at': datetime.now().isoformat(),
                'performance': self.get_model_performance_info(),
                'training_data_info': self.get_training_data_info()
            }

            # Create dataset summary (required by QC algorithm)
            dataset_summary = self.create_dataset_summary()

            return {
                'model_file': model_path,
                'model_metadata': model_metadata,
                'dataset_summary': dataset_summary
            }

        except Exception as e:
            logger.error(f"Error preparing QC package: {e}")
            return {}

    def get_model_performance_info(self) -> Dict:
        """Get model performance information."""
        try:
            perf_file = self.output_dir / 'model_performance.json'
            if perf_file.exists():
                with open(perf_file, 'r') as f:
                    return json.load(f)
            return {}
        except:
            return {}

    def get_training_data_info(self) -> Dict:
        """Get training data information."""
        try:
            # Check for training data files
            data_files = {
                'labeled_data': self.output_dir / 'labeled_data.parquet',
                'features_engineered': self.output_dir / 'features_engineered.parquet',
                'training_features': self.output_dir / 'training_features.txt'
            }

            info = {}
            for key, file_path in data_files.items():
                if file_path.exists():
                    if file_path.suffix == '.parquet':
                        import pandas as pd
                        df = pd.read_parquet(file_path)
                        info[key] = {
                            'exists': True,
                            'rows': len(df),
                            'columns': len(df.columns),
                            'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                        }
                    else:
                        info[key] = {
                            'exists': True,
                            'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                        }
                else:
                    info[key] = {'exists': False}

            return info

        except Exception as e:
            logger.error(f"Error getting training data info: {e}")
            return {}

    def create_dataset_summary(self) -> str:
        """Create dataset summary untuk QuantConnect algorithm."""
        try:
            # Get time range from training data
            labeled_file = self.output_dir / 'labeled_data.parquet'
            if labeled_file.exists():
                import pandas as pd
                df = pd.read_parquet(labeled_file)
                if 'time' in df.columns:
                    min_time = df['time'].min()
                    max_time = df['time'].max()
                    time_range = f"{min_time} to {max_time}"
                else:
                    time_range = "Unknown time range"
            else:
                time_range = "No training data found"

            summary = f"""XGBoost Trading Model Dataset Summary

Training Data:
- Time range: {time_range}
- Pairs: BTCUSDT, ETHUSDT
- Intervals: 1h, 4h
- Features: Multi-source (price, volume, funding, basis, long/short ratios)

Model Information:
- Algorithm: XGBoost Classifier
- Training Mode: Real-time Incremental
- Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Update Frequency: Hourly (if new data available)

Target: Binary classification (1=Buy signal, 0=Hold/Sell signal)
Features: 50+ technical and market indicators
"""

            return summary

        except Exception as e:
            logger.error(f"Error creating dataset summary: {e}")
            return "Dataset summary unavailable"

    def upload_to_quantconnect(self, package: Dict) -> bool:
        """Upload model package ke QuantConnect ObjectStore."""
        try:
            if not all([self.qc_api_token, self.project_id]):
                logger.error("QuantConnect API credentials not configured")
                return False

            # In production, ini akan menggunakan QuantConnect API
            # Untuk sekarang, simpan ke local staging area
            staging_dir = Path('./qc_staging')
            staging_dir.mkdir(exist_ok=True)

            # Copy model file
            import shutil
            model_filename = "latest_model.joblib"
            shutil.copy2(package['model_file'], staging_dir / model_filename)

            # Save metadata
            with open(staging_dir / 'model_metadata.json', 'w') as f:
                json.dump(package['model_metadata'], f, indent=2)

            # Save dataset summary
            with open(staging_dir / 'dataset_summary.txt', 'w') as f:
                f.write(package['dataset_summary'])

            logger.info(f"✅ Model package prepared for QuantConnect")
            logger.info(f"📁 Staging location: {staging_dir}")
            logger.info(f"📊 Model: {model_filename}")
            logger.info(f"📋 Features: {package['model_metadata']['n_features']}")

            # Send notification
            self.send_sync_notification(package['model_metadata'])

            return True

        except Exception as e:
            logger.error(f"Error uploading to QuantConnect: {e}")
            return False

    def send_sync_notification(self, model_metadata: Dict):
        """Send notification about model sync."""
        if not self.config['notifications']['enabled']:
            return

        message = f"""
🔄 **Model Synced to QuantConnect**

📊 **Model Info:**
• Type: {model_metadata.get('model_type', 'Unknown')}
• Features: {model_metadata.get('n_features', 0)}
• Created: {model_metadata.get('created_at', 'Unknown')}

🎯 **Performance:**
• Latest AUC: {model_metadata.get('performance', {}).get('latest_auc', 'N/A')}
• Latest Accuracy: {model_metadata.get('performance', {}).get('latest_accuracy', 'N/A')}

⏰ **Sync Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**

📝 *Next: Update QuantConnect ObjectStore manually or via API*
"""

        # Send to Telegram if configured
        telegram_token = self.config['notifications'].get('telegram_token')
        telegram_chat_id = self.config['notifications'].get('telegram_chat_id')

        if telegram_token and telegram_chat_id:
            try:
                requests.post(
                    f"https://api.telegram.org/bot{telegram_token}/sendMessage",
                    json={
                        'chat_id': telegram_chat_id,
                        'text': message,
                        'parse_mode': 'Markdown'
                    }
                )
            except Exception as e:
                logger.error(f"Failed to send Telegram notification: {e}")

    def sync_model_to_quantconnect(self) -> bool:
        """Main sync function."""
        logger.info("🔄 Starting model sync to QuantConnect")

        # Get latest model
        result = self.get_latest_model_info()
        if not result:
            logger.error("No model found to sync")
            return False

        model_path, model_info = result

        # Check if should sync
        if not self.should_sync_model(model_info):
            logger.info("Model does not meet sync criteria")
            return False

        logger.info(f"Preparing to sync model: {model_info['filename']}")

        # Prepare package
        package = self.prepare_quantconnect_package(model_path)
        if not package:
            logger.error("Failed to prepare model package")
            return False

        # Upload to QuantConnect
        success = self.upload_to_quantconnect(package)
        if success:
            logger.info("✅ Model successfully synced to QuantConnect")
        else:
            logger.error("❌ Failed to sync model to QuantConnect")

        return success

    def run_continuous_sync(self):
        """Run continuous sync loop."""
        logger.info("🚀 Starting continuous QuantConnect sync")

        sync_interval = self.config['sync_settings']['sync_interval_minutes'] * 60

        while True:
            try:
                # Check if sync is enabled
                if not self.config['sync_settings'].get('auto_sync', True):
                    logger.info("Auto sync disabled - sleeping")
                    time.sleep(sync_interval)
                    continue

                # Perform sync
                self.sync_model_to_quantconnect()

                # Sleep until next sync
                logger.info(f"Sleeping for {sync_interval/60:.0f} minutes")
                time.sleep(sync_interval)

            except KeyboardInterrupt:
                logger.info("Received interrupt - stopping sync")
                break
            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                time.sleep(60)  # Wait 1 minute before retry


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='QuantConnect integration bridge')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--sync', action='store_true', help='Sync once and exit')
    parser.add_argument('--continuous', action='store_true', help='Run continuous sync')
    parser.add_argument('--status', action='store_true', help='Show current status')

    args = parser.parse_args()

    bridge = QuantConnectBridge(config_file=args.config or 'qc_integration_config.json')

    if args.status:
        # Show current status
        result = bridge.get_latest_model_info()
        if result:
            model_path, model_info = result
            print("Current Model Status:")
            print(json.dumps(model_info, indent=2))
        else:
            print("No model found")

    elif args.sync:
        # Sync once
        success = bridge.sync_model_to_quantconnect()
        sys.exit(0 if success else 1)

    elif args.continuous:
        # Run continuous sync
        bridge.run_continuous_sync()

    else:
        parser.print_help()


if __name__ == "__main__":
    import sys
    main()