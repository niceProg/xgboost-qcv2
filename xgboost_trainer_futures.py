#!/usr/bin/env python3
"""
XGBoost trainer for binary classification of futures trend prediction.
Trains XGBoost model on engineered futures features and binary labels.
Optimized for futures trading with Open Interest, Liquidation, and Orderbook data.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import logging
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import json
import shutil
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class XGBoostTrainerFutures:
    """Train XGBoost model for futures trend prediction."""

    def __init__(self, data_filter, output_dir: str = './output_train_futures'):
        self.data_filter = data_filter
        self.output_dir = Path(output_dir)
        self.model = None
        self.feature_names = None
        self.best_params = None

    def load_training_data(self) -> tuple:
        """Load prepared training data."""
        logger.info("Loading training data...")

        datasets_dir = self.output_dir / 'datasets'
        X_file = datasets_dir / 'X_train_features_futures.parquet'
        y_file = datasets_dir / 'y_train_labels_futures.parquet'

        if not X_file.exists():
            X_file = self.output_dir / 'X_train_features_futures.parquet'
        if not y_file.exists():
            y_file = self.output_dir / 'y_train_labels_futures.parquet'

        if not X_file.exists() or not y_file.exists():
            logger.error("Training data files not found. Run label_builder_futures.py first.")
            sys.exit(1)

        try:
            X = pd.read_parquet(X_file)
            y = pd.read_parquet(y_file).iloc[:, 0]

            logger.info(f"Loaded features: {X.shape}")
            logger.info(f"Loaded labels: {y.shape}")
            logger.info(f"Label distribution: {y.value_counts().to_dict()}")

            self.feature_names = list(X.columns)
            return X, y

        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            sys.exit(1)

    def prepare_data_splits(self, X: pd.DataFrame, y: pd.Series,
                          test_size: float = 0.2,
                          validation_size: float = 0.2,
                          random_state: int = 42) -> tuple:
        """Prepare train/validation/test splits with temporal consideration."""
        logger.info("Preparing data splits...")

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=validation_size,
            random_state=random_state, stratify=y_temp
        )

        logger.info(f"Train set: {X_train.shape[0]} samples ({y_train.mean():.3f} bullish)")
        logger.info(f"Validation set: {X_val.shape[0]} samples ({y_val.mean():.3f} bullish)")
        logger.info(f"Test set: {X_test.shape[0]} samples ({y_test.mean():.3f} bullish)")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_default_xgboost_params(self) -> dict:
        """Get default XGBoost parameters for binary classification."""
        return {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc'],
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,
            'alpha': 0,
            'lambda': 1,
            'random_state': 42,
            'n_estimators': 100,
            'early_stopping_rounds': 10,
            'verbosity': 1
        }

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series,
                   params: dict = None) -> xgb.XGBClassifier:
        """Train XGBoost model with validation."""
        logger.info("Training XGBoost model...")

        if params is None:
            params = self.get_default_xgboost_params()

        logger.info(f"Using parameters: {params}")

        model = xgb.XGBClassifier(**params)

        eval_set = [(X_train, y_train), (X_val, y_val)]
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )

        results = model.evals_result()
        train_auc = results['validation_0']['auc'][-1]
        val_auc = results['validation_1']['auc'][-1]
        train_logloss = results['validation_0']['logloss'][-1]
        val_logloss = results['validation_1']['logloss'][-1]

        logger.info(f"Training completed:")
        logger.info(f"  Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
        logger.info(f"  Train LogLoss: {train_logloss:.4f}, Val LogLoss: {val_logloss:.4f}")
        logger.info(f"  Best iteration: {model.best_iteration}")

        self.model = model
        self.best_params = params
        return model

    def evaluate_model(self, model: xgb.XGBClassifier,
                      X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evaluate model performance on test set."""
        logger.info("Evaluating model performance...")

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='binary', zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }

        logger.info("Test Set Performance:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        logger.info("Confusion Matrix:")
        logger.info(f"  True Negatives: {cm[0,0]}")
        logger.info(f"  False Positives: {cm[0,1]}")
        logger.info(f"  False Negatives: {cm[1,0]}")
        logger.info(f"  True Positives: {cm[1,1]}")

        return metrics, cm

    def feature_importance_analysis(self, model: xgb.XGBClassifier) -> pd.DataFrame:
        """Analyze and log feature importance."""
        logger.info("Analyzing feature importance...")

        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        logger.info("Top 15 Most Important Features:")
        for idx, row in feature_importance.head(15).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        return feature_importance

    def cross_validation(self, X: pd.DataFrame, y: pd.Series,
                        cv_folds: int = 5, params: dict = None) -> dict:
        """Perform time series cross validation."""
        logger.info("Performing cross validation...")

        if params is None:
            params = self.get_default_xgboost_params()

        tscv = TimeSeriesSplit(n_splits=cv_folds)

        cv_params = params.copy()
        cv_params.pop('early_stopping_rounds', None)
        cv_params['n_estimators'] = 50

        model = xgb.XGBClassifier(**cv_params)

        cv_scores = cross_val_score(
            model, X, y, cv=tscv, scoring='roc_auc', n_jobs=-1
        )

        cv_results = {
            'mean_auc': cv_scores.mean(),
            'std_auc': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }

        logger.info(f"Cross Validation Results ({cv_folds} folds):")
        logger.info(f"  Mean AUC: {cv_results['mean_auc']:.4f} +/- {cv_results['std_auc']:.4f}")
        logger.info(f"  Scores: {[f'{score:.4f}' for score in cv_scores]}")

        return cv_results

    def save_model_and_results(self, model: xgb.XGBClassifier,
                             metrics: dict, cv_results: dict,
                             feature_importance: pd.DataFrame):
        """Save model and training results."""
        logger.info("Saving model and results...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"xgboost_futures_model_{timestamp}.joblib"

        models_dir = self.output_dir / 'models'
        models_dir.mkdir(exist_ok=True)

        model_path = models_dir / model_name
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")

        latest_model_path = models_dir / 'latest_futures_model.joblib'
        joblib.dump(model, latest_model_path)
        logger.info(f"Latest model saved to {latest_model_path}")

        root_latest_path = self.output_dir / 'latest_futures_model.joblib'
        if root_latest_path.exists():
            root_latest_path.unlink()
        shutil.copy2(latest_model_path, root_latest_path)
        logger.info(f"Latest model copied to root: {root_latest_path}")

        # Save training results
        results = {
            'model_name': model_name,
            'model_type': 'futures',
            'timestamp': timestamp,
            'parameters': self.best_params,
            'metrics': metrics,
            'cross_validation': cv_results,
            'feature_count': len(self.feature_names),
            'sample_count': len(self.feature_names)
        }

        results_path = self.output_dir / f'training_results_futures_{timestamp}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_path}")

        # Save feature importance
        importance_path = self.output_dir / f'feature_importance_futures_{timestamp}.csv'
        feature_importance.to_csv(importance_path, index=False)
        logger.info(f"Feature importance saved to {importance_path}")

        # Save feature names
        features_path = self.output_dir / 'model_features_futures.txt'
        with open(features_path, 'w') as f:
            f.write("Model Features (Futures):\n")
            f.write("=" * 30 + "\n")
            for feature in self.feature_names:
                f.write(f"{feature}\n")
        logger.info(f"Feature list saved to {features_path}")

        # Save dataset summary for QuantConnect
        summary_file, summary_content = self.save_dataset_summary(results, feature_importance)

        # Save to database
        if os.getenv('ENABLE_DB_STORAGE', 'true').lower() == 'true':
            self.save_to_database(model, model_name, results, summary_file, summary_content)

    def save_dataset_summary(self, results: dict, feature_importance: pd.DataFrame):
        """Save dataset summary for QuantConnect backtest."""
        try:
            import pytz
            jakarta_tz = pytz.timezone('Asia/Jakarta')
            current_time = datetime.now(jakarta_tz)

            summary_dir = self.output_dir / 'datasets' / 'summary'
            summary_dir.mkdir(parents=True, exist_ok=True)

            timestamp = current_time.strftime('%Y%m%d_%H%M%S')
            summary_filename = f'qc_summary_futures_{timestamp}.txt'
            summary_file = summary_dir / summary_filename

            summary_content = f"""============================================
XGBoost Futures Trading Model - Summary
============================================

Model Information:
- Model Name: {results['model_name']}
- Model Type: Futures Trading
- Timestamp: {current_time.strftime('%Y-%m-%d %H:%M:%S WIB')}
- Training Date: {results['timestamp']}

Training Parameters:
{json.dumps(results['parameters'], indent=2)}

Test Set Performance:
- Accuracy: {results['metrics']['accuracy']:.4f}
- Precision: {results['metrics']['precision']:.4f}
- Recall: {results['metrics']['recall']:.4f}
- F1 Score: {results['metrics']['f1']:.4f}
- ROC AUC: {results['metrics']['roc_auc']:.4f}

Cross Validation:
- Mean AUC: {results['cross_validation']['mean_auc']:.4f} +/- {results['cross_validation']['std_auc']:.4f}
- All Scores: {[f'{s:.4f}' for s in results['cross_validation']['cv_scores']]}

Feature Information:
- Total Features: {results['feature_count']}
- Top 10 Features:
"""

            for idx, row in feature_importance.head(10).iterrows():
                summary_content += f"  {idx+1}. {row['feature']}: {row['importance']:.4f}\n"

            summary_content += f"""
Model Files:
- Model: {results['model_name']}
- Latest: latest_futures_model.joblib
- Features: model_features_futures.txt
- Results: training_results_futures_{results['timestamp']}.json

============================================
Ready for QuantConnect Backtest
============================================
"""

            with open(summary_file, 'w') as f:
                f.write(summary_content)

            logger.info(f"QuantConnect summary saved to {summary_file}")
            return summary_filename, summary_content.encode('utf-8')

        except Exception as e:
            logger.warning(f"Could not save dataset summary: {e}")
            return None, None

    def save_to_database(self, model: xgb.XGBClassifier, model_name: str,
                        results: dict, summary_file: str, summary_data: bytes):
        """Save model and dataset summary to database."""
        try:
            from database_storage_futures import DatabaseStorageFutures

            # DatabaseStorageFutures will read config from .env automatically
            db_storage = DatabaseStorageFutures()

            # Store model with model_version='futures'
            model_id = db_storage.store_model(
                model=model,
                model_name=model_name,
                feature_names=self.feature_names if hasattr(self, 'feature_names') else [],
                hyperparams=results['parameters'],
                train_score=results['metrics'].get('train_auc', 0),
                val_score=results['metrics'].get('val_auc', 0),
                cv_scores=results['cross_validation'].get('cv_scores', []),
                model_version='futures'
            )

            logger.info(f"Model saved to database with ID: {model_id}")

            # Store dataset summary
            if summary_file and summary_data:
                summary_id = db_storage.store_dataset_summary(
                    summary_file=summary_file,
                    summary_data=summary_data,
                    model_version='futures'
                )
                logger.info(f"Dataset summary saved to database with ID: {summary_id}")

        except Exception as e:
            logger.warning(f"Failed to save to database: {e}")

    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series) -> dict:
        """Basic hyperparameter tuning."""
        logger.info("Performing hyperparameter tuning...")

        base_params = self.get_default_xgboost_params()
        best_auc = 0
        best_params = base_params.copy()

        param_grid = {
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [4, 6, 8],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }

        for lr in param_grid['learning_rate']:
            for md in param_grid['max_depth']:
                for mcw in param_grid['min_child_weight']:
                    for ss in param_grid['subsample']:
                        for csbt in param_grid['colsample_bytree']:
                            test_params = base_params.copy()
                            test_params.update({
                                'learning_rate': lr,
                                'max_depth': md,
                                'min_child_weight': mcw,
                                'subsample': ss,
                                'colsample_bytree': csbt
                            })

                            model = xgb.XGBClassifier(**test_params)
                            model.fit(
                                X_train, y_train,
                                eval_set=[(X_val, y_val)],
                                verbose=False
                            )

                            val_pred_proba = model.predict_proba(X_val)[:, 1]
                            val_auc = roc_auc_score(y_val, val_pred_proba)

                            if val_auc > best_auc:
                                best_auc = val_auc
                                best_params = test_params.copy()

        logger.info(f"Best validation AUC: {best_auc:.4f}")
        logger.info(f"Best parameters: {best_params}")

        return best_params


class DataFilter:
    """Simple data filter."""

    def __init__(self, args):
        self.args = args


def parse_arguments():
    """Parse command-line arguments."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Futures XGBoost Training Pipeline - Train Model"
    )
    parser.add_argument('--output-dir', type=str, default='./output_train_futures')
    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_arguments()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    data_filter = DataFilter(args)

    trainer = XGBoostTrainerFutures(data_filter, args.output_dir)

    try:
        # Load training data
        logger.info("Loading training data...")
        X, y = trainer.load_training_data()

        # Prepare data splits
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data_splits(X, y)

        # Hyperparameter tuning
        logger.info("Performing hyperparameter tuning...")
        best_params = trainer.hyperparameter_tuning(X_train, y_train, X_val, y_val)

        # Train final model
        logger.info("Training final model...")
        model = trainer.train_model(X_train, y_train, X_val, y_val, best_params)

        # Evaluate model
        metrics, cm = trainer.evaluate_model(model, X_test, y_test)

        # Feature importance
        feature_importance = trainer.feature_importance_analysis(model)

        # Cross validation
        cv_results = trainer.cross_validation(X, y)

        # Save everything
        trainer.save_model_and_results(model, metrics, cv_results, feature_importance)

        logger.info("\n=== Training Complete ===")
        logger.info("Model ready for QuantConnect backtest")
        logger.info(f"Model location: {args.output_dir}/latest_futures_model.joblib")

    except Exception as e:
        logger.error(f"Error in training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
