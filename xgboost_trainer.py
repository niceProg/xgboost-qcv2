#!/usr/bin/env python3
"""
XGBoost trainer for binary classification of trend prediction.
Trains XGBoost model on engineered features and binary labels.
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

# Import our command line options handler
from command_line_options import parse_arguments, validate_arguments, DataFilter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class XGBoostTrainer:
    """Train XGBoost model for trend prediction."""

    def __init__(self, data_filter: DataFilter, output_dir: str = './output_train', price_only_mode: bool = False):
        self.data_filter = data_filter
        self.output_dir = Path(output_dir)
        self.model = None
        self.feature_names = None
        self.best_params = None
        self.price_only_mode = price_only_mode

    def load_training_data(self) -> tuple:
        """Load prepared training data."""
        logger.info("Loading training data...")

        # Try datasets directory first (new structure), then root (compatibility)
        datasets_dir = self.output_dir / 'datasets'
        X_file = datasets_dir / 'X_train_features.parquet'
        y_file = datasets_dir / 'y_train_labels.parquet'

        if not X_file.exists() or not y_file.exists():
            # Fallback to root directory for backward compatibility
            X_file = self.output_dir / 'X_train_features.parquet'
            y_file = self.output_dir / 'y_train_labels.parquet'

        if not X_file.exists() or not y_file.exists():
            logger.error("Training data files not found. Run label_builder.py first.")
            logger.error(f"Checked: {datasets_dir / 'X_train_features.parquet'} and {self.output_dir / 'X_train_features.parquet'}")
            sys.exit(1)

        try:
            X = pd.read_parquet(X_file)
            y = pd.read_parquet(y_file).iloc[:, 0]  # Get first column as Series

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
        """
        Prepare train/validation/test splits with TIME-BASED split.

        For time-series data (futures trading), we MUST split by time, not randomly.
        Train = oldest 70%, Val = next 10%, Test = newest 20%

        This prevents data leakage from future into past.
        """
        logger.info("Preparing time-based data splits (no random split for time series)...")

        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate split points
        # Train: 70%, Val: 10%, Test: 20% (validation_size=0.2 of remaining = 0.2 * 0.8 = 0.16, adjusted to 0.1)
        train_end = int(n_samples * 0.70)
        val_end = int(n_samples * 0.80)  # 70% + 10% = 80%

        # Split indices
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        # Create splits
        X_train = X.iloc[train_idx].copy()
        y_train = y.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_val = y.iloc[val_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_test = y.iloc[test_idx].copy()

        logger.info(f"Time-based split:")
        logger.info(f"  Train set: {X_train.shape[0]} samples ({y_train.mean():.3f} bullish) - oldest 70%")
        logger.info(f"  Validation set: {X_val.shape[0]} samples ({y_val.mean():.3f} bullish) - next 10%")
        logger.info(f"  Test set: {X_test.shape[0]} samples ({y_test.mean():.3f} bullish) - newest 20%")
        logger.info(f"  Note: No data leakage - test data is strictly future data")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_default_xgboost_params(self) -> dict:
        """Get default XGBoost parameters for binary classification (V1 spot config)."""
        return {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc'],
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,
            'alpha': 0,  # L1 regularization
            'lambda': 1,  # L2 regularization
            'random_state': 42,
            'n_estimators': 100,
            'early_stopping_rounds': 10,
            'verbosity': 1
        }

    def calculate_scale_pos_weight(self, y: pd.Series) -> float:
        """Calculate scale_pos_weight for imbalanced classes."""
        pos_count = (y == 1).sum()
        neg_count = (y == 0).sum()

        if pos_count == 0:
            return 1.0

        scale = neg_count / pos_count
        logger.info(f"Class imbalance - Negative: {neg_count}, Positive: {pos_count}, scale_pos_weight: {scale:.2f}")
        return scale

    def select_features(self, X: pd.DataFrame, y: pd.Series,
                       importance_threshold: float = 0.01) -> tuple:
        """
        Feature selection using quick XGBoost to find importance.
        Drops features with importance below threshold (default 1%).

        Returns:
            X_selected: DataFrame with selected features only
            selected_features: List of selected feature names
        """
        logger.info(f"Performing feature selection (threshold={importance_threshold*100:.1f}%)...")

        # Train quick model for feature importance
        quick_params = {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc'],
            'learning_rate': 0.1,
            'max_depth': 4,  # Smaller for speed
            'n_estimators': 30,  # Fewer trees for speed
            'random_state': 42,
            'verbosity': 0
        }

        quick_model = xgb.XGBClassifier(**quick_params)
        quick_model.fit(X, y)

        # Get feature importance
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': quick_model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Select features above threshold
        selected_features = importance[importance['importance'] >= importance_threshold]['feature'].tolist()
        dropped_features = importance[importance['importance'] < importance_threshold]['feature'].tolist()

        logger.info(f"Feature selection: {len(X.columns)} -> {len(selected_features)} features")
        logger.info(f"Dropped {len(dropped_features)} low-importance features")

        if dropped_features:
            logger.info(f"Dropped features (importance < {importance_threshold*100:.1f}%):")
            for feat, imp in zip(dropped_features, importance[importance['importance'] < importance_threshold]['importance']):
                logger.info(f"  {feat}: {imp*100:.3f}%")

        return X[selected_features], selected_features

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series,
                   params: dict = None) -> xgb.XGBClassifier:
        """Train XGBoost model with validation."""
        logger.info("Training XGBoost model...")

        if params is None:
            params = self.get_default_xgboost_params()

        # Add scale_pos_weight for class imbalance (Quick Win #3)
        if 'scale_pos_weight' not in params:
            params['scale_pos_weight'] = self.calculate_scale_pos_weight(y_train)

        logger.info(f"Using parameters: {params}")

        # Create model
        model = xgb.XGBClassifier(**params)

        # Train with evaluation set
        eval_set = [(X_train, y_train), (X_val, y_val)]
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )

        # Log training results
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

        # Make predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'f1': f1_score(y_test, y_pred, average='binary'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }

        # Log metrics
        logger.info("Test Set Performance:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        # Log confusion matrix
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

        # Get feature importance
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        # Log top features
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

        # Create time series split
        tscv = TimeSeriesSplit(n_splits=cv_folds)

        # Create a copy of params without early_stopping_rounds for CV
        cv_params = params.copy()
        cv_params.pop('early_stopping_rounds', None)
        cv_params['n_estimators'] = 100  # Increased from 50 for better CV results

        # Perform cross validation
        model = xgb.XGBClassifier(**cv_params)

        # Cross validate on AUC
        cv_scores = cross_val_score(
            model, X, y, cv=tscv, scoring='roc_auc', n_jobs=-1
        )

        cv_results = {
            'mean_auc': cv_scores.mean(),
            'std_auc': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }

        logger.info(f"Cross Validation Results ({cv_folds} folds):")
        logger.info(f"  Mean AUC: {cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}")
        logger.info(f"  Scores: {[f'{score:.4f}' for score in cv_scores]}")

        return cv_results

    def save_model_and_results(self, model: xgb.XGBClassifier,
                             metrics: dict, cv_results: dict,
                             feature_importance: pd.DataFrame,
                             sample_count: int = None):
        """Save model and training results."""
        logger.info("Saving model and results...")

        # Create timestamp for model version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"xgboost_trading_model_{timestamp}.joblib"

        # Create models directory if it doesn't exist
        models_dir = self.output_dir / 'models'
        models_dir.mkdir(exist_ok=True)

        # Save model to models directory
        model_path = models_dir / model_name
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")

        # Save as latest model (overwrite existing)
        latest_model_path = models_dir / 'latest_model.joblib'
        joblib.dump(model, latest_model_path)
        logger.info(f"Latest model saved to {latest_model_path}")

        # No symlink creation - use direct file copies instead
        # Copy latest model to root for backward compatibility
        root_latest_path = self.output_dir / 'latest_model.joblib'
        if root_latest_path.exists():
            root_latest_path.unlink()
        shutil.copy2(latest_model_path, root_latest_path)
        logger.info(f"Latest model copied to root: {root_latest_path}")

        # Use provided sample_count or fallback to feature count (backward compat)
        actual_sample_count = sample_count if sample_count is not None else len(self.feature_names)

        # Save training results
        results = {
            'model_name': model_name,
            'timestamp': timestamp,
            'parameters': self.best_params,
            'metrics': metrics,
            'cross_validation': cv_results,
            'feature_count': len(self.feature_names),
            'sample_count': actual_sample_count
        }

        results_path = self.output_dir / f'training_results_{timestamp}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_path}")

        # Save feature importance
        importance_path = self.output_dir / f'feature_importance_{timestamp}.csv'
        feature_importance.to_csv(importance_path, index=False)
        logger.info(f"Feature importance saved to {importance_path}")

        # Save feature names
        features_path = self.output_dir / 'model_features.txt'
        with open(features_path, 'w') as f:
            f.write("Model Features:\n")
            f.write("=" * 20 + "\n")
            for feature in self.feature_names:
                f.write(f"{feature}\n")
        logger.info(f"Feature list saved to {features_path}")

        # Also save to database if enabled
        if os.getenv('ENABLE_DB_STORAGE', 'true').lower() == 'true':
            self.save_to_database(model, model_name, results)
            # DATABASE STORAGE DISABLED per client requirement
            # self.update_training_session(model_name, results)

    def save_to_database(self, model: xgb.XGBClassifier, model_name: str, results: dict):
        """Save model and results to xgboostqc database."""
        try:
            from database_storage import DatabaseStorage

            # Use xgboostqc database config
            db_storage = DatabaseStorage(
                db_config={
                    'host': os.getenv('DB_HOST', '103.150.81.86'),
                    'port': int(os.getenv('DB_PORT', 3306)),
                    'database': os.getenv('DB_NAME', 'xgboostqc'),
                    'user': os.getenv('DB_USER', 'xgboostqc'),
                    'password': os.getenv('DB_PASSWORD', '6SPxBDwXH6WyxpfT')
                }
            )

            # Store model
            db_storage.store_model(
                model=model,
                model_name=model_name,
                feature_names=self.feature_names if hasattr(self, 'feature_names') else [],
                hyperparams=results['parameters'],
                train_score=results['metrics'].get('train_auc', 0),
                val_score=results['metrics'].get('val_auc', 0),
                cv_scores=results['cross_validation'].get('cv_scores', []),
                model_version='futures'  # V2 is futures-only
            )

            logger.info("Model saved to database")

        except Exception as e:
            logger.warning(f"Failed to save model to database: {e}")

    # METHOD DISABLED per client requirement
    def update_training_session(self, model_name: str, results: dict):
        """Update training session record with complete metrics."""
        logger.info("❌ Training session update disabled per client requirement")

    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series) -> dict:
        """Basic hyperparameter tuning (V1 spot config)."""
        logger.info("Performing hyperparameter tuning...")

        base_params = self.get_default_xgboost_params()
        best_auc = 0
        best_params = base_params.copy()

        # Define parameter grid (V1 config)
        param_grid = {
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [4, 6, 8],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }

        # Simple grid search (limited combinations for speed)
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

                            # Train model
                            model = xgb.XGBClassifier(**test_params)
                            model.fit(
                                X_train, y_train,
                                eval_set=[(X_val, y_val)],
                                verbose=False
                            )

                            # Evaluate
                            val_pred_proba = model.predict_proba(X_val)[:, 1]
                            val_auc = roc_auc_score(y_val, val_pred_proba)

                            if val_auc > best_auc:
                                best_auc = val_auc
                                best_params = test_params.copy()

        logger.info(f"Best validation AUC: {best_auc:.4f}")
        logger.info(f"Best parameters: {best_params}")

        return best_params

def main():
    """Main training function."""
    args = parse_arguments()
    validate_arguments(args)

    # Set up logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check for price-only mode
    price_only_mode = os.getenv('PRICE_ONLY_MODE', 'false').lower() == 'true'

    # Create data filter
    data_filter = DataFilter(args)

    # Initialize trainer
    trainer = XGBoostTrainer(data_filter, args.output_dir, price_only_mode)

    try:
        # Load training data
        logger.info("Loading training data...")
        X, y = trainer.load_training_data()

        # In price-only mode, skip feature selection to keep all 18 features
        # This ensures compatibility with QuantConnect which has all 18 features
        if price_only_mode:
            logger.info("🔵 PRICE-ONLY MODE: Skipping feature selection to keep all price features")
            logger.info(f"Using all {X.shape[1]} features for training")
            selected_features = list(X.columns)
        else:
            # Quick Win #4: Feature selection (drop features < 0.5% importance)
            X, selected_features = trainer.select_features(X, y, importance_threshold=0.005)

        trainer.feature_names = selected_features  # Update feature names

        # Prepare data splits
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data_splits(X, y)

        # DISABLED: Hyperparameter tuning was causing overfitting
        # Using conservative default params instead to prevent overfitting
        logger.info("Using conservative default parameters (hyperparameter tuning disabled to prevent overfitting)...")
        best_params = trainer.get_default_xgboost_params()
        logger.info(f"Parameters: {best_params}")

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
        trainer.save_model_and_results(model, metrics, cv_results, feature_importance, sample_count=len(X))

        logger.info("\n=== Training Complete ===")
        logger.info("Ready for model_evaluation.py")

    except Exception as e:
        logger.error(f"Error in training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()