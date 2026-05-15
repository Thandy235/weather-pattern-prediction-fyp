"""
train_models.py — Trains Random Forest models for rainfall prediction.

For each of the four forecast horizons (1, 7, 30, 90 days ahead) we train
two models:
  1. A classifier — predicts whether it will rain at all (yes/no + probability)
  2. A regressor  — predicts how many mm of rain will fall (only on rainy days)

The two-model approach is standard for rainfall prediction because rainfall
has a "zero-inflated" distribution: most days have zero rain, and the amount
on rainy days follows a completely different distribution.  Training a single
regressor on all days would be dominated by the zeros.

Models are saved to the models/ directory as .pkl files.
A human-readable training summary is saved to models/training_summary.txt.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib  # For saving/loading model files
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class RainfallModelTrainer:
    """
    Trains, evaluates, and saves Random Forest models for all forecast horizons.

    The training pipeline for each horizon:
      1. Load the feature matrix from features_complete.csv
      2. Split into train / validation / test sets (temporal order preserved)
      3. Train a classifier for rainfall occurrence
      4. Train a regressor for rainfall amount (on rainy days only)
      5. Evaluate on both validation and test sets
      6. Save models, feature names, and training means to disk
    """

    def __init__(self):
        # Where the feature CSV lives
        self.data_dir = Path('data/processed')
        # Where to save trained models and plots
        self.model_dir = Path('models')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # The four forecast horizons we train separate models for
        self.forecast_horizons = {
            '1day': 1,    # Predict tomorrow
            '7day': 7,    # Predict one week ahead
            '30day': 30,  # Predict one month ahead
            '90day': 90   # Predict one season ahead
        }
        
    def load_features(self):
        """
        Load the engineered feature matrix from disk.

        This file is produced by feature_engineering.py and contains one row
        per day with hundreds of lag, rolling, and interaction features plus
        the target labels for all four horizons.

        Returns a DataFrame with a 'date' column and all feature/target columns.
        Raises FileNotFoundError if the file doesn't exist yet.
        """
        features_file = self.data_dir / 'features_complete.csv'
        
        if not features_file.exists():
            raise FileNotFoundError(
                f"{features_file} not found. Run feature_engineering.py first."
            )
        
        print("Loading features...")
        df = pd.read_csv(features_file, parse_dates=['date'])
        print(f"Data shape: {df.shape}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
    
    def prepare_data(self, df, horizon='1day'):
        """
        Extract features (X) and targets (y) for a specific forecast horizon.

        Steps:
          1. Identify the target columns for this horizon
          2. Drop metadata and all target columns from the feature set
          3. One-hot encode the 'season' categorical variable
          4. Keep only numeric columns
          5. Fill NaN values using training-set column means (no leakage from val/test)

        Why fill NaNs with training means only?  If we computed the mean across
        the whole dataset, we'd be "leaking" information from the future into the
        training set.  By computing means only on the training portion, the model
        never sees statistics from the validation or test periods during training.

        Parameters:
            df      — the full feature DataFrame
            horizon — which forecast horizon to prepare (e.g. '1day', '7day')

        Returns:
            X            — feature matrix (DataFrame)
            y_occurrence — binary target (0/1 rain occurrence)
            y_amount     — continuous target (mm of rainfall)
            feature_cols — list of feature column names
        """
        print(f"\nPreparing data for {horizon} forecast...")
        
        # The two target columns for this horizon
        target_occurrence = f'target_{horizon}_occurrence'
        target_amount = f'target_{horizon}_amount'
        
        # Columns to exclude from the feature set
        exclude_cols = [
            'date', 'rainfall', 'rainfall_occurrence',
            'Station_Name', 'season'
        ]
        
        # Also exclude all target columns — we don't want the model to cheat
        # by seeing the answer for a different horizon
        exclude_cols.extend([c for c in df.columns if c.startswith('target_')])
        
        # Start with all columns that aren't in the exclusion list
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        # One-hot encode the season column (converts 'hot_rainy' → season_hot_rainy=1, etc.)
        # This lets the model learn season-specific patterns without treating season as a number
        if 'season' in df.columns:
            season_dummies = pd.get_dummies(df['season'], prefix='season')
            df = pd.concat([df, season_dummies], axis=1)
            feature_cols.extend(season_dummies.columns.tolist())
        
        # Keep only numeric columns — Random Forest can't handle strings
        feature_cols = [c for c in feature_cols if df[c].dtype in ['int64', 'float64']]
        
        X = df[feature_cols].copy()
        y_occurrence = df[target_occurrence].copy()
        y_amount = df[target_amount].copy()
        
        # Compute the training/validation/test split boundary indices
        n = len(X)
        train_idx = int(n * 0.70)  # First 70% = training

        # Fill NaN values using ONLY the training portion's column means.
        # This prevents data leakage: the model never sees statistics from the future.
        train_means = X.iloc[:train_idx].mean()
        X = X.fillna(train_means)
        # Any columns that were entirely NaN in the training set → fill with 0
        X = X.fillna(0)
        
        print(f"Features: {X.shape[1]}")
        print(f"Samples: {X.shape[0]}")
        print(f"Occurrence distribution:\n{y_occurrence.value_counts()}")
        
        return X, y_occurrence, y_amount, feature_cols
    
    def train_classification_model(self, X, y, horizon):
        """
        Train a Random Forest classifier to predict rainfall occurrence (rain vs no-rain).

        We use a temporal split — the oldest 70% of data for training, the next
        15% for validation, and the most recent 15% for testing.  This is crucial
        for time-series data: shuffling would let the model "see the future" during
        training, giving unrealistically good results.

        The classifier uses class_weight='balanced' to handle the imbalance between
        rainy days (minority) and dry days (majority).  Without this, the model
        would learn to just predict "no rain" most of the time.

        Parameters:
            X       — feature matrix (already NaN-filled)
            y       — binary occurrence labels (0/1)
            horizon — forecast horizon label (e.g. '1day')

        Returns:
            model   — the trained RandomForestClassifier
            metrics — dict with 'validation' and 'test' metric dicts
        """
        print(f"\nTraining classification model ({horizon})...")
        
        # Temporal split: train on oldest data, test on most recent
        n = len(X)
        train_idx = int(n * 0.70)  # 70% training
        val_idx = int(n * 0.85)    # Next 15% validation, last 15% test
        
        X_train = X[:train_idx]
        y_train = y[:train_idx]
        
        X_val = X[train_idx:val_idx]
        y_val = y[train_idx:val_idx]
        
        X_test = X[val_idx:]
        y_test = y[val_idx:]
        
        print(f"Train size: {len(X_train)} ({len(X_train)/n*100:.1f}%)")
        print(f"Validation size: {len(X_val)} ({len(X_val)/n*100:.1f}%)")
        print(f"Test size: {len(X_test)} ({len(X_test)/n*100:.1f}%)")
        
        # Build and train the Random Forest classifier
        model = RandomForestClassifier(
            n_estimators=200,        # 200 trees — more stable than fewer trees
            max_depth=20,            # Limit tree depth to prevent overfitting
            min_samples_split=10,    # A node needs at least 10 samples to split
            min_samples_leaf=5,      # Each leaf must have at least 5 samples
            max_features='sqrt',     # Each tree sees √(n_features) random features
            random_state=42,         # Fixed seed for reproducibility
            n_jobs=-1,               # Use all CPU cores
            class_weight='balanced'  # Upweight rainy days to handle class imbalance
        )
        
        model.fit(X_train, y_train)
        
        # ── Evaluate on the validation set ───────────────────────────────
        y_val_pred = model.predict(X_val)
        y_val_pred_proba = model.predict_proba(X_val)[:, 1]  # Probability of rain
        
        val_metrics = {
            'accuracy': accuracy_score(y_val, y_val_pred),
            'precision': precision_score(y_val, y_val_pred, zero_division=0),
            'recall': recall_score(y_val, y_val_pred, zero_division=0),
            'f1': f1_score(y_val, y_val_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_val, y_val_pred_proba)
        }
        
        print(f"\nValidation Set Metrics ({horizon}):")
        for metric, value in val_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # ── Evaluate on the held-out test set ────────────────────────────
        # This is the unbiased final evaluation — we only look at this once
        y_test_pred = model.predict(X_test)
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'recall': recall_score(y_test, y_test_pred, zero_division=0),
            'f1': f1_score(y_test, y_test_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_test_pred_proba)
        }
        
        print(f"\nTest Set Metrics ({horizon}):")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        print(f"\nTest Set Confusion Matrix:")
        print(confusion_matrix(y_test, y_test_pred))
        
        print(f"\nTest Set Classification Report:")
        print(classification_report(y_test, y_test_pred))
        
        # ── Save the model and supporting files ──────────────────────────
        model_file = self.model_dir / f'rf_classifier_{horizon}.pkl'
        joblib.dump(model, model_file)
        print(f"✓ Model saved: {model_file}")

        # Save the exact list of feature names so predict.py can align columns correctly
        feature_names_file = self.model_dir / f'feature_names_{horizon}.pkl'
        joblib.dump(list(X.columns), feature_names_file)
        print(f"✓ Feature names saved: {feature_names_file}")

        # Save training-set column means so predict.py can fill NaNs the same way
        train_means_file = self.model_dir / f'feature_means_{horizon}.pkl'
        joblib.dump(X_train.mean().to_dict(), train_means_file)
        print(f"✓ Feature means saved: {train_means_file}")
        
        # Return both sets of metrics for the summary report
        metrics = {
            'validation': val_metrics,
            'test': test_metrics
        }
        
        return model, metrics
    
    def train_regression_model(self, X, y, horizon):
        """
        Train a Random Forest regressor to predict rainfall amount (in mm).

        We only train on days where it actually rained (y > 0).  This is the
        standard "two-stage" approach for rainfall modelling:
          Stage 1 (classifier): will it rain? → yes/no
          Stage 2 (regressor):  if yes, how much? → mm

        Training on all days (including dry days with y=0) would make the model
        predict near-zero amounts even on rainy days, because the zeros dominate.

        Parameters:
            X       — feature matrix (same as used for the classifier)
            y       — continuous rainfall amount targets (mm)
            horizon — forecast horizon label

        Returns:
            model   — the trained RandomForestRegressor (or None if too few rainy days)
            metrics — dict with 'validation' and 'test' metric dicts
        """
        print(f"\nTraining regression model ({horizon})...")
        
        # Filter to rainy days only
        mask = y > 0
        X_rain = X[mask]
        y_rain = y[mask]
        
        # We need a minimum number of rainy days to train a reliable regressor
        if len(X_rain) < 100:
            print(f"Warning: Only {len(X_rain)} rainy days. Skipping regression.")
            return None, {}
        
        # Same temporal split as the classifier
        n = len(X_rain)
        train_idx = int(n * 0.70)
        val_idx = int(n * 0.85)
        
        X_train = X_rain[:train_idx]
        y_train = y_rain[:train_idx]
        
        X_val = X_rain[train_idx:val_idx]
        y_val = y_rain[train_idx:val_idx]
        
        X_test = X_rain[val_idx:]
        y_test = y_rain[val_idx:]
        
        print(f"Train size: {len(X_train)} ({len(X_train)/n*100:.1f}%)")
        print(f"Validation size: {len(X_val)} ({len(X_val)/n*100:.1f}%)")
        print(f"Test size: {len(X_test)} ({len(X_test)/n*100:.1f}%)")
        
        # Build and train the Random Forest regressor
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
            # No class_weight here — regression doesn't have classes
        )
        
        model.fit(X_train, y_train)
        
        # ── Evaluate on the validation set ───────────────────────────────
        y_val_pred = model.predict(X_val)
        
        val_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),  # Root mean squared error (mm)
            'mae': mean_absolute_error(y_val, y_val_pred),           # Mean absolute error (mm)
            'r2': r2_score(y_val, y_val_pred)                        # R² — how much variance is explained
        }
        
        print(f"\nValidation Set Metrics ({horizon}):")
        for metric, value in val_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # ── Evaluate on the held-out test set ────────────────────────────
        y_test_pred = model.predict(X_test)
        
        test_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'mae': mean_absolute_error(y_test, y_test_pred),
            'r2': r2_score(y_test, y_test_pred)
        }
        
        print(f"\nTest Set Metrics ({horizon}):")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Save the regressor
        model_file = self.model_dir / f'rf_regressor_{horizon}.pkl'
        joblib.dump(model, model_file)
        print(f"✓ Model saved: {model_file}")

        # Save regressor feature names (same columns as the classifier)
        reg_feature_names_file = self.model_dir / f'feature_names_reg_{horizon}.pkl'
        joblib.dump(list(X_rain.columns), reg_feature_names_file)
        print(f"✓ Regressor feature names saved: {reg_feature_names_file}")
        
        metrics = {
            'validation': val_metrics,
            'test': test_metrics
        }
        
        return model, metrics
    
    def plot_feature_importance(self, model, feature_names, horizon, model_type):
        """
        Create a horizontal bar chart of the top 20 most important features.

        Feature importance in a Random Forest is measured by how much each
        feature reduces impurity (for classifiers) or variance (for regressors)
        when it's used as a split point, averaged across all trees.

        Higher importance = the model relies on that feature more heavily.

        Parameters:
            model        — trained RandomForest model
            feature_names — list of feature column names (same order as training)
            horizon      — forecast horizon label (for the plot title)
            model_type   — 'classifier' or 'regressor' (for the filename)
        """
        importances = model.feature_importances_
        # Get the indices of the top 20 features, sorted by importance (descending)
        indices = np.argsort(importances)[::-1][:20]
        
        plt.figure(figsize=(10, 8))
        plt.title(f'Top 20 Feature Importances - {model_type} ({horizon})')
        plt.barh(range(20), importances[indices])
        plt.yticks(range(20), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.tight_layout()
        
        output_file = self.model_dir / f'feature_importance_{model_type}_{horizon}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Feature importance plot saved: {output_file}")
    
    def train_all_models(self):
        """
        Run the full training pipeline for all four forecast horizons.

        For each horizon:
          1. Prepare the feature/target data
          2. Train and evaluate the classifier
          3. Plot classifier feature importances
          4. Train and evaluate the regressor
          5. Plot regressor feature importances (if regressor was trained)

        After all horizons, saves a human-readable summary to models/training_summary.txt.

        Returns a dict of results: {horizon: {classification: metrics, regression: metrics}}
        """
        # Load the feature matrix once and reuse it for all horizons
        df = self.load_features()
        
        all_results = {}
        
        for horizon in self.forecast_horizons.keys():
            print(f"\n{'='*60}")
            print(f"Training models for {horizon} forecast")
            print(f"{'='*60}")
            
            # Prepare features and targets for this horizon
            X, y_occurrence, y_amount, feature_names = self.prepare_data(df, horizon)
            
            # Train the occurrence classifier
            clf_model, clf_metrics = self.train_classification_model(
                X, y_occurrence, horizon
            )
            
            # Visualise which features the classifier found most useful
            self.plot_feature_importance(
                clf_model, feature_names, horizon, 'classifier'
            )
            
            # Train the amount regressor (on rainy days only)
            reg_model, reg_metrics = self.train_regression_model(
                X, y_amount, horizon
            )
            
            if reg_model is not None:
                self.plot_feature_importance(
                    reg_model, feature_names, horizon, 'regressor'
                )
            
            # Store results for the summary report
            all_results[horizon] = {
                'classification': clf_metrics,
                'regression': reg_metrics
            }
        
        # Write the summary file
        self._save_results_summary(all_results)
        
        print(f"\n{'='*60}")
        print("✓ All models trained successfully!")
        print(f"{'='*60}")
        
        return all_results
    
    def _save_results_summary(self, results):
        """
        Write a human-readable training summary to models/training_summary.txt.

        This file is read by policy_summary.py to include model performance
        metrics in the policy report, and by the web app's /api/stats endpoint.

        Parameters:
            results — dict of {horizon: {classification: metrics, regression: metrics}}
        """
        summary_file = self.model_dir / 'training_summary.txt'
        
        with open(summary_file, 'w') as f:
            f.write("Rainfall Prediction Model Training Summary\n")
            f.write("=" * 60 + "\n")
            f.write("Dataset Split: 70% Training, 15% Validation, 15% Testing\n")
            f.write("=" * 60 + "\n\n")
            
            for horizon, metrics in results.items():
                f.write(f"\n{horizon.upper()} Forecast:\n")
                f.write("=" * 60 + "\n")
                
                # ── Classification metrics ────────────────────────────────
                f.write("\nClassification (Occurrence Prediction):\n")
                f.write("-" * 40 + "\n")
                
                if 'validation' in metrics['classification']:
                    f.write("\nValidation Set:\n")
                    for metric, value in metrics['classification']['validation'].items():
                        f.write(f"  {metric}: {value:.4f}\n")
                    
                    f.write("\nTest Set:\n")
                    for metric, value in metrics['classification']['test'].items():
                        f.write(f"  {metric}: {value:.4f}\n")
                else:
                    # Handle older format where validation/test weren't separated
                    for metric, value in metrics['classification'].items():
                        f.write(f"  {metric}: {value:.4f}\n")
                
                # ── Regression metrics ────────────────────────────────────
                if metrics['regression']:
                    f.write("\nRegression (Amount Prediction):\n")
                    f.write("-" * 40 + "\n")
                    
                    if 'validation' in metrics['regression']:
                        f.write("\nValidation Set:\n")
                        for metric, value in metrics['regression']['validation'].items():
                            f.write(f"  {metric}: {value:.4f}\n")
                        
                        f.write("\nTest Set:\n")
                        for metric, value in metrics['regression']['test'].items():
                            f.write(f"  {metric}: {value:.4f}\n")
                    else:
                        for metric, value in metrics['regression'].items():
                            f.write(f"  {metric}: {value:.4f}\n")
                
                f.write("\n")
            
            # Add notes explaining what the metrics mean
            f.write("\n" + "=" * 60 + "\n")
            f.write("Notes:\n")
            f.write("- Validation set used for model tuning and hyperparameter selection\n")
            f.write("- Test set used for final unbiased performance evaluation\n")
            f.write("- Temporal order maintained (no data leakage)\n")
            f.write("- Test set represents most recent data\n")
            f.write("=" * 60 + "\n")
        
        print(f"\n✓ Training summary saved: {summary_file}")


# Run training directly when this script is called
if __name__ == '__main__':
    trainer = RainfallModelTrainer()
    results = trainer.train_all_models()
