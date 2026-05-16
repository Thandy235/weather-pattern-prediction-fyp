"""
predict.py — Loads trained models and makes rainfall predictions.

This module provides the RainfallPredictor class, which is used both by the
web application (app.py) and as a standalone script.  It loads the saved
Random Forest models from disk and exposes a simple interface for making
predictions across all four forecast horizons.

The two-stage prediction approach:
  1. The classifier answers: "Will it rain?" → probability + yes/no
  2. The regressor answers:  "If yes, how much?" → mm of rainfall
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib  # For loading saved model files
from datetime import datetime, timedelta

# Resolve paths relative to this file so the module works regardless of the
# working directory (important when launched via gunicorn or from project root).
_BASE = Path(__file__).parent.parent  # backend/


class RainfallPredictor:
    """
    Loads trained Random Forest models and makes rainfall predictions.

    On initialisation, all available models are loaded from the models/
    directory.  The predict() method takes a feature row and returns a
    prediction dict with occurrence probability, amount, and confidence level.
    """

    def __init__(self):
        # Where the trained model .pkl files are stored
        # Anchored to backend/ so this works from any working directory.
        self.model_dir = _BASE / 'models'
        self.models = {}         # Stores loaded model objects: {key: model}
        self.feature_names = {}  # {horizon: [col1, col2, ...]} — exact training column order
        self.feature_means = {}  # {horizon: {col: mean}} — for filling NaNs at inference time
        self.load_models()
        
    def load_models(self):
        """
        Load all trained models, feature name lists, and training-set column means.

        For each horizon we look for:
          - rf_classifier_{horizon}.pkl  — the occurrence classifier
          - rf_regressor_{horizon}.pkl   — the amount regressor
          - feature_names_{horizon}.pkl  — the exact list of feature columns
          - feature_means_{horizon}.pkl  — training-set means for NaN imputation

        Missing files are silently skipped — the predictor works with whatever
        models are available.
        """
        horizons = ['1day', '7day', '30day', '90day']

        print(f"[predict] Model directory: {self.model_dir.resolve()}")
        print(f"[predict] Model directory exists: {self.model_dir.exists()}")

        if self.model_dir.exists():
            found = list(self.model_dir.glob('*.pkl'))
            print(f"[predict] .pkl files found: {len(found)}")
        else:
            print("[predict] WARNING: model directory not found — no models will be loaded")

        for horizon in horizons:
            clf_file   = self.model_dir / f'rf_classifier_{horizon}.pkl'
            reg_file   = self.model_dir / f'rf_regressor_{horizon}.pkl'
            feat_file  = self.model_dir / f'feature_names_{horizon}.pkl'
            means_file = self.model_dir / f'feature_means_{horizon}.pkl'

            print(f"[predict] Loading {horizon}...")

            if clf_file.exists():
                try:
                    self.models[f'{horizon}_classifier'] = joblib.load(clf_file)
                    print(f"[predict]   ✓ classifier loaded")
                except Exception as e:
                    print(f"[predict]   ✗ classifier FAILED: {e}")
            else:
                print(f"[predict]   ✗ classifier not found: {clf_file}")

            if reg_file.exists():
                try:
                    self.models[f'{horizon}_regressor'] = joblib.load(reg_file)
                    print(f"[predict]   ✓ regressor loaded")
                except Exception as e:
                    print(f"[predict]   ✗ regressor FAILED: {e}")
            else:
                print(f"[predict]   ✗ regressor not found: {reg_file}")

            if feat_file.exists():
                try:
                    self.feature_names[horizon] = joblib.load(feat_file)
                    print(f"[predict]   ✓ feature names loaded ({len(self.feature_names[horizon])} features)")
                except Exception as e:
                    print(f"[predict]   ✗ feature names FAILED: {e}")
            else:
                print(f"[predict]   ✗ feature names not found: {feat_file}")

            if means_file.exists():
                try:
                    self.feature_means[horizon] = joblib.load(means_file)
                    print(f"[predict]   ✓ feature means loaded")
                except Exception as e:
                    print(f"[predict]   ✗ feature means FAILED: {e}")
            else:
                print(f"[predict]   - feature means not found (will use 0 for missing values): {means_file}")

        print(f"[predict] Done — {len(self.models)} model(s) loaded successfully")
    
    def prepare_features(self, input_data, horizon='1day'):
        """
        Align the input feature row to the exact column set used during training.

        The model was trained on a specific set of columns in a specific order.
        At prediction time, the input might have different columns (e.g. some
        features couldn't be computed, or the input comes from a different source).

        This method:
          1. Keeps only numeric columns
          2. Adds any missing columns, filled with the training-set mean (or 0)
          3. Reorders columns to match the training order exactly
          4. Fills any remaining NaNs with training means

        Why use training means for missing values?  It's the least-biased
        imputation strategy — it's equivalent to saying "this feature is average"
        which has minimal effect on the prediction.

        Parameters:
            input_data — a single-row DataFrame of features
            horizon    — which horizon's feature set to align to

        Returns a single-row DataFrame ready to pass to the model.
        """
        # Drop any non-numeric columns (strings, dates, etc.)
        input_data = input_data.select_dtypes(include=[np.number])

        if horizon in self.feature_names:
            expected = self.feature_names[horizon]  # The exact columns the model expects
            means    = self.feature_means.get(horizon, {})
            # Add any columns the model expects but the input doesn't have
            for col in expected:
                if col not in input_data.columns:
                    # Use the training mean if available, otherwise 0
                    input_data[col] = means.get(col, 0.0)
            # Reorder columns to exactly match the training order
            input_data = input_data[expected]
            # Fill any NaNs with training means
            if means:
                input_data = input_data.fillna(value=means)
        
        # Final safety net: replace any remaining NaNs with 0
        input_data = input_data.fillna(0.0)
        return input_data
    
    def predict(self, features, horizon='1day'):
        """
        Make a rainfall prediction for a single forecast horizon.

        Two-stage prediction:
          1. The classifier predicts the probability of rain and a yes/no decision
          2. If rain is predicted, the regressor estimates how many mm will fall

        Parameters:
            features — a single-row DataFrame of input features
            horizon  — which horizon to predict ('1day', '7day', '30day', '90day')

        Returns a dict with:
            horizon              — the forecast horizon
            occurrence           — 1 if rain predicted, 0 if not
            occurrence_probability — probability of rain (0.0–1.0)
            amount               — predicted rainfall in mm (0 if no rain predicted)
            confidence           — human-readable confidence label
        """
        clf_key = f'{horizon}_classifier'
        reg_key = f'{horizon}_regressor'
        
        if clf_key not in self.models:
            raise ValueError(f"Model for {horizon} not found")

        # Align the input features to the training column set
        aligned = self.prepare_features(features.copy(), horizon)
        
        # Stage 1: predict rain probability
        # predict_proba returns [[prob_no_rain, prob_rain]] — we want the second column
        occurrence_prob = self.models[clf_key].predict_proba(aligned)[0, 1]
        # Convert probability to a binary yes/no decision using 0.5 threshold
        occurrence = int(occurrence_prob > 0.5)
        
        # Stage 2: predict rainfall amount (only if rain is predicted)
        amount = 0.0
        if occurrence and reg_key in self.models:
            # clip to 0 ensures we never predict negative rainfall
            amount = max(0, self.models[reg_key].predict(aligned)[0])
        
        return {
            'horizon': horizon,
            'occurrence': occurrence,
            'occurrence_probability': float(occurrence_prob),
            'amount': float(amount),
            'confidence': self._calculate_confidence(occurrence_prob)
        }
    
    def predict_all_horizons(self, features):
        """
        Make predictions for all four forecast horizons at once.

        Calls predict() for each horizon and collects the results.
        If a horizon fails (e.g. model not loaded), it's stored as None.

        Parameters:
            features — a single-row DataFrame of input features

        Returns a dict: {horizon: prediction_dict or None}
        """
        predictions = {}
        
        for horizon in ['1day', '7day', '30day', '90day']:
            try:
                predictions[horizon] = self.predict(features, horizon)
            except Exception as e:
                print(f"Error predicting {horizon}: {e}")
                predictions[horizon] = None
        
        return predictions
    
    @staticmethod
    def _calculate_confidence(probability):
        """
        Convert a raw probability into a human-readable confidence label.

        The confidence is based on how far the probability is from 0.5 (maximum
        uncertainty).  A probability of 0.9 or 0.1 is very confident; 0.55 or
        0.45 is barely better than a coin flip.

        The formula: confidence = |probability - 0.5| × 2
        This maps [0.5, 1.0] → [0.0, 1.0] (and [0.0, 0.5] → [1.0, 0.0]).

        Parameters:
            probability — float between 0 and 1

        Returns one of: 'Very High', 'High', 'Medium', 'Low'
        """
        # Distance from 0.5 (the point of maximum uncertainty), scaled to [0, 1]
        confidence = abs(probability - 0.5) * 2
        
        if confidence > 0.8:
            return 'Very High'
        elif confidence > 0.6:
            return 'High'
        elif confidence > 0.4:
            return 'Medium'
        else:
            return 'Low'
    
    def get_forecast_dates(self):
        """
        Compute the calendar date that each forecast horizon corresponds to.

        For example, if today is 1 July, the '7day' forecast target date is 8 July.

        Returns a dict: {horizon: datetime.date}
        """
        today = datetime.now().date()
        
        return {
            '1day':  today + timedelta(days=1),
            '7day':  today + timedelta(days=7),
            '30day': today + timedelta(days=30),
            '90day': today + timedelta(days=90)
        }


# Example usage when run directly
if __name__ == '__main__':
    predictor = RainfallPredictor()
    
    # Load the feature matrix and use the most recent row as "current conditions"
    features_file = _BASE / 'data/processed/features_complete.csv'
    if features_file.exists():
        df = pd.read_csv(features_file)
        
        # Use the last row (most recent day in the dataset) as the input
        latest_features = df.iloc[[-1]].drop(
            columns=['date', 'rainfall', 'rainfall_occurrence'] + 
                    [c for c in df.columns if c.startswith('target_')],
            errors='ignore'
        )
        
        # Make predictions for all horizons
        predictions = predictor.predict_all_horizons(latest_features)
        
        print("\nRainfall Predictions:")
        print("=" * 60)
        
        forecast_dates = predictor.get_forecast_dates()
        
        for horizon, pred in predictions.items():
            if pred:
                print(f"\n{horizon.upper()} ({forecast_dates[horizon]}):")
                print(f"  Occurrence: {'Yes' if pred['occurrence'] else 'No'}")
                print(f"  Probability: {pred['occurrence_probability']:.2%}")
                print(f"  Amount: {pred['amount']:.2f} mm")
                print(f"  Confidence: {pred['confidence']}")
