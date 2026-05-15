"""
realtime_validation.py — Validates trained models against recent ERA5 observations.

This script fulfils Objective 4 of the project: real-time model validation.

The idea is simple: ERA5 provides near-real-time atmospheric data with about a
5-day delay.  We download the last N days of ERA5 data, build the same features
the models were trained on, run the models, and compare their predictions to
what ERA5 says actually happened.

This gives us an honest, ongoing measure of how well the models perform on
data they've never seen before — not just on the historical training set.

Output: data/validation/validation_results.csv
        data/validation/validation_metrics.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
import joblib

# Load the CDS_API_KEY from the .env file
load_dotenv()


class RealtimeValidator:
    """
    Downloads recent ERA5 data and validates the trained models against it.

    The validation pipeline:
      1. Fetch the last N days of ERA5 data from the CDS API
      2. Convert the NetCDF file to a daily DataFrame
      3. Build feature vectors matching the training schema
      4. Run the trained classifiers and compare to actual ERA5 rainfall
      5. Compute and report accuracy metrics
    """

    MODEL_DIR      = Path('models')
    VALIDATION_DIR = Path('data/validation')
    FEATURES_FILE  = Path('data/processed/features_complete.csv')

    # Choma, Zambia coordinates and bounding box
    LAT  = -16.8
    LON  =  26.9
    AREA = [-16.3, 26.4, -17.3, 27.4]   # [North, West, South, East]

    # A day counts as "rainy" if ERA5 precipitation is at least this many mm
    RAINFALL_THRESHOLD = 1.0   # mm

    def __init__(self):
        # Make sure the validation output directory exists
        self.VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  CDS API client                                                      #
    # ------------------------------------------------------------------ #
    def _get_cds_client(self):
        """
        Create and return an authenticated CDS API client.

        Reads the API key from the CDS_API_KEY environment variable.
        Raises a ValueError if the key is missing or still set to the placeholder.

        Returns an ecmwf_client.Client object.
        """
        api_key = os.getenv('CDS_API_KEY', '').strip()
        if not api_key or api_key == 'your_api_key_here':
            raise ValueError(
                "CDS_API_KEY not set. Add it to your .env file.\n"
                "Register at https://cds.climate.copernicus.eu/"
            )
        try:
            from ecmwf.datastores import client as ecmwf_client
            return ecmwf_client.Client(
                url='https://cds.climate.copernicus.eu/api',
                key=api_key
            )
        except ImportError:
            raise ImportError(
                "ecmwf-datastores not installed. Run: pip install ecmwf-datastores"
            )

    # ------------------------------------------------------------------ #
    #  Step 1 — Fetch recent ERA5 data                                     #
    # ------------------------------------------------------------------ #
    def fetch_recent_data(self, days_back: int = 30) -> Path | None:
        """
        Download the most recent ERA5 data from the CDS API.

        ERA5 has about a 5-day delay, so "real-time" means the most recent
        available data, not today's data.  We subtract 5 days from today to
        get the end date, then go back `days_back` days from there.

        If the file was already downloaded (same date range), we skip the
        download and return the existing file path.

        Parameters:
            days_back — how many days of recent data to download (default 30)

        Returns the Path to the downloaded NetCDF file, or None if download failed.
        """
        print(f"\n[1/4] Fetching last {days_back} days of ERA5 data...")

        client = self._get_cds_client()

        # ERA5 has ~5-day delay, so the most recent available data ends 5 days ago
        end_date   = datetime.now() - timedelta(days=5)
        start_date = end_date - timedelta(days=days_back)

        # Name the output file with the date range so we can detect existing downloads
        out_file = self.VALIDATION_DIR / (
            f"era5_validation_"
            f"{start_date.strftime('%Y%m%d')}_"
            f"{end_date.strftime('%Y%m%d')}.nc"
        )

        if out_file.exists():
            print(f"  ✓ Already downloaded: {out_file}")
            return out_file

        print(f"  Downloading {start_date.date()} → {end_date.date()}...")
        # Build the lists of years, months, and days covered by the date range
        dates  = pd.date_range(start_date, end_date, freq='D')
        years  = sorted({d.year  for d in dates})
        months = sorted({d.month for d in dates})
        days   = sorted({d.day   for d in dates})

        try:
            client.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': [
                        '2m_temperature', '2m_dewpoint_temperature',
                        'surface_pressure',
                        '10m_u_component_of_wind', '10m_v_component_of_wind',
                        'total_column_water_vapour', 'total_precipitation',
                    ],
                    'year':  [str(y) for y in years],
                    'month': [f'{m:02d}' for m in months],
                    'day':   [f'{d:02d}' for d in days],
                    'time':  ['00:00', '06:00', '12:00', '18:00'],
                    'area':  self.AREA,
                    'format': 'netcdf',
                },
                str(out_file)
            )
            print(f"  ✓ Downloaded: {out_file}")
            return out_file
        except Exception as e:
            print(f"  ✗ Download failed: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Step 2 — Process NetCDF → daily DataFrame                           #
    # ------------------------------------------------------------------ #
    def process_validation_data(self, nc_file: Path) -> pd.DataFrame:
        """
        Convert the downloaded ERA5 NetCDF file into a daily pandas DataFrame.

        This mirrors the conversion done in era5_downloader.py:
          1. Handle ZIP wrapping (new CDS API)
          2. Open with xarray and flatten to a DataFrame
          3. Aggregate 4 time steps per day to daily values
          4. Convert units (Kelvin → °C, Pa → hPa, m → mm)
          5. Compute derived variables (relative humidity, wind speed)

        Parameters:
            nc_file — path to the downloaded NetCDF file

        Returns a daily DataFrame with one row per day.
        """
        import zipfile
        import xarray as xr

        print("\n[2/4] Processing validation NetCDF to daily CSV...")

        # Check if the file is a ZIP archive (new CDS API wraps files in ZIP)
        with open(nc_file, 'rb') as fh:
            is_zip = fh.read(2) == b'PK'  # ZIP files start with the magic bytes 'PK'

        if is_zip:
            # Extract all files from the ZIP into a subdirectory
            extract_dir = self.VALIDATION_DIR / 'extracted'
            extract_dir.mkdir(exist_ok=True)
            with zipfile.ZipFile(nc_file, 'r') as zf:
                zf.extractall(extract_dir)
            nc_files = sorted(extract_dir.glob('*.nc'))
        else:
            nc_files = [nc_file]

        # Open all NetCDF files and merge them into one dataset
        datasets = [xr.open_dataset(f, engine='netcdf4') for f in nc_files]
        # compat='override' handles the 'expver' variable that the new CDS API adds
        # and which can cause merge conflicts between files
        ds = xr.merge(datasets, compat='override')
        df = ds.to_dataframe().reset_index()
        for d in datasets:
            d.close()  # Free memory

        # Extract the date from the time column
        time_col = 'valid_time' if 'valid_time' in df.columns else 'time'
        df['date'] = pd.to_datetime(df[time_col]).dt.date

        # Aggregate to daily: mean for most variables, sum for precipitation
        agg_map = {
            't2m': 'mean', 'd2m': 'mean', 'sp': 'mean',
            'u10': 'mean', 'v10': 'mean', 'tcwv': 'mean', 'tp': 'sum'
        }
        agg = {k: v for k, v in agg_map.items() if k in df.columns}
        daily = df.groupby('date').agg(agg).reset_index()

        # Rename to descriptive column names
        daily.rename(columns={
            't2m': 'temp_2m', 'd2m': 'dewpoint_2m', 'sp': 'surface_pressure',
            'u10': 'wind_u',  'v10': 'wind_v',       'tcwv': 'water_vapour',
            'tp':  'precipitation'
        }, inplace=True)

        # ── Unit conversions ──────────────────────────────────────────────
        if 'temp_2m'          in daily.columns: daily['temp_2m']           -= 273.15  # K → °C
        if 'dewpoint_2m'      in daily.columns: daily['dewpoint_2m']       -= 273.15  # K → °C
        if 'surface_pressure' in daily.columns: daily['surface_pressure']  /= 100     # Pa → hPa
        if 'precipitation'    in daily.columns: daily['precipitation']     *= 1000    # m → mm

        # ── Derived variables ─────────────────────────────────────────────
        # Relative humidity from temperature and dewpoint (Magnus formula)
        if 'temp_2m' in daily.columns and 'dewpoint_2m' in daily.columns:
            t, d = daily['temp_2m'], daily['dewpoint_2m']
            daily['relative_humidity'] = 100 * (
                np.exp(17.625 * d / (243.04 + d)) /
                np.exp(17.625 * t / (243.04 + t))
            )
        # Wind speed from east-west (u) and north-south (v) components
        if 'wind_u' in daily.columns and 'wind_v' in daily.columns:
            daily['wind_speed'] = np.sqrt(daily['wind_u']**2 + daily['wind_v']**2)

        daily['date'] = pd.to_datetime(daily['date'])
        daily = daily.sort_values('date').reset_index(drop=True)

        # Save the processed daily data for reference
        out = self.VALIDATION_DIR / 'validation_data.csv'
        daily.to_csv(out, index=False)
        print(f"  ✓ Processed: {len(daily)} days  |  columns: {list(daily.columns)}")
        return daily

    # ------------------------------------------------------------------ #
    #  Step 3 — Build features matching training schema                   #
    # ------------------------------------------------------------------ #
    def _build_features_for_row(self, val_df: pd.DataFrame,
                                 idx: int,
                                 feature_names: list) -> pd.DataFrame:
        """
        Build a single-row feature vector for one day in the validation window.

        The models were trained on hundreds of features (lags, rolling stats, etc.).
        We need to reconstruct as many of those features as possible from the
        short validation window.  Features that can't be computed (e.g. a 30-day
        lag when we only have 10 days of validation data) are filled with 0.

        Parameters:
            val_df       — the full validation DataFrame (all days)
            idx          — index of the current day we're building features for
            feature_names — the exact list of feature columns the model expects

        Returns a single-row DataFrame aligned to feature_names.
        """
        row = val_df.iloc[idx]
        features = {}

        # ── Direct ERA5 variables ─────────────────────────────────────────
        # These are the raw atmospheric measurements for the current day
        direct_map = {
            'temp_2m':            row.get('temp_2m',            np.nan),
            'relative_humidity':  row.get('relative_humidity',  np.nan),
            'wind_speed':         row.get('wind_speed',         np.nan),
            'surface_pressure':   row.get('surface_pressure',   np.nan),
            'water_vapour':       row.get('water_vapour',       np.nan),
            'precipitation':      row.get('precipitation',      np.nan),
        }
        features.update(direct_map)

        # ── Lag features ──────────────────────────────────────────────────
        # "What was the value N days ago?" — computed from earlier rows in val_df
        for lag in [1, 3, 7, 14, 30]:
            src_idx = idx - lag  # The row that is `lag` days before the current row
            if src_idx >= 0:
                features[f'precipitation_lag{lag}']      = val_df.iloc[src_idx]['precipitation']
                features[f'temp_2m_lag{lag}']            = val_df.iloc[src_idx]['temp_2m']
                features[f'relative_humidity_lag{lag}']  = val_df.iloc[src_idx]['relative_humidity']

        # ── Rolling features ──────────────────────────────────────────────
        # Statistics over the last W days up to and including the current day
        for w in [7, 14, 30]:
            # Slice the window: from (idx - w + 1) to idx (inclusive)
            window_data = val_df.iloc[max(0, idx - w + 1): idx + 1]
            for col in ['precipitation', 'temp_2m', 'relative_humidity']:
                if col in window_data.columns:
                    features[f'{col}_roll{w}_mean'] = window_data[col].mean()
                    features[f'{col}_roll{w}_std']  = window_data[col].std()
                    features[f'{col}_roll{w}_max']  = window_data[col].max()
                    features[f'{col}_roll{w}_min']  = window_data[col].min()

        # ── Align to training feature names ──────────────────────────────
        # Build a single-row DataFrame from the features we computed
        feat_row = pd.DataFrame([features])

        # Add any columns the model expects but we couldn't compute — fill with NaN
        for col in feature_names:
            if col not in feat_row.columns:
                feat_row[col] = np.nan

        # Select columns in the exact order the model expects, fill remaining NaNs with 0
        feat_row = feat_row[feature_names].fillna(0.0)
        return feat_row

    # ------------------------------------------------------------------ #
    #  Step 4 — Run models and compute metrics                             #
    # ------------------------------------------------------------------ #
    def validate_predictions(self, val_df: pd.DataFrame) -> pd.DataFrame | None:
        """
        Run the trained models on the validation data and compute accuracy metrics.

        For each day in the validation window:
          - Build a feature vector from the ERA5 data
          - Run each trained classifier to get a rain probability
          - Compare the prediction to the actual ERA5 precipitation for the next day

        Then compute accuracy, precision, recall, and F1 score for each horizon.

        Parameters:
            val_df — the processed daily validation DataFrame

        Returns a DataFrame with one row per day and columns for predictions
        and actual values, or None if no models are loaded.
        """
        print("\n[3/4] Running trained models on validation data...")

        # Load all available trained classifiers and their feature name lists
        models        = {}
        feature_names = {}
        for horizon in ['1day', '7day', '30day', '90day']:
            clf_f  = self.MODEL_DIR / f'rf_classifier_{horizon}.pkl'
            feat_f = self.MODEL_DIR / f'feature_names_{horizon}.pkl'
            if clf_f.exists():
                models[horizon]        = joblib.load(clf_f)
                feature_names[horizon] = joblib.load(feat_f) if feat_f.exists() else []

        if not models:
            print("  ✗ No trained models found. Run src/train_models.py first.")
            return None

        print(f"  Loaded models: {list(models.keys())}")

        # ── Prepend historical context for lag/rolling features ───────────
        # The validation window might only be 30 days, but some features need
        # up to 30 days of history (e.g. rainfall_lag30).  We prepend the last
        # 30 days of the training data to give the lag features proper context.
        context_df = None
        harmonized_file = Path('data/processed/choma_harmonized_unified.csv')
        if harmonized_file.exists():
            hist = pd.read_csv(harmonized_file, parse_dates=['date'])
            val_start = pd.to_datetime(val_df['date'].min())
            # Get the 30 days immediately before the validation window starts
            context = hist[hist['date'] < val_start].tail(30)
            if len(context) > 0:
                # Align the historical context columns to match the validation DataFrame
                common_cols = ['date'] + [c for c in val_df.columns if c in context.columns and c != 'date']
                context_aligned = context[common_cols].copy()
                # Add any columns present in val_df but not in the historical data
                for col in val_df.columns:
                    if col not in context_aligned.columns:
                        context_aligned[col] = np.nan
                context_aligned = context_aligned[val_df.columns]
                context_df = pd.concat([context_aligned, val_df], ignore_index=True)
                print(f"  Added {len(context)} days of historical context for lag features")

        working_df = context_df if context_df is not None else val_df
        context_len = len(context_df) - len(val_df) if context_df is not None else 0

        print(f"  Loaded models: {list(models.keys())}")

        results = []

        # Loop through each day in the validation window (not the context)
        for i in range(len(val_df) - 1):
            current_row  = val_df.iloc[i]
            # The "actual" outcome is the next day's precipitation
            next_row     = val_df.iloc[i + 1]
            actual_rain  = float(next_row.get('precipitation', 0))
            actual_occ   = int(actual_rain >= self.RAINFALL_THRESHOLD)

            row_result = {
                'date':              str(current_row['date'])[:10],
                'actual_rainfall':   round(actual_rain, 2),
                'actual_occurrence': actual_occ,
            }

            # Run each model and record its prediction
            for horizon, model in models.items():
                feat_names = feature_names.get(horizon, [])
                if not feat_names:
                    continue
                try:
                    feat_row = self._build_features_for_row(val_df, i, feat_names)
                    # predict_proba returns [[prob_no_rain, prob_rain]] — take the rain probability
                    prob     = model.predict_proba(feat_row)[0, 1]
                    pred_occ = int(prob > 0.5)
                    row_result[f'pred_{horizon}_occurrence']  = pred_occ
                    row_result[f'pred_{horizon}_probability'] = round(float(prob), 4)
                except Exception as e:
                    # If feature building fails, record NaN so we can skip this row in metrics
                    row_result[f'pred_{horizon}_occurrence']  = np.nan
                    row_result[f'pred_{horizon}_probability'] = np.nan

            results.append(row_result)

        results_df = pd.DataFrame(results)

        # ── Compute accuracy metrics ──────────────────────────────────────
        print("\n[4/4] Computing validation metrics...")
        print("\n" + "=" * 65)
        print("  REAL-TIME VALIDATION RESULTS")
        print("=" * 65)

        metrics_rows = []
        for horizon in models.keys():
            pred_col = f'pred_{horizon}_occurrence'
            if pred_col not in results_df.columns:
                continue

            # Drop rows where prediction failed
            valid = results_df.dropna(subset=[pred_col, 'actual_occurrence'])
            if len(valid) == 0:
                continue

            y_true = valid['actual_occurrence'].astype(int)
            y_pred = valid[pred_col].astype(int)

            # Compute confusion matrix components manually for clarity
            tp = ((y_pred == 1) & (y_true == 1)).sum()  # Correctly predicted rain
            tn = ((y_pred == 0) & (y_true == 0)).sum()  # Correctly predicted no rain
            fp = ((y_pred == 1) & (y_true == 0)).sum()  # Predicted rain but it didn't rain
            fn = ((y_pred == 0) & (y_true == 1)).sum()  # Predicted no rain but it did rain

            accuracy  = (tp + tn) / len(valid) if len(valid) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # Of predicted rain days, how many were right?
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0  # Of actual rain days, how many did we catch?
            f1        = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0)    # Harmonic mean of precision and recall

            print(f"\n  {horizon.upper()} Forecast:")
            print(f"    Samples   : {len(valid)}")
            print(f"    Accuracy  : {accuracy:.3f}  ({accuracy*100:.1f}%)")
            print(f"    Precision : {precision:.3f}")
            print(f"    Recall    : {recall:.3f}")
            print(f"    F1 Score  : {f1:.3f}")

            metrics_rows.append({
                'horizon': horizon, 'samples': len(valid),
                'accuracy': round(accuracy, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
            })

        # Save the per-day prediction results and the summary metrics
        results_file = self.VALIDATION_DIR / 'validation_results.csv'
        results_df.to_csv(results_file, index=False)

        metrics_file = self.VALIDATION_DIR / 'validation_metrics.csv'
        pd.DataFrame(metrics_rows).to_csv(metrics_file, index=False)

        print(f"\n  Results saved : {results_file}")
        print(f"  Metrics saved : {metrics_file}")
        print("=" * 65)

        return results_df

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #
    def run_validation(self, days_back: int = 30) -> pd.DataFrame | None:
        """
        Run the complete real-time validation pipeline end-to-end.

        Steps:
          1. Download recent ERA5 data
          2. Convert to daily DataFrame
          3. Run models and compute metrics

        Parameters:
            days_back — how many recent days to validate against (default 30)

        Returns the validation results DataFrame, or None if download failed.
        """
        print("\n" + "=" * 65)
        print("  OBJECTIVE 4: REAL-TIME MODEL VALIDATION")
        print("=" * 65)
        print(f"  Validating against last {days_back} days of ERA5 data")
        print("  (ERA5 has ~5-day delay, so 'real-time' = most recent available)")

        # Step 1: Download
        nc_file = self.fetch_recent_data(days_back)
        if nc_file is None:
            return None

        # Step 2: Process
        val_df  = self.process_validation_data(nc_file)

        # Step 3: Validate
        results = self.validate_predictions(val_df)
        return results


# Run validation directly when this script is called
if __name__ == '__main__':
    validator = RealtimeValidator()

    print("\nReal-time Validation — Objective 4")
    print("=" * 65)
    days = input("How many days back to validate? (default 30): ").strip()
    days = int(days) if days.isdigit() else 30

    try:
        results = validator.run_validation(days_back=days)
        if results is not None:
            print("\n✓ Objective 4 complete: model validated on real-time ERA5 data.")
    except Exception as e:
        print(f"\n✗ Validation failed: {e}")
        print("\nMake sure:")
        print("  1. CDS_API_KEY is set in .env")
        print("  2. Models are trained (run src/train_models.py)")
        print("  3. Internet connection is available")
