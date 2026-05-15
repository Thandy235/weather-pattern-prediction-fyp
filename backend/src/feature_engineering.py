"""
feature_engineering.py — Builds the full feature matrix used to train the ML models.

This script takes the cleaned, harmonized daily dataset (station + ERA5) and
transforms it into a rich set of input features: lag values (what happened
N days ago), rolling statistics (averages and spreads over recent windows),
interaction terms (combinations of variables that together carry more signal),
and the target labels the models are trained to predict.

The output is saved to data/processed/features_complete.csv and is the direct
input to train_models.py.
"""

import pandas as pd
import numpy as np
from pathlib import Path


class FeatureEngineer:
    """
    Handles all feature construction for the rainfall prediction pipeline.

    The class reads the harmonized dataset, builds several categories of
    features, attaches target labels for each forecast horizon, and saves
    the result.  Each method is responsible for one category of features so
    it's easy to add or remove feature groups without touching the rest.
    """

    def __init__(self):
        # Where the processed data files live
        self.processed_dir = Path('data/processed')

        # We prefer the harmonized file (station + ERA5 merged) because it has
        # more variables and fewer gaps.  If it doesn't exist yet, we fall back
        # to the station-only file.
        self.harmonized_file = self.processed_dir / 'choma_harmonized_unified.csv'
        self.station_file    = self.processed_dir / 'choma_daily_data.csv'

    # ------------------------------------------------------------------ #
    #  Data loading                                                        #
    # ------------------------------------------------------------------ #
    def load_data(self) -> pd.DataFrame:
        """
        Load the best available daily dataset.

        Tries the harmonized (station + ERA5) file first.  If that doesn't
        exist, falls back to the station-only file.  Raises an error if
        neither file is found — the preprocessing steps must run first.

        Returns a DataFrame sorted by date with the index reset.
        """
        if self.harmonized_file.exists():
            print(f"  Loading harmonized dataset: {self.harmonized_file}")
            df = pd.read_csv(self.harmonized_file, parse_dates=['date'])
            # Show which ERA5 columns are present so we know what extra features we have
            print(f"  Shape: {df.shape}  |  "
                  f"ERA5 columns present: "
                  f"{[c for c in df.columns if 'era5' in c.lower() or c in ['temp_2m','relative_humidity','wind_speed','surface_pressure','water_vapour']]}")
        elif self.station_file.exists():
            # Warn the user — ERA5 features won't be available
            print(f"  ⚠ Harmonized file not found. Using station-only data.")
            print(f"    Run src/harmonize_era5_station.py to produce the unified dataset.")
            df = pd.read_csv(self.station_file, parse_dates=['date'])
        else:
            raise FileNotFoundError(
                "No processed data found. Run data_preprocessing.py first, "
                "then harmonize_era5_station.py."
            )
        # Sort chronologically so lag/rolling operations work correctly
        return df.sort_values('date').reset_index(drop=True)

    # ------------------------------------------------------------------ #
    #  Lag features                                                        #
    # ------------------------------------------------------------------ #
    def create_lag_features(self, df: pd.DataFrame, variables: list,
                            lags: list = [1, 3, 7, 14, 30]) -> pd.DataFrame:
        """
        Create lagged copies of each variable — i.e. "what was the value N days ago?"

        Why lags?  Rainfall is autocorrelated: if it rained yesterday, there's a
        higher chance it rains today.  By giving the model yesterday's rainfall
        (lag=1), last week's (lag=7), etc., it can learn these persistence patterns.

        Parameters:
            df        — the daily DataFrame, must be sorted by date
            variables — list of column names to lag
            lags      — list of day offsets to create (e.g. [1, 3, 7, 14, 30])

        Returns the DataFrame with new columns like 'rainfall_lag7'.
        """
        print(f"  Creating lag features for {len(variables)} variables × {len(lags)} lags...")
        df = df.sort_values('date').reset_index(drop=True)
        for var in variables:
            if var not in df.columns:
                continue  # Skip variables that aren't in this dataset
            for lag in lags:
                # shift(lag) moves the column down by `lag` rows, so row i gets the value from row i-lag
                df[f'{var}_lag{lag}'] = df[var].shift(lag)
        return df

    # ------------------------------------------------------------------ #
    #  Rolling statistics                                                  #
    # ------------------------------------------------------------------ #
    def create_rolling_features(self, df: pd.DataFrame, variables: list,
                                windows: list = [7, 14, 30]) -> pd.DataFrame:
        """
        Compute rolling (moving window) statistics for each variable.

        Why rolling features?  A single day's value can be noisy.  The 7-day
        rolling mean of rainfall tells us whether we're in a wet or dry spell,
        which is much more informative for longer-horizon forecasts.

        For each variable and window size we compute:
          - mean  : average over the window (trend)
          - std   : standard deviation (how variable the weather has been)
          - max   : highest value in the window (recent extremes)
          - min   : lowest value in the window (recent dry/cold spells)

        min_periods=1 means we still compute a value even at the start of the
        series where there aren't enough rows to fill the full window.

        Parameters:
            df      — daily DataFrame sorted by date
            variables — columns to compute rolling stats for
            windows — window sizes in days

        Returns the DataFrame with new columns like 'rainfall_roll7_mean'.
        """
        print(f"  Creating rolling features for {len(variables)} variables × {len(windows)} windows...")
        for var in variables:
            if var not in df.columns:
                continue
            for w in windows:
                df[f'{var}_roll{w}_mean'] = df[var].rolling(w, min_periods=1).mean()
                df[f'{var}_roll{w}_std']  = df[var].rolling(w, min_periods=1).std()
                df[f'{var}_roll{w}_max']  = df[var].rolling(w, min_periods=1).max()
                df[f'{var}_roll{w}_min']  = df[var].rolling(w, min_periods=1).min()
        return df

    # ------------------------------------------------------------------ #
    #  Interaction / derived features                                      #
    # ------------------------------------------------------------------ #
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived and interaction features that combine existing variables.

        These capture physical relationships that a single variable can't express:
          - temp_range: the daily temperature swing (hot days + cold nights = convective storms)
          - humidity_temp_interaction: hot + humid conditions favour heavy rainfall
          - era5_heat_index: a proxy for atmospheric instability
          - moisture_transport: wind × water vapour = how much moisture is being advected in
          - rainfall_anomaly: today's rain vs the 30-day average (is it unusually wet/dry?)
          - consecutive_dry_days: how long since it last rained (soil moisture proxy)

        Returns the DataFrame with the new columns added.
        """
        print("  Creating interaction features...")

        # ── Station-based interactions ──────────────────────────────────
        if 'max_temp' in df.columns and 'min_temp' in df.columns:
            # Daily temperature range — large swings often precede convective rainfall
            df['temp_range'] = df['max_temp'] - df['min_temp']
            # Simple daily average temperature
            df['avg_temp']   = (df['max_temp'] + df['min_temp']) / 2

        if 'humidity' in df.columns and 'avg_temp' in df.columns:
            # Hot + humid = more energy available for rainfall
            df['humidity_temp_interaction'] = df['humidity'] * df['avg_temp']

        # ── ERA5-based interactions (only present in the harmonized dataset) ──
        # We try both the '_era5' suffixed name and the plain name to handle
        # different column naming conventions across dataset versions.
        t2m_col = next((c for c in ['temp_2m_era5', 'temp_2m'] if c in df.columns), None)
        rh_col  = next((c for c in ['relative_humidity_era5', 'relative_humidity'] if c in df.columns), None)
        ws_col  = next((c for c in ['wind_speed_era5', 'wind_speed'] if c in df.columns), None)
        wv_col  = next((c for c in ['water_vapour_era5', 'water_vapour'] if c in df.columns), None)

        if t2m_col and rh_col:
            # A simplified heat index — higher values mean more atmospheric energy
            df['era5_heat_index'] = df[t2m_col] + 0.5 * df[rh_col]

        if ws_col and wv_col:
            # Moisture transport: how much water vapour is being blown into the area.
            # High wind × high water vapour = conditions ripe for rainfall.
            df['moisture_transport'] = df[ws_col] * df[wv_col]

        # ── Rainfall anomaly ────────────────────────────────────────────
        if 'rainfall' in df.columns:
            # Compare today's rainfall to the 30-day rolling mean.
            # A positive anomaly means it's wetter than usual; negative means drier.
            roll30 = df['rainfall'].rolling(30, min_periods=1).mean()
            df['rainfall_anomaly'] = df['rainfall'] - roll30

        # ── Consecutive dry days ────────────────────────────────────────
        if 'rainfall_occurrence' in df.columns:
            # Count how many days in a row it hasn't rained up to the current day.
            # This is a proxy for soil moisture deficit — the longer the dry spell,
            # the drier the soil, which affects how much rain is absorbed vs runs off.
            dry = (df['rainfall_occurrence'] == 0).astype(int)
            # Group consecutive runs of the same value (0 or 1) and count within each run
            groups = dry.groupby((dry != dry.shift()).cumsum())
            df['consecutive_dry_days'] = groups.cumcount() + 1
            # Reset the counter to 0 on days when it actually rained
            df.loc[df['rainfall_occurrence'] == 1, 'consecutive_dry_days'] = 0

        return df

    # ------------------------------------------------------------------ #
    #  Target variables                                                    #
    # ------------------------------------------------------------------ #
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create the labels (targets) that the models are trained to predict.

        For each forecast horizon (1, 7, 30, 90 days ahead) we create two targets:
          - occurrence: will it rain at all? (0 = no, 1 = yes) — for the classifier
          - amount: how many mm will fall? — for the regressor

        We use shift(-days) to "look forward" in time.  For example, shift(-1)
        on row i gives the value from row i+1, which is tomorrow's rainfall.
        This means the last N rows will have NaN targets (there's no future data
        for them), and we drop those rows later.

        Returns the DataFrame with 8 new target columns.
        """
        print("  Creating target variables (1, 7, 30, 90 day horizons)...")
        df = df.sort_values('date').reset_index(drop=True)

        for days in [1, 7, 30, 90]:
            # Shift rainfall_occurrence forward by `days` rows to get the future label
            df[f'target_{days}day_occurrence'] = df['rainfall_occurrence'].shift(-days)
            # Same for the actual rainfall amount
            df[f'target_{days}day_amount']     = df['rainfall'].shift(-days)

        return df

    # ------------------------------------------------------------------ #
    #  Main pipeline                                                       #
    # ------------------------------------------------------------------ #
    def process_all(self) -> pd.DataFrame:
        """
        Run the complete feature engineering pipeline end-to-end.

        Steps:
          1. Load the harmonized (or station-only) dataset
          2. Identify which variable groups are available
          3. Build lag features, rolling features, interaction features
          4. Attach target labels for all four forecast horizons
          5. Drop rows where any target is NaN (the last 90 rows)
          6. Save the result to data/processed/features_complete.csv

        Returns the final feature DataFrame.
        """
        print("\n" + "=" * 65)
        print("  FEATURE ENGINEERING")
        print("=" * 65)

        # Step 1: Load the best available data
        df = self.load_data()

        # Step 2: Figure out which columns we actually have to work with.
        # We separate station variables from ERA5 variables so we can report
        # on both and apply lag/rolling to all of them.
        station_vars = [c for c in ['rainfall', 'humidity', 'max_temp', 'min_temp']
                        if c in df.columns]
        era5_vars    = [c for c in [
                            'temp_2m_era5', 'temp_2m',
                            'relative_humidity_era5', 'relative_humidity',
                            'wind_speed_era5', 'wind_speed',
                            'surface_pressure_era5', 'surface_pressure',
                            'water_vapour_era5', 'water_vapour',
                            'rainfall_era5'
                        ] if c in df.columns]

        all_vars = station_vars + era5_vars
        print(f"\n  Station variables : {station_vars}")
        print(f"  ERA5 variables    : {era5_vars}")

        # Step 3: Build all feature categories
        df = self.create_lag_features(df, all_vars)
        df = self.create_rolling_features(df, all_vars)
        df = self.create_interaction_features(df)

        # Step 4: Attach target labels
        df = self.create_target_variables(df)

        # Step 5: Drop rows where occurrence targets are NaN.
        # These are the last 90 rows — we shifted forward by up to 90 days,
        # so those rows have no future label to train against.
        target_cols = [c for c in df.columns if c.startswith('target_') and 'occurrence' in c]
        df = df.dropna(subset=target_cols)

        # Step 6: Save the complete feature matrix
        output_file = self.processed_dir / 'features_complete.csv'
        df.to_csv(output_file, index=False)

        # Print a breakdown of what was created
        lag_cols    = [c for c in df.columns if '_lag' in c]
        roll_cols   = [c for c in df.columns if '_roll' in c]
        target_cols = [c for c in df.columns if c.startswith('target_')]
        other_cols  = df.shape[1] - len(lag_cols) - len(roll_cols) - len(target_cols)

        print(f"\n  ✓ Feature engineering complete")
        print(f"  Output : {output_file}")
        print(f"  Shape  : {df.shape}")
        print(f"\n  Feature breakdown:")
        print(f"    Lag features     : {len(lag_cols)}")
        print(f"    Rolling features : {len(roll_cols)}")
        print(f"    Target variables : {len(target_cols)}")
        print(f"    Other features   : {other_cols}")
        print(f"\n  1-day target distribution:")
        print(df['target_1day_occurrence'].value_counts().to_string())

        return df


# Run feature engineering directly when this script is called
if __name__ == '__main__':
    engineer = FeatureEngineer()
    df = engineer.process_all()
