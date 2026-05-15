"""
harmonize_era5_station.py — Merges ERA5 reanalysis data with Choma ground station data.

This script fulfils Objectives 1 & 2 of the project:
  Objective 1: Integrate ERA5 satellite reanalysis data with ground station observations.
  Objective 2: Apply quality control, outlier removal, and gap-filling.

Steps performed:
  1. Load ground station daily data (already preprocessed)
  2. Load ERA5 yearly NetCDF files and convert to daily CSV
  3. Align both datasets to a common daily time axis (1990-2023)
  4. Quality control: flag and remove outliers (IQR method)
  5. Gap-fill station rainfall using ERA5 as reference (bias-corrected)
  6. Gap-fill temperature and humidity via linear interpolation
  7. Produce a single unified CSV: data/processed/choma_harmonized_unified.csv

Output columns:
  date, rainfall, rainfall_era5, rainfall_final, rainfall_occurrence,
  max_temp, min_temp, humidity, temp_2m_era5, relative_humidity_era5,
  wind_speed_era5, surface_pressure_era5, water_vapour_era5,
  data_source (station | era5_filled | interpolated),
  qc_flag (ok | outlier_removed | gap_filled)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ERA5StationHarmonizer:
    """
    Merges and cleans the Choma ground station record with ERA5 reanalysis data.

    The ground station has real observations but gaps and occasional bad values.
    ERA5 is a global model — it covers every day but can be biased relative to
    the actual station.  This class combines the best of both: real observations
    where available, ERA5 (bias-corrected) to fill the gaps.
    """

    # File paths — all relative to the project root
    STATION_FILE   = Path('data/processed/choma_daily_data.csv')
    ERA5_CSV       = Path('data/era5/era5_choma_daily.csv')
    ERA5_DIR       = Path('data/era5')
    OUTPUT_FILE    = Path('data/processed/choma_harmonized_unified.csv')

    # A day is "rainy" only if rainfall is at least this many mm
    RAINFALL_THRESHOLD = 1.0   # mm

    # Choma station GPS coordinates (used for ERA5 spatial extraction)
    LAT = -16.82
    LON =  26.97

    def __init__(self):
        # Make sure the output directory exists before we try to write to it
        Path('data/processed').mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  STEP 1 — Load station data                                          #
    # ------------------------------------------------------------------ #
    def load_station(self) -> pd.DataFrame:
        """
        Load the preprocessed ground station CSV.

        This file is produced by data_preprocessing.py and contains one row
        per calendar day with columns like rainfall, max_temp, min_temp, humidity.

        Returns a DataFrame sorted by date.
        Raises FileNotFoundError if the file doesn't exist yet.
        """
        if not self.STATION_FILE.exists():
            raise FileNotFoundError(
                f"{self.STATION_FILE} not found. Run data_preprocessing.py first."
            )
        df = pd.read_csv(self.STATION_FILE, parse_dates=['date'])
        df = df.sort_values('date').reset_index(drop=True)
        print(f"  Station data loaded: {len(df):,} rows  "
              f"({df['date'].min().date()} → {df['date'].max().date()})")
        return df

    # ------------------------------------------------------------------ #
    #  STEP 2 — Load ERA5 daily CSV (produced by era5_downloader.py)       #
    # ------------------------------------------------------------------ #
    def load_era5(self) -> pd.DataFrame | None:
        """
        Load the ERA5 daily CSV that was produced by era5_downloader.py.

        If the file doesn't exist (e.g. the user hasn't downloaded ERA5 yet),
        we return None and the harmonizer will fall back to climatology gap-filling.

        Returns a DataFrame sorted by date, or None if the file is missing.
        """
        if not self.ERA5_CSV.exists():
            print(f"  ⚠ ERA5 CSV not found at {self.ERA5_CSV}.")
            print("    Run src/era5_downloader.py first to download ERA5 data.")
            return None

        df = pd.read_csv(self.ERA5_CSV, parse_dates=['date'])
        df = df.sort_values('date').reset_index(drop=True)
        print(f"  ERA5 data loaded:    {len(df):,} rows  "
              f"({df['date'].min().date()} → {df['date'].max().date()})")
        return df

    # ------------------------------------------------------------------ #
    #  STEP 3 — Align to common daily time axis                            #
    # ------------------------------------------------------------------ #
    def align_to_common_axis(
        self,
        station: pd.DataFrame,
        era5: pd.DataFrame | None
    ) -> pd.DataFrame:
        """
        Create a complete daily time series covering every calendar day in the
        station record, then left-join both datasets onto it.

        Why an outer join on a full date range?  The station data may have
        missing dates (e.g. no entry for a day when the observer was absent).
        By building a complete date spine first, we ensure every day is
        represented — missing days will have NaN values that we fill later.

        Parameters:
            station — preprocessed station DataFrame
            era5    — ERA5 daily DataFrame, or None if not available

        Returns a merged DataFrame with one row per calendar day.
        """
        start = station['date'].min()
        end   = station['date'].max()
        # Build a complete list of every day from start to end
        full_index = pd.DataFrame({'date': pd.date_range(start, end, freq='D')})

        # Left-join station data onto the full date spine
        merged = full_index.merge(station, on='date', how='left')

        if era5 is not None:
            # Collect all ERA5 columns (including 'date') to merge in
            era5_cols = ['date']
            for col in era5.columns:
                if col != 'date':
                    era5_cols.append(col)
            # Left-join ERA5 onto the merged frame; suffix '_era5' handles name clashes
            merged = merged.merge(era5[era5_cols], on='date', how='left',
                                  suffixes=('', '_era5'))

        print(f"  Aligned time axis:   {len(merged):,} days  "
              f"({merged['date'].min().date()} → {merged['date'].max().date()})")
        return merged

    # ------------------------------------------------------------------ #
    #  STEP 4 — Quality control (outlier detection)                        #
    # ------------------------------------------------------------------ #
    def quality_control(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flag and remove physically impossible or statistically extreme values.

        Two-stage approach:
          1. Hard bounds: values outside the physically plausible range for
             Zambia are set to NaN (e.g. rainfall > 300 mm/day is impossible).
          2. IQR outlier detection: values more than 3× the interquartile range
             beyond the 1st/99th percentile are also removed.  We use the 1st/99th
             rather than the standard 25th/75th to be conservative — we only want
             to catch genuine data entry errors, not just unusual-but-real events.

        Why IQR instead of z-scores?  IQR is more robust to skewed distributions
        like rainfall (which has a long right tail) because it's based on medians
        rather than means.

        Flagged values are set to NaN so they can be gap-filled in the next step.
        The 'qc_flag' column records what happened to each row.

        Returns the DataFrame with bad values replaced by NaN.
        """
        df = df.copy()
        df['qc_flag'] = 'ok'  # Start with everything marked as clean

        # Physical bounds for each variable in the Zambia context
        qc_rules = {
            'rainfall':  (0,    300),   # mm/day — 300 mm is extreme but possible in tropical storms
            'max_temp':  (15,   50),    # °C — Zambia's range; below 15 or above 50 is a sensor error
            'min_temp':  (-5,   35),    # °C
            'humidity':  (5,    100),   # % — can't be negative or above 100
        }

        for col, (lo, hi) in qc_rules.items():
            if col not in df.columns:
                continue

            # Stage 1: hard physical bounds
            mask_bounds = (df[col] < lo) | (df[col] > hi)

            # Stage 2: statistical outliers using 1st/99th percentile IQR
            q1 = df[col].quantile(0.01)
            q3 = df[col].quantile(0.99)
            iqr = q3 - q1
            # Flag anything more than 3 IQR widths beyond the 1st/99th percentile
            mask_iqr = (df[col] < q1 - 3 * iqr) | (df[col] > q3 + 3 * iqr)

            # Combine both masks — flag if either condition is true
            mask = mask_bounds | mask_iqr
            n_flagged = mask.sum()
            if n_flagged > 0:
                df.loc[mask, col] = np.nan          # Replace bad value with NaN
                df.loc[mask, 'qc_flag'] = 'outlier_removed'
                print(f"    QC: {n_flagged} outliers removed from '{col}'")

        return df

    # ------------------------------------------------------------------ #
    #  STEP 5 — Gap-fill rainfall using ERA5 (bias-corrected)              #
    # ------------------------------------------------------------------ #
    def gap_fill_rainfall(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing station rainfall values using ERA5 precipitation as a proxy.

        ERA5 is a global model and tends to be systematically biased relative to
        local station observations — it might consistently over- or under-estimate
        rainfall in a particular month.  We correct for this by computing a
        monthly bias factor: the ratio of station mean to ERA5 mean for each month.

        IMPORTANT: The bias factor is computed ONLY from days where the station
        had genuine observations (not from days that were already gap-filled by
        an earlier step).  This prevents circular bias — we never correct ERA5
        against values that were themselves derived from ERA5.

        After ERA5 gap-filling, any days still missing are filled using the
        day-of-year climatological mean (last resort).

        Parameters:
            df — the aligned, QC'd DataFrame

        Returns the DataFrame with rainfall gaps filled and 'data_source' updated.
        """
        df = df.copy()
        df['data_source'] = 'station'  # Default: assume all rows are real observations

        # Find the ERA5 precipitation column (name varies by dataset version)
        era5_col = None
        for candidate in ['precipitation', 'tp', 'rainfall_era5']:
            if candidate in df.columns:
                era5_col = candidate
                break

        if era5_col is None:
            # No ERA5 data available — fall back to climatology for all gaps
            print("  ⚠ No ERA5 precipitation column found — skipping ERA5 gap-fill.")
            df = self._gap_fill_by_climatology(df, 'rainfall')
            return df

        # ── Compute bias correction from original station observations only ──
        # We identify "original" rows as those where rainfall is not NaN at this
        # point in the pipeline (before we do any filling here).
        original_mask = df['rainfall'].notna() & df[era5_col].notna()

        # If the preprocessing step saved a 'rainfall_observed' flag, use it
        # to restrict to genuine station observations only (not climatology fills).
        if 'rainfall_observed' in df.columns:
            original_mask = original_mask & (df['rainfall_observed'] == 1)
            print(f"  Using 'rainfall_observed' flag — excluding climatology-filled rows from bias correction")

        overlap = df[original_mask].copy()

        if len(overlap) > 30:
            # Compute the ratio of station mean to ERA5 mean for each calendar month.
            # For example, if in January the station averages 8 mm/day but ERA5 says 5 mm/day,
            # the January bias factor is 8/5 = 1.6 — we'll multiply ERA5 by 1.6 when filling.
            overlap['month'] = overlap['date'].dt.month
            monthly_bias = (
                overlap.groupby('month')
                .apply(lambda g: g['rainfall'].mean() / g[era5_col].mean()
                       if g[era5_col].mean() > 0 else 1.0)
                .rename('bias_factor')
                .reset_index()
            )
            n_months_with_data = len(monthly_bias[monthly_bias['bias_factor'] != 1.0])
            print(f"  Bias correction:     computed from {len(overlap):,} original station days "
                  f"across {n_months_with_data}/12 months")
            # Merge the monthly bias factors back onto the main DataFrame
            df['month'] = df['date'].dt.month
            df = df.merge(monthly_bias, on='month', how='left')
            df['bias_factor'] = df['bias_factor'].fillna(1.0)  # 1.0 = no correction
        else:
            # Not enough overlapping days to compute a reliable bias — use 1.0 (no correction)
            print(f"  ⚠ Only {len(overlap)} overlapping days — using bias_factor=1.0 (no correction)")
            df['bias_factor'] = 1.0

        # ── Fill only the genuinely missing station days ──────────────────
        missing_mask   = df['rainfall'].isna()          # Days with no station data
        era5_available = df[era5_col].notna()           # Days where ERA5 has a value
        fill_mask      = missing_mask & era5_available  # Fill only where both conditions met

        # Apply bias-corrected ERA5 value; clip to 0 so we never get negative rainfall
        df.loc[fill_mask, 'rainfall'] = (
            df.loc[fill_mask, era5_col] * df.loc[fill_mask, 'bias_factor']
        ).clip(lower=0)
        df.loc[fill_mask, 'data_source'] = 'era5_filled'
        # Update QC flag for filled rows (only if they weren't already flagged as outliers)
        df.loc[fill_mask, 'qc_flag'] = df.loc[fill_mask, 'qc_flag'].replace('ok', 'gap_filled')

        # Any days still missing after ERA5 fill → use day-of-year climatology as last resort
        still_missing = df['rainfall'].isna().sum()
        if still_missing > 0:
            df = self._gap_fill_by_climatology(df, 'rainfall')

        n_filled = fill_mask.sum()
        print(f"  Gap-fill rainfall:   {n_filled} days filled from ERA5  "
              f"({still_missing} remaining → climatology)")

        # Clean up temporary columns we added during this step
        df.drop(columns=['bias_factor', 'month'], errors='ignore', inplace=True)
        return df

    def _gap_fill_by_climatology(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Fill any remaining gaps using the historical average for that day of year.

        For example, if 15 January is missing, we fill it with the average of all
        15 January values across the full record.  This is the simplest possible
        gap-fill — it preserves the seasonal cycle but adds no year-to-year variation.

        Parameters:
            df  — DataFrame with a 'date' column
            col — name of the column to fill

        Returns the DataFrame with remaining NaNs in `col` replaced.
        """
        df = df.copy()
        df['_doy'] = df['date'].dt.dayofyear  # Day of year (1–365)
        # Compute the mean value for each day of year across all years
        clim = df.groupby('_doy')[col].mean().rename('_clim')
        df = df.merge(clim, on='_doy', how='left')
        still_missing = df[col].isna()
        # Fill only the rows that are still NaN
        df.loc[still_missing, col] = df.loc[still_missing, '_clim']
        df.loc[still_missing, 'data_source'] = 'interpolated'
        df.loc[still_missing, 'qc_flag'] = 'gap_filled'
        # Remove the temporary helper columns
        df.drop(columns=['_doy', '_clim'], inplace=True)
        n = still_missing.sum()
        if n > 0:
            print(f"    Climatology gap-fill: {n} days filled for '{col}'")
        return df

    # ------------------------------------------------------------------ #
    #  STEP 6 — Gap-fill temperature and humidity                          #
    # ------------------------------------------------------------------ #
    def gap_fill_met_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing temperature and humidity values using a three-tier approach:

          1. Linear interpolation: for short gaps (≤ 7 consecutive missing days),
             we interpolate between the surrounding valid values.  This works well
             because temperature changes smoothly from day to day.

          2. ERA5 fallback: for longer gaps, we use the ERA5 equivalent variable.
             For max/min temperature we offset the ERA5 mean temperature by ±5°C
             as a rough approximation of the daily range.

          3. Climatology: any remaining gaps are filled with the day-of-year mean
             (same approach as for rainfall).

        Parameters:
            df — the DataFrame after rainfall gap-filling

        Returns the DataFrame with temperature and humidity gaps filled.
        """
        df = df.copy()

        # Which ERA5 column to use as a fallback for each station variable
        fallback_map = {
            'max_temp':  'temp_2m_era5',           # ERA5 2m temperature → approximate max
            'min_temp':  'temp_2m_era5',           # ERA5 2m temperature → approximate min
            'humidity':  'relative_humidity_era5', # ERA5 relative humidity
        }

        for col, era5_col in fallback_map.items():
            if col not in df.columns:
                continue

            # Tier 1: linear interpolation for short gaps (up to 7 days)
            df[col] = df[col].interpolate(method='linear', limit=7)

            # Tier 2: ERA5 fallback for longer gaps
            if era5_col in df.columns:
                still_missing = df[col].isna()
                if still_missing.any():
                    if col == 'max_temp':
                        # ERA5 gives mean temperature; add 5°C to approximate the daily maximum
                        df.loc[still_missing, col] = df.loc[still_missing, era5_col] + 5
                    elif col == 'min_temp':
                        # Subtract 5°C from ERA5 mean to approximate the daily minimum
                        df.loc[still_missing, col] = df.loc[still_missing, era5_col] - 5
                    else:
                        # For humidity, use ERA5 directly
                        df.loc[still_missing, col] = df.loc[still_missing, era5_col]
                    n = still_missing.sum()
                    print(f"    ERA5 fallback: {n} days filled for '{col}'")

            # Tier 3: climatology for anything still missing after the above two steps
            df = self._gap_fill_by_climatology(df, col)

        return df

    # ------------------------------------------------------------------ #
    #  STEP 7 — Finalise and save                                          #
    # ------------------------------------------------------------------ #
    def finalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Recompute derived columns, enforce consistent data types, and sort.

        This is the last step before saving.  We:
          - Recompute rainfall_occurrence (binary rain/no-rain flag)
          - Rename the ERA5 precipitation column to 'rainfall_era5' for clarity
          - Add calendar columns (year, month, day, day_of_year, season)
          - Round all numeric columns to 4 decimal places
          - Sort by date

        Returns the cleaned, finalised DataFrame.
        """
        df = df.copy()

        # Recompute the binary rain/no-rain flag based on the filled rainfall column
        df['rainfall_occurrence'] = (
            df['rainfall'].fillna(0) >= self.RAINFALL_THRESHOLD
        ).astype(int)

        # Standardise the ERA5 precipitation column name
        for candidate in ['precipitation', 'tp']:
            if candidate in df.columns:
                df.rename(columns={candidate: 'rainfall_era5'}, inplace=True)
                break

        # Add calendar features — these are useful as model inputs (seasonality)
        df['year']        = df['date'].dt.year
        df['month']       = df['date'].dt.month
        df['day']         = df['date'].dt.day
        df['day_of_year'] = df['date'].dt.dayofyear
        df['season']      = df['month'].map(self._get_season)

        # Round all numeric columns to keep the CSV file size manageable
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].round(4)

        # Sort chronologically
        df = df.sort_values('date').reset_index(drop=True)
        return df

    @staticmethod
    def _get_season(month: int) -> str:
        """
        Map a month number to one of Zambia's three main seasons.

        Zambia's climate is dominated by a single rainy season (November–April)
        followed by a cool dry season (May–August) and a hot dry season (September–October).

        Parameters:
            month — integer month (1–12)

        Returns a string season label.
        """
        if month in [11, 12, 1, 2, 3, 4]:
            return 'hot_rainy'      # Nov–Apr: hot and rainy — the main growing season
        elif month in [5, 6, 7, 8]:
            return 'cool_dry'       # May–Aug: cool and dry — harvest and storage season
        else:
            return 'hot_dry'        # Sep–Oct: hot and dry — land preparation season

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #
    def harmonize(self) -> pd.DataFrame:
        """
        Run the full harmonization pipeline from raw inputs to unified output.

        Calls each step in order, prints progress, saves the result, and
        prints a summary report showing how many days came from each source.

        Returns the final harmonized DataFrame.
        """
        print("\n" + "=" * 65)
        print("  OBJECTIVE 1 & 2: DATA HARMONIZATION & QUALITY CONTROL")
        print("=" * 65)

        print("\n[1/6] Loading station data...")
        station = self.load_station()

        print("\n[2/6] Loading ERA5 data...")
        era5 = self.load_era5()

        print("\n[3/6] Aligning to common daily time axis...")
        df = self.align_to_common_axis(station, era5)

        print("\n[4/6] Quality control (outlier detection)...")
        df = self.quality_control(df)

        print("\n[5/6] Gap-filling rainfall (ERA5 bias-corrected)...")
        df = self.gap_fill_rainfall(df)

        print("\n[6/6] Gap-filling temperature & humidity...")
        df = self.gap_fill_met_variables(df)

        # Final cleanup and derived columns
        df = self.finalise(df)

        # Save the unified dataset
        df.to_csv(self.OUTPUT_FILE, index=False)

        # Print a summary so the user can see what happened
        n_total      = len(df)
        n_station    = (df['data_source'] == 'station').sum()
        n_era5_fill  = (df['data_source'] == 'era5_filled').sum()
        n_interp     = (df['data_source'] == 'interpolated').sum()
        n_outliers   = (df['qc_flag'] == 'outlier_removed').sum()
        n_gap_filled = (df['qc_flag'] == 'gap_filled').sum()
        # Average missing rate across the four key variables
        missing_pct  = df[['rainfall', 'max_temp', 'min_temp', 'humidity']].isna().mean().mean() * 100

        print(f"\n{'=' * 65}")
        print(f"  HARMONIZATION COMPLETE")
        print(f"{'=' * 65}")
        print(f"  Output file  : {self.OUTPUT_FILE}")
        print(f"  Total days   : {n_total:,}")
        print(f"  Date range   : {df['date'].min().date()} → {df['date'].max().date()}")
        print(f"\n  Data sources:")
        print(f"    Station observations : {n_station:,} days ({n_station/n_total*100:.1f}%)")
        print(f"    ERA5 gap-filled      : {n_era5_fill:,} days ({n_era5_fill/n_total*100:.1f}%)")
        print(f"    Climatology filled   : {n_interp:,} days ({n_interp/n_total*100:.1f}%)")
        print(f"\n  Quality control:")
        print(f"    Outliers removed     : {n_outliers:,}")
        print(f"    Gaps filled          : {n_gap_filled:,}")
        print(f"    Remaining missing    : {missing_pct:.2f}%")
        print(f"\n  Columns: {list(df.columns)}")
        print(f"{'=' * 65}")

        return df


# Run harmonization directly when this script is called
if __name__ == '__main__':
    harmonizer = ERA5StationHarmonizer()
    df = harmonizer.harmonize()
    print("\n✓ Objectives 1 & 2 complete: unified harmonized dataset produced.")
