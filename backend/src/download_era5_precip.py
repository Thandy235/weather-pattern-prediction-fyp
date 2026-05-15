"""
Download ERA5 total_precipitation for Choma 1990-2023.
Downloads year by year, extracts daily totals, saves to
data/era5/era5_choma_precip_daily.csv
Then merges with existing era5_choma_daily.csv.
"""

import os
import zipfile
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from dotenv import load_dotenv
from ecmwf.datastores import client as ecmwf_client

load_dotenv()

OUTPUT_DIR  = Path('data/era5/precip')
MERGED_OUT  = Path('data/era5/era5_choma_daily.csv')
ERA5_EXISTING = Path('data/era5/era5_choma_daily.csv')

LAT  = -16.8
LON  =  26.9
AREA = [LAT + 0.5, LON - 0.5, LAT - 0.5, LON + 0.5]  # N W S E

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_client():
    key = os.getenv('CDS_API_KEY', '').strip()
    if not key or key == 'your_api_key_here':
        raise ValueError("CDS_API_KEY not set in .env")
    return ecmwf_client.Client(url='https://cds.climate.copernicus.eu/api', key=key)


def download_precip_year(client, year: int) -> Path | None:
    out = OUTPUT_DIR / f'precip_{year}.nc'
    if out.exists():
        print(f"  {year} already downloaded")
        return out
    print(f"  Downloading precipitation {year}...")
    try:
        client.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': ['total_precipitation'],
                'year': str(year),
                'month': [f'{m:02d}' for m in range(1, 13)],
                'day':   [f'{d:02d}' for d in range(1, 32)],
                'time':  ['00:00', '06:00', '12:00', '18:00'],
                'area':  AREA,
                'format': 'netcdf',
            },
            str(out)
        )
        print(f"  ✓ {year} downloaded")
        return out
    except Exception as e:
        print(f"  ✗ {year} failed: {e}")
        return None


def nc_to_daily_precip(nc_file: Path) -> pd.DataFrame | None:
    """Extract daily precipitation total (mm) from a NetCDF file."""
    actual = nc_file

    # Handle ZIP wrapper from new CDS API
    with open(nc_file, 'rb') as fh:
        is_zip = fh.read(2) == b'PK'
    if is_zip:
        extract_dir = OUTPUT_DIR / 'extracted'
        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(nc_file, 'r') as zf:
            nc_members = [m for m in zf.namelist() if m.endswith('.nc')]
            if not nc_members:
                return None
            zf.extract(nc_members[0], extract_dir)
            actual = extract_dir / nc_members[0]

    ds  = xr.open_dataset(actual, engine='netcdf4')
    df  = ds.to_dataframe().reset_index()
    ds.close()

    time_col = 'valid_time' if 'valid_time' in df.columns else 'time'
    df['date'] = pd.to_datetime(df[time_col]).dt.date

    # Find precipitation column
    tp_col = next((c for c in df.columns if c in ['tp', 'total_precipitation']), None)
    if tp_col is None:
        print(f"  ⚠ No precipitation column in {nc_file.name}")
        return None

    daily = df.groupby('date')[tp_col].sum().reset_index()
    daily.columns = ['date', 'precipitation']
    # Convert m → mm
    daily['precipitation'] = (daily['precipitation'] * 1000).round(3).clip(lower=0)
    return daily


def main():
    print("\n" + "=" * 65)
    print("  DOWNLOADING ERA5 PRECIPITATION 1990–2023")
    print("=" * 65)

    client = get_client()
    all_precip = []

    for year in range(1990, 2024):
        nc_file = download_precip_year(client, year)
        if nc_file is None:
            continue
        daily = nc_to_daily_precip(nc_file)
        if daily is not None:
            all_precip.append(daily)

    if not all_precip:
        print("✗ No precipitation data downloaded.")
        return

    precip_df = pd.concat(all_precip, ignore_index=True)
    precip_df['date'] = pd.to_datetime(precip_df['date'])
    precip_df = precip_df.sort_values('date').reset_index(drop=True)

    print(f"\n  Precipitation data: {len(precip_df)} days")
    print(f"  Date range: {precip_df['date'].min().date()} → {precip_df['date'].max().date()}")
    print(f"  Mean daily precip: {precip_df['precipitation'].mean():.3f} mm")

    # Save standalone precip CSV
    precip_out = Path('data/era5/era5_choma_precip_daily.csv')
    precip_df.to_csv(precip_out, index=False)
    print(f"  Saved: {precip_out}")

    # Merge with existing ERA5 daily CSV
    if ERA5_EXISTING.exists():
        existing = pd.read_csv(ERA5_EXISTING, parse_dates=['date'])
        if 'precipitation' in existing.columns:
            existing = existing.drop(columns=['precipitation'])
        merged = existing.merge(precip_df, on='date', how='left')
        merged.to_csv(MERGED_OUT, index=False)
        print(f"  Merged into: {MERGED_OUT}")
        print(f"  Columns: {list(merged.columns)}")
        missing_precip = merged['precipitation'].isna().sum()
        print(f"  Missing precipitation values: {missing_precip}")
    else:
        print(f"  ⚠ {ERA5_EXISTING} not found — saving precip only")

    print("\n✓ ERA5 precipitation download complete.")


if __name__ == '__main__':
    main()
