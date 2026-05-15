"""
era5_downloader.py — Downloads ERA5 reanalysis data from the Copernicus Climate Data Store.

ERA5 is a global atmospheric reanalysis produced by ECMWF.  It provides hourly
estimates of dozens of meteorological variables going back to 1940, derived by
combining historical weather observations with a numerical weather model.

This script downloads the variables we need for Choma, Zambia (1990–2023),
saves them as yearly NetCDF files, and then converts them to a single daily CSV
that the harmonization step can use.

You need a free CDS API account to use this script:
  Register at: https://cds.climate.copernicus.eu/
  Then add your API key to the .env file as CDS_API_KEY=your_key_here
"""

import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from ecmwf.datastores import client as ecmwf_client

# Load the CDS_API_KEY from the .env file in the project root
load_dotenv()


class ERA5Downloader:
    """
    Handles downloading ERA5 data from the Copernicus CDS API and converting
    the resulting NetCDF files to a daily CSV for use in the pipeline.
    """

    def __init__(self):
        # Where to save the downloaded NetCDF files
        self.output_dir = Path('data/era5')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Choma station coordinates (decimal degrees; negative = south)
        self.latitude = -16.8
        self.longitude = 26.9
        
        # Bounding box for the ERA5 download: [North, West, South, East]
        # We add a 0.5° buffer around the station so we capture the nearest grid cell
        self.area = [
            self.latitude + 0.5,   # North boundary
            self.longitude - 0.5,  # West boundary
            self.latitude - 0.5,   # South boundary
            self.longitude + 0.5   # East boundary
        ]
        
    def setup_cds_client(self):
        """
        Create and return an authenticated CDS API client.

        Reads the API key from the CDS_API_KEY environment variable (set in .env).
        Raises a ValueError with helpful instructions if the key is missing.

        Returns an ecmwf_client.Client object ready to make API requests.
        """
        api_key = os.getenv('CDS_API_KEY')

        if not api_key or api_key == 'your_api_key_here':
            raise ValueError(
                "Please set CDS_API_KEY in .env file\n"
                "Register at: https://cds.climate.copernicus.eu/"
            )

        api_url = 'https://cds.climate.copernicus.eu/api'
        return ecmwf_client.Client(url=api_url, key=api_key)
    
    def download_era5_data(self, start_year=1990, end_year=2023):
        """
        Download ERA5 single-level reanalysis data for Choma, year by year.

        We download one year at a time to avoid API timeouts and to allow
        resuming if the download is interrupted — already-downloaded years
        are skipped automatically.

        Variables downloaded:
          - 2m temperature (air temperature near the surface)
          - 2m dewpoint temperature (used to calculate relative humidity)
          - Surface pressure (atmospheric pressure at ground level)
          - 10m u/v wind components (east-west and north-south wind)
          - Total column water vapour (total moisture in the atmosphere)
          - Total precipitation (accumulated rainfall)

        Each year is saved as a separate NetCDF file: era5_choma_YYYY.nc

        Parameters:
            start_year — first year to download (default 1990)
            end_year   — last year to download (default 2023)
        """
        
        print("Setting up CDS API client...")
        client = self.setup_cds_client()
        
        print(f"\nDownloading ERA5 data for Choma, Zambia")
        print(f"Coordinates: {self.latitude}°S, {self.longitude}°E")
        print(f"Period: {start_year}-{end_year}")
        print("\nThis may take a while...")
        
        # Loop through each year and download separately
        for year in range(start_year, end_year + 1):
            output_file = self.output_dir / f'era5_choma_{year}.nc'
            
            # Skip years we've already downloaded
            if output_file.exists():
                print(f"✓ {year} already downloaded")
                continue
            
            print(f"\nDownloading {year}...")
            
            try:
                # Request data from the CDS API
                client.retrieve(
                    'reanalysis-era5-single-levels',  # The ERA5 dataset name
                    {
                        'product_type': 'reanalysis',
                        'variable': [
                            '2m_temperature',
                            '2m_dewpoint_temperature',
                            'surface_pressure',
                            '10m_u_component_of_wind',
                            '10m_v_component_of_wind',
                            'total_column_water_vapour',
                            'total_precipitation',
                        ],
                        'year': str(year),
                        # All 12 months, zero-padded (e.g. '01', '02', ..., '12')
                        'month': [f'{m:02d}' for m in range(1, 13)],
                        # All possible days (the API ignores invalid dates like Feb 30)
                        'day': [f'{d:02d}' for d in range(1, 32)],
                        # Four time steps per day: midnight, 6am, noon, 6pm (UTC)
                        'time': [
                            '00:00', '06:00', '12:00', '18:00'
                        ],
                        'area': self.area,   # Bounding box around Choma
                        'format': 'netcdf',  # Download as NetCDF (standard climate format)
                    },
                    str(output_file)
                )
                print(f"✓ {year} downloaded successfully")
                
            except Exception as e:
                print(f"✗ Error downloading {year}: {e}")
                continue  # Skip this year and try the next one
        
        print("\n✓ ERA5 download complete!")
        self._create_download_summary()
    
    def _create_download_summary(self):
        """
        Print a summary of all downloaded ERA5 files: how many, which years,
        and total disk space used.  Useful for a quick sanity check after downloading.
        """
        files = list(self.output_dir.glob('era5_choma_*.nc'))
        
        if not files:
            print("\nNo ERA5 files found!")
            return
        
        print(f"\n{'='*50}")
        print("ERA5 Download Summary")
        print(f"{'='*50}")
        print(f"Total files: {len(files)}")
        print(f"Location: {self.output_dir}")
        
        # Extract the year from each filename and find the range
        years = sorted([int(f.stem.split('_')[-1]) for f in files])
        print(f"Years: {min(years)} - {max(years)}")
        
        # Sum up file sizes and convert bytes → megabytes
        total_size = sum(f.stat().st_size for f in files) / (1024**2)
        print(f"Total size: {total_size:.2f} MB")
        print(f"{'='*50}")
    
    def convert_to_csv(self):
        """
        Convert all downloaded ERA5 NetCDF files into a single daily CSV.

        NetCDF is a scientific data format that stores multi-dimensional arrays.
        We need to:
          1. Open each yearly NetCDF file
          2. Handle ZIP wrapping (the new CDS API wraps files in a ZIP archive)
          3. Aggregate the 4 daily time steps into a single daily value
          4. Convert units to human-friendly values (Kelvin → °C, Pa → hPa, etc.)
          5. Compute derived variables (relative humidity, wind speed)
          6. Combine all years and save as era5_choma_daily.csv

        The output CSV is what harmonize_era5_station.py reads.
        """
        import zipfile
        import xarray as xr  # xarray is the standard library for reading NetCDF files

        print("\nConverting NetCDF to CSV...")

        # Find all yearly NetCDF files, sorted chronologically
        nc_files = sorted(self.output_dir.glob('era5_choma_*.nc'))

        if not nc_files:
            print("No NetCDF files found. Run download first.")
            return

        all_data = []  # We'll collect one DataFrame per year and concatenate at the end

        for nc_file in nc_files:
            print(f"Processing {nc_file.name}...")

            # The new CDS API (post-Sept 2024) wraps NetCDF files inside a ZIP archive.
            # We detect this by checking the first two bytes — ZIP files start with 'PK'.
            actual_nc = nc_file
            with open(nc_file, 'rb') as fh:
                is_zip = fh.read(2) == b'PK'

            if is_zip:
                # Extract the .nc file from inside the ZIP
                extract_dir = self.output_dir / 'extracted'
                extract_dir.mkdir(exist_ok=True)
                with zipfile.ZipFile(nc_file, 'r') as zf:
                    members = zf.namelist()
                    nc_members = [m for m in members if m.endswith('.nc')]
                    if not nc_members:
                        print(f"  No .nc file inside {nc_file.name}, skipping")
                        continue
                    zf.extract(nc_members[0], extract_dir)
                    actual_nc = extract_dir / nc_members[0]

            # Open the NetCDF file and convert to a flat pandas DataFrame
            ds = xr.open_dataset(actual_nc, engine='netcdf4')
            df = ds.to_dataframe().reset_index()

            # Extract the date from the time column (ERA5 uses 'valid_time' in newer versions)
            df['date'] = pd.to_datetime(df['valid_time'] if 'valid_time' in df.columns else df['time']).dt.date
            
            # ERA5 variable names can differ between API versions.
            # This map lists the standard short name and possible alternative names.
            var_map = {
                't2m':  ['t2m', '2m_temperature'],
                'd2m':  ['d2m', '2m_dewpoint_temperature'],
                'sp':   ['sp', 'surface_pressure'],
                'u10':  ['u10', '10m_u_component_of_wind'],
                'v10':  ['v10', '10m_v_component_of_wind'],
                'tcwv': ['tcwv', 'total_column_water_vapour'],
                'tp':   ['tp', 'total_precipitation'],
            }
            
            # Build the aggregation dictionary: most variables → daily mean,
            # but precipitation → daily sum (we want total rain, not average)
            agg = {}
            col_rename = {}
            for std_name, candidates in var_map.items():
                for c in candidates:
                    if c in df.columns:
                        agg[c] = 'mean' if std_name != 'tp' else 'sum'
                        col_rename[c] = std_name
                        break

            # Aggregate from 4 time steps per day to 1 daily value
            daily = df.groupby('date').agg(agg).reset_index()
            daily.rename(columns=col_rename, inplace=True)

            all_data.append(daily)
            ds.close()  # Free memory

        if not all_data:
            print("No data could be processed.")
            return

        # Stack all years into one DataFrame
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.sort_values('date').reset_index(drop=True)

        # Rename to descriptive column names for clarity downstream
        combined.rename(columns={
            't2m': 'temp_2m', 'd2m': 'dewpoint_2m', 'sp': 'surface_pressure',
            'u10': 'wind_u', 'v10': 'wind_v', 'tcwv': 'water_vapour', 'tp': 'precipitation'
        }, inplace=True)

        # ── Unit conversions ──────────────────────────────────────────────
        # ERA5 stores temperature in Kelvin; subtract 273.15 to get Celsius
        if 'temp_2m' in combined.columns:
            combined['temp_2m'] = combined['temp_2m'] - 273.15
        if 'dewpoint_2m' in combined.columns:
            combined['dewpoint_2m'] = combined['dewpoint_2m'] - 273.15
        # ERA5 stores pressure in Pascals; divide by 100 to get hPa (millibars)
        if 'surface_pressure' in combined.columns:
            combined['surface_pressure'] = combined['surface_pressure'] / 100
        # ERA5 stores precipitation in metres; multiply by 1000 to get millimetres
        if 'precipitation' in combined.columns:
            combined['precipitation'] = combined['precipitation'] * 1000

        # ── Derived variables ─────────────────────────────────────────────
        # Relative humidity from temperature and dewpoint using the Magnus formula
        if 'temp_2m' in combined.columns and 'dewpoint_2m' in combined.columns:
            combined['relative_humidity'] = self._calculate_relative_humidity(
                combined['temp_2m'], combined['dewpoint_2m']
            )
        # Wind speed from the two wind components (Pythagoras: √(u² + v²))
        if 'wind_u' in combined.columns and 'wind_v' in combined.columns:
            combined['wind_speed'] = np.sqrt(combined['wind_u']**2 + combined['wind_v']**2)

        # Save the combined daily CSV
        output_file = self.output_dir / 'era5_choma_daily.csv'
        combined.to_csv(output_file, index=False)

        print(f"\n✓ Converted data saved to: {output_file}")
        print(f"Shape: {combined.shape}")
        print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
        print(f"Columns: {list(combined.columns)}")

        return combined
    
    @staticmethod
    def _calculate_relative_humidity(temp_c, dewpoint_c):
        """
        Calculate relative humidity (%) from temperature and dewpoint temperature.

        Uses the Magnus formula approximation, which is accurate to within ~0.4%
        for temperatures between -40°C and +60°C.

        The formula computes the ratio of the actual vapour pressure (at the
        dewpoint) to the saturation vapour pressure (at the air temperature).
        When the dewpoint equals the air temperature, RH = 100% (saturated air).

        Parameters:
            temp_c     — air temperature in Celsius (pandas Series or array)
            dewpoint_c — dewpoint temperature in Celsius

        Returns relative humidity as a percentage (0–100).
        """
        return 100 * (np.exp((17.625 * dewpoint_c) / (243.04 + dewpoint_c)) /
                      np.exp((17.625 * temp_c) / (243.04 + temp_c)))


# When run directly, offer a simple menu to download, convert, or both
if __name__ == '__main__':
    import sys

    downloader = ERA5Downloader()

    # Support non-interactive mode for use in automated pipelines
    auto_mode = '--auto' in sys.argv

    if auto_mode:
        choice = '3'  # Download + convert
        print("ERA5 Data Downloader (auto mode: download + convert)")
        print("=" * 50)
    else:
        print("ERA5 Data Downloader")
        print("=" * 50)
        print("\nOptions:")
        print("1. Download ERA5 data (requires CDS API key)")
        print("2. Convert existing NetCDF to CSV")
        print("3. Both")
        choice = input("\nEnter choice (1/2/3): ").strip()

    if choice in ['1', '3']:
        try:
            downloader.download_era5_data()
        except Exception as e:
            print(f"\nError: {e}")
            print("\nMake sure you have:")
            print("1. Registered at https://cds.climate.copernicus.eu/")
            print("2. Added your API key to .env file")

    if choice in ['2', '3']:
        downloader.convert_to_csv()
