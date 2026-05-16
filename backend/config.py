"""
config.py — Central configuration file for the Choma Rainfall Prediction project.

This file is the single place where you change settings that affect the whole
project: file paths, station coordinates, model hyperparameters, forecast
horizons, and so on.  Think of it as the project's control panel — tweak
values here and every other script picks them up automatically.
"""

from pathlib import Path

# Anchor all paths to the backend/ directory so they resolve correctly
# regardless of the working directory when the process is launched.
_BASE = Path(__file__).parent  # backend/

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# Where the raw Choma station spreadsheets live (Excel/CSV files from ZEMA)
DATA_DIR = _BASE / 'choma station data'

# Where cleaned and processed CSV files are written to
PROCESSED_DIR = _BASE / 'data/processed'

# Where ERA5 NetCDF and CSV files are stored
ERA5_DIR = _BASE / 'data/era5'

# Where trained model files (.pkl) are saved
MODEL_DIR = _BASE / 'models'

# GPS coordinates of the Choma weather station (decimal degrees)
# Negative latitude = Southern Hemisphere
STATION_LAT = -16.8
STATION_LON = 26.9

# The full historical record we work with (inclusive on both ends)
START_YEAR = 1990
END_YEAR = 2023

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

# How many days back to look when creating "lag" features.
# For example, lag=7 means "what was the rainfall 7 days ago?"
# These help the model learn patterns like "it rained last week, so..."
LAG_PERIODS = [1, 3, 7, 14, 30]

# Window sizes (in days) for rolling statistics like 7-day average rainfall.
# Larger windows capture longer-term trends; smaller windows capture recent changes.
ROLLING_WINDOWS = [7, 14, 30]

# A day counts as "rainy" only if rainfall is at least this many mm.
# 1 mm is a standard meteorological threshold — anything less is basically trace.
RAINFALL_THRESHOLD = 1.0

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# The four forecast horizons we train separate models for.
# Each key is a human-readable label; the value is how many days ahead we predict.
FORECAST_HORIZONS = {
    '1day': 1,    # Tomorrow
    '7day': 7,    # One week out
    '30day': 30,  # One month out
    '90day': 90   # One season out
}

# Hyperparameters for the Random Forest *classifier* (predicts rain vs no-rain).
# These were chosen to balance accuracy and overfitting:
#   - n_estimators=200: 200 decision trees — more trees = more stable predictions
#   - max_depth=20: trees can't grow too deep, which prevents memorising the training data
#   - min_samples_split/leaf: a node must have enough samples before it can split
#   - max_features='sqrt': each tree only sees a random subset of features (reduces correlation between trees)
#   - class_weight='balanced': automatically upweights the minority class (rainy days)
#     because dry days outnumber rainy days and we don't want the model to just predict "no rain" always
CLASSIFIER_PARAMS = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1,          # Use all available CPU cores
    'class_weight': 'balanced'
}

# Hyperparameters for the Random Forest *regressor* (predicts how many mm of rain).
# Same structure as the classifier but without class_weight (regression doesn't need it).
REGRESSOR_PARAMS = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1
}

# How we split the dataset into training, validation, and test sets.
# We keep temporal order — training is the oldest data, test is the most recent.
# This mimics real-world use: you train on the past and predict the future.
TRAIN_RATIO = 0.70       # 70% of data for training
VALIDATION_RATIO = 0.15  # 15% for tuning / early stopping decisions
TEST_RATIO = 0.15        # 15% held out for final unbiased evaluation

# ============================================================================
# ERA5 CONFIGURATION
# ============================================================================

# The atmospheric variables we download from the Copernicus Climate Data Store.
# These are the inputs that help the model understand the current weather state.
ERA5_VARIABLES = [
    '2m_temperature',              # Air temperature 2 metres above ground (°C after conversion)
    '2m_dewpoint_temperature',     # Dewpoint — used to calculate relative humidity
    'surface_pressure',            # Atmospheric pressure at the surface (hPa after conversion)
    '10m_u_component_of_wind',     # East-west wind component at 10 m height
    '10m_v_component_of_wind',     # North-south wind component at 10 m height
    'total_column_water_vapour',   # Total moisture in the atmosphere above the point (kg/m²)
    'total_precipitation',         # Accumulated precipitation (mm after conversion)
]

# ERA5 provides data at these four times each day (UTC).
# We average them to get a single daily value.
ERA5_TIMES = ['00:00', '06:00', '12:00', '18:00']

# How many degrees around the station to include in the ERA5 download.
# 0.5° ≈ 55 km — gives a small spatial buffer around the station point.
ERA5_AREA_BUFFER = 0.5

# ============================================================================
# WEB APPLICATION
# ============================================================================

# Flask server settings.
# '0.0.0.0' means "listen on all network interfaces" — accessible from other machines on the LAN.
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True  # Set to False in production to avoid exposing debug info

# How many days of historical data to show in the dashboard's time-series chart
HISTORICAL_DAYS = 365

# ============================================================================
# VISUALIZATION
# ============================================================================

# Seaborn plot style — 'whitegrid' gives clean charts with a white background and grid lines
PLOT_STYLE = 'whitegrid'

# Default figure size in inches (width × height)
DEFAULT_FIGSIZE = (12, 6)

# Colour palette for multi-series charts
COLOR_PALETTE = 'Set2'

# Resolution for saved figures — 150 DPI is a good balance of quality and file size
FIGURE_DPI = 150

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================

# We only train the rainfall *amount* regressor if there are at least this many
# rainy days in the training set.  Too few rainy days → unreliable regression.
MIN_RAINY_DAYS = 100

# Number of folds for cross-validation (used in some evaluation scripts)
CV_FOLDS = 5

# When plotting feature importances, only show the top N most important features
# to keep the chart readable
TOP_N_FEATURES = 20

# Fixed random seed so results are reproducible across runs
RANDOM_SEED = 42

# ============================================================================
# ZAMBIAN SEASONS
# ============================================================================

# Zambia has four recognisable seasons.  We encode these as features so the
# model can learn season-specific rainfall patterns.
SEASONS = {
    'rainy_hot': [12, 1, 2],      # Hot rainy season (Dec-Feb) — peak rainfall
    'dry_cool': [3, 4, 5],         # Cool dry season (Mar-May) — rainfall tailing off
    'dry_cold': [6, 7, 8],         # Cold dry season (Jun-Aug) — almost no rain
    'hot_dry': [9, 10, 11]         # Hot dry season (Sep-Nov) — building up to rains
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_season(month):
    """
    Given a month number (1–12), return the name of the Zambian season it belongs to.
    For example, get_season(1) returns 'rainy_hot' because January is in the hot rainy season.
    """
    for season, months in SEASONS.items():
        if month in months:
            return season
    return 'unknown'


def create_directories():
    """
    Make sure all the output folders exist before we try to write files into them.
    'parents=True' creates any missing parent folders too.
    'exist_ok=True' means it won't crash if the folder already exists.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ERA5_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """
    Run a quick sanity check on the configuration values.
    Prints any problems it finds and returns True if everything looks fine,
    False if there are errors that need fixing before the pipeline can run.
    """
    errors = []
    
    # Check data directory exists
    if not DATA_DIR.exists():
        errors.append(f"Data directory not found: {DATA_DIR}")
    
    # Check year range — start must come before end
    if START_YEAR >= END_YEAR:
        errors.append(f"START_YEAR ({START_YEAR}) must be less than END_YEAR ({END_YEAR})")
    
    # We need at least one forecast horizon defined
    if not FORECAST_HORIZONS:
        errors.append("FORECAST_HORIZONS cannot be empty")
    
    # We need at least one lag period for feature engineering
    if not LAG_PERIODS:
        errors.append("LAG_PERIODS cannot be empty")
    
    if errors:
        print("Configuration Errors:")
        for error in errors:
            print(f"  ✗ {error}")
        return False
    
    print("✓ Configuration validated successfully")
    return True


# When you run this file directly (python config.py), it prints a summary
# of the current settings and runs the validation check.
if __name__ == '__main__':
    print("Rainfall Prediction - Configuration")
    print("=" * 60)
    print(f"\nData Directory: {DATA_DIR}")
    print(f"Station Location: {STATION_LAT}°S, {STATION_LON}°E")
    print(f"Time Range: {START_YEAR}-{END_YEAR}")
    print(f"\nForecast Horizons: {list(FORECAST_HORIZONS.keys())}")
    print(f"Lag Periods: {LAG_PERIODS}")
    print(f"Rolling Windows: {ROLLING_WINDOWS}")
    print(f"\nClassifier: Random Forest with {CLASSIFIER_PARAMS['n_estimators']} trees")
    print(f"Regressor: Random Forest with {REGRESSOR_PARAMS['n_estimators']} trees")
    print("\n" + "=" * 60)
    
    validate_config()
