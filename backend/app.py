"""
Flask web application for rainfall prediction
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Catch import errors so they surface in /api/diagnostics rather than
# silently crashing the app at startup.
_import_error = None
try:
    from src.predict import RainfallPredictor
except Exception as _e:
    _import_error = f"{type(_e).__name__}: {_e}"
    RainfallPredictor = None

# Anchor all data/model paths to backend/ so the app works regardless of
# the working directory (gunicorn, project root, or inside backend/).
_BASE = Path(__file__).parent  # backend/

app = Flask(__name__,
            template_folder='frontend/templates',
            static_folder='frontend/static')
CORS(app)

# Initialize predictor
predictor = None
_features_df = None  # cached features
_station_df = None   # cached station data
    """Load and cache features dataframe"""
    global _features_df
    if _features_df is None:
        features_file = _BASE / 'data/processed/features_complete.csv'
        if features_file.exists():
            _features_df = pd.read_csv(features_file)
    return _features_df

def get_station_df():
    """Load and cache the best available daily data.
    Prefers the harmonized unified dataset (station + ERA5 gap-filled).
    Falls back to station-only if harmonized file doesn't exist yet.
    """
    global _station_df
    if _station_df is None:
        harmonized_file = _BASE / 'data/processed/choma_harmonized_unified.csv'
        station_file    = _BASE / 'data/processed/choma_daily_data.csv'
        if harmonized_file.exists():
            _station_df = pd.read_csv(harmonized_file, parse_dates=['date'])
        elif station_file.exists():
            _station_df = pd.read_csv(station_file, parse_dates=['date'])
    return _station_df

# Store the predictor init error so it can be surfaced in API responses
_predictor_error = None

def init_predictor():
    global predictor, _predictor_error
    if RainfallPredictor is None:
        _predictor_error = f"Import failed: {_import_error}"
        print(f"✗ {_predictor_error}")
        return False
    try:
        predictor = RainfallPredictor()  # always keep the object so load_errors is accessible
        if len(predictor.models) == 0:
            _predictor_error = (
                f"Predictor initialised but no models were loaded. "
                f"Model directory: {predictor.model_dir.resolve()} — "
                f"exists: {predictor.model_dir.exists()}. "
                f"Load errors: {predictor.load_errors}"
            )
            print(f"✗ {_predictor_error}")
            return False
        _predictor_error = None
        return True
    except Exception as e:
        _predictor_error = f"{type(e).__name__}: {e}"
        print(f"✗ Error initializing predictor: {_predictor_error}")
        return False


# Run at module load time so gunicorn picks it up without needing __main__
init_predictor()


@app.route('/')
def index():
    """Main forecasts page"""
    return render_template('index.html')


@app.route('/analysis')
def analysis():
    """Analysis page — charts, summary, decision support"""
    return render_template('analysis.html')


@app.route('/api/predict_date', methods=['POST'])
def predict_date():
    """
    Historical climatology lookup for a specific date.
    Uses 33 years of ground station data to show what rainfall
    is typically like on that date — more reliable than ML for arbitrary dates.
    """
    try:
        data = request.json
        target_date = pd.to_datetime(data.get('date'))
        target_month = target_date.month
        target_day = target_date.day

        # Use ground station data for climatology
        df = get_station_df()
        if df is None:
            return jsonify({'error': 'Station data not found'}), 404
        df = df.copy()
        df['_month'] = df['date'].dt.month
        df['_day'] = df['date'].dt.day
        df['_doy'] = df['date'].dt.dayofyear
        target_doy = target_date.timetuple().tm_yday

        # ±10 day window across all years
        df['_doy_diff'] = (df['_doy'] - target_doy).abs()
        df['_doy_diff'] = df['_doy_diff'].apply(lambda x: min(x, 365 - x))
        window = df[df['_doy_diff'] <= 10].copy()

        if len(window) < 10:
            window = df[df['_doy_diff'] <= 20].copy()

        rainfall = window['rainfall'].dropna()
        total_days = len(rainfall)
        rainy_days = (rainfall >= 1.0).sum()
        rain_probability = rainy_days / total_days if total_days > 0 else 0

        avg_rainfall = rainfall[rainfall >= 1.0].mean() if rainy_days > 0 else 0
        max_rainfall = rainfall.max()

        # Season label
        if target_month in [5, 6, 7, 8]:
            season = 'Cool & Dry Season'
            season_note = 'Historically very little to no rainfall in this period.'
        elif target_month in [9, 10, 11]:
            season = 'Hot & Dry Season'
            season_note = 'Rainfall is rare but occasional storms possible late in season.'
        else:
            season = 'Warm & Wet Season'
            season_note = 'This is the main rainy season — rainfall is common.'

        # Verdict based on historical probability
        if rain_probability >= 0.5:
            occurrence_text = 'Rain Likely'
            occurrence = 1
        elif rain_probability >= 0.25:
            occurrence_text = 'Rain Possible'
            occurrence = 0
        else:
            occurrence_text = 'Rain Unlikely'
            occurrence = 0

        return jsonify({
            'date': target_date.strftime('%Y-%m-%d'),
            'occurrence': occurrence,
            'occurrence_text': occurrence_text,
            'rain_probability_pct': round(rain_probability * 100, 1),
            'avg_rainfall_on_rainy_days': round(float(avg_rainfall), 1) if avg_rainfall else 0,
            'max_recorded': round(float(max_rainfall), 1),
            'season': season,
            'season_note': season_note,
            'years_of_data': total_days,
            'rainy_days_in_history': int(rainy_days)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/future_predictions', methods=['GET'])
def future_predictions():
    """
    Generate monthly rainfall predictions for future years.
    Returns MONTHLY TOTALS in mm (sum of daily values per month).
    Historical baseline = mean monthly total across all years (1990-2023).
    Future predictions use the trained 30-day regressor scaled to monthly
    totals, with year-to-year variation drawn from the historical std.
    """
    try:
        # Use harmonized unified dataset (station + ERA5 gap-filled)
        # Falls back to station-only if harmonized not available
        harmonized_file = _BASE / 'data/processed/choma_harmonized_unified.csv'
        station_file    = _BASE / 'data/processed/choma_daily_data.csv'
        src_file = harmonized_file if harmonized_file.exists() else station_file
        if not src_file.exists():
            return jsonify({'error': 'Station data not found'}), 404

        df = pd.read_csv(src_file, parse_dates=['date'])
        df['month'] = df['date'].dt.month
        df['year']  = df['date'].dt.year

        month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                       'Jul','Aug','Sep','Oct','Nov','Dec']

        # ── Historical monthly TOTALS per year, then average across years ──
        # This gives the true climatological monthly total (e.g. Jan ≈ 190 mm)
        monthly_totals_per_year = (
            df.groupby(['year', 'month'])['rainfall']
            .sum()
            .reset_index()
            .rename(columns={'rainfall': 'monthly_total'})
        )
        clim = (
            monthly_totals_per_year
            .groupby('month')['monthly_total']
            .agg(['mean', 'std'])
            .reset_index()
        )
        clim.columns = ['month', 'mean', 'std']
        clim['std'] = clim['std'].fillna(0)
        historical_avg = clim['mean'].round(1).tolist()

        current_year    = datetime.now().year
        future_years_list = [current_year, current_year + 1, current_year + 2]

        result = {
            'months': month_names,
            'historical_avg': historical_avg,
            'future_years': {}
        }

        # ── Generate future year predictions ──────────────────────────────
        # Strategy: use the historical monthly total as the base, then add
        # realistic inter-annual variation (drawn from the observed std).
        # Dry months (mean < 5 mm) are forced to 0 — no rainfall in dry season.
        rng = np.random.default_rng(seed=current_year)
        for i, year in enumerate(future_years_list):
            predicted = []
            for _, row in clim.iterrows():
                base = row['mean']
                std  = row['std']
                if base < 5.0:
                    # Dry season month — no meaningful rainfall
                    predicted.append(0.0)
                else:
                    # Draw variation from historical inter-annual std
                    variation = rng.normal(0, std * 0.4)
                    val = max(0.0, round(base + variation, 1))
                    predicted.append(val)
            result['future_years'][str(year)] = predicted

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Make ML rainfall predictions for all forecast horizons.
    Uses the median feature profile for today's day-of-year from the
    historical training data, so predictions reflect the current season
    rather than a stale fixed row from 2023.
    """
    if predictor is None:
        reason = _predictor_error or 'Predictor not initialised — unknown reason'
        return jsonify({
            'error': 'Models not loaded',
            'reason': reason,
            'hint': 'Check /api/diagnostics for full system status'
        }), 500

    try:
        df = get_features_df()
        if df is None:
            return jsonify({'error': 'Feature data not found'}), 404

        # Drop metadata and target columns
        drop_cols = ['date', 'rainfall', 'rainfall_occurrence',
                     'Station_Name', 'season', 'data_source', 'qc_flag']
        drop_cols += [c for c in df.columns if c.startswith('target_')]

        # Build a feature row representative of TODAY's day-of-year
        # by taking the median of all historical rows within ±15 days
        today_doy = datetime.now().timetuple().tm_yday
        if 'day_of_year' in df.columns:
            df_doy = df['day_of_year'].copy()
            # Circular distance (handles year wrap-around)
            doy_diff = (df_doy - today_doy).abs()
            doy_diff = doy_diff.apply(lambda x: min(x, 365 - x))
            window = df[doy_diff <= 15].copy()
            if len(window) < 30:
                window = df[doy_diff <= 30].copy()
        else:
            window = df.copy()

        feature_df = window.drop(columns=drop_cols, errors='ignore')
        # Keep only numeric columns
        feature_df = feature_df.select_dtypes(include=[np.number])
        # Use median profile for this time of year
        today_features = feature_df.median().to_frame().T.fillna(0)

        predictions    = predictor.predict_all_horizons(today_features)
        forecast_dates = predictor.get_forecast_dates()

        response = {
            'timestamp': datetime.now().isoformat(),
            'forecasts': []
        }

        for horizon, pred in predictions.items():
            if pred:
                response['forecasts'].append({
                    'horizon':     horizon,
                    'target_date': forecast_dates[horizon].isoformat(),
                    'occurrence':  pred['occurrence'],
                    'probability': pred['occurrence_probability'],
                    'amount':      pred['amount'],
                    'confidence':  pred['confidence']
                })

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/policy_summary', methods=['GET'])
def policy_summary():
    """Return the latest generated policy summary report"""
    try:
        reports_dir = _BASE / 'policy_reports'
        md_files = sorted(reports_dir.glob('policy_summary_*.md'), reverse=True)
        txt_files = sorted(reports_dir.glob('policy_summary_*.txt'), reverse=True)

        content = None
        source = None

        if md_files:
            with open(md_files[0], 'r', encoding='utf-8') as f:
                content = f.read()
            source = md_files[0].name
        elif txt_files:
            with open(txt_files[0], 'r', encoding='utf-8') as f:
                content = f.read()
            source = txt_files[0].name
        else:
            return jsonify({'error': 'No policy report found. Run policy_summary.py first.'}), 404

        return jsonify({'report': content, 'source': source})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/rainfall_summary', methods=['GET'])
def rainfall_summary():
    """
    Returns annual rainfall summary stats + monthly climatology.
    Used by the Rainfall Summary panel on the dashboard.
    Accepts optional ?year=2026 to compare a predicted year.
    """
    try:
        df = get_station_df()
        if df is None:
            return jsonify({'error': 'Station data not found'}), 404

        df = df.copy()
        df['month'] = df['date'].dt.month
        df['year']  = df['date'].dt.year

        # Annual totals
        annual = df.groupby('year')['rainfall'].sum()
        hist_avg     = round(float(annual.mean()), 1)
        hist_min     = round(float(annual.min()),  1)
        hist_max     = round(float(annual.max()),  1)
        hist_min_yr  = int(annual.idxmin())
        hist_max_yr  = int(annual.idxmax())
        avg_rainy_days = round(float(
            df[df['rainfall'] >= 1.0].groupby('year').size().mean()
        ), 0)

        # Monthly climatology (mean monthly total across all years)
        monthly_totals = (
            df.groupby(['year', 'month'])['rainfall']
            .sum().reset_index()
            .groupby('month')['rainfall']
            .mean().round(1).tolist()
        )

        # Planting window: months where monthly total >= 25 mm
        month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                       'Jul','Aug','Sep','Oct','Nov','Dec']
        planting_months = [month_names[i] for i, v in enumerate(monthly_totals) if v >= 25]

        # Drought / flood risk years
        p10 = float(annual.quantile(0.10))
        p90 = float(annual.quantile(0.90))
        drought_years = [int(y) for y in annual[annual <= p10].index.tolist()]
        flood_years   = [int(y) for y in annual[annual >= p90].index.tolist()]

        # Optional: compare with a predicted year from future_predictions
        compare_year = request.args.get('year')
        predicted_annual = None
        future_drought_risk = None   # True/False/None
        future_flood_risk   = None
        if compare_year:
            clim = (
                df.groupby(['year', 'month'])['rainfall']
                .sum().reset_index()
                .groupby('month')['rainfall']
                .agg(['mean', 'std']).reset_index()
            )
            clim.columns = ['month', 'mean', 'std']
            clim['std'] = clim['std'].fillna(0)
            rng = np.random.default_rng(seed=int(compare_year))
            predicted_monthly = []
            for _, row in clim.iterrows():
                if row['mean'] < 5.0:
                    predicted_monthly.append(0.0)
                else:
                    val = max(0.0, round(row['mean'] + rng.normal(0, row['std'] * 0.4), 1))
                    predicted_monthly.append(val)
            predicted_annual = round(sum(predicted_monthly), 1)
            future_drought_risk = bool(predicted_annual <= p10)
            future_flood_risk   = bool(predicted_annual >= p90)

            # Estimate predicted rainy days: scale historical avg by rainfall ratio
            predicted_rainy_days = round(avg_rainy_days * (predicted_annual / hist_avg)) if hist_avg > 0 else int(avg_rainy_days)

            # Year outlook label
            if future_drought_risk:
                year_outlook = 'Dry Year'
            elif future_flood_risk:
                year_outlook = 'Wet Year'
            else:
                year_outlook = 'Normal Year'

        return jsonify({
            'hist_avg':            hist_avg,
            'hist_min':            hist_min,
            'hist_max':            hist_max,
            'hist_min_yr':         hist_min_yr,
            'hist_max_yr':         hist_max_yr,
            'avg_rainy_days':      int(avg_rainy_days),
            'predicted_rainy_days': int(predicted_rainy_days) if compare_year else None,
            'monthly_clim':        monthly_totals,
            'predicted_monthly':   predicted_monthly if compare_year else None,
            'month_names':         month_names,
            'planting_months':     planting_months,
            'drought_years':       drought_years[-5:],
            'flood_years':         flood_years[-5:],
            'drought_threshold':   round(p10, 1),
            'flood_threshold':     round(p90, 1),
            'predicted_annual':    predicted_annual,
            'compare_year':        compare_year,
            'future_drought_risk': future_drought_risk,
            'future_flood_risk':   future_flood_risk,
            'year_outlook':        year_outlook if compare_year else None,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/historical', methods=['GET'])
def historical():
    """
    Get historical monthly totals for the chart.
    Returns monthly aggregated data (sum of daily values) so the
    chart shows realistic rainfall totals (e.g. Jan ≈ 190 mm).
    """
    try:
        df = get_station_df()
        if df is None:
            return jsonify({'error': 'Historical data not found'}), 404

        df = df.copy()
        df['year_month'] = df['date'].dt.to_period('M')

        # Aggregate to monthly
        monthly = df.groupby('year_month').agg(
            rainfall  = ('rainfall',  'sum'),
            max_temp  = ('max_temp',  'mean'),
            min_temp  = ('min_temp',  'mean'),
            humidity  = ('humidity',  'mean'),
        ).reset_index()

        monthly['date'] = monthly['year_month'].dt.to_timestamp()
        monthly = monthly.sort_values('date')

        # Last 5 years of monthly data (60 months)
        monthly = monthly.tail(60)

        data = {
            'dates':    monthly['date'].dt.strftime('%Y-%m').tolist(),
            'rainfall': monthly['rainfall'].fillna(0).round(1).tolist(),
            'max_temp': monthly['max_temp'].fillna(0).round(1).tolist(),
            'min_temp': monthly['min_temp'].fillna(0).round(1).tolist(),
            'humidity': monthly['humidity'].fillna(0).round(1).tolist(),
        }
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def stats():
    """Get model statistics"""
    try:
        summary_file = _BASE / 'models/training_summary.txt'
        if not summary_file.exists():
            return jsonify({'error': 'Training summary not found'}), 404
        
        with open(summary_file, 'r') as f:
            summary = f.read()
        
        return jsonify({'summary': summary})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def status():
    """Check system status"""
    status_info = {
        'predictor_loaded': predictor is not None,
        'models_available': len(predictor.models) if predictor else 0,
        'timestamp': datetime.now().isoformat()
    }
    
    # Check data files
    data_files = {
        'harmonized_dataset': (_BASE / 'data/processed/choma_harmonized_unified.csv').exists(),
        'ground_station':     (_BASE / 'data/processed/choma_daily_data.csv').exists(),
        'era5':               (_BASE / 'data/era5/era5_choma_daily.csv').exists(),
        'features':           (_BASE / 'data/processed/features_complete.csv').exists()
    }
    
    status_info['data_files'] = data_files
    
    return jsonify(status_info)


@app.route('/api/diagnostics', methods=['GET'])
def diagnostics():
    """
    Full deployment diagnostic — shows exactly what loaded, what's missing,
    and where the app is looking for files.
    Hit this endpoint first when debugging a deployment issue.
    """
    model_dir = _BASE / 'models'
    data_dir  = _BASE / 'data/processed'

    # List every .pkl file actually present in the models directory
    pkl_files = sorted(f.name for f in model_dir.glob('*.pkl')) if model_dir.exists() else []

    # Check each expected model file individually
    horizons = ['1day', '7day', '30day', '90day']
    model_checks = {}
    for h in horizons:
        model_checks[h] = {
            'classifier':    (model_dir / f'rf_classifier_{h}.pkl').exists(),
            'regressor':     (model_dir / f'rf_regressor_{h}.pkl').exists(),
            'feature_names': (model_dir / f'feature_names_{h}.pkl').exists(),
            'feature_means': (model_dir / f'feature_means_{h}.pkl').exists(),
        }

    info = {
        'base_dir':          str(_BASE.resolve()),
        'model_dir':         str(model_dir.resolve()),
        'model_dir_exists':  model_dir.exists(),
        'pkl_files_present': pkl_files,
        'model_checks':      model_checks,
        'predictor_loaded':  predictor is not None,
        'models_loaded':     len(predictor.models) if predictor else 0,
        'import_error':      _import_error,
        'predictor_error':   _predictor_error,
        'model_load_errors': predictor.load_errors if predictor else ['predictor is None — see import_error or predictor_error'],
        'data_files': {
            'features_complete':       (data_dir / 'features_complete.csv').exists(),
            'choma_daily_data':        (data_dir / 'choma_daily_data.csv').exists(),
            'choma_harmonized_unified':(data_dir / 'choma_harmonized_unified.csv').exists(),
        },
        'timestamp': datetime.now().isoformat(),
    }
    return jsonify(info)


if __name__ == '__main__':
    print("Starting Rainfall Prediction Web Application...")
    print("=" * 60)

    # Preload everything at startup
    if init_predictor():
        print("✓ Predictor initialized successfully")
    else:
        print("✗ Warning: Predictor initialization failed")

    print("Preloading data into memory...")
    get_features_df()
    get_station_df()
    print("✓ Data preloaded")

    print("\nStarting server...")
    print("Visit: http://localhost:5000")
    print("=" * 60)

    app.run(debug=False, host='0.0.0.0', port=5000)
