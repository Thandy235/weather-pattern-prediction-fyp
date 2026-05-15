import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime, timedelta

df = pd.read_csv('data/processed/features_complete.csv', parse_dates=['date'])
today_doy = datetime.now().timetuple().tm_yday

df['day_of_year'] = df['date'].dt.dayofyear
doy_diff = (df['day_of_year'] - today_doy).abs()
doy_diff = doy_diff.apply(lambda x: min(x, 365 - x))
window = df[doy_diff <= 15]

drop_cols = ['date', 'rainfall', 'rainfall_occurrence', 'Station_Name',
             'season', 'data_source', 'qc_flag']
drop_cols += [c for c in df.columns if c.startswith('target_')]
feat_cols = [c for c in window.columns
             if c not in drop_cols and window[c].dtype in ['int64', 'float64']]
today_feat = window[feat_cols].median().to_frame().T.fillna(0)

print("Today DOY:", today_doy, "| Window rows:", len(window))
print()

today = datetime.now().date()
horizons = {'1day': 1, '7day': 7, '30day': 30, '90day': 90}

for horizon, days in horizons.items():
    feat_file = Path(f'models/feature_names_{horizon}.pkl')
    clf_file  = Path(f'models/rf_classifier_{horizon}.pkl')
    if not feat_file.exists() or not clf_file.exists():
        print(horizon, "model not found")
        continue
    feat_names = joblib.load(feat_file)
    clf = joblib.load(clf_file)
    row = today_feat.copy()
    for col in feat_names:
        if col not in row.columns:
            row[col] = 0.0
    row = row[feat_names].fillna(0)
    prob = clf.predict_proba(row)[0, 1]
    target_date = today + timedelta(days=days)
    verdict = "Rain" if prob > 0.5 else "No Rain"
    print(f"{horizon:6s} ({target_date}): {prob*100:.1f}%  -> {verdict}")
