import requests

base = 'http://localhost:5000'

# Test predictions
r = requests.post(base + '/api/predict', json={}, timeout=15)
d = r.json()
print("=== FORECASTS ===")
for f in d.get('forecasts', []):
    h = f['horizon']
    dt = f['target_date']
    p = round(f['probability'] * 100, 1)
    v = 'Rain' if f['occurrence'] else 'No Rain'
    print(f"  {h:6s} {dt}  {p}%  -> {v}")

# Test other endpoints
for url in ['/api/future_predictions', '/api/historical', '/api/rainfall_summary']:
    r = requests.get(base + url, timeout=15)
    d = r.json()
    err = d.get('error', '')
    status = 'FAIL: ' + err if err else 'OK ' + str(list(d.keys())[:3])
    print(url, status)

print("\nAll checks complete.")
