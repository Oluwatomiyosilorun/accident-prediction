"""
generate_data.py
Generates a synthetic traffic accident dataset representative of
Nigerian urban road conditions (Lagos, Abuja, Port Harcourt).
Mimics the feature structure of the Kaggle US Accidents dataset
combined with SUMO simulation parameters.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

random.seed(42)
np.random.seed(42)

N = 50000  # total records

# ── Temporal features ────────────────────────────────────────────────────────
start_date = datetime(2019, 1, 1)
dates = [start_date + timedelta(hours=random.randint(0, 5 * 365 * 24)) for _ in range(N)]
hours   = [d.hour for d in dates]
days    = [d.weekday() for d in dates]   # 0=Mon, 6=Sun
months  = [d.month for d in dates]

# ── Weather features (tropical Nigerian conditions) ──────────────────────────
# Rainy season: April–October; Dry season: Nov–March
def rain_prob(month):
    return 0.55 if 4 <= month <= 10 else 0.1

precipitation = np.array([
    np.random.exponential(5) if random.random() < rain_prob(m) else 0.0
    for m in months
])
visibility    = np.clip(10 - precipitation * 0.8 + np.random.normal(0, 0.5, N), 0.5, 10)
temperature   = np.random.normal(30, 4, N)          # Nigerian avg ~30°C
humidity      = np.clip(60 + precipitation * 1.2 + np.random.normal(0, 5, N), 40, 100)
wind_speed    = np.abs(np.random.normal(8, 4, N))

# ── Road features ────────────────────────────────────────────────────────────
has_junction        = np.random.choice([0, 1], N, p=[0.6, 0.4])
has_traffic_signal  = np.random.choice([0, 1], N, p=[0.55, 0.45])
has_crossing        = np.random.choice([0, 1], N, p=[0.7, 0.3])
has_bump            = np.random.choice([0, 1], N, p=[0.65, 0.35])
speed_limit         = np.random.choice([30, 50, 60, 80, 100], N, p=[0.25, 0.35, 0.2, 0.15, 0.05])

# ── Traffic density (vehicles/km) — SUMO-calibrated ─────────────────────────
def traffic_density(hour, day):
    base = 30
    if 7 <= hour <= 9:    base = 90   # morning peak
    elif 12 <= hour <= 14: base = 55  # midday
    elif 17 <= hour <= 19: base = 95  # evening peak
    elif 22 <= hour or hour <= 5: base = 15  # night
    if day >= 5: base *= 0.7  # weekend
    return max(5, np.random.normal(base, base * 0.2))

traffic_density_arr = np.array([traffic_density(h, d) for h, d in zip(hours, days)])

# ── Weather condition category ────────────────────────────────────────────────
def weather_cat(prec, vis):
    if prec > 10:  return 2   # Heavy Rain
    elif prec > 2: return 1   # Light Rain
    else:          return 0   # Clear

weather_condition = np.array([weather_cat(p, v) for p, v in zip(precipitation, visibility)])

# ── Target variable: Accident Severity (0=None/Minor, 1=Moderate, 2=Severe) ──
def compute_severity(hour, prec, vis, junc, sig, spd, dens, bump):
    risk = 0.0
    # Time risk
    if 23 <= hour or hour <= 4:  risk += 0.35
    elif 7 <= hour <= 9:         risk += 0.15
    elif 17 <= hour <= 19:       risk += 0.20
    # Weather risk
    if prec > 10:    risk += 0.30
    elif prec > 2:   risk += 0.15
    if vis < 3:      risk += 0.20
    # Road risk
    if junc:         risk += 0.10
    if not sig:      risk += 0.10
    if bump:         risk += 0.05
    if spd >= 80:    risk += 0.15
    # Traffic density
    if dens > 80:    risk += 0.10
    # Add noise
    risk += np.random.uniform(-0.1, 0.1)
    risk = max(0, min(1, risk))
    if risk < 0.25:  return 0
    elif risk < 0.55: return 1
    else:             return 2

severity = np.array([
    compute_severity(hours[i], precipitation[i], visibility[i],
                     has_junction[i], has_traffic_signal[i],
                     speed_limit[i], traffic_density_arr[i], has_bump[i])
    for i in range(N)
])

# ── Assemble DataFrame ────────────────────────────────────────────────────────
df = pd.DataFrame({
    'hour':             hours,
    'day_of_week':      days,
    'month':            months,
    'temperature':      np.round(temperature, 1),
    'humidity':         np.round(humidity, 1),
    'precipitation':    np.round(precipitation, 2),
    'visibility':       np.round(visibility, 2),
    'wind_speed':       np.round(wind_speed, 1),
    'weather_condition': weather_condition,
    'traffic_density':  np.round(traffic_density_arr, 1),
    'speed_limit':      speed_limit,
    'has_junction':     has_junction,
    'has_traffic_signal': has_traffic_signal,
    'has_crossing':     has_crossing,
    'has_bump':         has_bump,
    'severity':         severity
})

out_path = os.path.join(os.path.dirname(__file__), 'data', 'traffic_data.csv')
df.to_csv(out_path, index=False)

print(f"Dataset generated: {N} records, {df.shape[1]} features")
print(f"Severity distribution:\n{df['severity'].value_counts().sort_index()}")
print(f"Saved to: {out_path}")
