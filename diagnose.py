"""
diagnose.py
Ablation study — find which of the 4 new features caused the regression
19 features (baseline) vs 23 features (tune.py) vs each feature added one at a time
"""

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
SEED = 42
N_FOLDS = 5

# ─────────────────────────────────────────────
# LOAD + ENGINEER (exact same as tune.py)
# ─────────────────────────────────────────────
train_raw = pd.read_csv("train.csv")
test_raw  = pd.read_csv("test.csv")

def base_engineer(df):
    df = df.copy()
    ts = df["timestamp"].str.split(":", expand=True).astype(int)
    df["hour"]         = ts[0]
    df["minute"]       = ts[1]
    df["time_minutes"] = df["hour"] * 60 + df["minute"]
    df["hour_sin"]     = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]     = np.cos(2 * np.pi * df["hour"] / 24)
    df["geo_prefix4"]  = df["geohash"].str[:4]
    df["geo_prefix5"]  = df["geohash"].str[:5]
    BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    def geohash_to_latlon(gh):
        lat_min, lat_max = -90.0, 90.0
        lon_min, lon_max = -180.0, 180.0
        is_lon = True
        for ch in gh:
            bits = BASE32.index(ch)
            for i in range(4, -1, -1):
                bit = (bits >> i) & 1
                if is_lon:
                    mid = (lon_min + lon_max) / 2
                    if bit: lon_min = mid
                    else:   lon_max = mid
                else:
                    mid = (lat_min + lat_max) / 2
                    if bit: lat_min = mid
                    else:   lat_max = mid
                is_lon = not is_lon
        return (lat_min + lat_max) / 2, (lon_min + lon_max) / 2
    latlons = df["geohash"].map(lambda g: geohash_to_latlon(g) if isinstance(g, str) else (np.nan, np.nan))
    df["lat"] = latlons.map(lambda x: x[0])
    df["lon"] = latlons.map(lambda x: x[1])
    df["temperature_missing"] = df["Temperature"].isna().astype(int)
    df["roadtype_missing"]    = df["RoadType"].isna().astype(int)
    df["weather_missing"]     = df["Weather"].isna().astype(int)
    df["RoadType"]    = df["RoadType"].fillna("Residential")
    df["Weather"]     = df["Weather"].fillna("Sunny")
    df["Temperature"] = df["Temperature"].fillna(df["Temperature"].median())
    df["large_vehicles_flag"] = (df["LargeVehicles"] == "Allowed").astype(int)
    df["has_landmark"]        = (df["Landmarks"] == "Yes").astype(int)
    df["road_type_ord"]       = df["RoadType"].map({"Residential": 0, "Street": 1, "Highway": 2}).fillna(0).astype(int)
    df["weather_severity"]    = df["Weather"].map({"Sunny": 0, "Foggy": 1, "Rainy": 2, "Snowy": 3}).fillna(0).astype(int)
    df["lanes_x_time"]        = df["NumberofLanes"] * df["time_minutes"]
    df["temp_x_weather"]      = df["Temperature"] * df["weather_severity"]
    return df

train = base_engineer(train_raw)
test  = base_engineer(test_raw)

TARGET = "demand"

def target_encode(train_df, test_df, col, target, n_folds=5, smoothing=10):
    global_mean = train_df[target].mean()
    oof_enc = np.zeros(len(train_df))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for tr_idx, val_idx in kf.split(train_df):
        fold_map = train_df.iloc[tr_idx].groupby(col)[target].agg(["mean","count"])
        fold_map["smooth"] = (fold_map["count"]*fold_map["mean"] + smoothing*global_mean) / (fold_map["count"]+smoothing)
        oof_enc[val_idx] = train_df.iloc[val_idx][col].map(fold_map["smooth"]).fillna(global_mean)
    full_map = train_df.groupby(col)[target].agg(["mean","count"])
    full_map["smooth"] = (full_map["count"]*full_map["mean"] + smoothing*global_mean) / (full_map["count"]+smoothing)
    return oof_enc, test_df[col].map(full_map["smooth"]).fillna(global_mean).values

for geo_col in ["geohash", "geo_prefix4", "geo_prefix5"]:
    train[f"{geo_col}_te"], test[f"{geo_col}_te"] = target_encode(train, test, geo_col, TARGET)

train["geo_te_x_time"] = train["geohash_te"] * train["time_minutes"]
test["geo_te_x_time"]  = test["geohash_te"]  * test["time_minutes"]

# ─────────────────────────────────────────────
# THE 4 NEW FEATURES (added in tune.py)
# ─────────────────────────────────────────────
# 1. geo_te_x_time   (geohash_te × time_minutes)
# 2. temperature_missing
# 3. roadtype_missing
# 4. weather_missing
# (geo_te_x_time was the actual new addition; the 3 missing flags were also new)

BASELINE_19 = [
    "day", "time_minutes", "hour", "minute",
    "hour_sin", "hour_cos",
    "NumberofLanes", "large_vehicles_flag", "has_landmark",
    "road_type_ord", "weather_severity", "Temperature",
    "lanes_x_time", "temp_x_weather",
    "lat", "lon",
    "geohash_te", "geo_prefix4_te", "geo_prefix5_te",
]

NEW_FEATURES = {
    "geo_te_x_time":       "geohash_te × time_minutes interaction",
    "temperature_missing": "missing flag: Temperature",
    "roadtype_missing":    "missing flag: RoadType",
    "weather_missing":     "missing flag: Weather",
}

# ─────────────────────────────────────────────
# CV HELPER
# ─────────────────────────────────────────────
PARAMS = dict(
    n_estimators=1000, learning_rate=0.05, num_leaves=127,
    max_depth=-1, min_child_samples=20, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
    n_jobs=-1, random_state=SEED, verbose=-1,
)

def cv_rmse(features):
    X = train[features]
    y = np.log1p(train[TARGET])
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    for tr_idx, val_idx in kf.split(X):
        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof[val_idx] = m.predict(X.iloc[val_idx])
    return np.sqrt(mean_squared_error(np.expm1(y), np.expm1(oof)))

# ─────────────────────────────────────────────
# ABLATION
# ─────────────────────────────────────────────
print("="*58)
print("ABLATION STUDY")
print("="*58)
print(f"{'Experiment':<40} {'RMSE':>8}  {'vs baseline':>12}")
print("-"*58)

baseline_rmse = cv_rmse(BASELINE_19)
print(f"{'Baseline (19 features)':<40} {baseline_rmse:.5f}  {'—':>12}")

results = {"baseline": baseline_rmse}

# Add each new feature one at a time
for feat, desc in NEW_FEATURES.items():
    rmse = cv_rmse(BASELINE_19 + [feat])
    delta = rmse - baseline_rmse
    sign  = "▲ WORSE" if delta > 0 else "▼ better"
    print(f"{'+ ' + feat:<40} {rmse:.5f}  {delta:+.5f} {sign}")
    results[feat] = rmse

# Add all 4 together (= tune.py's 23 features)
all23_rmse = cv_rmse(BASELINE_19 + list(NEW_FEATURES.keys()))
delta = all23_rmse - baseline_rmse
sign  = "▲ WORSE" if delta > 0 else "▼ better"
print(f"{'All 4 new features (23 total)':<40} {all23_rmse:.5f}  {delta:+.5f} {sign}")

# Add only the features that helped (delta < 0)
good = [f for f in NEW_FEATURES if results[f] < baseline_rmse]
if good:
    good_rmse = cv_rmse(BASELINE_19 + good)
    delta = good_rmse - baseline_rmse
    print(f"\n{'Baseline + only good features':<40} {good_rmse:.5f}  {delta:+.5f}")
    print(f"  Good features: {good}")
else:
    print(f"\nNo individual new feature improved the baseline.")
    print(f"All 4 features add noise. Stick with the 19-feature baseline.")

print("="*58)