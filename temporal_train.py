"""
train_temporal.py
=================
Reframes the problem as it actually is: forecast the rest of DAY 49, given
all of DAY 48 plus day 49's first two hours (00:00-02:00).

Key additions over train_lgbm.py:
  1. (geohash, hour) target encoding      -> per-location intraday demand curve
  2. (geohash, timestamp) target encoding -> injects day-48 same-slot value at test time
  3. (geo_prefix5, hour) target encoding  -> fallback for unseen geohashes
  4. prev_day_same_slot lag               -> day-48 demand at this (geohash, timestamp),
                                             leakage-free (only filled for day-49 rows)
  5. An HONEST validation: fit on day 48, predict day 49 train rows. Random KFold
     leaks place/time across folds and overstates accuracy on this forecasting task.

Run on the Mac venv:
    python train_temporal.py
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
OUT_DIR = "submissions"
MODEL_DIR = "models"
N_FOLDS = 5
SEED = 42
TARGET = "demand"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────
print("Loading data...")
train_raw = pd.read_csv(TRAIN_PATH)
test_raw = pd.read_csv(TEST_PATH)
print(f"Train: {train_raw.shape}  Test: {test_raw.shape}")


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING (same as train_lgbm.py + composite keys)
# ─────────────────────────────────────────────
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
                lon_min, lon_max = (mid, lon_max) if bit else (lon_min, mid)
            else:
                mid = (lat_min + lat_max) / 2
                lat_min, lat_max = (mid, lat_max) if bit else (lat_min, mid)
            is_lon = not is_lon
    return (lat_min + lat_max) / 2, (lon_min + lon_max) / 2


def engineer(df):
    df = df.copy()

    ts = df["timestamp"].str.split(":", expand=True).astype(int)
    df["hour"] = ts[0]
    df["minute"] = ts[1]
    df["time_minutes"] = df["hour"] * 60 + df["minute"]
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["geo_prefix4"] = df["geohash"].str[:4]
    df["geo_prefix5"] = df["geohash"].str[:5]

    latlons = df["geohash"].map(
        lambda g: geohash_to_latlon(g) if isinstance(g, str) else (np.nan, np.nan)
    )
    df["lat"] = latlons.map(lambda x: x[0])
    df["lon"] = latlons.map(lambda x: x[1])

    df["temperature_missing"] = df["Temperature"].isna().astype(int)
    df["roadtype_missing"] = df["RoadType"].isna().astype(int)
    df["weather_missing"] = df["Weather"].isna().astype(int)

    df["RoadType"] = df["RoadType"].fillna("Residential")
    df["Weather"] = df["Weather"].fillna("Sunny")
    df["Temperature"] = df["Temperature"].fillna(df["Temperature"].median())

    df["large_vehicles_flag"] = (df["LargeVehicles"] == "Allowed").astype(int)
    df["has_landmark"] = (df["Landmarks"] == "Yes").astype(int)
    df["road_type_ord"] = (
        df["RoadType"].map({"Residential": 0, "Street": 1, "Highway": 2}).fillna(0).astype(int)
    )
    df["weather_severity"] = (
        df["Weather"].map({"Sunny": 0, "Foggy": 1, "Rainy": 2, "Snowy": 3}).fillna(0).astype(int)
    )

    df["lanes_x_time"] = df["NumberofLanes"] * df["time_minutes"]
    df["temp_x_weather"] = df["Temperature"] * df["weather_severity"]

    # Composite keys for time-aware target encoding
    df["geo_hour"] = df["geohash"].astype(str) + "@" + df["hour"].astype(str)
    df["geo_ts"] = df["geohash"].astype(str) + "@" + df["timestamp"].astype(str)
    df["prefix5_hour"] = df["geo_prefix5"].astype(str) + "@" + df["hour"].astype(str)

    return df


print("Engineering features...")
train = engineer(train_raw)
test = engineer(test_raw)


# ─────────────────────────────────────────────
# 3. TARGET ENCODING (OOF on train, full-map on test)
# ─────────────────────────────────────────────
def target_encode(train_df, test_df, col, target, n_folds=5, smoothing=10):
    global_mean = train_df[target].mean()
    oof_enc = np.zeros(len(train_df))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for tr_idx, val_idx in kf.split(train_df):
        fold_map = train_df.iloc[tr_idx].groupby(col)[target].agg(["mean", "count"])
        fold_map["smooth"] = (
            fold_map["count"] * fold_map["mean"] + smoothing * global_mean
        ) / (fold_map["count"] + smoothing)
        oof_enc[val_idx] = train_df.iloc[val_idx][col].map(fold_map["smooth"]).fillna(global_mean)
    full_map = train_df.groupby(col)[target].agg(["mean", "count"])
    full_map["smooth"] = (
        full_map["count"] * full_map["mean"] + smoothing * global_mean
    ) / (full_map["count"] + smoothing)
    test_enc = test_df[col].map(full_map["smooth"]).fillna(global_mean).values
    return oof_enc, test_enc


print("Target encoding (location + time)...")
TE_SPECS = [
    ("geohash", 10),
    ("geo_prefix4", 10),
    ("geo_prefix5", 10),
    ("geo_hour", 5),       # per-location hourly profile  <-- key new signal
    ("geo_ts", 3),         # per-location 15-min slot; for test = day-48 same slot
    ("prefix5_hour", 8),   # fallback for sparse / unseen geohashes
]
for col, sm in TE_SPECS:
    train[f"{col}_te"], test[f"{col}_te"] = target_encode(train, test, col, TARGET, smoothing=sm)

train["geo_te_x_time"] = train["geohash_te"] * train["time_minutes"]
test["geo_te_x_time"] = test["geohash_te"] * test["time_minutes"]


# ─────────────────────────────────────────────
# 4. LEAKAGE-FREE "YESTERDAY" LAG
#    prev_day_same_slot = demand at (geohash, timestamp) on day-1.
#    Day-49 rows  -> day-48 value (real).  Day-48 rows -> NaN (no day 47).
# ─────────────────────────────────────────────
print("Building previous-day same-slot lag...")
day48 = train[train["day"] == 48]
lag_map = day48.groupby(["geohash", "timestamp"])[TARGET].mean()


def add_lag(df):
    key = list(zip(df["geohash"], df["timestamp"]))
    lag = pd.Series(key).map(lag_map).values
    # only valid where the row's "yesterday" is day 48
    lag = np.where(df["day"].values == 49, lag, np.nan)
    return lag


train["prev_day_same_slot"] = add_lag(train)
test["prev_day_same_slot"] = add_lag(test)
cov = test["prev_day_same_slot"].notna().mean()
print(f"  lag coverage on test: {cov:.1%}")


# ─────────────────────────────────────────────
# 5. FEATURE SET
# ─────────────────────────────────────────────
FEATURES = [
    "day", "time_minutes", "hour", "minute", "hour_sin", "hour_cos",
    "NumberofLanes", "large_vehicles_flag", "has_landmark",
    "road_type_ord", "weather_severity", "Temperature",
    "lanes_x_time", "temp_x_weather", "lat", "lon",
    "geohash_te", "geo_prefix4_te", "geo_prefix5_te",
    "geo_hour_te", "geo_ts_te", "prefix5_hour_te",   # new time-aware TE
    "geo_te_x_time",
    "prev_day_same_slot",                            # new lag
    "temperature_missing", "roadtype_missing", "weather_missing",
]

y_train = np.log1p(train[TARGET])
print(f"Features: {len(FEATURES)}")

PARAMS = dict(
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=127,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_jobs=-1,
    random_state=SEED,
    verbose=-1,
)


def rmse(a, b):
    return np.sqrt(mean_squared_error(a, b))


# ─────────────────────────────────────────────
# 6. HONEST VALIDATION  (fit on day 48, predict day-49 train rows)
#    This mirrors the real task far better than random KFold.
# ─────────────────────────────────────────────
print("\n" + "=" * 58)
print("HONEST VALIDATION: train on day 48 -> predict day 49")
print("=" * 58)
tr48 = train["day"] == 48
val49 = train["day"] == 49
m = lgb.LGBMRegressor(**PARAMS)
m.fit(train.loc[tr48, FEATURES], y_train[tr48])
val_pred = np.clip(np.expm1(m.predict(train.loc[val49, FEATURES])), 0, 1)
val_true = train.loc[val49, TARGET].values
print(f"  day-49 holdout RMSE = {rmse(val_true, val_pred):.5f}")
print(f"  day-49 holdout MAE  = {mean_absolute_error(val_true, val_pred):.5f}")
print(f"  day-49 holdout R^2  = {r2_score(val_true, val_pred):.4f}")
print("  (compare to your random-KFold ~0.031; this number is the realistic one)")

# ─────────────────────────────────────────────
# 7. STANDARD RANDOM KFOLD (for apples-to-apples with old scripts)
# ─────────────────────────────────────────────
print("\n" + "=" * 58)
print("Random 5-fold CV (optimistic, for comparison)")
print("=" * 58)
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(train))
for fold, (tr_idx, va_idx) in enumerate(kf.split(train), 1):
    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(train[FEATURES].iloc[tr_idx], y_train.iloc[tr_idx])
    oof[va_idx] = model.predict(train[FEATURES].iloc[va_idx])
    fr = rmse(np.expm1(y_train.iloc[va_idx]), np.expm1(oof[va_idx]))
    print(f"  fold {fold}/{N_FOLDS} — RMSE={fr:.5f}")
print(f"  ▶️ OOF RMSE={rmse(np.expm1(y_train), np.expm1(oof)):.5f}")

# ─────────────────────────────────────────────
# 8. FINAL MODEL + SUBMISSION
# ─────────────────────────────────────────────
print("\nRetraining on full train data...")
final = lgb.LGBMRegressor(**PARAMS)
final.fit(train[FEATURES], y_train)

with open(f"{MODEL_DIR}/lgbm_temporal.pkl", "wb") as f:
    pickle.dump(final, f)

fi = pd.DataFrame({"feature": FEATURES, "importance": final.feature_importances_}) \
    .sort_values("importance", ascending=False)
print("\nTop 12 features:")
print(fi.head(12).to_string(index=False))

test_pred = np.clip(np.expm1(final.predict(test[FEATURES])), 0, 1)
sub = pd.DataFrame({"Index": test["Index"].values, "demand": test_pred})
sub_path = f"{OUT_DIR}/submission_temporal.csv"
sub.to_csv(sub_path, index=False)
print(f"\n✅ Submission saved: {sub_path}")
print(f"   Rows: {len(sub)}  demand range: [{test_pred.min():.5f}, {test_pred.max():.5f}]")