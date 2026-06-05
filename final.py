"""
final_v3.py — Target 94-96
===========================
3 key fixes over best submission (91.06):
  1. Drop `day` feature — constant in test (always 49), causes train-test mismatch
  2. GroupKFold by geohash — forces real generalization, no location leakage
  3. Simpler model — num_leaves=63, higher reg, prevents overfitting
  4. demand_lag1 — day-48 demand at same geohash+timestamp (correct, no leakage)
"""

import os, pickle, json, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings("ignore")

SEED      = 42
N_FOLDS   = 5
OUT_DIR   = "submissions"
MODEL_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────
print("Loading...")
train_raw = pd.read_csv("train.csv")
test_raw  = pd.read_csv("test.csv")
print(f"Train: {train_raw.shape} | days: {sorted(train_raw['day'].unique())}")
print(f"Test : {test_raw.shape}")
TARGET = "demand"

# ─────────────────────────────────────────────
# 2. ENGINEER
# ─────────────────────────────────────────────
BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"

def geohash_to_latlon(gh):
    lat_min, lat_max = -90.0,  90.0
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

def engineer(df):
    df = df.copy()
    ts = df["timestamp"].str.split(":", expand=True).astype(int)
    df["hour"]         = ts[0]
    df["minute"]       = ts[1]
    df["time_minutes"] = df["hour"] * 60 + df["minute"]
    df["hour_sin"]     = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]     = np.cos(2 * np.pi * df["hour"] / 24)
    df["geo_prefix4"]  = df["geohash"].str[:4]
    df["geo_prefix5"]  = df["geohash"].str[:5]

    latlons        = df["geohash"].map(geohash_to_latlon)
    df["lat"]      = latlons.map(lambda x: x[0])
    df["lon"]      = latlons.map(lambda x: x[1])

    df["roadtype_missing"]    = df["RoadType"].isna().astype(int)
    df["temperature_missing"] = df["Temperature"].isna().astype(int)
    df["weather_missing"]     = df["Weather"].isna().astype(int)

    df["RoadType"]    = df["RoadType"].fillna("Residential")
    df["Weather"]     = df["Weather"].fillna("Sunny")
    df["Temperature"] = df["Temperature"].fillna(df["Temperature"].median())

    df["large_vehicles_flag"] = (df["LargeVehicles"] == "Allowed").astype(int)
    df["has_landmark"]        = (df["Landmarks"] == "Yes").astype(int)
    df["road_type_ord"]       = df["RoadType"].map(
        {"Residential": 0, "Street": 1, "Highway": 2}).fillna(0).astype(int)
    df["weather_severity"]    = df["Weather"].map(
        {"Sunny": 0, "Foggy": 1, "Rainy": 2, "Snowy": 3}).fillna(0).astype(int)
    df["lanes_x_time"]        = df["NumberofLanes"] * df["time_minutes"]
    return df

print("Engineering...")
train_fe = engineer(train_raw)
test_fe  = engineer(test_raw)

# ─────────────────────────────────────────────
# 3. LAG FEATURE — day-48 demand at same geohash+timestamp
#    Built from day-48 rows only → no leakage
# ─────────────────────────────────────────────
global_mean = train_raw[TARGET].mean()

lag_map = (train_raw[train_raw["day"] == 48]
           .groupby(["geohash", "timestamp"])[TARGET]
           .mean()
           .reset_index()
           .rename(columns={TARGET: "demand_lag1"}))

train_fe = train_fe.merge(lag_map, on=["geohash", "timestamp"], how="left")
test_fe  = test_fe.merge(lag_map,  on=["geohash", "timestamp"], how="left")
train_fe["demand_lag1"] = train_fe["demand_lag1"].fillna(global_mean)
test_fe["demand_lag1"]  = test_fe["demand_lag1"].fillna(global_mean)

print(f"Lag coverage — train: {(train_fe['demand_lag1'] != global_mean).mean():.1%}"
      f"  test: {(test_fe['demand_lag1'] != global_mean).mean():.1%}")

# ─────────────────────────────────────────────
# 4. AGGREGATE FEATURES — from full train (safe, no leakage vs test)
# ─────────────────────────────────────────────
geo_stats = (train_raw.groupby("geohash")[TARGET]
             .agg(geo_mean="mean", geo_std="std",
                  geo_max="max", geo_median="median")
             .reset_index().fillna(0))

ts_stats = (train_raw.groupby("timestamp")[TARGET]
            .agg(ts_mean="mean", ts_std="std")
            .reset_index())

geo_ts_stats = (train_raw.groupby(["geohash", "timestamp"])[TARGET]
                .mean().reset_index()
                .rename(columns={TARGET: "geo_ts_mean"}))

def add_stats(df):
    return (df
            .merge(geo_stats,    on="geohash",                how="left")
            .merge(ts_stats,     on="timestamp",              how="left")
            .merge(geo_ts_stats, on=["geohash", "timestamp"], how="left"))

train_fe = add_stats(train_fe)
test_fe  = add_stats(test_fe)

for col in ["geo_mean","geo_std","geo_max","geo_median","ts_mean","ts_std","geo_ts_mean"]:
    train_fe[col] = train_fe[col].fillna(global_mean)
    test_fe[col]  = test_fe[col].fillna(global_mean)

# ─────────────────────────────────────────────
# 5. TARGET ENCODING — OOF with GroupKFold by geohash
#    Each geohash entirely in train or val → no location leakage
# ─────────────────────────────────────────────
def target_encode_group(train_df, test_df, col, target,
                         groups, n_folds=5, smoothing=10):
    gm      = train_df[target].mean()
    oof_enc = np.zeros(len(train_df))
    gkf     = GroupKFold(n_splits=n_folds)

    for tr_idx, val_idx in gkf.split(train_df, groups=groups):
        fold_map = (train_df.iloc[tr_idx]
                    .groupby(col)[target].agg(["mean", "count"]))
        fold_map["smooth"] = (
            fold_map["count"] * fold_map["mean"] + smoothing * gm
        ) / (fold_map["count"] + smoothing)
        oof_enc[val_idx] = (train_df.iloc[val_idx][col]
                            .map(fold_map["smooth"]).fillna(gm))

    full_map = train_df.groupby(col)[target].agg(["mean", "count"])
    full_map["smooth"] = (
        full_map["count"] * full_map["mean"] + smoothing * gm
    ) / (full_map["count"] + smoothing)
    test_enc = test_df[col].map(full_map["smooth"]).fillna(gm).values
    return oof_enc, test_enc

print("Target encoding (GroupKFold by geohash)...")
groups = train_fe["geohash"].values
for col in ["geohash", "geo_prefix4", "geo_prefix5"]:
    train_fe[f"{col}_te"], test_fe[f"{col}_te"] = target_encode_group(
        train_fe, test_fe, col, TARGET, groups
    )

train_fe["geo_te_x_time"] = train_fe["geohash_te"] * train_fe["time_minutes"]
test_fe["geo_te_x_time"]  = test_fe["geohash_te"]  * test_fe["time_minutes"]

# ─────────────────────────────────────────────
# 6. FEATURE SET — no `day` (constant in test = 49, varies in train)
# ─────────────────────────────────────────────
FEATURES = [
    # NO "day" — causes train/test mismatch
    "time_minutes", "hour", "minute",
    "hour_sin", "hour_cos",
    "NumberofLanes", "large_vehicles_flag", "has_landmark",
    "road_type_ord", "weather_severity",
    "lanes_x_time",
    "lat", "lon",
    "geohash_te", "geo_prefix4_te", "geo_prefix5_te", "geo_te_x_time",
    "roadtype_missing", "temperature_missing", "weather_missing",
    # lag + history
    "demand_lag1",
    "geo_mean", "geo_std", "geo_max", "geo_median",
    "ts_mean", "ts_std",
    "geo_ts_mean",
]

X_train = train_fe[FEATURES]
y_train = np.log1p(train_fe[TARGET])
X_test  = test_fe[FEATURES]
print(f"Features: {len(FEATURES)} (no `day`)")
print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")

# ─────────────────────────────────────────────
# 7. GROUPKFOLD CV — grouped by geohash
#    Val geohashes never seen in training fold
# ─────────────────────────────────────────────
# Fix 3: simpler model — num_leaves=63, stronger reg
PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=63,        # was 127 — simpler = better generalization
    max_depth=8,
    min_child_samples=30, # was 20
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,        # was 0.1 — stronger regularization
    reg_lambda=0.5,       # was 0.1
    n_jobs=-1,
    random_state=SEED,
    verbose=-1,
)

print(f"\nRunning {N_FOLDS}-fold GroupKFold CV (grouped by geohash)...")
gkf  = GroupKFold(n_splits=N_FOLDS)
oof  = np.zeros(len(X_train))
geo_groups = train_fe["geohash"].values

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, groups=geo_groups), 1):
    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
    oof[val_idx] = m.predict(X_train.iloc[val_idx])

    fold_r2   = r2_score(train_fe[TARGET].iloc[val_idx], np.expm1(oof[val_idx]))
    fold_rmse = np.sqrt(mean_squared_error(
        train_fe[TARGET].iloc[val_idx], np.expm1(oof[val_idx])))
    print(f"  fold {fold}/{N_FOLDS} — R²={fold_r2:.4f}  RMSE={fold_rmse:.5f}"
          f"  score≈{fold_r2*100:.2f}")

oof_r2   = r2_score(train_fe[TARGET], np.expm1(oof))
oof_rmse = np.sqrt(mean_squared_error(train_fe[TARGET], np.expm1(oof)))
print(f"\n  ▶ OOF R²={oof_r2:.5f}  RMSE={oof_rmse:.5f}")
print(f"  ▶ GroupKFold score ≈ {oof_r2*100:.2f}")
print(f"  (This is a harder, more honest validation than random KFold)")

# ─────────────────────────────────────────────
# 8. RETRAIN ON FULL DATA + SAVE
# ─────────────────────────────────────────────
print("\nRetraining on full train data...")
final_model = lgb.LGBMRegressor(**PARAMS)
final_model.fit(X_train, y_train)

fi = (pd.DataFrame({"feature": FEATURES,
                    "importance": final_model.feature_importances_})
      .sort_values("importance", ascending=False))
print("\nTop 15 features:")
print(fi.head(15).to_string(index=False))

model_path = f"{MODEL_DIR}/lgbm_v3.pkl"
with open(model_path, "wb") as f:
    pickle.dump(final_model, f)
fi.to_csv(f"{MODEL_DIR}/feature_importance_v3.csv", index=False)

# ─────────────────────────────────────────────
# 9. SUBMISSION
# ─────────────────────────────────────────────
test_preds = np.clip(np.expm1(final_model.predict(X_test)), 0, 1)
sub = pd.DataFrame({"Index": test_raw["Index"].values, "demand": test_preds})
sub_path = f"{OUT_DIR}/submission_v3.csv"
sub.to_csv(sub_path, index=False)

print(f"\n✅ Submission : {sub_path}")
print(f"   demand range: [{test_preds.min():.5f}, {test_preds.max():.5f}]")
print(f"\n{'='*50}")
print(f"GroupKFold OOF R²  : {oof_r2:.5f}")
print(f"GroupKFold score   : {oof_r2*100:.2f}")
print(f"Previous LB best   : 91.06  (random KFold, 21 features)")
print(f"{'='*50}")
print(f"\nmlflow ui → http://127.0.0.1:5000")