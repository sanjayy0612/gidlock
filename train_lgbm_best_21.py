"""
train_lgbm_best_21.py
Clean final LightGBM trainer using the best 21-feature set:
- Temperature removed
- temp_x_weather removed

Uses the confirmed best LightGBM parameters, trains on full data,
and writes a separate submission file.
"""

import os
import pickle
import json
import warnings
import numpy as np
import pandas as pd
import mlflow
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
OUT_DIR = "submissions"
MODEL_DIR = "models"
SEED = 42
EXPERIMENT = "gridlock_final"
RUN_NAME = "lgbm_best_21_features_full_train"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
mlflow.set_experiment(EXPERIMENT)

# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────
print("Loading data...")
train_raw = pd.read_csv(TRAIN_PATH)
test_raw = pd.read_csv(TEST_PATH)
print(f"Train: {train_raw.shape}  Test: {test_raw.shape}")


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer(df):
    df = df.copy()

    # Timestamp
    ts = df["timestamp"].str.split(":", expand=True).astype(int)
    df["hour"] = ts[0]
    df["minute"] = ts[1]
    df["time_minutes"] = df["hour"] * 60 + df["minute"]
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Geohash prefixes
    df["geo_prefix4"] = df["geohash"].str[:4]
    df["geo_prefix5"] = df["geohash"].str[:5]

    # Geohash → lat/lon
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
                    if bit:
                        lon_min = mid
                    else:
                        lon_max = mid
                else:
                    mid = (lat_min + lat_max) / 2
                    if bit:
                        lat_min = mid
                    else:
                        lat_max = mid
                is_lon = not is_lon
        return (lat_min + lat_max) / 2, (lon_min + lon_max) / 2

    latlons = df["geohash"].map(
        lambda g: geohash_to_latlon(g) if isinstance(g, str) else (np.nan, np.nan)
    )
    df["lat"] = latlons.map(lambda x: x[0])
    df["lon"] = latlons.map(lambda x: x[1])

    # Missing flags
    df["temperature_missing"] = df["Temperature"].isna().astype(int)
    df["roadtype_missing"] = df["RoadType"].isna().astype(int)
    df["weather_missing"] = df["Weather"].isna().astype(int)

    # Impute
    df["RoadType"] = df["RoadType"].fillna("Residential")
    df["Weather"] = df["Weather"].fillna("Sunny")
    df["Temperature"] = df["Temperature"].fillna(df["Temperature"].median())

    # Encode
    df["large_vehicles_flag"] = (df["LargeVehicles"] == "Allowed").astype(int)
    df["has_landmark"] = (df["Landmarks"] == "Yes").astype(int)
    df["road_type_ord"] = (
        df["RoadType"].map({"Residential": 0, "Street": 1, "Highway": 2}).fillna(0).astype(int)
    )
    df["weather_severity"] = (
        df["Weather"].map({"Sunny": 0, "Foggy": 1, "Rainy": 2, "Snowy": 3}).fillna(0).astype(int)
    )

    # Interactions
    df["lanes_x_time"] = df["NumberofLanes"] * df["time_minutes"]
    df["temp_x_weather"] = df["Temperature"] * df["weather_severity"]

    return df


print("Engineering features...")
train = engineer(train_raw)
test = engineer(test_raw)

# ─────────────────────────────────────────────
# 3. TARGET ENCODING
# ─────────────────────────────────────────────
TARGET = "demand"


def target_encode(train_df, test_df, col, target, smoothing=10):
    global_mean = train_df[target].mean()
    full_map = train_df.groupby(col)[target].agg(["mean", "count"])
    full_map["smooth"] = (
        full_map["count"] * full_map["mean"] + smoothing * global_mean
    ) / (full_map["count"] + smoothing)
    train_enc = train_df[col].map(full_map["smooth"]).fillna(global_mean).values
    test_enc = test_df[col].map(full_map["smooth"]).fillna(global_mean).values
    return train_enc, test_enc


print("Target encoding...")
for geo_col in ["geohash", "geo_prefix4", "geo_prefix5"]:
    train[f"{geo_col}_te"], test[f"{geo_col}_te"] = target_encode(train, test, geo_col, TARGET)

# Interaction: now that TE is computed
train["geo_te_x_time"] = train["geohash_te"] * train["time_minutes"]
test["geo_te_x_time"] = test["geohash_te"] * test["time_minutes"]

# ─────────────────────────────────────────────
# 4. FEATURE SET (best 21)
# ─────────────────────────────────────────────
FEATURES = [
    "day", "time_minutes", "hour", "minute",
    "hour_sin", "hour_cos",
    "NumberofLanes", "large_vehicles_flag", "has_landmark",
    "road_type_ord", "weather_severity",
    "lanes_x_time",
    "lat", "lon",
    "geohash_te", "geo_prefix4_te", "geo_prefix5_te",
    "geo_te_x_time",
    "temperature_missing",
    "roadtype_missing",
    "weather_missing",
]

X_train = train[FEATURES]
y_train = np.log1p(train[TARGET])
X_test = test[FEATURES]
print(f"Features: {len(FEATURES)}  |  X_train: {X_train.shape}  X_test: {X_test.shape}")

# ─────────────────────────────────────────────
# 5. PARAMS
# ─────────────────────────────────────────────
PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.05,
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

# ─────────────────────────────────────────────
# 6. FULL TRAINING
# ─────────────────────────────────────────────
print("\nTraining final model on full training data...")
final_model = lgb.LGBMRegressor(**PARAMS)
final_model.fit(X_train, y_train)

with mlflow.start_run(run_name=RUN_NAME):
    mlflow.log_params(PARAMS)
    mlflow.log_param("n_features", len(FEATURES))
    mlflow.log_param("features", FEATURES)

    # Save model
    model_path = f"{MODEL_DIR}/lgbm_best_21_features.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)

    # Save feature importance
    fi = pd.DataFrame({
        "feature": FEATURES,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi_path = f"{MODEL_DIR}/feature_importance_best_21.csv"
    fi.to_csv(fi_path, index=False)

    # Save metadata
    meta = {
        "n_features": len(FEATURES),
        "features": FEATURES,
        "params": PARAMS,
    }
    meta_path = f"{MODEL_DIR}/lgbm_best_21_features_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Submission
    test_preds = np.clip(np.expm1(final_model.predict(X_test)), 0, 1)
    sub = pd.DataFrame({"Index": test["Index"].values, "demand": test_preds})
    sub_path = f"{OUT_DIR}/submission_best_21_features.csv"
    sub.to_csv(sub_path, index=False)

    mlflow.log_artifact(model_path)
    mlflow.log_artifact(fi_path)
    mlflow.log_artifact(meta_path)
    mlflow.log_artifact(sub_path)

print(f"✅ Model saved    : {model_path}")
print(f"✅ Submission     : {sub_path}")
print(f"✅ Importance CSV : {fi_path}")
print(f"\nTop 10 features:")
print(fi.head(10).to_string(index=False))
print(f"\nRows: {len(sub)}  demand range: [{test_preds.min():.5f}, {test_preds.max():.5f}]")
print("\nmlflow ui  →  http://127.0.0.1:5000")
