"""
train_lgbm_no_temperature.py
Controlled LightGBM experiment — 22-feature baseline without Temperature,
then remove selected features one by one and retrain with the same pipeline.
"""

import os
import pickle
import json
import warnings
import numpy as np
import pandas as pd
import mlflow
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
OUT_DIR = "submissions"
MODEL_DIR = "models"
N_FOLDS = 5
SEED = 42
EXPERIMENT = "gridlock_final"
RUN_NAME = "lgbm_22_features_no_temperature_ablation"
CURRENT_BEST_RMSE = 0.03138

BASELINE_FEATURES = [
    # Base
    "day", "time_minutes", "hour", "minute",
    "hour_sin", "hour_cos",
    "NumberofLanes", "large_vehicles_flag", "has_landmark",
    "road_type_ord", "weather_severity",
    "lanes_x_time", "temp_x_weather",
    "lat", "lon",
    # Target encoding
    "geohash_te", "geo_prefix4_te", "geo_prefix5_te",
    # Confirmed by ablation (+0.00040 improvement over 19 features)
    "geo_te_x_time",
    "temperature_missing",
    "roadtype_missing",
    "weather_missing",
]

CANDIDATE_REMOVALS = [
    "temp_x_weather",
    "geo_prefix4_te",
    "lanes_x_time",
    "minute",
    "weather_missing",
    "hour_sin",
]

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

    # Missing flags (confirmed helpful by ablation)
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
    return oof_enc, test_df[col].map(full_map["smooth"]).fillna(global_mean).values


print("Target encoding...")
for geo_col in ["geohash", "geo_prefix4", "geo_prefix5"]:
    train[f"{geo_col}_te"], test[f"{geo_col}_te"] = target_encode(train, test, geo_col, TARGET)

# Interaction: now that TE is computed
train["geo_te_x_time"] = train["geohash_te"] * train["time_minutes"]
test["geo_te_x_time"] = test["geohash_te"] * test["time_minutes"]

y_train = np.log1p(train[TARGET])

# ─────────────────────────────────────────────
# 4. PARAMS (baseline won — tuning confirmed)
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


def train_and_score(feature_list):
    x_train = train[feature_list]
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(x_train))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(x_train), 1):
        model = lgb.LGBMRegressor(**PARAMS)
        model.fit(x_train.iloc[tr_idx], y_train.iloc[tr_idx])
        oof[val_idx] = model.predict(x_train.iloc[val_idx])
        fold_rmse = np.sqrt(mean_squared_error(
            np.expm1(y_train.iloc[val_idx]),
            np.expm1(oof[val_idx])
        ))
        print(f"  fold {fold}/{N_FOLDS} — RMSE={fold_rmse:.5f}")

    oof_rmse = np.sqrt(mean_squared_error(np.expm1(y_train), np.expm1(oof)))
    oof_mae = mean_absolute_error(np.expm1(y_train), np.expm1(oof))
    oof_r2 = r2_score(np.expm1(y_train), np.expm1(oof))
    return oof_rmse, oof_mae, oof_r2


def save_artifacts(feature_list, artifact_suffix):
    x_train = train[feature_list]
    x_test = test[feature_list]

    print("\nRetraining on full train data...")
    final_model = lgb.LGBMRegressor(**PARAMS)
    final_model.fit(x_train, y_train)

    model_path = f"{MODEL_DIR}/lgbm_{artifact_suffix}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)

    fi = pd.DataFrame({
        "feature": feature_list,
        "importance": final_model.feature_importances_
    }).sort_values("importance", ascending=False)
    fi_path = f"{MODEL_DIR}/feature_importance_{artifact_suffix}.csv"
    fi.to_csv(fi_path, index=False)

    test_preds = np.clip(np.expm1(final_model.predict(x_test)), 0, 1)
    sub = pd.DataFrame({"Index": test["Index"].values, "demand": test_preds})
    sub_path = f"{OUT_DIR}/submission_{artifact_suffix}.csv"
    sub.to_csv(sub_path, index=False)

    meta = {
        "n_features": len(feature_list),
        "features": feature_list,
        "params": PARAMS,
        "model_path": model_path,
        "feature_importance_path": fi_path,
        "submission_path": sub_path,
    }
    meta_path = f"{MODEL_DIR}/lgbm_{artifact_suffix}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return model_path, fi_path, sub_path, meta_path


results = []

with mlflow.start_run(run_name=RUN_NAME):
    mlflow.log_params(PARAMS)
    mlflow.log_param("baseline_feature_count", len(BASELINE_FEATURES))
    mlflow.log_param("baseline_features", BASELINE_FEATURES)
    mlflow.log_param("candidate_removals", CANDIDATE_REMOVALS)

    print(f"\nBaseline 22-feature model")
    print(f"Features: {len(BASELINE_FEATURES)}")
    baseline_rmse, baseline_mae, baseline_r2 = train_and_score(BASELINE_FEATURES)
    mlflow.log_metric("baseline_oof_rmse", baseline_rmse)
    mlflow.log_metric("baseline_oof_mae", baseline_mae)
    mlflow.log_metric("baseline_oof_r2", baseline_r2)

    print(f"\n  ▶ Baseline OOF RMSE={baseline_rmse:.5f}  MAE={baseline_mae:.5f}  R²={baseline_r2:.4f}")
    print(f"  ▶ vs current best 23-feature model ({CURRENT_BEST_RMSE:.5f}): {baseline_rmse - CURRENT_BEST_RMSE:+.5f}")

    for feature_to_remove in CANDIDATE_REMOVALS:
        feature_subset = [f for f in BASELINE_FEATURES if f != feature_to_remove]
        suffix = f"no_temperature_no_{feature_to_remove}"

        print(f"\n{'=' * 60}")
        print(f"Testing without: {feature_to_remove}")
        print(f"Features: {len(feature_subset)}")
        print(f"{'=' * 60}")

        with mlflow.start_run(run_name=suffix, nested=True):
            mlflow.log_params(PARAMS)
            mlflow.log_param("feature_removed", feature_to_remove)
            mlflow.log_param("n_features", len(feature_subset))
            mlflow.log_param("features", feature_subset)

            oof_rmse, oof_mae, oof_r2 = train_and_score(feature_subset)
            delta_vs_22 = oof_rmse - baseline_rmse
            delta_vs_23 = oof_rmse - CURRENT_BEST_RMSE

            mlflow.log_metric("oof_rmse", oof_rmse)
            mlflow.log_metric("oof_mae", oof_mae)
            mlflow.log_metric("oof_r2", oof_r2)
            mlflow.log_metric("delta_vs_22_feature_baseline", delta_vs_22)
            mlflow.log_metric("delta_vs_23_feature_best", delta_vs_23)

            model_path, fi_path, sub_path, meta_path = save_artifacts(feature_subset, suffix)
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(fi_path)
            mlflow.log_artifact(sub_path)
            mlflow.log_artifact(meta_path)

        results.append({
            "feature_removed": feature_to_remove,
            "rmse": oof_rmse,
            "mae": oof_mae,
            "r2": oof_r2,
            "delta_vs_22": delta_vs_22,
            "delta_vs_23": delta_vs_23,
        })

results_df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
results_path = f"{MODEL_DIR}/no_temperature_feature_drop_results.csv"
results_df.to_csv(results_path, index=False)

print(f"\nSaved results: {results_path}")
print("\nFeature Removed                RMSE      Delta vs 22    Delta vs 23")
for row in results_df.itertuples(index=False):
    print(
        f"{row.feature_removed:<28} {row.rmse:>8.5f}   "
        f"{row.delta_vs_22:+.5f}       {row.delta_vs_23:+.5f}"
    )

best_row = results_df.iloc[0]
print("\nBest candidate from 22-feature baseline:")
print(f"Remove: {best_row['feature_removed']}")
print(f"RMSE  : {best_row['rmse']:.5f}")

if best_row["rmse"] < CURRENT_BEST_RMSE:
    print("NEW BEST MODEL")
else:
    print("Keep current best model")

print(f"\nmlflow ui  →  http://127.0.0.1:5000")
