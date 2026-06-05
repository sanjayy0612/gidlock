"""
Gridlock Hackathon — Traffic Demand Prediction
4-model pipeline: LightGBM | XGBoost | Random Forest | CatBoost
MLflow tracking | MPS/Apple Silicon aware | Auto best-model submission
"""

import os
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG — edit paths here
# ─────────────────────────────────────────────
TRAIN_PATH = "train.csv"
TEST_PATH  = "test.csv"
SUB_PATH   = "sample_submission.csv"
OUT_DIR    = "submissions"
N_FOLDS    = 5
SEED       = 42
EXPERIMENT = "gridlock_demand"

os.makedirs(OUT_DIR, exist_ok=True)
mlflow.set_experiment(EXPERIMENT)

# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────
print("Loading data...")
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
sub   = pd.read_csv(SUB_PATH)

print(f"Train: {train.shape}  Test: {test.shape}")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def engineer(df, is_train=True):
    df = df.copy()

    # --- Timestamp → numeric ---
    ts = df["timestamp"].str.split(":", expand=True).astype(int)
    df["hour"]         = ts[0]
    df["minute"]       = ts[1]
    df["time_minutes"] = df["hour"] * 60 + df["minute"]

    # Cyclical encoding (24h cycle)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # --- Geohash spatial features ---
    df["geo_prefix4"] = df["geohash"].str[:4]
    df["geo_prefix5"] = df["geohash"].str[:5]

    # Geohash → lat/lng approximation (base32 decode, rough)
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

    latlons = df["geohash"].map(
        lambda g: geohash_to_latlon(g) if isinstance(g, str) else (np.nan, np.nan)
    )
    df["lat"] = latlons.map(lambda x: x[0])
    df["lon"] = latlons.map(lambda x: x[1])

    # --- Impute nulls ---
    df["RoadType"]    = df["RoadType"].fillna("Residential")   # most frequent
    df["Weather"]     = df["Weather"].fillna("Sunny")
    df["Temperature"] = df["Temperature"].fillna(df["Temperature"].median())

    # --- Binary encode ---
    df["large_vehicles_flag"] = (df["LargeVehicles"] == "Allowed").astype(int)
    df["has_landmark"]        = (df["Landmarks"] == "Yes").astype(int)

    # --- Road type ordinal (traffic capacity proxy) ---
    road_order = {"Residential": 0, "Street": 1, "Highway": 2}
    df["road_type_ord"] = df["RoadType"].map(road_order).fillna(0).astype(int)

    # --- Weather severity ---
    weather_severity = {"Sunny": 0, "Foggy": 1, "Rainy": 2, "Snowy": 3}
    df["weather_severity"] = df["Weather"].map(weather_severity).fillna(0).astype(int)

    # --- Interaction features ---
    df["lanes_x_time"] = df["NumberofLanes"] * df["time_minutes"]
    df["temp_x_weather"] = df["Temperature"] * df["weather_severity"]

    return df


print("Engineering features...")
train = engineer(train, is_train=True)
test  = engineer(test,  is_train=False)

# ─────────────────────────────────────────────
# 3. TARGET ENCODING for geohash (on train)
# ─────────────────────────────────────────────
print("Target encoding geohash...")

def target_encode(train_df, test_df, col, target, n_folds=5, smoothing=10):
    """Out-of-fold target encoding to avoid leakage."""
    global_mean = train_df[target].mean()
    oof_enc = np.zeros(len(train_df))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    for tr_idx, val_idx in kf.split(train_df):
        fold_map = (
            train_df.iloc[tr_idx]
            .groupby(col)[target]
            .agg(["mean", "count"])
        )
        fold_map["smooth"] = (
            fold_map["count"] * fold_map["mean"] + smoothing * global_mean
        ) / (fold_map["count"] + smoothing)
        oof_enc[val_idx] = train_df.iloc[val_idx][col].map(fold_map["smooth"]).fillna(global_mean)

    # Full train encoding for test
    full_map = (
        train_df.groupby(col)[target]
        .agg(["mean", "count"])
    )
    full_map["smooth"] = (
        full_map["count"] * full_map["mean"] + smoothing * global_mean
    ) / (full_map["count"] + smoothing)

    test_enc = test_df[col].map(full_map["smooth"]).fillna(global_mean)
    return oof_enc, test_enc.values


TARGET = "demand"

for geo_col in ["geohash", "geo_prefix4", "geo_prefix5"]:
    enc_col = f"{geo_col}_te"
    train[enc_col], test[enc_col] = target_encode(train, test, geo_col, TARGET)

# ─────────────────────────────────────────────
# 4. FEATURE SET
# ─────────────────────────────────────────────
FEATURES = [
    "day", "time_minutes", "hour", "minute",
    "hour_sin", "hour_cos",
    "NumberofLanes", "large_vehicles_flag", "has_landmark",
    "road_type_ord", "weather_severity", "Temperature",
    "lanes_x_time", "temp_x_weather",
    "lat", "lon",
    "geohash_te", "geo_prefix4_te", "geo_prefix5_te",
]

X_train = train[FEATURES]
y_train = np.log1p(train[TARGET])   # log1p transform
X_test  = test[FEATURES]

print(f"Feature set: {len(FEATURES)} features")
print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")

# ─────────────────────────────────────────────
# 5. CROSS-VAL HELPER
# ─────────────────────────────────────────────
def cross_val_predict_oof(model_name, model_fn, X, y, X_test, n_folds=N_FOLDS):
    """Returns OOF preds, test preds (averaged), and per-fold metrics."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof_preds  = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_metrics = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = model_fn()
        model.fit(X_tr, y_tr)

        val_pred  = model.predict(X_val)
        test_pred = model.predict(X_test)

        oof_preds[val_idx] = val_pred
        test_preds += test_pred / n_folds

        # Metrics in original space
        val_orig  = np.expm1(y_val)
        pred_orig = np.expm1(val_pred)
        rmse = np.sqrt(mean_squared_error(val_orig, pred_orig))
        mae  = mean_absolute_error(val_orig, pred_orig)
        r2   = r2_score(val_orig, pred_orig)
        fold_metrics.append((rmse, mae, r2))
        print(f"  [{model_name}] fold {fold}/{n_folds} — RMSE={rmse:.5f}  MAE={mae:.5f}  R²={r2:.4f}")

    return oof_preds, test_preds, fold_metrics


# ─────────────────────────────────────────────
# 6. MODEL DEFINITIONS
# ─────────────────────────────────────────────
# LightGBM — uses Apple Silicon via OpenMP (no MPS needed, very fast on M2)
def make_lgbm():
    return lgb.LGBMRegressor(
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

# XGBoost — hist device uses CPU; set device='cuda' if CUDA available (not on M2)
def make_xgb():
    return xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",   # fastest CPU method, ARM-optimized
        device="cpu",
        n_jobs=-1,
        random_state=SEED,
        verbosity=0,
    )

# Random Forest — parallelized, solid baseline
def make_rf():
    return RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_leaf=5,
        max_features=0.6,
        n_jobs=-1,
        random_state=SEED,
    )

# CatBoost — native categorical support, fast on M2 (uses CPU BLAS)
def make_catboost():
    return cb.CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        loss_function="RMSE",
        task_type="CPU",       # MPS not yet supported in CatBoost; CPU is fast enough
        random_seed=SEED,
        verbose=0,
    )

MODELS = {
    "LightGBM":    make_lgbm,
    "XGBoost":     make_xgb,
    "RandomForest": make_rf,
    "CatBoost":    make_catboost,
}

# ─────────────────────────────────────────────
# 7. TRAIN + MLFLOW
# ─────────────────────────────────────────────
results = {}   # model_name → {oof, test_preds, metrics}

for model_name, model_fn in MODELS.items():
    print(f"\n{'='*50}")
    print(f"Training {model_name}...")
    print(f"{'='*50}")

    with mlflow.start_run(run_name=model_name):
        mlflow.set_tag("model", model_name)
        mlflow.log_param("n_folds", N_FOLDS)
        mlflow.log_param("features", len(FEATURES))
        mlflow.log_param("train_rows", len(X_train))

        oof_preds, test_preds, fold_metrics = cross_val_predict_oof(
            model_name, model_fn, X_train, y_train, X_test
        )

        # Aggregate metrics across folds
        rmse_cv = np.mean([m[0] for m in fold_metrics])
        mae_cv  = np.mean([m[1] for m in fold_metrics])
        r2_cv   = np.mean([m[2] for m in fold_metrics])
        rmse_std = np.std([m[0] for m in fold_metrics])

        # OOF overall score
        oof_orig  = np.expm1(oof_preds)
        y_orig    = np.expm1(y_train)
        oof_rmse  = np.sqrt(mean_squared_error(y_orig, oof_orig))
        oof_mae   = mean_absolute_error(y_orig, oof_orig)
        oof_r2    = r2_score(y_orig, oof_orig)

        mlflow.log_metric("cv_rmse_mean", rmse_cv)
        mlflow.log_metric("cv_rmse_std",  rmse_std)
        mlflow.log_metric("cv_mae_mean",  mae_cv)
        mlflow.log_metric("cv_r2_mean",   r2_cv)
        mlflow.log_metric("oof_rmse",     oof_rmse)
        mlflow.log_metric("oof_mae",      oof_mae)
        mlflow.log_metric("oof_r2",       oof_r2)

        print(f"\n  ▶ {model_name} OOF — RMSE={oof_rmse:.5f}  MAE={oof_mae:.5f}  R²={oof_r2:.4f}")

        # Save per-model submission (use test Index, not sample_submission rows)
        test_final = np.clip(np.expm1(test_preds), 0, 1)
        sub_model = pd.DataFrame({"Index": test["Index"].values, "demand": test_final})
        sub_path = f"{OUT_DIR}/submission_{model_name}.csv"
        sub_model.to_csv(sub_path, index=False)
        mlflow.log_artifact(sub_path)

        results[model_name] = {
            "oof_preds":  oof_preds,
            "test_preds": test_preds,
            "oof_rmse":   oof_rmse,
            "oof_mae":    oof_mae,
            "oof_r2":     oof_r2,
        }

# ─────────────────────────────────────────────
# 8. BEST MODEL SELECTION + ENSEMBLE
# ─────────────────────────────────────────────
print(f"\n{'='*50}")
print("RESULTS SUMMARY")
print(f"{'='*50}")
print(f"{'Model':<16} {'OOF RMSE':>10} {'OOF MAE':>10} {'OOF R²':>8}")
print("-" * 48)

for name, res in results.items():
    print(f"{name:<16} {res['oof_rmse']:>10.5f} {res['oof_mae']:>10.5f} {res['oof_r2']:>8.4f}")

best_name = min(results, key=lambda k: results[k]["oof_rmse"])
print(f"\n✅ Best model: {best_name} (lowest OOF RMSE = {results[best_name]['oof_rmse']:.5f})")

# Best model submission
best_preds = np.clip(np.expm1(results[best_name]["test_preds"]), 0, 1)
sub_best = pd.DataFrame({"Index": test["Index"].values, "demand": best_preds})
sub_best.to_csv(f"{OUT_DIR}/submission_BEST_{best_name}.csv", index=False)
print(f"✅ Best submission saved: {OUT_DIR}/submission_BEST_{best_name}.csv")

# Ensemble (equal-weight average of all 4)
print("\nBuilding ensemble (equal-weight average)...")
all_test_preds = np.stack([results[n]["test_preds"] for n in MODELS])
ensemble_preds = np.mean(all_test_preds, axis=0)
ensemble_final = np.clip(np.expm1(ensemble_preds), 0, 1)
sub_ens = pd.DataFrame({"Index": test["Index"].values, "demand": ensemble_final})
sub_ens.to_csv(f"{OUT_DIR}/submission_ensemble.csv", index=False)

# OOF RMSE for ensemble
all_oof = np.stack([results[n]["oof_preds"] for n in MODELS])
ens_oof = np.mean(all_oof, axis=0)
ens_rmse = np.sqrt(mean_squared_error(np.expm1(y_train), np.expm1(ens_oof)))
print(f"Ensemble OOF RMSE: {ens_rmse:.5f}")
print(f"✅ Ensemble submission saved: {OUT_DIR}/submission_ensemble.csv")

print("\nTo view MLflow UI run:")
print("  mlflow ui")
print("  → open http://127.0.0.1:5000")
