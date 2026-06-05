"""
tune_lgbm.py  (v2 — fixed)
Key fixes vs v1:
  - No early_stopping in CV (consistent with final retrain)
  - n_estimators fixed at 2000, lr tighter range (0.02–0.08)
  - Search space centered around baseline that scored 0.03178
  - Final retrain on FULL data with best params → model saved
"""

import os
import pickle
import json
import warnings
import numpy as np
import pandas as pd
import mlflow
import optuna
from optuna.samplers import TPESampler
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TRAIN_PATH  = "train.csv"
TEST_PATH   = "test.csv"
OUT_DIR     = "submissions"
MODEL_DIR   = "models"
N_FOLDS     = 5
N_TRIALS    = 50
SEED        = 42
EXPERIMENT  = "gridlock_lgbm_tuning_v2"
BASELINE_RMSE = 0.03178   # from train.py

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
mlflow.set_experiment(EXPERIMENT)

# ─────────────────────────────────────────────
# 1. LOAD + ENGINEER
# ─────────────────────────────────────────────
print("Loading data...")
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

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
    df["geo_te_x_time"]       = df["time_minutes"]   # filled after TE
    return df

print("Engineering features...")
train = engineer(train)
test  = engineer(test)

# ─────────────────────────────────────────────
# 2. TARGET ENCODING
# ─────────────────────────────────────────────
TARGET = "demand"

def target_encode(train_df, test_df, col, target, n_folds=5, smoothing=10):
    global_mean = train_df[target].mean()
    oof_enc = np.zeros(len(train_df))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for tr_idx, val_idx in kf.split(train_df):
        fold_map = train_df.iloc[tr_idx].groupby(col)[target].agg(["mean", "count"])
        fold_map["smooth"] = (fold_map["count"] * fold_map["mean"] + smoothing * global_mean) / (fold_map["count"] + smoothing)
        oof_enc[val_idx] = train_df.iloc[val_idx][col].map(fold_map["smooth"]).fillna(global_mean)
    full_map = train_df.groupby(col)[target].agg(["mean", "count"])
    full_map["smooth"] = (full_map["count"] * full_map["mean"] + smoothing * global_mean) / (full_map["count"] + smoothing)
    return oof_enc, test_df[col].map(full_map["smooth"]).fillna(global_mean).values

print("Target encoding...")
for geo_col in ["geohash", "geo_prefix4", "geo_prefix5"]:
    train[f"{geo_col}_te"], test[f"{geo_col}_te"] = target_encode(train, test, geo_col, TARGET)

train["geo_te_x_time"] = train["geohash_te"] * train["time_minutes"]
test["geo_te_x_time"]  = test["geohash_te"]  * test["time_minutes"]

FEATURES = [
    "day", "time_minutes", "hour", "minute",
    "hour_sin", "hour_cos",
    "NumberofLanes", "large_vehicles_flag", "has_landmark",
    "road_type_ord", "weather_severity", "Temperature",
    "lanes_x_time", "temp_x_weather", "geo_te_x_time",
    "lat", "lon",
    "geohash_te", "geo_prefix4_te", "geo_prefix5_te",
    "temperature_missing", "roadtype_missing", "weather_missing",
]

X_train = train[FEATURES]
y_train = np.log1p(train[TARGET])
X_test  = test[FEATURES]
print(f"Features: {len(FEATURES)}  |  X_train: {X_train.shape}  X_test: {X_test.shape}")

# ─────────────────────────────────────────────
# 3. OOF CV — NO early stopping (consistent with retrain)
# ─────────────────────────────────────────────
def oof_cv_rmse(params):
    kf  = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X_train))
    for tr_idx, val_idx in kf.split(X_train):
        m = lgb.LGBMRegressor(**params, random_state=SEED, verbose=-1, n_jobs=-1)
        m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        oof[val_idx] = m.predict(X_train.iloc[val_idx])
    return np.sqrt(mean_squared_error(np.expm1(y_train), np.expm1(oof)))

# ─────────────────────────────────────────────
# 4. OPTUNA OBJECTIVE — tighter search around baseline
# ─────────────────────────────────────────────
def objective(trial):
    params = {
        # Fix n_estimators high; let lr + regularization do the work
        "n_estimators":      2000,
        "learning_rate":     trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
        # num_leaves centered around baseline 127
        "num_leaves":        trial.suggest_int("num_leaves", 63, 255),
        "max_depth":         trial.suggest_int("max_depth", 6, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "subsample":         trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
        "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 0.5),
        "subsample_freq":    1,
    }
    rmse = oof_cv_rmse(params)
    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("oof_rmse", rmse)
    return rmse

# ─────────────────────────────────────────────
# 5. RUN OPTUNA — seed with baseline params
# ─────────────────────────────────────────────
print(f"\nRunning Optuna ({N_TRIALS} trials × {N_FOLDS} folds)...")
print(f"Baseline to beat: {BASELINE_RMSE}\n")

with mlflow.start_run(run_name="optuna_lgbm_v2"):
    sampler = TPESampler(seed=SEED)
    study   = optuna.create_study(direction="minimize", sampler=sampler)

    # Warm-start: tell Optuna the baseline params so it searches around them
    study.enqueue_trial({
        "learning_rate":     0.05,
        "num_leaves":        127,
        "max_depth":         8,
        "min_child_samples": 20,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "reg_alpha":         0.1,
        "reg_lambda":        0.1,
        "min_split_gain":    0.0,
    })

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best_params = study.best_params
    best_params["n_estimators"]   = 2000
    best_params["subsample_freq"] = 1
    best_rmse = study.best_value

    print(f"\n✅ Best OOF RMSE : {best_rmse:.5f}  (baseline: {BASELINE_RMSE:.5f})")
    print(f"   Best params   : {best_params}")
    mlflow.log_params(best_params)
    mlflow.log_metric("best_oof_rmse", best_rmse)

# ─────────────────────────────────────────────
# 6. RETRAIN ON FULL TRAIN DATA
# ─────────────────────────────────────────────
print("\nRetraining final model on FULL train data...")
final_model = lgb.LGBMRegressor(**best_params, random_state=SEED, verbose=-1, n_jobs=-1)
final_model.fit(X_train, y_train)

# ─────────────────────────────────────────────
# 7. SAVE MODEL + PARAMS
# ─────────────────────────────────────────────
model_path  = f"{MODEL_DIR}/lightgbm_final.pkl"
params_path = f"{MODEL_DIR}/lightgbm_best_params.json"

with open(model_path, "wb") as f:
    pickle.dump(final_model, f)
with open(params_path, "w") as f:
    json.dump({"best_params": best_params, "best_oof_rmse": best_rmse}, f, indent=2)

print(f"✅ Model saved  : {model_path}")
print(f"✅ Params saved : {params_path}")

# ─────────────────────────────────────────────
# 8. SUBMISSION
# ─────────────────────────────────────────────
test_preds = np.clip(np.expm1(final_model.predict(X_test)), 0, 1)
sub = pd.DataFrame({"Index": test["Index"].values, "demand": test_preds})
sub_path = f"{OUT_DIR}/submission_tuned_LightGBM_v2.csv"
sub.to_csv(sub_path, index=False)

print(f"✅ Submission   : {sub_path}")
print(f"   Rows: {len(sub)}  demand range: [{test_preds.min():.5f}, {test_preds.max():.5f}]")

improvement = (BASELINE_RMSE - best_rmse) / BASELINE_RMSE * 100
print(f"\n{'='*50}")
print(f"Baseline RMSE : {BASELINE_RMSE:.5f}")
print(f"Tuned RMSE    : {best_rmse:.5f}")
print(f"Improvement   : {improvement:+.2f}%")
print(f"{'='*50}")
print(f"\nmlflow ui  →  http://127.0.0.1:5000")