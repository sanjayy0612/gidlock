import os
import json
import pickle
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ==================================================
# CONFIG
# ==================================================
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"

OUT_DIR = "submissions"
MODEL_DIR = "models"

SEED = 42
N_FOLDS = 5

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ==================================================
# LOAD
# ==================================================
print("Loading data...")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print(train.shape)
print(test.shape)

# ==================================================
# BASIC FEATURE ENGINEERING ONLY
# ==================================================
def engineer(df):
    df = df.copy()

    ts = df["timestamp"].str.split(":", expand=True).astype(int)

    df["hour"] = ts[0]
    df["minute"] = ts[1]

    df["time_minutes"] = (
        df["hour"] * 60 +
        df["minute"]
    )

    return df

train = engineer(train)
test = engineer(test)

# ==================================================
# MISSING VALUES
# ==================================================
train["RoadType"] = train["RoadType"].fillna("Unknown")
test["RoadType"] = test["RoadType"].fillna("Unknown")

train["Weather"] = train["Weather"].fillna("Unknown")
test["Weather"] = test["Weather"].fillna("Unknown")

temp_median = train["Temperature"].median()

train["Temperature"] = train["Temperature"].fillna(temp_median)
test["Temperature"] = test["Temperature"].fillna(temp_median)

# ==================================================
# FEATURES
# ==================================================
TARGET = "demand"

FEATURES = [
    "geohash",
    "day",
    "hour",
    "minute",
    "time_minutes",
    "RoadType",
    "NumberofLanes",
    "LargeVehicles",
    "Landmarks",
    "Temperature",
    "Weather",
]

CAT_FEATURES = [
    "geohash",
    "RoadType",
    "LargeVehicles",
    "Landmarks",
    "Weather",
]

X = train[FEATURES]
y = np.log1p(train[TARGET])

X_test = test[FEATURES]

# ==================================================
# CV
# ==================================================
print("\nRunning CatBoost CV...")

kf = KFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=SEED
)

oof = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):

    X_tr = X.iloc[tr_idx]
    y_tr = y.iloc[tr_idx]

    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]

    train_pool = Pool(
        X_tr,
        y_tr,
        cat_features=CAT_FEATURES
    )

    val_pool = Pool(
        X_val,
        y_val,
        cat_features=CAT_FEATURES
    )

    test_pool = Pool(
        X_test,
        cat_features=CAT_FEATURES
    )

    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=SEED,
        verbose=200
    )

    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True
    )

    oof[val_idx] = model.predict(val_pool)

    test_preds += (
        model.predict(test_pool) /
        N_FOLDS
    )

    fold_rmse = np.sqrt(
        mean_squared_error(
            np.expm1(y_val),
            np.expm1(oof[val_idx])
        )
    )

    print(
        f"Fold {fold}/{N_FOLDS} "
        f"RMSE={fold_rmse:.5f}"
    )

# ==================================================
# METRICS
# ==================================================
oof_rmse = np.sqrt(
    mean_squared_error(
        np.expm1(y),
        np.expm1(oof)
    )
)

oof_mae = mean_absolute_error(
    np.expm1(y),
    np.expm1(oof)
)

oof_r2 = r2_score(
    np.expm1(y),
    np.expm1(oof)
)

print("\n============================")
print("CATBOOST RESULTS")
print("============================")
print(f"OOF RMSE : {oof_rmse:.5f}")
print(f"OOF MAE  : {oof_mae:.5f}")
print(f"OOF R²   : {oof_r2:.5f}")

# ==================================================
# SAVE MODEL
# ==================================================
final_pool = Pool(
    X,
    y,
    cat_features=CAT_FEATURES
)

final_model = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.03,
    depth=8,
    loss_function="RMSE",
    random_seed=SEED,
    verbose=200
)

final_model.fit(final_pool)

model_path = (
    "models/catboost_baseline.pkl"
)

with open(model_path, "wb") as f:
    pickle.dump(final_model, f)

# ==================================================
# SUBMISSION
# ==================================================
preds = np.clip(
    np.expm1(test_preds),
    0,
    1
)

sub = pd.DataFrame({
    "Index": test["Index"],
    "demand": preds
})

sub_path = (
    "submissions/submission_catboost.csv"
)

sub.to_csv(
    sub_path,
    index=False
)

print("\nSaved:")
print(model_path)
print(sub_path)

meta = {
    "oof_rmse": float(oof_rmse),
    "oof_mae": float(oof_mae),
    "oof_r2": float(oof_r2),
    "features": FEATURES,
    "cat_features": CAT_FEATURES
}

with open(
    "models/catboost_baseline_meta.json",
    "w"
) as f:
    json.dump(meta, f, indent=2)