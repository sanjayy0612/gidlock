import pandas as pd

lgb = pd.read_csv("submissions/submission_best_21_features.csv")
cat = pd.read_csv("submissions/submission_catboost.csv")

blend = lgb.copy()



blend["demand"] = (
    0.62 * lgb["demand"] +
    0.38 * cat["demand"]
)
blend.to_csv(
    "submissions/submission_blend_62_38.csv",
    index=False
)