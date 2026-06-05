import pandas as pd

lgb = pd.read_csv("submissions/submission_100.csv")
cat = pd.read_csv("submissions/submission_catboost.csv")

blend = lgb.copy()

blend["demand"] = (
    0.9 * lgb["demand"]
    + 0.1 * cat["demand"]
)

blend.to_csv(
    "submissions/submission_blend_90_10.csv",
    index=False
)