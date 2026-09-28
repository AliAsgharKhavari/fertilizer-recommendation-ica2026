"""
03 - Baseline methods (Table 2, lower block).

Four baselines under the same 10-seed / sigma = 10% protocol:
  - Global mean       : predict the training-set mean dose
  - Crop-wise GRD     : predict each crop's mean dose (crop identity only)
  - Deterministic GRD : apply the nutrient-balance rule (Eqs. 1-2) directly
  - Linear regression : OLS on the eight inputs
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

FEATURES = ["crop_encoded", "N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGETS = ["fert_N_kg_ha", "fert_P2O5_kg_ha", "fert_K2O_kg_ha"]
SEEDS = range(10)
SIGMA = 0.10


def m(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    md = y_true.mean()
    return r2, rmse, mae, 100 * rmse / md, 100 * mae / md


def run():
    df = pd.read_csv("df_prepared.csv")
    res = {k: [] for k in ["Global mean", "Crop-wise GRD", "Deterministic GRD rule", "Linear regression"]}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        y = np.clip(df[TARGETS].values * rng.normal(1, SIGMA, (len(df), 3)), 0, None)
        idx = np.arange(len(df))
        tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=df["crop"])
        ytr, yte = y[tr], y[te]
        crop_tr, crop_te = df["crop"].values[tr], df["crop"].values[te]

        # global mean
        res["Global mean"].append(m(yte, np.tile(ytr.mean(0), (len(te), 1))))
        # crop-wise mean dose
        means = pd.DataFrame(ytr, columns=TARGETS).assign(crop=crop_tr).groupby("crop").mean()
        pred_cw = means.reindex(crop_te).values
        res["Crop-wise GRD"].append(m(yte, pred_cw))
        # deterministic GRD rule = clean (unperturbed) labels as the oracle
        res["Deterministic GRD rule"].append(m(yte, df[TARGETS].values[te]))
        # linear regression
        lr = LinearRegression().fit(df[FEATURES].values[tr], ytr)
        res["Linear regression"].append(m(yte, lr.predict(df[FEATURES].values[te])))

    print("=== Baselines (sigma = 10%, mean over 10 seeds) ===")
    for k, v in res.items():
        a = np.array(v).mean(0)
        print(f"{k:24s} R2={a[0]:.3f}  RMSE={a[1]:.2f}  MAE={a[2]:.2f}  "
              f"nRMSE={a[3]:.1f}  RelErr={a[4]:.1f}")


if __name__ == "__main__":
    run()
