"""
05 - Per-crop performance (Table 4).

Random Forest, sigma = 10%, ten seeds. Reports per-crop RMSE, MAE and nRMSE
(nRMSE as a percentage of that crop's mean recommended dose).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

FEATURES = ["crop_encoded", "N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGETS = ["fert_N_kg_ha", "fert_P2O5_kg_ha", "fert_K2O_kg_ha"]
SEEDS = range(10)
SIGMA = 0.10


def run():
    df = pd.read_csv("df_prepared.csv")
    acc = {c: {"rmse": [], "mae": [], "nrmse": []} for c in df["crop"].unique()}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        y = np.clip(df[TARGETS].values * rng.normal(1, SIGMA, (len(df), 3)), 0, None)
        idx = np.arange(len(df))
        tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=df["crop"])
        mdl = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
        mdl.fit(df[FEATURES].values[tr], y[tr])
        pred = mdl.predict(df[FEATURES].values[te])
        crop_te = df["crop"].values[te]
        for c in np.unique(crop_te):
            mask = crop_te == c
            yt, yp = y[te][mask], pred[mask]
            rmse = np.sqrt(mean_squared_error(yt, yp))
            mae = mean_absolute_error(yt, yp)
            acc[c]["rmse"].append(rmse)
            acc[c]["mae"].append(mae)
            acc[c]["nrmse"].append(100 * rmse / yt.mean())

    rows = [dict(Crop=c, RMSE=np.mean(v["rmse"]), MAE=np.mean(v["mae"]),
                 nRMSE=np.mean(v["nrmse"])) for c, v in acc.items()]
    out = pd.DataFrame(rows).sort_values("nRMSE").round(2)
    out.to_csv("table_per_crop.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    run()
