"""
02 - Multi-output regression benchmark (Table 2, learned models).

Seven algorithms, multi-output regression of N / P2O5 / K2O amounts, repeated
over ten independent seeds (0-9). Under each seed a fresh 80:20 split (stratified
by crop), a fresh multiplicative-Gaussian noise realisation on the targets
(sigma = 10%), and a fresh model initialisation are drawn.

    F_tilde_i = max(0, F_i * eps),  eps ~ N(1, sigma^2)

Scale-sensitive models (SVR, k-NN, MLP) are standardised within each split with
the scaler fitted on the training partition only.

Outputs: results_multiseed.csv (per model / seed / sigma R2, RMSE, MAE, nRMSE, rel.err.)
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

FEATURES = ["crop_encoded", "N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGETS = ["fert_N_kg_ha", "fert_P2O5_kg_ha", "fert_K2O_kg_ha"]
SCALE_SENSITIVE = {"SVR", "k-NN", "MLP"}
SEEDS = range(10)
SIGMAS = [0.05, 0.10, 0.15]


def make_models(seed):
    return {
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1),
        "XGBoost": MultiOutputRegressor(XGBRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            random_state=seed, n_jobs=-1, verbosity=0)),
        "LightGBM": MultiOutputRegressor(LGBMRegressor(
            n_estimators=200, learning_rate=0.1, random_state=seed, n_jobs=-1, verbose=-1)),
        "CatBoost": MultiOutputRegressor(CatBoostRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            random_state=seed, verbose=0)),
        "SVR": MultiOutputRegressor(SVR(kernel="rbf", C=100, gamma="scale")),
        "k-NN": KNeighborsRegressor(n_neighbors=5, weights="uniform"),
        "MLP": MLPRegressor(hidden_layer_sizes=(128, 64), learning_rate_init=0.001,
                            max_iter=500, early_stopping=True, n_iter_no_change=10,
                            random_state=seed),
    }


def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mean_dose = y_true.mean()
    return r2, rmse, mae, 100 * rmse / mean_dose, 100 * mae / mean_dose


def run():
    df = pd.read_csv("df_prepared.csv")
    X_all = df[FEATURES].values
    rows = []
    for sigma in SIGMAS:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            y = df[TARGETS].values.copy()
            y = np.clip(y * rng.normal(1.0, sigma, size=y.shape), 0, None)
            Xtr, Xte, ytr, yte = train_test_split(
                X_all, y, test_size=0.2, random_state=seed, stratify=df["crop"])
            for name, model in make_models(seed).items():
                if name in SCALE_SENSITIVE:
                    sc = StandardScaler().fit(Xtr)
                    Xtr_m, Xte_m = sc.transform(Xtr), sc.transform(Xte)
                else:
                    Xtr_m, Xte_m = Xtr, Xte
                model.fit(Xtr_m, ytr)
                pred = model.predict(Xte_m)
                r2, rmse, mae, nrmse, rel = metrics(yte, pred)
                rows.append(dict(Model=name, sigma=sigma, seed=seed,
                                 R2=r2, RMSE=rmse, MAE=mae, nRMSE=nrmse, RelErr=rel))
            print(f"sigma={sigma:.2f} seed={seed} done")
    out = pd.DataFrame(rows)
    out.to_csv("results_multiseed.csv", index=False)

    print("\n=== Table 2 (sigma = 10%, mean +/- SD over 10 seeds) ===")
    g = out[out.sigma == 0.10].groupby("Model")
    summ = g[["R2", "RMSE", "MAE", "nRMSE", "RelErr"]].agg(["mean", "std"])
    print(summ.sort_values(("R2", "mean"), ascending=False).round(4))


if __name__ == "__main__":
    run()
