"""
06 - Figures.

Fig. 2  predicted vs. observed (Random Forest, sigma = 10%)
Fig. 3  critical-difference diagram (regression, sigma = 10%)

(Fig. 1, the framework schematic, is a hand-drawn diagram; the version used in
the paper is provided under ../figures/.)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

FEATURES = ["crop_encoded", "N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGETS = ["fert_N_kg_ha", "fert_P2O5_kg_ha", "fert_K2O_kg_ha"]

plt.rcParams.update({"savefig.dpi": 600, "savefig.bbox": "tight", "font.size": 10})


def fig_pred_vs_obs():
    df = pd.read_csv("df_prepared.csv")
    rng = np.random.default_rng(0)
    y = np.clip(df[TARGETS].values * rng.normal(1, 0.10, (len(df), 3)), 0, None)
    Xtr, Xte, ytr, yte = train_test_split(df[FEATURES].values, y,
                                          test_size=0.2, random_state=0, stratify=df["crop"])
    pred = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1).fit(Xtr, ytr).predict(Xte)
    names = ["Nitrogen (N)", "Phosphorus (P2O5)", "Potassium (K2O)"]
    cols = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for i, (ax, nm, col) in enumerate(zip(axes, names, cols)):
        ax.scatter(yte[:, i], pred[:, i], s=10, alpha=0.5, color=col, edgecolor="none")
        lim = [0, max(yte[:, i].max(), pred[:, i].max()) * 1.05]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel("Observed (kg/ha)"); ax.set_ylabel("Predicted (kg/ha)")
        ax.set_title(f"{nm}\nR2 = {r2_score(yte[:, i], pred[:, i]):.3f}")
        ax.set_xlim(lim); ax.set_ylim(lim)
    plt.suptitle("Predicted vs. observed fertilizer amounts - Random Forest (sigma = 10%)",
                 fontweight="bold")
    plt.tight_layout(); plt.savefig("Fig2_predicted_vs_observed.png"); plt.close()
    print("saved Fig2_predicted_vs_observed.png")


def cd_diagram(ranks, cd, title, fname):
    ranks = ranks.sort_values()
    models, vals = list(ranks.index), list(ranks.values)
    k, lo, hi = len(models), 1, len(models); half = (k + 1) // 2
    fig, ax = plt.subplots(figsize=(10, 0.55 * k + 2.2))
    ax.set_xlim(lo - 2, hi + 2); ax.set_ylim(-(half * 0.6) - 0.6, 1.7); ax.axis("off")
    ax.plot([lo, hi], [0, 0], "k-", lw=1.5)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [0, 0.10], "k-", lw=1); ax.text(r, 0.26, str(r), ha="center", fontsize=9)
    ax.text((lo + hi) / 2, 1.45, title, ha="center", fontsize=11, fontweight="bold")
    for depth, i in enumerate(range(half)):
        v = vals[i]; y = -0.6 * (depth + 1)
        ax.plot([v, v], [0, y], "k-", lw=0.8); ax.plot([v, lo - 1.9], [y, y], "k-", lw=0.8)
        ax.text(lo - 1.95, y, f"{models[i]} ({v:.2f})", ha="right", va="center", fontsize=9)
    for depth, i in enumerate(range(k - 1, half - 1, -1)):
        v = vals[i]; y = -0.6 * (depth + 1)
        ax.plot([v, v], [0, y], "k-", lw=0.8); ax.plot([v, hi + 1.9], [y, y], "k-", lw=0.8)
        ax.text(hi + 1.95, y, f"{models[i]} ({v:.2f})", ha="left", va="center", fontsize=9)
    ax.plot([lo, lo + cd], [1.05, 1.05], "k-", lw=2.5)
    ax.plot([lo, lo], [0.98, 1.12], "k-"); ax.plot([lo + cd, lo + cd], [0.98, 1.12], "k-")
    ax.text(lo + cd / 2, 1.22, f"CD = {cd:.2f}", ha="center", fontsize=9)
    plt.tight_layout(); plt.savefig(fname); plt.close(); print(f"saved {fname}")


if __name__ == "__main__":
    fig_pred_vs_obs()
    ranks = pd.read_csv("table_mean_ranks.csv", index_col=0).iloc[:, 0]
    cd_diagram(ranks, 2.849,
               "Critical-difference diagram - fertilizer amount (sigma = 10%)",
               "Fig3_CD_regression.png")
