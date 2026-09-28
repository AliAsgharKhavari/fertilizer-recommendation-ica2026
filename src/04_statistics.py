"""
04 - Statistical comparison (Section 5.1, Fig. 3).

Friedman test over the ten seeds (blocks) at sigma = 10%, Kendall's W for ranking
consistency, and Nemenyi post-hoc pairwise comparison. Critical difference for
k = 7 algorithms and n = 10 blocks at alpha = 0.05 is CD = 2.849.
"""
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata

try:
    from scikit_posthocs import posthoc_nemenyi_friedman
    HAVE_PH = True
except Exception:
    HAVE_PH = False

MODELS = ["Random Forest", "CatBoost", "XGBoost", "LightGBM", "MLP", "k-NN", "SVR"]


def run():
    df = pd.read_csv("results_multiseed.csv")
    d = df[df.sigma == 0.10].pivot(index="seed", columns="Model", values="R2")[MODELS]

    stat, p = friedmanchisquare(*[d[m].values for m in MODELS])
    print(f"Friedman: chi2({len(MODELS)-1}) = {stat:.2f}, p = {p:.3e}")

    # ranks per seed (rank 1 = best R2)
    ranks = np.apply_along_axis(lambda r: rankdata(-r), 1, d.values)
    mean_ranks = pd.Series(ranks.mean(0), index=MODELS).sort_values()
    print("\nMean ranks (1 = best):")
    print(mean_ranks.round(2))

    # Kendall's W
    n, k = d.shape
    W = (12 * (ranks.sum(0) ** 2).sum() - 3 * n**2 * k * (k + 1) ** 2) / \
        (n**2 * k * (k**2 - 1))
    print(f"\nKendall's W = {W:.3f}")
    print("CD (k=7, n=10, alpha=0.05) = 2.849")

    mean_ranks.to_csv("table_mean_ranks.csv", header=["mean_rank"])

    if HAVE_PH:
        pv = posthoc_nemenyi_friedman(d.values)
        pv.index = pv.columns = MODELS
        print("\nNemenyi pairwise p-values:")
        print(pv.round(3))
    else:
        print("\n(install scikit-posthocs for exact Nemenyi pairwise p-values)")


if __name__ == "__main__":
    run()
