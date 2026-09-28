# Machine Learning for Field-Specific Fertilizer Quantity Recommendation in Indian Staple Crops

**Paper ID 179 — ICA 2026 (4th International Conference on Agriculture-Centric Computation)**  
Springer Lecture Notes in Computer Science / CCIS series, IIT Ropar, India

**Authors:**  
Ali Asghar Khavari · Shuchita Upadhyaya · Pardeep Kumar · Monika Poriye  
Department of Computer Science and Applications, Kurukshetra University, Kurukshetra-136119, Haryana, India

---

## Overview

This repository contains the dataset, source code, and figures for the above conference paper.

The study proposes a machine learning framework for field-specific fertilizer amount recommendation for 17 Indian staple crops. A method-derived labelled dataset of 1,700 records is built using an STCR-inspired nutrient-balance rule anchored to verified ICAR/AICRP/SAU General Recommended Doses (GRD). Seven algorithms are benchmarked under a multiplicative measurement-uncertainty model, evaluated with the Friedman/Nemenyi statistical comparison protocol.

**This repository reproduces:**
- Table 1 (GRD reference doses) — in `data/fertilizer_dataset_sources_citations.csv`
- Table 2 (main results, σ = 10%) — from `src/02_regression_benchmark.py` + `src/03_baselines.py`
- Table 3 (σ sensitivity) — from `src/02_regression_benchmark.py`
- Table 4 (per-crop performance) — from `src/05_per_crop.py`
- Section 5.1 statistics — from `src/04_statistics.py`
- Figures 2 and 3 — from `src/06_figures.py`

---

## Repository structure

```
├── data/
│   ├── fertilizer_recommendation_labelled.csv   # 1,700 records, 17 crops, 8 features + labels
│   ├── fertilizer_dataset_sources_citations.csv # GRD source citations (ICAR/AICRP/SAU)
│   └── DATASET_METHODOLOGY.md                   # full derivation methodology and provenance
├── src/
│   ├── 01_data_preparation.py    # load + encode dataset → df_prepared.csv
│   ├── 02_regression_benchmark.py # 7 models × 10 seeds × 3 sigma levels → results_multiseed.csv
│   ├── 03_baselines.py           # 4 baseline methods (Table 2 lower block)
│   ├── 04_statistics.py          # Friedman test, Kendall's W, Nemenyi post-hoc, CD diagram
│   ├── 05_per_crop.py            # per-crop RMSE/MAE/nRMSE (Table 4)
│   └── 06_figures.py             # Figures 2 and 3
├── figures/
│   ├── Fig1_framework.{pdf,png}             # dataset + evaluation framework schematic
│   ├── Fig2_predicted_vs_observed.{pdf,png} # predicted vs. observed (Random Forest, σ=10%)
│   └── Fig3_CD_regression.{pdf,png}         # critical-difference diagram
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Reproducibility

### Setup

```bash
git clone https://github.com/<your-username>/fertilizer-recommendation-ica2026.git
cd fertilizer-recommendation-ica2026
pip install -r requirements.txt
```

Python 3.11+ recommended. Exact library versions used in the paper:
`scikit-learn 1.6.1 · XGBoost 3.3.0 · LightGBM 4.6.0 · CatBoost 1.2.10 · SciPy 1.16.3`

### Run the full pipeline

```bash
cd src
python 01_data_preparation.py   # produces df_prepared.csv
python 02_regression_benchmark.py  # ~10–20 min; produces results_multiseed.csv
python 03_baselines.py          # baselines (fast)
python 04_statistics.py         # requires results_multiseed.csv
python 05_per_crop.py           # per-crop table
python 06_figures.py            # requires df_prepared.csv + table_mean_ranks.csv
```

All scripts must be run from within the `src/` directory.

---

## Dataset

`fertilizer_recommendation_labelled.csv` — 1,700 records (17 crops × 100 field conditions)

| Column | Description |
|---|---|
| crop | crop name (string) |
| N, P, K | soil-available N, P, K (kg/ha) |
| temperature | mean air temperature (°C) |
| humidity | relative humidity (%) |
| ph | soil pH |
| rainfall | annual rainfall (mm) |
| fert_N_kg_ha | recommended N dose (kg/ha) — label |
| fert_P2O5_kg_ha | recommended P₂O₅ dose (kg/ha) — label |
| fert_K2O_kg_ha | recommended K₂O dose (kg/ha) — label |
| total_fertilizer_kg_ha | sum of three nutrients |
| fertilizer_type | dominant product type (categorical) |

Labels are method-derived (not field-measured). See `data/DATASET_METHODOLOGY.md` for the full nutrient-balance derivation, GRD sources, and scope/limitations.

---

## How to cite

If you use this code or dataset, please cite:

> Khavari, A. A., Upadhyaya, S., Kumar, P., & Poriye, M. (2026).
> Machine Learning for Field-Specific Fertilizer Quantity Recommendation in Indian Staple Crops.
> *Proceedings of the 4th International Conference on Agriculture-Centric Computation (ICA 2026)*, IIT Ropar.
> Springer CCIS. Paper ID 179.

---

## Note on scope

This repository contains the code and data for the ICA 2026 conference paper only. Additional analyses are part of ongoing work.

---

## Contact

Ali Asghar Khavari — dcsa2126ali@kuk.ac.in  
Kurukshetra University, Kurukshetra-136119, Haryana, India
