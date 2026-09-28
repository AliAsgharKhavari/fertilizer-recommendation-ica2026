"""
01 - Data preparation / encoding.

Loads the released labelled dataset and produces the model-ready frame used by
all downstream scripts. The dataset itself was derived by an STCR-inspired
nutrient-balance rule anchored to verified ICAR/AICRP/SAU General Recommended
Doses; see ../data/DATASET_METHODOLOGY.md for full provenance.

Output: df_prepared.csv (adds an integer `crop_encoded` column).
"""
import pandas as pd

DATA = "../data/fertilizer_recommendation_labelled.csv"

FEATURES = ["crop_encoded", "N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGETS = ["fert_N_kg_ha", "fert_P2O5_kg_ha", "fert_K2O_kg_ha"]


def load_prepared():
    df = pd.read_csv(DATA)
    # deterministic label-encode crop identity (alphabetical, stable)
    crops = sorted(df["crop"].unique())
    mapping = {c: i for i, c in enumerate(crops)}
    df["crop_encoded"] = df["crop"].map(mapping)
    return df


if __name__ == "__main__":
    df = load_prepared()
    df.to_csv("df_prepared.csv", index=False)
    print(f"Records: {len(df)}  Crops: {df['crop'].nunique()}")
    print(f"Features: {FEATURES}")
    print(f"Targets:  {TARGETS}")
    print("Saved df_prepared.csv")
