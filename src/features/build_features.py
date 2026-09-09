import pandas as pd
from src.db.connection import get_engine

RARE_MEDICATION_FEATURES = [
    "acetohexamide", "glimepiride-pioglitazone", "metformin-pioglitazone",
    "metformin-rosiglitazone", "troglitazone", "glipizide-metformin",
    "tolbutamide", "miglitol", "tolazamide", "chlorpropamide"
]

REDUNDANT_ENGINEERED_FEATURES = [
    "medication_changed", "diabetes_medication_used",
    "max_glu_serum_documented", "A1C_documented",
    "max_glu_serum_level", "A1C_level"
]

RAW_DIAGNOSIS_FEATURES = ["diag_1", "diag_2", "diag_3"]
WEIGHT_FEATURES_TO_DROP = ["weight"]

EXCLUDED_COLUMNS = [
    "encounter_id", "patient_nbr", "readmitted", "readmitted_30d",
    "admission_type_id", "discharge_disposition_id", "admission_source_id"
    # NOTE: original notebook also excluded 'death_related_disposition',
    # which doesn't exist in this pipeline's cleaned_encounters table —
    # it was a bookkeeping-only column, never part of the final 39 features,
    # so omitting it here doesn't change the resulting feature set.
]

TARGET_COLUMN = "readmitted_30d"


def load_cleaned_from_postgres():
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM cleaned_encounters", engine)
    print(f"Loaded {len(df):,} cleaned rows, {len(df.columns)} columns from Postgres")
    return df


def get_provisional_features(df):
    provisional = [c for c in df.columns if c not in EXCLUDED_COLUMNS]
    print(f"[1/4] Provisional features: {len(provisional)} (expected: 59)")
    assert len(provisional) == 59, \
        f"Provisional feature count mismatch — expected 59, got {len(provisional)}"
    return provisional


def get_features_to_drop():
    features_to_drop = (
        RARE_MEDICATION_FEATURES
        + REDUNDANT_ENGINEERED_FEATURES
        + RAW_DIAGNOSIS_FEATURES
        + WEIGHT_FEATURES_TO_DROP
    )
    print(f"[2/4] Features to drop: {len(features_to_drop)} (expected: 20)")
    assert len(features_to_drop) == 20, \
        f"Drop-list count mismatch — expected 20, got {len(features_to_drop)}"
    return features_to_drop


def select_final_features(df, provisional_features, features_to_drop):
    selected = [f for f in provisional_features if f not in features_to_drop]
    print(f"[3/4] Selected features: {len(selected)} (expected: 39)")
    assert len(selected) == 39, \
        f"Final feature count mismatch — expected 39, got {len(selected)}"

    # Target leakage guard
    assert TARGET_COLUMN not in selected, "Target leakage — target found in feature list!"

    return selected


def build_features():
    df = load_cleaned_from_postgres()

    provisional_features = get_provisional_features(df)
    features_to_drop = get_features_to_drop()
    selected_features = select_final_features(df, provisional_features, features_to_drop)

    # Keep encounter_id + patient_nbr alongside the features, in the SAME
    # table — avoids the row-order-dependent CSV join that caused issues
    # in the original notebook pipeline.
    final_columns = ["encounter_id", "patient_nbr"] + selected_features + [TARGET_COLUMN]
    df_final = df[final_columns].copy()

    print(f"[4/4] Final feature table shape: {df_final.shape} "
          f"(expected: (100114, 42) — 39 features + target + 2 ID columns)")

    engine = get_engine()
    df_final.to_sql("model_ready_data", engine, if_exists="replace", index=False)
    print(f"\n✓ Wrote {len(df_final):,} rows to Postgres table 'model_ready_data'")

    return df_final


if __name__ == "__main__":
    build_features()