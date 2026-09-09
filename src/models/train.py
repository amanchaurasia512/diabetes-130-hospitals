import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier

from src.db.connection import get_engine

MODELS_DIR = Path("Models")


def load_model_ready_data():
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM model_ready_data", engine)
    print(f"[1/7] Loaded {len(df):,} rows, {len(df.columns)} columns from Postgres")
    return df


def split_features_target_groups(df):
    X = df.drop(columns=["encounter_id", "patient_nbr", "readmitted_30d"])
    y = df["readmitted_30d"]
    groups = df["patient_nbr"]

    print(f"[2/7] X shape: {X.shape} (expected: (100114, 39)) | "
          f"target distribution: {dict(y.value_counts())}")
    assert X.shape[1] == 39, f"Expected 39 features, got {X.shape[1]}"

    return X, y, groups


def identify_feature_types(X):
    numerical_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    print(f"[3/7] Numerical features: {len(numerical_features)} | "
          f"Categorical features: {len(categorical_features)}")

    return numerical_features, categorical_features


def patient_grouped_split(X, y, groups):
    group_split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(group_split.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

    # Verify no patient appears in both sets
    overlap = set(groups_train) & set(groups_test)
    print(f"[4/7] Train: {len(X_train):,} rows | Test: {len(X_test):,} rows | "
          f"Patient overlap: {len(overlap)} (expected: 0)")
    assert len(overlap) == 0, "Patient leakage detected between train and test!"

    return X_train, X_test, y_train, y_test


def build_preprocessor(numerical_features, categorical_features):
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features)
        ],
        remainder="drop"
    )

    print(f"[5/7] Preprocessor built")
    return preprocessor


def build_and_train_model(preprocessor, X_train, y_train):
    negative_class = (y_train == 0).sum()
    positive_class = (y_train == 1).sum()
    scale_pos_weight = negative_class / positive_class

    print(f"[6/7] scale_pos_weight: {scale_pos_weight:.4f} "
          f"(neg={negative_class:,}, pos={positive_class:,})")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)
    print("      Model fitted successfully")

    return pipeline


def train():
    df = load_model_ready_data()
    X, y, groups = split_features_target_groups(df)
    numerical_features, categorical_features = identify_feature_types(X)
    X_train, X_test, y_train, y_test = patient_grouped_split(X, y, groups)
    preprocessor = build_preprocessor(numerical_features, categorical_features)
    pipeline = build_and_train_model(preprocessor, X_train, y_train)

    # Quick sanity check against the known notebook result before saving anything
    from sklearn.metrics import roc_auc_score
    test_probs = pipeline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_probs)
    print(f"[7/7] Test ROC-AUC: {test_auc:.4f} (notebook 04 reference: 0.668)")

    MODELS_DIR.mkdir(exist_ok=True)
    output_path = MODELS_DIR / "tuned_balanced_xgb_pipeline_v2.joblib"
    joblib.dump(pipeline, output_path)
    print(f"\n✓ Saved to {output_path} (kept separate from the original "
          f"tuned_balanced_xgb_pipeline.joblib for comparison)")

    return pipeline


if __name__ == "__main__":
    train()