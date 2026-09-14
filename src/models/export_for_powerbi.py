import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.inspection import permutation_importance

from src.db.connection import get_engine
from src.models.train import load_model_ready_data, split_features_target_groups, patient_grouped_split
from src.evaluation import evaluate_thresholds

MODELS_DIR = Path("Models")


def load_promoted_model():
    model = joblib.load(MODELS_DIR / "calibrated_xgb_pipeline_v2.joblib")
    with open(MODELS_DIR / "calibrated_xgb_metadata_v2.json") as f:
        metadata = json.load(f)
    print(f"[1/6] Loaded promoted model — threshold: {metadata['decision_threshold']}")
    return model, metadata


def rebuild_holdout():
    """Same holdout used for calibration — patient-grouped, deterministic."""
    df = load_model_ready_data()
    X, y, groups = split_features_target_groups(df)
    X_train, X_test, y_train, y_test = patient_grouped_split(X, y, groups)

    calib_split = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    test_idx_groups = groups.loc[X_test.index]
    calib_idx, holdout_idx = next(calib_split.split(X_test, y_test, groups=test_idx_groups))

    X_holdout = X_test.iloc[holdout_idx].copy()
    y_holdout = y_test.iloc[holdout_idx].copy()

    # Bring back encounter_id/patient_nbr for the export (dropped for modeling in train.py)
    id_cols = df.loc[X_holdout.index, ["encounter_id", "patient_nbr"]]

    print(f"[2/6] Rebuilt holdout — shape: {X_holdout.shape}")
    return X_holdout, y_holdout, id_cols


def build_predictions_export(model, metadata, X_holdout, y_holdout, id_cols):
    probs = model.predict_proba(X_holdout)[:, 1]
    threshold = metadata["decision_threshold"]
    preds = (probs >= threshold).astype(int)
    y_true = y_holdout.to_numpy()

    # FN/TP/FP/TN label, per row
    outcome_label = np.select(
        [
            (y_true == 1) & (preds == 1),
            (y_true == 1) & (preds == 0),
            (y_true == 0) & (preds == 1),
            (y_true == 0) & (preds == 0),
        ],
        ["TP", "FN", "FP", "TN"],
        default="Unknown" 
    )

    is_obstetric = (
        (X_holdout["diag_1_group"] == "Pregnancy & childbirth") |
        (X_holdout["diag_2_group"] == "Pregnancy & childbirth") |
        (X_holdout["diag_3_group"] == "Pregnancy & childbirth")
    ).astype(int)

    export_df = X_holdout.copy()
    export_df["encounter_id"] = id_cols["encounter_id"].values
    export_df["patient_nbr"] = id_cols["patient_nbr"].values
    export_df["actual_readmitted_30d"] = y_true
    export_df["predicted_probability"] = probs.round(4)
    export_df["risk_flag"] = preds
    export_df["outcome_label"] = outcome_label
    export_df["is_obstetric_related"] = is_obstetric.values

    print(f"[3/6] Built predictions export — shape: {export_df.shape} | "
          f"outcome distribution: {pd.Series(outcome_label).value_counts().to_dict()}")
    return export_df


def build_feature_importance_export(model, X_holdout, y_holdout):
    perm_result = permutation_importance(
        model, X_holdout, y_holdout,
        n_repeats=10, random_state=42, scoring="average_precision", n_jobs=-1
    )
    importance_df = pd.DataFrame({
        "feature": X_holdout.columns,
        "importance": perm_result.importances_mean
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    importance_df["rank"] = importance_df.index + 1

    print(f"[4/6] Built feature importance export — top feature: {importance_df.iloc[0]['feature']}")
    return importance_df


def build_threshold_sensitivity_export(model, X_holdout, y_holdout):
    probs = model.predict_proba(X_holdout)[:, 1]
    cost_ratios = [2, 3, 5, 8, 10, 15]
    rows = []

    for ratio in cost_ratios:
        results = evaluate_thresholds(
            y_true=y_holdout, y_prob=probs,
            thresholds=np.arange(0.02, 0.95, 0.01)
        )
        results["Total_Cost"] = ratio * results["FN"] + 1 * results["FP"]
        best = results.loc[results["Total_Cost"].idxmin()]
        rows.append({
            "cost_ratio": ratio,
            "best_threshold": best["Threshold"],
            "recall": best["Recall"],
            "precision": best["Precision"],
            "false_negatives": int(best["FN"]),
            "false_positives": int(best["FP"])
        })

    sensitivity_df = pd.DataFrame(rows)
    print(f"[5/6] Built threshold sensitivity export — {len(sensitivity_df)} cost ratios")
    return sensitivity_df


def export_all():
    model, metadata = load_promoted_model()
    X_holdout, y_holdout, id_cols = rebuild_holdout()

    predictions_df = build_predictions_export(model, metadata, X_holdout, y_holdout, id_cols)
    importance_df = build_feature_importance_export(model, X_holdout, y_holdout)
    sensitivity_df = build_threshold_sensitivity_export(model, X_holdout, y_holdout)

    engine = get_engine()
    predictions_df.to_sql("predictions_export", engine, if_exists="replace", index=False)
    importance_df.to_sql("feature_importance_export", engine, if_exists="replace", index=False)
    sensitivity_df.to_sql("threshold_sensitivity_export", engine, if_exists="replace", index=False)

    print(f"[6/6] Wrote 3 tables to Postgres: predictions_export ({len(predictions_df)} rows), "
          f"feature_importance_export ({len(importance_df)} rows), "
          f"threshold_sensitivity_export ({len(sensitivity_df)} rows)")


if __name__ == "__main__":
    export_all()