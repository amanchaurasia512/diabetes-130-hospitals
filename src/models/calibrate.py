import json
import joblib
import numpy as np
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss

from src.models.train import (
    load_model_ready_data,
    split_features_target_groups,
    patient_grouped_split,
)
from src.evaluation import evaluate_thresholds

MODELS_DIR = Path("Models")


def load_v2_model():
    model = joblib.load(MODELS_DIR / "tuned_balanced_xgb_pipeline_v2.joblib")
    print("[1/6] Loaded tuned_balanced_xgb_pipeline_v2.joblib")
    return model


def rebuild_test_split():
    """
    Reuses train.py's own functions so this produces the EXACT same
    X_test/y_test the model was evaluated on during training — same
    random_state=42, same GroupShuffleSplit — rather than risking a
    slightly different split from re-implementing it separately.
    """
    df = load_model_ready_data()
    X, y, groups = split_features_target_groups(df)
    X_train, X_test, y_train, y_test = patient_grouped_split(X, y, groups)
    print(f"[2/6] Rebuilt test split — X_test shape: {X_test.shape}")
    return X_test, y_test, groups


def carve_calibration_holdout(X_test, y_test, groups, test_idx_groups):
    calib_split = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    calib_idx, holdout_idx = next(calib_split.split(X_test, y_test, groups=test_idx_groups))

    X_calib, X_holdout = X_test.iloc[calib_idx].copy(), X_test.iloc[holdout_idx].copy()
    y_calib, y_holdout = y_test.iloc[calib_idx].copy(), y_test.iloc[holdout_idx].copy()

    overlap = set(test_idx_groups.iloc[calib_idx]) & set(test_idx_groups.iloc[holdout_idx])
    print(f"[3/6] Calibration set: {len(X_calib):,} | Holdout: {len(X_holdout):,} | "
          f"Patient overlap: {len(overlap)} (expected: 0)")
    assert len(overlap) == 0

    return X_calib, X_holdout, y_calib, y_holdout


def calibrate_model(model, X_calib, y_calib, X_holdout, y_holdout):
    raw_probs = model.predict_proba(X_holdout)[:, 1]
    raw_brier = brier_score_loss(y_holdout, raw_probs)

    calibrated_model = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    calibrated_model.fit(X_calib, y_calib)

    calibrated_probs = calibrated_model.predict_proba(X_holdout)[:, 1]
    calibrated_brier = brier_score_loss(y_holdout, calibrated_probs)

    print(f"[4/6] Brier score — raw: {raw_brier:.4f} | calibrated: {calibrated_brier:.4f}")
    return calibrated_model, calibrated_probs


def find_threshold(y_holdout, calibrated_probs, cost_ratio=8):
    results = evaluate_thresholds(
        y_true=y_holdout, y_prob=calibrated_probs,
        thresholds=np.arange(0.02, 0.95, 0.01)
    )
    results["Total_Cost"] = cost_ratio * results["FN"] + 1 * results["FP"]
    best = results.loc[results["Total_Cost"].idxmin()]

    print(f"[5/6] Best threshold at {cost_ratio}:1 cost ratio: {best['Threshold']:.2f} "
          f"(Recall: {best['Recall']:.4f})")
    return float(best["Threshold"]), float(best["Recall"])


def calibrate():
    model = load_v2_model()
    X_test, y_test, groups = rebuild_test_split()
    test_idx_groups = groups.loc[X_test.index]

    X_calib, X_holdout, y_calib, y_holdout = carve_calibration_holdout(
        X_test, y_test, groups, test_idx_groups
    )
    calibrated_model, calibrated_probs = calibrate_model(model, X_calib, y_calib, X_holdout, y_holdout)

    # NEW — check recall at the original model's threshold before deciding on v2's own
    check_recall_at_original_threshold(y_holdout, calibrated_probs, original_threshold=0.09)

    threshold, recall = find_threshold(y_holdout, calibrated_probs, cost_ratio=8)

       
    model_path = MODELS_DIR / "calibrated_xgb_pipeline_v2.joblib"
    joblib.dump(calibrated_model, model_path)
    
     

    metadata = {
        "model_file": model_path.name,
        "base_model_file": "tuned_balanced_xgb_pipeline_v2.joblib",
        "calibration_method": "isotonic",
        "decision_threshold": threshold,
        "cost_ratio_fn_to_fp": 8,
        "recall_at_threshold": recall,
        "expected_features": list(X_test.columns),
        "target_column": "readmitted_30d",
        "notes": "Trained via src/models/train.py pipeline (Postgres-sourced), "
                 "not the original notebook. Compared against original raw model "
                 "before calibration: AUC 0.6690 vs 0.668, Brier 0.2119 vs 0.2142."
    }
    metadata_path = MODELS_DIR / "calibrated_xgb_metadata_v2.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    
    print(f"[6/6] Saved {model_path.name} and {metadata_path.name}")
    return calibrated_model, metadata
    


from src.evaluation import evaluate_model
def check_recall_at_original_threshold(y_holdout, calibrated_probs, original_threshold=0.09):
    """
    Isolates whether the recall drop vs. the original model is caused by
    the threshold moving (0.09 -> 0.11), or something more concerning
    with the model itself.
    """
    result = evaluate_model(
        model_name="v2 at original threshold",
        y_true=y_holdout,
        y_prob=calibrated_probs,
        threshold=original_threshold
    )
    print(f"      Recall at original threshold ({original_threshold}): "
          f"{result['Recall']:.4f} (original model's recall: 0.6730)")
    return result

if __name__ == "__main__":
    calibrate()


    
