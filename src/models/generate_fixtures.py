import json
import joblib
import numpy as np
from pathlib import Path

from src.models.train import load_model_ready_data, split_features_target_groups, patient_grouped_split

MODELS_DIR = Path("Models")
FIXTURES_DIR = Path("api/tests/fixtures")


def load_v2_models():
    raw_model = joblib.load(MODELS_DIR / "tuned_balanced_xgb_pipeline_v2.joblib")
    calibrated_model = joblib.load(MODELS_DIR / "calibrated_xgb_pipeline_v2.joblib")
    print("[1/5] Loaded v2 raw model and v2 calibrated model")
    return raw_model, calibrated_model


def rebuild_test_set():
    """Same split logic as train.py/calibrate.py, so this X_test is identical."""
    df = load_model_ready_data()
    X, y, groups = split_features_target_groups(df)
    X_train, X_test, y_train, y_test = patient_grouped_split(X, y, groups)
    print(f"[2/5] Rebuilt X_test — shape: {X_test.shape}")
    return X_test, y_test


def find_case_positions(raw_model, X_test, y_test, threshold=0.50):
    """
    Identifies the same four case types as notebook 05, Section 5.5:
    borderline FN, confident-miss FN, confident FP, borderline FP —
    based on the RAW model's predictions at threshold 0.50.
    """
    y_test_prob = raw_model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= threshold).astype(int)
    y_test_np = y_test.to_numpy()

    false_negative_mask = (y_test_np == 1) & (y_test_pred == 0)
    false_positive_mask = (y_test_np == 0) & (y_test_pred == 1)

    fn_positions = np.where(false_negative_mask)[0]
    fp_positions = np.where(false_positive_mask)[0]

    fn_probs = y_test_prob[fn_positions]
    fp_probs = y_test_prob[fp_positions]

    positions = {
        "borderline_fn": fn_positions[np.argmax(fn_probs)],
        "confident_miss_fn": fn_positions[np.argmin(fn_probs)],
        "confident_fp": fp_positions[np.argmax(fp_probs)],
        "borderline_fp": fp_positions[np.argmin(fp_probs)],
    }

    print(f"[3/5] Found case positions — FN count: {len(fn_positions)}, "
          f"FP count: {len(fp_positions)}")
    for name, pos in positions.items():
        print(f"      {name}: position {pos}, raw P={y_test_prob[pos]:.4f}")

    return positions, y_test_prob


def build_fixtures(positions, X_test, calibrated_model):
    fixtures = {}
    for name, pos in positions.items():
        row = X_test.iloc[[pos]]
        row_dict = row.to_dict(orient="records")[0]
        calibrated_prob = float(calibrated_model.predict_proba(row)[:, 1][0])
        fixtures[name] = {
            "input": row_dict,
            "expected_calibrated_probability": round(calibrated_prob, 4)
        }
    print(f"[4/5] Built {len(fixtures)} fixtures with calibrated probabilities")
    return fixtures


def save_fixtures(fixtures):
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIXTURES_DIR / "known_cases.json"

    with open(output_path, "w") as f:
        json.dump(fixtures, f, indent=2)

    print(f"[5/5] Saved to {output_path}")
    for name, data in fixtures.items():
        print(f"      {name}: P={data['expected_calibrated_probability']}")


def generate_fixtures():
    raw_model, calibrated_model = load_v2_models()
    X_test, y_test = rebuild_test_set()
    positions, _ = find_case_positions(raw_model, X_test, y_test)
    fixtures = build_fixtures(positions, X_test, calibrated_model)
    save_fixtures(fixtures)
    return fixtures


if __name__ == "__main__":
    generate_fixtures()