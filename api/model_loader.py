import joblib
import json
from pathlib import Path
from functools import lru_cache

MODELS_DIR = Path(__file__).resolve().parent.parent / "Models"


@lru_cache
def load_model_and_metadata():
    model = joblib.load(MODELS_DIR / "calibrated_xgb_pipeline.joblib")
    with open(MODELS_DIR / "calibrated_xgb_metadata.json") as f:
        metadata = json.load(f)
    return model, metadata