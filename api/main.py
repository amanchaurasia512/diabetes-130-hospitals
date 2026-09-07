import pandas as pd
from fastapi import FastAPI, HTTPException
from sklearn.exceptions import NotFittedError

from api.schemas import EncounterInput, PredictionResponse
from api.model_loader import load_model_and_metadata

app = FastAPI(
    title="Hospital Readmission Risk API",
    description="Predicts 30-day readmission risk for diabetic inpatient encounters. "
                 "Decision-support only — not a diagnostic tool.",
    version="1.0.0"
)


@app.get("/health")
def health_check():
    model, metadata = load_model_and_metadata()
    return {"status": "ok", "model_file": metadata["model_file"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(encounter: EncounterInput):
    model, metadata = load_model_and_metadata()

    # Build a single-row DataFrame with columns in the exact order the pipeline expects
    input_dict = encounter.model_dump(by_alias=True)
    row = pd.DataFrame([input_dict])[metadata["expected_features"]]

    try:
        probability = float(model.predict_proba(row)[:, 1][0])
    except ValueError as e:
        # Most likely cause: an unrecognized category value not seen during training
        raise HTTPException(
            status_code=422,
            detail=f"Could not score this encounter — likely an unrecognized "
                    f"category value. Details: {e}"
        )

    threshold = metadata["decision_threshold"]

    return PredictionResponse(
        readmission_probability=round(probability, 4),
        risk_flag=probability >= threshold,
        decision_threshold=threshold,
        model_version=metadata["model_file"]
    )