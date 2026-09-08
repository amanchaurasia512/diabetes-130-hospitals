import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

FIXTURES_PATH = Path(__file__).parent / "fixtures" / "known_cases.json"

with open(FIXTURES_PATH) as f:
    KNOWN_CASES = json.load(f)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.parametrize("case_name", KNOWN_CASES.keys())
def test_predict_matches_notebook(case_name):
    case = KNOWN_CASES[case_name]
    response = client.post("/predict", json=case["input"])

    assert response.status_code == 200, response.text
    predicted_prob = response.json()["readmission_probability"]
    expected_prob = case["expected_calibrated_probability"]

    assert predicted_prob == pytest.approx(expected_prob, abs=1e-4), (
        f"{case_name}: API returned {predicted_prob}, "
        f"notebook expected {expected_prob}"
    )


def test_predict_rejects_missing_field():
    incomplete_input = {"time_in_hospital": 5}
    response = client.post("/predict", json=incomplete_input)
    assert response.status_code == 422


def test_predict_handles_unknown_category():
    case = KNOWN_CASES["borderline_fn"]["input"].copy()
    case["medical_specialty"] = "SomeSpecialtyNotInTrainingData"
    response = client.post("/predict", json=case)
    assert response.status_code == 200