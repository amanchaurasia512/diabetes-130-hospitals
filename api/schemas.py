from typing import List
from pydantic import BaseModel, Field


class EncounterInput(BaseModel):
    # --- Numeric features (13) ---
    time_in_hospital: int = Field(..., ge=1, le=14)
    num_lab_procedures: int = Field(..., ge=0)
    num_procedures: int = Field(..., ge=0)
    num_medications: int = Field(..., ge=0)
    number_outpatient: int = Field(..., ge=0)
    number_emergency: int = Field(..., ge=0)
    number_inpatient: int = Field(..., ge=0)
    number_diagnoses: int = Field(..., ge=0)
    num_medications_active: int = Field(..., ge=0)
    num_medications_up: int = Field(..., ge=0)
    num_medications_down: int = Field(..., ge=0)
    num_medications_steady: int = Field(..., ge=0)
    weight_documented: int = Field(..., ge=0, le=1)

    # --- Categorical features (26) ---
    race: str
    gender: str
    age: str
    payer_code: str
    medical_specialty: str
    max_glu_serum: str
    A1Cresult: str
    metformin: str
    repaglinide: str
    nateglinide: str
    glimepiride: str
    glipizide: str
    glyburide: str
    pioglitazone: str
    rosiglitazone: str
    insulin: str
    glyburide_metformin: str = Field(..., alias="glyburide-metformin")
    acarbose: str
    change: str
    diabetesMed: str
    admission_type: str
    discharge_disposition: str
    admission_source: str
    diag_1_group: str
    diag_2_group: str
    diag_3_group: str

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    readmission_probability: float
    risk_flag: bool
    decision_threshold: float
    model_version: str


class BatchPredictionRequest(BaseModel):
    encounters: List[EncounterInput]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    