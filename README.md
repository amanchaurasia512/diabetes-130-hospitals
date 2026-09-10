# Hospital Readmission Risk Prediction System

Predicting 30-day hospital readmission risk for diabetic inpatients, end-to-end — from a raw clinical dataset through a PostgreSQL pipeline, a calibrated XGBoost model, a tested FastAPI service, and a Django web application.

> **Decision-support tool, not a diagnostic system.** Findings describe statistical association in a historical dataset, not causal clinical relationships. Nothing here should inform individual clinical decisions.

---

## Overview

Diabetic patients who are readmitted within 30 days of discharge represent a real cost and quality-of-care problem for hospitals — and in the US, are directly tied to CMS reimbursement penalties under the Hospital Readmissions Reduction Program. This project builds a model that flags which inpatient encounters are at elevated risk of a 30-day readmission, so discharge-planning teams can prioritize follow-up outreach.

Built on the [UCI Diabetes 130-US Hospitals dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) (101,766 encounters, 1999–2008) — the same clinical extract published in [Strack et al., 2014](https://doi.org/10.1155/2014/781670).

## Key Results

| Metric | Value |
|---|---:|
| ROC-AUC | 0.668 |
| Brier score, after calibration | 0.100 (from 0.214 raw) |
| Recall at production threshold | 67.3% |
| Features | 39, selected from 59 provisional |

The headline number isn't accuracy — with an 88.66%/11.34% class split, a model that predicts "no readmission" every time scores 88.66% accuracy while catching zero real readmissions. Every decision in this project — metric choice, threshold selection, error analysis — is built around that asymmetry instead.

## What Makes This More Than a Notebook Exercise

- **Patient-grouped validation** — no patient's encounters ever cross between train and test, verified programmatically on every run
- **Threshold selection made explicit** — the model's decision cutoff is derived from an explicit, stated cost ratio (a missed readmission costs ~8x an unnecessary follow-up call), not a silent default
- **A subgroup finding that changed the conclusion** — obstetric-related encounters looked like a model blind spot at first; digging in showed the model actually *ranks risk correctly* for this subgroup (AUC 0.743, above the overall model's 0.668) — the real issue was a single global threshold, not the model's understanding
- **External validation against published research** — this project's cleaned data independently reproduces the core finding of Strack et al. (2014); a follow-up ablation study precisely quantifies how much of that signal the model's final feature set actually captures versus proxies through other features
- **A fully reproducible pipeline, not just a trained artifact** — raw CSV → PostgreSQL → cleaned data → engineered features → trained model, as verifiable scripts with checkpoint assertions at every stage, not a one-off notebook run

## Architecture

```
Data/raw/diabetic_data.csv
        │
        ▼
src/data/load_raw_data.py ──────► PostgreSQL: raw_encounters
        │
        ▼
src/data/clean_data.py ─────────► PostgreSQL: cleaned_encounters
        │
        ▼
src/features/build_features.py ─► PostgreSQL: model_ready_data (39 features)
        │
        ▼
src/models/train.py ────────────► Models/tuned_balanced_xgb_pipeline_v2.joblib
        │
        ▼
src/models/calibrate.py ────────► Models/calibrated_xgb_pipeline_v2.joblib
        │
        ▼
   ┌────┴────┐
   ▼         ▼
FastAPI    (tested via
(/predict)   pytest against
   │         known ground truth)
   ▼
Django webapp (dropdown-driven form,
consumes the API over HTTP)
```

## Tech Stack

Python · Pandas · scikit-learn · XGBoost · SQLAlchemy · PostgreSQL · FastAPI · Django · pytest · Power BI

## Project Structure

```
├── Notebook/           01–05: data profiling through model improvement
├── src/
│   ├── db/              shared Postgres connection
│   ├── data/             raw-load and cleaning scripts
│   ├── features/          feature engineering script
│   └── models/             train / calibrate / fixture generation / ablation
├── api/                 FastAPI service + pytest suite
├── webapp/              Django application
├── Models/               trained + calibrated model artifacts
└── PROJECT_DOCUMENTATION.md   full methodology, findings, and limitations
```

See [`PROJECT_DOCUMENTATION.md`](./PROJECT_DOCUMENTATION.md) for the complete methodology, every modeling decision and its justification, the full error analysis and explainability work, and documented limitations.

## Running It Locally

**1. Database**
```bash
# Create a PostgreSQL database named diabetes_readmission,
# then set credentials in src/db/connection.py
python -m src.data.load_raw_data
python -m src.data.clean_data
python -m src.features.build_features
python -m src.models.train
python -m src.models.calibrate
```

**2. API**
```bash
pip install -r requirements.txt
python -m uvicorn api.main:app --reload
# → http://127.0.0.1:8000/docs
```

**3. Tests**
```bash
python -m pytest api/tests/ -v
```

**4. Web application**
```bash
cd webapp
python manage.py migrate
python manage.py runserver 8001
# → http://127.0.0.1:8001/
```

The API must be running for the Django app to return predictions.

## API Example

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{ "time_in_hospital": 4, "race": "Caucasian", "gender": "Female", ... }'
```
```json
{
  "readmission_probability": 0.111,
  "risk_flag": true,
  "decision_threshold": 0.11,
  "model_version": "calibrated_xgb_pipeline_v2.joblib"
}
```

Full interactive docs, with every field and valid value, are auto-generated at `/docs` once the API is running.

## Limitations

- ROC-AUC 0.668 is moderate discrimination, consistent with published results on this dataset
- A single global decision threshold structurally disadvantages lower-base-rate subgroups (most clearly demonstrated for pregnancy-related encounters)
- All findings are associational — no causal claims are made or implied
- Dataset spans 1999–2008; may not generalize to current clinical practice

Full limitations and model-card-style documentation in [`PROJECT_DOCUMENTATION.md`](./PROJECT_DOCUMENTATION.md).

## Data Source

Strack, B., DeShazo, J.P., Gennings, C., et al. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.* BioMed Research International, 2014. https://doi.org/10.1155/2014/781670
