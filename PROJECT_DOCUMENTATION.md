# Hospital Readmission Risk Prediction System
### Diabetes 130-US Hospitals (1999–2008) — Project Documentation

**Status as of this document:** Full pipeline (Postgres → clean → features → train → calibrate) built, verified, and promoted to production. API and Application layers serving the promoted model. Power BI and formal monitoring not yet started.

---

## 1. Problem Statement

Predict whether a diabetic inpatient encounter will result in a **30-day
hospital readmission**, using structured EHR-style data from the UCI
Diabetes 130-US Hospitals dataset (1999–2008).

This is a **decision-support tool, not a diagnostic system**. Findings
describe statistical association within this historical dataset, not
causal clinical relationships. No output from this project should be
used to make individual clinical decisions.

**Target users (hypothetical):** hospital discharge-planning teams who
want to prioritize follow-up outreach (calls, appointments, medication
review) toward the encounters most likely to bounce back within 30 days.

**Data provenance:** This dataset is the same clinical extract published
in Strack et al., "Impact of HbA1c Measurement on Hospital Readmission
Rates: Analysis of 70,000 Clinical Database Patient Records,"
*BioMed Research International*, 2014 (Health Facts database, Cerner
Corporation). The paper's five inclusion criteria (inpatient, diabetic
diagnosis, 1–14 day stay, labs performed, medications administered)
match this project's raw row count of 101,766 exactly, confirming
provenance.

---

## 2. Technology Stack

| Layer | Tools |
|---|---|
| Data wrangling | Python, Pandas, SQLAlchemy |
| Database | PostgreSQL |
| Modeling | scikit-learn, XGBoost |
| API | FastAPI |
| Application | Django |
| BI / Reporting | Power BI (planned) |
| Version control | GitHub |

Docker is **not** used in this project.

---

## 3. Project Structure

```text
diabetes-130-us-hospitals-for-years-1999-2008/
├── .venv/
├── Data/
│   ├── raw/                          # original UCI files + IDs_mapping.csv
│   └── Processed/                    # legacy CSV outputs from early notebook runs
├── Notebook/
│   ├── 01_data_profiling.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_machine_learning_modeling.ipynb
│   └── 05_model_validation_and_improvement.ipynb
├── Models/
│   ├── tuned_balanced_xgb_pipeline.joblib          # original, notebook-trained
│   ├── calibrated_xgb_pipeline.joblib               # original, calibrated
│   ├── calibrated_xgb_metadata.json
│   ├── tuned_balanced_xgb_pipeline_v2.joblib        # script-pipeline-trained
│   ├── calibrated_xgb_pipeline_v2.joblib            # PROMOTED — served by API
│   └── calibrated_xgb_metadata_v2.json              # PROMOTED — served by API
├── src/
│   ├── evaluation.py                  # evaluate_model / evaluate_thresholds / find_best_threshold
│   ├── db/
│   │   └── connection.py              # shared Postgres connection (SQLAlchemy)
│   ├── data/
│   │   ├── load_raw_data.py           # CSV -> Postgres raw_encounters
│   │   ├── clean_data.py              # raw_encounters -> cleaned_encounters
│   │   └── get_categorical_values.py  # pulls real category lists for Django dropdowns
│   ├── features/
│   │   └── build_features.py          # cleaned_encounters -> model_ready_data (39 features)
│   └── models/
│       ├── train.py                   # model_ready_data -> tuned_balanced_xgb_pipeline_v2
│       ├── calibrate.py               # v2 -> calibrated_xgb_pipeline_v2 + metadata
│       ├── generate_fixtures.py       # rebuilds pytest ground-truth fixtures for the live model
│       └── ablation_check.py          # tests whether dominant features mask other signal
├── api/
│   ├── main.py, schemas.py, model_loader.py
│   └── tests/
│       ├── test_predict.py
│       └── fixtures/known_cases.json  # regenerated against v2
├── webapp/                             # Django application
│   └── predictor/
│       ├── forms.py                    # 26 categorical fields as real dropdowns
│       ├── views.py                    # Post/Redirect/Get pattern (no resubmission bug)
│       └── templates/predictor/form.html
├── .gitignore
├── PROJECT_DOCUMENTATION.md
└── requirements.txt
```

**Data flow, end to end:**
```
Data/raw/diabetic_data.csv
  → src/data/load_raw_data.py         → Postgres: raw_encounters
  → src/data/clean_data.py            → Postgres: cleaned_encounters
  → src/features/build_features.py    → Postgres: model_ready_data
  → src/models/train.py               → Models/tuned_balanced_xgb_pipeline_v2.joblib
  → src/models/calibrate.py           → Models/calibrated_xgb_pipeline_v2.joblib (LIVE)
  → api/model_loader.py               → served via FastAPI
  → webapp/predictor/                 → consumed via Django
```
Every script prints numbered checkpoints with expected values and includes
assertions that halt execution if a stage's output doesn't match known
ground truth — this was deliberately built in after several early
debugging sessions where a script "ran successfully" while silently
producing wrong output.

---

## 4. Data Understanding & Cleaning (Notebooks 01–02, reproduced in `clean_data.py`)

**Raw data:** 101,766 encounters, 71,518 unique patients.

**Key cleaning decisions**, all reproduced and verified identically in
`src/data/clean_data.py`:
- High-missingness columns (`weight`, `payer_code`, `max_glu_serum`,
  `A1Cresult`) converted into indicator + level pairs rather than dropped.
  `weight` itself is deliberately left as `NaN`, never imputed or filled.
- Discharge dispositions split: **death-related** (IDs 11, 19, 20, 21)
  excluded (1,652 encounters); **hospice** (IDs 13, 14) deliberately kept.
- Admin ID codes (`admission_type_id`, `discharge_disposition_id`,
  `admission_source_id`) mapped to readable labels via `IDs_mapping.csv`.
- ICD-9 diagnosis codes bucketed into 17 broad clinical chapters, applied
  to all three diagnosis fields.
- Medication columns summarized into aggregate features.

**Modeling population:** 100,114 encounters (101,766 − 1,652
death-related). Target distribution: **88.66% negative / 11.34%
positive**.

---

## 5. Feature Engineering (Notebook 03, reproduced in `build_features.py`)

Started from 59 provisional features (67 raw columns minus IDs, target,
and leakage-risk columns; `death_related_disposition`, present in the
original notebook's 67-column input but not persisted in this pipeline's
`cleaned_encounters`, was a constant post-filter bookkeeping column with
no effect on the final feature count).

**Removed:** 10 sparse medication columns (<0.1% active prevalence) + 6
redundant engineered columns + 3 raw diagnosis codes (superseded by
diagnosis groups) + `weight` (96.86% missing).

**Final: 39 features + target**, verified via assertion (`assert
len(selected) == 39`) on every pipeline run.

---

## 6. Machine Learning (Notebook 04, reproduced in `train.py`)

**Validation strategy:** `GroupShuffleSplit` on `patient_nbr`, 80/20,
`random_state=42`, zero patient overlap verified on every run.

**Final model comparison (original, notebook-trained):**

| Model | ROC-AUC | Avg. Precision |
|---|---:|---:|
| Dummy (majority class) | ~0.50 | — |
| Logistic Regression (balanced) | ~0.66 | — |
| Random Forest (baseline) | 0.653 | 0.207 |
| XGBoost (baseline) | 0.659 | 0.217 |
| **XGBoost (Tuned, Balanced) — selected** | **0.668** | **0.230** |

**Why not accuracy:** an always-negative classifier scores 88.66%
accuracy while catching zero real readmissions.

**Why recall over precision:** a missed readmission (false negative) is
costlier in this use case than an unnecessary follow-up call (false
positive) — formalized later via cost-ratio threshold selection.

**Final hyperparameters** (`n_estimators=200, learning_rate=0.05,
max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
scale_pos_weight` computed per-training-set): reproduced exactly in
`src/models/train.py`.

---

## 7. Model Validation & Error Analysis (Notebook 05)

**Calibration (initial):** raw model Brier score 0.2142 — overconfident.

**FN vs. TP vs. FP vs. TN:** all four groups sit along a single apparent
axis of "recent care intensity." `number_inpatient` increases
monotonically across the four groups: TN (0.15) → FN (0.23) → FP (1.46)
→ TP (1.96). Same pattern holds for ER admission, length of stay, and
discharge to SNF/rehab.

**Conclusion:** the model relies heavily on encounter-level acuity as a
proxy for readmission risk — genuinely predictive, but a blunt
instrument at the margins.

---

## 8. Known Issue & Correction: Notebook 05 Data Source

`05` initially loaded the pre-feature-selection 67-column file instead
of the 39-feature model-ready file. Model predictions were unaffected
(the pipeline's `ColumnTransformer` selects columns by name and
ignores extras), but early feature-level error-analysis tables were
contaminated until corrected.

---

## 9. Explainability (Notebook 05, Section 5)

**Global importance**, triangulated across three methods: `number_inpatient`
and `discharge_disposition` are the dominant drivers by every method that
measures fairly (native gain, permutation, SHAP). `medical_specialty`
looked scattered in native importance (60-category dilution) but is the
3rd most important feature by permutation importance. `payer_code` is a
real, moderate contributor, flagged as a fairness-relevant feature
(possible proxy for socioeconomic/disability status).

**Local case studies:** four individual encounters (borderline FN,
confident-miss FN, confident FP, borderline FP) confirmed the global
pattern at the individual-prediction level via SHAP waterfall
decomposition. Borderline FN and borderline FP had nearly identical
SHAP profiles — near-threshold predictions are genuine toss-ups, not
model failures.

### 9.3 — Subgroup Verification: Obstetric/Pregnancy-Related Encounters

Pregnancy/childbirth-related encounters (692 of 100,114, 0.69%) have a
genuinely lower real-world readmission rate (6.07% vs. 11.34% overall).
Within-subgroup ranking is strong (ROC-AUC 0.743, exceeding the overall
model's 0.668), but a fixed global threshold underperforms for this
lower-base-rate subgroup — a **threshold-calibration limitation**, not
evidence the model fails to understand the subgroup.

### 9.4 — External Validation Against Published Literature

A direct SQL replication of Strack et al. (2014)'s core finding was run
against `cleaned_encounters`:

| A1Cresult group | This project's readmission rate | Strack et al.'s rate |
|---|---:|---:|
| Not tested | 11.63% | 9.4% |
| Tested, >7% | 10.12% | — |
| Tested, >8% | 9.95% | 8.9% |
| Tested, Normal | 9.76% | 8.9% |

The direction matches exactly across both datasets: not being tested has
the highest readmission rate, and every tested group sits lower. This
independently confirms the cleaning pipeline preserved a genuine,
previously-published clinical signal, despite this project's population
being larger and more inclusive (100,114 vs. 69,984 — this project keeps
every encounter and retains hospice discharges, unlike the paper's
first-encounter-per-patient, hospice-excluded design).

**However**, `A1Cresult` does not appear in the model's top 20 features
by permutation importance (Section 9.1). An ablation experiment
(`src/models/ablation_check.py`) removed the two dominant features
(`number_inpatient`, `discharge_disposition`) and retrained: `A1Cresult`
rose only to rank 11 of 37, with importance (0.0016) roughly 6x smaller
than the new top feature (`number_emergency`, 0.0094).

**Conclusion:** the paper's finding replicates in raw association, but
the model's own feature-importance analysis indicates this is
substantially a **proxy relationship** — patients who receive A1C
testing also tend to have more prior utilization and longer stays,
signal the model already captures more directly through
`number_inpatient`/`number_emergency`/`time_in_hospital`. Some
independent A1C-testing signal likely remains but is not a major driver
of this model's predictions. This is presented as a worked example of
distinguishing genuine feature redundancy from a missed signal, per the
project's feature-disagreement framework: (1) check for redundancy with
existing features, (2) ablate and check if importance rises, (3) check
for subgroup/interaction effects before concluding either the research
or the model is "wrong."

---

## 10. Model Improvement (Notebook 05, Section 6 — original model)

Isotonic recalibration reduced Brier score from 0.2142 to 0.0996.
Cost-sensitive threshold selection (replacing the original F1-based
choice) revealed the original threshold implicitly assumed a ~7–8:1
FN:FP cost ratio; explicit selection at 8:1 improved recall from 54.8%
to 67.3%. A residual subgroup limitation was confirmed for the
obstetric subgroup (Section 9.3) — improved from ~30% to 43% recall,
still trailing the population average, a structural limit of any
single global threshold.

---

## 11. Data Pipeline: PostgreSQL Integration

Following the original notebook-based workflow, the full pipeline was
rebuilt as reusable, verifiable scripts reading from and writing to a
real PostgreSQL database (`diabetes_readmission`), rather than only
ever reading/writing CSV files. This closes the gap between the
project's stated architecture (`Problem → Data → Database → ...`) and
what was actually, verifiably running.

**`src/db/connection.py`** — single shared SQLAlchemy engine factory.
Passwords are URL-encoded via `urllib.parse.quote_plus` before being
inserted into the connection string (a raw `@` in a password will
otherwise be misparsed as the user/host separator).

**`src/data/load_raw_data.py`** — loads `Data/raw/diabetic_data.csv`
into table `raw_encounters` (101,766 rows). Safely re-runnable
(`if_exists="replace"`).

**`src/data/clean_data.py`** — `raw_encounters` → `cleaned_encounters`.
Verified via 11 numbered checkpoints, each printing a real number
checked against known ground truth (e.g. weight-missing count 98,569;
gender `Not_Documented` count 3; readmitted_30d distribution
90,409/11,357; zero nulls across all three admin-ID mapped columns;
final shape (100114, 66)). All checkpoints matched exactly.

**`src/features/build_features.py`** — `cleaned_encounters` →
`model_ready_data` (39 features + target + `encounter_id` +
`patient_nbr`, kept in the same table to avoid the row-order-dependent
CSV join that caused the Notebook 05 bug in Section 8 — a direct
architectural improvement over the original notebook pipeline).
Verified via assertions at every stage (59 provisional features, 20
dropped, 39 selected).

**`src/models/train.py`** — `model_ready_data` → a freshly-trained
Tuned Balanced XGBoost pipeline, using the exact hyperparameters from
Notebook 04. Test ROC-AUC 0.6690 vs. the original notebook's 0.668 —
confirms the script pipeline reproduces the original model, not merely
an approximation of it.

---

## 12. Model v2: Calibration, Verification, and Promotion

The script-trained model (`tuned_balanced_xgb_pipeline_v2.joblib`) was
calibrated and promoted to production through the same rigor applied to
the original model in Section 10, plus an additional verification step
made necessary by having two candidate artifacts.

**Raw comparison (both uncalibrated):**

| Metric | Original | v2 |
|---:|---:|---:|
| ROC-AUC | 0.668 | 0.6690 |
| Brier score | 0.2142 | 0.2133 |

**Calibration (`src/models/calibrate.py`):** isotonic regression on a
patient-grouped 50/50 split of the test set, mirroring Section 10's
methodology exactly. Brier score: 0.2133 → 0.0994 (near-identical to
the original's 0.2142 → 0.0996).

**Threshold:** cost-sensitive search at an 8:1 FN:FP ratio found v2's
own optimal threshold at 0.11 (vs. the original's 0.09) — expected,
since v2 used its own fresh `GroupShuffleSplit` calls, producing a
slightly different calibration/holdout carve.

**Verification the threshold difference wasn't a regression:** recall
at v2's own threshold (0.11) was 0.6241, noticeably below the original's
0.6730 at threshold 0.09. Rather than accept this gap, recall was
checked at the **same** threshold (0.09) for both models: v2 achieved
**0.7095** — higher than the original. This confirmed the apparent
recall gap was purely an artifact of v2's own (correctly, independently
derived) stricter threshold, not a weaker model. v2's own threshold
(0.11) was retained for production, since it was derived the same
rigorous way as the original's.

**Promotion:** `api/model_loader.py` updated to serve
`calibrated_xgb_pipeline_v2.joblib` / `calibrated_xgb_metadata_v2.json`.
`api/tests/fixtures/known_cases.json` regenerated
(`src/models/generate_fixtures.py`) against v2's own predictions for
the same four case types (borderline FN, confident-miss FN, confident
FP, borderline FP) identified the same way as the original notebook's
Section 5.5. Full pytest suite re-verified against the live, promoted
API before this was considered complete.

---

## 13. API and Application Layers

**FastAPI (`api/`):** `/health`, `/predict`, `/predict/batch`. Pydantic
schema validation (`EncounterInput`) rejects malformed requests with a
422 before they reach the model. `OneHotEncoder(handle_unknown="ignore")`
means an unrecognized category value is silently zero-encoded rather
than crashing the request — documented, not a bug. Model loaded once via
`@lru_cache`, not per-request. Automated pytest suite (7 tests) checks
the live API's predictions against `known_cases.json` ground truth on
every run, plus schema-validation and unknown-category edge cases.

**Django (`webapp/`):** thin consumer of the FastAPI service — no model
logic duplicated. All 26 categorical form fields are real dropdowns
(`ChoiceField`), populated from real Postgres category values via
`src/data/get_categorical_values.py`, eliminating the possibility of a
typo silently zero-encoding a field. Uses the Post/Redirect/Get pattern
(`views.py`) to prevent duplicate-submission-on-refresh, a common Django
pitfall.

Both layers were verified end-to-end multiple times: after initial
build, and again after the v2 promotion, using real rows from
`known_cases.json` typed manually into the Django form and checked
against the fixture's expected probability.

---

## 14. Roadmap

| # | Step | Status |
|---|---|---|
| 1 | Data cleaning, feature engineering, modeling (notebooks) | ✅ Done |
| 2 | Error analysis, explainability, calibration (notebooks) | ✅ Done |
| 3 | FastAPI + Django, tested end-to-end | ✅ Done |
| 4 | PostgreSQL-backed script pipeline (raw → clean → features → train) | ✅ Done |
| 5 | v2 model calibration, verification, and promotion to production | ✅ Done |
| 6 | Django dropdown fields + POST-refresh fix | ✅ Done |
| 7 | External validation against Strack et al. (2014) + ablation study | ✅ Done |
| 8 | Power BI dashboard (performance, threshold tradeoff, error analysis, drivers, fairness) | ⬜ Next |
| 9 | Monitoring & drift-tracking plan | ⬜ |
| 10 | GitHub push of all pipeline/promotion work | ✅ Done |

---

## 15. Limitations (for the eventual model card)

- ROC-AUC 0.668–0.669 (0.743 within the obstetric subgroup) is moderate
  discrimination, consistent with published results on this dataset.
- The model relies heavily on `number_inpatient` and
  `discharge_disposition` as an acuity proxy — drives most correct
  predictions but also most errors in both directions.
- A single global decision threshold structurally disadvantages
  lower-base-rate subgroups, most clearly demonstrated for
  pregnancy/childbirth-related encounters (0.69% of the dataset).
- `payer_code` is a moderately important predictor that may proxy for
  socioeconomic or disability status; disclose explicitly in any
  deployment context.
- `A1Cresult` (HbA1c testing status) shows a real, literature-replicated
  association with readmission, but contributes only modestly to this
  model's predictions once utilization features are accounted for —
  most of its raw association is proxied through other features.
- Dataset spans 1999–2008 US hospitals; may not generalize to current
  clinical practice, other health systems, or other countries.
- All findings are associational. No causal claims should be drawn or
  implied from feature-importance, SHAP, or error-analysis output.
- No production monitoring or drift-detection is currently implemented.
