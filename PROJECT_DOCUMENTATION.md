# Hospital Readmission Risk Prediction System
### Diabetes 130-US Hospitals (1999–2008) — Project Documentation

**Status as of this document:** Modeling, error analysis, explainability, and post-hoc model improvement complete. API/Application/Power BI/final documentation layers not yet started.

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

---

## 2. Technology Stack

| Layer | Tools |
|---|---|
| Data wrangling | Python, Pandas |
| Database | PostgreSQL |
| Modeling | scikit-learn, XGBoost |
| API | FastAPI |
| Application | Django |
| BI / Reporting | Power BI |
| Version control | GitHub |

Docker is **not** used in this project.

---

## 3. Project Structure

```text
diabetes-130-us-hospitals-for-years-1999-2008/
├── .venv/
├── Data/
│   ├── raw/                          # original UCI files + IDs_mapping.csv
│   └── Processed/
│       ├── diabetes_cleaned_data.csv     # df_clean — 101,766 rows, 67 cols
│       ├── diabetes_modeling_data.csv    # df_model — 100,114 rows, 67 cols
│       ├── Diabetes_orginal_file.csv     # df_model_feature — 100,114 rows, all cols incl. patient_nbr
│       └── diabetes_model_ready.csv      # FINAL model input — 100,114 rows, 39 features + target
├── Notebook/
│   ├── 01_data_profiling.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_machine_learning_modeling.ipynb
│   └── 05_model_validation_and_improvement.ipynb
├── Models/
│   └── tuned_balanced_xgb_pipeline.joblib
├── src/
│   ├── __init__.py
│   └── evaluation.py                  # evaluate_model / evaluate_thresholds / find_best_threshold
├── .gitignore
└── requirements.txt
```

**Which processed file is "the" model input?** `diabetes_model_ready.csv`
is the only file that should ever be loaded to reproduce, retrain, or
evaluate the model. The others are intermediate artifacts kept for
traceability, not for direct modeling use. `Diabetes_orginal_file.csv`
is needed alongside it only to recover `patient_nbr` for group-aware
splitting, since the model-ready file intentionally excludes identifiers.

---

## 4. Data Understanding & Cleaning (Notebooks 01–02)

**Raw data:** 101,766 encounters, 71,518 unique patients. Encounter-level,
not patient-level — a patient can appear multiple times (max 40 encounters
for one patient).

**Key profiling findings (01):**
- `?` used as a placeholder for missing values in `weight`, `payer_code`,
  `medical_specialty`, `race`, diagnosis fields
- `admission_type_id`, `discharge_disposition_id`, `admission_source_id`
  are coded categoricals, not continuous numbers
- Original 3-class target `readmitted` (`NO` / `>30` / `<30`): `<30` is
  11,357 encounters — a minority class
- `examide`, `citoglipton` have zero variance (single category, all rows)
- Race and gender are consistent within a patient across repeat
  encounters; age band can legitimately change

**Key cleaning decisions (02):**
- High-missingness columns (`weight`, `payer_code`, `max_glu_serum`,
  `A1Cresult`) were **not dropped**. Each was converted into an
  indicator + level pair (e.g. `weight` → `weight_documented` flag),
  preserving the "was this measured" signal instead of discarding it.
- Discharge dispositions were split into **two distinct categories**,
  not one blanket "expired/hospice" exclusion:
  - **Death-related** (`Expired`, `Expired at home/medical
    facility/unknown` — IDs 11, 19, 20, 21): **excluded** from the
    modeling population, because a deceased patient cannot be
    readmitted. **1,652 encounters excluded.**
  - **Hospice** (IDs 13, 14): **kept**. A hospice discharge can still
    be followed by a real 30-day readmission, so excluding it would
    have thrown away valid signal.
- ICD-9 diagnosis codes bucketed into broad diagnosis groups
  (`diag_1_group`, `diag_2_group`, `diag_3_group`)
- Admin ID codes mapped to human-readable labels via `IDs_mapping.csv`
  (`admission_type`, `discharge_disposition`, `admission_source`)
- Medication columns summarized into aggregate features
  (`num_medications_active`, `num_medications_up/down/steady`,
  `medication_changed`) while raw per-drug columns were also retained
  at this stage (later thinned in 03)
- Binary target created: `readmitted_30d` (1 = `<30`, 0 = otherwise)

**Modeling population:** `df_model` = 100,114 encounters (101,766 raw
− 1,652 death-related). Target distribution: **88.66% negative /
11.34% positive** — a real class imbalance that shapes every modeling
decision downstream.

---

## 5. Feature Engineering (Notebook 03)

Started from 59 provisional features (67 raw columns minus IDs, target,
and leakage-risk columns).

**Removed:**
- 10 medication columns with <0.1% active prevalence (e.g.
  `acetohexamide`, `tolbutamide`, `troglitazone`) — active in as few as
  1–85 of 100,114 encounters
- 6 redundant engineered columns once a cleaner indicator/level version
  existed (e.g. dropped `max_glu_serum_level` once
  `max_glu_serum_documented` covered the useful signal for modeling)

**Final modeling dataset:** **39 features + target**, saved to
`diabetes_model_ready.csv`. No target leakage columns present (verified
programmatically). Patient identifier deliberately excluded from the
feature set — but preserved separately in `Diabetes_orginal_file.csv`
for group-aware train/test splitting.

---

## 6. Machine Learning (Notebook 04)

**Validation strategy:** `GroupShuffleSplit` on `patient_nbr` (not a
plain random split) — ensures the same patient never appears in both
train and test, preventing leakage from repeat encounters of the same
person. 80/20 split, zero patient overlap confirmed. Threshold selection
used out-of-fold predictions only; the test set was touched exactly once,
for final evaluation.

**Class imbalance handling:** class-weighting / balanced variants
compared against SMOTE-based approaches during iteration.

**Model comparison (test set):**

| Model | ROC-AUC | Avg. Precision (PR-AUC) | Recall @ threshold | F1 |
|---|---:|---:|---:|---:|
| Dummy (majority class) | ~0.50 | — | ~0 | ~0 |
| Logistic Regression (balanced) | ~0.66 | — | 0.555 | — |
| Random Forest (baseline) | 0.653 | 0.207 | ~0 (broken threshold) | ~0 |
| XGBoost (baseline) | 0.659 | 0.217 | 0.430 | 0.281 |
| **XGBoost (tuned, balanced) — SELECTED** | **0.668** | **0.230** | **0.548** | **0.281** |

**Why not accuracy?** With an 88.66/11.34 class split, a model that
always predicts "no readmission" scores 88.66% accuracy while catching
zero real readmissions. Accuracy is not reported as a decision metric
anywhere in this project for that reason.

**Why prioritize recall over precision?** The two error types have
asymmetric real-world cost in this use case:
- **False negative** (missed readmission): a genuinely high-risk patient
  gets no follow-up outreach — the costly, harmful miss.
- **False positive** (unnecessary flag): a lower-risk patient gets an
  extra phone call or check-in — mildly wasteful, not harmful.

This asymmetry is why threshold selection favored recall, while
ROC-AUC/PR-AUC guarded against the degenerate "flag everyone" solution
that maximizes recall alone.

**Final artifact:** the Tuned Balanced XGBoost pipeline (preprocessing +
model together) was refit on the full training data and saved to
`Models/tuned_balanced_xgb_pipeline.joblib`. Reload-verified.

**Preprocessing pipeline:** `ColumnTransformer` selecting `numerical_features`
and `categorical_features` **by name**, `remainder="drop"`. This detail
matters — see Section 8.

---

## 7. Model Validation & Error Analysis (Notebook 05)

**Calibration (initial):** Brier score 0.213. The calibration curve
showed the model was **overconfident** — predicted probabilities
exceeded observed readmission rates across most bins — while retaining
useful **rank-ordering** ability. See Section 9 for the correction
applied to this issue.

**Error rates at the original threshold (0.50):** False Positive Rate
30.55%, False Negative Rate 45.25%.

### 7.1 — False Negatives vs. True Positives

Both numeric and categorical comparisons point to one consistent
pattern: encounters the model correctly catches (true positives) look
more medically "severe" at this specific encounter — more prior
inpatient visits, longer stays, more medications, admission via the
Emergency Room, discharge to a skilled nursing/rehab facility.
Encounters the model misses (false negatives) look comparatively
routine — shorter stays, fewer medications, planned (Physician
Referral) admission, discharge straight home — despite still resulting
in a real readmission.

**Interpretation:** the model relies heavily on encounter-level acuity
as a proxy for readmission risk. This works when severity and true risk
coincide, but misses patients whose risk isn't visible in how serious
the current encounter looked.

### 7.2 — False Positives vs. True Negatives

The false-positive comparison mirrors the false-negative comparison
exactly, confirming a single underlying axis rather than two separate
error modes. Across all four outcome groups, `number_inpatient`
increases monotonically with predicted risk:

| Group | number_inpatient (mean) | Predicted | Actual |
|---|---:|---|---|
| True Negative | 0.15 | Low risk | No readmission (correct) |
| False Negative | 0.23 | Low risk | Readmitted (missed) |
| False Positive | 1.46 | High risk | No readmission (false alarm) |
| True Positive | 1.96 | High risk | Readmitted (correct) |

The same ordering holds for prior ER admission, length of stay, and
discharge to SNF/rehab. False positives skew toward age 80–90 and
recent medication changes — plausibly cases where in-hospital
management successfully resolved an acute risk that had been real at
admission.

**Conclusion:** the residual error does not stem from a feature
engineering mistake, but from an information ceiling — factors outside
this dataset (medication adherence, home support, follow-up access)
likely drive much of the remaining outcome variance.

---

## 8. Known Issue & Correction: Notebook 05 Data Source

**Issue identified:** `05` initially loaded `diabetes_modeling_data.csv`
(the 67-column, pre-feature-selection output of `02`) instead of
`diabetes_model_ready.csv` (the 39-feature file `03` actually produced
and `04` actually trained on). This left `X_test` with 64 columns
instead of 39, including columns `03` deliberately removed (raw numeric
ID duplicates of already-encoded string categories, 10 zero-variance
sparse medication columns, redundant engineered columns).

**Impact assessment:**
- **Model predictions, ROC-AUC, and calibration were unaffected.** The
  trained pipeline's `ColumnTransformer` selects columns by name with
  `remainder="drop"`, so it silently ignored the 41 extra columns and
  used only the 39 it was trained on.
- **The feature-level error-analysis tables were contaminated** prior
  to the fix, since they included columns the model never sees.

**Fix applied:** `05` now loads `diabetes_model_ready.csv` for
features/target and separately loads `Diabetes_orginal_file.csv` for
`patient_nbr` (mirroring `04`'s approach) before rebuilding the split.
All results in Sections 7–10 of this document reflect the corrected
39-feature `X_test`.

---

## 9. Explainability (Notebook 05, Section 5)

Three complementary methods were used to directly measure what the
error analysis (Section 7) inferred indirectly.

### 9.1 — Global Feature Importance

| Feature | Native (rank) | SHAP (rank) | Permutation (rank) |
|---|:---:|:---:|:---:|
| `number_inpatient` | 1 | 1 | **1** |
| `discharge_disposition` | 2 (split across dummies) | 2, 8 | **2** |
| `medical_specialty` | scattered/misleading | — | **3** |
| `payer_code` | 8 | 5 | 5 |

`number_inpatient` and `discharge_disposition` are, by every method
that measures fairly, the two dominant drivers of the model's
predictions — directly confirming the Section 7 hypothesis.

**Methodological note:** native (gain) importance is biased toward
high-cardinality one-hot fields. `medical_specialty` (60 categories)
looked scattered and unimportant in the native chart (individual rare
dummies appearing sporadically) but is the **3rd most important
feature overall** by permutation importance, which evaluates whole
original features fairly. Permutation importance is the most
trustworthy of the three for cross-feature comparison.

**Fairness note:** `payer_code` (insurance type) is a real, moderate
contributor. It should be disclosed in any deployment documentation as
a feature that may act as a proxy for socioeconomic or disability
status rather than a purely clinical signal.

`age` and `gender` showed differences in the Section 7 group
comparisons but do not appear in the permutation-importance top 20,
suggesting their apparent influence is correlated with — rather than
independent of — the two dominant utilization features.

### 9.2 — Local Case Studies (four individual encounters)

| Case | Predicted probability | Key finding |
|---|---:|---|
| Borderline False Negative | 0.4996 | Near-identical SHAP profile to the borderline FP below — a genuine toss-up, not a model failure |
| Confident-Miss False Negative | 0.115 | Driven by pregnancy/childbirth diagnosis codes — see subgroup verification below |
| Confident False Positive | 0.905 | `number_inpatient` alone (4.2 SD above average) contributed >60% of the total log-odds — illustrates over-reliance on one feature at extreme values |
| Borderline False Positive | 0.5000 | Mirrors the borderline FN almost exactly (same top two features, same direction, near-identical magnitude) |

**Key finding:** the borderline FN and borderline FP cases had nearly
identical SHAP explanations (`number_inpatient` ≈ −0.25,
"not discharged home" ≈ +0.23) and landed within 0.0001 of each other
in log-odds — one patient returned, one didn't. Near the decision
threshold, the model is correctly expressing genuine uncertainty, not
making an error that could be "fixed."

### 9.3 — Subgroup Verification: Obstetric/Pregnancy-Related Encounters

The confident-miss case above prompted a direct check, since one
anecdote should not become a documented limitation without numbers
behind it.

**Findings:**
- Pregnancy/childbirth-related encounters are rare (692 of 100,114,
  0.69%) with a genuinely lower real-world readmission rate (6.07% vs.
  11.34% overall) — the model's learned association is accurate, not
  biased.
- Within this subgroup (test set, n=124, 10 actual readmissions),
  ranking ability was strong: **ROC-AUC 0.743**, exceeding the overall
  model's 0.668. True positives were correctly scored higher (mean
  P=0.385) than non-readmissions (mean P=0.246).
- However, **7 of 10 subgroup readmissions fell below the 0.50
  threshold** despite being correctly ranked as elevated risk relative
  to peers.

**Revised conclusion:** this is not evidence the model fails to
understand obstetric-related encounters — ranking performance there is
above average. It is evidence of a **threshold-calibration
limitation**: a single global cutoff, tuned for an 11.34% base rate,
cannot simultaneously suit a subgroup whose true base rate is closer to
6–8%. Even a correctly-identified "riskier than peers" score in this
subgroup may not clear a threshold set for the general population.
(Note: n=10 actual positives is small; the specific miss ratio should
be read as directional.)

---

## 10. Model Improvement (Notebook 05, Section 6)

Two post-hoc corrections were applied to the existing trained pipeline
— no retraining required. A calibration/holdout split (patient-grouped,
carved from the original test set) was used so the correction was
fitted and evaluated on disjoint data.

### 10.1 — Recalibration

Isotonic regression (via `CalibratedClassifierCV` with a frozen/prefit
base estimator) reduced the Brier score from **0.2142 to 0.0996** —
predicted probabilities now track observed readmission rates
substantially more closely. Side effect: the calibrated model's
probability range compresses somewhat at the top end, which changes
where any given threshold should be set (see below) but does not by
itself indicate a decision-quality problem.

### 10.2 — Cost-Sensitive Threshold Selection

The original threshold was selected via F1, an objective that does not
account for the asymmetric real-world cost of a missed readmission vs.
a false alarm. Re-selecting the threshold to explicitly minimize
`(FN_cost × FN) + (FP_cost × FP)` revealed:

- At a naive 5:1 cost ratio, both raw and calibrated probabilities
  produced **identical total cost** (5,226) despite very different
  recall/threshold values — confirming that comparing calibrated vs.
  raw recall at a single, unvalidated cost ratio is misleading. The
  right comparison is cost-at-optimal-threshold, not recall-at-arbitrary-threshold.
- A sensitivity analysis across cost ratios (2, 3, 5, 8, 10, 15) showed
  the original model's F1-selected threshold (54.8% recall) implicitly
  behaves as though a missed readmission costs **roughly 7–8x** an
  unnecessary follow-up call — an assumption that had never been stated
  explicitly until this analysis.
- At a ratio of 8:1, the **calibrated** probabilities outperform raw:
  **67.3% recall** (calibrated) vs. 58.8% (raw), at comparable cost.

**Recommended configuration:** isotonic-calibrated model, decision
threshold corresponding to an assumed 8:1 FN:FP cost ratio (≈0.09 on
calibrated probabilities). This is adjustable if a real operational
cost estimate becomes available.

### 10.3 — Residual Subgroup Limitation

Re-checking the pregnancy-related subgroup (Section 9.3) at the
improved configuration: recall improved from ~30% to **43%** (3 of 7
caught in the final holdout), but still trails the overall population's
67.3% recall.

**Conclusion:** recalibration and cost-based threshold selection provide
a genuine, measurable improvement (Brier score more than halved, recall
improved from 54.8% to 67.3%) without materially harming subgroup
performance — and improve it somewhat. However, they do not fully
close the subgroup gap identified in Section 9.3. This is a structural
limit of any single global threshold: subgroups with a genuinely lower
base rate will always trail the population average to some degree.
Fully resolving this would require subgroup-specific thresholds or a
probability-first (no hard cutoff) deployment approach — noted as
future work, since the pregnancy subgroup's small size (n=62–124
across holdouts) makes a dedicated threshold statistically unreliable
to tune with currently available data.

---

## 11. Roadmap

| # | Step | Status |
|---|---|---|
| 1 | Data cleaning, feature engineering, modeling | ✅ Done |
| 2 | Error analysis (FN/TP/FP/TN) | ✅ Done |
| 3 | Explainability (native/permutation/SHAP, local cases, subgroup check) | ✅ Done |
| 4 | Model improvement (recalibration, cost-sensitive threshold) | ✅ Done |
| 5 | Fill in `04` cell 178's summary template with final numbers | ⬜ Next |
| 6 | FastAPI service wrapping the improved pipeline (`/predict` endpoint, returning calibrated probability) | ⬜ |
| 7 | Django application layer consuming the API | ⬜ |
| 8 | Power BI dashboard (model performance + risk-driver views) | ⬜ |
| 9 | Final documentation pass: README, model card, limitations | ⬜ |

---

## 12. Limitations (for the eventual model card)

- ROC-AUC 0.668 (0.743 within the obstetric subgroup) is moderate
  discrimination, consistent with published results on this dataset —
  readmission is inherently hard to predict from claims-style data
  alone (no clinical notes, no post-discharge information).
- The model relies heavily on two features (`number_inpatient`,
  `discharge_disposition`) as an acuity proxy. This drives most correct
  predictions but also most errors in both directions (Section 7.2).
- Even after recalibration, a single global decision threshold
  structurally disadvantages lower-base-rate subgroups (Section 9.3,
  10.3) — most clearly demonstrated for pregnancy/childbirth-related
  encounters (n=692, 0.69% of the dataset).
- `payer_code` is a moderately important predictor that may act as a
  proxy for socioeconomic or disability status; disclose this
  explicitly in any deployment context.
- Dataset spans 1999–2008 US hospitals; may not generalize to current
  clinical practice, other health systems, or other countries.
- All findings are associational. No causal claims should be drawn or
  implied from feature-importance, SHAP, or error-analysis output.
