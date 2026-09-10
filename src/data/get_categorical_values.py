from src.db.connection import get_engine

CATEGORICAL_COLUMNS = [
    "race", "gender", "age", "payer_code", "medical_specialty",
    "max_glu_serum", "A1Cresult", "metformin", "repaglinide", "nateglinide",
    "glimepiride", "glipizide", "glyburide", "pioglitazone", "rosiglitazone",
    "insulin", "glyburide-metformin", "acarbose", "change", "diabetesMed",
    "admission_type", "discharge_disposition", "admission_source",
    "diag_1_group", "diag_2_group", "diag_3_group"
]


def get_categorical_values():
    engine = get_engine()

    for col in CATEGORICAL_COLUMNS:
        query = f'SELECT DISTINCT "{col}" FROM model_ready_data ORDER BY "{col}"'
        with engine.connect() as conn:
            result = conn.exec_driver_sql(query)
            values = sorted([row[0] for row in result if row[0] is not None])

        # Prints in a format you can paste directly into forms.py
        python_var_name = col.replace("-", "_")
        print(f"{python_var_name}_choices = {values}")
        print()


if __name__ == "__main__":
    get_categorical_values()