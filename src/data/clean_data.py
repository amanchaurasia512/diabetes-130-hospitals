import pandas as pd
from src.db.connection import get_engine

MEDICATION_COLS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
    'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
    'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
    'miglitol', 'troglitazone', 'tolazamide', 'insulin',
    'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone',
    'metformin-pioglitazone'
]
DIAGNOSIS_COLS = ['diag_1', 'diag_2', 'diag_3']


def load_raw_from_postgres():
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM raw_encounters", engine)
    print(f"Loaded {len(df):,} raw rows from Postgres")
    return df


def replace_missing_placeholders(df):
    """The raw data uses '?' as a missing-value placeholder — convert to real NaN."""
    return df.replace("?", pd.NA)


def fill_categorical_missing(df):
    """
    max_glu_serum, A1Cresult, medical_specialty, payer_code, race, and the
    diagnosis columns are explicitly filled with 'Not_Documented' rather
    than imputed with an existing category — this preserves the fact that
    the value was genuinely missing, rather than inventing information.
    weight is intentionally NOT filled — 96.86% missing, imputing would
    add far more artificial information than it's worth.
    """
    for col in ['max_glu_serum', 'A1Cresult', 'medical_specialty', 'payer_code',
                'race', 'diag_1', 'diag_2', 'diag_3']:
        df[col] = df[col].fillna('Not_Documented')

    return df


def fix_invalid_gender(df):
    df['gender'] = df['gender'].replace('Unknown/Invalid', 'Not_Documented')

    return df


def drop_constant_medications(df):
    """citoglipton and examide are 100% 'No' for every encounter — no signal."""
    return df.drop(columns=['citoglipton', 'examide'])


def create_target(df):
    df['readmitted_30d'] = (df['readmitted'] == '<30').astype(int)
    return df

def apply_admin_id_mapping(df, ids_mapping_path="Data/raw/IDs_mapping.csv"):
    mapping = pd.read_csv(ids_mapping_path)

    admission_mapping = mapping.iloc[0:8].copy()
    discharge_mapping = mapping.iloc[10:40].copy()
    source_mapping = mapping.iloc[42:].copy()

    for table in [admission_mapping, discharge_mapping, source_mapping]:
        table['admission_type_id'] = pd.to_numeric(
            table['admission_type_id'], errors='coerce'
        ).astype('Int64')  # nullable integer, not float

    admission_mapping = admission_mapping.dropna(subset=['admission_type_id'])
    discharge_mapping = discharge_mapping.dropna(subset=['admission_type_id'])
    source_mapping = source_mapping.dropna(subset=['admission_type_id'])

    for table in [admission_mapping, discharge_mapping, source_mapping]:
        table['description'] = table['description'].fillna('Not_Documented')

    admission_map = {k: v.strip() for k, v in
                      dict(zip(admission_mapping['admission_type_id'], admission_mapping['description'])).items()}
    discharge_map = {k: v.strip() for k, v in
                      dict(zip(discharge_mapping['admission_type_id'], discharge_mapping['description'])).items()}
    source_map = {k: v.strip() for k, v in
                  dict(zip(source_mapping['admission_type_id'], source_mapping['description'])).items()}

    df['admission_type_id'] = df['admission_type_id'].astype('Int64')
    df['discharge_disposition_id'] = df['discharge_disposition_id'].astype('Int64')
    df['admission_source_id'] = df['admission_source_id'].astype('Int64')

    df['admission_type'] = df['admission_type_id'].map(admission_map)
    df['discharge_disposition'] = df['discharge_disposition_id'].map(discharge_map)
    df['admission_source'] = df['admission_source_id'].map(source_map)

    return df     

def classify_icd9(code):
    """Buckets a raw ICD-9 diagnosis code into a broad clinical chapter."""
    if code == 'Not_Documented':
        return 'Not_Documented'
    code = str(code).strip()
    if code.startswith('V'):
        return 'Supplementary factors (V-code)'
    if code.startswith('E'):
        return 'External causes (E-code)'
    try:
        value = float(code)
    except ValueError:
        return 'Other/Unknown'

    ranges = [
        (1, 139, 'Infectious & parasitic diseases'),
        (140, 239, 'Neoplasms'),
        (240, 279, 'Endocrine, nutritional & metabolic diseases'),
        (280, 289, 'Blood diseases'),
        (290, 319, 'Mental disorders'),
        (320, 389, 'Nervous system & sense organs'),
        (390, 459, 'Circulatory diseases'),
        (460, 519, 'Respiratory diseases'),
        (520, 579, 'Digestive diseases'),
        (580, 629, 'Genitourinary diseases'),
        (630, 679, 'Pregnancy & childbirth'),
        (680, 709, 'Skin & subcutaneous tissue'),
        (710, 739, 'Musculoskeletal system'),
        (740, 759, 'Congenital anomalies'),
        (760, 779, 'Perinatal conditions'),
        (780, 799, 'Symptoms/signs/ill-defined conditions'),
        (800, 999, 'Injury & poisoning'),
    ]
    for low, high, label in ranges:
        if low <= value <= high:
            return label
    return 'Other/Unknown'


def bucket_diagnosis_codes(df):
    for col in DIAGNOSIS_COLS:
        df[f'{col}_group'] = df[col].apply(classify_icd9)
    return df


def summarize_medications(df):
    df['num_medications_active'] = (df[MEDICATION_COLS] != 'No').sum(axis=1)
    df['num_medications_up'] = (df[MEDICATION_COLS] == 'Up').sum(axis=1)
    df['num_medications_down'] = (df[MEDICATION_COLS] == 'Down').sum(axis=1)
    df['num_medications_steady'] = (df[MEDICATION_COLS] == 'Steady').sum(axis=1)
    df['medication_changed'] = (df['change'] == 'Ch').astype(int)
    df['diabetes_medication_used'] = (df['diabetesMed'] == 'Yes').astype(int)
    return df


def add_documentation_indicators(df):
    df['weight_documented'] = df['weight'].notna().astype(int)

    glucose_level_map = {'Norm': 0, '>200': 1, '>300': 2}
    df['max_glu_serum_level'] = df['max_glu_serum'].map(glucose_level_map)
    df['max_glu_serum_documented'] = (df['max_glu_serum'] != 'Not_Documented').astype(int)

    a1c_level_map = {'Norm': 0, '>7': 1, '>8': 2}
    df['A1C_level'] = df['A1Cresult'].map(a1c_level_map)
    df['A1C_documented'] = (df['A1Cresult'] != 'Not_Documented').astype(int)

    return df


def exclude_death_related(df):
    """
    IDs 11, 19, 20, 21 are death-related — excluded (can't be readmitted).
    IDs 13, 14 (hospice) are deliberately KEPT.
    """
    death_related_ids = [11, 19, 20, 21]
    before = len(df)
    df = df[~df['discharge_disposition_id'].isin(death_related_ids)].copy()
    print(f"Excluded {len(before) - len(df) if hasattr(before,'__len__') else before - len(df):,} "
          f"death-related encounters (expected: 1,652)")
    return df
def clean_data():
    df = load_raw_from_postgres()
    print(f"[1/11] Raw data loaded — shape: {df.shape}")

    df = replace_missing_placeholders(df)
    print(f"[2/11] '?' placeholders replaced with NaN — "
          f"weight still missing: {df['weight'].isna().sum():,}")

    df = fill_categorical_missing(df)
    print(f"[3/11] Categorical missing values filled — "
          f"payer_code nulls remaining: {df['payer_code'].isna().sum()}")

    df = fix_invalid_gender(df)
    print(f"[4/11] Gender fixed — value counts: {dict(df['gender'].value_counts())}")

    df = drop_constant_medications(df)
    print(f"[5/11] Constant medications dropped — shape now: {df.shape}")

    df = create_target(df)
    print(f"[6/11] Target created — readmitted_30d distribution: "
          f"{dict(df['readmitted_30d'].value_counts())}")

    df = apply_admin_id_mapping(df)
    print(f"[7/11] Admin IDs mapped — nulls: "
          f"admission_type={df['admission_type'].isna().sum()}, "
          f"discharge_disposition={df['discharge_disposition'].isna().sum()}, "
          f"admission_source={df['admission_source'].isna().sum()}")

    df = bucket_diagnosis_codes(df)
    print(f"[8/11] Diagnosis codes bucketed — "
          f"diag_1_group unique values: {df['diag_1_group'].nunique()}")

    df = summarize_medications(df)
    print(f"[9/11] Medication summary built — "
          f"mean num_medications_active: {df['num_medications_active'].mean():.2f}")

    df = add_documentation_indicators(df)
    print(f"[10/11] Documentation indicators added — "
          f"weight_documented=1 count: {df['weight_documented'].sum():,}")

    df = exclude_death_related(df)
    print(f"[11/11] Death-related encounters excluded — final shape: {df.shape}")

    engine = get_engine()
    df.to_sql("cleaned_encounters", engine, if_exists="replace", index=False)
    print(f"\n✓ Wrote {len(df):,} cleaned rows to Postgres table 'cleaned_encounters'")

    return df



if __name__ == "__main__":
    clean_data()