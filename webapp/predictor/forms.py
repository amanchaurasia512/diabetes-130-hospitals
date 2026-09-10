from django import forms

# ============================================================
# Real category values, pulled directly from Postgres
# (src/data/get_categorical_values.py)
# ============================================================

race_choices = ['AfricanAmerican', 'Asian', 'Caucasian', 'Hispanic', 'Not_Documented', 'Other']
gender_choices = ['Female', 'Male', 'Not_Documented']
age_choices = ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']
payer_code_choices = ['BC', 'CH', 'CM', 'CP', 'DM', 'FR', 'HM', 'MC', 'MD', 'MP', 'Not_Documented', 'OG', 'OT', 'PO', 'SI', 'SP', 'UN', 'WC']
medical_specialty_choices = ['AllergyandImmunology', 'Anesthesiology', 'Anesthesiology-Pediatric', 'Cardiology', 'Cardiology-Pediatric', 'DCPTEAM', 'Dentistry', 'Dermatology', 'Emergency/Trauma', 'Endocrinology', 'Endocrinology-Metabolism', 'Family/GeneralPractice', 'Gastroenterology', 'Gynecology', 'Hematology', 'Hematology/Oncology', 'Hospitalist', 'InfectiousDiseases', 'InternalMedicine', 'Nephrology', 'Neurology', 'Neurophysiology', 'Not_Documented', 'Obsterics&Gynecology-GynecologicOnco', 'Obstetrics', 'ObstetricsandGynecology', 'Oncology', 'Ophthalmology', 'Orthopedics', 'Orthopedics-Reconstructive', 'Osteopath', 'Otolaryngology', 'OutreachServices', 'Pathology', 'Pediatrics', 'Pediatrics-AllergyandImmunology', 'Pediatrics-CriticalCare', 'Pediatrics-EmergencyMedicine', 'Pediatrics-Endocrinology', 'Pediatrics-Hematology-Oncology', 'Pediatrics-InfectiousDiseases', 'Pediatrics-Neurology', 'Pediatrics-Pulmonology', 'Perinatology', 'PhysicalMedicineandRehabilitation', 'PhysicianNotFound', 'Podiatry', 'Proctology', 'Psychiatry', 'Psychiatry-Addictive', 'Psychiatry-Child/Adolescent', 'Psychology', 'Pulmonology', 'Radiologist', 'Radiology', 'Resident', 'Rheumatology', 'Speech', 'SportsMedicine', 'Surgeon', 'Surgery-Cardiovascular', 'Surgery-Cardiovascular/Thoracic', 'Surgery-Colon&Rectal', 'Surgery-General', 'Surgery-Maxillofacial', 'Surgery-Neuro', 'Surgery-Pediatric', 'Surgery-Plastic', 'Surgery-PlasticwithinHeadandNeck', 'Surgery-Thoracic', 'Surgery-Vascular', 'SurgicalSpecialty', 'Urology']
max_glu_serum_choices = ['>200', '>300', 'Norm', 'Not_Documented']
A1Cresult_choices = ['>7', '>8', 'Norm', 'Not_Documented']
metformin_choices = ['Down', 'No', 'Steady', 'Up']
repaglinide_choices = ['Down', 'No', 'Steady', 'Up']
nateglinide_choices = ['Down', 'No', 'Steady', 'Up']
glimepiride_choices = ['Down', 'No', 'Steady', 'Up']
glipizide_choices = ['Down', 'No', 'Steady', 'Up']
glyburide_choices = ['Down', 'No', 'Steady', 'Up']
pioglitazone_choices = ['Down', 'No', 'Steady', 'Up']
rosiglitazone_choices = ['Down', 'No', 'Steady', 'Up']
insulin_choices = ['Down', 'No', 'Steady', 'Up']
glyburide_metformin_choices = ['Down', 'No', 'Steady', 'Up']
acarbose_choices = ['Down', 'No', 'Steady', 'Up']
change_choices = ['Ch', 'No']
diabetesMed_choices = ['No', 'Yes']
admission_type_choices = ['Elective', 'Emergency', 'Newborn', 'Not Available', 'Not Mapped', 'Not_Documented', 'Trauma Center', 'Urgent']
discharge_disposition_choices = ['Admitted as an inpatient to this hospital', 'Discharged to home', 'Discharged/transferred to ICF', 'Discharged/transferred to SNF', 'Discharged/transferred to a federal health care facility.', 'Discharged/transferred to a long term care hospital.', 'Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare.', 'Discharged/transferred to another rehab fac including rehab units of a hospital .', 'Discharged/transferred to another short term hospital', 'Discharged/transferred to another type of inpatient care institution', 'Discharged/transferred to home under care of Home IV provider', 'Discharged/transferred to home with home health service', 'Discharged/transferred within this institution to Medicare approved swing bed', 'Discharged/transferred/referred another institution for outpatient services', 'Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital', 'Discharged/transferred/referred to this institution for outpatient services', 'Hospice / home', 'Hospice / medical facility', 'Left AMA', 'Neonate discharged to another hospital for neonatal aftercare', 'Not Mapped', 'Not_Documented', 'Still patient or expected to return for outpatient services']
admission_source_choices = ['Clinic Referral', 'Court/Law Enforcement', 'Emergency Room', 'Extramural Birth', 'HMO Referral', 'Normal Delivery', 'Not Available', 'Not Mapped', 'Not_Documented', 'Physician Referral', 'Sick Baby', 'Transfer from Ambulatory Surgery Center', 'Transfer from a Skilled Nursing Facility (SNF)', 'Transfer from a hospital', 'Transfer from another health care facility', 'Transfer from critial access hospital', 'Transfer from hospital inpt/same fac reslt in a sep claim']
diag_group_choices = ['Blood diseases', 'Circulatory diseases', 'Congenital anomalies', 'Digestive diseases', 'Endocrine, nutritional & metabolic diseases', 'External causes (E-code)', 'Genitourinary diseases', 'Infectious & parasitic diseases', 'Injury & poisoning', 'Mental disorders', 'Musculoskeletal system', 'Neoplasms', 'Nervous system & sense organs', 'Not_Documented', 'Pregnancy & childbirth', 'Respiratory diseases', 'Skin & subcutaneous tissue', 'Supplementary factors (V-code)', 'Symptoms/signs/ill-defined conditions']


def as_choices(values):
    """Django ChoiceField needs (value, label) pairs — same string for both here."""
    return [(v, v) for v in values]


class EncounterForm(forms.Form):
    # --- Numeric fields (unchanged) ---
    time_in_hospital = forms.IntegerField(min_value=1, max_value=14)
    num_lab_procedures = forms.IntegerField(min_value=0)
    num_procedures = forms.IntegerField(min_value=0)
    num_medications = forms.IntegerField(min_value=0)
    number_outpatient = forms.IntegerField(min_value=0)
    number_emergency = forms.IntegerField(min_value=0)
    number_inpatient = forms.IntegerField(min_value=0)
    number_diagnoses = forms.IntegerField(min_value=0)
    num_medications_active = forms.IntegerField(min_value=0)
    num_medications_up = forms.IntegerField(min_value=0)
    num_medications_down = forms.IntegerField(min_value=0)
    num_medications_steady = forms.IntegerField(min_value=0)
    weight_documented = forms.ChoiceField(choices=[(0, "No"), (1, "Yes")])

    # --- Categorical fields — now real dropdowns ---
    race = forms.ChoiceField(choices=as_choices(race_choices))
    gender = forms.ChoiceField(choices=as_choices(gender_choices))
    age = forms.ChoiceField(choices=as_choices(age_choices))
    payer_code = forms.ChoiceField(choices=as_choices(payer_code_choices))
    medical_specialty = forms.ChoiceField(choices=as_choices(medical_specialty_choices))
    max_glu_serum = forms.ChoiceField(choices=as_choices(max_glu_serum_choices))
    A1Cresult = forms.ChoiceField(choices=as_choices(A1Cresult_choices))
    metformin = forms.ChoiceField(choices=as_choices(metformin_choices))
    repaglinide = forms.ChoiceField(choices=as_choices(repaglinide_choices))
    nateglinide = forms.ChoiceField(choices=as_choices(nateglinide_choices))
    glimepiride = forms.ChoiceField(choices=as_choices(glimepiride_choices))
    glipizide = forms.ChoiceField(choices=as_choices(glipizide_choices))
    glyburide = forms.ChoiceField(choices=as_choices(glyburide_choices))
    pioglitazone = forms.ChoiceField(choices=as_choices(pioglitazone_choices))
    rosiglitazone = forms.ChoiceField(choices=as_choices(rosiglitazone_choices))
    insulin = forms.ChoiceField(choices=as_choices(insulin_choices))
    glyburide_metformin = forms.ChoiceField(
        choices=as_choices(glyburide_metformin_choices), label="glyburide-metformin"
    )
    acarbose = forms.ChoiceField(choices=as_choices(acarbose_choices))
    change = forms.ChoiceField(choices=as_choices(change_choices))
    diabetesMed = forms.ChoiceField(choices=as_choices(diabetesMed_choices))
    admission_type = forms.ChoiceField(choices=as_choices(admission_type_choices))
    discharge_disposition = forms.ChoiceField(choices=as_choices(discharge_disposition_choices))
    admission_source = forms.ChoiceField(choices=as_choices(admission_source_choices))
    diag_1_group = forms.ChoiceField(choices=as_choices(diag_group_choices))
    diag_2_group = forms.ChoiceField(choices=as_choices(diag_group_choices))
    diag_3_group = forms.ChoiceField(choices=as_choices(diag_group_choices))