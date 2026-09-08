from django import forms


class EncounterForm(forms.Form):
    # Numeric fields
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

    # Categorical fields — plain text for now, upgraded to real dropdowns
    # once we have the exact category list from the notebook
    race = forms.CharField(max_length=50)
    gender = forms.CharField(max_length=20)
    age = forms.CharField(max_length=20)
    payer_code = forms.CharField(max_length=20)
    medical_specialty = forms.CharField(max_length=50)
    max_glu_serum = forms.CharField(max_length=20)
    A1Cresult = forms.CharField(max_length=20)
    metformin = forms.CharField(max_length=20)
    repaglinide = forms.CharField(max_length=20)
    nateglinide = forms.CharField(max_length=20)
    glimepiride = forms.CharField(max_length=20)
    glipizide = forms.CharField(max_length=20)
    glyburide = forms.CharField(max_length=20)
    pioglitazone = forms.CharField(max_length=20)
    rosiglitazone = forms.CharField(max_length=20)
    insulin = forms.CharField(max_length=20)
    glyburide_metformin = forms.CharField(max_length=20, label="glyburide-metformin")
    acarbose = forms.CharField(max_length=20)
    change = forms.CharField(max_length=20)
    diabetesMed = forms.CharField(max_length=20)
    admission_type = forms.CharField(max_length=30)
    discharge_disposition = forms.CharField(max_length=60)
    admission_source = forms.CharField(max_length=40)
    diag_1_group = forms.CharField(max_length=50)
    diag_2_group = forms.CharField(max_length=50)
    diag_3_group = forms.CharField(max_length=50)
    