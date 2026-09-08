from django.shortcuts import render

# Create your views here.
import requests
from django.shortcuts import render
from .forms import EncounterForm

API_URL = "http://127.0.0.1:8000/predict"


def predict_view(request):
    result = None
    error = None

    if request.method == "POST":
        form = EncounterForm(request.POST)
        if form.is_valid():
            payload = dict(form.cleaned_data)
            # Django can't have a field literally named "glyburide-metformin"
            # (hyphens aren't valid Python identifiers), so we map it back
            # to the key the API actually expects, right before sending.
            payload["glyburide-metformin"] = payload.pop("glyburide_metformin")

            try:
                response = requests.post(API_URL, json=payload, timeout=5)
                if response.status_code == 200:
                    result = response.json()
                else:
                    error = f"API returned {response.status_code}: {response.text}"
            except requests.exceptions.ConnectionError:
                error = "Could not reach the prediction API. Is it running on port 8000?"
    else:
        form = EncounterForm()

    return render(request, "predictor/form.html", {
        "form": form,
        "result": result,
        "error": error,
    })