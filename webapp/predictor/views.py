import requests
from django.shortcuts import render, redirect
from .forms import EncounterForm

API_URL = "http://127.0.0.1:8000/predict"


def predict_view(request):
    if request.method == "POST":
        form = EncounterForm(request.POST)
        if form.is_valid():
            payload = dict(form.cleaned_data)
            payload["glyburide-metformin"] = payload.pop("glyburide_metformin")

            try:
                response = requests.post(API_URL, json=payload, timeout=5)
                if response.status_code == 200:
                    # Store the result in the session, NOT in this response —
                    # then redirect. This is what breaks the POST-refresh loop.
                    request.session["prediction_result"] = response.json()
                    request.session["prediction_error"] = None
                else:
                    request.session["prediction_result"] = None
                    request.session["prediction_error"] = (
                        f"API returned {response.status_code}: {response.text}"
                    )
            except requests.exceptions.ConnectionError:
                request.session["prediction_result"] = None
                request.session["prediction_error"] = (
                    "Could not reach the prediction API. Is it running on port 8000?"
                )

            return redirect("predict")  # <-- the key change: redirect, don't render

    else:
        form = EncounterForm()

    # This branch handles BOTH the initial GET and the GET after redirect
    result = request.session.pop("prediction_result", None)
    error = request.session.pop("prediction_error", None)

    return render(request, "predictor/form.html", {
        "form": form,
        "result": result,
        "error": error,
    })