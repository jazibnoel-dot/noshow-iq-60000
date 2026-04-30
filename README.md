noshow-iq-60000
==============================

A mid term Project of Jazib

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

API Usage
---------

Start the FastAPI app and then send a POST request to `/predict` with JSON input. The service derives the model features from the appointment timestamps.

Start the app with `uvicorn`:

```powershell
uvicorn noshow_iq.api:app --host 0.0.0.0 --port 7860
```

Then send a request.

Example PowerShell request:

```powershell
$body = @{
  age = 45
  gender = "F"
  scheduled_day = "2025-04-10T08:00:00"
  appointment_day = "2025-04-20T10:00:00"
  scholarship = 0
  hipertension = 1
  diabetes = 0
  alcoholism = 0
  handcap = 0
  sms_received = 1
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:7860/predict -Method Post -Body $body -ContentType "application/json"
```

If you prefer `curl.exe` in PowerShell:

```powershell
curl.exe -X POST http://localhost:7860/predict `
  -H "Content-Type: application/json" `
  -d '{
    "age": 45,
    "gender": "F",
    "scheduled_day": "2025-04-10T08:00:00",
    "appointment_day": "2025-04-20T10:00:00",
    "scholarship": 0,
    "hipertension": 1,
    "diabetes": 0,
    "alcoholism": 0,
    "handcap": 0,
    "sms_received": 1
  }'
```

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
