import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

"""
This file contains unit tests for the FastAPI application.

Purpose of these tests:
1. Check if API endpoints are working correctly.
2. Avoid using the real model and database (we use mock objects instead).
3. Ensure API returns correct structure and values.

We use:
- MagicMock → to fake model and database
- patch → to replace real functions with fake ones
- TestClient → to simulate API requests
"""

# Sample input data (same format as API expects)
SAMPLE = {
    "age": 30, "scholarship": 0, "hipertension": 0,
    "diabetes": 0, "alcoholism": 0, "handcap": 0,
    "sms_received": 1, "days_in_advance": 5, "hour_of_booking": 10,
}

# Create a fake (mock) model
mock_model = MagicMock()

# Mock the model's predict_proba function
# It returns probabilities: [prob_not_no_show, prob_no_show]
mock_model.predict_proba.return_value = np.array([[0.6, 0.4]])

# Create a fake database collection
mock_col = MagicMock()

# Mock database functions (so no real DB is used)
mock_col.insert_one = MagicMock()
mock_col.find.return_value.sort.return_value.limit.return_value = []
mock_col.aggregate.return_value = []

# Create a fake database object
mock_db = MagicMock()

# Whenever db["collection_name"] is called, return mock_col
mock_db.__getitem__ = MagicMock(return_value=mock_col)

# Replace real functions with mock versions
with patch("noshow_iq.api.get_model", return_value=mock_model), \
        patch("noshow_iq.api.get_db", return_value=mock_db):

    # Import API module after patching
    import noshow_iq.api as api_module

    # Set global model to mock model
    api_module.model = mock_model

    # Create test client to call API endpoints
    client = TestClient(api_module.app)


def test_health_returns_200():
    """
    Test if /health endpoint returns HTTP 200 (success).
    """
    assert client.get("/health").status_code == 200


def test_health_has_status_key():
    """
    Test if /health response contains 'status' key.
    """
    assert "status" in client.get("/health").json()


def test_predict_returns_200():
    """
    Test if /predict endpoint returns HTTP 200.
    """
    api_module.model = mock_model
    assert client.post("/predict", json=SAMPLE).status_code == 200


def test_predict_has_risk_level():
    """
    Test if prediction response contains 'risk_level'.
    """
    api_module.model = mock_model
    assert "risk_level" in client.post("/predict", json=SAMPLE).json()


def test_predict_risk_level_valid():
    """
    Test if 'risk_level' is either 'high' or 'low'.
    """
    api_module.model = mock_model
    assert client.post("/predict", json=SAMPLE).json()["risk_level"] in ["high", "low"]


def test_predict_has_recommendation():
    """
    Test if prediction response contains 'recommendation'.
    """
    api_module.model = mock_model
    assert "recommendation" in client.post("/predict", json=SAMPLE).json()