import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

SAMPLE = {
    "age": 30, "scholarship": 0, "hipertension": 0,
    "diabetes": 0, "alcoholism": 0, "handcap": 0,
    "sms_received": 1, "days_in_advance": 5, "hour_of_booking": 10,
}

mock_model = MagicMock()
mock_model.predict_proba.return_value = np.array([[0.6, 0.4]])
mock_col = MagicMock()
mock_col.insert_one = MagicMock()
mock_col.find.return_value.sort.return_value.limit.return_value = []
mock_col.aggregate.return_value = []
mock_db = MagicMock()
mock_db.__getitem__ = MagicMock(return_value=mock_col)

with patch("noshow_iq.api.get_model", return_value=mock_model), \
        patch("noshow_iq.api.get_db", return_value=mock_db):
    import noshow_iq.api as api_module
    api_module.model = mock_model
    client = TestClient(api_module.app)


def test_health_returns_200():
    assert client.get("/health").status_code == 200


def test_health_has_status_key():
    assert "status" in client.get("/health").json()


def test_predict_returns_200():
    api_module.model = mock_model
    assert client.post("/predict", json=SAMPLE).status_code == 200


def test_predict_has_risk_level():
    api_module.model = mock_model
    assert "risk_level" in client.post("/predict", json=SAMPLE).json()


def test_predict_risk_level_valid():
    api_module.model = mock_model
    assert client.post("/predict", json=SAMPLE).json()["risk_level"] in ["high", "low"]


def test_predict_has_recommendation():
    api_module.model = mock_model
    assert "recommendation" in client.post("/predict", json=SAMPLE).json()