import numpy as np
from unittest.mock import MagicMock, patch

"""
This code sets up mock (fake) objects for testing purposes.

Purpose:
1. Replace the real machine learning model with a fake one.
2. Avoid loading the actual model file.
3. Avoid connecting to a real database.
4. Make testing faster and safer.

We use:
- MagicMock → to create fake objects
- patch → to temporarily replace real functions/variables
"""

# Create a fake (mock) model
mock_model = MagicMock()

# Mock the model's predict_proba method
# This simulates prediction output:
# [[0.6, 0.4]] means:
# 60% probability of "No-show = 0"
# 40% probability of "No-show = 1"
mock_model.predict_proba.return_value = np.array([[0.6, 0.4]])

# Replace the real get_model() function with our mock model
# So whenever the app tries to load the model, it gets this fake one
patch("noshow_iq.api.get_model", return_value=mock_model).start()

# Replace the database connection function with a mock
# This prevents real database connections during testing
patch("noshow_iq.api.get_db").start()

# Replace the global model variable inside the API with mock_model
# Ensures the API uses the fake model instead of a real one
patch("noshow_iq.api.model", mock_model).start()