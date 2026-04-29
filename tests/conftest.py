import numpy as np
from unittest.mock import MagicMock, patch

mock_model = MagicMock()
mock_model.predict_proba.return_value = np.array([[0.6, 0.4]])

patch("noshow_iq.api.get_model", return_value=mock_model).start()
patch("noshow_iq.api.get_db").start()
patch("noshow_iq.api.model", mock_model).start()