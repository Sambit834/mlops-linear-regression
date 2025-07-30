import os
import sys

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import fetch_data_split, build_regressor, persist_model, restore_model


class TestTraining:
    """Test cases for training pipeline."""

    def test_dataset_loading(self):
        """Test dataset loading functionality."""
        train_x, test_x, train_y, test_y = fetch_data_split()

        # Check if data is loaded correctly
        assert train_x is not None
        assert test_x is not None
        assert train_y is not None
        assert test_y is not None

        # Check shapes
        assert train_x.shape[1] == 8  # California housing has 8 features
        assert test_x.shape[1] == 8
        assert len(train_x) == len(train_y)
        assert len(test_x) == len(test_y)

        # Check train/test split ratio (approximately 80/20)
        total_samples = len(train_x) + len(test_x)
        train_ratio = len(train_x) / total_samples
        assert 0.75 <= train_ratio <= 0.85

    def test_model_creation(self):
        """Test model creation."""
        reg = build_regressor()

        # Check if model is LinearRegression instance
        assert isinstance(reg, LinearRegression)
        assert hasattr(reg, 'fit')
        assert hasattr(reg, 'predict')

    def test_model_training(self):
        """Test if the model can be trained and has required attributes."""
        train_x, test_x, train_y, test_y = fetch_data_split()
        reg = build_regressor()

        # Train model
        reg.fit(train_x, train_y)

        # Check if model was trained (coefficients exist)
        assert hasattr(reg, 'coef_')
        assert hasattr(reg, 'intercept_')
        assert reg.coef_ is not None
        assert reg.intercept_ is not None

        # Check coefficient shape
        assert reg.coef_.shape == (8,)  # 8 features
        assert isinstance(reg.intercept_, (float, np.float64))

    def test_model_performance(self):
        """Test if R² score exceeds minimum threshold."""
        from utils import regression_metrics

        train_x, test_x, train_y, test_y = fetch_data_split()
        reg = build_regressor()
        reg.fit(train_x, train_y)

        # Make predictions
        preds = reg.predict(test_x)

        # Calculate R² score
        r2, mse = regression_metrics(test_y, preds)

        # Check if R² score exceeds minimum threshold (0.5)
        assert r2 > 0.5, f"R² score {r2:.4f} is below minimum threshold of 0.5"
        assert mse > 0, "MSE should be positive"

        print(f"Model R² Score: {r2:.4f}")
        print(f"Model MSE: {mse:.4f}")

    def test_model_save_load(self):
        """Test model saving and loading."""
        train_x, test_x, train_y, test_y = fetch_data_split()
        reg = build_regressor()
        reg.fit(train_x, train_y)

        # Save model
        test_path = "test_model.joblib"
        persist_model(reg, test_path)

        # Check if file exists
        assert os.path.exists(test_path)

        # Load model
        loaded_reg = restore_model(test_path)

        # Check if loaded model works
        pred_original = reg.predict(test_x[:5])
        pred_loaded = loaded_reg.predict(test_x[:5])

        # Predictions should be identical
        np.testing.assert_array_almost_equal(pred_original, pred_loaded)

        # Cleanup
        os.remove(test_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])