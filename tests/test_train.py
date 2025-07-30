import os
import sys

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import load_dataset, create_model, save_model, load_model


class TestTraining:
    """Test cases for training pipeline."""

    def test_dataset_loading(self):
        """Test dataset loading functionality."""
        X_train, X_test, y_train, y_test = load_dataset()

        # Check if data is loaded correctly
        assert X_train is not None
        assert X_test is not None
        assert y_train is not None
        assert y_test is not None

        # Check shapes
        assert X_train.shape[1] == 8  # California housing has 8 features
        assert X_test.shape[1] == 8
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

        # Check train/test split ratio (approximately 80/20)
        total_samples = len(X_train) + len(X_test)
        train_ratio = len(X_train) / total_samples
        assert 0.75 <= train_ratio <= 0.85

    def test_model_creation(self):
        """Test model creation."""
        model = create_model()

        # Check if model is LinearRegression instance
        assert isinstance(model, LinearRegression)
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')

    def test_model_training(self):
        """Test if the model can be trained and has required attributes."""
        X_train, X_test, y_train, y_test = load_dataset()
        model = create_model()

        # Train model
        model.fit(X_train, y_train)

        # Check if model was trained (coefficients exist)
        assert hasattr(model, 'coef_')
        assert hasattr(model, 'intercept_')
        assert model.coef_ is not None
        assert model.intercept_ is not None

        # Check coefficient shape
        assert model.coef_.shape == (8,)  # 8 features
        assert isinstance(model.intercept_, (float, np.float64))

    def test_model_performance(self):
        """Test if R² score exceeds minimum threshold."""
        from utils import calculate_metrics

        X_train, X_test, y_train, y_test = load_dataset()
        model = create_model()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate R² score
        r2, mse = calculate_metrics(y_test, y_pred)

        # Check if R² score exceeds minimum threshold (0.5)
        assert r2 > 0.5, f"R² score {r2:.4f} is below minimum threshold of 0.5"
        assert mse > 0, "MSE should be positive"

        print(f"Model R² Score: {r2:.4f}")
        print(f"Model MSE: {mse:.4f}")

    def test_model_save_load(self):
        """Test model saving and loading."""
        X_train, X_test, y_train, y_test = load_dataset()
        model = create_model()
        model.fit(X_train, y_train)

        # Save model
        test_path = "test_model.joblib"
        save_model(model, test_path)

        # Check if file exists
        assert os.path.exists(test_path)

        # Load model
        loaded_model = load_model(test_path)

        # Check if loaded model works
        pred_original = model.predict(X_test[:5])
        pred_loaded = loaded_model.predict(X_test[:5])

        # Predictions should be identical
        np.testing.assert_array_almost_equal(pred_original, pred_loaded)

        # Cleanup
        os.remove(test_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])