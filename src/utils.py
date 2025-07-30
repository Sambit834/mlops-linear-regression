import os

import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


def load_dataset():
    """Load and split California Housing dataset."""
    data = fetch_california_housing()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def create_model():
    """Create LinearRegression model instance."""
    return LinearRegression()


def save_model(model, filepath):
    """Save model using joblib."""
    # Only create directory if filepath contains a directory
    directory = os.path.dirname(filepath)
    if directory:  # Only create directory if it's not empty
        os.makedirs(directory, exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath):
    """Load model using joblib."""
    return joblib.load(filepath)


def calculate_metrics(y_true, y_pred):
    """Calculate R2 score and MSE."""
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return r2, mse


def quantize_to_uint16(values, scale_factor=None):
    """Quantize float values to uint16 with adaptive scaling."""
    if np.all(values == 0):
        return np.zeros(values.shape, dtype=np.uint16), 0.0, 0.0, 1.0
    if scale_factor is None:
        abs_max = np.abs(values).max()
        if abs_max > 0:
            scale_factor = 65500.0 / abs_max
        else:
            scale_factor = 1.0
    scaled_values = values * scale_factor
    min_val, max_val = scaled_values.min(), scaled_values.max()
    if max_val == min_val:
        quantized = np.full(values.shape, 32767, dtype=np.uint16)
        return quantized, min_val, max_val, scale_factor
    value_range = max_val - min_val
    normalized = ((scaled_values - min_val) / value_range * 65535)
    normalized = np.clip(normalized, 0, 65535)
    quantized = normalized.astype(np.uint16)
    return quantized, min_val, max_val, scale_factor


def dequantize_from_uint16(quantized, min_val, max_val, scale_factor):
    """Dequantize uint16 values back to float using stored metadata."""
    value_range = max_val - min_val
    if value_range == 0:
        return np.full(quantized.shape, min_val / scale_factor)
    scaled = (quantized.astype(np.float32) / 65535.0) * value_range + min_val
    return scaled / scale_factor


def quantize_to_uint8(values, scale_factor=None):
    """Quantize float values to uint8 with adaptive scaling."""
    if np.all(values == 0):
        return np.zeros(values.shape, dtype=np.uint8), 0.0, 0.0, 1.0
    if scale_factor is None:
        abs_max = np.abs(values).max()
        if abs_max > 0:
            scale_factor = 250.0 / abs_max
        else:
            scale_factor = 1.0
    scaled_values = values * scale_factor
    min_val, max_val = scaled_values.min(), scaled_values.max()
    if max_val == min_val:
        quantized = np.full(values.shape, 127, dtype=np.uint8)
        return quantized, min_val, max_val, scale_factor
    value_range = max_val - min_val
    normalized = ((scaled_values - min_val) / value_range * 255)
    normalized = np.clip(normalized, 0, 255)
    quantized = normalized.astype(np.uint8)
    return quantized, min_val, max_val, scale_factor


def dequantize_from_uint8(quantized, min_val, max_val, scale_factor):
    """Dequantize uint8 values back to float using stored metadata."""
    value_range = max_val - min_val
    if value_range == 0:
        return np.full(quantized.shape, min_val / scale_factor)
    scaled = (quantized.astype(np.float32) / 255.0) * value_range + min_val
    return scaled / scale_factor