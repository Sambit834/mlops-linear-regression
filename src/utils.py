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


def quantize_to_uint8(values, scale_factor=None):
    """Quantize float values to uint8 with adaptive scaling."""
    # Handle edge cases
    if np.all(values == 0):
        return np.zeros(values.shape, dtype=np.uint8), 0.0, 0.0, 1.0

    if scale_factor is None:
        abs_max = np.abs(values).max()
        if abs_max > 0:
            # Scale so that max absolute value uses most of the uint8 range
            scale_factor = 200.0 / abs_max
        else:
            scale_factor = 1.0

    scaled_values = values * scale_factor
    min_val, max_val = scaled_values.min(), scaled_values.max()

    # Check for constant values
    if max_val == min_val:
        # If all values are the same, return middle value (127) for all
        quantized = np.full(values.shape, 127, dtype=np.uint8)
        return quantized, min_val, max_val, scale_factor

    # Normalize to 0-255 range (avoiding division by zero)
    value_range = max_val - min_val
    normalized = ((scaled_values - min_val) / value_range * 255)

    # Clip to valid range and convert to uint8
    normalized = np.clip(normalized, 0, 255)
    quantized = normalized.astype(np.uint8)

    return quantized, min_val, max_val, scale_factor


def quantize_to_uint8_individual(values):
    quantized = np.zeros(values.shape, dtype=np.uint8)
    metadata = []

    for i, val in enumerate(values):
        if val == 0:
            quantized[i] = 127  # Middle value for zero
            metadata.append({'min_val': 0.0, 'max_val': 0.0, 'scale': 1.0})
        else:
            # Scale individual value to use full uint8 range
            abs_val = abs(val)
            scale_factor = 127.0 / abs_val  # Use 127 to preserve sign

            # Map to 0-255 range: negative -> 0-126, positive -> 128-255
            if val < 0:
                quantized_val = int(127 - (abs_val * scale_factor))
            else:
                quantized_val = int(128 + (abs_val * scale_factor))

            quantized[i] = np.clip(quantized_val, 0, 255)
            metadata.append({
                'min_val': val,
                'max_val': val,
                'scale': scale_factor,
                'original': val
            })

    return quantized, metadata


def dequantize_from_uint8_individual(quantized_values, metadata):
    dequantized = np.zeros(quantized_values.shape, dtype=np.float32)

    for i, (quant_val, meta) in enumerate(zip(quantized_values, metadata)):
        if meta['scale'] == 1.0:  # Was zero
            dequantized[i] = 0.0
        else:
            # Reverse the quantization process
            if quant_val <= 127:  # Was negative
                abs_val = (127 - quant_val) / meta['scale']
                dequantized[i] = -abs_val
            else:  # Was positive
                abs_val = (quant_val - 128) / meta['scale']
                dequantized[i] = abs_val

    return dequantized


def dequantize_from_uint8(quantized_values, min_val, max_val, scale_factor):
    # Handle edge case where min_val == max_val
    if max_val == min_val:
        return np.full(quantized_values.shape, min_val / scale_factor, dtype=np.float32)

    # Denormalize from 0-255 range
    value_range = max_val - min_val
    denormalized = (quantized_values.astype(np.float32) / 255.0) * value_range + min_val
    # Descale
    original_values = denormalized / scale_factor
    return original_values
