import os

import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


def fetch_data_split():
    """Fetch and split California Housing dataset."""
    housing = fetch_california_housing()
    features, targets = housing.data, housing.target
    train_x, test_x, train_y, test_y = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )
    return train_x, test_x, train_y, test_y


def build_regressor():
    """Instantiate LinearRegression model."""
    return LinearRegression()


def persist_model(regressor, out_path):
    """Persist model using joblib."""
    dir_path = os.path.dirname(out_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    joblib.dump(regressor, out_path)


def restore_model(in_path):
    """Restore model using joblib."""
    return joblib.load(in_path)


def regression_metrics(y_actual, y_predicted):
    """Compute R2 and MSE metrics."""
    r2_val = r2_score(y_actual, y_predicted)
    mse_val = mean_squared_error(y_actual, y_predicted)
    return r2_val, mse_val


def float_to_uint16(arr, scale=None):
    """Quantize float array to uint16 with scaling."""
    if np.all(arr == 0):
        return np.zeros(arr.shape, dtype=np.uint16), 0.0, 0.0, 1.0
    if scale is None:
        max_abs = np.abs(arr).max()
        if max_abs > 0:
            scale = 65500.0 / max_abs
        else:
            scale = 1.0
    arr_scaled = arr * scale
    arr_min, arr_max = arr_scaled.min(), arr_scaled.max()
    if arr_max == arr_min:
        quant = np.full(arr.shape, 32767, dtype=np.uint16)
        return quant, arr_min, arr_max, scale
    rng = arr_max - arr_min
    normed = ((arr_scaled - arr_min) / rng * 65535)
    normed = np.clip(normed, 0, 65535)
    quant = normed.astype(np.uint16)
    return quant, arr_min, arr_max, scale


def uint16_to_float(quant, arr_min, arr_max, scale):
    """Dequantize uint16 array to float using metadata."""
    rng = arr_max - arr_min
    if rng == 0:
        return np.full(quant.shape, arr_min / scale)
    arr_scaled = (quant.astype(np.float32) / 65535.0) * rng + arr_min
    return arr_scaled / scale


def float_to_uint8(arr, scale=None):
    """Quantize float array to uint8 with scaling."""
    if np.all(arr == 0):
        return np.zeros(arr.shape, dtype=np.uint8), 0.0, 0.0, 1.0
    if scale is None:
        max_abs = np.abs(arr).max()
        if max_abs > 0:
            scale = 250.0 / max_abs
        else:
            scale = 1.0
    arr_scaled = arr * scale
    arr_min, arr_max = arr_scaled.min(), arr_scaled.max()
    if arr_max == arr_min:
        quant = np.full(arr.shape, 127, dtype=np.uint8)
        return quant, arr_min, arr_max, scale
    rng = arr_max - arr_min
    normed = ((arr_scaled - arr_min) / rng * 255)
    normed = np.clip(normed, 0, 255)
    quant = normed.astype(np.uint8)
    return quant, arr_min, arr_max, scale


def uint8_to_float(quant, arr_min, arr_max, scale):
    """Dequantize uint8 array to float using metadata."""
    rng = arr_max - arr_min
    if rng == 0:
        return np.full(quant.shape, arr_min / scale)
    arr_scaled = (quant.astype(np.float32) / 255.0) * rng + arr_min
    return arr_scaled / scale
