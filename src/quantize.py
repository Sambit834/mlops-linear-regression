import os

import joblib
import numpy as np

from utils import quantize_to_uint16, dequantize_from_uint16, quantize_to_uint8, dequantize_from_uint8, load_model


def main():
    """Main quantization function."""
    print("Loading trained model...")
    model = load_model("models/linear_regression_model.joblib")

    # Extract coefficients and intercept
    coef = model.coef_
    intercept = model.intercept_

    print(f"Original coefficients shape: {coef.shape}")
    print(f"Original intercept: {intercept}")
    print(f"Original coef values: {coef}")

    # Save raw parameters
    raw_params = {
        'coef': coef,
        'intercept': intercept
    }
    os.makedirs("models", exist_ok=True)
    joblib.dump(raw_params, "models/unquant_params.joblib")

    quant_coef16, coef16_min, coef16_max, coef16_scale = quantize_to_uint16(coef)
    quant_intercept16, int16_min, int16_max, int16_scale = quantize_to_uint16(np.array([intercept]))

    print(f"\nQuantizing intercept...")
    print(f"Intercept value: {intercept:.8f}")
    print(f"Intercept scale factor: {int16_scale:.2f}")

    # Save quantized parameters with metadata
    quant_params = {
        'quant_coef16': quant_coef16,
        'coef16_min': coef16_min,
        'coef16_max': coef16_max,
        'coef16_scale': coef16_scale,
        'quant_intercept16': quant_intercept16[0],
        'int16_min': int16_min,
        'int16_max': int16_max,
        'int16_scale': int16_scale
    }
    joblib.dump(quant_params, "models/quant_params.joblib")
    print("Quantized parameters saved to models/quant_params.joblib")

    # Print model size difference

    orig_size = os.path.getsize("models/linear_regression_model.joblib")
    quant_size = os.path.getsize("models/quant_params.joblib")
    print(f"\nModel size before quantization: {orig_size / 1024:.2f} KB")
    print(f"Model size after quantization:  {quant_size / 1024:.2f} KB")
    print(f"Size reduction:                {(orig_size - quant_size) / 1024:.2f} KB")

    # Dequantize for inference
    dequant_coef16 = dequantize_from_uint16(quant_coef16, coef16_min, coef16_max, coef16_scale)
    dequant_intercept16 = \
    dequantize_from_uint16(np.array([quant_params['quant_intercept16']]), int16_min, int16_max, int16_scale)[0]

    # Calculate coefficient and intercept errors
    coef_error = np.abs(coef - dequant_coef16).max()
    intercept_error = np.abs(intercept - dequant_intercept16)
    print(f"Max coefficient error (16-bit): {coef_error:.8f}")
    print(f"Intercept error (16-bit): {intercept_error:.8f}")

    # Inference test and output formatting
    from utils import load_dataset
    X_train, X_test, y_train, y_test = load_dataset()

    # First 5 samples
    original_pred = model.predict(X_test[:5])
    manual_original_pred = X_test[:5] @ coef + intercept
    manual_dequant_pred = X_test[:5] @ dequant_coef16 + dequant_intercept16

    print("\nInference Test (first 5 samples):")
    print(f"Original predictions (sklearn): {original_pred}")
    print(f"Manual original predictions:    {manual_original_pred}")
    print(f"Manual dequant predictions:     {manual_dequant_pred}")

    print("\nDifferences:")
    print(f"Sklearn vs manual original: {np.abs(original_pred - manual_original_pred)}")
    print(f"Original vs dequant manual:  {np.abs(manual_original_pred - manual_dequant_pred)}")
    original_vs_dequant_diff = np.abs(original_pred - manual_dequant_pred)
    print(f"Absolute differences: {original_vs_dequant_diff}")
    print(f"Max difference: {original_vs_dequant_diff.max()}")
    print(f"Mean difference: {original_vs_dequant_diff.mean()}")
    max_diff = original_vs_dequant_diff.max()
    if max_diff < 0.1:
        print(f"Quantization quality is good (max diff: {max_diff:.6f})")
    elif max_diff < 1.0:
        print(f"Quantization quality is acceptable (max diff: {max_diff:.6f})")
    else:
        print(f"Quantization quality is poor (max diff: {max_diff:.6f})")

    # Calculate prediction errors for quantized (dequantized) model
    max_pred_error = np.max(np.abs(y_test - (X_test @ dequant_coef16 + dequant_intercept16)))
    mean_pred_error = np.mean(np.abs(y_test - (X_test @ dequant_coef16 + dequant_intercept16)))
    print(f"Max Prediction Error (quantized 16-bit): {max_pred_error:.4f}")
    print(f"Mean Prediction Error (quantized 16-bit): {mean_pred_error:.4f}")

    # 8-bit quantization
    quant_coef8, coef8_min, coef8_max, coef8_scale = quantize_to_uint8(coef)
    quant_intercept8, int8_min, int8_max, int8_scale = quantize_to_uint8(np.array([intercept]))
    print(f"\nQuantizing intercept (8-bit)...")
    print(f"Intercept value: {intercept:.8f}")
    print(f"Intercept scale factor (8-bit): {int8_scale:.2f}")
    quant_params8 = {
        'quant_coef8': quant_coef8,
        'coef8_min': coef8_min,
        'coef8_max': coef8_max,
        'coef8_scale': coef8_scale,
        'quant_intercept8': quant_intercept8[0],
        'int8_min': int8_min,
        'int8_max': int8_max,
        'int8_scale': int8_scale
    }
    joblib.dump(quant_params8, "models/quant_params8.joblib")
    print("Quantized parameters (8-bit) saved to models/quant_params8.joblib")

    quant_size8 = os.path.getsize("models/quant_params8.joblib")
    print(f"Model size after 8-bit quantization:  {quant_size8 / 1024:.2f} KB")
    print(f"Size reduction (8-bit):                {(orig_size - quant_size8) / 1024:.2f} KB")

    # Dequantize for inference (8-bit)
    dequant_coef8 = dequantize_from_uint8(quant_coef8, coef8_min, coef8_max, coef8_scale)
    dequant_intercept8 = \
    dequantize_from_uint8(np.array([quant_params8['quant_intercept8']]), int8_min, int8_max, int8_scale)[0]

    # Calculate coefficient and intercept errors (8-bit)
    coef_error8 = np.abs(coef - dequant_coef8).max()
    intercept_error8 = np.abs(intercept - dequant_intercept8)
    print(f"Max coefficient error (8-bit): {coef_error8:.8f}")
    print(f"Intercept error (8-bit): {intercept_error8:.8f}")

    # Inference test and output formatting (8-bit)
    from utils import load_dataset
    X_train, X_test, y_train, y_test = load_dataset()
    manual_dequant_pred8 = X_test[:5] @ dequant_coef8 + dequant_intercept8
    print("\nManual dequant predictions (8-bit):     ", manual_dequant_pred8)
    original_vs_dequant_diff8 = np.abs(model.predict(X_test[:5]) - manual_dequant_pred8)
    print(f"Absolute differences (8-bit): {original_vs_dequant_diff8}")
    print(f"Max difference (8-bit): {original_vs_dequant_diff8.max()}")
    print(f"Mean difference (8-bit): {original_vs_dequant_diff8.mean()}")
    max_diff8 = original_vs_dequant_diff8.max()
    if max_diff8 < 0.1:
        print(f"Quantization quality (8-bit) is good (max diff: {max_diff8:.6f})")
    elif max_diff8 < 1.0:
        print(f"Quantization quality (8-bit) is acceptable (max diff: {max_diff8:.6f})")
    else:
        print(f"Quantization quality (8-bit) is poor (max diff: {max_diff8:.6f})")

    # Calculate prediction errors for quantized (dequantized) model (8-bit)
    max_pred_error8 = np.max(np.abs(y_test - (X_test @ dequant_coef8 + dequant_intercept8)))
    mean_pred_error8 = np.mean(np.abs(y_test - (X_test @ dequant_coef8 + dequant_intercept8)))
    print(f"Max Prediction Error (quantized 8-bit): {max_pred_error8:.4f}")
    print(f"Mean Prediction Error (quantized 8-bit): {mean_pred_error8:.4f}")
    print("\nQuantization completed successfully!\n")

    # Calculate R2 and MSE for 8-bit quantized model
    y_pred8 = X_test @ dequant_coef8 + dequant_intercept8
    from utils import calculate_metrics
    r2_8, mse_8 = calculate_metrics(y_test, y_pred8)
    print(f"R2 score (quantized 8-bit model): {r2_8:.4f}")
    print(f"MSE (quantized 8-bit model): {mse_8:.4f}")

    # Calculate R2 and MSE for 16-bit quantized model
    y_pred16 = X_test @ dequant_coef16 + dequant_intercept16
    from utils import calculate_metrics
    r2_16, mse_16 = calculate_metrics(y_test, y_pred16)
    print(f"R2 score (quantized 16-bit model): {r2_16:.4f}")
    print(f"MSE (quantized 16-bit model): {mse_16:.4f}")


if __name__ == "__main__":
    main()