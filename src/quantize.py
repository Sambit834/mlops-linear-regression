import os

import joblib
import numpy as np

from utils import float_to_uint16, uint16_to_float, float_to_uint8, uint8_to_float, restore_model


def quantize_main():
    """Main quantization entry point."""
    print("Restoring trained regressor...")
    reg = restore_model("artifacts/linear_regression_model.joblib")

    # Extract coefficients and intercept
    weights = reg.coef_
    bias = reg.intercept_

    print(f"Original weights shape: {weights.shape}")
    print(f"Original bias: {bias}")
    print(f"Original weight values: {weights}")

    # Save raw parameters
    params_raw = {
        'weights': weights,
        'bias': bias
    }
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(params_raw, "artifacts/unquant_params.joblib")

    q_weights16, w16_min, w16_max, w16_scale = float_to_uint16(weights)
    q_bias16, b16_min, b16_max, b16_scale = float_to_uint16(np.array([bias]))

    print(f"\nQuantizing bias...")
    print(f"Bias value: {bias:.8f}")
    print(f"Bias scale factor: {b16_scale:.2f}")

    # Save quantized parameters with metadata
    params_quant = {
        'q_weights16': q_weights16,
        'w16_min': w16_min,
        'w16_max': w16_max,
        'w16_scale': w16_scale,
        'q_bias16': q_bias16[0],
        'b16_min': b16_min,
        'b16_max': b16_max,
        'b16_scale': b16_scale
    }
    joblib.dump(params_quant, "artifacts/quant_params.joblib")
    print("Quantized parameters saved to artifacts/quant_params.joblib")

    # Print model size difference
    orig_sz = os.path.getsize("artifacts/linear_regression_model.joblib")
    quant_sz = os.path.getsize("artifacts/quant_params.joblib")
    print(f"\nModel size before quantization: {orig_sz / 1024:.2f} KB")
    print(f"Model size after quantization:  {quant_sz / 1024:.2f} KB")
    print(f"Size reduction:                {(orig_sz - quant_sz) / 1024:.2f} KB")

    # Dequantize for inference
    dq_weights16 = uint16_to_float(q_weights16, w16_min, w16_max, w16_scale)
    dq_bias16 = uint16_to_float(np.array([params_quant['q_bias16']]), b16_min, b16_max, b16_scale)[0]

    # Calculate coefficient and intercept errors
    w_err = np.abs(weights - dq_weights16).max()
    b_err = np.abs(bias - dq_bias16)
    print(f"Max weight error (16-bit): {w_err:.8f}")
    print(f"Bias error (16-bit): {b_err:.8f}")

    # Inference test and output formatting
    from utils import fetch_data_split
    train_x, test_x, train_y, test_y = fetch_data_split()

    # First 5 samples
    pred_orig = reg.predict(test_x[:5])
    pred_manual_orig = test_x[:5] @ weights + bias
    pred_manual_dq = test_x[:5] @ dq_weights16 + dq_bias16

    print("\nInference Test (first 5 samples):")
    print(f"Original predictions (sklearn): {pred_orig}")
    print(f"Manual original predictions:    {pred_manual_orig}")
    print(f"Manual dequant predictions:     {pred_manual_dq}")

    print("\nDifferences:")
    print(f"Sklearn vs manual original: {np.abs(pred_orig - pred_manual_orig)}")
    print(f"Original vs dequant manual:  {np.abs(pred_manual_orig - pred_manual_dq)}")
    orig_vs_dq_diff = np.abs(pred_orig - pred_manual_dq)
    print(f"Absolute differences: {orig_vs_dq_diff}")
    print(f"Max difference: {orig_vs_dq_diff.max()}")
    print(f"Mean difference: {orig_vs_dq_diff.mean()}")
    max_diff = orig_vs_dq_diff.max()
    if max_diff < 0.1:
        print(f"Quantization quality is good (max diff: {max_diff:.6f})")
    elif max_diff < 1.0:
        print(f"Quantization quality is acceptable (max diff: {max_diff:.6f})")
    else:
        print(f"Quantization quality is poor (max diff: {max_diff:.6f})")

    # Calculate prediction errors for quantized (dequantized) model
    max_pred_err = np.max(np.abs(test_y - (test_x @ dq_weights16 + dq_bias16)))
    mean_pred_err = np.mean(np.abs(test_y - (test_x @ dq_weights16 + dq_bias16)))
    print(f"Max Prediction Error (quantized 16-bit): {max_pred_err:.4f}")
    print(f"Mean Prediction Error (quantized 16-bit): {mean_pred_err:.4f}")

    # 8-bit quantization
    q_weights8, w8_min, w8_max, w8_scale = float_to_uint8(weights)
    q_bias8, b8_min, b8_max, b8_scale = float_to_uint8(np.array([bias]))
    print(f"\nQuantizing bias (8-bit)...")
    print(f"Bias value: {bias:.8f}")
    print(f"Bias scale factor (8-bit): {b8_scale:.2f}")
    params_quant8 = {
        'q_weights8': q_weights8,
        'w8_min': w8_min,
        'w8_max': w8_max,
        'w8_scale': w8_scale,
        'q_bias8': q_bias8[0],
        'b8_min': b8_min,
        'b8_max': b8_max,
        'b8_scale': b8_scale
    }
    joblib.dump(params_quant8, "artifacts/quant_params8.joblib")
    print("Quantized parameters (8-bit) saved to artifacts/quant_params8.joblib")

    quant_sz8 = os.path.getsize("artifacts/quant_params8.joblib")
    print(f"Model size after 8-bit quantization:  {quant_sz8 / 1024:.2f} KB")
    print(f"Size reduction (8-bit):                {(orig_sz - quant_sz8) / 1024:.2f} KB")

    # Dequantize for inference (8-bit)
    dq_weights8 = uint8_to_float(q_weights8, w8_min, w8_max, w8_scale)
    dq_bias8 = uint8_to_float(np.array([params_quant8['q_bias8']]), b8_min, b8_max, b8_scale)[0]

    # Calculate coefficient and intercept errors (8-bit)
    w_err8 = np.abs(weights - dq_weights8).max()
    b_err8 = np.abs(bias - dq_bias8)
    print(f"Max weight error (8-bit): {w_err8:.8f}")
    print(f"Bias error (8-bit): {b_err8:.8f}")

    # Inference test and output formatting (8-bit)
    from utils import fetch_data_split
    train_x, test_x, train_y, test_y = fetch_data_split()
    pred_manual_dq8 = test_x[:5] @ dq_weights8 + dq_bias8
    print("\nManual dequant predictions (8-bit):     ", pred_manual_dq8)
    orig_vs_dq_diff8 = np.abs(reg.predict(test_x[:5]) - pred_manual_dq8)
    print(f"Absolute differences (8-bit): {orig_vs_dq_diff8}")
    print(f"Max difference (8-bit): {orig_vs_dq_diff8.max()}")
    print(f"Mean difference (8-bit): {orig_vs_dq_diff8.mean()}")
    max_diff8 = orig_vs_dq_diff8.max()
    if max_diff8 < 0.1:
        print(f"Quantization quality (8-bit) is good (max diff: {max_diff8:.6f})")
    elif max_diff8 < 1.0:
        print(f"Quantization quality (8-bit) is acceptable (max diff: {max_diff8:.6f})")
    else:
        print(f"Quantization quality (8-bit) is poor (max diff: {max_diff8:.6f})")

    # Calculate prediction errors for quantized (dequantized) model (8-bit)
    max_pred_err8 = np.max(np.abs(test_y - (test_x @ dq_weights8 + dq_bias8)))
    mean_pred_err8 = np.mean(np.abs(test_y - (test_x @ dq_weights8 + dq_bias8)))
    print(f"Max Prediction Error (quantized 8-bit): {max_pred_err8:.4f}")
    print(f"Mean Prediction Error (quantized 8-bit): {mean_pred_err8:.4f}")
    print("\nQuantization completed successfully!\n")

    # Calculate R2 and MSE for 8-bit quantized model
    y_pred8 = test_x @ dq_weights8 + dq_bias8
    from utils import regression_metrics
    r2_8, mse_8 = regression_metrics(test_y, y_pred8)
    print(f"R2 score (quantized 8-bit model): {r2_8:.4f}")
    print(f"MSE (quantized 8-bit model): {mse_8:.4f}")

    # Calculate R2 and MSE for 16-bit quantized model
    y_pred16 = test_x @ dq_weights16 + dq_bias16
    r2_16, mse_16 = regression_metrics(test_y, y_pred16)
    print(f"R2 score (quantized 16-bit model): {r2_16:.4f}")
    print(f"MSE (quantized 16-bit model): {mse_16:.4f}")


if __name__ == "__main__":
    quantize_main()
