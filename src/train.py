import numpy as np

from utils import (
    fetch_data_split, build_regressor, persist_model,
    regression_metrics
)


def run_training():
    """Entry point for model training."""
    print("Fetching California Housing data...")
    train_x, test_x, train_y, test_y = fetch_data_split()

    print("Building LinearRegression estimator...")
    reg = build_regressor()

    print("Fitting regressor...")
    reg.fit(train_x, train_y)

    # Generate predictions
    preds = reg.predict(test_x)

    # Compute metrics
    r2_val, mse_val = regression_metrics(test_y, preds)
    max_err = np.max(np.abs(test_y - preds))
    mean_err = np.mean(np.abs(test_y - preds))

    print(f"R² Score: {r2_val:.4f}")
    print(f"Mean Squared Error (Loss): {mse_val:.4f}")
    print(f"Max Prediction Error: {max_err:.4f}")
    print(f"Mean Prediction Error: {mean_err:.4f}")

    # Save model
    out_model = "artifacts/linear_regression_model.joblib"
    persist_model(reg, out_model)
    print(f"Model saved to {out_model}")

    return reg, r2_val, mse_val


if __name__ == "__main__":
    run_training()
