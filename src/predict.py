from utils import restore_model, fetch_data_split, regression_metrics


def run_prediction():
    """Prediction entry point for Docker container."""
    print("Restoring trained regressor...")
    reg = restore_model("artifacts/linear_regression_model.joblib")

    print("Loading test split...")
    train_x, test_x, train_y, test_y = fetch_data_split()

    print("Generating predictions...")
    preds = reg.predict(test_x)

    # Compute metrics
    r2_val, mse_val = regression_metrics(test_y, preds)

    print(f"Model Performance:")
    print(f"R² Score: {r2_val:.4f}")
    print(f"Mean Squared Error: {mse_val:.4f}")

    print("\nSample Predictions (first 10):")
    for idx in range(10):
        print(f"True: {test_y[idx]:.2f} | Predicted: {preds[idx]:.2f} | Diff: {abs(test_y[idx] - preds[idx]):.2f}")

    print("\nPrediction completed successfully!")
    return True


if __name__ == "__main__":
    run_prediction()
