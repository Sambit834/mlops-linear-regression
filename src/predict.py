from utils import load_model, load_dataset, calculate_metrics


def main():
    """Main prediction function for Docker container."""
    print("Loading trained model...")
    model = load_model("models/linear_regression_model.joblib")

    print("Loading test dataset...")
    X_train, X_test, y_train, y_test = load_dataset()

    print("Making predictions...")
    y_pred = model.predict(X_test)

    # Calculate metrics
    r2, mse = calculate_metrics(y_test, y_pred)

    print(f"Model Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")

    print("\nSample Predictions (first 10):")
    for i in range(10):
        print(f"True: {y_test[i]:.2f} | Predicted: {y_pred[i]:.2f} | Diff: {abs(y_test[i] - y_pred[i]):.2f}")

    print("\nPrediction completed successfully!")
    return True


if __name__ == "__main__":
    main()