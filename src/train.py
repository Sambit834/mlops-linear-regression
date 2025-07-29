from utils import (
    load_dataset, create_model, save_model,
    calculate_metrics
)


def main():
    """Main training function."""
    print("Loading California Housing dataset...")
    X_train, X_test, y_train, y_test = load_dataset()

    print("Creating LinearRegression model...")
    model = create_model()

    print("Training model...")
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    r2, mse = calculate_metrics(y_test, y_pred)

    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error (Loss): {mse:.4f}")

    # Save model
    model_path = "models/linear_regression_model.joblib"
    save_model(model, model_path)
    print(f"Model saved to {model_path}")

    return model, r2, mse


if __name__ == "__main__":
    main()
