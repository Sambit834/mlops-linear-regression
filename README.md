# MLOps Linear Regression Project

A complete MLOps pipeline for training, quantizing, and deploying a linear regression model using the California Housing
dataset. This project demonstrates best practices for reproducible machine learning workflows, including CI/CD,
Dockerization, and model quantization.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
    - [Training](#training)
    - [Quantization](#quantization)
    - [Prediction](#prediction)
- [Testing](#testing)
- [Docker Usage](#docker-usage)
- [CI/CD Pipeline](#cicd-pipeline)
- [Artifacts](#artifacts)
- [License](#license)

---

## Project Overview

This repository provides a simple, reproducible MLOps pipeline for a linear regression model using scikit-learn. The
pipeline includes:

- Data loading and preprocessing
- Model training and evaluation
- Model quantization (16-bit and 8-bit)
- Automated testing
- Docker-based deployment
- CI/CD with GitHub Actions

## Project Structure

```
mlops-linear-regression/
├── Dockerfile                # Docker build instructions
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── src/                      # Source code
│   ├── utils.py              # Utility functions (data, quantization, metrics)
│   ├── train.py              # Model training script
│   ├── predict.py            # Model inference script
│   ├── quantize.py           # Model quantization script
│   └── artifacts/            # Model artifacts (created at runtime)
├── models/                   # (Optional) Pretrained model files
├── tests/                    # Unit tests
│   └── test_train.py         # Test suite for training pipeline
└── .github/
    └── workflows/
        └── ci.yml            # GitHub Actions CI/CD workflow
```

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd mlops-linear-regression
   ```
2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

Train a linear regression model on the California Housing dataset:

```bash
python src/train.py
```

**Output:**

```
Loading California Housing dataset...
Creating LinearRegression model...
Training model...
R² Score: 0.5758
Mean Squared Error (Loss): 0.5559
Max Prediction Error: 9.8753
Mean Prediction Error: 0.5332
Model saved to models/linear_regression_model.joblib
```

- Model and metrics will be saved to `src/artifacts/`.

### Quantization

Quantize the trained model to 16-bit and 8-bit representations:

```bash
python src/quantize.py
```

**Output:**

```
Loading trained model...
Original coefficients shape: (8,)
Original intercept: -37.02327770606389
Original coef values: [ 4.48674910e-01  9.72425752e-03 -1.23323343e-01  7.83144907e-01
 -2.02962058e-06 -3.52631849e-03 -4.19792487e-01 -4.33708065e-01]

Quantizing intercept...
Intercept value: -37.02327771
Intercept scale factor: 1769.16
Quantized parameters saved to models/quant_params.joblib

Model size before quantization: 0.65 KB
Model size after quantization:  0.55 KB
Size reduction:                0.10 KB
Max coefficient error (16-bit): 0.00001722
Intercept error (16-bit): 0.00000000

Inference Test (first 5 samples):
Original predictions (sklearn): [0.71912284 1.76401657 2.70965883 2.83892593 2.60465725]
Manual original predictions:    [0.71912284 1.76401657 2.70965883 2.83892593 2.60465725]
Manual dequant predictions:     [0.69951698 1.74201847 2.69089815 2.81510208 2.5894276 ]

Differences:
Sklearn vs manual original: [0. 0. 0. 0. 0.]
Original vs dequant manual:  [0.01960586 0.0219981  0.01876068 0.02382385 0.01522965]
Absolute differences: [0.01960586 0.0219981  0.01876068 0.02382385 0.01522965]
Max difference: 0.02382384825320827
Mean difference: 0.01988362812321611
Quantization quality is good (max diff: 0.023824)
Max Prediction Error (quantized 16-bit): 9.8738
Mean Prediction Error (quantized 16-bit): 0.5305

Quantizing intercept (8-bit)...
Intercept value: -37.02327771
Intercept scale factor (8-bit): 6.75
Quantized parameters (8-bit) saved to models/quant_params8.joblib
Model size after 8-bit quantization:  0.53 KB
Size reduction (8-bit):                0.12 KB
Max coefficient error (8-bit): 0.00441088
Intercept error (8-bit): 0.00000000

Manual dequant predictions (8-bit):      [-5.4457549  -5.1534455  -3.24117255 -4.62411163 -2.21934695]
Absolute differences (8-bit): [6.16487775 6.91746207 5.95083138 7.46303756 4.8240042 ]
Max difference (8-bit): 7.463037557632468
Mean difference (8-bit): 6.26404259166095
Quantization quality (8-bit) is poor (max diff: 7.463038)
Max Prediction Error (quantized 8-bit): 68.9402
Mean Prediction Error (quantized 8-bit): 6.3301

Quantization completed successfully!

R2 score (quantized 8-bit model): -46.6831
MSE (quantized 8-bit model): 62.4844
R2 score (quantized 16-bit model): 0.5752
MSE (quantized 16-bit model): 0.5567
```

- Quantized parameters are saved to `src/artifacts/`.

### Prediction

Run inference using the trained model:

python src/predict.py

***Output:***

````

Restoring trained regressor...
Loading test split...
Generating predictions...
Model Performance:
R² Score: 0.5758
Mean Squared Error: 0.5559

Sample Predictions (first 10):
True: 0.48 | Predicted: 0.72 | Diff: 0.24
True: 0.46 | Predicted: 1.76 | Diff: 1.31
True: 5.00 | Predicted: 2.71 | Diff: 2.29
True: 2.19 | Predicted: 2.84 | Diff: 0.65
True: 2.78 | Predicted: 2.60 | Diff: 0.18
True: 1.59 | Predicted: 2.01 | Diff: 0.42
True: 1.98 | Predicted: 2.65 | Diff: 0.66
True: 1.57 | Predicted: 2.17 | Diff: 0.59
True: 3.40 | Predicted: 2.74 | Diff: 0.66
True: 4.47 | Predicted: 3.92 | Diff: 0.55

Prediction completed successfully!

````


- Outputs model performance and sample predictions.

## Model Performance

### Performance Comparison Table

| Metric         | Original Model | Quantized Model (16-bit) | Quantized Model (8-bit) | 16-bit vs Original | 8-bit vs Original |
|----------------|----------------|--------------------------|-------------------------|--------------------|-------------------|
| **R² Score**   | 0.5758         | 0.5752                   | -46.6831                | -0.0006            | -47.2589          |
| **MSE**        | 0.5559         | 0.5567                   | 62.4844                 | +0.0008            | +61.9285          |
| **Model Size** | 0.65 KB        | 0.55 KB                  | 0.53 KB                 | -0.10 KB           | -0.12 KB          |

## Testing

Run the test suite using pytest:

```bash
pytest tests/ -v
````

- Tests ensure the training pipeline works correctly, including:
    - Data loading
    - Model training
    - Model saving/loading
    - Quantization
    - Prediction accuracy

  ### Expected Output:
  ============================= test session starts =============================
  collecting ... collected 5 items

test_train.py::TestTraining::test_dataset_loading PASSED                 [ 20%]
test_train.py::TestTraining::test_model_creation PASSED                  [ 40%]
test_train.py::TestTraining::test_model_training PASSED                  [ 60%]
test_train.py::TestTraining::test_model_performance PASSED               [ 80%]Model R² Score: 0.5758
Model MSE: 0.5559

test_train.py::TestTraining::test_model_save_load PASSED                 [100%]

============================== 5 passed in 3.06s ==============================

````
    
    

- Tests cover data loading, model training, saving/loading, and performance.

## Docker Usage

Build and run the project in a Docker container:

1. **Build the Docker image:**
   ```bash
   docker build -t mlops-linear-regression .
   ```
2. **Run the container:**
   ```bash
   docker run --rm mlops-linear-regression
 
## CI/CD Pipeline

- Automated with GitHub Actions (`.github/workflows/ci.yml`):
    - Runs tests on every push/PR
    - Trains and quantizes the model
    - Uploads model artifacts
    - Builds and validates the Docker image

## Artifacts

- All model files and quantized parameters are saved in `src/artifacts/` (or `artifacts/` in the Docker image).
- Key files:
    - `linear_regression_model.joblib`: Trained model
    - `quant_params.joblib`, `quant_params8.joblib`: Quantized model parameters
    - `unquant_params.joblib`: Raw model parameters


