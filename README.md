# Breast Cancer Classification using Logistic Regression

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Project Overview

This project implements **Logistic Regression from scratch** to classify breast cancer tumors as malignant or benign. The goal is to demonstrate a deep understanding of machine learning algorithms by implementing the core optimization logic (Gradient Descent, Maximum Likelihood Estimation) without relying on high-level libraries for the model itself.

The project uses the **Breast Cancer Wisconsin (Diagnostic)** dataset.

## Features

- **Custom Logistic Regression Implementation**:
  - Gradient Descent optimization
  - L2 Regularization
  - Probability output using Sigmoid activation
- **Performance Evaluation**:
  - Confusion Matrix
  - ROC Curve & AUC Score
  - Accuracy, Precision, Recall, F1-Score
- **Visualization**:
  - Cost history during training to monitor convergence
  - Decision boundaries and metric plots

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd A2_SophieLam_24618528
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Using the Jupyter Notebook
Open `A2_SophieLam_24618528.ipynb` to see the step-by-step analysis, training process, and visualization of results.

```bash
jupyter notebook A2_SophieLam_24618528.ipynb
```

### 2. Using the Python Module
You can also use the `LogisticRegression` class directly in your own scripts:

```python
import numpy as np
from src.model import LogisticRegression

# Initialize model
model = LogisticRegression(learning_rate=0.01, max_iterations=1000)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
```

## Results

The custom implementation achieves high accuracy on the Breast Cancer dataset, comparable to Scikit-Learn's implementation.

- **Accuracy**: ~97%
- **Precision**: ~98%
- **Recall**: ~97%

## Author

**Sophie Lam**
*31005 Machine Learning Project*
