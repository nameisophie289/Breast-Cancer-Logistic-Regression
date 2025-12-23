
import numpy as np
from typing import Any, Dict, Optional, Tuple, List

class LogisticRegression:
    """
    Logistic Regression classifier implemented from scratch using NumPy.

    This implementation uses maximum likelihood estimation and gradient descent
    to learn the parameters of a logistic regression model for binary classification.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for gradient descent optimization.
    max_iterations : int, default=1000
        Maximum number of iterations for gradient descent.
    regularization : str or None, default=None
        Type of regularization to apply. Options: None, 'l2'
    lambda_reg : float, default=0.01
        Regularization strength parameter.
    tolerance : float, default=1e-6
        Convergence tolerance for cost function.
    verbose : bool, default=False
        Whether to print training progress.

    Attributes
    ----------
    weights_ : ndarray of shape (n_features,)
        Learned weights after fitting.
    bias_ : float
        Learned bias term after fitting.
    cost_history_ : list
        History of cost function values during training.
    is_fitted_ : bool
        Whether the model has been fitted.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        regularization: Optional[str] = None,
        lambda_reg: float = 0.01,
        tolerance: float = 1e-6,
        verbose: bool = False,
    ):
        # Hyperparameters
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.tolerance = tolerance
        self.verbose = verbose

        # Model parameters (will be set during fitting)
        self.weights_ = None
        self.bias_ = None

        # Training history
        self.cost_history_ = []
        self.is_fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """
        Fit the logistic regression model to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data features.
        y : array-like of shape (n_samples,)
            Training data labels (0 or 1).

        Returns
        -------
        self : LogisticRegression
            Returns self for method chaining.
        """
        # Initialize weights with small random values
        self.weights_ = np.random.normal(0, 0.01, X.shape[1])
        self.bias_ = 0.0

        # Reset training history
        self.cost_history_ = []

        # Gradient descent optimization
        for iteration in range(self.max_iterations):
            # Forward pass
            z = self._compute_linear_combination(X)
            y_pred = self._sigmoid(z)

            # Compute cost
            cost = self._compute_cost(y, y_pred)
            self.cost_history_.append(cost)

            # Compute gradients
            dw, db = self._compute_gradients(X, y, y_pred)

            # Update parameters
            self.weights_ -= self.learning_rate * dw
            self.bias_ -= self.learning_rate * db

            # Check for convergence
            if (
                iteration > 0
                and abs(self.cost_history_[-2] - self.cost_history_[-1])
                < self.tolerance
            ):
                if self.verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break

            # Print progress
            if self.verbose and (iteration + 1) % 100 == 0:
                print(
                    f"Iteration {iteration + 1}/{self.max_iterations}, Cost: {cost:.6f}"
                )

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        probabilities : ndarray of shape (n_samples,)
            Predicted probabilities for the positive class.
        """
        self._check_is_fitted()

        z = self._compute_linear_combination(X)
        return self._sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary class labels for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        threshold : float, default=0.5
            Decision threshold for classification.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Predicted binary class labels (0 or 1).
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def get_training_history(self) -> Dict[str, Any]:
        """
        Get training history information.

        Returns
        -------
        history : dict
            Dictionary containing training history information.
        """
        self._check_is_fitted()
        return {
            "cost_history": self.cost_history_.copy(),
            "final_cost": self.cost_history_[-1],
            "iterations": len(self.cost_history_),
            "converged": len(self.cost_history_) < self.max_iterations,
        }

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Compute sigmoid activation function with numerical stability.

        Parameters
        ----------
        z : array-like
            Linear combination values.

        Returns
        -------
        sigmoid_values : ndarray
            Sigmoid activation values.
        """
        # Clip z to prevent overflow/underflow
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))

    def _compute_linear_combination(self, X: np.ndarray) -> np.ndarray:
        """Compute linear combination w^T * X + b."""
        return np.dot(X, self.weights_) + self.bias_

    def _compute_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute binary cross-entropy cost with optional regularization.

        Parameters
        ----------
        y_true : array-like
            True binary labels.
        y_pred : array-like
            Predicted probabilities.

        Returns
        -------
        cost : float
            Computed cost value.
        """
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)

        # Binary cross-entropy
        n_samples = len(y_true)
        cost = -np.mean(
            y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
        )

        # Add regularization if specified
        if self.regularization == "l2":
            l2_penalty = self.lambda_reg * np.sum(self.weights_**2) / (2 * n_samples)
            cost += l2_penalty

        return cost

    def _compute_gradients(
        self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Compute gradients for weights and bias.

        Parameters
        ----------
        X : array-like
            Input features.
        y_true : array-like
            True binary labels.
        y_pred : array-like
            Predicted probabilities.

        Returns
        -------
        dw : ndarray
            Gradient with respect to weights.
        db : float
            Gradient with respect to bias.
        """
        n_samples = len(y_true)

        # Compute gradients
        error = y_pred - y_true
        dw = np.dot(X.T, error) / n_samples
        db = np.mean(error)

        # Add regularization gradient if specified
        if self.regularization == "l2":
            dw += self.lambda_reg * self.weights_ / n_samples

        return dw, db

    def _check_is_fitted(self) -> None:
        """Check if the model has been fitted."""
        if not self.is_fitted_:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"LogisticRegression(learning_rate={self.learning_rate}, "
            f"max_iterations={self.max_iterations}, "
            f"regularization={self.regularization}, "
            f"lambda_reg={self.lambda_reg})"
        )
