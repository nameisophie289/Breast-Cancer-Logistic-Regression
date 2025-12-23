
import unittest
import numpy as np
from src import (
    LogisticRegression, 
    ExperimentRunner, 
    compute_classification_metrics
)

class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        # Create synthetic data
        np.random.seed(42)
        self.X = np.random.randn(100, 2)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(int)
        
    def test_fit_predict(self):
        model = LogisticRegression(learning_rate=0.1, max_iterations=100)
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        self.assertEqual(len(preds), 100)
        self.assertTrue(set(preds).issubset({0, 1}))
        
    def test_metrics(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 0])
        metrics = compute_classification_metrics(y_true, y_pred)
        self.assertEqual(metrics['accuracy'], 0.75)

class TestExperimentRunner(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X_train = np.random.randn(50, 2)
        self.y_train = (self.X_train[:, 0] + self.X_train[:, 1] > 0).astype(int)
        self.X_test = np.random.randn(20, 2)
        self.y_test = (self.X_test[:, 0] + self.X_test[:, 1] > 0).astype(int)
        self.runner = ExperimentRunner(verbose=False)

    def test_experiment_learning_rates(self):
        results = self.runner.experiment_learning_rates(
            self.X_train, self.y_train, self.X_test, self.y_test,
            learning_rates=[0.01, 0.1]
        )
        self.assertIn("learning_rates", results)
        self.assertEqual(len(results["learning_rates"]), 2)
        self.assertEqual(len(results["test_metrics"]), 2)

    def test_experiment_regularization(self):
        results = self.runner.experiment_regularization(
            self.X_train, self.y_train, self.X_test, self.y_test,
            lambda_values=[0, 0.1]
        )
        self.assertIn("lambda_values", results)
        self.assertEqual(len(results["lambda_values"]), 2)

    def test_compare_with_sklearn(self):
        # Just ensure it runs without error
        results = self.runner.compare_with_sklearn(
            self.X_train, self.y_train, self.X_test, self.y_test
        )
        self.assertIn("our_metrics", results)
        self.assertIn("sklearn_metrics", results)

if __name__ == '__main__':
    unittest.main()
