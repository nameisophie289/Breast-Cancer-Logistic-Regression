
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.model_selection import cross_val_score

from .logistic_regression import LogisticRegression
from .metrics import compute_classification_metrics
from .plotting import plot_learning_rate_comparison

class ExperimentRunner:
    """
    Class for running systematic experiments on logistic regression.

    This class provides methods for hyperparameter analysis, model comparison,
    and performance evaluation across different settings.

    Parameters
    ----------
    random_state : int, default=42
        Random seed for reproducibility.
    verbose : bool, default=True
        Whether to print experiment progress.
    """

    def __init__(self, random_state: int = 42, verbose: bool = True):
        self.random_state = random_state
        self.verbose = verbose
        self.results = {}
        np.random.seed(random_state)

    def experiment_learning_rates(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        learning_rates: List[float] = None,
    ) -> Dict[str, Any]:
        """
        Experiment with different learning rates.

        Parameters
        ----------
        X_train, y_train : array-like
            Training data.
        X_test, y_test : array-like
            Test data.
        learning_rates : list, optional
            List of learning rates to test.

        Returns
        -------
        results : dict
            Experiment results including metrics and training histories.
        """
        if learning_rates is None:
            learning_rates = [0.001, 0.01, 0.1, 1.0]

        if self.verbose:
            print("Experimenting with different learning rates...")
            print(f"Learning rates to test: {learning_rates}")

        results = {
            "learning_rates": learning_rates,
            "cost_histories": [],
            "final_costs": [],
            "training_times": [],
            "test_metrics": [],
            "convergence_info": [],
        }

        for i, lr in enumerate(learning_rates):
            if self.verbose:
                print(f"\\nTesting learning rate {lr} ({i+1}/{len(learning_rates)})")

            # Train model
            start_time = time.time()
            model = LogisticRegression(
                learning_rate=lr, max_iterations=1000, verbose=False
            )

            try:
                model.fit(X_train, y_train)
                training_time = time.time() - start_time

                # Get training history
                history = model.get_training_history()
                results["cost_histories"].append(history["cost_history"])
                results["final_costs"].append(history["final_cost"])
                results["training_times"].append(training_time)
                results["convergence_info"].append(
                    {
                        "converged": history["converged"],
                        "iterations": history["iterations"],
                    }
                )

                # Evaluate on test set
                y_pred = model.predict(X_test)
                # y_proba = model.predict_proba(X_test) # Unused
                test_metrics = compute_classification_metrics(y_test, y_pred)
                test_metrics["training_time"] = training_time
                results["test_metrics"].append(test_metrics)

                if self.verbose:
                    print(f"  Final cost: {history['final_cost']:.6f}")
                    print(f"  Iterations: {history['iterations']}")
                    print(f"  Test accuracy: {test_metrics['accuracy']:.4f}")
                    print(f"  Training time: {training_time:.3f}s")

            except Exception as e:
                if self.verbose:
                    print(f"  Failed with learning rate {lr}: {e}")
                # Add placeholder results for failed runs
                results["cost_histories"].append([])
                results["final_costs"].append(np.inf)
                results["training_times"].append(0)
                results["test_metrics"].append(
                    {"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0}
                )
                results["convergence_info"].append(
                    {"converged": False, "iterations": 0}
                )

        # Store results
        self.results["learning_rate_experiment"] = results

        if self.verbose:
            print("\\Learning Rate Experiment Summary:")
            for lr, metrics, conv_info in zip(
                learning_rates, results["test_metrics"], results["convergence_info"]
            ):
                status = (
                    "Converged" if conv_info["converged"] else "Max iterations"
                )
                print(f"  LR {lr:>6}: Accuracy={metrics['accuracy']:.4f}, {status}")

        return results

    def experiment_regularization(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        lambda_values: List[float] = None,
    ) -> Dict[str, Any]:
        """
        Experiment with different regularization strengths.

        Parameters
        ----------
        X_train, y_train : array-like
            Training data.
        X_test, y_test : array-like
            Test data.
        lambda_values : list, optional
            List of regularization strengths to test.

        Returns
        -------
        results : dict
            Experiment results including metrics and weight magnitudes.
        """
        if lambda_values is None:
            lambda_values = [0, 0.001, 0.01, 0.1, 1.0]

        if self.verbose:
            print("Experimenting with different regularization strengths...")
            print(f"Lambda values to test: {lambda_values}")

        results = {
            "lambda_values": lambda_values,
            "train_metrics": [],
            "test_metrics": [],
            "weight_magnitudes": [],
            "cost_histories": [],
        }

        for i, lambda_reg in enumerate(lambda_values):
            if self.verbose:
                print(f"\\nTesting lambda = {lambda_reg} ({i+1}/{len(lambda_values)})")

            # Determine regularization type
            reg_type = None if lambda_reg == 0 else "l2"

            # Train model
            model = LogisticRegression(
                learning_rate=0.01,
                max_iterations=1000,
                regularization=reg_type,
                lambda_reg=lambda_reg,
                verbose=False,
            )

            model.fit(X_train, y_train)

            # Get training history
            history = model.get_training_history()
            results["cost_histories"].append(history["cost_history"])

            # Evaluate on training set
            y_train_pred = model.predict(X_train)
            train_metrics = compute_classification_metrics(y_train, y_train_pred)
            results["train_metrics"].append(train_metrics)

            # Evaluate on test set
            y_test_pred = model.predict(X_test)
            test_metrics = compute_classification_metrics(y_test, y_test_pred)
            results["test_metrics"].append(test_metrics)

            # Calculate weight magnitude
            weight_magnitude = np.linalg.norm(model.weights_)
            results["weight_magnitudes"].append(weight_magnitude)

            if self.verbose:
                print(f"  Train accuracy: {train_metrics['accuracy']:.4f}")
                print(f"  Test accuracy: {test_metrics['accuracy']:.4f}")
                print(f"  Weight magnitude: {weight_magnitude:.4f}")

        # Store results
        self.results["regularization_experiment"] = results

        if self.verbose:
            print("\\nRegularization Experiment Summary:")
            for lam, train_acc, test_acc, weight_mag in zip(
                lambda_values,
                [m["accuracy"] for m in results["train_metrics"]],
                [m["accuracy"] for m in results["test_metrics"]],
                results["weight_magnitudes"],
            ):
                print(
                    f"  \u03bb {lam:>6}: Train={train_acc:.4f}, Test={test_acc:.4f}, ||w||={weight_mag:.4f}"
                )

        return results

    def compare_with_sklearn(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        tolerance: float = 1e-3,
    ) -> Dict[str, Any]:
        """
        Compare implementation with scikit-learn's LogisticRegression.

        Parameters
        ----------
        X_train, y_train : array-like
            Training data.
        X_test, y_test : array-like
            Test data.
        tolerance : float, default=1e-3
            Tolerance for numerical comparison.

        Returns
        -------
        results : dict
            Comparison results.
        """
        if self.verbose:
            print("\ud83d\udd2c Comparing with scikit-learn implementation...")

        # Train our implementation
        our_model = LogisticRegression(
            learning_rate=0.01, max_iterations=1000, verbose=False
        )
        our_model.fit(X_train, y_train)

        # Train scikit-learn model
        sklearn_model = SklearnLogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            solver="lbfgs",  # Use LBFGS for better convergence
        )
        sklearn_model.fit(X_train, y_train)

        # Compare predictions
        our_pred = our_model.predict(X_test)
        sklearn_pred = sklearn_model.predict(X_test)

        our_proba = our_model.predict_proba(X_test)
        sklearn_proba = sklearn_model.predict_proba(X_test)[
            :, 1
        ]  # Get positive class probabilities

        # Compare metrics
        our_metrics = compute_classification_metrics(y_test, our_pred)
        sklearn_metrics = compute_classification_metrics(y_test, sklearn_pred)

        # Compare parameters (note: sklearn may use different conventions)
        weight_diff = np.linalg.norm(our_model.weights_ - sklearn_model.coef_[0])
        bias_diff = abs(our_model.bias_ - sklearn_model.intercept_[0])

        # Prediction agreement
        prediction_agreement = np.mean(our_pred == sklearn_pred)
        proba_mae = np.mean(np.abs(our_proba - sklearn_proba))

        results = {
            "our_metrics": our_metrics,
            "sklearn_metrics": sklearn_metrics,
            "parameter_differences": {
                "weight_l2_norm_diff": weight_diff,
                "bias_diff": bias_diff,
            },
            "prediction_agreement": prediction_agreement,
            "probability_mae": proba_mae,
            "numerical_agreement": {
                "weights": weight_diff < tolerance,
                "bias": bias_diff < tolerance,
                "predictions": prediction_agreement > 0.95,
                "probabilities": proba_mae < tolerance,
            },
        }

        # Store results
        self.results["sklearn_comparison"] = results

        if self.verbose:
            print(f"\\nComparison with scikit-learn:")
            print(f"  Our accuracy:      {our_metrics['accuracy']:.4f}")
            print(f"  Sklearn accuracy:  {sklearn_metrics['accuracy']:.4f}")
            print(f"  Prediction agreement: {prediction_agreement:.4f}")
            print(f"  Probability MAE:   {proba_mae:.6f}")
            print(f"  Weight difference: {weight_diff:.6f}")
            print(f"  Bias difference:   {bias_diff:.6f}")

            # Check if implementations agree
            agrees = all(results["numerical_agreement"].values())
            status = (
                "Implementations agree!" if agrees else "Implementations differ"
            )
            print(f"  Status: {status}")

        return results

    def convergence_analysis(
        self, X_train: np.ndarray, y_train: np.ndarray, max_iterations: List[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze convergence behavior with different iteration limits.

        Parameters
        ----------
        X_train, y_train : array-like
            Training data.
        max_iterations : list, optional
            List of maximum iteration limits to test.

        Returns
        -------
        results : dict
            Convergence analysis results.
        """
        if max_iterations is None:
            max_iterations = [50, 100, 200, 500, 1000]

        if self.verbose:
            print("\ud83d\udd2c Analyzing convergence behavior...")

        results = {
            "max_iterations": max_iterations,
            "final_costs": [],
            "cost_histories": [],
            "convergence_rates": [],
        }

        for max_iter in max_iterations:
            model = LogisticRegression(
                learning_rate=0.01, max_iterations=max_iter, verbose=False
            )

            model.fit(X_train, y_train)
            history = model.get_training_history()

            results["final_costs"].append(history["final_cost"])
            results["cost_histories"].append(history["cost_history"])

            # Calculate convergence rate (cost reduction per iteration)
            if len(history["cost_history"]) > 1:
                initial_cost = history["cost_history"][0]
                final_cost = history["final_cost"]
                iterations = len(history["cost_history"])
                convergence_rate = (initial_cost - final_cost) / iterations
                results["convergence_rates"].append(convergence_rate)
            else:
                results["convergence_rates"].append(0)

        self.results["convergence_analysis"] = results

        if self.verbose:
            print("\\nConvergence Analysis:")
            for max_iter, final_cost, conv_rate in zip(
                max_iterations, results["final_costs"], results["convergence_rates"]
            ):
                print(
                    f"  {max_iter:>4} iter: Final cost={final_cost:.6f}, Rate={conv_rate:.8f}"
                )

        return results

    def cross_validation_analysis(
        self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Perform cross-validation analysis.

        Parameters
        ----------
        X, y : array-like
            Full dataset.
        cv_folds : int, default=5
            Number of cross-validation folds.

        Returns
        -------
        results : dict
            Cross-validation results.
        """
        if self.verbose:
            print(f"\ud83d\udd2c Performing {cv_folds}-fold cross-validation...")

        # Use sklearn's LogisticRegression for cross-validation
        # (Our implementation doesn't have built-in CV support)
        sklearn_model = SklearnLogisticRegression(
            max_iter=1000, random_state=self.random_state
        )

        cv_scores = cross_val_score(
            sklearn_model, X, y, cv=cv_folds, scoring="accuracy"
        )

        results = {
            "cv_scores": cv_scores,
            "mean_score": np.mean(cv_scores),
            "std_score": np.std(cv_scores),
            "cv_folds": cv_folds,
        }

        self.results["cross_validation"] = results

        if self.verbose:
            print(f"\\nCross-Validation Results:")
            print(
                f"  Mean accuracy: {results['mean_score']:.4f} \u00b1 {results['std_score']:.4f}"
            )
            print(f"  Individual scores: {[f'{score:.4f}' for score in cv_scores]}")

        return results

    def generate_experiment_report(self, save_path: str = None) -> str:
        """
        Generate a comprehensive experiment report.

        Parameters
        ----------
        save_path : str, optional
            Path to save the report. If None, returns the report as string.

        Returns
        -------
        report : str
            Formatted experiment report.
        """
        report = []
        report.append("# \ud83e\uddea Logistic Regression Experimental Analysis Report")
        report.append("=" * 60)
        report.append("")

        # Learning rate experiment
        if "learning_rate_experiment" in self.results:
            lr_results = self.results["learning_rate_experiment"]
            report.append("## Learning Rate Analysis")
            report.append("")
            report.append(
                "| Learning Rate | Final Cost | Test Accuracy | Converged | Training Time |"
            )
            report.append(
                "|---------------|------------|---------------|-----------|---------------|"
            )

            for lr, cost, metrics, conv_info, time_taken in zip(
                lr_results["learning_rates"],
                lr_results["final_costs"],
                lr_results["test_metrics"],
                lr_results["convergence_info"],
                lr_results["training_times"],
            ):
                converged = "Yes" if conv_info["converged"] else "No"
                report.append(
                    f"| {lr:>12} | {cost:>9.6f} | {metrics['accuracy']:>12.4f} | {converged:>8} | {time_taken:>12.3f}s |"
                )

            report.append("")

        # Regularization experiment
        if "regularization_experiment" in self.results:
            reg_results = self.results["regularization_experiment"]
            report.append("## Regularization Analysis")
            report.append("")
            report.append("| Lambda | Train Acc | Test Acc | Weight Norm |")
            report.append("|--------|-----------|----------|-------------|")

            for lam, train_m, test_m, weight_mag in zip(
                reg_results["lambda_values"],
                reg_results["train_metrics"],
                reg_results["test_metrics"],
                reg_results["weight_magnitudes"],
            ):
                report.append(
                    f"| {lam:>6} | {train_m['accuracy']:>8.4f} | {test_m['accuracy']:>7.4f} | {weight_mag:>10.4f} |"
                )

            report.append("")

        # Sklearn comparison
        if "sklearn_comparison" in self.results:
            comp_results = self.results["sklearn_comparison"]
            report.append("## \u2696\ufe0f Comparison with Scikit-learn")
            report.append("")
            report.append(
                f"- **Our Implementation Accuracy**: {comp_results['our_metrics']['accuracy']:.4f}"
            )
            report.append(
                f"- **Scikit-learn Accuracy**: {comp_results['sklearn_metrics']['accuracy']:.4f}"
            )
            report.append(
                f"- **Prediction Agreement**: {comp_results['prediction_agreement']:.4f}"
            )
            report.append(
                f"- **Probability MAE**: {comp_results['probability_mae']:.6f}"
            )
            report.append(
                f"- **Weight L2 Difference**: {comp_results['parameter_differences']['weight_l2_norm_diff']:.6f}"
            )
            report.append(
                f"- **Bias Difference**: {comp_results['parameter_differences']['bias_diff']:.6f}"
            )
            report.append("")

        # Cross-validation
        if "cross_validation" in self.results:
            cv_results = self.results["cross_validation"]
            report.append("## Cross-Validation Analysis")
            report.append("")
            report.append(
                f"- **Mean Accuracy**: {cv_results['mean_score']:.4f} \u00b1 {cv_results['std_score']:.4f}"
            )
            report.append(f"- **CV Folds**: {cv_results['cv_folds']}")
            report.append("")

        report_text = "\\n".join(report)

        if save_path:
            with open(save_path, "w") as f:
                f.write(report_text)
            if self.verbose:
                print(f"Experiment report saved to: {save_path}")

        return report_text

    def plot_experiment_results(self, save_dir: str = None) -> None:
        """
        Generate plots for all experiment results.

        Parameters
        ----------
        save_dir : str, optional
            Directory to save plots. If None, displays plots.
        """
        if self.verbose:
            print("Generating experiment plots...")

        # Learning rate comparison
        if "learning_rate_experiment" in self.results:
            lr_results = self.results["learning_rate_experiment"]
            plot_learning_rate_comparison(
                lr_results["learning_rates"],
                lr_results["cost_histories"],
                save_path=(
                    f"{save_dir}/learning_rate_comparison.png" if save_dir else None
                ),
            )

        # Regularization effect
        if "regularization_experiment" in self.results:
            reg_results = self.results["regularization_experiment"]

            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            # Plot accuracy vs regularization
            axes[0].plot(
                reg_results["lambda_values"],
                [m["accuracy"] for m in reg_results["train_metrics"]],
                "bo-",
                label="Training",
            )
            axes[0].plot(
                reg_results["lambda_values"],
                [m["accuracy"] for m in reg_results["test_metrics"]],
                "ro-",
                label="Test",
            )
            axes[0].set_xlabel("Regularization Strength (\u03bb)")
            axes[0].set_ylabel("Accuracy")
            axes[0].set_title("Regularization Effect on Accuracy")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].set_xscale("log")

            # Plot weight magnitude vs regularization
            axes[1].plot(
                reg_results["lambda_values"], reg_results["weight_magnitudes"], "go-"
            )
            axes[1].set_xlabel("Regularization Strength (\u03bb)")
            axes[1].set_ylabel("Weight L2 Norm")
            axes[1].set_title("Regularization Effect on Weight Magnitude")
            axes[1].grid(True, alpha=0.3)
            axes[1].set_xscale("log")

            plt.tight_layout()

            if save_dir:
                plt.savefig(
                    f"{save_dir}/regularization_analysis.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                if self.verbose:
                    print(
                        f"Regularization analysis plot saved to: {save_dir}/regularization_analysis.png"
                    )
            else:
                plt.show()
