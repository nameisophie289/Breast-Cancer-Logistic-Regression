
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
from sklearn.metrics import confusion_matrix
from .metrics import compute_roc_metrics

def plot_cost_history(
    cost_history: List[float],
    title: str = "Training Cost History",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot training cost history.

    Parameters
    ----------
    cost_history : list
        List of cost values during training.
    title : str, default="Training Cost History"
        Plot title.
    save_path : str, optional
        Path to save the plot. If None, displays the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, "b-", linewidth=2)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Cost (Binary Cross-Entropy)", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)

    # Add convergence information
    if len(cost_history) > 1:
        final_cost = cost_history[-1]
        initial_cost = cost_history[0]
        reduction = ((initial_cost - final_cost) / initial_cost) * 100
        plt.text(
            0.7,
            0.95,
            f"Cost reduction: {reduction:.1f}%",
            transform=plt.gca().transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Cost history plot saved to: {save_path}")
    else:
        plt.show()


def plot_learning_rate_comparison(
    learning_rates: List[float],
    cost_histories: List[List[float]],
    save_path: Optional[str] = None,
) -> None:
    """
    Compare training curves for different learning rates.

    Parameters
    ----------
    learning_rates : list
        List of learning rates used.
    cost_histories : list
        List of cost histories for each learning rate.
    save_path : str, optional
        Path to save the plot. If None, displays the plot.
    """
    plt.figure(figsize=(12, 8))

    colors = plt.cm.viridis(np.linspace(0, 1, len(learning_rates)))

    for lr, history, color in zip(learning_rates, cost_histories, colors):
        plt.plot(history, color=color, linewidth=2, label=f"LR = {lr}")

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Cost (Binary Cross-Entropy)", fontsize=12)
    plt.title("Learning Rate Comparison", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")  # Log scale for better visualization

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Learning rate comparison plot saved to: {save_path}")
    else:
        plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted binary labels.
    save_path : str, optional
        Path to save the plot. If None, displays the plot.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
    )
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix plot saved to: {save_path}")
    else:
        plt.show()


def plot_roc_curve(
    y_true: np.ndarray, y_proba: np.ndarray, save_path: Optional[str] = None
) -> None:
    """
    Plot ROC curve.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_proba : array-like
        Predicted probabilities for positive class.
    save_path : str, optional
        Path to save the plot. If None, displays the plot.
    """
    roc_metrics = compute_roc_metrics(y_true, y_proba)

    plt.figure(figsize=(8, 8))
    plt.plot(
        roc_metrics["fpr"],
        roc_metrics["tpr"],
        "b-",
        linewidth=2,
        label=f'ROC Curve (AUC = {roc_metrics["auc"]:.3f})',
    )
    plt.plot([0, 1], [0, 1], "r--", linewidth=1, label="Random Classifier")

    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(
        "Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"ROC curve plot saved to: {save_path}")
    else:
        plt.show()
