
import numpy as np
from typing import Dict
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted binary labels.

    Returns
    -------
    metrics : dict
        Dictionary containing various classification metrics.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def compute_roc_metrics(
    y_true: np.ndarray, y_proba: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute ROC curve and AUC score.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_proba : array-like
        Predicted probabilities for positive class.

    Returns
    -------
    roc_metrics : dict
        Dictionary containing FPR, TPR, thresholds, and AUC.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc_score = auc(fpr, tpr)

    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": auc_score}
