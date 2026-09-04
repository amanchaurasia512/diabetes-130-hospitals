import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def evaluate_model(
    model_name,
    y_true,
    y_prob,
    threshold=0.50
):
    """
    Evaluate binary classification performance at a given threshold.

    Parameters
    ----------
    model_name : str
        Name of the model.
    y_true : array-like
        Actual binary target values.
    y_prob : array-like
        Predicted probabilities for the positive class.
    threshold : float, default=0.50
        Probability threshold used to convert probabilities into predictions.

    Returns
    -------
    dict
        Classification and ranking metrics.
    """

    y_pred = (np.asarray(y_prob) >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "Model": model_name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "F1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "Average_Precision": average_precision_score(
            y_true,
            y_prob
        ),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }


def evaluate_thresholds(
    y_true,
    y_prob,
    thresholds=None,
    model_name="Model"
):
    """
    Evaluate a model across multiple probability thresholds.

    Parameters
    ----------
    y_true : array-like
        Actual binary target values.
    y_prob : array-like
        Predicted probabilities for the positive class.
    thresholds : array-like, optional
        Probability thresholds to evaluate.
    model_name : str, default="Model"
        Name of the model.

    Returns
    -------
    pandas.DataFrame
        Metrics for each threshold.
    """

    if thresholds is None:
        thresholds = np.arange(0.05, 0.51, 0.01)

    results = []

    for threshold in thresholds:
        results.append(
            evaluate_model(
                model_name=model_name,
                y_true=y_true,
                y_prob=y_prob,
                threshold=threshold
            )
        )

    return pd.DataFrame(results)


def find_best_threshold(
    y_true,
    y_prob,
    thresholds=None,
    metric="F1"
):
    """
    Find the threshold that maximizes a selected evaluation metric.

    Parameters
    ----------
    y_true : array-like
        Actual binary target values.
    y_prob : array-like
        Predicted probabilities.
    thresholds : array-like, optional
        Thresholds to evaluate.
    metric : str, default="F1"
        Metric used for selecting the best threshold.

    Returns
    -------
    dict
        Best threshold and corresponding evaluation metrics.
    """

    results = evaluate_thresholds(
        y_true=y_true,
        y_prob=y_prob,
        thresholds=thresholds
    )

    best_idx = results[metric].idxmax()

    return results.loc[best_idx].to_dict()
    