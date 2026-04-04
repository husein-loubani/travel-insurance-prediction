"""
predict.py
----------
Model inference and evaluation utilities.

CCDS role: code to run model inference and evaluate trained models.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from travel_insurance.config import TARGET


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
) -> dict:
    """
    Comprehensive evaluation on a held-out test set.

    Returns a metrics dict and prints a formatted report.
    The test set should only be passed here once, after all tuning is complete.
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy":          round(accuracy_score(y_test, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_test, y_pred), 4),
        "precision":         round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":            round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":                round(f1_score(y_test, y_pred, zero_division=0), 4),
    }
    if y_proba is not None:
        metrics["roc_auc"] = round(roc_auc_score(y_test, y_proba), 4)
        metrics["pr_auc"]  = round(average_precision_score(y_test, y_proba), 4)

    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred)

    sep = "=" * 50
    print(f"\n{sep}")
    print(f"  {model_name}: Test Set Evaluation")
    print(sep)
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  Balanced accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"  Precision         : {metrics['precision']:.4f}")
    print(f"  Recall            : {metrics['recall']:.4f}")
    print(f"  F1-score          : {metrics['f1']:.4f}")
    if "roc_auc" in metrics:
        print(f"  ROC-AUC           : {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC            : {metrics['pr_auc']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"\n  Confusion matrix:")
    print(f"    TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"    FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    print(sep)
    print()
    print(classification_report(y_test, y_pred, target_names=["No Insurance", "Insurance"]))
    return metrics


def compare_final_metrics(metrics_dict: dict) -> pd.DataFrame:
    """
    Combine per-model evaluation dicts into a sortable comparison DataFrame.
    Excludes the confusion_matrix key.
    """
    rows = []
    for name, m in metrics_dict.items():
        row = {"model": name}
        row.update({k: v for k, v in m.items() if k != "confusion_matrix"})
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .set_index("model")
        .sort_values("roc_auc", ascending=False)
    )
