"""
train.py
--------
Model training and cross-validation utilities.

CCDS role: code to train models and run hyperparameter search.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from travel_insurance.config import RANDOM_SEED


def cv_compare(
    pipelines: dict,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
    scoring: str = "roc_auc",
) -> pd.DataFrame:
    """
    Run stratified k-fold CV on a dict of named pipelines.

    Returns a DataFrame with mean / std of the chosen scoring metric,
    mean train score, and mean fit time, sorted by mean score descending.

    Parameters
    ----------
    pipelines : dict
        {name: fitted_or_unfitted_pipeline}
    X, y : features and target from the *training* set only.
    cv_folds : number of stratified folds.
    scoring : sklearn scoring string (e.g. 'roc_auc', 'f1').
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    rows = []
    for name, pipe in pipelines.items():
        scores = cross_validate(
            pipe, X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
        )
        rows.append({
            "model":                    name,
            f"cv_{scoring}_mean":       round(scores["test_score"].mean(), 4),
            f"cv_{scoring}_std":        round(scores["test_score"].std(), 4),
            "train_score_mean":         round(scores["train_score"].mean(), 4),
            "fit_time_s":               round(scores["fit_time"].mean(), 2),
        })
    return (
        pd.DataFrame(rows)
        .set_index("model")
        .sort_values(f"cv_{scoring}_mean", ascending=False)
    )
