"""
features.py
-----------
Preprocessing pipeline construction for the Travel Insurance dataset.

CCDS role: create features for modelling.

All transformations are encapsulated in a leakage-safe ColumnTransformer.
The transformer is always fitted only on training data and applied to
validation / test data; never the reverse.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


# ── Feature groups ────────────────────────────────────────────────────────────

# Continuous numerics: need StandardScaler for distance-based models (KNN, SVM, LR).
NUMERICAL_SCALE: list[str] = ["age", "annual_income", "family_members"]

# Binary categorical encoded as int {0, 1}: already numeric, pass through unchanged.
BINARY_INT_PASS: list[str] = ["chronic_diseases"]

# Yes/No string columns: OrdinalEncoder maps No -> 0, Yes -> 1.
# OneHotEncoder would add a redundant column for two-level binary features.
BINARY_ORDINAL: list[str] = ["graduate_or_not", "frequent_flyer", "ever_travelled_abroad"]

# Two-level nominal: OneHotEncoder with drop='first' avoids the dummy variable trap
# in logistic regression.
NOMINAL_OHE: list[str] = ["employment_type"]


# ── Preprocessor factory ──────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    Build and return a ColumnTransformer that handles all feature groups.

    Returns a fresh (unfitted) transformer. Fit it on training data only:

        pre = build_preprocessor()
        pre.fit(X_train)
        X_test_transformed = pre.transform(X_test)

    Or use make_pipeline() which wraps this inside a Pipeline for you.
    """
    return ColumnTransformer(
        transformers=[
            (
                "scale",
                StandardScaler(),
                NUMERICAL_SCALE,
            ),
            (
                "pass",
                "passthrough",
                BINARY_INT_PASS,
            ),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[["No", "Yes"]] * len(BINARY_ORDINAL),
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                BINARY_ORDINAL,
            ),
            (
                "ohe",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
                NOMINAL_OHE,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_pipeline(clf) -> Pipeline:
    """
    Wrap a fresh preprocessor and a classifier into a single sklearn Pipeline.

    A new preprocessor is built per call so pipelines are fully independent
    and can be fitted in separate GridSearchCV folds without sharing state.

    Usage:
        pipe = make_pipeline(LogisticRegression(class_weight='balanced'))
        pipe.fit(X_train, y_train)
        pipe.predict(X_test)
    """
    return Pipeline([
        ("pre", build_preprocessor()),
        ("clf", clf),
    ])


# ── Engineered-feature groups ─────────────────────────────────────────────────

ENG_NUM_SCALE: list[str] = ["age", "annual_income", "family_members", "income_per_member"]
ENG_NUM_PASS: list[str] = ["chronic_diseases", "travel_risk"]
ENG_BIN_ORDINAL: list[str] = ["graduate_or_not", "frequent_flyer", "ever_travelled_abroad"]
ENG_NOMINAL_OHE: list[str] = ["employment_type", "age_group"]
ENG_FEATURES: list[str] = ENG_NUM_SCALE + ENG_NUM_PASS + ENG_BIN_ORDINAL + ENG_NOMINAL_OHE


def add_engineered_features(df) -> "pd.DataFrame":
    """Row-level feature engineering. No fitting required, so leakage-safe."""
    import pandas as pd
    d = df.copy()
    d["income_per_member"] = d["annual_income"] / d["family_members"]
    d["travel_risk"] = (
        (d["frequent_flyer"] == "Yes") | (d["ever_travelled_abroad"] == "Yes")
    ).astype(int)
    d["age_group"] = pd.cut(
        d["age"], bins=[0, 28, 32, 100], labels=["Young", "Mid", "Senior"]
    )
    return d


def build_eng_preprocessor() -> ColumnTransformer:
    """ColumnTransformer for the expanded engineered feature set."""
    return ColumnTransformer(
        transformers=[
            ("scale",   StandardScaler(), ENG_NUM_SCALE),
            ("pass",    "passthrough",    ENG_NUM_PASS),
            ("ordinal", OrdinalEncoder(
                categories=[["No", "Yes"]] * len(ENG_BIN_ORDINAL),
                handle_unknown="use_encoded_value", unknown_value=-1,
            ), ENG_BIN_ORDINAL),
            ("ohe", OneHotEncoder(
                drop="first", sparse_output=False, handle_unknown="ignore",
            ), ENG_NOMINAL_OHE),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_eng_pipeline(clf) -> Pipeline:
    """Pipeline using the engineered feature preprocessor."""
    return Pipeline([("pre", build_eng_preprocessor()), ("clf", clf)])
