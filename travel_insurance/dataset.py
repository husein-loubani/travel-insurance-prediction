"""
dataset.py
----------
Functions for loading, auditing, and splitting the Travel Insurance dataset.

CCDS role: download / generate / load raw data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split

from travel_insurance.config import (
    ALPHA,
    CATEGORICAL_FEATURES,
    INDEX_COL,
    NUMERICAL_FEATURES,
    RANDOM_SEED,
    TARGET,
    TEST_SIZE,
)

# ── Loading ───────────────────────────────────────────────────────────────────

def load_data(path: str | Path) -> pd.DataFrame:
    """Load the raw CSV and drop the anonymous row-index column."""
    df = pd.read_csv(path)
    if INDEX_COL in df.columns:
        df = df.drop(columns=[INDEX_COL])
    return df


# ── Auditing ──────────────────────────────────────────────────────────────────

def audit_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Print a structured data-quality report and return a per-column summary DataFrame.

    Covers: shape, dtypes, missing values, duplicates, unique counts, and
    numeric range statistics.
    """
    n_rows, n_cols = df.shape
    print(f"Shape          : {n_rows:,} rows x {n_cols} columns")
    print(f"Duplicated rows: {df.duplicated().sum():,}")
    print()

    missing = df.isnull().sum()
    missing_pct = (missing / n_rows * 100).round(2)
    miss_df = pd.DataFrame({"missing_count": missing, "missing_%": missing_pct})
    miss_df = miss_df[miss_df["missing_count"] > 0]
    if miss_df.empty:
        print("Missing values : None detected.")
    else:
        print("Missing values detected:")
        print(miss_df.to_string())
    print()

    rows = []
    for col in df.columns:
        row = {
            "column":   col,
            "dtype":    str(df[col].dtype),
            "n_unique": df[col].nunique(),
            "missing":  df[col].isnull().sum(),
        }
        if df[col].dtype in [np.float64, np.int64, np.float32, np.int32]:
            row["min"]    = df[col].min()
            row["max"]    = df[col].max()
            row["mean"]   = round(df[col].mean(), 2)
            row["median"] = df[col].median()
        else:
            row["min"] = row["max"] = row["mean"] = row["median"] = "N/A"
        rows.append(row)

    return pd.DataFrame(rows).set_index("column")


def anomaly_screen(
    df: pd.DataFrame,
    features: list[str],
    iqr_factor: float = 1.5,
) -> pd.DataFrame:
    """
    Light IQR-based anomaly screen for numeric features (initial audit stage).

    Computes Tukey fences (Q1 - factor*IQR, Q3 + factor*IQR) and counts values
    outside those bounds. Features with IQR == 0 (e.g. binary indicators) are
    included in the summary but cannot be screened with this method.

    Results are flagged as potential anomalies only. Confirmation and any
    treatment decisions are deferred to the EDA section.
    """
    indices, rows = [], []
    for col in features:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            rows.append({
                "Q1":          round(q1, 2),
                "Q3":          round(q3, 2),
                "IQR":         0.0,
                "lower_fence": np.nan,
                "upper_fence": np.nan,
                "n_flagged":   np.nan,
                "pct_flagged": np.nan,
                "note":        "IQR=0 (binary/constant); screen not applicable",
            })
        else:
            lower = q1 - iqr_factor * iqr
            upper = q3 + iqr_factor * iqr
            n_flagged = int(((s < lower) | (s > upper)).sum())
            note = "potential anomalies" if n_flagged > 0 else "none flagged"
            rows.append({
                "Q1":          round(q1, 2),
                "Q3":          round(q3, 2),
                "IQR":         round(iqr, 2),
                "lower_fence": round(lower, 2),
                "upper_fence": round(upper, 2),
                "n_flagged":   n_flagged,
                "pct_flagged": round(n_flagged / len(s) * 100, 1),
                "note":        note,
            })
        indices.append(col)
    return pd.DataFrame(rows, index=pd.Index(indices, name="feature"))


def descriptive_stats(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return a transposed describe() for the given features, rounded to 2 dp."""
    return df[features].describe().T.round(2)


def target_distribution(df: pd.DataFrame, target: str = None) -> pd.DataFrame:
    """Return value counts and percentage for the target column."""
    col = target if target is not None else TARGET
    counts = df[col].value_counts()
    pct    = (counts / len(df) * 100).round(1)
    return pd.DataFrame({"count": counts, "%": pct})


def inspect_categoricals(df: pd.DataFrame, features: list = None) -> None:
    """Print distinct value counts for every categorical feature and the target."""
    cols = features if features is not None else CATEGORICAL_FEATURES + [TARGET]
    for col in cols:
        if col in df.columns:
            print(f"  {col:<25} -> {df[col].value_counts().to_dict()}")


# ── Data Cleaning ─────────────────────────────────────────────────────────────

def duplicate_summary(df: pd.DataFrame) -> None:
    """Print a concise duplicate-row summary for the DataFrame."""
    n_dups = df.duplicated().sum()
    print(f"Original  : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Unique    : {df.drop_duplicates().shape[0]:,} rows")
    print(f"Duplicates: {n_dups:,} rows ({n_dups / len(df) * 100:.1f}%)")


def duplicate_impact_numerical(
    df: pd.DataFrame,
    features: list[str] | None = None,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """
    Assess the impact of duplicate rows on numerical feature distributions.

    Compares unique records against the extra duplicate rows using a
    Mann-Whitney U test (non-parametric, no normality assumption).
    Returns a per-feature summary DataFrame with mean, median, percentage
    shift, p-value, and significance flag.
    """
    if features is None:
        features = NUMERICAL_FEATURES
    uniq_recs = df.drop_duplicates()
    dup_extra = df[df.duplicated(keep="first")]

    rows = []
    for col in features:
        mean_all   = round(df[col].mean(), 2)
        mean_dedup = round(uniq_recs[col].mean(), 2)
        med_all    = round(df[col].median(), 2)
        med_dedup  = round(uniq_recs[col].median(), 2)
        _, p_mw    = stats.mannwhitneyu(
            uniq_recs[col], dup_extra[col], alternative="two-sided"
        )
        rows.append({
            "feature":           col,
            "mean (all)":        mean_all,
            "mean (dedup)":      mean_dedup,
            "mean Δ%":           round((mean_dedup - mean_all) / mean_all * 100, 2),
            "median (all)":      med_all,
            "median (dedup)":    med_dedup,
            "median Δ%":         round((med_dedup - med_all) / med_all * 100, 2),
            "p (Mann-Whitney)":  round(p_mw, 4),
            f"sig (α={alpha})":  "Yes" if p_mw < alpha else "No",
        })
    return pd.DataFrame(rows).set_index("feature")


def duplicate_impact_categorical(
    df: pd.DataFrame,
    features: list[str] | None = None,
    alpha: float = ALPHA,
) -> pd.DataFrame:
    """
    Assess the impact of duplicate rows on categorical feature distributions.

    Compares category proportions between unique records and extra duplicate
    rows using a chi-square test. Returns a per-feature summary DataFrame
    with chi2, p-value, maximum proportion shift, and significance flag.
    """
    if features is None:
        features = CATEGORICAL_FEATURES
    uniq_recs = df.drop_duplicates()
    dup_flag  = df.duplicated(keep="first").map({True: "duplicate", False: "unique"})

    rows = []
    for col in features:
        ct = pd.crosstab(df[col], dup_flag)
        chi2, p, _, _ = chi2_contingency(ct)
        prop_all   = df[col].value_counts(normalize=True).to_dict()
        prop_dedup = uniq_recs[col].value_counts(normalize=True).to_dict()
        max_shift  = max(
            abs(prop_dedup.get(k, 0) - prop_all.get(k, 0)) for k in prop_all
        )
        rows.append({
            "feature":          col,
            "chi2":             round(chi2, 3),
            "p-value":          round(p, 4),
            "max prop. shift":  round(max_shift * 100, 2),
            f"sig (α={alpha})": "Yes" if p < alpha else "No",
        })
    return pd.DataFrame(rows).set_index("feature")


def drop_duplicates_clean(df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop duplicate rows, reset the index, and print a cleaning summary.

    Parameters
    ----------
    df     : DataFrame to clean.
    raw_df : Original (uncleaned) DataFrame; used to report rows removed.

    Returns the deduplicated DataFrame.
    """
    cleaned = df.drop_duplicates().reset_index(drop=True)
    print(f"Cleaned dataset : {cleaned.shape[0]:,} rows x {cleaned.shape[1]} columns")
    print(f"Rows removed    : {len(raw_df) - len(cleaned):,} duplicate rows")
    print(
        f"Target balance  : {cleaned[TARGET].mean() * 100:.1f}% positive "
        f"({cleaned[TARGET].sum()} of {len(cleaned)})"
    )
    return cleaned


# ── Splitting ─────────────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified 80/20 train / test split.

    Returns (train_df, test_df). The test set is treated as a held-out set
    and must not be touched until final evaluation.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df[TARGET],
    )
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)
    print(f"Train set : {len(train_df):,} rows  ({train_df[TARGET].mean()*100:.1f}% positive)")
    print(f"Test set  : {len(test_df):,} rows  ({test_df[TARGET].mean()*100:.1f}% positive)")
    return train_df, test_df
