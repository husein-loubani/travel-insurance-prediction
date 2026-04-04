"""
statistics.py
-------------
Statistical inference helpers: hypothesis tests and confidence intervals.

CCDS note: this module has no direct CCDS equivalent; it is a custom addition
for the sprint's statistical inference requirement. Training and evaluation
logic lives in modeling/train.py and modeling/predict.py respectively.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from travel_insurance.config import ALPHA, CONFIDENCE_LEVEL


# ── Two-sample t-test ─────────────────────────────────────────────────────────

def two_sample_ttest(
    group0: pd.Series,
    group1: pd.Series,
    feature_name: str,
    h0: str,
    h1: str,
    alpha: float = ALPHA,
) -> dict:
    """
    Independent two-sample Welch's t-test with Levene's variance pre-test.

    Uses Welch's t-test (equal_var=False) as the default; does not assume
    homoscedasticity, which is the more conservative choice for observational data.
    """
    levene_stat, levene_p = stats.levene(group0, group1)
    equal_var = levene_p > alpha

    t_stat, p_value = stats.ttest_ind(group0, group1, equal_var=equal_var)

    n0, n1     = len(group0), len(group1)
    mean_diff  = group0.mean() - group1.mean()
    se_diff    = np.sqrt(group0.var(ddof=1) / n0 + group1.var(ddof=1) / n1)
    df_denom   = (
        (group0.var(ddof=1) / n0 + group1.var(ddof=1) / n1) ** 2
        / (
            (group0.var(ddof=1) / n0) ** 2 / (n0 - 1)
            + (group1.var(ddof=1) / n1) ** 2 / (n1 - 1)
        )
    )
    t_crit  = stats.t.ppf((1 + CONFIDENCE_LEVEL) / 2, df=df_denom)
    ci_low  = mean_diff - t_crit * se_diff
    ci_high = mean_diff + t_crit * se_diff

    return {
        "feature":      feature_name,
        "H0":           h0,
        "H1":           h1,
        "n_group0":     n0,
        "n_group1":     n1,
        "mean_group0":  round(group0.mean(), 4),
        "mean_group1":  round(group1.mean(), 4),
        "mean_diff":    round(mean_diff, 4),
        "levene_p":     round(levene_p, 4),
        "equal_var":    equal_var,
        "t_statistic":  round(t_stat, 4),
        "p_value":      round(p_value, 4),
        "ci_low":       round(ci_low, 4),
        "ci_high":      round(ci_high, 4),
        "alpha":        alpha,
        "reject_H0":    p_value < alpha,
    }


# ── Two-proportion z-test ─────────────────────────────────────────────────────

def two_proportion_ztest(
    successes_a: int,
    n_a: int,
    successes_b: int,
    n_b: int,
    group_a_name: str,
    group_b_name: str,
    feature_name: str,
    h0: str,
    h1: str,
    alpha: float = ALPHA,
) -> dict:
    """
    Two-proportion z-test using a pooled standard error under H0.

    Normal approximation holds when n*p >= 10 and n*(1-p) >= 10 for both groups.
    """
    p_a    = successes_a / n_a
    p_b    = successes_b / n_b
    p_pool = (successes_a + successes_b) / (n_a + n_b)

    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z_stat  = (p_a - p_b) / se_pool
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    z_crit      = stats.norm.ppf((1 + CONFIDENCE_LEVEL) / 2)
    se_unpooled = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    diff        = p_a - p_b
    ci_low      = diff - z_crit * se_unpooled
    ci_high     = diff + z_crit * se_unpooled

    return {
        "feature":     feature_name,
        "H0":          h0,
        "H1":          h1,
        "group_a":     group_a_name,
        "group_b":     group_b_name,
        "n_a":         n_a,
        "n_b":         n_b,
        "p_a":         round(p_a, 4),
        "p_b":         round(p_b, 4),
        "diff":        round(diff, 4),
        "z_statistic": round(z_stat, 4),
        "p_value":     round(p_value, 6),
        "ci_low":      round(ci_low, 4),
        "ci_high":     round(ci_high, 4),
        "alpha":       alpha,
        "reject_H0":   p_value < alpha,
    }


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_test_result(result: dict) -> None:
    """Pretty-print a hypothesis test result dictionary."""
    sep = "─" * 62
    print(sep)
    print(f"  Feature  : {result['feature']}")
    print(f"  H0       : {result['H0']}")
    print(f"  H1       : {result['H1']}")
    print(f"  alpha    : {result['alpha']}")
    print()
    if "t_statistic" in result:
        print(f"  Levene p-value : {result['levene_p']}  "
              f"(equal variance assumed: {result['equal_var']})")
        print(f"  Group 0 mean   : {result['mean_group0']:,.2f}  (n={result['n_group0']:,})")
        print(f"  Group 1 mean   : {result['mean_group1']:,.2f}  (n={result['n_group1']:,})")
        print(f"  Difference     : {result['mean_diff']:,.4f}")
        print(f"  t-statistic    : {result['t_statistic']:.4f}")
    elif "z_statistic" in result:
        print(f"  {result['group_a']} rate : {result['p_a']:.4f}  (n={result['n_a']:,})")
        print(f"  {result['group_b']} rate : {result['p_b']:.4f}  (n={result['n_b']:,})")
        print(f"  Difference        : {result['diff']:.4f}")
        print(f"  z-statistic       : {result['z_statistic']:.4f}")
    print(f"  p-value    : {result['p_value']:.6f}")
    print(f"  95% CI     : [{result['ci_low']:.4f}, {result['ci_high']:.4f}]")
    verdict = "REJECT H0" if result["reject_H0"] else "FAIL TO REJECT H0"
    print(f"\n  Verdict    : {verdict}")
    print(sep)
