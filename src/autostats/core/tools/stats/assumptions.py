"""Shared assumption-check primitives, reused by every parametric, regression,
and time-series tool. A test tool always runs these BEFORE computing its main
result and returns both -- it never silently substitutes a different test;
that decision is left to the LLM/user via `recommended_alternative`.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from autostats.core.schemas.stat_result import AssumptionCheck


def check_normality(series: pd.Series, label: str) -> AssumptionCheck:
    series = series.dropna()
    n = len(series)
    if n < 3:
        return AssumptionCheck(
            name=f"normality_{label}",
            passed=False,
            detail=f"Sample size ({n}) too small to test normality for '{label}'.",
        )
    if n < 5000:
        stat, p = scipy_stats.shapiro(series)
        method = "Shapiro-Wilk"
    else:
        stat, p = scipy_stats.normaltest(series)
        method = "D'Agostino-Pearson"
    passed = p > 0.05
    return AssumptionCheck(
        name=f"normality_{label}",
        passed=passed,
        statistic=float(stat),
        p_value=float(p),
        detail=(
            f"{method} test for '{label}': "
            f"{'consistent with' if passed else 'inconsistent with'} normality (p={p:.4f})."
        ),
    )


def check_homogeneity_of_variance(*groups: pd.Series, labels: list[str] | None = None) -> AssumptionCheck:
    clean = [g.dropna() for g in groups]
    stat, p = scipy_stats.levene(*clean, center="median")
    passed = p > 0.05
    names = ", ".join(labels) if labels else "groups"
    return AssumptionCheck(
        name="homogeneity_of_variance",
        passed=passed,
        statistic=float(stat),
        p_value=float(p),
        detail=(
            f"Levene's test (Brown-Forsythe) across {names}: "
            f"variances {'appear equal' if passed else 'appear unequal'} (p={p:.4f})."
        ),
    )


def check_sample_size(n: int, min_n: int, label: str) -> AssumptionCheck:
    passed = n >= min_n
    return AssumptionCheck(
        name=f"sample_size_{label}",
        passed=passed,
        statistic=float(n),
        detail=(
            f"Sample size for '{label}' is {n} (recommended minimum {min_n})."
            + ("" if passed else " Results may be unreliable due to small sample size.")
        ),
    )


def check_independence_note() -> AssumptionCheck:
    return AssumptionCheck(
        name="independence",
        passed=True,
        detail=(
            "Independence of observations cannot be verified statistically from the data alone; "
            "it is assumed based on the study/collection design. Verify this holds for your data."
        ),
    )


def check_vif(design_matrix: pd.DataFrame) -> list[AssumptionCheck]:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    checks: list[AssumptionCheck] = []
    numeric = design_matrix.select_dtypes(include=[np.number])
    for i, col in enumerate(numeric.columns):
        try:
            vif = variance_inflation_factor(numeric.values, i)
        except Exception:
            continue
        passed = vif <= 5
        detail = f"VIF for '{col}' is {vif:.2f}."
        if vif > 10:
            detail += " Strong multicollinearity."
        elif vif > 5:
            detail += " Moderate multicollinearity."
        checks.append(
            AssumptionCheck(
                name=f"multicollinearity_{col}",
                passed=passed,
                statistic=float(vif),
                detail=detail,
            )
        )
    return checks
