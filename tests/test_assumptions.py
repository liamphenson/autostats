import numpy as np
import pandas as pd

from autostats.core.tools.stats.assumptions import (
    check_homogeneity_of_variance,
    check_normality,
    check_sample_size,
)


def test_check_normality_passes_for_normal_data(rng):
    series = pd.Series(rng.normal(size=200))
    result = check_normality(series, "x")
    assert result.passed is True
    assert result.p_value is not None and result.p_value > 0.05


def test_check_normality_fails_for_skewed_data(rng):
    series = pd.Series(rng.exponential(scale=1.0, size=200))
    result = check_normality(series, "x")
    assert result.passed is False


def test_check_homogeneity_of_variance_equal(rng):
    a = pd.Series(rng.normal(scale=1.0, size=100))
    b = pd.Series(rng.normal(scale=1.0, size=100))
    result = check_homogeneity_of_variance(a, b, labels=["a", "b"])
    assert result.passed is True


def test_check_homogeneity_of_variance_unequal(rng):
    a = pd.Series(rng.normal(scale=1.0, size=200))
    b = pd.Series(rng.normal(scale=10.0, size=200))
    result = check_homogeneity_of_variance(a, b, labels=["a", "b"])
    assert result.passed is False


def test_check_sample_size_flags_small_n():
    assert check_sample_size(5, 30, "x").passed is False
    assert check_sample_size(50, 30, "x").passed is True
