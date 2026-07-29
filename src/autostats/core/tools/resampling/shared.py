"""Shared helpers for resampling tools (jackknife, bootstrap, ...)."""

import numpy as np

STAT_FUNCS = {
    "mean": np.mean,
    "median": np.median,
    "std": np.std,
    "var": np.var,
}
