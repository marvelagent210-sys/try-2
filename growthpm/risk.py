"""Covariance estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .signals import TRADING_DAYS


def shrunk_covariance(returns: pd.DataFrame, shrink: float) -> np.ndarray:
    """Annualised covariance shrunk toward a constant-correlation target.

    Sample covariance over ~100 days of 20-50 assets is too noisy for an
    optimizer to invert; pulling correlations toward their average stops it
    from betting on spurious hedges.
    """
    r = returns.fillna(0.0).to_numpy()
    n = r.shape[1]
    if n == 0:
        return np.zeros((0, 0))
    sample = np.atleast_2d(np.cov(r, rowvar=False)) * TRADING_DAYS
    sd = np.sqrt(np.clip(np.diag(sample), 1e-10, None))
    if n > 1:
        corr = sample / np.outer(sd, sd)
        avg_corr = (corr.sum() - n) / (n * (n - 1))
        target = np.full((n, n), avg_corr) * np.outer(sd, sd)
        np.fill_diagonal(target, sd**2)
        sample = (1 - shrink) * sample + shrink * target
    return sample + np.eye(n) * 1e-6
