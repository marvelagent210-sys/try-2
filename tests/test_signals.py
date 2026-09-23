import numpy as np
import pandas as pd

from growthpm.signals import compute_panel, zscore


def test_signals_never_use_future_prices(synthetic_prices, etf_cfg):
    full = compute_panel(synthetic_prices, etf_cfg.strategy, etf_cfg.regime, "SPY")
    cut = synthetic_prices.index[-120]
    past = compute_panel(synthetic_prices.loc[:cut], etf_cfg.strategy, etf_cfg.regime, "SPY")
    for name in ("momentum", "vol", "abs_momentum", "sma_entry", "sma_exit", "beta"):
        a = getattr(full, name).loc[:cut]
        b = getattr(past, name)
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, obj=name)


def test_changing_the_future_does_not_change_the_present(synthetic_prices, etf_cfg):
    cut = synthetic_prices.index[-50]
    shocked = synthetic_prices.copy()
    shocked.loc[shocked.index > cut] *= 3.0
    a = compute_panel(synthetic_prices, etf_cfg.strategy, etf_cfg.regime, "SPY").row(cut)
    b = compute_panel(shocked, etf_cfg.strategy, etf_cfg.regime, "SPY").row(cut)
    for key in a:
        pd.testing.assert_series_equal(a[key], b[key])


def test_zscore_is_winsorised_and_ignores_nans():
    s = pd.Series([1.0, 2.0, 3.0, np.nan, 1000.0] + [2.0] * 20)
    z = zscore(s)
    assert np.isnan(z.iloc[3])
    assert z.max() == 3.0
    assert zscore(pd.Series([1.0, 1.0])).eq(0).all()
