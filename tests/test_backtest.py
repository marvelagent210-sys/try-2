import numpy as np
import pandas as pd

from growthpm.backtest import performance, run_backtest
from growthpm.universe import resolve


def test_backtest_runs_and_respects_limits(synthetic_prices, etf_cfg):
    uni = resolve(etf_cfg.universe)
    r = run_backtest(synthetic_prices, etf_cfg, uni, rebalance_days=5)
    assert set(r.metrics) == {"strategy", "SPY"}
    assert (r.equity > 0).all() and len(r.equity) == len(r.benchmark)
    s = etf_cfg.strategy
    assert (r.weights.gt(0).sum(axis=1) <= s.max_positions).all()
    assert (r.weights.max(axis=1) <= s.max_weight + s.weight_drift + 1e-9).all()
    assert (r.weights.sum(axis=1) <= s.max_gross + 1e-9).all()
    assert r.info["trades"] == len(r.trades) > 0


def test_backtest_has_no_look_ahead(synthetic_prices, etf_cfg):
    uni = resolve(etf_cfg.universe)
    cut = synthetic_prices.index[-100]
    base = run_backtest(synthetic_prices, etf_cfg, uni)
    shocked = synthetic_prices.copy()
    rng = np.random.default_rng(0)
    shocked.loc[shocked.index > cut] *= rng.uniform(0.5, 2.0, size=shocked.loc[shocked.index > cut].shape)
    other = run_backtest(shocked, etf_cfg, uni)
    pd.testing.assert_series_equal(base.equity.loc[:cut], other.equity.loc[:cut])


def test_performance_metrics():
    eq = pd.Series(np.linspace(1, 2, 253))
    m = performance(eq)
    assert abs(m["total_return"] - 1.0) < 1e-12
    assert abs(m["cagr"] - 1.0) < 1e-9
    assert m["max_drawdown"] == 0.0
