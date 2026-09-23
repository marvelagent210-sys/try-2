"""Signal panel: every indicator the strategy uses, computed once over the full
price history with backward-looking windows only. The row at date t depends
only on prices up to and including t, so backtests and live runs share one
code path and the backtest cannot peek at the future."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import RegimeConfig, StrategyConfig

TRADING_DAYS = 252


@dataclass
class SignalPanel:
    prices: pd.DataFrame
    returns: pd.DataFrame
    momentum: pd.DataFrame      # blended risk-adjusted momentum (a t-stat-like score)
    vol: pd.DataFrame           # annualised realised volatility
    abs_momentum: pd.DataFrame  # plain 126-day return, for the absolute-momentum filter
    sma_entry: pd.DataFrame
    sma_exit: pd.DataFrame
    beta: pd.DataFrame          # 252-day beta to the benchmark
    nobs: pd.DataFrame          # count of valid prices so far
    benchmark: str
    bench_sma: pd.Series        # regime moving average of the benchmark

    def row(self, date: pd.Timestamp) -> dict[str, pd.Series]:
        return {
            "price": self.prices.loc[date],
            "momentum": self.momentum.loc[date],
            "vol": self.vol.loc[date],
            "abs_momentum": self.abs_momentum.loc[date],
            "sma_entry": self.sma_entry.loc[date],
            "sma_exit": self.sma_exit.loc[date],
            "beta": self.beta.loc[date],
            "nobs": self.nobs.loc[date],
        }


def compute_panel(prices: pd.DataFrame, cfg: StrategyConfig, regime: RegimeConfig,
                  benchmark: str) -> SignalPanel:
    if benchmark not in prices.columns:
        raise ValueError(f"benchmark {benchmark} missing from prices")
    returns = prices.pct_change(fill_method=None)
    vol = returns.rolling(cfg.vol_window, min_periods=cfg.vol_window // 2).std() * np.sqrt(TRADING_DAYS)
    vol = vol.clip(lower=0.05)

    momentum = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for k, w in zip(cfg.lookbacks, cfg.lookback_weights):
        period_ret = prices / prices.shift(k) - 1.0
        momentum = momentum + w * period_ret / (vol * np.sqrt(k / TRADING_DAYS))
    momentum = momentum / float(sum(cfg.lookback_weights))
    if cfg.smoothing > 1:
        # Backward-looking EMA; NaN until every lookback has data, so no warm-up bias.
        raw = momentum
        momentum = raw.ewm(span=cfg.smoothing, min_periods=cfg.smoothing).mean().where(raw.notna())

    def sma(n: int) -> pd.DataFrame:
        return prices.rolling(n, min_periods=n).mean()

    bench = returns[benchmark]
    mask = returns.notna()
    b = pd.DataFrame(np.where(mask, bench.to_numpy()[:, None], np.nan),
                     index=returns.index, columns=returns.columns)
    win, minp = TRADING_DAYS, TRADING_DAYS // 2
    mean_rb = (returns * b).rolling(win, min_periods=minp).mean()
    mean_r = returns.rolling(win, min_periods=minp).mean()
    mean_b = b.rolling(win, min_periods=minp).mean()
    var_b = (b * b).rolling(win, min_periods=minp).mean() - mean_b**2
    beta = ((mean_rb - mean_r * mean_b) / var_b.where(var_b > 0)).clip(-1.0, 4.0)

    return SignalPanel(
        prices=prices,
        returns=returns,
        momentum=momentum,
        vol=vol,
        abs_momentum=prices / prices.shift(126) - 1.0,
        sma_entry=sma(cfg.entry_sma),
        sma_exit=sma(cfg.exit_sma),
        beta=beta,
        nobs=prices.notna().cumsum(),
        benchmark=benchmark,
        bench_sma=prices[benchmark].rolling(regime.sma, min_periods=regime.sma).mean(),
    )


def zscore(values: pd.Series, clip: float = 3.0) -> pd.Series:
    """Cross-sectional z-score, winsorised so one outlier cannot dominate."""
    v = values.dropna()
    if len(v) < 2 or v.std(ddof=0) == 0:
        return pd.Series(0.0, index=values.index)
    z = (values - v.mean()) / v.std(ddof=0)
    return z.clip(-clip, clip)
