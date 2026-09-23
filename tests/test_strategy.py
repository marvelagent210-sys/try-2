import numpy as np
import pandas as pd

from growthpm.config import Config
from growthpm.signals import compute_panel
from growthpm.strategy import PositionMeta, decide
from growthpm.universe import resolve

from .conftest import MARKET, trend_prices


def small_cfg() -> Config:
    cfg = Config()
    cfg.universe.preset = "none"
    cfg.universe.tickers = ["WIN", "LAG", "MID", "DOG"]
    cfg.strategy.max_positions = 1
    cfg.strategy.max_weight = 0.9
    return cfg


UNI = ["WIN", "LAG", "MID", "DOG"]


def small_panel(cfg, prices):
    return compute_panel(prices, cfg.strategy, cfg.regime, "SPY")


def test_targets_respect_all_limits(synthetic_prices, etf_cfg):
    panel = compute_panel(synthetic_prices, etf_cfg.strategy, etf_cfg.regime, "SPY")
    date = synthetic_prices.index[-1]
    d = decide(panel, date, pd.Series(dtype=float), {}, {}, resolve(etf_cfg.universe), etf_cfg)
    s = etf_cfg.strategy
    held = d.target[d.target > 0]
    assert len(held) <= s.max_positions
    assert (held <= s.max_weight + 1e-9).all()
    assert (held >= s.min_weight - 1e-9).all()
    assert d.target.sum() <= d.regime.gross_cap + 1e-9
    assert all(a.kind == "BUY" for a in d.trades)


def test_decision_is_stable_once_positioned(synthetic_prices, etf_cfg):
    panel = compute_panel(synthetic_prices, etf_cfg.strategy, etf_cfg.regime, "SPY")
    date = synthetic_prices.index[-1]
    uni = resolve(etf_cfg.universe)
    first = decide(panel, date, pd.Series(dtype=float), {}, {}, uni, etf_cfg)
    price = synthetic_prices.loc[date]
    meta = {t: PositionMeta(date, price[t], price[t]) for t in first.target[first.target > 0].index}
    second = decide(panel, date, first.target, meta, {}, uni, etf_cfg)
    assert second.trades == []


def test_rotates_out_of_a_laggard_into_a_better_opportunity():
    cfg = small_cfg()
    prices = trend_prices(MARKET)
    panel = small_panel(cfg, prices)
    date = prices.index[-1]
    meta = {"LAG": PositionMeta(date - pd.Timedelta(days=60), 90.0, prices["LAG"].iloc[-1])}
    d = decide(panel, date, pd.Series({"LAG": 0.5}), meta, {}, UNI, cfg)
    kinds = {a.ticker: a for a in d.trades}
    assert kinds["LAG"].kind == "SELL" and "WIN" in kinds["LAG"].reason
    assert kinds["WIN"].kind == "BUY"


def test_trailing_stop_forces_exit():
    cfg = small_cfg()
    prices = trend_prices(MARKET)
    panel = small_panel(cfg, prices)
    date = prices.index[-1]
    peak = prices["WIN"].iloc[-1] / 0.7   # price is 30% under its peak
    meta = {"WIN": PositionMeta(date - pd.Timedelta(days=90), 100.0, peak)}
    d = decide(panel, date, pd.Series({"WIN": 0.5}), meta, {}, UNI, cfg)
    assert d.target["WIN"] == 0
    assert "trailing stop" in d.forced_exits["WIN"]
    assert any(a.ticker == "WIN" and a.kind == "SELL" and "trailing stop" in a.reason for a in d.trades)


def test_trend_break_and_universe_removal_force_exits():
    cfg = small_cfg()
    prices = trend_prices({**MARKET, "DROP": (-0.5, 0.2)})
    panel = small_panel(cfg, prices)
    date = prices.index[-1]
    current = pd.Series({"DROP": 0.2, "MID": 0.2})
    meta = {t: PositionMeta(date, prices[t].iloc[-1], prices[t].iloc[-1]) for t in current.index}
    d = decide(panel, date, current, meta, {}, ["WIN", "LAG", "DOG", "DROP"], cfg)
    assert "trend break" in d.forced_exits["DROP"]
    assert d.forced_exits["MID"] == "removed from universe"
    assert d.target["DROP"] == 0 and d.target["MID"] == 0


def test_cooldown_blocks_reentry():
    cfg = small_cfg()
    prices = trend_prices(MARKET)
    panel = small_panel(cfg, prices)
    date = prices.index[-1]
    cooldown = {"WIN": date + pd.Timedelta(days=5)}
    d = decide(panel, date, pd.Series(dtype=float), {}, cooldown, UNI, cfg)
    assert d.target["WIN"] == 0
    assert d.target.sum() > 0      # capital goes to the next-best idea instead


def test_risk_off_regime_caps_exposure():
    cfg = small_cfg()
    cfg.strategy.max_positions = 3
    # Market in a downtrend: benchmark below its 200-day average.
    prices = trend_prices({**MARKET, "SPY": (-0.4, 0.15)}, days=420)
    panel = small_panel(cfg, prices)
    date = prices.index[-1]
    d = decide(panel, date, pd.Series(dtype=float), {}, {}, UNI, cfg)
    assert d.regime.name in ("risk_off", "neutral")
    assert d.target.sum() <= cfg.strategy.max_gross * cfg.regime.gross_neutral + 1e-9
    assert np.isfinite(d.growth_target)
