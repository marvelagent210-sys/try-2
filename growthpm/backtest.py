"""Walk-forward backtest that calls the exact same decide() as live trading.

At each close t the strategy sees prices up to t, trades at t's close (paying
costs on turnover), and earns t -> t+1 returns. Trailing stops and trend exits
are checked every day; the full re-optimisation runs every `rebalance_days`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .config import Config
from .data import align
from .signals import TRADING_DAYS, compute_panel
from .strategy import EPS, PositionMeta, decide, forced_exits


@dataclass
class BacktestResult:
    equity: pd.Series
    benchmark: pd.Series
    weights: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, dict[str, float]]
    info: dict[str, object]


def performance(equity: pd.Series, rf: float = 0.0) -> dict[str, float]:
    rets = equity.pct_change().dropna()
    years = max(len(rets) / TRADING_DAYS, 1e-9)
    total = float(equity.iloc[-1] / equity.iloc[0] - 1)
    cagr = (1 + total) ** (1 / years) - 1
    vol = float(rets.std() * np.sqrt(TRADING_DAYS))
    downside = float(rets[rets < 0].std() * np.sqrt(TRADING_DAYS))
    excess = float(rets.mean() * TRADING_DAYS - rf)
    max_dd = float((equity / equity.cummax() - 1).min())
    return {
        "cagr": cagr, "vol": vol, "sharpe": excess / vol if vol > 0 else 0.0,
        "sortino": excess / downside if downside > 0 else 0.0, "max_drawdown": max_dd,
        "calmar": cagr / abs(max_dd) if max_dd < 0 else 0.0, "total_return": total,
    }


def run_backtest(prices: pd.DataFrame, cfg: Config, universe: Sequence[str], *,
                 vix: pd.Series | None = None, start: str | None = None, end: str | None = None,
                 rebalance_days: int = 5, synthetic: bool = False) -> BacktestResult:
    s = cfg.strategy
    bench = cfg.universe.benchmark
    prices = align(prices, bench)
    panel = compute_panel(prices, s, cfg.regime, bench)
    universe = [t for t in universe if t in prices.columns]
    uni = set(universe)

    warmup = max(s.min_history, s.cov_lookback, cfg.regime.sma, max(s.lookbacks)) + 1
    dates = prices.index[warmup:]
    if start:
        dates = dates[dates >= pd.Timestamp(start)]
    if end:
        dates = dates[dates <= pd.Timestamp(end)]
    if len(dates) < 2:
        raise ValueError("not enough history for a backtest after the warm-up period")

    rets = panel.returns.fillna(0.0)
    rf_daily = s.risk_free_rate / TRADING_DAYS
    cost_rate = cfg.execution.trade_cost_bps / 1e4
    w = pd.Series(0.0, index=prices.columns)
    meta: dict[str, PositionMeta] = {}
    cooldown: dict[str, pd.Timestamp] = {}
    equity, curve, snapshots, trades = 1.0, [], {}, []
    turnover_total, positions = 0.0, []
    regime = None

    for i, d in enumerate(dates):
        if i > 0:
            r = rets.loc[d]
            port = float((w * r).sum() + (1 - w.sum()) * rf_daily)
            equity *= 1 + port
            w = w * (1 + r) / (1 + port)
        px = prices.loc[d]
        for t, m in meta.items():
            if np.isfinite(px[t]):
                m.peak_price = max(m.peak_price, float(px[t]))

        if i % rebalance_days == 0:
            v = float(vix.asof(d)) if vix is not None else None
            dec = decide(panel, d, w, meta, cooldown, universe, cfg, v, regime)
            regime = dec.regime.name
            new_w, exits = dec.target.copy(), dec.forced_exits
            reasons = {a.ticker: a.reason for a in dec.actions}
            snapshots[d] = new_w[new_w > 0]
        else:
            held = list(w[w > EPS].index)
            exits = forced_exits(panel.row(d), held, meta, uni, cfg)
            new_w = w.copy()
            new_w[list(exits)] = 0.0
            reasons = exits

        delta = new_w - w
        turnover = float(delta.abs().sum())
        if turnover > 0:
            equity *= 1 - turnover * cost_rate
            turnover_total += turnover
            for t in delta[delta.abs() > EPS].index:
                was, now = float(w[t]), float(new_w[t])
                kind = "BUY" if was <= EPS else "SELL" if now <= EPS else "ADD" if now > was else "TRIM"
                trades.append({"date": d, "ticker": t, "action": kind, "from": was, "to": now,
                               "reason": reasons.get(t, "")})
                if kind == "BUY":
                    meta[t] = PositionMeta(d, float(px[t]), float(px[t]))
                elif kind == "SELL":
                    meta.pop(t, None)
                    if t in exits:
                        cooldown[t] = d + pd.offsets.BDay(s.reentry_cooldown_days)
        w = new_w
        curve.append(equity)
        positions.append(int((w > EPS).sum()))

    eq = pd.Series(curve, index=dates, name="strategy")
    bm = (prices[bench].loc[dates] / prices[bench].loc[dates[0]]).rename(bench)
    years = len(dates) / TRADING_DAYS
    metrics = {"strategy": performance(eq, s.risk_free_rate), bench: performance(bm, s.risk_free_rate)}
    info = {
        "start": f"{dates[0]:%Y-%m-%d}", "end": f"{dates[-1]:%Y-%m-%d}", "trades": len(trades),
        "avg_positions": float(np.mean(positions)), "turnover": turnover_total / max(years, 1e-9),
        "rebalance_days": rebalance_days, "synthetic": synthetic,
    }
    weights = pd.DataFrame(snapshots).T.fillna(0.0) if snapshots else pd.DataFrame()
    return BacktestResult(eq, bm, weights, pd.DataFrame(trades), metrics, info)
