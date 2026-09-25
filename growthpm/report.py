"""Human-readable Markdown reports (printed to the terminal and saved to disk)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .broker import Fill, Order
from .config import Config
from .strategy import Decision, PositionMeta


def table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    if not rows:
        return "_none_"
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def _pct(x: float, signed: bool = False) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:+.1%}" if signed else f"{x:.1%}"


def render_run(decision: Decision, orders: list[Order], fills: list[Fill], equity: float, cash: float,
               meta: dict[str, PositionMeta], cfg: Config, *, executed: bool, broker: str,
               synthetic: bool, run_time: pd.Timestamp, blocked: str | None = None) -> str:
    d = decision
    lines = [f"# Growth portfolio: {run_time:%Y-%m-%d %H:%M} ET (prices as of {d.date:%Y-%m-%d})", ""]
    if synthetic:
        lines += ["> **SYNTHETIC DATA**: demo prices, not the real market.", ""]
    if executed:
        mode = f"EXECUTED on {broker} broker"
    elif blocked:
        mode = f"NOT EXECUTED: {blocked}"
    else:
        mode = "DRY RUN: no orders sent (use --execute)"
    lines += [
        f"**Mode:** {mode}  ",
        f"**Regime:** {d.regime.describe(cfg.universe.benchmark)}; exposure cap {d.regime.gross_cap:.0%}  ",
        f"**Equity:** ${equity:,.2f} | cash ${cash:,.2f}  ",
        f"**Invested:** {d.current.sum():.1%} in {int((d.current > 0).sum())} positions "
        f"-> target {d.target.sum():.1%} in {int((d.target > 0).sum())}  ",
        f"**Model growth rate** (annual log, excess of cash): current {_pct(d.growth_current)} "
        f"-> target {_pct(d.growth_target)}",
        "",
    ]

    reasons = {a.ticker: a for a in d.actions}
    rows = []
    for o in orders:
        a = reasons.get(o.ticker)
        wt = f"{a.from_weight:.1%} -> {a.to_weight:.1%}" if a else ""
        rows.append([o.side.upper(), o.ticker, f"{o.shares:,.4f}".rstrip("0").rstrip("."),
                     f"${o.value:,.0f}", wt, o.reason])
    lines += [f"## Orders ({len(orders)})", "", table(["Side", "Ticker", "Shares", "Value", "Weight", "Why"], rows), ""]
    if executed:
        frows = [[f.side.upper(), f.ticker, f"{f.shares:,.4f}", f"${f.price:,.2f}", f"${f.value:,.0f}"] for f in fills]
        lines += [f"## Fills ({len(fills)})", "", table(["Side", "Ticker", "Shares", "Price", "Value"], frows), ""]

    held = d.table[d.table["target"] > 0].sort_values("target", ascending=False)
    trows = []
    for t, r in held.iterrows():
        stop = meta[t].peak_price * (1 - cfg.strategy.trailing_stop) if t in meta else r["price"] * (1 - cfg.strategy.trailing_stop)
        trows.append([t, _pct(r["target"]), _pct(r["exp_return"]), _pct(r["vol"]),
                      f"{r['momentum_z']:+.2f}", f"${r['price']:,.2f}", f"${stop:,.2f}"])
    lines += ["## Target portfolio", "",
              table(["Ticker", "Weight", "Exp. return", "Vol", "Momentum z", "Price", "Stop"], trows), ""]

    ideas = d.table[(d.table["target"] <= 0) & d.table["entry_ok"]].head(8)
    irows = [[t, _pct(r["exp_return"]), _pct(r["vol"]), f"{r['momentum_z']:+.2f}", f"${r['price']:,.2f}"]
             for t, r in ideas.iterrows()]
    lines += ["## Next-best opportunities (not held)", "",
              table(["Ticker", "Exp. return", "Vol", "Momentum z", "Price"], irows), ""]

    holds = [a for a in d.actions if a.kind == "HOLD"]
    if holds:
        lines += ["## Unchanged", "", ", ".join(f"{a.ticker} ({a.to_weight:.1%})" for a in holds), ""]
    lines += ["_Expected returns are model estimates from momentum and regime, not forecasts. "
              "Past performance does not guarantee future results._", ""]
    return "\n".join(lines)


def render_backtest(metrics: dict[str, dict[str, float]], info: dict[str, object]) -> str:
    keys = [("cagr", "CAGR", True), ("vol", "Volatility", True), ("sharpe", "Sharpe", False),
            ("sortino", "Sortino", False), ("max_drawdown", "Max drawdown", True),
            ("calmar", "Calmar", False), ("total_return", "Total return", True)]
    names = list(metrics)
    rows = []
    for key, label, is_pct in keys:
        rows.append([label] + [_pct(metrics[n][key]) if is_pct else f"{metrics[n][key]:.2f}" for n in names])
    lines = [f"# Backtest {info['start']} -> {info['end']}", ""]
    if info.get("synthetic"):
        lines += ["> **SYNTHETIC DATA**: this only exercises the machinery; it says nothing about real markets.", ""]
    lines += [table(["Metric"] + names, rows), "",
              f"Trades: {info['trades']} | avg positions: {info['avg_positions']:.1f} | "
              f"annual turnover: {info['turnover']:.0%} | rebalance check every {info['rebalance_days']} day(s)",
              "",
              "_Caveat: the default universe is today's winners, so a historical backtest on it carries "
              "survivorship bias and will look better than real-time results would have._", ""]
    return "\n".join(lines)
