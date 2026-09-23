"""One management cycle: fetch live data -> decide -> (optionally) trade -> report."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .broker import AlpacaBroker, Broker, Fill, Order, PaperBroker, build_orders
from .config import Config
from .data import PriceProvider, align
from .portfolio import PortfolioState
from .report import render_run
from .signals import compute_panel
from .strategy import Decision, decide
from .universe import resolve

log = logging.getLogger(__name__)
NEW_YORK = ZoneInfo("America/New_York")


@dataclass
class RunResult:
    decision: Decision
    orders: list[Order]
    fills: list[Fill]
    equity: float
    cash: float
    report: str
    report_path: Path | None


def make_broker(cfg: Config, state: PortfolioState) -> Broker:
    if cfg.account.broker == "alpaca":
        return AlpacaBroker.from_env(paper=cfg.account.alpaca_paper)
    return PaperBroker(state, cfg.execution.trade_cost_bps)


def us_market_open(now: pd.Timestamp | None = None) -> bool:
    """Regular NYSE session, Mon-Fri 09:30-16:00 New York time (exchange holidays not modelled)."""
    now = pd.Timestamp.now(tz=NEW_YORK) if now is None else pd.Timestamp(now)
    ny = now.tz_localize(NEW_YORK) if now.tzinfo is None else now.tz_convert(NEW_YORK)
    return ny.weekday() < 5 and dtime(9, 30) <= ny.time() <= dtime(16, 0)


def _latest_vix(provider: PriceProvider, symbol: str) -> float | None:
    if not symbol:
        return None
    try:
        v = provider.history([symbol], days=30)
        return float(v.iloc[:, 0].dropna().iloc[-1])
    except Exception as exc:  # the volatility index is informative, not essential
        log.warning("could not fetch %s: %s", symbol, exc)
        return None


def run_once(cfg: Config, provider: PriceProvider, *, execute: bool = False,
             state_path: str | Path | None = None, broker: Broker | None = None,
             now: pd.Timestamp | None = None, write_report: bool = True) -> RunResult:
    now = now or pd.Timestamp.now()
    state_path = Path(state_path or cfg.account.state_file)
    state = PortfolioState.load(state_path) or PortfolioState.new(cfg.account.starting_cash)
    broker = broker or make_broker(cfg, state)
    cash, shares = broker.snapshot()

    universe = resolve(cfg.universe)
    bench = cfg.universe.benchmark
    wanted = sorted(set(universe) | set(shares) | {bench})
    prices = align(provider.history(wanted, cfg.account.history_days), bench)
    vix = _latest_vix(provider, cfg.universe.vix)
    panel = compute_panel(prices, cfg.strategy, cfg.regime, bench)
    date = prices.index[-1]
    last = prices.iloc[-1]

    unpriced = [t for t in shares if not np.isfinite(last.get(t, np.nan))]
    if unpriced:
        log.warning("no price for held %s: excluded from valuation and left untouched", unpriced)
    priced = {t: q for t, q in shares.items() if t not in unpriced}
    state.sync(shares, prices, date)
    equity = cash + sum(q * float(last[t]) for t, q in priced.items())
    current = pd.Series({t: q * float(last[t]) / equity for t, q in priced.items()}, dtype=float)

    prev_regime = state.history[-1]["regime"] if state.history else None
    decision = decide(panel, date, current, state.meta(), state.cooldown_until(), universe, cfg, vix,
                      prev_regime)
    reasons = {a.ticker: a.reason for a in decision.actions}
    orders = build_orders(decision.target, current, priced, last, equity, cash, cfg.execution, reasons)

    fills: list[Fill] = []
    if execute and orders:
        fills = broker.execute(orders, now)
        cash, shares = broker.snapshot()
        state.sync(shares, prices, date)
        reopen = (date + pd.offsets.BDay(cfg.strategy.reentry_cooldown_days)).date().isoformat()
        for t in decision.forced_exits:
            if shares.get(t, 0) <= 0:
                state.cooldown[t] = reopen
        by_ticker = {o.ticker: o for o in orders}
        for f in fills:
            state.trades.append({
                "time": now.isoformat(timespec="seconds"), "ticker": f.ticker, "side": f.side,
                "shares": round(f.shares, 6), "price": round(f.price, 4),
                "reason": by_ticker[f.ticker].reason if f.ticker in by_ticker else "",
            })
        equity = cash + sum(q * float(last.get(t, 0.0)) for t, q in shares.items()
                            if np.isfinite(last.get(t, np.nan)))

    state.history.append({
        "time": now.isoformat(timespec="seconds"), "price_date": date.date().isoformat(),
        "equity": round(equity, 2), "cash": round(cash, 2), "regime": decision.regime.name,
        "executed": bool(execute and orders), "synthetic": provider.is_synthetic,
    })
    state.save(state_path)

    report = render_run(decision, orders, fills, equity, cash, state.meta(), cfg,
                        executed=execute, broker=getattr(broker, "name", "?"),
                        synthetic=provider.is_synthetic, run_time=now)
    path = None
    if write_report:
        path = Path(cfg.account.reports_dir) / f"run_{now:%Y%m%d_%H%M%S}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report)
    return RunResult(decision, orders, fills, equity, cash, report, path)
