"""The decision engine: from a signal snapshot and current holdings to target
weights, with a plain-English reason for every change."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from .config import Config
from .optimizer import expected_log_growth, optimize, trade_cost
from .risk import shrunk_covariance
from .signals import SignalPanel, zscore

EPS = 1e-6


@dataclass
class PositionMeta:
    entry_date: pd.Timestamp
    entry_price: float
    peak_price: float


@dataclass
class Regime:
    name: str               # risk_on | neutral | risk_off
    premium: float          # expected equity premium used for beta exposure
    gross_cap: float        # max total weight in risky assets
    bench_price: float
    bench_sma: float
    breadth: float
    vix: float | None

    def describe(self, benchmark: str) -> str:
        parts = []
        if np.isfinite(self.bench_sma):
            gap = self.bench_price / self.bench_sma - 1
            parts.append(f"{benchmark} {gap:+.1%} vs 200d avg")
        if np.isfinite(self.breadth):
            parts.append(f"breadth {self.breadth:.0%}")
        if self.vix is not None and np.isfinite(self.vix):
            parts.append(f"VIX {self.vix:.1f}")
        return f"{self.name.upper().replace('_', '-')} ({', '.join(parts)})"


@dataclass
class Action:
    ticker: str
    kind: str               # BUY | SELL | ADD | TRIM | HOLD
    from_weight: float
    to_weight: float
    reason: str


@dataclass
class Decision:
    date: pd.Timestamp
    regime: Regime
    target: pd.Series
    current: pd.Series
    table: pd.DataFrame
    actions: list[Action]
    growth_current: float
    growth_target: float
    forced_exits: dict[str, str] = field(default_factory=dict)

    @property
    def trades(self) -> list[Action]:
        return [a for a in self.actions if a.kind != "HOLD"]


def assess_regime(panel: SignalPanel, date: pd.Timestamp, universe: Iterable[str],
                  cfg: Config, vix: float | None, prev: str | None = None) -> Regime:
    """Market regime from the benchmark trend, breadth and volatility. `prev` (the
    last regime) adds hysteresis: leaving a regime takes a clear move past the
    threshold, so a benchmark hovering at its average does not flip daily."""
    r = cfg.regime
    bench_price = float(panel.prices.at[date, panel.benchmark])
    bench_sma = float(panel.bench_sma.loc[date])
    cols = [t for t in universe if t in panel.prices.columns]
    price, sma = panel.prices.loc[date, cols], panel.sma_exit.loc[date, cols]
    ok = sma.notna() & price.notna()
    breadth = float((price[ok] > sma[ok]).mean()) if ok.any() else float("nan")
    vix_high = vix is not None and np.isfinite(vix) and vix > r.vix_high
    sticky = {"risk_on": -1, "risk_off": 1}.get(prev or "", 0)
    above = bench_price > bench_sma * (1 + sticky * r.buffer)
    broad = not np.isfinite(breadth) or breadth >= r.breadth_threshold + sticky * 0.05

    if not np.isfinite(bench_sma):
        name = "neutral"
    elif above and broad and not vix_high:
        name = "risk_on"
    elif not above and (not broad or vix_high):
        name = "risk_off"
    else:
        name = "neutral"
    gmax = cfg.strategy.max_gross
    premium, gross = {
        "risk_on": (r.premium_risk_on, gmax),
        "neutral": (r.premium_neutral, gmax * r.gross_neutral),
        "risk_off": (r.premium_risk_off, gmax * r.gross_risk_off),
    }[name]
    return Regime(name, premium, gross, bench_price, bench_sma, breadth, vix)


def forced_exits(snap: dict[str, pd.Series], held: Iterable[str], meta: dict[str, PositionMeta],
                 universe: set[str], cfg: Config) -> dict[str, str]:
    """Hard risk exits that bypass the optimizer."""
    s = cfg.strategy
    out: dict[str, str] = {}
    for t in held:
        price = snap["price"].get(t, np.nan)
        if t not in universe:
            out[t] = "removed from universe"
        elif not np.isfinite(price):
            out[t] = "no current price data"
        elif t in meta and price <= meta[t].peak_price * (1 - s.trailing_stop):
            dd = price / meta[t].peak_price - 1
            out[t] = f"trailing stop: {dd:.0%} from post-entry peak {meta[t].peak_price:,.2f}"
        elif np.isfinite(snap["sma_exit"].get(t, np.nan)) and price < snap["sma_exit"][t]:
            out[t] = f"trend break: closed below its {s.exit_sma}-day average"
    return out


def decide(panel: SignalPanel, date: pd.Timestamp, current: pd.Series,
           meta: dict[str, PositionMeta], cooldown: dict[str, pd.Timestamp],
           universe: Iterable[str], cfg: Config, vix: float | None = None,
           prev_regime: str | None = None) -> Decision:
    s, x = cfg.strategy, cfg.execution
    universe = [t for t in universe if t in panel.prices.columns]
    uni = set(universe)
    snap = panel.row(date)
    price = snap["price"]
    tickers = panel.prices.columns
    current = current.reindex(tickers).fillna(0.0)
    held = [t for t in tickers if current[t] > EPS]

    regime = assess_regime(panel, date, universe, cfg, vix, prev_regime)
    in_uni = tickers.isin(list(uni))
    valid = (
        in_uni & (snap["nobs"] >= s.min_history) & price.notna() & snap["momentum"].notna()
        & snap["vol"].notna() & snap["beta"].notna()
    )
    z = zscore(snap["momentum"].where(valid)).where(valid)
    mu = (regime.premium * snap["beta"] + s.ic * snap["vol"] * z * np.sqrt(12)).where(valid)
    cooling = pd.Series(False, index=tickers)
    for t, until in cooldown.items():
        if t in cooling.index and date < until:
            cooling[t] = True
    # New money needs an uptrend on both averages, so a fresh buy can never be an
    # immediate trend-break sale.
    entry_ok = (
        valid & (price > snap["sma_entry"]) & (price > snap["sma_exit"])
        & (snap["abs_momentum"] > 0) & (mu > 0) & ~cooling
    )
    exits = forced_exits(snap, held, meta, uni, cfg)

    new_ideas = mu[entry_ok & ~tickers.isin(held)].sort_values(ascending=False)
    cand = held + [t for t in new_ideas.index[: s.candidate_pool]]
    n = len(cand)
    w0 = current[cand].to_numpy()
    mu_c = mu.reindex(cand).fillna(0.0).to_numpy()
    upper = np.empty(n)
    for i, t in enumerate(cand):
        if t in exits:
            upper[i] = 0.0
        elif t in held and not entry_ok[t]:
            upper[i] = min(w0[i], s.max_weight)   # still OK to hold, not to add
        elif s.max_weight < w0[i] <= s.max_weight + s.weight_drift:
            upper[i] = w0[i]                      # let a winner run instead of trimming it
        else:
            upper[i] = s.max_weight

    window = panel.returns.loc[:date, cand].iloc[-s.cov_lookback:]
    cov = shrunk_covariance(window, s.cov_shrink)
    buy_cost, sell_cost = x.turnover_penalty()
    budget = regime.gross_cap

    def solve() -> np.ndarray:
        return optimize(mu_c, cov, w0, upper, budget, s.kelly_fraction, buy_cost, sell_cost)

    w = solve() if n else np.zeros(0)
    for _ in range(8):  # enforce max_positions and min_weight, re-solving each time
        # Keep the names adding the most to the objective, not the largest weights:
        # Kelly gives big weights to low-volatility assets that add little growth.
        contribution = (w * (mu_c - cov @ w / (2 * s.kelly_fraction))
                        - trade_cost(w, w0, buy_cost, sell_cost))
        order = [i for i in np.argsort(-contribution) if w[i] > EPS]
        keep = {i for i in order[: s.max_positions] if w[i] >= s.min_weight - EPS}
        drop = [i for i in range(n) if w[i] > EPS and i not in keep]
        if not drop:
            break
        upper[drop] = 0.0
        w = solve()
    w = _apply_band(w, w0, upper, budget, x.rebalance_band)
    w[w < EPS] = 0.0

    target = pd.Series(0.0, index=tickers)
    target[cand] = w
    growth_now = expected_log_growth(w0, mu_c, cov) if n else 0.0
    growth_new = expected_log_growth(w, mu_c, cov) if n else 0.0

    rank = mu[valid].rank(ascending=False)
    table = pd.DataFrame({
        "price": price, "momentum_z": z, "exp_return": mu, "vol": snap["vol"], "beta": snap["beta"],
        "entry_ok": entry_ok, "rank": rank, "current": current, "target": target,
    }).loc[valid | (current > EPS) | (target > EPS)].sort_values("exp_return", ascending=False)

    actions = _explain(cand, w0, w, upper, exits, mu, z, snap["vol"], rank, regime, entry_ok)
    return Decision(date, regime, target, current, table, actions, growth_now, growth_new, exits)


def _apply_band(w, w0, upper, budget, band) -> np.ndarray:
    """Skip small adjustments to existing positions; they cost more than they earn.
    Holdings that price drift has pushed a hair over the exposure cap are left
    alone too, but never beyond what is already held (no new margin)."""
    w = w.copy()
    limit = max(budget, min(float(w0.sum()), budget + band))
    snapped = []
    for i in range(len(w)):
        if w0[i] > EPS and w[i] > EPS and abs(w[i] - w0[i]) < band and w0[i] <= upper[i] + EPS:
            snapped.append((w0[i] - w[i], i, w[i]))
            w[i] = w0[i]
    # Snapping up can breach the limit: undo the largest upward snaps first.
    for delta, i, orig in sorted(snapped, reverse=True):
        if w.sum() <= limit + 1e-9 or delta <= 0:
            break
        w[i] = orig
    if w.sum() > limit + 1e-9:
        w *= limit / w.sum()
    return w


def _explain(cand, w0, w, upper, exits, mu, z, vol, rank, regime, entry_ok) -> list[Action]:
    growing = sorted(((mu[t], t) for i, t in enumerate(cand) if w[i] > w0[i] + EPS), reverse=True)
    funds = ", ".join(f"{t} ({m:.0%}/yr)" for m, t in growing[:3])
    exposure_cut = w.sum() < w0.sum() - 0.005 and w.sum() >= regime.gross_cap - 1e-4
    regime_txt = f"regime {regime.name}: total exposure capped at {regime.gross_cap:.0%}"
    actions = []
    for i, t in enumerate(cand):
        a, b = w0[i], w[i]
        if a <= EPS and b <= EPS:
            continue
        m = mu.get(t, np.nan)
        m_txt = f"{m:.0%}/yr" if np.isfinite(m) else "n/a"
        if a <= EPS:
            kind = "BUY"
            reason = (f"new opportunity: momentum rank #{int(rank[t])} (z {z[t]:+.1f}), "
                      f"expected {m_txt}, vol {vol[t]:.0%}")
        elif b <= EPS:
            kind = "SELL"
            if t in exits:
                reason = exits[t]
            elif np.isfinite(m) and m <= 0:
                reason = f"expected return turned negative ({m:.1%}/yr)"
            elif funds:
                reason = f"displaced (expected {m_txt}): capital rotates into {funds}"
            elif exposure_cut:
                reason = regime_txt
            else:
                reason = f"no longer earns its risk (expected {m_txt})"
        elif b > a + EPS:
            kind, reason = "ADD", f"raise toward growth-optimal weight (expected {m_txt})"
        elif b < a - EPS:
            kind = "TRIM"
            if upper[i] < a - EPS:
                reason = f"above the position cap ({a:.1%})"
            elif exposure_cut and not funds:
                reason = regime_txt
            elif funds:
                reason = f"trimmed (expected {m_txt}) to fund {funds}"
            elif not entry_ok.get(t, False):
                reason = f"momentum fading (expected {m_txt}): reducing"
            else:
                reason = f"rebalance toward growth-optimal weight (expected {m_txt})"
        else:
            kind, reason = "HOLD", f"expected {m_txt}"
        actions.append(Action(t, kind, float(a), float(b), reason))
    order = {"SELL": 0, "TRIM": 1, "BUY": 2, "ADD": 3, "HOLD": 4}
    actions.sort(key=lambda a: (order[a.kind], -abs(a.to_weight - a.from_weight)))
    return actions
