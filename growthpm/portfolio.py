"""Persistent portfolio state: cash, holdings with entry/peak metadata, cooldowns,
equity history and the trade log. Stored as JSON so it is easy to inspect."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .strategy import PositionMeta


@dataclass
class Holding:
    shares: float
    entry_date: str
    entry_price: float
    peak_price: float


@dataclass
class PortfolioState:
    cash: float
    holdings: dict[str, Holding] = field(default_factory=dict)
    cooldown: dict[str, str] = field(default_factory=dict)   # ticker -> ISO date re-entry opens
    history: list[dict] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)
    created: str = field(default_factory=lambda: pd.Timestamp.now().isoformat(timespec="seconds"))

    @classmethod
    def new(cls, cash: float) -> PortfolioState:
        return cls(cash=float(cash))

    @classmethod
    def load(cls, path: str | Path) -> PortfolioState | None:
        path = Path(path)
        if not path.exists():
            return None
        raw = json.loads(path.read_text())
        raw["holdings"] = {t: Holding(**h) for t, h in raw.get("holdings", {}).items()}
        return cls(**raw)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".state-", suffix=".json")
        with os.fdopen(fd, "w") as fh:
            json.dump(asdict(self), fh, indent=2, default=str)
        os.replace(tmp, path)   # atomic: a crash never leaves a half-written ledger

    def shares(self) -> dict[str, float]:
        return {t: h.shares for t, h in self.holdings.items() if h.shares > 0}

    def meta(self) -> dict[str, PositionMeta]:
        return {
            t: PositionMeta(pd.Timestamp(h.entry_date), h.entry_price, h.peak_price)
            for t, h in self.holdings.items()
        }

    def cooldown_until(self) -> dict[str, pd.Timestamp]:
        return {t: pd.Timestamp(d) for t, d in self.cooldown.items()}

    def sync(self, shares: dict[str, float], prices: pd.DataFrame, date: pd.Timestamp) -> None:
        """Mirror broker share counts and refresh each position's post-entry peak.
        Positions the ledger has never seen (e.g. bought by hand) start today."""
        last = prices.iloc[-1]
        for t in list(self.holdings):
            if shares.get(t, 0) <= 0:
                del self.holdings[t]
        for t, q in shares.items():
            if q <= 0:
                continue
            px = float(last.get(t, np.nan))
            h = self.holdings.get(t)
            if h is None:
                h = self.holdings[t] = Holding(q, date.date().isoformat(), px, px)
            h.shares = float(q)
            if t in prices.columns:
                since = prices.loc[prices.index >= pd.Timestamp(h.entry_date), t].max()
                if np.isfinite(since):
                    h.peak_price = float(np.nanmax([h.peak_price, since]))
        today = date.normalize()
        self.cooldown = {t: d for t, d in self.cooldown.items() if pd.Timestamp(d) > today}
