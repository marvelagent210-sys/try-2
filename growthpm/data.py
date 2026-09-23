"""Market data providers.

Every provider returns a wide DataFrame of split/dividend-adjusted closes: a
DatetimeIndex (tz-naive, ascending) by one column per ticker. During market
hours Yahoo's last row is today's live, partial bar, so each run sees current
prices.
"""

from __future__ import annotations

import hashlib
import logging
import zlib
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DEFENSIVE_HINTS = {"TLT", "IEF", "GLD", "SHY", "BIL", "SGOV", "AGG", "BND", "XLU", "XLP"}
INDEX_HINTS = {"SPY", "VOO", "IVV", "QQQ", "IWM", "DIA", "VTI"}


class DataError(RuntimeError):
    pass


class PriceProvider(Protocol):
    name: str
    is_synthetic: bool

    def history(self, tickers: Sequence[str], days: int) -> pd.DataFrame: ...


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    idx = pd.DatetimeIndex(frame.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    frame = frame.copy()
    frame.index = idx
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame.dropna(how="all").dropna(axis=1, how="all").astype(float)


class YahooProvider:
    """Yahoo Finance via yfinance. Full history is cached once per day and only the
    last few sessions are re-downloaded on later runs, which keeps an intraday
    watch loop light on requests."""

    name = "yahoo"
    is_synthetic = False

    def __init__(self, cache_dir: str | Path | None = ".cache", refresh_days: int = 7):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.refresh_days = refresh_days

    def history(self, tickers: Sequence[str], days: int) -> pd.DataFrame:
        tickers = sorted({t.upper() for t in tickers})
        today = pd.Timestamp.now().normalize()
        cache = self._cache_path(tickers, days)
        base = None
        if cache is not None and cache.exists():
            try:
                stored = pd.read_pickle(cache)
                if stored.get("fetched") == today.date().isoformat():
                    base = stored["prices"]
            except Exception as exc:  # corrupt cache: just refetch
                log.warning("ignoring unreadable price cache %s: %s", cache, exc)
        if base is None:
            prices = self._download(tickers, today - pd.Timedelta(days=days))
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                pd.to_pickle({"fetched": today.date().isoformat(), "prices": prices}, cache)
            return prices
        try:
            recent = self._download(tickers, today - pd.Timedelta(days=self.refresh_days))
        except DataError as exc:
            log.warning("intraday refresh failed (%s); using this morning's cached prices", exc)
            return base
        return _normalise(recent.combine_first(base))

    def _cache_path(self, tickers: Sequence[str], days: int) -> Path | None:
        if self.cache_dir is None:
            return None
        key = hashlib.sha1((",".join(tickers) + f"|{days}").encode()).hexdigest()[:16]
        return self.cache_dir / f"yahoo_{key}.pkl"

    @staticmethod
    def _download(tickers: Sequence[str], start: pd.Timestamp) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover
            raise DataError("yfinance is not installed: pip install yfinance") from exc
        raw = yf.download(
            list(tickers), start=start.strftime("%Y-%m-%d"), auto_adjust=True,
            progress=False, threads=True, group_by="column",
        )
        if raw is None or raw.empty:
            raise DataError("Yahoo Finance returned no data (network blocked or rate limited?)")
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw[["Close"]].rename(columns={"Close": tickers[0]})
        if isinstance(close, pd.Series):
            close = close.to_frame(tickers[0])
        close = _normalise(close)
        missing = [t for t in tickers if t not in close.columns]
        if missing:
            log.warning("no Yahoo data for: %s", ", ".join(missing))
        return close


class CSVProvider:
    """Wide CSV: first column is the date, then one adjusted-close column per ticker."""

    name = "csv"
    is_synthetic = False

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def history(self, tickers: Sequence[str], days: int) -> pd.DataFrame:
        frame = pd.read_csv(self.path, index_col=0, parse_dates=True)
        frame.columns = [str(c).upper() for c in frame.columns]
        wanted = [t.upper() for t in tickers if t.upper() in frame.columns]
        missing = sorted(set(t.upper() for t in tickers) - set(wanted))
        if missing:
            log.warning("CSV has no column for: %s", ", ".join(missing))
        frame = _normalise(frame[wanted])
        cutoff = frame.index[-1] - pd.Timedelta(days=days)
        return frame.loc[frame.index >= cutoff]


class SyntheticProvider:
    """Deterministic fake market for offline demos and tests. NOT real prices.

    A two-state (bull/bear) market factor drives every asset through its beta,
    and each asset carries a slowly mean-reverting drift. The persistent drift
    gives momentum something real to find, so the machinery can be exercised
    end to end without network access. The path is anchored at a fixed origin,
    so moving `end` forward reveals more of the same market, as time would.
    """

    name = "synthetic"
    is_synthetic = True
    ORIGIN = pd.Timestamp("2008-01-01")
    TOTAL_DAYS = 252 * 32

    def __init__(self, seed: int = 7, end: str | pd.Timestamp | None = None):
        self.seed = seed
        self.end = pd.Timestamp(end).normalize() if end is not None else pd.Timestamp.now().normalize()
        self._market_cache: np.ndarray | None = None

    def history(self, tickers: Sequence[str], days: int) -> pd.DataFrame:
        dates = pd.bdate_range(start=self.ORIGIN, periods=self.TOTAL_DAYS)
        stop = int(dates.searchsorted(self.end, side="right"))
        if stop == 0 or self.end > dates[-1]:
            raise DataError(f"synthetic market covers {dates[0]:%Y-%m-%d}..{dates[-1]:%Y-%m-%d}")
        n = min(int(days * 252 / 365) + 1, stop)
        market = self._market()
        cols = {}
        for t in tickers:
            t = t.upper()
            if t.startswith("^"):
                vol = pd.Series(market).rolling(21, min_periods=5).std().bfill().to_numpy()
                cols[t] = np.maximum(10.0, 100 * vol * np.sqrt(252) + 3.0)
            else:
                cols[t] = 100.0 * np.exp(np.cumsum(self._asset_log_returns(t, market)))
        return pd.DataFrame(cols, index=dates).iloc[stop - n:stop]

    def _market(self) -> np.ndarray:
        if self._market_cache is None:
            rng = np.random.default_rng(self.seed)
            u, eps = rng.random(self.TOTAL_DAYS), rng.standard_normal(self.TOTAL_DAYS)
            bull = np.empty(self.TOTAL_DAYS, dtype=bool)
            state = True
            for i in range(self.TOTAL_DAYS):
                state = not (u[i] < 1 / 500) if state else u[i] < 1 / 120
                bull[i] = state
            mu = np.where(bull, 0.14, -0.30)
            sigma = np.where(bull, 0.14, 0.28)
            self._market_cache = mu / 252 + sigma / np.sqrt(252) * eps
        return self._market_cache

    def _asset_log_returns(self, ticker: str, market: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng([self.seed, zlib.crc32(ticker.encode())])
        n = len(market)
        if ticker in INDEX_HINTS:      # broad index: the market plus a little noise
            beta, idio, sigma_a = rng.uniform(0.95, 1.15), rng.uniform(0.02, 0.05), 0.03
        elif ticker in DEFENSIVE_HINTS:
            beta, idio, sigma_a = rng.uniform(-0.3, 0.2), rng.uniform(0.06, 0.15), 0.08
        else:
            beta, idio, sigma_a = rng.uniform(0.7, 1.6), rng.uniform(0.15, 0.45), 0.35
        phi = 1 - 1 / 126
        shocks = rng.standard_normal(n) * sigma_a * np.sqrt(1 - phi**2)
        alpha = np.empty(n)
        a = rng.standard_normal() * sigma_a
        for i in range(n):
            a = phi * a + shocks[i]
            alpha[i] = a
        noise = rng.standard_normal(n) * idio / np.sqrt(252)
        return beta * market + alpha / 252 + noise - 0.5 * (idio**2) / 252


def align(prices: pd.DataFrame, calendar: str, ffill_limit: int = 5) -> pd.DataFrame:
    """Put everything on the benchmark's trading calendar (drops crypto weekends)."""
    if calendar not in prices.columns:
        raise DataError(f"benchmark {calendar} missing from price data")
    idx = prices[calendar].dropna().index
    return prices.reindex(idx).ffill(limit=ffill_limit).dropna(axis=1, how="all")


def make_provider(spec: str, cache_dir: str | Path | None = ".cache") -> PriceProvider:
    """'yahoo' (default), 'synthetic', 'synthetic:<seed>', or 'csv:<path>'."""
    if spec == "yahoo":
        return YahooProvider(cache_dir=cache_dir)
    if spec.startswith("synthetic"):
        _, _, seed = spec.partition(":")
        return SyntheticProvider(seed=int(seed) if seed else 7)
    if spec.startswith("csv:"):
        return CSVProvider(spec[4:])
    raise ValueError(f"unknown data provider '{spec}'")
