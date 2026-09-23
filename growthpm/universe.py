"""Tradable universes. Momentum decides what is actually bought; these are just the menu."""

from __future__ import annotations

from .config import UniverseConfig

GROWTH_STOCKS = [
    # Mega-cap platforms
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "NFLX", "ORCL",
    # Semiconductors
    "AMD", "MU", "AMAT", "LRCX", "KLAC", "TSM", "ASML", "QCOM", "ARM",
    # Software / internet
    "CRM", "ADBE", "NOW", "INTU", "PANW", "CRWD", "NET", "DDOG", "SNOW", "PLTR",
    "APP", "SHOP", "MELI", "UBER", "SPOT", "ANET",
    # Other secular growth
    "LLY", "ISRG", "COST", "V", "MA", "JPM", "GE", "CAT", "AXON",
    "VRT", "CEG", "VST", "COIN", "HOOD",
]

SECTOR_ETFS = [
    "SPY", "QQQ", "IWM", "SMH", "XLK", "IGV", "XLC", "XLY", "XLF", "XLI",
    "XLE", "XLV", "XLU", "XLP", "EEM", "EFA",
]

# Low or negative equity beta: what momentum rotates into when stocks break down.
DEFENSIVE = ["TLT", "IEF", "GLD"]

AGGRESSIVE_EXTRAS = [
    "TQQQ", "SOXL", "UPRO", "TECL",   # 3x leveraged ETFs (daily reset, volatility decay)
    "IBIT", "ETHA", "MSTR",           # crypto exposure through US-listed securities
]

PRESETS: dict[str, list[str]] = {
    "growth": GROWTH_STOCKS + SECTOR_ETFS + DEFENSIVE,
    "aggressive": GROWTH_STOCKS + SECTOR_ETFS + DEFENSIVE + AGGRESSIVE_EXTRAS,
    "etf": SECTOR_ETFS + DEFENSIVE,
    "none": [],
}


def resolve(cfg: UniverseConfig) -> list[str]:
    """Tradable symbols. The benchmark is fetched separately and only traded if listed here."""
    if cfg.preset not in PRESETS:
        raise ValueError(f"unknown universe preset '{cfg.preset}' (choose from {sorted(PRESETS)})")
    excluded = {t.upper() for t in cfg.exclude}
    out: list[str] = []
    for t in PRESETS[cfg.preset] + [t.upper() for t in cfg.tickers]:
        if t not in excluded and t not in out:
            out.append(t)
    if not out:
        raise ValueError("the universe is empty: pick a preset or list tickers")
    return out
