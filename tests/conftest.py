import numpy as np
import pandas as pd
import pytest

from growthpm.config import Config
from growthpm.data import SyntheticProvider, align
from growthpm.universe import resolve

END = "2026-06-30"


@pytest.fixture
def cfg() -> Config:
    return Config()


@pytest.fixture
def etf_cfg() -> Config:
    c = Config()
    c.universe.preset = "etf"
    return c


@pytest.fixture
def synthetic_prices(etf_cfg) -> pd.DataFrame:
    uni = resolve(etf_cfg.universe)
    prices = SyntheticProvider(seed=3, end=END).history(uni + ["SPY"], 365 * 4)
    return align(prices, "SPY")


def trend_prices(spec: dict[str, tuple[float, float]], days: int = 400, seed: int = 0,
                 end: str = END) -> pd.DataFrame:
    """Deterministic trending series: {ticker: (annual drift, annual volatility)}."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=end, periods=days)
    cols = {}
    for t, (drift, vol) in spec.items():
        steps = drift / 252 + vol / np.sqrt(252) * rng.standard_normal(days)
        cols[t] = 100 * np.exp(np.cumsum(steps))
    return pd.DataFrame(cols, index=idx)


# A strong grower, a sleepy low-volatility laggard, a middling name and a dud.
MARKET = {"SPY": (0.15, 0.10), "WIN": (1.0, 0.25), "LAG": (0.08, 0.08), "MID": (0.25, 0.15),
          "DOG": (-0.1, 0.25)}
