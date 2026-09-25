"""Configuration: every tunable knob of the manager, loadable from TOML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


@dataclass
class UniverseConfig:
    preset: str = "growth"              # growth | aggressive | etf | none
    tickers: list[str] = field(default_factory=list)   # added to the preset
    exclude: list[str] = field(default_factory=list)
    benchmark: str = "SPY"
    vix: str = "^VIX"                   # "" disables the volatility-index check


@dataclass
class StrategyConfig:
    # Momentum: risk-adjusted return over each lookback, blended with these weights.
    lookbacks: list[int] = field(default_factory=lambda: [21, 63, 126, 252])
    lookback_weights: list[float] = field(default_factory=lambda: [1.0, 2.0, 2.0, 1.0])
    vol_window: int = 63
    smoothing: int = 10                 # EMA span (days) on the momentum score; damps churn
    entry_sma: int = 100                # new buys need price above this SMA ...
    exit_sma: int = 200                 # ... held names are sold below this one
    min_history: int = 260              # trading days of data required to trade a name
    # Expected return model: mu = regime_premium * beta + ic * vol * z * sqrt(12)
    ic: float = 0.08
    # Sizing: fractional Kelly on the candidates, subject to caps.
    kelly_fraction: float = 0.5
    max_positions: int = 10
    max_weight: float = 0.20
    weight_drift: float = 0.05          # winners may drift this far above max_weight before a trim
    min_weight: float = 0.03
    max_gross: float = 1.0              # >1.0 means margin
    candidate_pool: int = 25            # best-ranked non-held names fed to the optimizer
    cov_lookback: int = 126
    cov_shrink: float = 0.3
    # Risk exits.
    trailing_stop: float = 0.25         # sell after this drawdown from the post-entry peak
    reentry_cooldown_days: int = 10     # after a forced exit, block re-entry this long
    risk_free_rate: float = 0.0         # annual rate earned on idle cash (backtests)


@dataclass
class RegimeConfig:
    sma: int = 200
    buffer: float = 0.02                # hysteresis around the benchmark's average (no whipsaw)
    breadth_threshold: float = 0.5
    vix_high: float = 30.0
    premium_risk_on: float = 0.06
    premium_neutral: float = 0.02
    premium_risk_off: float = -0.04
    gross_neutral: float = 0.8          # multiplier on max_gross
    gross_risk_off: float = 0.5


@dataclass
class ExecutionConfig:
    trade_cost_bps: float = 10.0        # commission + spread + slippage, one way
    expected_hold_days: int = 63        # amortises trade cost into an annual hurdle
    switch_hurdle: float = 0.06         # extra annual return a switch must earn
    rebalance_band: float = 0.02        # ignore weight changes smaller than this
    fractional_shares: bool = True
    cash_buffer: float = 0.005
    min_order_value: float = 50.0

    def turnover_penalty(self) -> tuple[float, float]:
        """(buy, sell) annualised growth a trade must add per unit of weight moved.
        Deploying idle cash pays only the amortised trading cost; selling also
        pays the switch hurdle, so a full swap must clear cost + cost + hurdle."""
        one_way = self.trade_cost_bps / 1e4 * 252.0 / max(self.expected_hold_days, 1)
        return one_way, one_way + self.switch_hurdle


@dataclass
class AccountConfig:
    broker: str = "paper"               # paper | alpaca
    starting_cash: float = 100_000.0
    state_file: str = "portfolio_state.json"
    reports_dir: str = "reports"
    dashboard: str = "dashboard.html"   # phone-friendly page rewritten every cycle ("" = off)
    alpaca_paper: bool = True           # False sends orders to a LIVE Alpaca account
    history_days: int = 800             # calendar days of prices to fetch


@dataclass
class Config:
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    account: AccountConfig = field(default_factory=AccountConfig)

    @classmethod
    def load(cls, path: str | Path | None) -> Config:
        if path is None or not Path(path).exists():
            return cls()
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        cfg = cls()
        for section, values in raw.items():
            if not hasattr(cfg, section):
                raise ValueError(f"unknown config section [{section}]")
            _apply(getattr(cfg, section), values, section)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        s = self.strategy
        if len(s.lookbacks) != len(s.lookback_weights):
            raise ValueError("strategy.lookbacks and strategy.lookback_weights differ in length")
        if not 0 < s.kelly_fraction <= 1.5:
            raise ValueError("strategy.kelly_fraction must be in (0, 1.5]")
        if not 0 < s.min_weight <= s.max_weight <= 1:
            raise ValueError("need 0 < min_weight <= max_weight <= 1")
        if s.max_positions < 1:
            raise ValueError("strategy.max_positions must be >= 1")
        if not 0 < s.trailing_stop < 1:
            raise ValueError("strategy.trailing_stop must be in (0, 1)")
        if self.account.broker not in ("paper", "alpaca"):
            raise ValueError("account.broker must be 'paper' or 'alpaca'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _apply(target: Any, values: dict[str, Any], section: str) -> None:
    known = {f.name: f for f in fields(target)}
    for key, value in values.items():
        if key not in known:
            raise ValueError(f"unknown key '{key}' in [{section}]")
        current = getattr(target, key)
        if is_dataclass(current):
            _apply(current, value, f"{section}.{key}")
        elif isinstance(current, float) and isinstance(value, int):
            setattr(target, key, float(value))
        else:
            setattr(target, key, value)


EXAMPLE_TOML = """\
# growthpm configuration. Every key is optional; defaults are shown.

[universe]
preset = "growth"        # growth | aggressive (adds leveraged ETFs + crypto ETFs) | etf | none
tickers = []             # extra symbols to consider, e.g. ["RKLB", "IONQ"]
exclude = []
benchmark = "SPY"
vix = "^VIX"

[strategy]
kelly_fraction = 0.5     # 1.0 = full Kelly (max theoretical growth, brutal drawdowns)
max_positions = 10
max_weight = 0.20
min_weight = 0.03
max_gross = 1.0          # set >1.0 only with a margin account
ic = 0.08                # how much the model trusts momentum
trailing_stop = 0.25
entry_sma = 100
exit_sma = 200

[regime]
premium_risk_on = 0.06
premium_neutral = 0.02
premium_risk_off = -0.04
gross_neutral = 0.8
gross_risk_off = 0.5

[execution]
trade_cost_bps = 10
expected_hold_days = 63
switch_hurdle = 0.06     # a new idea must beat a holding by ~this much per year to swap
rebalance_band = 0.02
fractional_shares = true

[account]
broker = "paper"         # paper | alpaca (reads APCA_API_KEY_ID / APCA_API_SECRET_KEY)
starting_cash = 100000
state_file = "portfolio_state.json"
reports_dir = "reports"
dashboard = "dashboard.html"  # open on your phone, or publish it (see README)
alpaca_paper = true
"""
