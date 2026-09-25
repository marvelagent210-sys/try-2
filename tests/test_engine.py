import pandas as pd
import pytest

from growthpm.cli import main
from growthpm.config import Config
from growthpm.data import SyntheticProvider
from growthpm.engine import last_session, run_once, us_market_open
from growthpm.portfolio import PortfolioState

from .conftest import END


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_dry_run_changes_nothing_but_records_history(workdir, etf_cfg):
    provider = SyntheticProvider(seed=3, end=END)
    result = run_once(etf_cfg, provider, execute=False, now=pd.Timestamp("2026-06-30 15:45"))
    state = PortfolioState.load(workdir / "portfolio_state.json")
    assert state.cash == etf_cfg.account.starting_cash and not state.holdings
    assert len(state.history) == 1 and state.history[0]["synthetic"] is True
    assert result.orders and result.report_path.exists()
    assert "DRY RUN" in result.report and "SYNTHETIC DATA" in result.report


def test_execute_then_rerun_is_idempotent(workdir, etf_cfg):
    provider = SyntheticProvider(seed=3, end=END)
    first = run_once(etf_cfg, provider, execute=True, now=pd.Timestamp("2026-06-30 15:45"))
    assert first.fills
    state = PortfolioState.load(workdir / "portfolio_state.json")
    assert state.holdings and state.cash < etf_cfg.account.starting_cash
    assert len(state.trades) == len(first.fills)
    assert first.equity == pytest.approx(etf_cfg.account.starting_cash, rel=0.01)

    second = run_once(etf_cfg, provider, execute=True, now=pd.Timestamp("2026-06-30 15:50"))
    assert second.orders == []


def test_config_file_round_trip_and_validation(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('[strategy]\nmax_positions = 4\nmax_weight = 1\n[universe]\npreset = "etf"\n')
    cfg = Config.load(path)
    assert cfg.strategy.max_positions == 4 and cfg.strategy.max_weight == 1.0
    path.write_text("[strategy]\nbogus = 1\n")
    with pytest.raises(ValueError, match="bogus"):
        Config.load(path)


def test_cli_run_and_status(workdir, capsys):
    (workdir / "config.toml").write_text('[universe]\npreset = "etf"\n')
    assert main(["run", "--provider", "synthetic:3", "--execute"]) == 0
    assert "EXECUTED on paper broker" in capsys.readouterr().out
    assert main(["status"]) == 0
    assert "recent trades" in capsys.readouterr().out


def test_market_hours():
    assert us_market_open(pd.Timestamp("2026-06-30 10:00", tz="America/New_York"))
    assert not us_market_open(pd.Timestamp("2026-06-30 08:00", tz="America/New_York"))
    assert not us_market_open(pd.Timestamp("2026-06-27 12:00", tz="America/New_York"))  # Saturday
    assert us_market_open(pd.Timestamp("2026-06-30 15:00", tz="UTC"))                   # 11:00 NY


class LiveLikeProvider(SyntheticProvider):
    """Synthetic prices that the engine treats as real market data."""
    is_synthetic = False


@pytest.mark.parametrize("now, reason", [
    ("2026-07-02 10:00", "stale prices"),     # Thursday session open, newest bar is Tuesday
    ("2026-06-30 18:00", "market closed"),    # after the close: no paper fills at a stale close
    ("2026-07-01 08:00", "market closed"),    # Wednesday pre-open: Tuesday's bar is fresh, but no fills yet
])
def test_real_data_guards_block_execution(workdir, etf_cfg, now, reason):
    result = run_once(etf_cfg, LiveLikeProvider(seed=3, end=END), execute=True, now=pd.Timestamp(now))
    assert result.orders and not result.fills
    assert reason in result.report and "NOT EXECUTED" in result.report
    state = PortfolioState.load(workdir / "portfolio_state.json")
    assert not state.holdings and reason in state.history[-1]["blocked"]


def test_fresh_prices_during_session_trade(workdir, etf_cfg):
    result = run_once(etf_cfg, LiveLikeProvider(seed=3, end=END), execute=True,
                      now=pd.Timestamp("2026-06-30 15:45"))
    assert result.fills and "EXECUTED on paper broker" in result.report


def test_last_session():
    ny = "America/New_York"
    assert last_session(pd.Timestamp("2026-09-23 03:24", tz=ny)) == pd.Timestamp("2026-09-22")
    assert last_session(pd.Timestamp("2026-09-23 07:24", tz="UTC")) == pd.Timestamp("2026-09-22")
    assert last_session(pd.Timestamp("2026-09-23 10:00", tz=ny)) == pd.Timestamp("2026-09-23")
    assert last_session(pd.Timestamp("2026-09-26 12:00", tz=ny)) == pd.Timestamp("2026-09-25")  # Saturday
    assert last_session(pd.Timestamp("2026-09-28 08:00", tz=ny)) == pd.Timestamp("2026-09-25")  # Monday pre-open
