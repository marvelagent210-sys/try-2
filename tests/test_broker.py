import pandas as pd
import pytest

from growthpm.broker import AlpacaBroker, Order, PaperBroker, build_orders
from growthpm.config import ExecutionConfig
from growthpm.portfolio import Holding, PortfolioState

PRICES = pd.Series({"AAA": 100.0, "BBB": 50.0, "CCC": 20.0})


def test_orders_sell_first_skip_holds_and_close_exits():
    shares = {"AAA": 100.0, "BBB": 200.0}          # 10k + 10k, plus 5k cash = 25k equity
    current = pd.Series({"AAA": 0.4, "BBB": 0.4})
    target = pd.Series({"AAA": 0.4, "BBB": 0.0, "CCC": 0.4})
    orders = build_orders(target, current, shares, PRICES, 25_000, 5_000, ExecutionConfig())
    assert [o.ticker for o in orders] == ["BBB", "CCC"]
    sell, buy = orders
    assert sell.side == "sell" and sell.close_position and sell.shares == 200
    assert buy.side == "buy" and buy.value == pytest.approx(10_000, rel=1e-4)


def test_buys_are_scaled_to_available_cash():
    target = pd.Series({"AAA": 0.6, "CCC": 0.6})   # asks for 120% of equity
    orders = build_orders(target, pd.Series(dtype=float), {}, PRICES, 10_000, 10_000, ExecutionConfig())
    spent = sum(o.value for o in orders) * 1.001
    assert spent <= 10_000 * (1 - ExecutionConfig().cash_buffer) + 1e-6


def test_whole_share_mode_rounds_down_buys():
    cfg = ExecutionConfig(fractional_shares=False)
    orders = build_orders(pd.Series({"AAA": 0.5}), pd.Series(dtype=float), {}, PRICES, 10_150, 10_150, cfg)
    assert orders[0].shares == 50


def test_paper_broker_accounts_for_costs():
    state = PortfolioState.new(10_000)
    state.holdings["AAA"] = Holding(10, "2026-01-02", 90.0, 110.0)
    broker = PaperBroker(state, cost_bps=10)
    when = pd.Timestamp("2026-06-30 15:45")
    fills = broker.execute([Order("AAA", "sell", 10, 100.0, close_position=True),
                            Order("BBB", "buy", 20, 50.0)], when)
    assert [f.ticker for f in fills] == ["AAA", "BBB"]
    assert state.cash == pytest.approx(10_000 + 1000 * 0.999 - 1000 * 1.001)
    assert "AAA" not in state.holdings
    assert state.holdings["BBB"].shares == 20 and state.holdings["BBB"].entry_date == "2026-06-30"


def test_state_round_trips(tmp_path):
    state = PortfolioState.new(1234.5)
    state.holdings["AAA"] = Holding(1.5, "2026-01-02", 90.0, 110.0)
    state.cooldown["BBB"] = "2026-07-10"
    path = tmp_path / "s.json"
    state.save(path)
    loaded = PortfolioState.load(path)
    assert loaded.cash == 1234.5 and loaded.holdings["AAA"] == state.holdings["AAA"]
    assert loaded.cooldown == {"BBB": "2026-07-10"}


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status
        self.content = b"x"
        self.text = str(payload)

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.headers = {}
        self.calls = []

    def request(self, method, url, timeout=None, json=None):
        self.calls.append((method, url, json))
        path = url.split(".markets", 1)[1]
        if path == "/v2/account":
            return FakeResponse({"cash": "2500.50", "equity": "10000"})
        if path == "/v2/positions":
            return FakeResponse([{"symbol": "AAA", "qty": "3.5"}])
        if method == "DELETE":
            return FakeResponse({"id": "o-close", "symbol": "AAA", "side": "sell"})
        if method == "POST":
            return FakeResponse({"id": "o-" + json["symbol"], "symbol": json["symbol"], "side": json["side"]})
        if path.startswith("/v2/orders/"):
            sym = "AAA" if path.endswith("close") else path.rsplit("-", 1)[1]
            side = "sell" if sym == "AAA" else "buy"
            return FakeResponse({"id": path, "symbol": sym, "side": side, "status": "filled",
                                 "filled_qty": "3.5", "filled_avg_price": "101.0"})
        raise AssertionError(f"unexpected call {method} {url}")


def test_alpaca_defaults_to_paper_and_places_orders_in_order():
    session = FakeSession()
    broker = AlpacaBroker("key", "secret", session=session, poll_interval=0)
    assert broker.base == AlpacaBroker.PAPER_URL
    assert session.headers["APCA-API-KEY-ID"] == "key"
    cash, shares = broker.snapshot()
    assert cash == 2500.50 and shares == {"AAA": 3.5}

    orders = [Order("AAA", "sell", 3.5, 100.0, close_position=True), Order("BBB", "buy", 1.25, 50.0)]
    fills = broker.execute(orders, pd.Timestamp("2026-06-30"))
    methods = [(m, u.split(".markets")[1]) for m, u, _ in session.calls]
    assert ("DELETE", "/v2/positions/AAA") in methods
    post = [c for c in session.calls if c[0] == "POST"][0]
    assert post[2] == {"symbol": "BBB", "qty": "1.25", "side": "buy", "type": "market", "time_in_force": "day"}
    # the sell is confirmed filled before the buy is sent
    assert methods.index(("GET", "/v2/orders/o-close")) < methods.index(("POST", "/v2/orders"))
    assert [f.ticker for f in fills] == ["AAA", "BBB"]


def test_alpaca_live_requires_explicit_opt_out_of_paper():
    assert AlpacaBroker("k", "s", paper=False, session=FakeSession()).base == AlpacaBroker.LIVE_URL
