"""Order generation and execution: a local paper broker plus an Alpaca adapter."""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from .config import ExecutionConfig
from .portfolio import Holding, PortfolioState

log = logging.getLogger(__name__)


@dataclass
class Order:
    ticker: str
    side: str               # buy | sell
    shares: float
    price: float            # reference price used for sizing
    reason: str = ""
    close_position: bool = False

    @property
    def value(self) -> float:
        return self.shares * self.price


@dataclass
class Fill:
    ticker: str
    side: str
    shares: float
    price: float

    @property
    def value(self) -> float:
        return self.shares * self.price


class Broker(Protocol):
    name: str

    def snapshot(self) -> tuple[float, dict[str, float]]: ...

    def execute(self, orders: list[Order], when: pd.Timestamp) -> list[Fill]: ...


def build_orders(target: pd.Series, current: pd.Series, shares: dict[str, float], prices: pd.Series,
                 equity: float, cash: float, cfg: ExecutionConfig,
                 reasons: dict[str, str] | None = None) -> list[Order]:
    """Turn target weights into share orders. Sells come first so they fund the buys;
    buys are scaled down if they would eat into the cash buffer."""
    reasons = reasons or {}
    names = set(shares) | set(target[target > 0].index)
    sells, buys = [], []
    for t in sorted(names):
        want, have = float(target.get(t, 0.0)), float(current.get(t, 0.0))
        if abs(want - have) < 1e-9:
            continue                                   # a HOLD never trades
        price = float(prices.get(t, np.nan))
        if not np.isfinite(price) or price <= 0:
            log.warning("no usable price for %s; leaving it untouched", t)
            continue
        cur = float(shares.get(t, 0.0))
        if want <= 0:
            if cur > 0:
                sells.append(Order(t, "sell", cur, price, reasons.get(t, ""), close_position=True))
            continue
        delta = want * equity / price - cur
        if abs(delta) * price < cfg.min_order_value:
            continue
        order = Order(t, "buy" if delta > 0 else "sell", abs(delta), price, reasons.get(t, ""))
        (buys if delta > 0 else sells).append(order)

    cost = cfg.trade_cost_bps / 1e4
    spendable = cash + sum(o.value for o in sells) * (1 - cost) - equity * cfg.cash_buffer
    wanted = sum(o.value for o in buys) * (1 + cost)
    scale = min(1.0, max(spendable, 0.0) / wanted) if wanted > 0 else 1.0
    rounded = []
    for o in sorted(buys, key=lambda o: -o.value):
        qty = o.shares * scale
        qty = math.floor(qty * 1e6) / 1e6 if cfg.fractional_shares else math.floor(qty)
        if qty * o.price >= cfg.min_order_value:
            rounded.append(Order(o.ticker, "buy", qty, o.price, o.reason))
    if not cfg.fractional_shares:
        for o in sells:
            o.shares = min(float(math.ceil(o.shares - 1e-9)), float(shares.get(o.ticker, 0.0)))
        sells = [o for o in sells if o.shares > 0]
    return sells + rounded


class PaperBroker:
    """Simulated fills at the reference price plus/minus the configured trading cost."""

    name = "paper"

    def __init__(self, state: PortfolioState, cost_bps: float):
        self.state = state
        self.cost = cost_bps / 1e4

    def snapshot(self) -> tuple[float, dict[str, float]]:
        return self.state.cash, self.state.shares()

    def execute(self, orders: list[Order], when: pd.Timestamp) -> list[Fill]:
        fills = []
        for o in [o for o in orders if o.side == "sell"] + [o for o in orders if o.side == "buy"]:
            h = self.state.holdings.get(o.ticker)
            if o.side == "sell":
                qty = min(o.shares, h.shares if h else 0.0)
                if qty <= 0:
                    continue
                px = o.price * (1 - self.cost)
                self.state.cash += qty * px
                h.shares -= qty
                if o.close_position or h.shares < 1e-6:
                    del self.state.holdings[o.ticker]
            else:
                px = o.price * (1 + self.cost)
                qty = min(o.shares, math.floor(self.state.cash / px * 1e6) / 1e6)
                if qty <= 0:
                    log.warning("insufficient cash to buy %s", o.ticker)
                    continue
                self.state.cash -= qty * px
                if h is None:
                    self.state.holdings[o.ticker] = Holding(qty, when.date().isoformat(), px, o.price)
                else:
                    h.entry_price = (h.entry_price * h.shares + px * qty) / (h.shares + qty)
                    h.shares += qty
            fills.append(Fill(o.ticker, o.side, qty, px))
        return fills


class AlpacaBroker:
    """Alpaca Trading API v2. Defaults to the paper-trading endpoint; a live account
    is only used when explicitly configured with account.alpaca_paper = false."""

    name = "alpaca"
    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL = "https://api.alpaca.markets"
    DONE = {"filled", "canceled", "expired", "rejected", "done_for_day"}

    def __init__(self, key_id: str, secret: str, paper: bool = True, session=None,
                 fill_timeout: float = 90.0, poll_interval: float = 1.0):
        import requests

        self.base = self.PAPER_URL if paper else self.LIVE_URL
        self.session = session or requests.Session()
        self.session.headers.update({"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret})
        self.fill_timeout = fill_timeout
        self.poll_interval = poll_interval

    @classmethod
    def from_env(cls, paper: bool = True) -> AlpacaBroker:
        key, secret = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError("set APCA_API_KEY_ID and APCA_API_SECRET_KEY to use the Alpaca broker")
        return cls(key, secret, paper=paper)

    def _request(self, method: str, path: str, **kwargs):
        resp = self.session.request(method, self.base + path, timeout=30, **kwargs)
        if resp.status_code >= 400:
            raise RuntimeError(f"Alpaca {method} {path} failed: {resp.status_code} {resp.text}")
        return resp.json() if resp.content else {}

    def is_market_open(self) -> bool:
        return bool(self._request("GET", "/v2/clock").get("is_open"))

    def snapshot(self) -> tuple[float, dict[str, float]]:
        account = self._request("GET", "/v2/account")
        positions = self._request("GET", "/v2/positions")
        return float(account["cash"]), {p["symbol"]: float(p["qty"]) for p in positions}

    def execute(self, orders: list[Order], when: pd.Timestamp) -> list[Fill]:
        sells = [self._submit(o) for o in orders if o.side == "sell"]
        fills = self._await(sells)          # free up buying power before buying
        buys = [self._submit(o) for o in orders if o.side == "buy"]
        return fills + self._await(buys)

    def _submit(self, o: Order) -> dict:
        if o.close_position:
            return self._request("DELETE", f"/v2/positions/{o.ticker}")
        qty = f"{o.shares:.6f}".rstrip("0").rstrip(".")
        body = {"symbol": o.ticker, "qty": qty, "side": o.side, "type": "market", "time_in_force": "day"}
        return self._request("POST", "/v2/orders", json=body)

    def _await(self, submitted: list[dict]) -> list[Fill]:
        pending = {o["id"]: o for o in submitted if o.get("id")}
        deadline = time.monotonic() + self.fill_timeout
        done: dict[str, dict] = {}
        while pending and time.monotonic() < deadline:
            for oid in list(pending):
                order = self._request("GET", f"/v2/orders/{oid}")
                if order.get("status") in self.DONE:
                    done[oid] = order
                    del pending[oid]
            if pending:
                time.sleep(self.poll_interval)
        for oid in pending:
            log.warning("order %s still open after %.0fs", oid, self.fill_timeout)
        return [
            Fill(o["symbol"], o["side"], float(o["filled_qty"]), float(o["filled_avg_price"]))
            for o in done.values() if float(o.get("filled_qty") or 0) > 0
        ]
