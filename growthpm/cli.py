"""Command line interface: growthpm {init,run,watch,status,backtest}."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from .config import EXAMPLE_TOML, Config
from .data import make_provider
from .engine import make_broker, run_once, us_market_open
from .portfolio import PortfolioState
from .report import table
from .universe import resolve

log = logging.getLogger("growthpm")


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", default="config.toml", help="TOML config (defaults used if missing)")
    p.add_argument("--provider", default="yahoo",
                   help="market data: yahoo (live), synthetic[:seed] (offline demo), csv:<path>")
    p.add_argument("--state", help="portfolio state file (overrides config)")
    p.add_argument("-v", "--verbose", action="store_true")


def cmd_init(args) -> int:
    path = Path(args.config)
    if path.exists():
        print(f"{path} already exists; not overwriting")
        return 1
    path.write_text(EXAMPLE_TOML)
    print(f"wrote {path}. Next: growthpm run   (dry run), then growthpm run --execute")
    return 0


def cmd_run(args) -> int:
    cfg = Config.load(args.config)
    result = run_once(cfg, make_provider(args.provider), execute=args.execute, state_path=args.state)
    print(result.report)
    if result.report_path:
        print(f"report saved to {result.report_path}")
    return 0


def cmd_watch(args) -> int:
    cfg = Config.load(args.config)
    provider = make_provider(args.provider)
    interval = max(1.0, args.interval) * 60
    print(f"watching: a cycle every {args.interval:g} min; Ctrl-C to stop")
    while True:
        try:
            if args.always or _market_open(cfg, args):
                result = run_once(cfg, provider, execute=args.execute, state_path=args.state)
                trades = result.decision.trades
                stamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                print(f"[{stamp}] equity ${result.equity:,.2f} | {result.decision.regime.name} | "
                      f"{len(result.orders)} order(s)" + (f" -> {result.report_path}" if result.report_path else ""))
                for a in trades:
                    print(f"    {a.kind:<4} {a.ticker:<6} {a.from_weight:6.1%} -> {a.to_weight:6.1%}  {a.reason}")
            else:
                log.info("market closed; sleeping")
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # keep the loop alive through transient data/broker errors
            log.exception("cycle failed: %s", exc)
        time.sleep(interval)


def _market_open(cfg: Config, args) -> bool:
    if cfg.account.broker == "alpaca":
        try:
            state = PortfolioState.load(args.state or cfg.account.state_file) or PortfolioState.new(0)
            return make_broker(cfg, state).is_market_open()
        except Exception as exc:
            log.warning("Alpaca clock unavailable (%s); using local session hours", exc)
    return us_market_open()


def cmd_status(args) -> int:
    cfg = Config.load(args.config)
    path = args.state or cfg.account.state_file
    state = PortfolioState.load(path)
    if state is None:
        print(f"no portfolio at {path} yet; run `growthpm run` first")
        return 1
    rows = [[t, f"{h.shares:,.4f}", h.entry_date, f"${h.entry_price:,.2f}", f"${h.peak_price:,.2f}",
             f"${h.peak_price * (1 - cfg.strategy.trailing_stop):,.2f}"] for t, h in sorted(state.holdings.items())]
    print(f"cash ${state.cash:,.2f} | created {state.created}")
    print(table(["Ticker", "Shares", "Since", "Avg cost", "Peak", "Stop"], rows))
    if state.history:
        first, last = state.history[0], state.history[-1]
        change = last["equity"] / first["equity"] - 1 if first["equity"] else 0.0
        print(f"\nequity ${last['equity']:,.2f} at {last['time']} ({change:+.2%} since {first['time']}), "
              f"regime {last['regime']}")
    if state.trades:
        print("\nrecent trades:")
        print(table(["Time", "Side", "Ticker", "Shares", "Price", "Reason"],
                    [[t["time"], t["side"].upper(), t["ticker"], t["shares"], t["price"], t["reason"]]
                     for t in state.trades[-10:]]))
    return 0


def cmd_backtest(args) -> int:
    from .backtest import run_backtest
    from .report import render_backtest

    cfg = Config.load(args.config)
    provider = make_provider(args.provider)
    universe = resolve(cfg.universe)
    days = int((args.years + 2) * 365)
    prices = provider.history(sorted(set(universe) | {cfg.universe.benchmark}), days)
    vix = None
    if cfg.universe.vix:
        try:
            vix = provider.history([cfg.universe.vix], days).iloc[:, 0]
        except Exception as exc:
            log.warning("no %s history (%s); regime ignores volatility", cfg.universe.vix, exc)
    start = args.start or (prices.index[-1] - pd.DateOffset(years=args.years)).strftime("%Y-%m-%d")
    result = run_backtest(prices, cfg, universe, vix=vix, start=start, end=args.end,
                          rebalance_days=args.rebalance_days, synthetic=provider.is_synthetic)
    print(render_backtest(result.metrics, result.info))
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        pd.concat([result.equity, result.benchmark], axis=1).to_csv(out / "equity.csv")
        result.trades.to_csv(out / "trades.csv", index=False)
        result.weights.to_csv(out / "weights.csv")
        print(f"equity curve, trades and weights written to {out}/")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="growthpm", description="Active growth portfolio manager")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="write a starter config.toml")
    p.add_argument("-c", "--config", default="config.toml")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("run", help="one cycle: fetch prices, decide, show (or execute) trades")
    _common(p)
    p.add_argument("--execute", action="store_true", help="send the orders (paper ledger or Alpaca)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("watch", help="run a cycle repeatedly during market hours")
    _common(p)
    p.add_argument("--interval", type=float, default=60, help="minutes between cycles")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--always", action="store_true", help="ignore market hours")
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("status", help="show holdings, stops and recent trades from the ledger")
    _common(p)
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("backtest", help="walk-forward backtest of the strategy")
    _common(p)
    p.add_argument("--years", type=float, default=8)
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--rebalance-days", type=int, default=5)
    p.add_argument("--out", help="directory for equity/trades/weights CSVs")
    p.set_defaults(func=cmd_backtest)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
