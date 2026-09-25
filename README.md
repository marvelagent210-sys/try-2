# growthpm: an active growth portfolio manager

`growthpm` manages a stock/ETF portfolio for **maximum long-run growth**. Every
cycle it pulls current market data, re-scores the whole universe, and moves
capital when (and only when) a better opportunity beats a current holding by
more than the cost of switching. Every trade comes with a plain-English reason.

```
SELL  CEG   displaced (expected 3%/yr): capital rotates into IGV (9%/yr)
BUY   IGV   new opportunity: momentum rank #4 (z +1.4), expected 9%/yr, vol 34%
SELL  XLK   trend break: closed below its 200-day average
SELL  AMD   trailing stop: -26% from post-entry peak 187.40
```

> Not financial advice. Expected returns are model estimates, not forecasts. Start
> on the paper ledger, and read [Limits](#limits-read-this) before you trust it
> with real money.

## How it decides

Each cycle runs this pipeline (`growthpm/strategy.py`):

1. **Live data.** Adjusted daily closes for ~70 liquid growth stocks, sector
   ETFs and defensive assets (bonds, gold) from Yahoo Finance. During market hours
   the last bar is today's live price.
2. **Regime.** Risk-on, neutral or risk-off from the benchmark vs its 200-day
   average, market breadth and the VIX, with hysteresis so the regime doesn't
   whipsaw. The regime sets the equity premium and the exposure cap (100%, 80% or 50%).
3. **Expected return per asset.** Blended, risk-adjusted momentum over 1, 3, 6
   and 12 months, smoothed and z-scored across the universe, then turned into an
   expected excess return: `mu = premium * beta + IC * vol * z * sqrt(12)`.
4. **Filters.** New money only goes into uptrends: price above the 100- and
   200-day averages, positive 6-month return, not in a post-stop cooldown.
5. **Growth-optimal sizing.** A fractional-Kelly optimizer maximizes expected log
   growth, `mu'w - (1/2k) w'Σw`, using a shrunk covariance matrix. It also charges
   a **trade penalty**. Putting idle cash to work pays only the trading cost.
   Selling a holding also pays a *switch hurdle* (default 6%/yr). So a position
   is replaced only when the new idea's edge clears
   cost + cost + hurdle: "update a position if a better opportunity exists",
   without churning on noise (`growthpm/optimizer.py`).
6. **Limits.** At most 10 positions, 3–20% each (winners may drift to 25% before a
   trim). When the position count is capped, the positions that add the most to
   expected growth are kept, not simply the largest ones.
7. **Hard exits.** A 25% trailing stop from the post-entry peak, or a close below
   the 200-day average, sells regardless of the optimizer. After a forced
   exit, re-entry is blocked for 10 trading days.

Why Kelly: maximizing *expected log growth* is what maximizes long-run
compounded wealth. Full Kelly is famously violent, so the default is half-Kelly
(`kelly_fraction = 0.5`). The regime filter and stops exist because deep
drawdowns destroy compounding: a 50% loss needs a 100% gain to recover.

## Quick start

```bash
pip install -e .            # Python 3.10+
growthpm init               # writes config.toml with every knob documented
growthpm run                # dry run: live data -> report of what it WOULD trade
growthpm run --execute      # trade the paper ledger (portfolio_state.json)
growthpm status             # holdings, stop levels, P&L, recent trades
```

Each run prints a Markdown report and saves it under `reports/`. The report shows
the regime, orders with reasons, the target portfolio with stop levels, and the
next-best opportunities not held.

No network? Add `--provider synthetic` to any command to run it on a
deterministic fake market (clearly labeled as such in every report).

## Keeping it actively updated

Pick one:

| Option | How |
|---|---|
| **Watch loop** | `growthpm watch --execute --interval 60` runs a cycle every hour while the NYSE is open. |
| **Cron** | `35 15 * * 1-5 cd /path/to/repo && growthpm run --execute` (machine in New York time; one cycle near the close). |
| **GitHub Actions** | `.github/workflows/manage.yml` runs a cycle three times each weekday session (GitHub often starts scheduled jobs hours late, so several slots keep one inside market hours) and commits `portfolio/state.json` plus a report (`portfolio/LATEST.md`). Run it by hand from the Actions tab. |

A once-a-day cycle near the close is the recommended cadence. The signals are
daily, and intraday runs mainly let the trailing stops react faster.

With real data, `--execute` refuses to trade (and says why in the report) when
the newest price bar is older than the last session that has opened, or when a
paper order would fill while the market is closed. Either would give the ledger
fills at prices nobody could actually trade at.

### Real brokerage (Alpaca)

Set `broker = "alpaca"` under `[account]` and export `APCA_API_KEY_ID` /
`APCA_API_SECRET_KEY`. It trades Alpaca's **paper** endpoint unless you also set
`alpaca_paper = false`. Sells go first and the buys wait (up to 90 s) for them to fill, and
full exits close the whole position. For GitHub Actions, add the keys as
repository secrets.

## Backtesting

```bash
growthpm backtest --years 8 --out results/     # equity.csv, trades.csv, weights.csv
```

The backtest calls the same `decide()` the live manager uses. At each close it
sees only past prices (a test shocks future prices and checks that nothing before
them changes), pays trading costs, checks stops daily and re-optimizes every
`--rebalance-days` (default 5).

## Configuration

Every knob lives in `config.toml` (`growthpm init` writes a commented template).
These matter most:

| Key | Default | Effect |
|---|---|---|
| `universe.preset` | `growth` | `aggressive` adds 3x ETFs (TQQQ, SOXL, UPRO, TECL) and crypto exposure (IBIT, ETHA, MSTR); `etf` is ETF-only |
| `universe.tickers` | `[]` | extra symbols to consider |
| `strategy.kelly_fraction` | `0.5` | higher = more growth *and* much larger drawdowns |
| `strategy.max_positions` / `max_weight` | `10` / `0.20` | concentration |
| `strategy.max_gross` | `1.0` | above 1.0 uses margin |
| `strategy.trailing_stop` | `0.25` | exit after this drop from the post-entry peak |
| `execution.switch_hurdle` | `0.06` | how much better (per year) a new idea must be to replace a holding |
| `execution.trade_cost_bps` | `10` | assumed one-way cost (spread + slippage) |

## Limits (read this)

- **Model, not oracle.** Momentum is one of the best-documented return
  anomalies, but it has crashes (e.g. sharp bear-market rebounds). The expected
  returns in reports are estimates.
- **Survivorship bias.** The default universe is today's successful companies.
  A backtest on it looks better than trading it in real time would have been.
  Treat backtest numbers as an upper bound.
- **Synthetic results prove mechanics only.** The synthetic market is built so
  that momentum works. Its numbers say nothing about real markets.
- **Not modeled:** taxes (short-term gains can be a big drag in a taxable account),
  exchange holidays (the watch loop's local clock doesn't know them; Alpaca's
  clock does), borrowing costs on margin, and dividends beyond Yahoo's
  adjusted closes.
- **Yahoo Finance** is free and unofficial. It can rate-limit or return gaps.
  The provider caches the day's history and only refreshes recent days, and
  held names with no price are left untouched.

## Layout

```
growthpm/
  data.py        Yahoo / CSV / synthetic price providers, calendar alignment
  signals.py     momentum, volatility, trend and beta panels (backward-looking only)
  risk.py        shrunk covariance
  optimizer.py   fractional-Kelly optimizer with asymmetric trade penalty (FISTA)
  strategy.py    regime, filters, hard exits, sizing, trade reasons
  portfolio.py   JSON ledger: cash, holdings, peaks, cooldowns, history, trades
  broker.py      order building, paper broker, Alpaca broker
  engine.py      one live cycle: data -> decision -> orders -> report
  backtest.py    walk-forward backtest on the same decision code
  cli.py         growthpm init | run | watch | status | backtest
tests/           pytest suite (no network needed)
```

Run the tests with `pip install -e ".[dev]" && pytest`.
