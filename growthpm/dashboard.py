"""Phone-first HTML dashboard of the portfolio, regenerated every cycle.

`render()` returns the page as three parts (title, style, body) so the same
content can be published as a fragment or wrapped into a standalone document
(`document()`) for GitHub Pages. Everything is inline: no external requests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from html import escape

import numpy as np
import pandas as pd

from .broker import Fill, Order
from .config import Config
from .portfolio import PortfolioState
from .strategy import Decision

TITLE = "growthpm Portfolio"


@dataclass
class Page:
    title: str
    style: str
    body: str

    def fragment(self) -> str:
        return f"<title>{escape(self.title)}</title>\n<style>{self.style}</style>\n{self.body}"


def document(page: Page) -> str:
    """A standalone, installable (Add to Home Screen) HTML document."""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#f4f4f0" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#0d0d0d" media="(prefers-color-scheme: dark)">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="growthpm">
<meta name="robots" content="noindex">
<title>{escape(page.title)}</title>
<style>:root{{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}}
body{{margin:0}}[hidden]{{display:none!important}}{page.style}</style>
</head>
<body>
{page.body}
</body>
</html>
"""


# ---------------------------------------------------------------- formatting

def _money(x: float, cents: bool = False) -> str:
    return f"${x:,.2f}" if cents else f"${x:,.0f}"


def _pct(x: float, signed: bool = True, digits: int = 1) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    s = f"{abs(x):.{digits}%}"
    if not signed:
        return f"{x:.{digits}%}"
    return ("+" if x > 0 else "−" if x < 0 else "") + s


def _delta(x: float, digits: int = 1) -> str:
    """Signed change with a direction glyph, so it never relies on color alone."""
    if x is None or not np.isfinite(x):
        return '<span class="flat">n/a</span>'
    cls, glyph = ("up", "▲") if x > 0 else ("down", "▼") if x < 0 else ("flat", "◆")
    return f'<span class="{cls}"><span aria-hidden="true">{glyph}</span> {_pct(x, digits=digits)}</span>'


def _day(d: pd.Timestamp) -> str:
    return f"{d:%a %b} {d.day}"


REGIME = {"risk_on": ("good", "Risk-on"), "neutral": ("warn", "Neutral"), "risk_off": ("crit", "Risk-off")}
KIND = {"BUY": "buy", "ADD": "buy", "SELL": "sell", "TRIM": "sell"}


# ---------------------------------------------------------------- data

def _equity_series(state: PortfolioState, prices: pd.DataFrame, bench: str, base: float) -> list[dict]:
    """One point per price date (the last run that day), with the benchmark on the same dates."""
    by_date: dict[str, float] = {}
    for h in state.history:
        by_date[h["price_date"]] = float(h["equity"])
    points = []
    first_bench = None
    for d, eq in sorted(by_date.items()):
        ts = pd.Timestamp(d)
        if ts not in prices.index or bench not in prices.columns:
            continue
        b = float(prices.at[ts, bench])
        first_bench = first_bench or b
        points.append({"d": d, "label": f"{ts:%b} {ts.day}", "eq": round(eq, 2),
                       "p": eq / base - 1, "b": b / first_bench - 1})
    return points


def render(state: PortfolioState, decision: Decision, orders: list[Order], fills: list[Fill],
           prices: pd.DataFrame, cfg: Config, *, equity: float, cash: float, executed: bool,
           blocked: str | None, broker: str, synthetic: bool, run_time: pd.Timestamp) -> Page:
    bench = cfg.universe.benchmark
    last, prev = prices.iloc[-1], prices.iloc[-2] if len(prices) > 1 else prices.iloc[-1]
    base = cfg.account.starting_cash if broker == "paper" else (state.history[0]["equity"] if state.history else equity)
    series = _equity_series(state, prices, bench, base)
    since = pd.Timestamp(series[0]["d"]) if series else decision.date
    total = equity / base - 1 if base else float("nan")
    bench_ret = series[-1]["b"] if series else float("nan")

    holdings = []
    day_pnl = 0.0
    for t, h in state.holdings.items():
        px = float(last.get(t, np.nan))
        if not np.isfinite(px):
            continue
        value = h.shares * px
        p0 = float(prev.get(t, np.nan))
        if np.isfinite(p0) and pd.Timestamp(h.entry_date) < decision.date:
            day_pnl += h.shares * (px - p0)
        stop = h.peak_price * (1 - cfg.strategy.trailing_stop)
        exp = decision.table["exp_return"].get(t, np.nan) if "exp_return" in decision.table else np.nan
        holdings.append({"t": t, "value": value, "w": value / equity if equity else 0.0, "px": px,
                         "pnl": px / h.entry_price - 1 if h.entry_price else np.nan, "since": h.entry_date,
                         "stop": stop, "drop": 1 - stop / px, "exp": exp})
    holdings.sort(key=lambda r: -r["value"])
    invested = sum(r["value"] for r in holdings)
    day_ret = day_pnl / (equity - day_pnl) if equity - day_pnl else float("nan")

    tone, regime_label = REGIME.get(decision.regime.name, ("warn", decision.regime.name))
    body = []
    body.append('<main class="wrap">')
    body.append('<header class="top">'
                '<div class="brand">growthpm <span class="sub">paper portfolio</span></div>'
                f'<span class="chip {tone}" title="Market regime"><span class="dot" aria-hidden="true"></span>'
                f'{regime_label}</span></header>')
    if synthetic:
        body.append('<p class="banner">Synthetic demo data. These are not real market prices.</p>')

    # Hero: the one number this page leads with.
    rel = total - bench_ret if np.isfinite(total) and np.isfinite(bench_ret) else float("nan")
    body.append('<section class="hero" aria-label="Portfolio value">'
                f'<div class="hero-value">{_money(equity, cents=True)}</div>'
                f'<div class="hero-line">{_delta(total, 2)} <span class="muted">since {_day(since)} start</span></div>'
                f'<div class="hero-line small muted">{escape(bench)} {_pct(bench_ret, digits=2)} over the same days'
                + (f' · {"+" if rel >= 0 else chr(0x2212)}{abs(rel) * 100:.2f} pts vs {escape(bench)}'
                   if np.isfinite(rel) else "")
                + '</div></section>')

    stats = [
        (f"Last session · {decision.date:%b} {decision.date.day}", _delta(day_ret, 2),
         f'{"+" if day_pnl >= 0 else chr(0x2212)}{_money(abs(day_pnl))}'),
        ("Invested", _pct(invested / equity if equity else 0, signed=False),
         f"{len(holdings)} of {cfg.strategy.max_positions} positions"),
        ("Cash", _money(cash), "available"),
        ("Model growth rate", _pct(decision.growth_target, signed=False) + "/yr", "model estimate"),
    ]
    body.append('<section class="stats" aria-label="Key figures">' + "".join(
        f'<div class="stat"><div class="label">{escape(k)}</div><div class="value">{v}</div>'
        f'<div class="note">{escape(n)}</div></div>' for k, v, n in stats) + "</section>")

    # Chart (drawn client-side at the real width) with a table fallback.
    body.append('<section class="card" aria-labelledby="perf-h"><div class="card-head">'
                f'<h2 id="perf-h">Return since start</h2><div class="legend">'
                '<span><i class="key key-p"></i>Portfolio</span>'
                f'<span><i class="key key-b"></i>{escape(bench)}</span></div></div>'
                '<div id="chart" class="chart"><div id="tip" class="tip" hidden></div></div>'
                '<details class="table-view"><summary>Show as table</summary><div class="scroll"><table>'
                f'<thead><tr><th>Date</th><th>Value</th><th>Portfolio</th><th>{escape(bench)}</th></tr></thead><tbody>'
                + "".join(f'<tr><td>{escape(p["label"])}</td><td>{_money(p["eq"])}</td>'
                          f'<td>{_pct(p["p"], digits=2)}</td><td>{_pct(p["b"], digits=2)}</td></tr>'
                          for p in reversed(series))
                + "</tbody></table></div></details></section>")

    # What the manager did this cycle, and why.
    trades = decision.trades
    if executed:
        status = f"{len(fills)} order{'s' if len(fills) != 1 else ''} filled" if fills else "No trades needed"
    elif blocked:
        status = f"Held back: {blocked}"
    else:
        status = "Dry run: no orders sent"
    body.append('<section aria-labelledby="cycle-h"><div class="sec-head">'
                f'<h2 id="cycle-h">Latest cycle</h2><span class="muted small">{run_time:%b} {run_time.day}, '
                f'{run_time:%H:%M} ET · prices {_day(decision.date)}</span></div>'
                f'<p class="status">{escape(status)}</p>')
    if trades:
        body.append('<ul class="rows">' + "".join(
            f'<li class="row action"><span class="tag {KIND.get(a.kind, "")}">{a.kind}</span>'
            f'<span class="tk">{escape(a.ticker)}</span>'
            f'<span class="num muted">{a.from_weight:.1%} → {a.to_weight:.1%}</span>'
            f'<p class="why">{escape(a.reason)}</p></li>' for a in trades) + "</ul>")
    else:
        body.append(f'<p class="muted">All {len(holdings)} positions held. Nothing beat them by enough to pay for a switch.</p>')
    body.append("</section>")

    # Holdings.
    cap = cfg.strategy.max_weight + cfg.strategy.weight_drift
    rows = []
    for r in holdings:
        exp = f'expected {_pct(r["exp"], signed=False, digits=0)}/yr · ' if np.isfinite(r["exp"]) else ""
        rows.append(
            f'<li class="row holding"><div class="line1"><span class="tk">{escape(r["t"])}</span>'
            f'<span class="num">{_money(r["value"])}</span>{_delta(r["pnl"])}</div>'
            f'<div class="meter" role="img" aria-label="{r["w"]:.1%} of the portfolio">'
            f'<span style="width:{min(r["w"] / cap, 1) * 100:.1f}%"></span></div>'
            f'<div class="line2 muted small"><span>{r["w"]:.1%} weight · {exp}since {escape(r["since"])}</span>'
            f'<span>stop ${r["stop"]:,.2f} · {_pct(r["drop"], signed=False)} drop away</span></div></li>')
    body.append('<section aria-labelledby="hold-h"><div class="sec-head"><h2 id="hold-h">Holdings</h2>'
                '<span class="muted small">P&amp;L since entry</span></div>'
                + (f'<ul class="rows">{"".join(rows)}</ul>' if rows else '<p class="muted">No positions yet.</p>')
                + "</section>")

    # Watchlist: best ideas not held.
    tbl = decision.table
    ideas = tbl[(tbl["target"] <= 0) & tbl["entry_ok"]].head(6) if len(tbl) else tbl
    if len(ideas):
        body.append('<section aria-labelledby="watch-h"><div class="sec-head"><h2 id="watch-h">Next in line</h2>'
                    '<span class="muted small">would need to beat a holding by the switch hurdle</span></div>'
                    '<ul class="rows compact">' + "".join(
                        f'<li class="row idea"><span class="tk">{escape(str(t))}</span>'
                        f'<span class="num">{_pct(r["exp_return"], signed=False, digits=0)}/yr</span>'
                        f'<span class="num muted">z {r["momentum_z"]:+.2f}</span>'
                        f'<span class="num muted">${r["price"]:,.2f}</span></li>' for t, r in ideas.iterrows())
                    + "</ul></section>")

    # Recent trades.
    recent = list(reversed(state.trades[-8:]))
    if recent:
        body.append('<section aria-labelledby="trades-h"><div class="sec-head"><h2 id="trades-h">Recent trades</h2></div>'
                    '<ul class="rows">' + "".join(
                        f'<li class="row action"><span class="tag {"buy" if t["side"] == "buy" else "sell"}">'
                        f'{escape(t["side"].upper())}</span><span class="tk">{escape(t["ticker"])}</span>'
                        f'<span class="num muted">{float(t["shares"]):,.2f} @ ${float(t["price"]):,.2f}</span>'
                        f'<p class="why"><span class="muted">{escape(t["time"][:10])}</span> · {escape(t["reason"])}</p></li>'
                        for t in recent) + "</ul></section>")

    body.append('<footer class="foot muted small"><p>Paper trading, not financial advice. Expected returns are '
                'model estimates from momentum and market regime, not forecasts.</p>'
                f'<p>Updated {run_time:%Y-%m-%d %H:%M} ET by growthpm · {escape(broker)} broker</p></footer>')
    body.append("</main>")

    data = {"series": series, "bench": bench}
    payload = json.dumps(data).replace("</", "<\\/")
    body.append(f'<script type="application/json" id="gp-data">{payload}</script>')
    body.append(f"<script>{SCRIPT}</script>")
    return Page(TITLE, STYLE, "\n".join(body))


STYLE = """
:root{
  --page:#f4f4f0;--surface:#fcfcfb;--ink:#0b0b0b;--ink-2:#52514e;--muted:#6b6963;
  --grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);
  --accent:#2a78d6;--track:#dbe8f8;--bench:#8d8b84;
  --up:#006300;--down:#c23434;--good:#0ca30c;--warn:#d99400;--crit:#d03b3b;
  --tag-buy:#e3eefb;--tag-sell:#ecebe6;
  --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#ffffff;--ink-2:#c3c2b7;--muted:#9c9a92;
  --grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
  --accent:#3987e5;--track:#1d3350;--bench:#8d8b84;
  --up:#2dbb2d;--down:#e66767;--good:#0ca30c;--warn:#fab219;--crit:#e05252;
  --tag-buy:#1b2f48;--tag-sell:#2a2a28;}}
:root[data-theme="dark"]{
  color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#ffffff;--ink-2:#c3c2b7;--muted:#9c9a92;
  --grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);
  --accent:#3987e5;--track:#1d3350;--bench:#8d8b84;
  --up:#2dbb2d;--down:#e66767;--good:#0ca30c;--warn:#fab219;--crit:#e05252;
  --tag-buy:#1b2f48;--tag-sell:#2a2a28;}
body{background:var(--page);color:var(--ink);font:15px/1.45 var(--sans);-webkit-text-size-adjust:100%}
.wrap{max-width:680px;margin:0 auto;padding-inline:16px;padding-block:12px 32px;display:flex;flex-direction:column;gap:22px}
h2{font-size:17px;margin:0;font-weight:650;text-wrap:balance}
p{margin:0}
.muted{color:var(--muted)}.small{font-size:13px}
.num{font-variant-numeric:tabular-nums}
.tk{font-family:var(--mono);font-weight:600;letter-spacing:.02em}
.top{display:flex;align-items:center;justify-content:space-between;gap:12px;padding-top:4px}
.brand{font-family:var(--mono);font-weight:700;font-size:15px}
.brand .sub{font-family:var(--sans);font-weight:400;color:var(--muted);font-size:13px;margin-left:4px}
.chip{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--border);border-radius:999px;
  padding:4px 10px;font-size:13px;font-weight:600;background:var(--surface)}
.chip .dot{width:8px;height:8px;border-radius:50%;background:var(--warn)}
.chip.good .dot{background:var(--good)}.chip.crit .dot{background:var(--crit)}
.banner{background:var(--tag-sell);border-radius:8px;padding:8px 12px;font-size:13px;font-weight:600}
.hero{display:flex;flex-direction:column;gap:4px}
.hero-value{font-size:44px;font-weight:650;letter-spacing:-.02em;line-height:1.1}
.hero-line{font-size:16px}
.up{color:var(--up);font-weight:600}.down{color:var(--down);font-weight:600}.flat{color:var(--muted);font-weight:600}
.stats{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1px;background:var(--border);
  border:1px solid var(--border);border-radius:12px;overflow:hidden}
@media (min-width:560px){.stats{grid-template-columns:repeat(4,minmax(0,1fr))}}
.stat{background:var(--surface);padding:12px 14px;display:flex;flex-direction:column;gap:2px;min-width:0}
.stat .label{font-size:12px;color:var(--muted);letter-spacing:.01em}
.stat .value{font-size:19px;font-weight:650}
.stat .note{font-size:12px;color:var(--muted)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:14px 14px 10px;
  display:flex;flex-direction:column;gap:10px}
.card-head,.sec-head{display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap}
.legend{display:flex;gap:14px;font-size:13px;color:var(--ink-2)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.key{display:inline-block;width:14px;height:2px;border-radius:2px}
.key-p{background:var(--accent)}.key-b{background:var(--bench)}
.chart{position:relative;min-height:200px;touch-action:pan-y}
.chart svg{display:block;width:100%;height:200px;overflow:visible}
.chart .empty{padding:60px 0;text-align:center;color:var(--muted);font-size:14px}
.chart svg:focus{outline:none}.chart svg:focus-visible{outline:2px solid var(--accent);outline-offset:4px;border-radius:4px}
.tip{position:absolute;top:0;pointer-events:none;background:var(--surface);border:1px solid var(--border);
  border-radius:8px;padding:8px 10px;font-size:13px;box-shadow:0 4px 14px rgba(0,0,0,.12);min-width:130px}
.tip .d{color:var(--muted);font-size:12px;margin-bottom:4px}
.tip .r{display:flex;align-items:center;gap:8px}.tip .r b{font-variant-numeric:tabular-nums}
.tip .r span:last-child{color:var(--muted)}
.table-view summary{cursor:pointer;color:var(--ink-2);font-size:13px;padding:4px 0}
.table-view summary:focus-visible{outline:2px solid var(--accent);border-radius:4px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:6px 8px;border-bottom:1px solid var(--grid);white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:500}
section{display:flex;flex-direction:column;gap:8px}
.status{font-weight:600}
.rows{list-style:none;margin:0;padding:0;border-top:1px solid var(--grid)}
.row{border-bottom:1px solid var(--grid);padding:10px 0}
.row.action{display:grid;grid-template-columns:auto auto 1fr;align-items:center;column-gap:10px;row-gap:4px}
.row.action .num{text-align:right;font-size:13px}
.row.action .why{grid-column:1/-1;font-size:13px;color:var(--ink-2);overflow-wrap:anywhere}
.tag{font-size:11px;font-weight:700;letter-spacing:.06em;padding:2px 7px;border-radius:5px;background:var(--tag-sell);color:var(--ink-2)}
.tag.buy{background:var(--tag-buy);color:var(--accent)}
.row.holding{display:flex;flex-direction:column;gap:6px}
.line1{display:grid;grid-template-columns:1fr auto 5.5em;align-items:baseline;gap:10px}
.line1 .num{font-weight:600}.line1>span:last-child{text-align:right}
.line2{display:flex;justify-content:space-between;gap:4px 12px;flex-wrap:wrap}
.meter{height:6px;border-radius:3px;background:var(--track);overflow:hidden}
.meter span{display:block;height:100%;background:var(--accent);border-radius:3px}
.rows.compact .row.idea{display:grid;grid-template-columns:1fr 5em 5.5em 6.5em;gap:8px;align-items:baseline}
.rows.compact .row.idea .num{text-align:right;font-size:14px}
.foot{display:flex;flex-direction:column;gap:4px;padding-top:4px}
@media (prefers-reduced-motion:no-preference){.meter span{transition:width .4s ease}}
"""

SCRIPT = r"""
(function(){
  var data = JSON.parse(document.getElementById('gp-data').textContent);
  var host = document.getElementById('chart'), tip = document.getElementById('tip');
  var NS = 'http://www.w3.org/2000/svg', pts = data.series, idx = pts.length - 1, svg = null, geo = null;
  function fmt(v){ var s = Math.abs(v*100).toFixed(2) + '%'; return v > 0 ? '+' + s : v < 0 ? '−' + s : s; }
  function el(n, a){ var e = document.createElementNS(NS, n); for (var k in a) e.setAttribute(k, a[k]); return e; }
  function niceStep(span){ var steps=[.001,.002,.0025,.005,.01,.02,.025,.05,.1,.2,.25,.5,1];
    for (var i=0;i<steps.length;i++) if (span/steps[i] <= 5) return steps[i]; return 1; }
  function draw(){
    if (svg) svg.remove();
    if (pts.length < 2){ if (!host.querySelector('.empty')){ var p=document.createElement('p'); p.className='empty';
      p.textContent='The chart fills in after the second daily cycle.'; host.appendChild(p);} return; }
    var W = host.clientWidth, H = 200, L = 46, R = 58, T = 10, B = 24;
    var vals = [0]; pts.forEach(function(p){ vals.push(p.p, p.b); });
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    var step = niceStep((hi - lo) || .01); lo = Math.floor(lo/step)*step; hi = Math.ceil(hi/step)*step;
    if (hi === lo) hi = lo + step;
    var x = function(i){ return L + (W - L - R) * i / (pts.length - 1); };
    var y = function(v){ return T + (H - T - B) * (hi - v) / (hi - lo); };
    svg = el('svg', {viewBox:'0 0 '+W+' '+H, role:'img', tabindex:'0',
      'aria-label':'Return since start: portfolio '+fmt(pts[pts.length-1].p)+', '+data.bench+' '+fmt(pts[pts.length-1].b)});
    for (var v = lo; v <= hi + 1e-9; v += step){
      var yy = y(v), zero = Math.abs(v) < 1e-9;
      svg.appendChild(el('line', {x1:L, x2:W-R, y1:yy, y2:yy, stroke: zero ? 'var(--axis)' : 'var(--grid)', 'stroke-width':1}));
      var t = el('text', {x:L-8, y:yy+4, 'text-anchor':'end', 'font-size':11, fill:'var(--muted)'});
      t.textContent = (v === 0 ? '0' : (Math.abs(v*100) < 1 ? fmt(v) : (v>0?'+':'−')+Math.abs(v*100).toFixed(step<.01?1:0)+'%'));
      svg.appendChild(t);
    }
    [0, pts.length-1].concat(pts.length > 4 ? [Math.round((pts.length-1)/2)] : []).forEach(function(i){
      var t = el('text', {x:x(i), y:H-6, 'text-anchor': i===0 ? 'start' : i===pts.length-1 ? 'end' : 'middle', 'font-size':11, fill:'var(--muted)'});
      t.textContent = pts[i].label; svg.appendChild(t); });
    function line(key, color){ var d = pts.map(function(p,i){ return (i?'L':'M') + x(i).toFixed(1) + ' ' + y(p[key]).toFixed(1); }).join(' ');
      svg.appendChild(el('path', {d:d, fill:'none', stroke:color, 'stroke-width':2, 'stroke-linejoin':'round', 'stroke-linecap':'round'})); }
    var area = 'M' + x(0) + ' ' + y(0) + ' ' + pts.map(function(p,i){ return 'L' + x(i).toFixed(1) + ' ' + y(p.p).toFixed(1); }).join(' ') + ' L' + x(pts.length-1) + ' ' + y(0) + ' Z';
    svg.appendChild(el('path', {d:area, fill:'var(--accent)', 'fill-opacity':.10, stroke:'none'}));
    line('b', 'var(--bench)'); line('p', 'var(--accent)');
    var n = pts.length - 1, yp = y(pts[n].p), yb = y(pts[n].b);
    [['b','var(--bench)',yb],['p','var(--accent)',yp]].forEach(function(s){
      svg.appendChild(el('circle', {cx:x(n), cy:s[2], r:4, fill:s[1], stroke:'var(--surface)', 'stroke-width':2})); });
    function endLabel(v, yy){ var t = el('text', {x:x(n)+10, y:yy+4, 'font-size':12, 'font-weight':600, fill:'var(--ink)'}); t.textContent = fmt(v); svg.appendChild(t); }
    endLabel(pts[n].p, yp); if (Math.abs(yp - yb) >= 16) endLabel(pts[n].b, yb);
    var cross = el('line', {y1:T, y2:H-B, stroke:'var(--axis)', 'stroke-width':1, visibility:'hidden'});
    svg.appendChild(cross);
    geo = {x:x, y:y, cross:cross, W:W};
    host.insertBefore(svg, tip);
    function show(i){
      idx = Math.max(0, Math.min(pts.length-1, i)); var p = pts[idx], cx = x(idx);
      cross.setAttribute('x1', cx); cross.setAttribute('x2', cx); cross.setAttribute('visibility', 'visible');
      tip.textContent = '';
      var d = document.createElement('div'); d.className = 'd'; d.textContent = p.label + ' · $' + Math.round(p.eq).toLocaleString('en-US'); tip.appendChild(d);
      [['p','Portfolio','var(--accent)'],['b',data.bench,'var(--bench)']].forEach(function(s){
        var r = document.createElement('div'); r.className = 'r';
        var k = document.createElement('i'); k.className = 'key'; k.style.background = s[2];
        var b = document.createElement('b'); b.textContent = fmt(p[s[0]]);
        var nm = document.createElement('span'); nm.textContent = s[1];
        r.appendChild(k); r.appendChild(b); r.appendChild(nm); tip.appendChild(r); });
      tip.hidden = false;
      var tw = tip.offsetWidth, left = cx + 12; if (left + tw > W) left = cx - tw - 12;
      tip.style.left = Math.max(0, left) + 'px';
    }
    function hide(){ cross.setAttribute('visibility', 'hidden'); tip.hidden = true; }
    function nearest(ev){ var r = svg.getBoundingClientRect(); var px = ev.clientX - r.left;
      return Math.round((px - L) / ((W - L - R) / (pts.length - 1))); }
    svg.addEventListener('pointermove', function(ev){ show(nearest(ev)); });
    svg.addEventListener('pointerdown', function(ev){ show(nearest(ev)); });
    svg.addEventListener('pointerleave', hide);
    svg.addEventListener('focus', function(){ show(idx); });
    svg.addEventListener('blur', hide);
    svg.addEventListener('keydown', function(ev){
      if (ev.key === 'ArrowLeft'){ show(idx-1); ev.preventDefault(); }
      else if (ev.key === 'ArrowRight'){ show(idx+1); ev.preventDefault(); }
      else if (ev.key === 'Escape'){ hide(); } });
  }
  draw();
  var lastW = host.clientWidth;
  if (window.ResizeObserver) new ResizeObserver(function(){ if (host.clientWidth !== lastW){ lastW = host.clientWidth; tip.hidden = true; draw(); } }).observe(host);
})();
"""
