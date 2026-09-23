import pandas as pd
import pytest

from growthpm.data import DataError, SyntheticProvider, YahooProvider, align, make_provider


def fake_yahoo_frame(tickers, start, end=None, multi=True):
    idx = pd.bdate_range(start=start, end=end or pd.Timestamp.now().normalize(), name="Date")
    fields = ["Close", "High", "Low", "Open", "Volume"]
    base = 100 + (idx - pd.Timestamp("2020-01-01")).days.to_numpy() * 0.01   # a function of the date only
    data = {(f, t): base * (1 + i) for f in fields for i, t in enumerate(tickers)}
    frame = pd.DataFrame(data, index=idx)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns, names=["Price", "Ticker"])
    if not multi:
        frame.columns = frame.columns.get_level_values(0)
    return frame


@pytest.fixture
def fake_download(monkeypatch):
    calls = []
    import yfinance

    def download(tickers, start=None, **kwargs):
        calls.append((list(tickers), start))
        assert kwargs["auto_adjust"] is True
        return fake_yahoo_frame([t for t in tickers if t != "NOPE"], start)

    monkeypatch.setattr(yfinance, "download", download)
    return calls


def test_yahoo_parses_multiindex_and_caches_daily(tmp_path, fake_download):
    p = YahooProvider(cache_dir=tmp_path)
    first = p.history(["msft", "AAPL", "NOPE"], days=400)
    assert list(first.columns) == ["AAPL", "MSFT"]
    assert first.index.tz is None and first.index.is_monotonic_increasing
    second = p.history(["AAPL", "MSFT", "NOPE"], days=400)
    assert len(fake_download) == 2
    # the second call only refreshes the last few days and merges them in
    first_start, second_start = pd.Timestamp(fake_download[0][1]), pd.Timestamp(fake_download[1][1])
    assert second_start - first_start > pd.Timedelta(days=300)
    pd.testing.assert_frame_equal(first, second)


def test_yahoo_single_ticker_flat_columns(monkeypatch):
    import yfinance

    monkeypatch.setattr(yfinance, "download", lambda tickers, start=None, **kw: fake_yahoo_frame(tickers, start, multi=False))
    out = YahooProvider(cache_dir=None).history(["SPY"], days=100)
    assert list(out.columns) == ["SPY"]


def test_yahoo_empty_response_raises_but_refresh_falls_back_to_cache(tmp_path, monkeypatch):
    import yfinance

    monkeypatch.setattr(yfinance, "download", lambda *a, **k: pd.DataFrame())
    with pytest.raises(DataError):
        YahooProvider(cache_dir=None).history(["SPY"], days=100)

    monkeypatch.setattr(yfinance, "download", lambda tickers, start=None, **kw: fake_yahoo_frame(tickers, start))
    p = YahooProvider(cache_dir=tmp_path)
    fresh = p.history(["SPY"], days=100)
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: pd.DataFrame())
    pd.testing.assert_frame_equal(p.history(["SPY"], days=100), fresh)


def test_csv_provider(tmp_path):
    frame = fake_yahoo_frame(["SPY", "QQQ"], "2025-01-01")["Close"]
    path = tmp_path / "px.csv"
    frame.to_csv(path)
    out = make_provider(f"csv:{path}").history(["spy", "qqq", "zzz"], days=30)
    assert list(out.columns) == ["SPY", "QQQ"]
    assert out.index[-1] - out.index[0] <= pd.Timedelta(days=30)


def test_synthetic_is_deterministic_and_align_drops_off_calendar_rows():
    a = SyntheticProvider(seed=5, end="2026-06-30").history(["SPY", "NVDA"], 200)
    b = SyntheticProvider(seed=5, end="2026-06-30").history(["NVDA", "SPY"], 200)
    pd.testing.assert_frame_equal(a[["SPY", "NVDA"]], b[["SPY", "NVDA"]])
    weekend = a.copy()
    weekend.loc[pd.Timestamp("2026-06-27"), "NVDA"] = 1.0   # a crypto-style Saturday print
    aligned = align(weekend.sort_index().assign(SPY=weekend["SPY"].where(weekend.index.dayofweek < 5)), "SPY")
    assert pd.Timestamp("2026-06-27") not in aligned.index
