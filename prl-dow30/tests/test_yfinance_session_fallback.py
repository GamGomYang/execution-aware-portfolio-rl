import pandas as pd
import pytest

from prl.data_sources import fetch_yfinance


def test_yfinance_session_fallback_uses_all_tickers(monkeypatch):
    tickers = ["AAA", "BBB"]
    calls = []

    def _fake_download(*args, **kwargs):
        calls.append(kwargs.get("tickers") or args[0])
        if "session" in kwargs:
            raise TypeError("session not supported")
        dates = pd.date_range("2020-01-01", periods=3, freq="B")
        data = pd.DataFrame(
            {
                ("Adj Close", "AAA"): [1.0, 1.1, 1.2],
                ("Adj Close", "BBB"): [2.0, 2.1, 2.2],
            },
            index=dates,
        )
        data.columns = pd.MultiIndex.from_tuples(data.columns)
        return data

    monkeypatch.setattr("prl.data_sources.yf.download", _fake_download)

    df = fetch_yfinance(tickers, "2020-01-01", "2020-01-05", session_opts={"verify_ssl": True})
    assert len(calls) == 2
    assert set(df.columns) == set(tickers)
    assert calls[-1] == tickers  # second call received full list
