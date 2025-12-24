import pandas as pd
import pytest

from prl.data import load_market_data


def test_paper_mode_requires_cache_missing(tmp_path):
    processed_dir = tmp_path / "processed"
    with pytest.raises(RuntimeError, match="CACHE_MISSING"):
        load_market_data(
            start_date="2020-01-01",
            end_date="2020-01-10",
            processed_dir=processed_dir,
            tickers=["AAA"],
            force_refresh=False,
            source="yfinance_only",
            offline=True,
            require_cache=True,
            paper_mode=True,
        )


def test_paper_mode_uses_existing_cache(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2020-01-01", periods=6, freq="B")
    prices = pd.DataFrame({"AAA": 100.0 + pd.RangeIndex(len(dates))}, index=dates)
    returns = prices.pct_change().dropna()
    prices.to_parquet(processed_dir / "prices.parquet")
    returns.to_parquet(processed_dir / "returns.parquet")

    market = load_market_data(
        start_date="2020-01-01",
        end_date="2020-01-10",
        processed_dir=processed_dir,
        tickers=["AAA"],
        force_refresh=False,
        source="yfinance_only",
        offline=True,
        require_cache=True,
        paper_mode=True,
    )
    assert market.prices.equals(prices)
