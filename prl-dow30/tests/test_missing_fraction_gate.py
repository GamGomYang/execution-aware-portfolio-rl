import numpy as np
import pandas as pd
import pytest

from prl.data import load_market_data


def _make_index():
    idx = pd.bdate_range(start="2020-01-01", periods=100)
    start = idx[0].date().isoformat()
    end = idx[-1].date().isoformat()
    return start, end


def _quality_params(max_missing_fraction):
    return {
        "min_vol_std": 0.0,
        "min_max_abs_return": 0.0,
        "max_flat_fraction": 1.0,
        "max_missing_fraction": max_missing_fraction,
    }


def _fake_fetch_factory(missing_tickers=None, missing_fraction=0.30):
    missing_tickers = set(missing_tickers or [])

    def _fake_fetch(tickers, start, end, session_opts=None):
        ticker = list(tickers)[0]
        idx = pd.date_range(start=start, end=end, freq="B")
        values = (100.0 + np.arange(len(idx))).astype(float)
        if ticker in missing_tickers:
            missing_count = int(len(idx) * missing_fraction)
            if missing_count:
                values[1 : 1 + missing_count] = np.nan
        return pd.DataFrame({ticker: values}, index=idx)

    return _fake_fetch


def _make_cfg(processed_dir, start, end, tickers, quality_params, data_overrides=None):
    data_cfg = {
        "processed_dir": str(processed_dir),
        "source": "yfinance_only",
        "universe_policy": "availability_filtered",
        "min_assets": 1,
        "min_history_days": 20,
        "history_tolerance_days": 0,
        "ticker_substitutions": {},
        "quality_params": quality_params,
        "tickers": tickers,
    }
    if data_overrides:
        data_cfg.update(data_overrides)
    return {"dates": {"train_start": start, "test_end": end}, "data": data_cfg}


def test_raw_missing_fraction_detected(tmp_path, monkeypatch):
    start, end = _make_index()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("prl.data.fetch_yfinance", _fake_fetch_factory({"BBB"}, 0.30))
    cfg = _make_cfg(
        tmp_path / "processed",
        start,
        end,
        ["AAA", "BBB", "CCC"],
        _quality_params(1.0),
    )

    (
        _prices,
        _returns,
        _manifest,
        _quality_summary,
        _raw_prices,
        _filled_prices,
        raw_missing_fraction,
        _drop_decisions,
        _market_closed_info,
        _debug_info,
    ) = load_market_data(
        cfg,
        offline=False,
        require_cache=False,
        cache_only=False,
        force_refresh=True,
        debug_return_intermediates=True,
    )

    assert raw_missing_fraction["BBB"] >= 0.30
    report_path = tmp_path / "outputs" / "reports" / "data_quality_summary.csv"
    assert report_path.exists()
    report_df = pd.read_csv(report_path)
    report_row = report_df.loc[report_df["ticker"] == "BBB", "missing_fraction_raw"]
    assert report_row.iloc[0] == pytest.approx(raw_missing_fraction["BBB"], rel=1e-6)


def test_gate_drops_ticker_under_availability_filtered(tmp_path, monkeypatch):
    start, end = _make_index()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("prl.data.fetch_yfinance", _fake_fetch_factory({"BBB"}, 0.30))
    cfg = _make_cfg(
        tmp_path / "processed",
        start,
        end,
        ["AAA", "BBB", "CCC"],
        _quality_params(0.10),
        data_overrides={"min_assets": 2},
    )

    _prices, _returns, manifest, quality_summary = load_market_data(
        cfg,
        offline=False,
        require_cache=False,
        cache_only=False,
        force_refresh=True,
    )

    assert "BBB" in manifest["dropped_tickers"]
    assert "BBB" not in manifest["kept_tickers"]
    assert manifest["dropped_reasons"]["BBB"] == "RAW_MISSING_FRACTION_EXCEEDED"
    row = quality_summary.loc[quality_summary["ticker"] == "BBB", "missing_fraction_raw"]
    assert not row.empty
    assert row.iloc[0] > 0.0

    report_path = tmp_path / "outputs" / "reports" / "data_quality_summary.csv"
    assert report_path.exists()
    report_df = pd.read_csv(report_path)
    report_row = report_df.loc[report_df["ticker"] == "BBB", "missing_fraction_raw"]
    assert report_row.iloc[0] > 0.0


def test_universe_too_small(tmp_path, monkeypatch):
    start, end = _make_index()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("prl.data.fetch_yfinance", _fake_fetch_factory({"BBB", "CCC"}, 0.30))
    cfg = _make_cfg(
        tmp_path / "processed",
        start,
        end,
        ["AAA", "BBB", "CCC"],
        _quality_params(0.10),
        data_overrides={"min_assets": 3},
    )

    with pytest.raises(RuntimeError, match="DATA_UNIVERSE_TOO_SMALL"):
        load_market_data(
            cfg,
            offline=False,
            require_cache=False,
            cache_only=False,
            force_refresh=True,
        )
