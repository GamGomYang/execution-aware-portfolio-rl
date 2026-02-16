import pandas as pd

from prl.train import prepare_market_and_features


def _make_market_frames():
    dates = pd.date_range("2020-01-01", periods=15, freq="B")
    returns = pd.DataFrame(0.001, index=dates, columns=["AAA", "BBB"])
    prices = pd.DataFrame(100.0, index=dates, columns=["AAA", "BBB"])
    return prices, returns


def _make_config(processed_dir, train_start, train_end, test_end):
    return {
        "dates": {
            "train_start": train_start,
            "train_end": train_end,
            "test_start": train_start,
            "test_end": test_end,
        },
        "data": {
            "raw_dir": "data/raw",
            "processed_dir": str(processed_dir),
        },
    }


def test_vol_stats_keyed_by_lv(tmp_path, monkeypatch):
    prices, returns = _make_market_frames()
    manifest = {"data_manifest_hash": "a" * 64}

    def _fake_load_market_data(*args, **kwargs):
        return prices, returns, manifest, pd.DataFrame()

    monkeypatch.setattr("prl.train.load_market_data", _fake_load_market_data)

    config = _make_config(tmp_path / "processed", "2020-01-01", "2020-01-10", "2020-01-21")

    _, features_lv2 = prepare_market_and_features(config=config, lv=2, force_refresh=False)
    _, features_lv3 = prepare_market_and_features(config=config, lv=3, force_refresh=False)

    assert features_lv2.stats_path.name != features_lv3.stats_path.name
    assert features_lv2.stats_path.exists()
    assert features_lv3.stats_path.exists()


def test_vol_stats_keyed_by_train_window(tmp_path, monkeypatch):
    prices, returns = _make_market_frames()
    manifest = {"data_manifest_hash": "b" * 64}

    def _fake_load_market_data(*args, **kwargs):
        return prices, returns, manifest, pd.DataFrame()

    monkeypatch.setattr("prl.train.load_market_data", _fake_load_market_data)

    config_short = _make_config(tmp_path / "processed", "2020-01-01", "2020-01-08", "2020-01-21")
    config_long = _make_config(tmp_path / "processed", "2020-01-01", "2020-01-10", "2020-01-21")

    _, features_short = prepare_market_and_features(config=config_short, lv=2, force_refresh=False)
    _, features_long = prepare_market_and_features(config=config_long, lv=2, force_refresh=False)

    assert features_short.stats_path.name != features_long.stats_path.name
    assert features_short.stats_path.exists()
    assert features_long.stats_path.exists()
