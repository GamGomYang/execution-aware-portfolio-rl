from pathlib import Path

from tests.helpers import run_stubbed_run_all


def test_run_index_written(tmp_path, monkeypatch):
    output_root = tmp_path / "outputs_run_index"
    ctx = run_stubbed_run_all(tmp_path, monkeypatch, output_root=output_root, model_types=["baseline", "prl"], seeds=[0])
    run_index = ctx["run_index"]

    required_keys = {
        "exp_name",
        "timestamp",
        "config_path",
        "model_types",
        "seeds",
        "eval_windows",
        "run_ids",
        "metrics_path",
        "regime_metrics_path",
        "reports_dir",
        "output_root",
    }
    assert required_keys.issubset(run_index.keys())
    expected_run_ids = {"baseline_strategies_seed0", "runid_baseline_seed0", "runid_prl_seed0"}
    assert set(run_index["run_ids"]) == expected_run_ids

    metrics_path = Path(run_index["metrics_path"])
    regime_path = Path(run_index["regime_metrics_path"])
    assert metrics_path.exists()
    assert regime_path.exists()
    assert run_index["reports_dir"] == str(ctx["reports_dir"])
    assert run_index["output_root"] == str(output_root)
    assert run_index["seeds"] == [0]
    assert run_index["model_types"] == ["baseline", "prl"]
    assert run_index["eval_windows"]["test_start"] == "2020-01-01"
    assert run_index["eval_windows"]["test_end"] == "2020-01-10"
