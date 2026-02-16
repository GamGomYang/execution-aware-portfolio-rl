from tests.helpers import run_stubbed_run_all


def test_output_root_creates_dirs(tmp_path, monkeypatch):
    output_root = tmp_path / "alt_outputs"
    ctx = run_stubbed_run_all(tmp_path, monkeypatch, output_root=output_root, model_types=["baseline"], seeds=[0])
    for name in ["reports", "traces", "models", "logs"]:
        path = output_root / name
        assert path.exists(), f"{path} not created"
    assert (output_root / "reports" / "metrics.csv").exists()
    assert ctx["output_root"] == output_root
