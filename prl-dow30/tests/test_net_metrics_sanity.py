from scripts.sanity_check_net_metrics import run_checks
from tests.helpers import run_stubbed_run_all


def test_net_metrics_sanity(tmp_path, monkeypatch):
    ctx = run_stubbed_run_all(tmp_path, monkeypatch, model_types=["baseline"], seeds=[0])
    run_checks(ctx["output_root"], require_trace=True)
