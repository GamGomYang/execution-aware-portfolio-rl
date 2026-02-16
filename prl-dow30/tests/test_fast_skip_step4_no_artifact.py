from tests.helpers import run_stubbed_run_all


def test_fast_skip_step4_no_artifact(tmp_path, monkeypatch):
    ctx = run_stubbed_run_all(
        tmp_path,
        monkeypatch,
        eval_cfg={"write_step4": False},
        model_types=["baseline", "prl"],
        seeds=[0],
    )
    reports_dir = ctx["reports_dir"]
    run_ids = ctx["run_index"]["run_ids"]
    for run_id in run_ids:
        assert not (reports_dir / f"step4_report_{run_id}.md").exists()
