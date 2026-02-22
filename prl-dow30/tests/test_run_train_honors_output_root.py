from pathlib import Path

import yaml


def _write_config(path: Path, *, output_root: Path | None = None) -> None:
    cfg = {
        "data": {
            "offline": True,
            "require_cache": False,
            "paper_mode": False,
        }
    }
    if output_root is not None:
        cfg["output"] = {"root": str(output_root)}
    path.write_text(yaml.safe_dump(cfg))


def test_run_train_honors_cli_output_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "config.yaml"
    cli_root = tmp_path / "cli_root"
    _write_config(cfg_path)

    captured = {}

    def _fake_run_training(**kwargs):
        captured.update(kwargs)
        model_path = Path(kwargs["output_dir"]) / "stub_final.zip"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("stub")
        return model_path

    monkeypatch.setattr("scripts.run_train.resolve_signal_configuration", lambda cfg: {})
    monkeypatch.setattr("scripts.run_train.run_training", _fake_run_training)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_train.py",
            "--config",
            str(cfg_path),
            "--model-type",
            "prl",
            "--seed",
            "7",
            "--offline",
            "--output-root",
            str(cli_root),
        ],
    )

    from scripts import run_train as run_train_script

    run_train_script.main()

    assert Path(captured["output_dir"]) == cli_root / "models"
    assert Path(captured["reports_dir"]) == cli_root / "reports"
    assert Path(captured["logs_dir"]) == cli_root / "logs"


def test_run_train_falls_back_to_config_output_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_path = tmp_path / "config.yaml"
    cfg_root = tmp_path / "cfg_root"
    _write_config(cfg_path, output_root=cfg_root)

    captured = {}

    def _fake_run_training(**kwargs):
        captured.update(kwargs)
        model_path = Path(kwargs["output_dir"]) / "stub_final.zip"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("stub")
        return model_path

    monkeypatch.setattr("scripts.run_train.resolve_signal_configuration", lambda cfg: {})
    monkeypatch.setattr("scripts.run_train.run_training", _fake_run_training)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_train.py",
            "--config",
            str(cfg_path),
            "--model-type",
            "baseline",
            "--seed",
            "0",
            "--offline",
        ],
    )

    from scripts import run_train as run_train_script

    run_train_script.main()

    assert Path(captured["output_dir"]) == cfg_root / "models"
    assert Path(captured["reports_dir"]) == cfg_root / "reports"
    assert Path(captured["logs_dir"]) == cfg_root / "logs"
