from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import step6_run_matrix as run_matrix


def _make_args(*, model_path: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        model_root="outputs",
        offline=True,
        max_steps=252,
        model_path=model_path,
    )


def test_resolve_model_path_by_seed_independent_uses_per_seed_models(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_build_eval_context(**kwargs):
        calls.append(kwargs)
        seed = int(kwargs["seed"])
        return SimpleNamespace(model_path=Path(f"/tmp/model_seed{seed}.zip"))

    monkeypatch.setattr(run_matrix, "build_eval_context", fake_build_eval_context)

    model_paths = run_matrix._resolve_model_path_by_seed(
        args=_make_args(),
        config_path=Path("configs/step6_main.yaml"),
        seeds=[0, 1, 2],
        model_type="prl",
        seed_model_mode="independent",
    )

    assert model_paths == {
        0: "/tmp/model_seed0.zip",
        1: "/tmp/model_seed1.zip",
        2: "/tmp/model_seed2.zip",
    }
    assert len(calls) == 3
    assert all(call["model_path_arg"] is None for call in calls)
    assert all(bool(call["prefer_metadata_config"]) for call in calls)


def test_resolve_model_path_by_seed_shared_uses_single_model(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_build_eval_context(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(model_path=Path("/tmp/shared_model.zip"))

    monkeypatch.setattr(run_matrix, "build_eval_context", fake_build_eval_context)

    model_paths = run_matrix._resolve_model_path_by_seed(
        args=_make_args(),
        config_path=Path("configs/step6_main.yaml"),
        seeds=[0, 1, 2],
        model_type="prl",
        seed_model_mode="shared",
    )

    assert model_paths == {
        0: "/tmp/shared_model.zip",
        1: "/tmp/shared_model.zip",
        2: "/tmp/shared_model.zip",
    }
    assert len(calls) == 1
    assert int(calls[0]["seed"]) == 0


def test_resolve_model_path_by_seed_independent_missing_seed_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_build_eval_context(**kwargs):
        seed = int(kwargs["seed"])
        if seed == 1:
            raise FileNotFoundError("Model not found: outputs/models/prl_seed1_final.zip")
        return SimpleNamespace(model_path=Path(f"/tmp/model_seed{seed}.zip"))

    monkeypatch.setattr(run_matrix, "build_eval_context", fake_build_eval_context)

    with pytest.raises(FileNotFoundError) as excinfo:
        run_matrix._resolve_model_path_by_seed(
            args=_make_args(),
            config_path=Path("configs/step6_main.yaml"),
            seeds=[0, 1],
            model_type="prl",
            seed_model_mode="independent",
        )

    msg = str(excinfo.value)
    assert "missing_seed=1" in msg
    assert "expected_path=outputs/models/prl_seed1_final.zip" in msg
    assert "--seed-model-mode shared" in msg


def test_resolve_model_path_by_seed_independent_with_model_path_and_multiseed_raises() -> None:
    with pytest.raises(ValueError) as excinfo:
        run_matrix._resolve_model_path_by_seed(
            args=_make_args(model_path="/tmp/one_model.zip"),
            config_path=Path("configs/step6_main.yaml"),
            seeds=[0, 1],
            model_type="prl",
            seed_model_mode="independent",
        )

    assert "--model-path with multiple seeds is incompatible" in str(excinfo.value)
