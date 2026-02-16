import json
from pathlib import Path

import pandas as pd

from analysis import step5_final_gate


def _make_metrics_rows(exp_tag: str, seeds: list[int], *, sharpe_shift: float, turnover: float, mdd_shift: float) -> list[dict]:
    rows = []
    for seed in seeds:
        run_id = f"{exp_tag}_seed{seed}_prl_abcd"
        rows.append(
            {
                "run_id": run_id,
                "eval_id": run_id,
                "eval_window": "W1",
                "model_type": "prl_sac",
                "seed": seed,
                "period": "test",
                "sharpe_net_exp": 0.6 + 0.02 * seed + sharpe_shift,
                "avg_turnover_exec": turnover,
                "max_drawdown_net_exp": -0.20 - 0.01 * seed + mdd_shift,
            }
        )
    return rows


def _write_prl_logs(logs_dir: Path, run_ids: list[str]) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    for run_id in run_ids:
        pd.DataFrame(
            [
                {
                    "schema_version": "1.1",
                    "run_id": run_id,
                    "model_type": "prl",
                    "seed": 0,
                    "timesteps": 100,
                    "emergency_rate": 0.01,
                    "prl_prob_p05": 0.1,
                    "prl_prob_p95": 0.9,
                    "prl_prob_std": 0.3,
                    "prl_prob_min": 0.0,
                    "prl_prob_max": 1.0,
                }
            ]
        ).to_csv(logs_dir / f"train_{run_id}.csv", index=False)


def test_step5_final_gate_pass(tmp_path):
    input_root = tmp_path / "step5"
    archive_dir = input_root / "reports" / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    seeds = [0, 1, 2, 3, 4]
    base_rows = _make_metrics_rows(
        "exp_S5_final_baseline_eta010",
        seeds,
        sharpe_shift=0.0,
        turnover=0.20,
        mdd_shift=0.0,
    )
    prl_rows = _make_metrics_rows(
        "exp_S5_final_prl_eta010",
        seeds,
        sharpe_shift=0.02,
        turnover=0.205,
        mdd_shift=0.01,
    )

    pd.DataFrame(base_rows).to_csv(archive_dir / "metrics_exp_S5_final_baseline_eta010__aaaa.csv", index=False)
    pd.DataFrame(prl_rows).to_csv(archive_dir / "metrics_exp_S5_final_prl_eta010__bbbb.csv", index=False)

    _write_prl_logs(input_root / "logs", [row["run_id"] for row in prl_rows])

    step5_final_gate.main(["--input-root", str(input_root)])

    result_path = input_root / "reports" / "paper" / "step5" / "step5_gate_result.json"
    assert result_path.exists()

    result = json.loads(result_path.read_text())
    assert result["step5_gate_pass"] is True
    assert result["comparison"]["baseline"]["prl_gate_pass"] == "SKIP"
    assert result["comparison"]["prl"]["prl_gate_pass"] is True


def test_step5_final_gate_fails_turnover_rule(tmp_path):
    input_root = tmp_path / "step5"
    archive_dir = input_root / "reports" / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    seeds = [0, 1, 2, 3, 4]
    base_rows = _make_metrics_rows(
        "exp_S5_final_baseline_eta010",
        seeds,
        sharpe_shift=0.0,
        turnover=0.20,
        mdd_shift=0.0,
    )
    prl_rows = _make_metrics_rows(
        "exp_S5_final_prl_eta010",
        seeds,
        sharpe_shift=0.03,
        turnover=0.25,
        mdd_shift=0.01,
    )

    pd.DataFrame(base_rows).to_csv(archive_dir / "metrics_exp_S5_final_baseline_eta010__aaaa.csv", index=False)
    pd.DataFrame(prl_rows).to_csv(archive_dir / "metrics_exp_S5_final_prl_eta010__bbbb.csv", index=False)

    _write_prl_logs(input_root / "logs", [row["run_id"] for row in prl_rows])

    step5_final_gate.main(["--input-root", str(input_root)])

    result = json.loads((input_root / "reports" / "paper" / "step5" / "step5_gate_result.json").read_text())
    assert result["step5_gate_pass"] is False
    assert "turnover_increase_forbidden" in result["step5_gate_reason"]
