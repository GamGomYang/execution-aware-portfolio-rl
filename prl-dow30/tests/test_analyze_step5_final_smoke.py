from pathlib import Path

import pandas as pd

from analysis import analyze_step5_final


def _make_metrics_rows(exp_tag: str, seeds: list[int], *, sharpe_bias: float, cumret_bias: float, mdd_bias: float, turnover: float, std_daily: float, target_turnover: float) -> list[dict]:
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
                "sharpe_net_exp": 0.70 + 0.05 * seed + sharpe_bias,
                "cumulative_return_net_exp": 0.12 + 0.02 * seed + cumret_bias,
                "max_drawdown_net_exp": -0.22 - 0.01 * seed + mdd_bias,
                "avg_turnover_exec": turnover,
                "avg_turnover_target": target_turnover,
                "std_daily_net_return_exp": std_daily,
            }
        )
    return rows


def _make_regime_rows(metrics_rows: list[dict]) -> list[dict]:
    regime_rows = []
    regime_offsets = {"low": 0.05, "mid": 0.0, "high": -0.05}
    for row in metrics_rows:
        for regime, offset in regime_offsets.items():
            regime_rows.append(
                {
                    "run_id": row["run_id"],
                    "model_type": row["model_type"],
                    "seed": row["seed"],
                    "regime": regime,
                    "period": "test",
                    "eval_window": row["eval_window"],
                    "sharpe_net_exp": row["sharpe_net_exp"] + offset,
                    "cumulative_return_net_exp": row["cumulative_return_net_exp"] + 0.01 * offset,
                    "max_drawdown_net_exp": row["max_drawdown_net_exp"] + 0.02 * offset,
                    "avg_turnover_exec": row["avg_turnover_exec"] + 0.01,
                    "avg_turnover_target": row["avg_turnover_target"] + 0.01,
                    "std_daily_net_return_exp": row["std_daily_net_return_exp"] + 0.001,
                }
            )
    return regime_rows


def _write_trace(path: Path, run_id: str, seed: int, *, base_return: float, turnover_exec: float, turnover_target: float) -> None:
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    df = pd.DataFrame(
        {
            "date": dates,
            "seed": seed,
            "net_return_exp": [base_return] * len(dates),
            "turnover_exec": [turnover_exec] * len(dates),
            "turnover_target": [turnover_target] * len(dates),
            "eval_window": ["W1"] * len(dates),
        }
    )
    df.to_parquet(path, index=False)


def test_analyze_step5_final_smoke(tmp_path):
    input_root = tmp_path / "step5"
    reports_dir = input_root / "reports"
    archive_dir = reports_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    seeds = [0, 1]

    metrics_a = _make_metrics_rows(
        "exp_S5_final_baseline_eta010",
        seeds,
        sharpe_bias=0.00,
        cumret_bias=0.00,
        mdd_bias=0.00,
        turnover=0.15,
        std_daily=0.020,
        target_turnover=0.14,
    )
    metrics_b = _make_metrics_rows(
        "exp_S5_final_prl_eta010",
        seeds,
        sharpe_bias=0.08,
        cumret_bias=0.03,
        mdd_bias=0.02,
        turnover=0.14,
        std_daily=0.018,
        target_turnover=0.13,
    )
    metrics_c = _make_metrics_rows(
        "exp_S5_ablate_baseline_etaNone",
        seeds,
        sharpe_bias=-0.05,
        cumret_bias=-0.03,
        mdd_bias=-0.03,
        turnover=0.22,
        std_daily=0.024,
        target_turnover=0.20,
    )
    metrics_d = _make_metrics_rows(
        "exp_S5_ablate_prl_etaNone",
        seeds,
        sharpe_bias=-0.01,
        cumret_bias=-0.01,
        mdd_bias=-0.02,
        turnover=0.19,
        std_daily=0.022,
        target_turnover=0.18,
    )

    pd.DataFrame(metrics_a).to_csv(archive_dir / "metrics_exp_S5_final_baseline_eta010__aaaa.csv", index=False)
    pd.DataFrame(metrics_b).to_csv(archive_dir / "metrics_exp_S5_final_prl_eta010__bbbb.csv", index=False)
    pd.DataFrame(metrics_c).to_csv(archive_dir / "metrics_exp_S5_ablate_baseline_etaNone__cccc.csv", index=False)
    pd.DataFrame(metrics_d).to_csv(archive_dir / "metrics_exp_S5_ablate_prl_etaNone__dddd.csv", index=False)

    pd.DataFrame(_make_regime_rows(metrics_a)).to_csv(
        archive_dir / "regime_metrics_exp_S5_final_baseline_eta010__aaaa.csv", index=False
    )
    pd.DataFrame(_make_regime_rows(metrics_b)).to_csv(
        archive_dir / "regime_metrics_exp_S5_final_prl_eta010__bbbb.csv", index=False
    )
    pd.DataFrame(_make_regime_rows(metrics_c)).to_csv(
        archive_dir / "regime_metrics_exp_S5_ablate_baseline_etaNone__cccc.csv", index=False
    )
    pd.DataFrame(_make_regime_rows(metrics_d)).to_csv(
        archive_dir / "regime_metrics_exp_S5_ablate_prl_etaNone__dddd.csv", index=False
    )

    for row in metrics_a + metrics_b:
        _write_trace(
            reports_dir / f"trace_{row['run_id']}.parquet",
            row["run_id"],
            int(row["seed"]),
            base_return=0.001 + 0.0002 * int(row["seed"]),
            turnover_exec=float(row["avg_turnover_exec"]),
            turnover_target=float(row["avg_turnover_target"]),
        )

    out_dir = reports_dir / "paper" / "step5"
    analyze_step5_final.main(["--input-root", str(input_root), "--out-dir", str(out_dir)])

    expected_files = [
        "table_main.csv",
        "table_main_robust.tex",
        "robust_stats_summary.csv",
        "robust_delta_prl_minus_base.csv",
        "table_regime.csv",
        "table_ablation.csv",
        "stats_tests.csv",
        "fig_equity_curve_net_exp.png",
        "fig_drawdown_net_exp.png",
        "fig_turnover_exec_vs_target.png",
        "summary_step5.md",
    ]
    for name in expected_files:
        assert (out_dir / name).exists(), name
