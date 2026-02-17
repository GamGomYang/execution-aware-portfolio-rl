from __future__ import annotations

import argparse
from math import comb
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step6 paper-ready CSV pack from a fixed run directory.")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory, e.g. outputs/step6_runs/20260217_171039")
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    return pd.read_csv(path)


def _load_metrics(step_dir: Path) -> pd.DataFrame:
    files = sorted(step_dir.glob("kappa_*/*/seed_*/metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No metrics.csv found under: {step_dir}")
    rows = [pd.read_csv(path) for path in files]
    return pd.concat(rows, ignore_index=True)


def _exact_sign_test_two_sided(n_pos: int, n_neg: int) -> float:
    n = int(n_pos + n_neg)
    if n <= 0:
        return 1.0
    k = int(n_pos)
    probs = [comb(n, i) * (0.5**n) for i in range(n + 1)]
    cdf = sum(probs[: k + 1])
    sf = sum(probs[k:])
    return float(min(1.0, 2.0 * min(cdf, sf)))


def _make_eta_t_summary_if_missing(rule_vol_dir: Path) -> pd.DataFrame:
    out_path = rule_vol_dir / "eta_t_summary.csv"
    if out_path.exists():
        return pd.read_csv(out_path)

    rows: list[dict] = []
    for path in sorted(rule_vol_dir.glob("kappa_*/*/seed_*/trace.parquet")):
        trace = pd.read_parquet(path)
        kappa = float(path.parts[-4].replace("kappa_", ""))
        arm = path.parts[-3]
        seed = int(path.parts[-2].replace("seed_", ""))
        rule_vol_a = None
        if arm.startswith("rule_vol_a_"):
            rule_vol_a = float(arm.replace("rule_vol_a_", ""))

        eta = pd.to_numeric(trace["eta_t"], errors="coerce").dropna()
        rows.append(
            {
                "kappa": kappa,
                "seed": seed,
                "arm": arm,
                "rule_vol_a": rule_vol_a,
                "eta_mean": float(eta.mean()),
                "eta_std": float(eta.std()),
                "eta_p05": float(eta.quantile(0.05)),
                "eta_p50": float(eta.quantile(0.50)),
                "eta_p95": float(eta.quantile(0.95)),
                "eta_min": float(eta.min()),
                "eta_max": float(eta.max()),
                "clip_max_hit_ratio": float((eta >= 0.5).mean()),
            }
        )

    df = pd.DataFrame(rows).sort_values(["kappa", "seed", "arm"]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"Failed to build eta_t_summary.csv from traces in: {rule_vol_dir}")
    df.to_csv(out_path, index=False)
    return df


def _build_step6_main_eta_table(run_dir: Path) -> pd.DataFrame:
    step_dir = run_dir / "step6_main"
    agg = _read_csv(step_dir / "aggregate.csv")
    metrics = _load_metrics(step_dir)

    median_cagr = (
        metrics.groupby(["kappa", "arm"], as_index=False)["cagr"]
        .median()
        .rename(columns={"cagr": "median_cagr"})
    )
    out = agg.merge(median_cagr, on=["kappa", "arm"], how="left")
    out = out[
        [
            "kappa",
            "eta",
            "arm",
            "median_sharpe",
            "median_turnover_exec",
            "median_cagr",
            "collapse_rate",
        ]
    ].copy()
    arm_order = {"baseline": 0, "main": 1}
    out["__arm_order"] = out["arm"].map(arm_order).fillna(99)
    out = out.sort_values(["kappa", "__arm_order"]).drop(columns="__arm_order").reset_index(drop=True)
    return out


def _build_step6_main_paired_summary(run_dir: Path) -> pd.DataFrame:
    paired = _read_csv(run_dir / "step6_main" / "paired_delta.csv")
    out = (
        paired.groupby("kappa", as_index=False)
        .agg(
            n_pairs=("delta_sharpe", "size"),
            median_delta_sharpe=("delta_sharpe", "median"),
            median_delta_cagr=("delta_cagr", "median"),
            win_rate_sharpe=("delta_sharpe", lambda x: float((x > 0).mean())),
        )
        .sort_values("kappa")
        .reset_index(drop=True)
    )
    return out


def _build_step6_main_seed_table(run_dir: Path) -> pd.DataFrame:
    metrics = _load_metrics(run_dir / "step6_main")
    out = metrics[
        ["kappa", "arm", "seed", "sharpe_net_lin", "avg_turnover_exec", "cagr", "maxdd"]
    ].sort_values(["kappa", "arm", "seed"])
    return out.reset_index(drop=True)


def _build_step6_main_sign_test(run_dir: Path) -> pd.DataFrame:
    paired = _read_csv(run_dir / "step6_main" / "paired_delta.csv")

    def _row(scope: str, kappa_label: str, frame: pd.DataFrame) -> dict:
        pos = int((frame["delta_sharpe"] > 0).sum())
        neg = int((frame["delta_sharpe"] < 0).sum())
        n = pos + neg
        return {
            "scope": scope,
            "kappa": kappa_label,
            "n": n,
            "n_pos": pos,
            "n_neg": neg,
            "p_two_sided": _exact_sign_test_two_sided(pos, neg),
        }

    out = pd.DataFrame(
        [
            _row("all_kappa", "all", paired),
            _row("kappa_pos_only", ">0", paired[paired["kappa"] > 0]),
        ]
    )
    return out


def _build_step6_rule_vol_negative_summary(run_dir: Path) -> pd.DataFrame:
    rule_dir = run_dir / "step6_rule_vol"
    agg = _read_csv(rule_dir / "aggregate.csv")
    paired = _read_csv(rule_dir / "paired_delta.csv")
    eta_summary = _make_eta_t_summary_if_missing(rule_dir)

    fixed = agg[agg["arm"] == "fixed_comparison"][
        ["kappa", "median_sharpe", "median_turnover_exec"]
    ].rename(columns={"median_sharpe": "fixed_sharpe", "median_turnover_exec": "fixed_turnover"})

    rule = agg[agg["arm"] == "rule_vol"][["kappa", "median_sharpe", "median_turnover_exec"]].rename(
        columns={"median_sharpe": "rule_sharpe", "median_turnover_exec": "rule_turnover"}
    )

    delta = (
        paired.groupby("kappa", as_index=False)["delta_sharpe"]
        .median()
        .rename(columns={"delta_sharpe": "median_delta_sharpe"})
    )

    clip = (
        eta_summary[eta_summary["arm"].str.startswith("rule_vol_a_")]
        .groupby("kappa", as_index=False)["clip_max_hit_ratio"]
        .mean()
    )

    out = fixed.merge(rule, on="kappa", how="inner")
    out = out.merge(delta, on="kappa", how="left")
    out = out.merge(clip, on="kappa", how="left")
    out = out[
        [
            "kappa",
            "fixed_sharpe",
            "rule_sharpe",
            "median_delta_sharpe",
            "fixed_turnover",
            "rule_turnover",
            "clip_max_hit_ratio",
        ]
    ].sort_values("kappa")
    return out.reset_index(drop=True)


def _validate_outputs(output_dir: Path) -> None:
    checks: dict[str, list[str]] = {
        "step6_main_eta_table.csv": [
            "kappa",
            "eta",
            "arm",
            "median_sharpe",
            "median_turnover_exec",
            "median_cagr",
            "collapse_rate",
        ],
        "step6_main_paired_summary.csv": [
            "kappa",
            "n_pairs",
            "median_delta_sharpe",
            "median_delta_cagr",
            "win_rate_sharpe",
        ],
        "step6_main_seed_table.csv": [
            "kappa",
            "arm",
            "seed",
            "sharpe_net_lin",
            "avg_turnover_exec",
            "cagr",
            "maxdd",
        ],
        "step6_main_sign_test.csv": ["scope", "kappa", "n", "n_pos", "n_neg", "p_two_sided"],
        "step6_rule_vol_negative_summary.csv": [
            "kappa",
            "fixed_sharpe",
            "rule_sharpe",
            "median_delta_sharpe",
            "fixed_turnover",
            "rule_turnover",
            "clip_max_hit_ratio",
        ],
    }

    for filename, required_columns in checks.items():
        path = output_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing output: {path}")
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"Output is empty: {path}")
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        if df[required_columns].isna().any().any():
            raise ValueError(f"{path} has NaN in required columns: {required_columns}")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    output_dir = run_dir / "paper_pack"
    output_dir.mkdir(parents=True, exist_ok=True)

    step6_main_eta_table = _build_step6_main_eta_table(run_dir)
    step6_main_paired_summary = _build_step6_main_paired_summary(run_dir)
    step6_main_seed_table = _build_step6_main_seed_table(run_dir)
    step6_main_sign_test = _build_step6_main_sign_test(run_dir)
    step6_rule_vol_negative_summary = _build_step6_rule_vol_negative_summary(run_dir)

    step6_main_eta_table.to_csv(output_dir / "step6_main_eta_table.csv", index=False)
    step6_main_paired_summary.to_csv(output_dir / "step6_main_paired_summary.csv", index=False)
    step6_main_seed_table.to_csv(output_dir / "step6_main_seed_table.csv", index=False)
    step6_main_sign_test.to_csv(output_dir / "step6_main_sign_test.csv", index=False)
    step6_rule_vol_negative_summary.to_csv(output_dir / "step6_rule_vol_negative_summary.csv", index=False)

    _validate_outputs(output_dir)

    print(f"paper_pack created: {output_dir}")
    for filename in [
        "step6_main_eta_table.csv",
        "step6_main_paired_summary.csv",
        "step6_main_seed_table.csv",
        "step6_main_sign_test.csv",
        "step6_rule_vol_negative_summary.csv",
    ]:
        print(output_dir / filename)


if __name__ == "__main__":
    main()
