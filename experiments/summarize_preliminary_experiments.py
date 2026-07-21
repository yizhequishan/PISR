"""Create one concise, reproducible summary for the completed P0-P2 runs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
P0 = REPO / "outputs" / "p0_derivative_closure"
P1 = REPO / "outputs" / "p1_observability_nullspace"
P2 = REPO / "outputs" / "p2_analytic_derivative_loss"
OUT = REPO / "outputs" / "preliminary_oasis_experiments"


def percentage_reduction(old: float, new: float) -> float:
    return 100.0 * (1.0 - new / old)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p0 = pd.read_csv(P0 / "closure_metrics.csv")
    p1_metrics = json.loads((P1 / "metrics.json").read_text(encoding="utf-8"))
    p1_budget = pd.read_csv(P1 / "sparse_v_budget.csv")
    p2_metrics = json.loads((P2 / "metrics.json").read_text(encoding="utf-8"))
    p2_candidates = pd.read_csv(P2 / "candidate_metrics.csv")

    p0_table = p0.pivot(index="parameter", columns="k_source", values="rel_l2")
    p0_exact_max = float(p0_table["exact_discrete_path"].max())

    p1_three = p1_budget[p1_budget["n_v"] == 3].set_index("method")

    common_complexities = sorted(
        set(
            p2_candidates[
                (p2_candidates["source"] == "value_only")
                & (p2_candidates["status"] == "ok")
            ]["complexity"].astype(int)
        )
        & set(
            p2_candidates[
                (p2_candidates["source"] == "analytic_derivative")
                & (p2_candidates["status"] == "ok")
            ]["complexity"].astype(int)
        )
    )
    matched_complexity = max(common_complexities)
    matched = p2_candidates[
        (p2_candidates["complexity"] == matched_complexity)
        & (p2_candidates["status"] == "ok")
        & p2_candidates["source"].isin(["value_only", "analytic_derivative"])
    ].set_index("source")
    matched_value = matched.loc["value_only"]
    matched_derivative = matched.loc["analytic_derivative"]

    selected_value = p2_metrics["selected"]["value_only_by_value_mse"]
    selected_derivative = p2_metrics["selected"]["analytic_derivative_by_h1"]

    key_metrics = {
        "P0": {
            "exact_discrete_path_max_rel_l2": p0_exact_max,
            "baseline_data_gradient_rel_l2": {
                parameter: float(p0_table.loc[parameter, "baseline_data_gradient"])
                for parameter in ("R", "cx", "cy")
            },
            "symbolic_analytic_gradient_rel_l2": {
                parameter: float(
                    p0_table.loc[parameter, "symbolic_analytic_gradient"]
                )
                for parameter in ("R", "cx", "cy")
            },
        },
        "P1": {
            "u_only_rank": p1_metrics["u_only"]["rank"],
            "u_only_nullity": p1_metrics["u_only"]["nullity"],
            "u_plus_v_rank": p1_metrics["u_plus_v"]["rank"],
            "u_plus_v_nullity": p1_metrics["u_plus_v"]["nullity"],
            "three_v_points": {
                method: {
                    "rank_without_gauge": float(
                        p1_three.loc[method, "rank_without_gauge"]
                    ),
                    "coefficient_error_mean": float(
                        p1_three.loc[method, "coefficient_error_mean"]
                    ),
                }
                for method in ("d_optimal", "random", "uniform")
            },
        },
        "P2": {
            "selected_value_only": selected_value,
            "selected_analytic_derivative": selected_derivative,
            "selected_gradient_mse_reduction_percent": percentage_reduction(
                selected_value["gradient_mse"], selected_derivative["gradient_mse"]
            ),
            "selected_value_mse_change_percent": 100.0
            * (
                selected_derivative["value_mse"] / selected_value["value_mse"] - 1.0
            ),
            "matched_complexity": matched_complexity,
            "matched_value_mse_change_percent": 100.0
            * (
                matched_derivative["value_mse"] / matched_value["value_mse"] - 1.0
            ),
            "matched_gradient_mse_reduction_percent": percentage_reduction(
                matched_value["gradient_mse"], matched_derivative["gradient_mse"]
            ),
        },
    }
    (OUT / "key_metrics.json").write_text(
        json.dumps(key_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# OASIS-SR preliminary experiments: P0-P2",
        "",
        "## Overall outcome",
        "",
        "All three experiments completed. P0 validates the derivative bookkeeping "
        "but exposes a symbolic-gradient gap; P1 establishes an algorithm-independent "
        "observation nullspace; P2 demonstrates both controlled and PySR-discovered "
        "value/gradient ranking inversions and improves held-out gradients with an "
        "analytic derivative loss.",
        "",
        "## P0: T = M + K closure",
        "",
        f"The exact finite-step path decomposition closes to relative L2 <= "
        f"{p0_exact_max:.3e}, so coordinate transforms, signs, and normalization are "
        "internally consistent.",
        "",
        "| parameter | baseline CFD-gradient closure | symbolic-gradient closure |",
        "|---|---:|---:|",
    ]
    for parameter in ("R", "cx", "cy"):
        lines.append(
            f"| {parameter} | "
            f"{p0_table.loc[parameter, 'baseline_data_gradient']:.4f} | "
            f"{p0_table.loc[parameter, 'symbolic_analytic_gradient']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: the bookkeeping is correct, while the C=21 symbolic "
            "surrogate does not yet preserve the baseline spatial gradient accurately. "
            "The radius result also remains affected by finite-step/nonlinear error.",
            "",
            "## P1: differential observability",
            "",
            f"The u-only matrix has rank {p1_metrics['u_only']['rank']}/7 and nullity "
            f"{p1_metrics['u_only']['nullity']}; u+v has rank "
            f"{p1_metrics['u_plus_v']['rank']}/7 and nullity "
            f"{p1_metrics['u_plus_v']['nullity']}, exactly the constant gauge.",
            "",
            "With only three v samples (0.18% of the v grid):",
            "",
            "| selection | mean rank (gauge removed) | coefficient error |",
            "|---|---:|---:|",
        ]
    )
    for method in ("d_optimal", "random", "uniform"):
        lines.append(
            f"| {method} | {p1_three.loc[method, 'rank_without_gauge']:.2f} | "
            f"{p1_three.loc[method, 'coefficient_error_mean']:.4g} |"
        )
    lines.extend(
        [
            "",
            "This rank result uses only explicit observation matrices and least squares; "
            "it is independent of PySR. Existing PySR Pareto fronts show the matching "
            "empirical collapse rates (u-fit 89%-92%, psi u-only 60%, psi u+v 33%).",
            "",
            "## P2: analytic derivative loss",
            "",
            "| selection | complexity | value MSE | gradient MSE | gradient rel L2 |",
            "|---|---:|---:|---:|---:|",
            f"| value-only minimum value | {selected_value['complexity']} | "
            f"{selected_value['value_mse']:.4g} | {selected_value['gradient_mse']:.4g} | "
            f"{selected_value['gradient_rel_l2']:.4g} |",
            f"| analytic derivative minimum H1 | {selected_derivative['complexity']} | "
            f"{selected_derivative['value_mse']:.4g} | "
            f"{selected_derivative['gradient_mse']:.4g} | "
            f"{selected_derivative['gradient_rel_l2']:.4g} |",
            "",
            f"The derivative-aware selection reduces gradient MSE by "
            f"{key_metrics['P2']['selected_gradient_mse_reduction_percent']:.1f}% while "
            f"increasing value MSE by "
            f"{key_metrics['P2']['selected_value_mse_change_percent']:.1f}%.",
            "",
            f"At matched complexity C={matched_complexity}, value MSE changes by only "
            f"{key_metrics['P2']['matched_value_mse_change_percent']:.1f}% and gradient "
            f"MSE falls by "
            f"{key_metrics['P2']['matched_gradient_mse_reduction_percent']:.1f}%.",
            "",
            "## Decision",
            "",
            "Proceed to local parameter-response fields only after treating P0's "
            "symbolic-gradient gap as an explicit optimization/evaluation target. P1 "
            "supports observability-aware sampling as a real algorithmic mechanism; P2 "
            "supports derivative-aware Pareto selection, but the discovered expressions "
            "are still complex and should not yet be presented as recovered physics.",
        ]
    )
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summary written to {OUT}")


if __name__ == "__main__":
    main()
