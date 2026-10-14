"""Summarize P0-P2 while retaining source identities and evaluation protocols."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pipeline_contracts import atomic_json, file_hash, read_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, folder in (("p0", "p0_derivative_closure"), ("p1", "p1_observability_nullspace"),
                         ("p2", "p2_analytic_derivative_loss")):
        parser.add_argument(f"--{name}-dir", type=Path, default=REPO / "outputs" / folder)
    parser.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "preliminary_oasis_experiments")
    return parser.parse_args()


def percentage_reduction(old: float, new: float):
    return 100.0 * (1.0 - new / old) if old != 0 else None


def number(value, spec=".4g"):
    return "undefined (zero reference or unavailable)" if value is None else format(value, spec)


def p2_protocol(metrics):
    protocol = metrics.get("evaluation_protocol", {})
    independent = (protocol.get("protocol") == "train_validate_select_then_test_v2"
                   and all(protocol.get(key, {}).get("sha256") for key in ("train", "validation", "test"))
                   and len({protocol[key]["sha256"] for key in ("train", "validation", "test")}) == 3
                   and all("expression_sha256" in row and "validation_value_mse" in row
                           for row in metrics["selected"].values()))
    regime = "independent_test_after_validation_selection" if independent else "legacy_or_unverified_selection_grid"
    explanation = ("Expressions were frozen using an independent validation set; the selected-model metrics below use the test grid."
                   if independent else
                   "Legacy/unverified protocol: selection and evaluation separation is not established. These metrics are selection-grid diagnostics, not independent test evidence.")
    return regime, explanation, protocol


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    p0 = pd.read_csv(args.p0_dir / "closure_metrics.csv")
    p0_meta_path = args.p0_dir / "metrics.json"
    p0_meta = read_json(p0_meta_path) if p0_meta_path.exists() else {}
    p1 = read_json(args.p1_dir / "metrics.json")
    budgets = pd.read_csv(args.p1_dir / "sparse_v_budget.csv")
    p2 = read_json(args.p2_dir / "metrics.json")
    candidates = pd.read_csv(args.p2_dir / "candidate_metrics.csv")
    regime, protocol_note, protocol = p2_protocol(p2)
    selected_value = p2["selected"]["value_only_by_value_mse"]
    selected_derivative = p2["selected"]["analytic_derivative_by_h1"]
    closure = p0.pivot(index="parameter", columns="k_source", values="rel_l2")
    three = budgets[budgets.n_v == 3].set_index("method")
    uniform_protocol = p1.get("sampling_protocol", "legacy_prefix_uniform_unverified")
    valid = candidates[(candidates.status == "ok") & candidates.source.isin(["value_only", "analytic_derivative"])]
    common = set(valid[valid.source == "value_only"].complexity) & set(valid[valid.source == "analytic_derivative"].complexity)
    matched_complexity = int(max(common)) if common else None
    matched_value_change = matched_gradient_reduction = None
    if common:
        rows = valid[valid.complexity == matched_complexity]
        value_rows, derivative_rows = rows[rows.source == "value_only"], rows[rows.source == "analytic_derivative"]
        # A duplicate complexity does not specify an expression identity; do not choose by test error.
        if len(value_rows) == len(derivative_rows) == 1:
            a, b = value_rows.iloc[0], derivative_rows.iloc[0]
            delta = percentage_reduction(a.value_mse, b.value_mse)
            matched_value_change = -delta if delta is not None else None
            matched_gradient_reduction = percentage_reduction(a.gradient_mse, b.gradient_mse)
    value_delta = percentage_reduction(selected_value["value_mse"], selected_derivative["value_mse"])
    sources = {}
    for label, directory, names in (
        ("P0", args.p0_dir, ("closure_metrics.csv", "metrics.json")),
        ("P1", args.p1_dir, ("metrics.json", "sparse_v_budget.csv")),
        ("P2", args.p2_dir, ("metrics.json", "candidate_metrics.csv")),
    ):
        sources[label] = {"directory": str(directory.resolve()),
                          "files_sha256": {name: file_hash(directory / name) for name in names if (directory / name).exists()}}
    result = {
        "sources": sources,
        "P0": {
            "exact_discrete_path_max_rel_l2": float(closure.exact_discrete_path.max()),
            "baseline_data_gradient_rel_l2": closure.baseline_data_gradient.to_dict(),
            "symbolic_analytic_gradient_rel_l2": closure.symbolic_analytic_gradient.to_dict(),
            "model": p0_meta.get("model"), "provenance": p0_meta.get("provenance"),
            "version_status": "versioned" if p0_meta.get("provenance") else "legacy_unverified",
        },
        "P1": {
            "u_only_rank": p1["u_only"]["rank"], "u_only_nullity": p1["u_only"]["nullity"],
            "u_plus_v_rank": p1["u_plus_v"]["rank"], "u_plus_v_nullity": p1["u_plus_v"]["nullity"],
            "sampling_protocol": uniform_protocol,
            "three_v_points": {method: {key: float(row[key]) for key in ("rank_without_gauge", "coefficient_error_mean")}
                               for method, row in three.iterrows()},
        },
        "P2": {
            "evaluation_regime": regime, "evaluation_protocol": protocol,
            "selected_value_only": selected_value, "selected_analytic_derivative": selected_derivative,
            "selected_gradient_mse_reduction_percent": percentage_reduction(selected_value["gradient_mse"], selected_derivative["gradient_mse"]),
            "selected_value_mse_change_percent": -value_delta if value_delta is not None else None,
            "matched_complexity": matched_complexity,
            "matched_value_mse_change_percent": matched_value_change,
            "matched_gradient_mse_reduction_percent": matched_gradient_reduction,
            "matched_comparison_role": "post_selection_diagnostic_only",
        },
    }
    atomic_json(args.out_dir / "key_metrics.json", result)
    lines = ["# OASIS-SR preliminary experiments: P0-P2", "",
             "Each section describes the supplied result snapshot. Source paths and file hashes are saved in key_metrics.json.", "",
             "## P0: T = M + K closure", "",
             f"Version status: {result['P0']['version_status']}. Exact finite-step path closure: relative L2 <= {result['P0']['exact_discrete_path_max_rel_l2']:.3e}.",
             "This tests coordinate/sign bookkeeping; it does not establish physical sensitivity accuracy.", "",
             "| parameter | baseline gradient closure | selected symbolic gradient closure |", "|---|---:|---:|"]
    for parameter, row in closure.iterrows():
        lines.append(f"| {parameter} | {row.baseline_data_gradient:.4g} | {row.symbolic_analytic_gradient:.4g} |")
    model = p0_meta.get("model", {})
    lines.extend(["", f"Symbolic expression identity: {model.get('expression_sha256', 'unverified historical expression')}; complexity: {model.get('complexity', p0_meta.get('symbolic_complexity', 'unknown'))}.", "",
                  "## P1: differential observability", "",
                  f"u-only rank/nullity: {p1['u_only']['rank']}/{p1['u_only']['nullity']}; u+v: {p1['u_plus_v']['rank']}/{p1['u_plus_v']['nullity']}.",
                  f"Sampling protocol: {uniform_protocol}.",
                  ("Uniform rows span the full candidate domain separately at every budget."
                   if uniform_protocol == "full_domain_uniform_per_budget_v2" else
                   "Historical uniform-prefix sampling did not cover the domain at each budget. Rerun P1 before comparing sparse sampling methods."), "",
                  "Three complementary v observations:", "", "| method | rank without gauge | coefficient error |", "|---|---:|---:|"])
    for method, row in three.iterrows():
        lines.append(f"| {method} | {row.rank_without_gauge:.2f} | {row.coefficient_error_mean:.4g} |")
    lines.extend(["", "## P2: analytic derivative loss", "", protocol_note, "",
                  "| selection | complexity | value MSE | gradient MSE | gradient rel L2 |", "|---|---:|---:|---:|---:|"])
    for name, row in (("value-only (value criterion)", selected_value), ("analytic derivative (H1 criterion)", selected_derivative)):
        lines.append(f"| {name} | {row['complexity']} | {row['value_mse']:.4g} | {row['gradient_mse']:.4g} | {row['gradient_rel_l2']:.4g} |")
    lines.extend(["", f"On the stated evaluation data, gradient MSE reduction: {number(result['P2']['selected_gradient_mse_reduction_percent'], '.1f')}%; value MSE change: {number(result['P2']['selected_value_mse_change_percent'], '.1f')}%.", "",
                  f"Matched-complexity diagnostic (C={matched_complexity}): gradient MSE reduction {number(matched_gradient_reduction, '.1f')}%. This post-selection comparison is not used to choose an expression.", "",
                  "These empirical comparisons do not establish that the discovered expressions recover physical laws."])
    (args.out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summary written to {args.out_dir}")
    return result


if __name__ == "__main__":
    main()
