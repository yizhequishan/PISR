"""Audit every existing stream-function Pareto candidate by transport error.

This is a retrospective, zero-training audit.  It reads the saved Strategy A/B
PySR candidate tables, differentiates every candidate stream function, builds
the R/cx/cy transport fields, and compares them with the empirical transport
reference ``K_emp = S_total - M`` produced by Step 5.

The primary ranking metric is the R/u transport error

    epsilon_K = ||K_hat - K_emp||_2 / ||K_emp||_2.

Outputs are written below ``outputs_mgn/fd_check/psi_pareto_gradient_audit``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp


REPO = Path(__file__).resolve().parents[1]
DEFAULT_PSI_DIR = REPO / "outputs_mgn" / "comsol" / "eta_sym_psi"
DEFAULT_FD_DIR = REPO / "outputs_mgn" / "fd_check"
DEFAULT_OUT_DIR = DEFAULT_FD_DIR / "psi_pareto_gradient_audit"
DEFAULT_SELECTED_STRATEGY = "B"
DEFAULT_SELECTED_COMPLEXITY = 21


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--psi-dir", type=Path, default=DEFAULT_PSI_DIR)
    parser.add_argument("--fd-dir", type=Path, default=DEFAULT_FD_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--selected-strategy", default=DEFAULT_SELECTED_STRATEGY,
        help="Previously selected strategy used for the comparison baseline.",
    )
    parser.add_argument(
        "--selected-complexity", type=int, default=DEFAULT_SELECTED_COMPLEXITY,
        help="Previously selected complexity used for the comparison baseline.",
    )
    return parser.parse_args()


def ensure_array(value: Any, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(shape, float(array), dtype=float)
    return np.broadcast_to(array, shape).astype(float, copy=False)


def relative_l2(truth: np.ndarray, estimate: np.ndarray) -> float:
    denominator = float(np.linalg.norm(truth.ravel()))
    if denominator == 0.0:
        return float("nan")
    return float(np.linalg.norm((estimate - truth).ravel()) / denominator)


def correlation(truth: np.ndarray, estimate: np.ndarray) -> float:
    truth_flat = truth.ravel()
    estimate_flat = estimate.ravel()
    if np.std(truth_flat) == 0.0 or np.std(estimate_flat) == 0.0:
        return float("nan")
    return float(np.corrcoef(truth_flat, estimate_flat)[0, 1])


def evaluate(expression: sp.Expr, xi: sp.Symbol, eta: sp.Symbol,
             xi_values: np.ndarray, eta_values: np.ndarray) -> np.ndarray:
    function = sp.lambdify((xi, eta), expression, modules="numpy")
    with np.errstate(all="ignore"):
        values = ensure_array(function(xi_values, eta_values), xi_values.shape)
    if not np.all(np.isfinite(values)):
        count = int(np.size(values) - np.count_nonzero(np.isfinite(values)))
        raise ValueError(f"expression produced {count} non-finite values")
    return values


def candidate_fields(
    psi_text: str,
    xi_values: np.ndarray,
    eta_values: np.ndarray,
    diameter: float,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    xi, eta = sp.symbols("x_D y_D")
    psi = sp.sympify(psi_text)
    unsupported = psi.free_symbols - {xi, eta}
    if unsupported:
        names = ", ".join(sorted(str(symbol) for symbol in unsupported))
        raise ValueError(f"unsupported symbols: {names}")

    velocity_expr = {
        "u": sp.diff(psi, eta),
        "v": -sp.diff(psi, xi),
    }
    velocity: dict[str, np.ndarray] = {}
    transport: dict[str, dict[str, np.ndarray]] = {
        parameter: {} for parameter in ("R", "cx", "cy")
    }

    for component, expression in velocity_expr.items():
        d_dxi = sp.diff(expression, xi)
        d_deta = sp.diff(expression, eta)
        operator_expr = {
            "R": -(2.0 / diameter) * (xi * d_dxi + eta * d_deta),
            "cx": -(1.0 / diameter) * d_dxi,
            "cy": -(1.0 / diameter) * d_deta,
        }
        velocity[component] = evaluate(
            expression, xi, eta, xi_values, eta_values
        )
        for parameter, k_expression in operator_expr.items():
            transport[parameter][component] = evaluate(
                k_expression, xi, eta, xi_values, eta_values
            )

    return velocity, transport


def load_candidates(psi_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for strategy in ("A", "B"):
        path = psi_dir / f"results_strategy_{strategy.lower()}.csv"
        frame = pd.read_csv(path)
        frame.insert(0, "strategy", strategy)
        frame.insert(1, "source_file", path.name)
        frames.append(frame)
    candidates = pd.concat(frames, ignore_index=True)
    candidates["complexity"] = candidates["complexity"].astype(int)
    candidates["equation_index"] = candidates["equation_index"].astype(int)
    return candidates


def audit_candidate(
    row: pd.Series,
    xi_values: np.ndarray,
    eta_values: np.ndarray,
    diameter: float,
    fd_fields: dict[str, np.ndarray],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "strategy": row["strategy"],
        "equation_index": int(row["equation_index"]),
        "complexity": int(row["complexity"]),
        "original_mse_u": float(row["mse"]),
        "original_mse_v": float(row["mse_v"]),
        "original_combined_loss": float(row["combined_loss"]),
        "has_x_D": bool(row["has_x_D"]),
        "psi_sympy": str(row["psi_sympy"]),
        "status": "ok",
        "error": "",
    }
    try:
        velocity, transport = candidate_fields(
            result["psi_sympy"], xi_values, eta_values, diameter
        )
        for component in ("u", "v"):
            truth_field = fd_fields[f"{component}_base"]
            estimate_field = velocity[component]
            result[f"dense_mse_{component}"] = float(
                np.mean((estimate_field - truth_field) ** 2)
            )
            result[f"dense_rel_l2_{component}"] = relative_l2(
                truth_field, estimate_field
            )

        # A dimensionless balanced field score prevents the larger u energy
        # from completely hiding v while keeping the two components comparable.
        result["dense_balanced_field_score"] = (
            result["dense_rel_l2_u"] ** 2 + result["dense_rel_l2_v"] ** 2
        )

        for parameter in ("R", "cx", "cy"):
            for component in ("u", "v"):
                prefix = f"{parameter}_{component}"
                k_empirical = fd_fields[f"{prefix}_K_empirical"]
                s_total = fd_fields[f"{prefix}_S_total"]
                response = fd_fields[f"{prefix}_M"]
                k_candidate = transport[parameter][component]
                epsilon = relative_l2(k_empirical, k_candidate)
                kappa = float(
                    np.linalg.norm(k_empirical.ravel())
                    / np.linalg.norm(s_total.ravel())
                )
                closure = relative_l2(s_total, k_candidate + response)
                result[f"epsilon_K_{prefix}"] = epsilon
                result[f"kappa_{prefix}"] = kappa
                result[f"predicted_closure_{prefix}"] = epsilon * kappa
                result[f"actual_closure_{prefix}"] = closure
                result[f"closure_identity_error_{prefix}"] = abs(
                    epsilon * kappa - closure
                )
                result[f"K_correlation_{prefix}"] = correlation(
                    k_empirical, k_candidate
                )
    except Exception as exc:  # Preserve all candidates in the audit table.
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def pareto_mask(frame: pd.DataFrame, error_column: str) -> pd.Series:
    """Return candidates not dominated in (complexity, error)."""
    valid = frame[frame["status"] == "ok"].copy()
    mask = pd.Series(False, index=frame.index)
    for index, row in valid.iterrows():
        dominated = valid[
            (valid["complexity"] <= row["complexity"])
            & (valid[error_column] <= row[error_column])
            & (
                (valid["complexity"] < row["complexity"])
                | (valid[error_column] < row[error_column])
            )
        ]
        if dominated.empty:
            mask.loc[index] = True
    return mask


def candidate_id(row: pd.Series) -> str:
    return f"{row['strategy']}-E{int(row['equation_index'])}-C{int(row['complexity'])}"


def record(row: pd.Series, metric: str) -> dict[str, Any]:
    return {
        "candidate": candidate_id(row),
        "strategy": str(row["strategy"]),
        "equation_index": int(row["equation_index"]),
        "complexity": int(row["complexity"]),
        "metric": metric,
        "value": float(row[metric]),
        "epsilon_K_R_u": float(row["epsilon_K_R_u"]),
        "actual_closure_R_u": float(row["actual_closure_R_u"]),
        "original_mse_u": float(row["original_mse_u"]),
        "dense_balanced_field_score": float(row["dense_balanced_field_score"]),
        "psi_sympy": str(row["psi_sympy"]),
    }


def make_plots(frame: pd.DataFrame, selected_index: int, out_dir: Path) -> None:
    valid = frame[frame["status"] == "ok"].copy()
    selected = frame.loc[selected_index]
    best_epsilon = valid.loc[valid["epsilon_K_R_u"].idxmin()]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for strategy, marker in (("A", "o"), ("B", "s")):
        subset = valid[valid["strategy"] == strategy]
        scatter = axes[0].scatter(
            subset["original_mse_u"],
            100.0 * subset["epsilon_K_R_u"],
            c=subset["complexity"],
            cmap="viridis",
            marker=marker,
            s=55,
            alpha=0.85,
            label=f"Strategy {strategy}",
        )
        axes[1].plot(
            subset["complexity"],
            100.0 * subset["epsilon_K_R_u"],
            marker=marker,
            linewidth=1.2,
            alpha=0.8,
            label=f"Strategy {strategy}",
        )

    axes[0].set_xscale("log")
    axes[0].set_xlabel("original field MSE (u)")
    axes[0].set_ylabel("epsilon_K for R/u [%]")
    axes[0].set_title("Field fit does not define derivative ranking")
    axes[0].set_ylim(5, 105)
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    colorbar = fig.colorbar(scatter, ax=axes[0])
    colorbar.set_label("complexity")

    axes[1].set_xlabel("complexity")
    axes[1].set_ylabel("epsilon_K for R/u [%]")
    axes[1].set_title("Derivative-aware candidate audit")
    axes[1].set_ylim(5, 105)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    for axis, row, color, label, offset in (
        (axes[0], selected, "#d62728", "recorded psi-B", (-2, 12)),
        (axes[0], best_epsilon, "#1f77b4", "min epsilon_K", (8, -18)),
    ):
        axis.scatter(
            row["original_mse_u"], 100.0 * row["epsilon_K_R_u"],
            s=150, facecolors="none", edgecolors=color, linewidths=2.2,
        )
        axis.annotate(
            label,
            (row["original_mse_u"], 100.0 * row["epsilon_K_R_u"]),
            xytext=offset, textcoords="offset points", color=color,
        )
    for row, color, label, offset in (
        (selected, "#d62728", "recorded psi-B", (8, 10)),
        (best_epsilon, "#1f77b4", "min epsilon_K", (8, -18)),
    ):
        axes[1].scatter(
            row["complexity"], 100.0 * row["epsilon_K_R_u"],
            s=150, facecolors="none", edgecolors=color, linewidths=2.2,
        )
        axes[1].annotate(
            label,
            (row["complexity"], 100.0 * row["epsilon_K_R_u"]),
            xytext=offset, textcoords="offset points", color=color,
        )

    fig.savefig(out_dir / "01_mse_vs_epsilon_K.png", dpi=190)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, limit: int = 10) -> list[str]:
    columns = [
        "candidate", "complexity", "original_mse_u", "epsilon_K_R_u",
        "actual_closure_R_u", "K_correlation_R_u",
    ]
    rows = frame.nsmallest(limit, "epsilon_K_R_u").copy()
    if "candidate" not in rows.columns:
        rows.insert(0, "candidate", rows.apply(candidate_id, axis=1))
    lines = [
        "| candidate | complexity | original u MSE | epsilon_K R/u | closure R/u | K corr R/u |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in rows[columns].iterrows():
        lines.append(
            f"| {row['candidate']} | {int(row['complexity'])} | "
            f"{row['original_mse_u']:.6g} | {row['epsilon_K_R_u']:.4f} | "
            f"{row['actual_closure_R_u']:.4f} | {row['K_correlation_R_u']:.4f} |"
        )
    return lines


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(args.psi_dir)
    fields_path = args.fd_dir / "data_gradient_closure_fields.npz"
    with np.load(fields_path) as loaded:
        x_d = loaded["x_D"].copy()
        y_d = loaded["y_D"].copy()
        fd_fields = {name: loaded[name].copy() for name in loaded.files}
    with np.load(args.fd_dir / "fd_sensitivity_fields.npz") as loaded:
        fd_fields["u_base"] = loaded["u_base"].copy()
        fd_fields["v_base"] = loaded["v_base"].copy()

    with (args.fd_dir / "decomposition_run_info.json").open(
        "r", encoding="utf-8"
    ) as handle:
        run_info = json.load(handle)
    diameter = float(run_info["D0_m"])

    xx, yy = np.meshgrid(x_d, y_d, indexing="ij")
    rows = [
        audit_candidate(row, xx, yy, diameter, fd_fields)
        for _, row in candidates.iterrows()
    ]
    audit = pd.DataFrame(rows)
    audit["derivative_pareto_R_u"] = pareto_mask(audit, "epsilon_K_R_u")
    audit.insert(0, "candidate", audit.apply(candidate_id, axis=1))
    audit.to_csv(args.out_dir / "candidate_gradient_audit.csv", index=False)
    audit[audit["derivative_pareto_R_u"]].sort_values(
        ["complexity", "epsilon_K_R_u"]
    ).to_csv(args.out_dir / "derivative_pareto_R_u.csv", index=False)

    valid = audit[audit["status"] == "ok"].copy()
    if valid.empty:
        raise RuntimeError("Every candidate failed evaluation; inspect audit CSV")
    selected_rows = valid[
        (valid["strategy"].str.upper() == args.selected_strategy.upper())
        & (valid["complexity"] == args.selected_complexity)
    ]
    if len(selected_rows) != 1:
        raise ValueError(
            "Expected exactly one recorded selected candidate, found "
            f"{len(selected_rows)}"
        )
    selected_index = int(selected_rows.index[0])
    selected = valid.loc[selected_index]
    best_epsilon = valid.loc[valid["epsilon_K_R_u"].idxmin()]
    best_original_mse = valid.loc[valid["original_mse_u"].idxmin()]
    best_dense_balanced = valid.loc[
        valid["dense_balanced_field_score"].idxmin()
    ]
    spearman_mse_epsilon = float(
        valid["original_mse_u"].corr(valid["epsilon_K_R_u"], method="spearman")
    )

    summary = {
        "candidate_count": int(len(audit)),
        "valid_candidate_count": int(len(valid)),
        "failed_candidate_count": int((audit["status"] != "ok").sum()),
        "grid_shape": [int(x_d.size), int(y_d.size)],
        "diameter_m": diameter,
        "primary_metric": "epsilon_K_R_u",
        "recorded_selected": record(selected, "epsilon_K_R_u"),
        "min_epsilon_K_R_u": record(best_epsilon, "epsilon_K_R_u"),
        "min_original_mse_u": record(best_original_mse, "original_mse_u"),
        "min_dense_balanced_field_score": record(
            best_dense_balanced, "dense_balanced_field_score"
        ),
        "argmin_epsilon_differs_from_recorded": bool(
            best_epsilon["candidate"] != selected["candidate"]
        ),
        "epsilon_improvement_factor": float(
            selected["epsilon_K_R_u"] / best_epsilon["epsilon_K_R_u"]
        ),
        "epsilon_improvement_percentage_points": float(
            100.0
            * (selected["epsilon_K_R_u"] - best_epsilon["epsilon_K_R_u"])
        ),
        "closure_improvement_percentage_points": float(
            100.0
            * (
                selected["actual_closure_R_u"]
                - best_epsilon["actual_closure_R_u"]
            )
        ),
        "best_epsilon_original_mse_ratio_vs_recorded": float(
            best_epsilon["original_mse_u"] / selected["original_mse_u"]
        ),
        "spearman_original_mse_u_vs_epsilon_K_R_u": spearman_mse_epsilon,
        "note": (
            "Retrospective in-sample audit: K_emp=S-M is used for both ranking "
            "and reporting. Independent held-out validation is still required."
        ),
    }
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    make_plots(audit, selected_index, args.out_dir)

    report_lines = [
        "# Stream-function Pareto gradient audit",
        "",
        "This is a zero-training retrospective audit of every saved Strategy A/B "
        "candidate. The primary metric is `epsilon_K_R_u` against `K_emp=S-M` "
        f"on the current `{x_d.size} x {y_d.size}` FD grid.",
        "",
        "## Main result",
        "",
        f"- Recorded psi-B: `{candidate_id(selected)}`, epsilon_K="
        f"`{selected['epsilon_K_R_u']:.4f}`, closure="
        f"`{selected['actual_closure_R_u']:.4f}`.",
        f"- Minimum epsilon_K: `{candidate_id(best_epsilon)}`, epsilon_K="
        f"`{best_epsilon['epsilon_K_R_u']:.4f}`, closure="
        f"`{best_epsilon['actual_closure_R_u']:.4f}`.",
        f"- Minimum original u-MSE candidate: `{candidate_id(best_original_mse)}`, "
        f"epsilon_K=`{best_original_mse['epsilon_K_R_u']:.4f}`.",
        f"- `argmin epsilon_K != recorded selection`: "
        f"`{summary['argmin_epsilon_differs_from_recorded']}`.",
        f"- Epsilon improvement: "
        f"`{summary['epsilon_improvement_percentage_points']:.2f}` percentage "
        f"points (`{summary['epsilon_improvement_factor']:.2f}x`).",
        f"- The derivative-best candidate has "
        f"`{summary['best_epsilon_original_mse_ratio_vs_recorded']:.2f}x` the "
        "recorded candidate's u-MSE, so this is a direct pairwise ranking "
        "inversion.",
        f"- Across all candidates, Spearman(original u-MSE, epsilon_K)="
        f"`{spearman_mse_epsilon:.3f}`; MSE is broadly informative but not a "
        "sufficient final selection metric.",
        "",
        "## Best candidates by R/u transport error",
        "",
        *markdown_table(valid),
        "",
        "## Interpretation boundary",
        "",
        "The same empirical transport field is used to rank and report the "
        "candidates. This establishes whether the saved search already contains "
        "a derivative-better equation, but it is not an out-of-sample validation "
        "of derivative-aware selection.",
        "",
        "![MSE versus epsilon_K](01_mse_vs_epsilon_K.png)",
        "",
    ]
    (args.out_dir / "REPORT.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
