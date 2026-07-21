"""Step 6: select and validate the symbolic radius-response stream function.

Run ``step2_pisr_psi.py`` on ``fd_check/phi_R_input`` first.  This script then
selects a Pareto candidate using a component-balanced derivative error, derives

    M_R^u = partial_eta phi_R,   M_R^v = -partial_xi phi_R,

and tests the fully symbolic sensitivity ``K_R[psi_0] + M_R[phi_R]`` against
the fixed-physical-point finite difference produced by Step 5.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp


REPO = Path(__file__).resolve().parent
FD_DIR = REPO / "outputs_mgn" / "fd_check"
PHI_DIR = FD_DIR / "phi_R_pysr"
RESULTS_PATH = PHI_DIR / "results_strategy_b.csv"
FIELDS_PATH = FD_DIR / "fd_sensitivity_fields.npz"
BASE_RESULTS = REPO / "outputs_mgn" / "comsol" / "eta_sym_psi" / "results_strategy_b.csv"
BASE_COMPLEXITY = 21


def ensure_array(value: object, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(shape, float(array))
    return np.broadcast_to(array, shape).astype(float, copy=False)


def field_metrics(truth: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    residual = estimate - truth
    truth_flat = truth.ravel()
    estimate_flat = estimate.ravel()
    truth_centered = truth_flat - truth_flat.mean()
    estimate_centered = estimate_flat - estimate_flat.mean()
    denominator = np.linalg.norm(truth_centered) * np.linalg.norm(estimate_centered)
    correlation = (
        float(np.dot(truth_centered, estimate_centered) / denominator)
        if denominator > 0
        else float("nan")
    )
    return {
        "truth_rms": float(np.sqrt(np.mean(truth**2))),
        "estimate_rms": float(np.sqrt(np.mean(estimate**2))),
        "truth_mean_abs": float(np.mean(np.abs(truth))),
        "estimate_mean_abs": float(np.mean(np.abs(estimate))),
        "rel_l2": float(np.linalg.norm(residual) / max(np.linalg.norm(truth), 1e-12)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "max_abs_error": float(np.max(np.abs(residual))),
        "correlation": correlation,
    }


def evaluate_expression(
    expression: sp.Expr, xi_grid: np.ndarray, eta_grid: np.ndarray
) -> np.ndarray:
    xi, eta = sp.symbols("x_D y_D")
    function = sp.lambdify((xi, eta), expression, modules="numpy")
    return ensure_array(function(xi_grid, eta_grid), xi_grid.shape)


def candidate_table(results: pd.DataFrame, m_u: np.ndarray, m_v: np.ndarray) -> pd.DataFrame:
    u_scale = float(np.mean(m_u**2))
    v_scale = float(np.mean(m_v**2))
    table = results.copy()
    table["u_rel_l2"] = np.sqrt(table["mse"] / u_scale)
    table["v_rel_l2"] = np.sqrt(table["mse_v"] / v_scale)
    table["balanced_score"] = table["mse"] / u_scale + table["mse_v"] / v_scale
    table["selection"] = ""
    selected_index = table["balanced_score"].idxmin()
    table.loc[selected_index, "selection"] = "selected_min_balanced_score"
    return table


def plot_triplets(
    path: Path,
    x_d: np.ndarray,
    y_d: np.ndarray,
    rows: list[tuple[str, np.ndarray, np.ndarray]],
    title: str,
) -> None:
    fig, axes = plt.subplots(len(rows), 3, figsize=(16, 4.6 * len(rows)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row_index, (component, truth, prediction) in enumerate(rows):
        residual = prediction - truth
        main_limit = max(float(np.max(np.abs(truth))), float(np.max(np.abs(prediction))), 1e-12)
        residual_limit = max(float(np.max(np.abs(residual))), 1e-12)
        for column_index, (label, field, limit) in enumerate(
            (
                ("FD target", truth, main_limit),
                ("symbolic", prediction, main_limit),
                ("residual", residual, residual_limit),
            )
        ):
            axis = axes[row_index, column_index]
            image = axis.pcolormesh(
                x_d,
                y_d,
                field.T,
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                shading="auto",
            )
            axis.set_aspect("equal")
            axis.set_xlabel("xi")
            axis.set_ylabel("eta")
            metrics = field_metrics(truth, prediction)
            axis.set_title(
                f"{component}: {label}\nrel L2={metrics['rel_l2']:.3f}, r={metrics['correlation']:.3f}"
            )
            fig.colorbar(image, ax=axis, shrink=0.84)
    fig.suptitle(title)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> dict:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {RESULTS_PATH}. Run the phi_R PySR search before this analysis."
        )

    with np.load(FIELDS_PATH) as loaded:
        fields = {name: loaded[name].copy() for name in loaded.files}
    x_d = fields["x_D"]
    y_d = fields["y_D"]
    xx, yy = np.meshgrid(x_d, y_d, indexing="ij")
    m_target = {"u": fields["M_R_u"], "v": fields["M_R_v"]}
    total_target = {"u": fields["S_total_R_u"], "v": fields["S_total_R_v"]}
    k_psi = {"u": fields["K_R_u_psi"], "v": fields["K_R_v_psi"]}
    k_data = {
        "u": fields["K_R_u_data"],
        "v": fields["K_R_v_data"],
    }

    raw_results = pd.read_csv(RESULTS_PATH)
    candidates = candidate_table(raw_results, m_target["u"], m_target["v"])
    selected = candidates.loc[candidates["balanced_score"].idxmin()]
    candidates.to_csv(PHI_DIR / "candidate_metrics.csv", index=False)

    xi, eta = sp.symbols("x_D y_D")
    phi = sp.sympify(str(selected["psi_sympy"]))
    m_u_expression = sp.diff(phi, eta)
    m_v_expression = -sp.diff(phi, xi)
    symbolic_divergence = sp.simplify(
        sp.diff(m_u_expression, xi) + sp.diff(m_v_expression, eta)
    )
    m_symbolic = {
        "u": evaluate_expression(m_u_expression, xx, yy),
        "v": evaluate_expression(m_v_expression, xx, yy),
    }
    total_symbolic = {
        component: k_psi[component] + m_symbolic[component]
        for component in ("u", "v")
    }
    hybrid_data_k_symbolic_m = {
        component: k_data[component] + m_symbolic[component]
        for component in ("u", "v")
    }

    base_results = pd.read_csv(BASE_RESULTS)
    base_row = base_results[base_results["complexity"] == BASE_COMPLEXITY].iloc[0]
    psi0 = sp.sympify(str(base_row["psi_sympy"]))
    u0 = float(fields["U0"])
    u_prime = float(fields["Uprime"])
    phi_dimensional = sp.expand(u0 * phi + u_prime * psi0)
    dimensional_total = {
        "u": fields["S_total_R_u_dim"],
        "v": fields["S_total_R_v_dim"],
    }
    dimensional_symbolic = {
        "u": u0 * (k_psi["u"] + m_symbolic["u"]) + u_prime * fields["u_base"],
        "v": u0 * (k_psi["v"] + m_symbolic["v"]) + u_prime * fields["v_base"],
    }

    metric_rows = []
    for component in ("u", "v"):
        metric_rows.append(
            {
                "quantity": "intrinsic_response_M",
                "component": component,
                "units": "1/m",
                **field_metrics(m_target[component], m_symbolic[component]),
            }
        )
        metric_rows.append(
            {
                "quantity": "total_symbolic_K_plus_M",
                "component": component,
                "units": "1/m",
                **field_metrics(total_target[component], total_symbolic[component]),
            }
        )
        metric_rows.append(
            {
                "quantity": "hybrid_data_K_plus_symbolic_M",
                "component": component,
                "units": "1/m",
                **field_metrics(
                    total_target[component],
                    hybrid_data_k_symbolic_m[component],
                ),
            }
        )
        metric_rows.append(
            {
                "quantity": "total_symbolic_dimensional_with_Uprime",
                "component": component,
                "units": "m/s/m",
                **field_metrics(dimensional_total[component], dimensional_symbolic[component]),
            }
        )
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(PHI_DIR / "phi_R_validation_metrics.csv", index=False)

    dmu_dxi = np.gradient(m_target["u"], x_d, axis=0, edge_order=2)
    dmv_deta = np.gradient(m_target["v"], y_d, axis=1, edge_order=2)
    target_divergence = dmu_dxi + dmv_deta
    dmu_sym_dxi = np.gradient(m_symbolic["u"], x_d, axis=0, edge_order=2)
    dmv_sym_deta = np.gradient(m_symbolic["v"], y_d, axis=1, edge_order=2)
    symbolic_grid_divergence = dmu_sym_dxi + dmv_sym_deta

    np.savez_compressed(
        PHI_DIR / "phi_R_symbolic_fields.npz",
        x_D=x_d,
        y_D=y_d,
        M_R_u_target=m_target["u"],
        M_R_v_target=m_target["v"],
        M_R_u_symbolic=m_symbolic["u"],
        M_R_v_symbolic=m_symbolic["v"],
        S_total_R_u_target=total_target["u"],
        S_total_R_v_target=total_target["v"],
        S_total_R_u_symbolic=total_symbolic["u"],
        S_total_R_v_symbolic=total_symbolic["v"],
        S_total_R_u_hybrid_data_K_symbolic_M=hybrid_data_k_symbolic_m["u"],
        S_total_R_v_hybrid_data_K_symbolic_M=hybrid_data_k_symbolic_m["v"],
        target_divergence=target_divergence,
        symbolic_grid_divergence=symbolic_grid_divergence,
    )

    plot_triplets(
        PHI_DIR / "06_phi_R_response_fit.png",
        x_d,
        y_d,
        [
            ("M_R^u", m_target["u"], m_symbolic["u"]),
            ("M_R^v", m_target["v"], m_symbolic["v"]),
        ],
        "Symbolic response stream function: derivative-field fit",
    )
    plot_triplets(
        PHI_DIR / "07_total_symbolic_sensitivity.png",
        x_d,
        y_d,
        [
            ("S_total^u", total_target["u"], total_symbolic["u"]),
            ("S_total^v", total_target["v"], total_symbolic["v"]),
        ],
        "Fully symbolic sensitivity: K_R[psi_0] + M_R[phi_R]",
    )
    plot_triplets(
        PHI_DIR / "09_hybrid_data_K_symbolic_M.png",
        x_d,
        y_d,
        [
            (
                "S_total^u",
                total_target["u"],
                hybrid_data_k_symbolic_m["u"],
            ),
            (
                "S_total^v",
                total_target["v"],
                hybrid_data_k_symbolic_m["v"],
            ),
        ],
        "Error isolation: base-CFD data K_R + symbolic M_R[phi_R]",
    )

    result = {
        "selection": {
            "strategy": "B",
            "complexity": int(selected["complexity"]),
            "balanced_score": float(selected["balanced_score"]),
            "u_rel_l2_from_candidate_table": float(selected["u_rel_l2"]),
            "v_rel_l2_from_candidate_table": float(selected["v_rel_l2"]),
        },
        "phi_R_sympy": str(phi),
        "phi_R_latex": sp.latex(phi),
        "M_R_u_sympy": str(m_u_expression),
        "M_R_v_sympy": str(m_v_expression),
        "symbolic_divergence": str(symbolic_divergence),
        "phi_R_dimensional_sympy": str(phi_dimensional),
        "U0": u0,
        "Uprime": u_prime,
        "target_divergence_rms_finite_difference": float(
            np.sqrt(np.mean(target_divergence**2))
        ),
        "symbolic_grid_divergence_rms": float(
            np.sqrt(np.mean(symbolic_grid_divergence**2))
        ),
        "metrics": metric_rows,
    }
    with (PHI_DIR / "phi_R_results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    u_m = metrics_frame[
        (metrics_frame["quantity"] == "intrinsic_response_M")
        & (metrics_frame["component"] == "u")
    ].iloc[0]
    v_m = metrics_frame[
        (metrics_frame["quantity"] == "intrinsic_response_M")
        & (metrics_frame["component"] == "v")
    ].iloc[0]
    u_total = metrics_frame[
        (metrics_frame["quantity"] == "total_symbolic_K_plus_M")
        & (metrics_frame["component"] == "u")
    ].iloc[0]
    v_total = metrics_frame[
        (metrics_frame["quantity"] == "total_symbolic_K_plus_M")
        & (metrics_frame["component"] == "v")
    ].iloc[0]
    u_hybrid = metrics_frame[
        (metrics_frame["quantity"] == "hybrid_data_K_plus_symbolic_M")
        & (metrics_frame["component"] == "u")
    ].iloc[0]
    v_hybrid = metrics_frame[
        (metrics_frame["quantity"] == "hybrid_data_K_plus_symbolic_M")
        & (metrics_frame["component"] == "v")
    ].iloc[0]
    report = [
        "# Radius-response stream-function experiment",
        "",
        "## Selected symbolic response",
        "",
        f"Strategy B, complexity {int(selected['complexity'])}, selected by the minimum "
        "component-balanced derivative score over the PySR Pareto front.",
        "",
        "```text",
        f"phi_R(xi, eta) = {phi}",
        "```",
        "",
        "The response velocity is recovered structurally as",
        "`M_R^u = d(phi_R)/deta`, `M_R^v = -d(phi_R)/dxi`. Its symbolic "
        f"divergence simplifies exactly to `{symbolic_divergence}`.",
        "",
        "## Validation",
        "",
        "| quantity | component | rel L2 | RMSE | correlation |",
        "|---|---|---:|---:|---:|",
        f"| intrinsic response M | u | {u_m['rel_l2']:.4f} | {u_m['rmse']:.4f} | {u_m['correlation']:.4f} |",
        f"| intrinsic response M | v | {v_m['rel_l2']:.4f} | {v_m['rmse']:.4f} | {v_m['correlation']:.4f} |",
        f"| total K[psi0] + M[phi_R] | u | {u_total['rel_l2']:.4f} | {u_total['rmse']:.4f} | {u_total['correlation']:.4f} |",
        f"| total K[psi0] + M[phi_R] | v | {v_total['rel_l2']:.4f} | {v_total['rmse']:.4f} | {v_total['correlation']:.4f} |",
        f"| hybrid K[data] + M[phi_R] | u | {u_hybrid['rel_l2']:.4f} | {u_hybrid['rmse']:.4f} | {u_hybrid['correlation']:.4f} |",
        f"| hybrid K[data] + M[phi_R] | v | {v_hybrid['rel_l2']:.4f} | {v_hybrid['rmse']:.4f} | {v_hybrid['correlation']:.4f} |",
        "",
        "The total-error row includes both the response-model error and the base psi-B "
        "transport-derivative error. The hybrid row replaces the latter with the matched-step "
        "base-CFD data gradient, thereby isolating the response-symbolization error.",
        "",
        "For dimensional velocity, define `phi_R_dim = U0*phi_R + Uprime*psi0`; then "
        "`du/dR = U0*K_R[psi0] + d(phi_R_dim)/deta`, which contains the inlet-speed "
        "term explicitly and remains divergence-free by construction.",
    ]
    (PHI_DIR / "PHI_R_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("=== Selected phi_R ===")
    print(f"complexity={int(selected['complexity'])}")
    print(f"phi_R = {phi}")
    print(f"symbolic divergence = {symbolic_divergence}")
    print("\n=== Validation metrics ===")
    print(metrics_frame.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(f"\nOutputs: {PHI_DIR}")
    return result


if __name__ == "__main__":
    main()
