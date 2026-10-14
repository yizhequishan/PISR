"""P0: close total, material, and coordinate derivatives on existing CFD fields.

The finite-difference archive stores each perturbed case on its own normalized
coordinates.  Those centered differences are therefore material/reference
responses M = d ubar / dp |_(xi, eta).  This experiment additionally evaluates
the perturbed fields at the same *physical* points to form total derivatives T,
and then checks T = M + K using three versions of the coordinate term K:

1. an exact finite-step path decomposition (bookkeeping check),
2. the gradient of a spline through the baseline CFD field, and
3. the analytic gradient of the selected symbolic expression.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
from scipy.interpolate import RectBivariateSpline


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pipeline_contracts import selected_model, verify_fd_model, archive_provenance
DEFAULT_INPUT = REPO / "outputs_mgn" / "fd_check" / "fd_sensitivity_fields.npz"
DEFAULT_SYMBOLIC = REPO / "outputs_mgn" / "comsol" / "eta_sym_psi" / "sensitivity_results.csv"
DEFAULT_OUT = REPO / "outputs" / "p0_derivative_closure"

D0 = 0.100
CX0 = 0.200
CY0 = 0.200

# (finite-difference step, plus field/D/cx/cy, minus field/D/cx/cy)
PARAMETERS = {
    "R": (
        0.001,
        ("u_Rp", 0.102, 0.200, 0.200),
        ("u_Rn", 0.098, 0.200, 0.200),
    ),
    "cx": (
        0.002,
        ("u_cxp", 0.100, 0.202, 0.200),
        ("u_cxn", 0.100, 0.198, 0.200),
    ),
    "cy": (
        0.002,
        ("u_cyp", 0.100, 0.200, 0.202),
        ("u_cyn", 0.100, 0.200, 0.198),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--psi-dir", type=Path, default=DEFAULT_SYMBOLIC.parent)
    parser.add_argument("--symbolic-csv", type=Path, default=None, help="Optional explicit historical diagnostic table; requires --complexity")
    parser.add_argument("--strategy", default="B")
    parser.add_argument("--complexity", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if (args.symbolic_csv is None) != (args.complexity is None):
        parser.error("Use --symbolic-csv and --complexity together for an explicit historical diagnostic")
    return args


def eval_spline(
    spline: RectBivariateSpline, xq: np.ndarray, yq: np.ndarray
) -> np.ndarray:
    return spline.ev(xq.ravel(), yq.ravel()).reshape(xq.shape)


def mapped_coordinates(
    xi: np.ndarray, eta: np.ndarray, *, D: float, cx: float, cy: float
) -> tuple[np.ndarray, np.ndarray]:
    physical_x = CX0 + D0 * xi
    physical_y = CY0 + D0 * eta
    return (physical_x - cx) / D, (physical_y - cy) / D


def rel_l2(reference: np.ndarray, estimate: np.ndarray) -> float:
    denom = np.linalg.norm(reference)
    return float(np.linalg.norm(reference - estimate) / max(denom, 1e-15))


def field_metrics(
    total: np.ndarray, estimate: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    a = total[mask].ravel()
    b = estimate[mask].ravel()
    if np.std(a) > 0 and np.std(b) > 0:
        correlation = float(np.corrcoef(a, b)[0, 1])
    else:
        correlation = float("nan")
    residual = a - b
    return {
        "rel_l2": rel_l2(a, b),
        "correlation": correlation,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "max_abs": float(np.max(np.abs(residual))),
        "sign_agreement": float(np.mean(np.sign(a) == np.sign(b))),
    }


def load_symbolic_gradients(
    csv_path: Path,
    strategy: str,
    complexity: int,
    xi_grid: np.ndarray,
    eta_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    frame = pd.read_csv(csv_path)
    selected = frame[
        (frame["strategy"] == strategy) & (frame["complexity"] == complexity)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one symbolic row for strategy={strategy}, complexity={complexity}; "
            f"found {len(selected)}"
        )
    expression_text = str(selected.iloc[0]["sympy"])
    xi_symbol, eta_symbol = sp.symbols("x_D y_D")
    expression = sp.sympify(expression_text)
    dxi = sp.lambdify(
        (xi_symbol, eta_symbol), sp.diff(expression, xi_symbol), modules="numpy"
    )
    deta = sp.lambdify(
        (xi_symbol, eta_symbol), sp.diff(expression, eta_symbol), modules="numpy"
    )
    grad_xi = np.broadcast_to(
        np.asarray(dxi(xi_grid, eta_grid), dtype=float), xi_grid.shape
    ).copy()
    grad_eta = np.broadcast_to(
        np.asarray(deta(xi_grid, eta_grid), dtype=float), eta_grid.shape
    ).copy()
    return grad_xi, grad_eta, expression_text


def coordinate_term(
    parameter: str,
    xi_grid: np.ndarray,
    eta_grid: np.ndarray,
    grad_xi: np.ndarray,
    grad_eta: np.ndarray,
) -> np.ndarray:
    if parameter == "R":
        return -(2.0 / D0) * (xi_grid * grad_xi + eta_grid * grad_eta)
    if parameter == "cx":
        return -(1.0 / D0) * grad_xi
    if parameter == "cy":
        return -(1.0 / D0) * grad_eta
    raise KeyError(parameter)


def write_report(out_dir: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# P0 derivative-closure experiment",
        "",
        "The exact discrete path decomposition checks coordinate/sign bookkeeping. "
        "The data-gradient and symbolic-gradient rows check the first-order chain rule "
        "with progressively stronger modeling assumptions.",
        "",
        "| parameter | K source | rel L2 | correlation | RMSE | sign agreement |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {parameter} | {k_source} | {rel_l2:.6g} | {correlation:.6g} | "
            "{rmse:.6g} | {sign_agreement:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Interpretation: exact-path closure should be at floating-point precision. "
            "A larger data-gradient error measures finite-step/interpolation/nonlinear "
            "effects; the additional symbolic-gradient gap measures derivative error in "
            "the selected symbolic surrogate.",
        ]
    )
    (out_dir / "P0_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global D0, CX0, CY0, PARAMETERS
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.input, allow_pickle=False) as loaded:
        archive = {name: loaded[name].copy() for name in loaded.files}
    required = {"x_D", "y_D", "u_base", "S_R", "S_cx", "S_cy",
                "u_Rp", "u_Rn", "u_cxp", "u_cxn", "u_cyp", "u_cyn"}
    missing = required - archive.keys()
    if missing:
        raise ValueError(f"P0 needs the Step 5 v2 reference-field archive; missing {sorted(missing)}. Rerun Step 5.")
    provenance = archive_provenance(archive)
    geometry = provenance["cases"]["base"]
    D0, CX0, CY0 = geometry["D"], geometry["cx"], geometry["cy"]
    PARAMETERS = {}
    for parameter, prefix in (("R", "R"), ("cx", "cx"), ("cy", "cy")):
        plus, minus = provenance["cases"][parameter + "+"], provenance["cases"][parameter + "-"]
        PARAMETERS[parameter] = (provenance["parameter_steps"][parameter],
            (f"u_{prefix}p", plus["D"], plus["cx"], plus["cy"]),
            (f"u_{prefix}n", minus["D"], minus["cx"], minus["cy"]))
    xi = archive["x_D"]
    eta = archive["y_D"]
    xi_grid, eta_grid = np.meshgrid(xi, eta, indexing="ij")

    base_spline = RectBivariateSpline(xi, eta, archive["u_base"], kx=3, ky=3, s=0)
    base_grad_xi = base_spline(xi, eta, dx=1, dy=0)
    base_grad_eta = base_spline(xi, eta, dx=0, dy=1)
    if args.symbolic_csv is not None:
        symbolic_grad_xi, symbolic_grad_eta, symbolic_expression = load_symbolic_gradients(
            args.symbolic_csv, args.strategy, args.complexity, xi_grid, eta_grid)
        model = {"selection_role": "explicit_historical_diagnostic", "complexity": args.complexity}
    else:
        model = selected_model(args.psi_dir, args.strategy)
        verify_fd_model(archive, model)
        symbolic_expression = model["sympy"]
        xi_symbol, eta_symbol = sp.symbols("x_D y_D")
        expression = sp.sympify(symbolic_expression)
        symbolic_grad_xi = np.broadcast_to(sp.lambdify((xi_symbol, eta_symbol), sp.diff(expression, xi_symbol), "numpy")(xi_grid, eta_grid), xi_grid.shape).copy()
        symbolic_grad_eta = np.broadcast_to(sp.lambdify((xi_symbol, eta_symbol), sp.diff(expression, eta_symbol), "numpy")(xi_grid, eta_grid), xi_grid.shape).copy()

    summary_rows: list[dict[str, object]] = []
    saved_fields: dict[str, np.ndarray] = {"x_D": xi, "y_D": eta}
    plot_rows: list[tuple[str, list[np.ndarray]]] = []

    for parameter, (step, plus, minus) in PARAMETERS.items():
        mapped_values: list[np.ndarray] = []
        mapped_base_values: list[np.ndarray] = []
        reference_values: list[np.ndarray] = []
        valid = np.ones_like(xi_grid, dtype=bool)

        for field_name, D, cx, cy in (plus, minus):
            mapped_xi, mapped_eta = mapped_coordinates(
                xi_grid, eta_grid, D=D, cx=cx, cy=cy
            )
            valid &= (
                (mapped_xi >= xi.min())
                & (mapped_xi <= xi.max())
                & (mapped_eta >= eta.min())
                & (mapped_eta <= eta.max())
            )
            field_spline = RectBivariateSpline(
                xi, eta, archive[field_name], kx=3, ky=3, s=0
            )
            mapped_values.append(eval_spline(field_spline, mapped_xi, mapped_eta))
            mapped_base_values.append(eval_spline(base_spline, mapped_xi, mapped_eta))
            reference_values.append(np.asarray(archive[field_name], dtype=float))

        total = (mapped_values[0] - mapped_values[1]) / (2.0 * step)
        material = np.asarray(archive[f"S_{parameter}"], dtype=float)

        # Exact algebraic remainder for this finite step.  It is the coordinate
        # part along the two perturbed fields, not yet a baseline linearization.
        coordinate_path = (
            (mapped_values[0] - reference_values[0])
            + (reference_values[1] - mapped_values[1])
        ) / (2.0 * step)

        # A matched-step coordinate difference through the baseline field is
        # retained as a diagnostic next to the analytic baseline gradient.
        coordinate_base_fd = (mapped_base_values[0] - mapped_base_values[1]) / (
            2.0 * step
        )
        coordinate_data = coordinate_term(
            parameter, xi_grid, eta_grid, base_grad_xi, base_grad_eta
        )
        coordinate_symbolic = coordinate_term(
            parameter, xi_grid, eta_grid, symbolic_grad_xi, symbolic_grad_eta
        )

        variants = {
            "exact_discrete_path": coordinate_path,
            "baseline_matched_fd": coordinate_base_fd,
            "baseline_data_gradient": coordinate_data,
            "symbolic_analytic_gradient": coordinate_symbolic,
        }
        for source, coordinate in variants.items():
            metrics = field_metrics(total, material + coordinate, valid)
            summary_rows.append(
                {
                    "parameter": parameter,
                    "k_source": source,
                    "valid_fraction": float(np.mean(valid)),
                    **metrics,
                }
            )

        saved_fields[f"{parameter}_mask"] = valid
        saved_fields[f"{parameter}_T"] = total
        saved_fields[f"{parameter}_M"] = material
        saved_fields[f"{parameter}_K_path"] = coordinate_path
        saved_fields[f"{parameter}_K_data"] = coordinate_data
        saved_fields[f"{parameter}_K_symbolic"] = coordinate_symbolic
        saved_fields[f"{parameter}_residual_data"] = total - material - coordinate_data
        saved_fields[f"{parameter}_residual_symbolic"] = (
            total - material - coordinate_symbolic
        )
        plot_rows.append(
            (
                parameter,
                [
                    total,
                    material,
                    coordinate_path,
                    total - material - coordinate_data,
                    total - material - coordinate_symbolic,
                ],
            )
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "closure_metrics.csv", index=False)
    np.savez_compressed(args.out_dir / "closure_fields.npz", **saved_fields)

    metadata = {
        "input": str(args.input.resolve()),
        "symbolic_csv": str(args.symbolic_csv.resolve()) if args.symbolic_csv else None,
        "symbolic_strategy": args.strategy,
        "symbolic_complexity": model["complexity"],
        "model": model,
        "provenance": provenance,
        "symbolic_expression": symbolic_expression,
        "baseline": {"D": D0, "cx": CX0, "cy": CY0},
        "metrics": summary_rows,
    }
    (args.out_dir / "metrics.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(args.out_dir, summary_rows)

    titles = [
        "T: fixed physical FD",
        "M: fixed reference FD",
        "K: exact path",
        "residual: data gradient",
        "residual: symbolic gradient",
    ]
    fig, axes = plt.subplots(3, 5, figsize=(18, 10), constrained_layout=True)
    for row_index, (parameter, fields) in enumerate(plot_rows):
        for col_index, (title, field) in enumerate(zip(titles, fields)):
            ax = axes[row_index, col_index]
            limit = max(float(np.nanmax(np.abs(field))), 1e-12)
            image = ax.pcolormesh(
                xi,
                eta,
                field.T,
                shading="auto",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
            )
            ax.set_title(f"{parameter}: {title}")
            ax.set_xlabel(r"$\xi$")
            ax.set_ylabel(r"$\eta$")
            fig.colorbar(image, ax=ax, shrink=0.78)
    fig.savefig(args.out_dir / "p0_closure_fields.png", dpi=180)
    plt.close(fig)

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(f"P0 outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
