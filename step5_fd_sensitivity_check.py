"""Step 5: close the radius-sensitivity chain-rule decomposition.

The old experiment mapped every perturbed run with its own ``(cx, cy, D)``.
Consequently its finite difference was taken at fixed ``(xi, eta)`` and measured
the intrinsic response ``M_R``, not the fixed-physical-point total derivative.

This script now constructs both derivatives for u and v:

    S_tot* = d q*/dR |_(x,y)
    M_R*   = partial q*/partial R |_(xi,eta)
    K_R*   = -(2/D) (xi partial_xi q* + eta partial_eta q*)

and checks ``S_tot* ~= K_R* + M_R*``.  It also performs the same check for the
dimensional velocity ``q = U_in q*`` with the inlet-speed term made explicit:

    d q/dR |_(x,y) ~= U0 (K_R* + M_R*) + U'_in q*_0.

The psi-B formula supplies the analytic transport term.  A numerical-gradient
transport term from the base CFD field is retained as a diagnostic.  The
dimensionless intrinsic response pair ``(M_R^u*, M_R^v*)`` is exported as the
training target for the Step 6 response-stream-function experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import sympy as sp
from pipeline_contracts import (resolve_run_dir, dataset_manifest, selected_model, grid_manifest,
                                array_hash, content_hash, atomic_json, write_data_manifest)


REPO = Path(__file__).resolve().parent
BASE_GRID_DIR = REPO / "outputs_mgn" / "comsol" / "eta_sym"
PSI_RUN_DIR = REPO / "outputs_mgn" / "comsol" / "eta_sym_psi"
OUT_DIR = REPO / "outputs_mgn" / "fd_check"
CACHE_DIR = OUT_DIR / "case_cache"
PHI_INPUT_DIR = OUT_DIR / "phi_R_input"
SOURCE_DIR = REPO


CASES = {
    "base": dict(file="cylinder_flow_result", D=0.100, cx=0.200, cy=0.200, Umax=1.5000),
    "R-": dict(file="R-.txt", D=0.098, cx=0.200, cy=0.200, Umax=1.5306),
    "R+": dict(file="R+.txt", D=0.102, cx=0.200, cy=0.200, Umax=1.4706),
    "cx-": dict(file="cx-.txt", D=0.100, cx=0.198, cy=0.200, Umax=1.5000),
    "cx+": dict(file="cx+.txt", D=0.100, cx=0.202, cy=0.200, Umax=1.5000),
    "cy-": dict(file="cy-.txt", D=0.100, cx=0.200, cy=0.198, Umax=1.5000),
    "cy+": dict(file="cy+.txt", D=0.100, cx=0.200, cy=0.202, Umax=1.5000),
}

DR = 0.001  # R = D/2, while D+/- differs by 0.002 m.
DCX = 0.002
DCY = 0.002
T_START_FRAC = 0.5
INLET_WIDTH = 0.02
PSI_STRATEGY = "B"
TIME_START_INDEX = None
TIME_END_INDEX = None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=REPO)
    parser.add_argument("--base-grid-dir", type=Path, default=BASE_GRID_DIR)
    parser.add_argument("--psi-dir", type=Path, default=PSI_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--case-config", type=Path, help="JSON with cases using the base/R-/R+/cx-/cx+/cy-/cy+ keys")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def read_comsol(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return mesh coordinates and u/v time histories from one COMSOL export."""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("%"):
                first = stripped
                break
        else:
            raise ValueError(f"No numeric data found in {path}")

    n_cols = len(first.split())
    n_time = (n_cols - 2) // 6
    usecols = [0, 1]
    for index in range(n_time):
        usecols.extend([2 + 6 * index, 3 + 6 * index])
    data = np.loadtxt(path, comments="%", usecols=usecols, dtype=np.float64)
    return data[:, :2], data[:, 2::2], data[:, 3::2], n_time


def estimate_uin(coords: np.ndarray, u_mean: np.ndarray) -> float:
    x_min = float(coords[:, 0].min())
    mask = coords[:, 0] <= x_min + INLET_WIDTH
    if not np.any(mask):
        mask = np.isclose(coords[:, 0], x_min)
    value = float(np.mean(u_mean[mask]))
    return value if abs(value) > 1e-8 else 1.0


def load_case(case_name: str, info: dict) -> dict:
    """Load and time-average a case, using a validated small cache on reruns."""
    path = SOURCE_DIR / info["file"]
    stat = path.stat()
    cache_path = CACHE_DIR / f"{case_name.replace('+', 'p').replace('-', 'm')}.npz"
    cache_key = f"{stat.st_size}:{stat.st_mtime_ns}:{T_START_FRAC}:{TIME_START_INDEX}:{TIME_END_INDEX}:{INLET_WIDTH}"

    if cache_path.exists():
        with np.load(cache_path) as cached:
            if str(cached["cache_key"].item()) == cache_key:
                result = {
                    "coords": cached["coords"],
                    "u_mean": cached["u_mean"],
                    "v_mean": cached["v_mean"],
                    "U_in": float(cached["U_in"]),
                    "n_time": int(cached["n_time"]),
                    **info,
                }
                print(
                    f"[{case_name}] cache: n_nodes={len(result['coords']):5d} "
                    f"n_time={result['n_time']} U_in={result['U_in']:.8f}"
                )
                return result

    print(f"[{case_name}] loading {path.name} ...")
    coords, u_time, v_time, n_time = read_comsol(path)
    start = int(n_time * T_START_FRAC) if TIME_START_INDEX is None else TIME_START_INDEX
    end = n_time if TIME_END_INDEX is None else TIME_END_INDEX
    if not 0 <= start < end <= n_time:
        raise ValueError(f"Case {case_name}: time window [{start}, {end}) is invalid for {n_time} samples")
    u_mean = u_time[:, start:end].mean(axis=1)
    v_mean = v_time[:, start:end].mean(axis=1)
    u_in = estimate_uin(coords, u_mean)
    np.savez_compressed(
        cache_path,
        cache_key=cache_key,
        coords=coords,
        u_mean=u_mean,
        v_mean=v_mean,
        U_in=u_in,
        n_time=n_time,
    )
    print(
        f"    n_nodes={len(coords):5d} n_time={n_time} "
        f"U_in={u_in:.8f} Umax={info['Umax']:.4f}"
    )
    return dict(
        coords=coords,
        u_mean=u_mean,
        v_mean=v_mean,
        U_in=u_in,
        n_time=n_time,
        **info,
    )


def normalized_grid_points(case: dict, x_d: np.ndarray, y_d: np.ndarray) -> np.ndarray:
    xx, yy = np.meshgrid(x_d, y_d, indexing="ij")
    return np.column_stack(
        [case["cx"] + case["D"] * xx.ravel(), case["cy"] + case["D"] * yy.ravel()]
    )


def interpolate_velocity(case: dict, points: np.ndarray) -> np.ndarray:
    """Interpolate dimensional (u,v) to points, with nearest fallback only for NaNs."""
    values = np.column_stack([case["u_mean"], case["v_mean"]])
    linear = griddata(case["coords"], values, points, method="linear")
    missing = np.isnan(linear).any(axis=1)
    if np.any(missing):
        linear[missing] = griddata(
            case["coords"], values, points[missing], method="nearest"
        )
    return linear


def ensure_array(value: object, n_points: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(n_points, float(array))
    return np.broadcast_to(array, (n_points,)).astype(float, copy=False)


def load_psi_b_transport(
    x_d: np.ndarray, y_d: np.ndarray, base_d: float, model=None
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str]:
    """Evaluate psi-B velocity and analytic Euler transport terms for u and v."""
    row = selected_model(PSI_RUN_DIR, PSI_STRATEGY) if model is None else model
    xi, eta = sp.symbols("x_D y_D")
    psi = sp.sympify(str(row["psi_sympy"]))
    u_expr = sp.diff(psi, eta)
    v_expr = -sp.diff(psi, xi)
    xx, yy = np.meshgrid(x_d, y_d, indexing="ij")
    xi_flat, eta_flat = xx.ravel(), yy.ravel()
    n_points = xi_flat.size

    fields: dict[str, np.ndarray] = {}
    transport: dict[str, np.ndarray] = {}
    for component, expression in (("u", u_expr), ("v", v_expr)):
        function = sp.lambdify((xi, eta), expression, modules="numpy")
        values = ensure_array(function(xi_flat, eta_flat), n_points).reshape(xx.shape)
        euler = xi * sp.diff(expression, xi) + eta * sp.diff(expression, eta)
        euler_function = sp.lambdify((xi, eta), euler, modules="numpy")
        euler_values = ensure_array(
            euler_function(xi_flat, eta_flat), n_points
        ).reshape(xx.shape)
        fields[component] = values
        transport[component] = -(2.0 / base_d) * euler_values
    return fields, transport, str(psi)


def numerical_transport(
    field: np.ndarray, x_d: np.ndarray, y_d: np.ndarray, base_d: float
) -> np.ndarray:
    d_dxi, d_deta = np.gradient(field, x_d, y_d, edge_order=2)
    xx, yy = np.meshgrid(x_d, y_d, indexing="ij")
    return -(2.0 / base_d) * (xx * d_dxi + yy * d_deta)


def matched_step_base_cfd_transport(
    base_case: dict,
    base_points: np.ndarray,
    plus_case: dict,
    minus_case: dict,
    step: float,
    shape: tuple[int, int],
) -> np.ndarray:
    """Finite-difference the frozen base CFD field under the exact coordinate map.

    For a perturbed geometry ``case``, a fixed physical point has normalized
    coordinates ``((x-cx_case)/D_case, (y-cy_case)/D_case)``.  To evaluate the
    *frozen* base field at those coordinates, map them back through the base
    geometry and interpolate the original base COMSOL field there.  Using the
    same parameter step as the CFD pair avoids adding a coarse-grid derivative
    or a mismatched Taylor-truncation error to the pipeline check.
    """

    def frozen_for(case: dict) -> np.ndarray:
        xi = (base_points[:, 0] - case["cx"]) / case["D"]
        eta = (base_points[:, 1] - case["cy"]) / case["D"]
        sample_points = np.column_stack(
            [
                base_case["cx"] + base_case["D"] * xi,
                base_case["cy"] + base_case["D"] * eta,
            ]
        )
        return (
            interpolate_velocity(base_case, sample_points).reshape(*shape, 2)
            / base_case["U_in"]
        )

    frozen_plus = frozen_for(plus_case)
    frozen_minus = frozen_for(minus_case)
    return (frozen_plus - frozen_minus) / (2.0 * step)


def field_stats(field: np.ndarray) -> dict[str, float]:
    return {
        "mean_abs": float(np.mean(np.abs(field))),
        "max_abs": float(np.max(np.abs(field))),
        "rms": float(np.sqrt(np.mean(field**2))),
    }


def comparison_metrics(truth: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    residual = estimate - truth
    truth_flat = truth.ravel()
    estimate_flat = estimate.ravel()
    truth_centered = truth_flat - truth_flat.mean()
    estimate_centered = estimate_flat - estimate_flat.mean()
    corr_denominator = np.linalg.norm(truth_centered) * np.linalg.norm(estimate_centered)
    correlation = (
        float(np.dot(truth_centered, estimate_centered) / corr_denominator)
        if corr_denominator > 0
        else float("nan")
    )
    return {
        "rel_l2": float(np.linalg.norm(residual) / max(np.linalg.norm(truth), 1e-12)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "max_abs_error": float(np.max(np.abs(residual))),
        "correlation": correlation,
    }


def add_comparison_row(
    rows: list[dict],
    component: str,
    units: str,
    reconstruction: str,
    truth: np.ndarray,
    estimate: np.ndarray,
) -> None:
    rows.append(
        {
            "component": component,
            "units": units,
            "reconstruction": reconstruction,
            "truth_mean_abs": field_stats(truth)["mean_abs"],
            "estimate_mean_abs": field_stats(estimate)["mean_abs"],
            **comparison_metrics(truth, estimate),
        }
    )


def save_phi_input(
    x_d: np.ndarray,
    y_d: np.ndarray,
    m_star: dict[str, np.ndarray],
    base_meta: dict,
    auxiliary: dict[str, np.ndarray | float | str],
) -> None:
    xx, yy = np.meshgrid(x_d, y_d, indexing="ij")
    pd.DataFrame(
        {
            "x_D": xx.ravel(),
            "y_D": yy.ravel(),
            "u_star": m_star["u"].ravel(),
        }
    ).to_csv(PHI_INPUT_DIR / "wake_data.csv", index=False)
    np.save(PHI_INPUT_DIR / "v_mean_wake.npy", m_star["v"])
    np.savez(PHI_INPUT_DIR / "response_fields.npz", x_D=x_d, y_D=y_d, **auxiliary)

    response_meta = dict(base_meta)
    response_meta.update(
        {
            "U_in": 1.0,
            "pipeline": "radius_intrinsic_response_stream_function_input",
            "response_parameter": "R",
            "response_step_m": DR,
            "response_units": "1/m",
            "fd_target_id": json.loads(str(auxiliary["provenance_json"]))["fd_target_id"],
            "u_target": "partial u_star / partial R at fixed (xi, eta)",
            "v_target": "partial v_star / partial R at fixed (xi, eta)",
            "source": str(OUT_DIR / "fd_sensitivity_fields.npz"),
        }
    )
    with (PHI_INPUT_DIR / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(response_meta, handle, indent=2, ensure_ascii=False)
    write_data_manifest(PHI_INPUT_DIR)


def plot_decomposition(
    x_d: np.ndarray,
    y_d: np.ndarray,
    total: dict[str, np.ndarray],
    transport: dict[str, np.ndarray],
    intrinsic: dict[str, np.ndarray],
    reconstructed: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(22, 9), constrained_layout=True)
    columns = [
        ("total FD", total),
        ("K_R[psi_0]", transport),
        ("M_R FD", intrinsic),
        ("K_R + M_R", reconstructed),
    ]
    for row_index, component in enumerate(("u", "v")):
        limit = max(
            float(np.max(np.abs(fields[component])))
            for _, fields in columns
        )
        limit = max(limit, 1e-12)
        for column_index, (title, fields) in enumerate(columns):
            axis = axes[row_index, column_index]
            image = axis.pcolormesh(
                x_d,
                y_d,
                fields[component].T,
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                shading="auto",
            )
            axis.set_aspect("equal")
            axis.set_xlabel("xi = (x-cx)/D")
            axis.set_ylabel("eta = (y-cy)/D")
            axis.set_title(
                f"{component}: {title}\nmean |.|={np.mean(np.abs(fields[component])):.3g}"
            )
            fig.colorbar(image, ax=axis, shrink=0.84)
    fig.suptitle("Radius sensitivity decomposition at fixed physical points (dimensionless velocity)")
    fig.savefig(OUT_DIR / "04_R_decomposition_fields.png", dpi=170)
    plt.close(fig)


def plot_parity(
    total_star: dict[str, np.ndarray],
    reconstructed_star: dict[str, np.ndarray],
    total_dim: dict[str, np.ndarray],
    reconstructed_dim: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    configurations = [
        ("u", total_star["u"], reconstructed_star["u"], "1/m"),
        ("v", total_star["v"], reconstructed_star["v"], "1/m"),
        ("u", total_dim["u"], reconstructed_dim["u"], "m/s/m"),
        ("v", total_dim["v"], reconstructed_dim["v"], "m/s/m"),
    ]
    for axis, (component, truth, estimate, units) in zip(axes.ravel(), configurations):
        axis.scatter(truth, estimate, s=12, alpha=0.35, edgecolors="none")
        limit = max(float(np.max(np.abs(truth))), float(np.max(np.abs(estimate))), 1e-12)
        axis.plot([-limit, limit], [-limit, limit], "k--", linewidth=1)
        metrics = comparison_metrics(truth, estimate)
        axis.set_xlabel(f"total FD [{units}]")
        axis.set_ylabel(f"K + M reconstruction [{units}]")
        axis.set_title(
            f"{component}, rel L2={metrics['rel_l2']:.3f}, r={metrics['correlation']:.3f}"
        )
        axis.grid(alpha=0.25)
    fig.suptitle("Chain-rule closure using analytic psi-B transport")
    fig.savefig(OUT_DIR / "05_R_decomposition_parity.png", dpi=170)
    plt.close(fig)


def main() -> tuple[pd.DataFrame, dict]:
    global BASE_GRID_DIR, PSI_RUN_DIR, OUT_DIR, CACHE_DIR, PHI_INPUT_DIR, SOURCE_DIR
    global CASES, DR, DCX, DCY, TIME_START_INDEX, TIME_END_INDEX, INLET_WIDTH
    args = parse_args()
    BASE_GRID_DIR = resolve_run_dir(args.base_grid_dir)
    PSI_RUN_DIR = resolve_run_dir(args.psi_dir)
    SOURCE_DIR = args.source_dir.resolve()
    OUT_DIR = args.out_dir.resolve()
    CACHE_DIR, PHI_INPUT_DIR = OUT_DIR / "case_cache", OUT_DIR / "phi_R_input"
    if args.case_config:
        CASES = json.loads(args.case_config.read_text(encoding="utf-8"))["cases"]
    required_cases = {"base", "R-", "R+", "cx-", "cx+", "cy-", "cy+"}
    if set(CASES) != required_cases:
        raise ValueError(f"Expected case keys {sorted(required_cases)}")
    DR = (CASES["R+"]["D"] - CASES["R-"]["D"]) / 4.0
    DCX = (CASES["cx+"]["cx"] - CASES["cx-"]["cx"]) / 2.0
    DCY = (CASES["cy+"]["cy"] - CASES["cy-"]["cy"] ) / 2.0
    for parameter, key, step in (("R", "D", DR * 2), ("cx", "cx", DCX), ("cy", "cy", DCY)):
        if step <= 0:
            raise ValueError("Centered parameter steps must be positive")
        for suffix, sign in (("+", 1), ("-", -1)):
            case = CASES[parameter + suffix]
            for coordinate in ("D", "cx", "cy"):
                expected = CASES["base"][coordinate] + (sign * step if coordinate == key else 0.0)
                if not np.isclose(case[coordinate], expected):
                    raise ValueError("Cases must be symmetric single-parameter perturbations")
    base_data_manifest = dataset_manifest(BASE_GRID_DIR)
    base_model = selected_model(PSI_RUN_DIR, PSI_STRATEGY)
    if "psi_sympy" not in base_model:
        raise ValueError("Step 5 requires a stream-function base model")
    for directory in (OUT_DIR, CACHE_DIR, PHI_INPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    # The reporting/evaluation grid is a Step 1 artifact and may be refined
    # independently of the historical psi-B fit.  Keep its provenance separate
    # from PSI_RUN_DIR so a dense-grid audit cannot silently fall back to the
    # legacy 16-column wake_data snapshot stored beside the psi-B model.
    base_frame = pd.read_csv(BASE_GRID_DIR / "wake_data.csv")
    with (BASE_GRID_DIR / "meta.json").open("r", encoding="utf-8") as handle:
        base_meta = json.load(handle)
    TIME_START_INDEX = base_meta.get("t_avg_start_index")
    TIME_END_INDEX = base_meta.get("t_avg_end_index")
    INLET_WIDTH = float(base_meta.get("inlet_width", 0.02))
    for key, meta_key in (("D", "D"), ("cx", "cyl_cx"), ("cy", "cyl_cy")):
        if not np.isclose(CASES["base"][key], float(base_meta[meta_key])):
            raise ValueError("Case geometry differs from the Step 1 baseline metadata")
    model_normalization = base_model["training_data"]["normalization"]
    for key in ("D", "U_in", "cyl_cx", "cyl_cy"):
        if not np.isclose(float(model_normalization[key]), float(base_meta[key])):
            raise ValueError("Base model and Step 1 normalization differ; retrain or explicitly convert the model")
    n_x = int(base_meta["n_x_wake"])
    n_y = int(base_meta["n_y"])
    x_d = np.sort(base_frame["x_D"].unique())
    y_d = np.sort(base_frame["y_D"].unique())
    if x_d.size != n_x or y_d.size != n_y:
        raise ValueError("Grid mismatch with Step 1 base-grid meta.json")
    print(
        f"Grid: {n_x} x {n_y}; xi=[{x_d[0]:.3f}, {x_d[-1]:.3f}], "
        f"eta=[{y_d[0]:.3f}, {y_d[-1]:.3f}]"
    )

    cases = {name: load_case(name, info) for name, info in CASES.items()}
    if not np.isclose(cases["base"]["U_in"], float(base_meta["U_in"]), rtol=1e-8, atol=1e-12):
        raise ValueError("Step 1/5 inlet normalization differs; align time window and inlet estimate")
    base_points = normalized_grid_points(cases["base"], x_d, y_d)

    own_dimensional: dict[str, np.ndarray] = {}
    own_star: dict[str, np.ndarray] = {}
    for name, case in cases.items():
        points = normalized_grid_points(case, x_d, y_d)
        own_dimensional[name] = interpolate_velocity(case, points).reshape(n_x, n_y, 2)
        own_star[name] = own_dimensional[name] / case["U_in"]
    expected_u = base_frame.sort_values(["x_D", "y_D"])["u_star"].to_numpy().reshape(n_x, n_y)
    expected_v = np.load(BASE_GRID_DIR / "v_mean_wake.npy")
    if not (np.allclose(own_star["base"][..., 0], expected_u, rtol=1e-8, atol=1e-10)
            and np.allclose(own_star["base"][..., 1], expected_v, rtol=1e-8, atol=1e-10)):
        raise ValueError("Step 1 baseline differs from the current source/time window; regenerate Step 1")

    fixed_dimensional = {
        name: interpolate_velocity(cases[name], base_points).reshape(n_x, n_y, 2)
        for name in cases
        if name != "base"
    }
    fixed_star = {
        name: fixed_dimensional[name] / cases[name]["U_in"]
        for name in fixed_dimensional
    }

    component_index = {"u": 0, "v": 1}
    total_star: dict[str, np.ndarray] = {}
    total_dim: dict[str, np.ndarray] = {}
    m_star: dict[str, np.ndarray] = {}
    m_dim_direct: dict[str, np.ndarray] = {}
    base_star: dict[str, np.ndarray] = {}
    for component, index in component_index.items():
        total_star[component] = (
            fixed_star["R+"][..., index] - fixed_star["R-"][..., index]
        ) / (2.0 * DR)
        total_dim[component] = (
            fixed_dimensional["R+"][..., index]
            - fixed_dimensional["R-"][..., index]
        ) / (2.0 * DR)
        m_star[component] = (
            own_star["R+"][..., index] - own_star["R-"][..., index]
        ) / (2.0 * DR)
        m_dim_direct[component] = (
            own_dimensional["R+"][..., index]
            - own_dimensional["R-"][..., index]
        ) / (2.0 * DR)
        base_star[component] = own_star["base"][..., index]

    _, k_psi, psi_expression = load_psi_b_transport(x_d, y_d, cases["base"]["D"], base_model)
    k_cfd = {
        component: numerical_transport(
            base_star[component], x_d, y_d, cases["base"]["D"]
        )
        for component in component_index
    }
    k_empirical = {
        component: total_star[component] - m_star[component]
        for component in component_index
    }

    # Primary data-gradient closure.  Unlike k_cfd above (np.gradient on the
    # 16x64 reporting grid), this differentiates the frozen base COMSOL field
    # on its original mesh using the exact R/cx/cy coordinate maps and the same
    # finite-difference step as each perturbation pair.
    parameter_specs = {
        "R": {"minus": "R-", "plus": "R+", "step": DR},
        "cx": {"minus": "cx-", "plus": "cx+", "step": DCX},
        "cy": {"minus": "cy-", "plus": "cy+", "step": DCY},
    }
    data_gradient_fields: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    data_gradient_rows: list[dict] = []
    for parameter, spec in parameter_specs.items():
        minus_name = spec["minus"]
        plus_name = spec["plus"]
        step = spec["step"]
        k_data_vector = matched_step_base_cfd_transport(
            cases["base"],
            base_points,
            cases[plus_name],
            cases[minus_name],
            step,
            (n_x, n_y),
        )
        data_gradient_fields[parameter] = {}
        for component, index in component_index.items():
            s_total = (
                fixed_star[plus_name][..., index]
                - fixed_star[minus_name][..., index]
            ) / (2.0 * step)
            m_response = (
                own_star[plus_name][..., index]
                - own_star[minus_name][..., index]
            ) / (2.0 * step)
            k_emp = s_total - m_response
            k_data = k_data_vector[..., index]
            reconstructed = k_data + m_response
            condition_number = float(
                np.linalg.norm(k_emp) / max(np.linalg.norm(s_total), 1e-12)
            )
            transport_error = float(
                np.linalg.norm(k_data - k_emp) / max(np.linalg.norm(k_emp), 1e-12)
            )
            closure = comparison_metrics(s_total, reconstructed)
            only_k = comparison_metrics(s_total, k_data)
            k_comparison = comparison_metrics(k_emp, k_data)
            row = {
                "parameter": parameter,
                "component": component,
                "units": "1/m",
                "S_total_mean_abs": field_stats(s_total)["mean_abs"],
                "M_mean_abs": field_stats(m_response)["mean_abs"],
                "K_empirical_mean_abs": field_stats(k_emp)["mean_abs"],
                "K_data_mean_abs": field_stats(k_data)["mean_abs"],
                "condition_number_kappa": condition_number,
                "K_data_vs_empirical_rel_l2": transport_error,
                "kappa_times_transport_error": condition_number * transport_error,
                "closure_rel_l2": closure["rel_l2"],
                "closure_mae_ratio": closure["mae"]
                / max(field_stats(s_total)["mean_abs"], 1e-12),
                "closure_correlation": closure["correlation"],
                "only_K_rel_l2": only_k["rel_l2"],
                "K_data_vs_empirical_correlation": k_comparison["correlation"],
            }
            if parameter == "R":
                psi_transport_error = float(
                    np.linalg.norm(k_psi[component] - k_emp)
                    / max(np.linalg.norm(k_emp), 1e-12)
                )
                psi_closure = comparison_metrics(
                    s_total, k_psi[component] + m_response
                )
                row.update(
                    {
                        "K_psi_vs_empirical_rel_l2": psi_transport_error,
                        "kappa_times_psi_transport_error": condition_number
                        * psi_transport_error,
                        "psi_closure_rel_l2": psi_closure["rel_l2"],
                    }
                )
            data_gradient_rows.append(row)
            data_gradient_fields[parameter][component] = {
                "S_total": s_total,
                "M": m_response,
                "K_empirical": k_emp,
                "K_data": k_data,
                "reconstructed": reconstructed,
            }

    data_gradient_table = pd.DataFrame(data_gradient_rows)
    data_gradient_table.to_csv(
        OUT_DIR / "data_gradient_closure_summary.csv", index=False
    )

    # Resolution audit: vary only the virtual derivative step on the frozen
    # base field while keeping the real perturbation S and M unchanged.
    gradient_scale_rows: list[dict] = []
    for parameter in ("R", "cy"):
        real_step = parameter_specs[parameter]["step"]
        real_fields = data_gradient_fields[parameter]["u"]
        for scale in (0.25, 0.5, 1.0, 2.0):
            delta = scale * real_step
            plus_virtual = dict(cases["base"])
            minus_virtual = dict(cases["base"])
            if parameter == "R":
                plus_virtual["D"] = cases["base"]["D"] + 2.0 * delta
                minus_virtual["D"] = cases["base"]["D"] - 2.0 * delta
            else:
                plus_virtual[parameter] = cases["base"][parameter] + delta
                minus_virtual[parameter] = cases["base"][parameter] - delta
            k_scaled = matched_step_base_cfd_transport(
                cases["base"],
                base_points,
                plus_virtual,
                minus_virtual,
                delta,
                (n_x, n_y),
            )[..., 0]
            scaled_closure = comparison_metrics(
                real_fields["S_total"], k_scaled + real_fields["M"]
            )
            gradient_scale_rows.append(
                {
                    "parameter": parameter,
                    "component": "u",
                    "virtual_gradient_step_scale": scale,
                    "K_data_mean_abs": field_stats(k_scaled)["mean_abs"],
                    "closure_rel_l2": scaled_closure["rel_l2"],
                    "closure_mae_ratio": scaled_closure["mae"]
                    / max(field_stats(real_fields["S_total"])["mean_abs"], 1e-12),
                }
            )
    gradient_scale_table = pd.DataFrame(gradient_scale_rows)
    gradient_scale_table.to_csv(
        OUT_DIR / "data_gradient_step_scale_sensitivity.csv", index=False
    )
    k_data_r = {
        component: data_gradient_fields["R"][component]["K_data"]
        for component in component_index
    }
    reconstructed_star_data = {
        component: data_gradient_fields["R"][component]["reconstructed"]
        for component in component_index
    }

    u0 = cases["base"]["U_in"]
    u_prime = (cases["R+"]["U_in"] - cases["R-"]["U_in"]) / (2.0 * DR)
    reconstructed_star_psi = {
        component: k_psi[component] + m_star[component]
        for component in component_index
    }
    reconstructed_star_cfd = {
        component: k_cfd[component] + m_star[component]
        for component in component_index
    }
    reconstructed_dim_explicit = {
        component: u0 * (k_psi[component] + m_star[component])
        + u_prime * base_star[component]
        for component in component_index
    }
    reconstructed_dim_direct = {
        component: u0 * k_psi[component] + m_dim_direct[component]
        for component in component_index
    }

    rows: list[dict] = []
    for component in component_index:
        add_comparison_row(
            rows,
            component,
            "1/m",
            "psi-B K + fixed-normalized M",
            total_star[component],
            reconstructed_star_psi[component],
        )
        add_comparison_row(
            rows,
            component,
            "1/m",
            "matched-step base-CFD K + fixed-normalized M",
            total_star[component],
            reconstructed_star_data[component],
        )
        add_comparison_row(
            rows,
            component,
            "1/m",
            "base-CFD-gradient K + fixed-normalized M",
            total_star[component],
            reconstructed_star_cfd[component],
        )
        add_comparison_row(
            rows,
            component,
            "1/m",
            "psi-B K versus empirical (S_tot-M)",
            k_empirical[component],
            k_psi[component],
        )
        add_comparison_row(
            rows,
            component,
            "m/s/m",
            "U0*(psi-B K+M*) + Uprime*q0*",
            total_dim[component],
            reconstructed_dim_explicit[component],
        )
        add_comparison_row(
            rows,
            component,
            "m/s/m",
            "U0*psi-B K + fixed-normalized-coordinate dimensional FD",
            total_dim[component],
            reconstructed_dim_direct[component],
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "decomposition_closure_summary.csv", index=False)

    magnitude_rows = []
    for component in component_index:
        for name, field in (
            ("S_total_FD", total_star[component]),
            ("K_psi_B", k_psi[component]),
            ("K_base_CFD_matched_step", k_data_r[component]),
            ("K_empirical", k_empirical[component]),
            ("M_FD", m_star[component]),
            ("K_psi_plus_M", reconstructed_star_psi[component]),
            ("K_data_plus_M", reconstructed_star_data[component]),
        ):
            magnitude_rows.append(
                {"component": component, "term": name, **field_stats(field)}
            )
    magnitude_table = pd.DataFrame(magnitude_rows)
    magnitude_table.to_csv(OUT_DIR / "decomposition_term_magnitudes.csv", index=False)

    legacy_response = {
        "R": (own_star["R+"][..., 0] - own_star["R-"][..., 0]) / (2 * DR),
        "cx": (own_star["cx+"][..., 0] - own_star["cx-"][..., 0]) / (2 * DCX),
        "cy": (own_star["cy+"][..., 0] - own_star["cy-"][..., 0]) / (2 * DCY),
    }
    fixed_coordinate_table = pd.DataFrame(
        [
            {"param": parameter, "quantity": "intrinsic response M (not total sensitivity)", **field_stats(field)}
            for parameter, field in legacy_response.items()
        ]
    )
    fixed_coordinate_table.to_csv(
        OUT_DIR / "fixed_normalized_response_summary.csv", index=False
    )
    # Overwrite the legacy filename so it cannot silently preserve the invalid
    # FD-vs-psi comparison from older runs.
    fixed_coordinate_table.to_csv(OUT_DIR / "fd_vs_pisr_summary.csv", index=False)

    fd_payload = dict(
        x_D=x_d,
        y_D=y_d,
        u_base=base_star["u"],
        v_base=base_star["v"],
        M_R_u=m_star["u"],
        M_R_v=m_star["v"],
        S_total_R_u=total_star["u"],
        S_total_R_v=total_star["v"],
        K_R_u_psi=k_psi["u"],
        K_R_v_psi=k_psi["v"],
        K_R_u_cfd=k_cfd["u"],
        K_R_v_cfd=k_cfd["v"],
        K_R_u_data=k_data_r["u"],
        K_R_v_data=k_data_r["v"],
        K_R_u_empirical=k_empirical["u"],
        K_R_v_empirical=k_empirical["v"],
        S_total_R_u_dim=total_dim["u"],
        S_total_R_v_dim=total_dim["v"],
        M_R_u_dim=m_dim_direct["u"],
        M_R_v_dim=m_dim_direct["v"],
        U0=u0,
        Uprime=u_prime,
        S_R=legacy_response["R"],
        S_cx=legacy_response["cx"],
        S_cy=legacy_response["cy"],
    )

    # Version 2 restores the reference-coordinate perturbed fields used by P0.
    for case_name, suffix in (("R+", "Rp"), ("R-", "Rn"), ("cx+", "cxp"), ("cx-", "cxn"), ("cy+", "cyp"), ("cy-", "cyn")):
        for component, index in component_index.items():
            fd_payload[f"{component}_{suffix}"] = own_star[case_name][..., index]
    provenance = {
        "schema_version": 2, "evaluation_grid": grid_manifest(x_d, y_d),
        "evaluation_data": base_data_manifest, "base_model": base_model,
        "cases": CASES, "parameter_steps": {"R": DR, "cx": DCX, "cy": DCY},
        "normalization": {name: case["U_in"] for name, case in cases.items()},
        "time_window": [TIME_START_INDEX, TIME_END_INDEX],
        "legacy_S_fields": "intrinsic_response_M",
        "fd_target_id": content_hash({key: array_hash(value) for key, value in sorted(fd_payload.items())}),
    }
    fd_payload["provenance_json"] = np.array(json.dumps(provenance))
    np.savez_compressed(OUT_DIR / "fd_sensitivity_fields.npz", **fd_payload)
    atomic_json(OUT_DIR / "fd_manifest.json", provenance)

    data_field_payload: dict[str, np.ndarray] = {"x_D": x_d, "y_D": y_d}
    for parameter, component_fields in data_gradient_fields.items():
        for component, named_fields in component_fields.items():
            for name, field in named_fields.items():
                data_field_payload[f"{parameter}_{component}_{name}"] = field
    closure_provenance = {**provenance, "archive_role": "data_gradient_closure",
        "archive_fields_sha256": content_hash({key: array_hash(value) for key, value in sorted(data_field_payload.items())})}
    data_field_payload["provenance_json"] = np.array(json.dumps(closure_provenance))
    np.savez_compressed(
        OUT_DIR / "data_gradient_closure_fields.npz", **data_field_payload
    )

    save_phi_input(
        x_d,
        y_d,
        m_star,
        base_meta,
        {
            "M_R_u": m_star["u"],
            "M_R_v": m_star["v"],
            "S_total_R_u": total_star["u"],
            "S_total_R_v": total_star["v"],
            "K_R_u_psi": k_psi["u"],
            "K_R_v_psi": k_psi["v"],
            "K_R_u_data": k_data_r["u"],
            "K_R_v_data": k_data_r["v"],
            "U0": u0,
            "Uprime": u_prime,
            "psi0_sympy": psi_expression,
            "provenance_json": json.dumps(provenance),
        },
    )

    if not args.no_plots:
        fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
        image = axis.pcolormesh(x_d, y_d, base_star["u"].T, cmap="viridis", shading="auto")
        axis.set_xlabel("xi")
        axis.set_ylabel("eta")
        axis.set_aspect("equal")
        axis.set_title("Base time-averaged u* field")
        fig.colorbar(image, ax=axis, label="u*")
        fig.savefig(OUT_DIR / "01_u_star_base.png", dpi=170)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
        for axis, parameter in zip(axes, ("R", "cx", "cy")):
            field = legacy_response[parameter]
            limit = max(float(np.max(np.abs(field))), 1e-12)
            image = axis.pcolormesh(
                x_d,
                y_d,
                field.T,
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                shading="auto",
            )
            axis.set_xlabel("xi")
            axis.set_ylabel("eta")
            axis.set_aspect("equal")
            axis.set_title(
                f"M_{parameter} at fixed (xi,eta)\nnot total sensitivity; mean |.|={np.mean(np.abs(field)):.3g}"
            )
            fig.colorbar(image, ax=axis, shrink=0.84)
        fig.suptitle("Legacy fixed-normalized-coordinate FD reclassified as intrinsic response")
        fig.savefig(OUT_DIR / "02_fd_sensitivity_heatmaps.png", dpi=170)
        plt.close(fig)

        labels = ["total FD", "K psi-B", "M FD", "K+M"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
        for axis, component in zip(axes, ("u", "v")):
            values = [
                np.mean(np.abs(total_star[component])),
                np.mean(np.abs(k_psi[component])),
                np.mean(np.abs(m_star[component])),
                np.mean(np.abs(reconstructed_star_psi[component])),
            ]
            axis.bar(labels, values, color=["#355C7D", "#C06C84", "#6C5B7B", "#F8B195"])
            axis.set_ylabel("mean absolute value [1/m]")
            axis.set_title(f"Radius decomposition: {component}*")
            axis.grid(axis="y", alpha=0.25)
            axis.tick_params(axis="x", rotation=18)
        fig.suptitle("Corrected comparison: total sensitivity versus its K and M terms")
        fig.savefig(OUT_DIR / "03_fd_vs_pisr_bar.png", dpi=170)
        plt.close(fig)

        plot_decomposition(
            x_d, y_d, total_star, k_psi, m_star, reconstructed_star_psi
        )
        plot_parity(
            total_star,
            reconstructed_star_psi,
            total_dim,
            reconstructed_dim_explicit,
        )

        # Clean pipeline test: R and cy, u component, all terms from CFD data.
        fig, axes = plt.subplots(2, 4, figsize=(20, 8.5), constrained_layout=True)
        for row_index, parameter in enumerate(("R", "cy")):
            fields = data_gradient_fields[parameter]["u"]
            panels = [
                ("total FD", fields["S_total"]),
                ("K: base CFD data gradient", fields["K_data"]),
                ("M: fixed normalized FD", fields["M"]),
                ("K + M", fields["reconstructed"]),
            ]
            limit = max(float(np.max(np.abs(field))) for _, field in panels)
            for column_index, (label, field) in enumerate(panels):
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
                axis.set_xlabel("xi")
                axis.set_ylabel("eta")
                axis.set_aspect("equal")
                axis.set_title(
                    f"{parameter}, u*: {label}\nmean |.|={np.mean(np.abs(field)):.3g}"
                )
                fig.colorbar(image, ax=axis, shrink=0.82)
        fig.suptitle("Data-gradient chain-rule closure on the original base CFD field")
        fig.savefig(OUT_DIR / "06_data_gradient_closure_fields.png", dpi=170)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
        for axis, parameter in zip(axes, ("R", "cx", "cy")):
            fields = data_gradient_fields[parameter]["u"]
            truth = fields["S_total"].ravel()
            estimate = fields["reconstructed"].ravel()
            metric = comparison_metrics(truth, estimate)
            limit = max(np.max(np.abs(truth)), np.max(np.abs(estimate)), 1e-12)
            axis.scatter(truth, estimate, s=12, alpha=0.35, edgecolors="none")
            axis.plot([-limit, limit], [-limit, limit], "k--", linewidth=1)
            axis.set_xlabel("total FD [1/m]")
            axis.set_ylabel("data-gradient K + M [1/m]")
            axis.set_title(
                f"{parameter}: rel L2={metric['rel_l2']:.3f}, r={metric['correlation']:.3f}"
            )
            axis.grid(alpha=0.25)
        fig.suptitle("Fixed-physical-point closure with matched-step base-CFD gradients")
        fig.savefig(OUT_DIR / "07_data_gradient_closure_parity.png", dpi=170)
        plt.close(fig)

        r_u_row = data_gradient_table[
            (data_gradient_table["parameter"] == "R")
            & (data_gradient_table["component"] == "u")
        ].iloc[0]
        fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        labels = ["psi-B transport", "base-CFD transport"]
        transport_errors = [
            100.0 * r_u_row["K_psi_vs_empirical_rel_l2"],
            100.0 * r_u_row["K_data_vs_empirical_rel_l2"],
        ]
        closure_errors = [
            100.0 * r_u_row["psi_closure_rel_l2"],
            100.0 * r_u_row["closure_rel_l2"],
        ]
        positions = np.arange(2)
        width = 0.36
        axis.bar(
            positions - width / 2,
            transport_errors,
            width,
            label="transport error vs S-M",
            color="#6C5B7B",
        )
        axis.bar(
            positions + width / 2,
            closure_errors,
            width,
            label="amplified total closure error",
            color="#F8B195",
        )
        axis.axhspan(
            10,
            15,
            color="#7FB069",
            alpha=0.15,
            label="10-15% closure band (orange bars)",
        )
        axis.set_xticks(positions)
        axis.set_xticklabels(labels)
        axis.set_ylabel("relative L2 error [%]")
        axis.set_title(
            f"R, u*: gradient error amplified by kappa={r_u_row['condition_number_kappa']:.3f}"
        )
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
        fig.savefig(OUT_DIR / "08_gradient_error_kappa_amplification.png", dpi=170)
        plt.close(fig)

    run_info = {
        "D0_m": cases["base"]["D"],
        "dR_m": DR,
        "U_in": {name: cases[name]["U_in"] for name in ("R-", "base", "R+")},
        "Uprime_per_m": u_prime,
        "Umax_times_D": {
            name: cases[name]["Umax"] * cases[name]["D"] for name in ("R-", "base", "R+")
        },
        "psi_strategy": PSI_STRATEGY,
        "psi_complexity": base_model["complexity"],
        "psi_sympy": psi_expression,
        "base_model": base_model,
        "provenance": provenance,
        "base_grid_dir": str(BASE_GRID_DIR),
        "base_grid_revision": base_meta.get("grid_revision", "legacy_or_unknown"),
        "n_x_wake": n_x,
        "n_y": n_y,
        "psi_run_dir": str(PSI_RUN_DIR),
    }
    with (OUT_DIR / "decomposition_run_info.json").open("w", encoding="utf-8") as handle:
        json.dump(run_info, handle, indent=2, ensure_ascii=False)

    print("\n=== Radius sensitivity term magnitudes (dimensionless velocity per metre) ===")
    print(magnitude_table.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print("\n=== Chain-rule closure ===")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print("\n=== Matched-step base-CFD data-gradient closure ===")
    print(
        data_gradient_table.to_string(
            index=False, float_format=lambda value: f"{value:.6g}"
        )
    )
    print("\n=== Frozen-base numerical-gradient step-scale audit (u*) ===")
    print(
        gradient_scale_table.to_string(
            index=False, float_format=lambda value: f"{value:.6g}"
        )
    )
    print("\nInlet-speed control:")
    print(json.dumps(run_info["U_in"], indent=2))
    print(f"U'_in(R) = {u_prime:.8g} (m/s)/m")
    print(f"Outputs: {OUT_DIR}")
    return summary, run_info


if __name__ == "__main__":
    main()
