"""
Step 4: Euler Coordinate Transport Analysis
  Maps SR formula u* = f(x/D, y/D) to coordinate transport K_R
  via Euler operator: K_R = -(2/D) * [xi * df/dxi + eta * df/deta]

This script can run on the default outputs_mgn/ directory or on a self-contained
sub-run directory created by step2_pisr.py.
"""
import argparse
import json
import os
from pathlib import Path
from pipeline_contracts import resolve_run_dir, load_strategies, selected_models, dataset_manifest, atomic_json

import numpy as np
import pandas as pd
import sympy as sp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.size"] = 11
rcParams["figure.dpi"] = 150

BASE_OUT_DIR = str(Path(__file__).resolve().parent / "outputs_mgn")
SELF_CHECK_DELTA_FRAC = 0.05
xi_sym, eta_sym = sp.symbols("x_D y_D")
SUPPORTED_EULER_SYMBOLS = {xi_sym, eta_sym}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Step 4 Euler sensitivity analysis on outputs_mgn/ or one of its sub-runs."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Absolute or relative path to a self-contained run directory.",
    )
    group.add_argument(
        "--out-subdir",
        type=str,
        default=None,
        help="Relative subdirectory under outputs_mgn/ for this run.",
    )
    return parser.parse_args()


def resolve_run_out_dir(base_out_dir, out_dir=None, out_subdir=None):
    if out_dir:
        return os.path.abspath(out_dir)

    if not out_subdir:
        return base_out_dir

    if os.path.isabs(out_subdir):
        raise ValueError("--out-subdir must be a relative path inside outputs_mgn/.")

    normalized = os.path.normpath(out_subdir)
    if normalized in ("", ".") or normalized.startswith(".."):
        raise ValueError(
            "--out-subdir must stay inside outputs_mgn/ and cannot use parent traversal."
        )

    run_out_dir = os.path.join(base_out_dir, normalized)
    abs_base = os.path.abspath(base_out_dir)
    abs_run = os.path.abspath(run_out_dir)
    if os.path.commonpath([abs_base, abs_run]) != abs_base:
        raise ValueError("--out-subdir resolves outside outputs_mgn/.")
    return abs_run


def ensure_numpy_array(values, n_points):
    if np.isscalar(values):
        return np.full(n_points, float(values))
    arr = np.array(values, dtype=float)
    if arr.shape == ():
        return np.full(n_points, float(arr))
    return arr.reshape(-1)


def build_formula_tools(sympy_expr_str):
    expr = sp.sympify(sympy_expr_str)
    unsupported = expr.free_symbols - SUPPORTED_EULER_SYMBOLS
    if unsupported:
        raise ValueError(
            "Euler analysis only supports closed-form f(x_D, y_D); "
            f"found auxiliary symbols: {sorted(str(s) for s in unsupported)}"
        )

    df_dxi = sp.diff(expr, xi_sym)
    df_deta = sp.diff(expr, eta_sym)
    euler_expr = xi_sym * df_dxi + eta_sym * df_deta
    formula_func = sp.lambdify([xi_sym, eta_sym], expr, modules="numpy")
    euler_func = sp.lambdify([xi_sym, eta_sym], euler_expr, modules="numpy")
    return expr, formula_func, euler_expr, euler_func


def compute_euler_sensitivity(sympy_expr_str, X_data, D_val):
    _, _, euler_expr, euler_func = build_formula_tools(sympy_expr_str)
    euler_vals = ensure_numpy_array(euler_func(X_data[:, 0], X_data[:, 1]), len(X_data))
    S_R_vals = -(2.0 / D_val) * euler_vals
    S_R_integrated = np.mean(np.abs(S_R_vals))
    S_R_max = np.max(np.abs(S_R_vals))
    return euler_vals, S_R_vals, S_R_integrated, S_R_max, euler_expr


def compute_translation_sensitivity(sympy_expr_str, X_data, D_val):
    """Coordinate transport: K_cx = -(1/D) df/dxi, K_cy = -(1/D) df/deta.

    These are the c_x and c_y frozen-field transport terms derived by chain rule from
    xi = (x - c_x)/D and eta = (y - c_y)/D. They are siblings of the Euler scaling
    transport K_R and obey K_R = 2 (xi*K_cx + eta*K_cy).
    The returned S_* keys are compatibility aliases for K, not total response T.
    """
    expr = sp.sympify(sympy_expr_str)
    unsupported = expr.free_symbols - SUPPORTED_EULER_SYMBOLS
    if unsupported:
        raise ValueError(
            "Translation analysis only supports closed-form f(x_D, y_D); "
            f"found auxiliary symbols: {sorted(str(s) for s in unsupported)}"
        )

    df_dxi_expr = sp.diff(expr, xi_sym)
    df_deta_expr = sp.diff(expr, eta_sym)
    tx_func = sp.lambdify([xi_sym, eta_sym], df_dxi_expr, modules="numpy")
    ty_func = sp.lambdify([xi_sym, eta_sym], df_deta_expr, modules="numpy")

    tx_vals = ensure_numpy_array(tx_func(X_data[:, 0], X_data[:, 1]), len(X_data))
    ty_vals = ensure_numpy_array(ty_func(X_data[:, 0], X_data[:, 1]), len(X_data))

    S_cx_vals = -(1.0 / D_val) * tx_vals
    S_cy_vals = -(1.0 / D_val) * ty_vals

    return {
        "S_cx_vals": S_cx_vals,
        "S_cy_vals": S_cy_vals,
        "S_cx_integrated": float(np.mean(np.abs(S_cx_vals))),
        "S_cy_integrated": float(np.mean(np.abs(S_cy_vals))),
        "S_cx_max": float(np.max(np.abs(S_cx_vals))),
        "S_cy_max": float(np.max(np.abs(S_cy_vals))),
        "df_dxi_expr": df_dxi_expr,
        "df_deta_expr": df_deta_expr,
        "has_xi_dep": bool(xi_sym in df_dxi_expr.free_symbols)
        or bool(df_dxi_expr != 0),
    }


def verify_euler_decomposition(S_R_vals, S_cx_vals, S_cy_vals, X_data):
    """Math identity: K_R = 2 (xi * K_cx + eta * K_cy). Should hold to machine precision."""
    xi = X_data[:, 0]
    eta = X_data[:, 1]
    S_R_recon = 2.0 * (xi * S_cx_vals + eta * S_cy_vals)
    diff = S_R_vals - S_R_recon
    rel_l2 = float(np.linalg.norm(diff) / max(np.linalg.norm(S_R_vals), 1e-12))
    max_abs = float(np.max(np.abs(diff)))
    return {
        "S_R_reconstructed": S_R_recon,
        "rel_l2": rel_l2,
        "max_abs": max_abs,
    }


def compute_self_consistency(sympy_expr_str, X_data, D_val, delta_frac):
    _, formula_func, _, euler_func = build_formula_tools(sympy_expr_str)

    xi_old = X_data[:, 0]
    eta_old = X_data[:, 1]
    n_points = len(X_data)
    scale = 1.0 / (1.0 + delta_frac)

    u_old = ensure_numpy_array(formula_func(xi_old, eta_old), n_points)
    u_new = ensure_numpy_array(formula_func(xi_old * scale, eta_old * scale), n_points)
    delta_formula = u_new - u_old

    euler_vals = ensure_numpy_array(euler_func(xi_old, eta_old), n_points)
    delta_D = delta_frac * D_val
    delta_euler = (-(2.0 / D_val) * euler_vals) * (delta_D / 2.0)

    delta_error = delta_formula - delta_euler
    rmse = np.sqrt(np.mean(delta_error ** 2))
    mae = np.mean(np.abs(delta_error))
    max_abs = np.max(np.abs(delta_error))
    rel_l2 = np.linalg.norm(delta_error) / max(np.linalg.norm(delta_formula), 1e-12)

    return {
        "delta_formula": delta_formula,
        "delta_euler": delta_euler,
        "delta_error": delta_error,
        "rmse": rmse,
        "mae": mae,
        "max_abs": max_abs,
        "rel_l2": rel_l2,
        "delta_formula_rms": np.sqrt(np.mean(delta_formula ** 2)),
        "delta_euler_rms": np.sqrt(np.mean(delta_euler ** 2)),
    }


def select_best_supported(sens_df, strategy, preferred_metric):
    sub = sens_df[sens_df["strategy"] == strategy]
    if sub.empty:
        return None, None

    metric = preferred_metric
    if metric not in sub.columns or sub[metric].isna().all():
        metric = "mse"
    idx = sub[metric].astype(float).idxmin()
    return idx, metric


def load_inputs(run_out_dir):
    run_out_dir = resolve_run_dir(run_out_dir)
    strategies = load_strategies(run_out_dir)
    df = pd.read_csv(os.path.join(run_out_dir, "wake_data.csv")).sort_values(["x_D", "y_D"]).reset_index(drop=True)
    with open(os.path.join(run_out_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    res_a = strategies.get("A", pd.DataFrame())
    res_b = strategies.get("B", pd.DataFrame())
    return df, meta, res_a, res_b


def main():
    args = parse_args()
    run_out_dir = resolve_run_out_dir(
        BASE_OUT_DIR, out_dir=args.out_dir, out_subdir=args.out_subdir
    )
    run_out_dir = resolve_run_dir(run_out_dir)
    chosen = selected_models(run_out_dir)
    data_version = dataset_manifest(run_out_dir)
    fig_dir = os.path.join(run_out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print(f"Using run directory: {run_out_dir}")
    df, meta, res_a, res_b = load_inputs(run_out_dir)

    X = df[["x_D", "y_D"]].values
    y_true = df["u_star"].values
    n_x = int(meta["n_x_wake"])
    n_y = int(meta["n_y"])
    D = float(meta["D"])
    x_vals = np.unique(df["x_D"])
    y_vals = np.unique(df["y_D"])

    print(f"Data loaded: {len(df)} points, n_x={n_x}, n_y={n_y}, D={D:.4f}")
    print("\n" + "=" * 60)
    print("Euler Coordinate Transport Analysis")
    print("=" * 60)

    sensitivity_rows = []
    skipped_rows = []

    for label, res_df in [("A", res_a), ("B", res_b)]:
        print(f"\n--- Strategy {label} ---")
        for _, row in res_df.iterrows():
            try:
                _, S_R_vals, S_R_int, S_R_max, euler_expr = compute_euler_sensitivity(
                    row["sympy"], X, D
                )
                self_check = compute_self_consistency(
                    row["sympy"], X, D, SELF_CHECK_DELTA_FRAC
                )
                trans = compute_translation_sensitivity(row["sympy"], X, D)
                ident = verify_euler_decomposition(
                    S_R_vals, trans["S_cx_vals"], trans["S_cy_vals"], X
                )

                sensitivity_rows.append(
                    {
                        "strategy": label,
                        "equation_index": int(row.get("equation_index", row.name)),
                        "quantity": "coordinate_transport_K",
                        "complexity": row["complexity"],
                        "mse": row["mse"],
                        "combined_loss": row["combined_loss"]
                        if "combined_loss" in row.index
                        else np.nan,
                        "div_rms": row["div_rms"],
                        "S_R_integrated": S_R_int,
                        "S_R_max": S_R_max,
                        "S_cx_integrated": trans["S_cx_integrated"],
                        "S_cy_integrated": trans["S_cy_integrated"],
                        "S_cx_max": trans["S_cx_max"],
                        "S_cy_max": trans["S_cy_max"],
                        "identity_rel_l2": ident["rel_l2"],
                        "identity_max_abs": ident["max_abs"],
                        "self_check_delta_frac": SELF_CHECK_DELTA_FRAC,
                        "self_check_rel_l2": self_check["rel_l2"],
                        "self_check_rmse": self_check["rmse"],
                        "self_check_mae": self_check["mae"],
                        "self_check_max_abs": self_check["max_abs"],
                        "equation": row["equation"],
                        "sympy": row["sympy"],
                        "euler_expr": str(euler_expr),
                        "df_dxi_expr": str(trans["df_dxi_expr"]),
                        "df_deta_expr": str(trans["df_deta_expr"]),
                    }
                )
                degenerate_flag = "[xi-DEGEN]" if trans["S_cx_integrated"] < 1e-10 else ""
                print(
                    f"  C={row['complexity']:>2.0f}  MSE={row['mse']:.6f}  "
                    f"|K_R|={S_R_int:.4f}  |K_cx|={trans['S_cx_integrated']:.4e}  "
                    f"|K_cy|={trans['S_cy_integrated']:.4f}  "
                    f"identity relL2={ident['rel_l2']:.2e}  {degenerate_flag}"
                )
            except Exception as exc:
                skipped_rows.append(
                    {
                        "strategy": label,
                        "complexity": row["complexity"],
                        "mse": row["mse"] if "mse" in row.index else np.nan,
                        "combined_loss": row["combined_loss"]
                        if "combined_loss" in row.index
                        else np.nan,
                        "equation": row["equation"],
                        "sympy": row["sympy"],
                        "reason": str(exc),
                    }
                )
                print(f"  C={row['complexity']:>2.0f}  SKIPPED: {exc}")

    sens_df = pd.DataFrame(sensitivity_rows)
    if sens_df.empty:
        raise RuntimeError("No Euler-compatible formulas were found in the selected run directory.")

    for column in list(sens_df.columns):
        if column.startswith(("S_R_", "S_cx_", "S_cy_")):
            sens_df["K" + column[1:]] = sens_df[column]
    atomic_json(run_out_dir / "transport_manifest.json", {"schema_version": 1,
        "quantity": "coordinate_transport_K", "legacy_S_columns": "aliases for K, not total T",
        "training_data": chosen["training_data"], "evaluation_data": data_version,
        "selected_models": chosen["models"], "self_check": "frozen-formula Taylor check, not CFD validation"})
    sens_path = os.path.join(run_out_dir, "sensitivity_results.csv")
    sens_df.to_csv(sens_path, index=False)
    print("\nCoordinate transport results saved to sensitivity_results.csv")

    self_check_table = sens_df[
        [
            "strategy",
            "complexity",
            "mse",
            "combined_loss",
            "self_check_delta_frac",
            "self_check_rel_l2",
            "self_check_rmse",
            "self_check_mae",
            "self_check_max_abs",
        ]
    ].copy()
    self_check_path = os.path.join(run_out_dir, "self_consistency_validation.csv")
    self_check_table.to_csv(self_check_path, index=False)
    print("Self-consistency table saved to self_consistency_validation.csv")

    skipped_path = None
    if skipped_rows:
        skipped_path = os.path.join(run_out_dir, "sensitivity_skipped.csv")
        pd.DataFrame(skipped_rows).to_csv(skipped_path, index=False)
        print("Skipped-formula table saved to sensitivity_skipped.csv")

    def chosen_index(label):
        if label not in chosen["models"]:
            return None, None
        model = chosen["models"][label]
        matches = sens_df[(sens_df.strategy == label) & (sens_df.equation_index == model["equation_index"])]
        if len(matches) != 1:
            raise ValueError(f"Selected {label} model could not be evaluated by Step 4")
        return matches.index[0], model["selection_metric"]

    best_a_idx, best_a_metric = chosen_index("A")
    best_b_idx, best_b_metric = chosen_index("B")

    selected = []
    for label, idx, metric in [
        ("A", best_a_idx, best_a_metric),
        ("B", best_b_idx, best_b_metric),
    ]:
        if idx is None:
            print(f"Strategy {label}: no Euler-compatible formula available for plotting.")
            continue
        row = sens_df.loc[idx]
        _, S_R_vals, _, _, _ = compute_euler_sensitivity(row["sympy"], X, D)
        self_check = compute_self_consistency(row["sympy"], X, D, SELF_CHECK_DELTA_FRAC)
        trans = compute_translation_sensitivity(row["sympy"], X, D)
        ident = verify_euler_decomposition(
            S_R_vals, trans["S_cx_vals"], trans["S_cy_vals"], X
        )
        selected.append(
            {
                "strategy": label,
                "row": row,
                "selection_metric": metric,
                "S_R_2d": S_R_vals.reshape(n_x, n_y),
                "S_cx_2d": trans["S_cx_vals"].reshape(n_x, n_y),
                "S_cy_2d": trans["S_cy_vals"].reshape(n_x, n_y),
                "S_R_recon_2d": ident["S_R_reconstructed"].reshape(n_x, n_y),
                "identity_rel_l2": ident["rel_l2"],
                "identity_max_abs": ident["max_abs"],
                "self_check": self_check,
            }
        )

    if not selected:
        raise RuntimeError("No Euler-compatible formula could be selected for plotting.")

    fig, axes = plt.subplots(1, len(selected), figsize=(7 * len(selected), 5))
    axes = np.atleast_1d(axes)
    sr_abs_max = max(np.abs(item["S_R_2d"]).max() for item in selected)
    for ax, item in zip(axes, selected):
        row = item["row"]
        im = ax.pcolormesh(
            x_vals,
            y_vals,
            item["S_R_2d"].T,
            cmap="RdBu_r",
            vmin=-sr_abs_max,
            vmax=sr_abs_max,
            shading="auto",
        )
        ax.set_xlabel("x/D")
        ax.set_ylabel("y/D")
        ax.set_title(
            f"Strategy {item['strategy']} "
            f"(C={row['complexity']:.0f}, by {item['selection_metric']})"
        )
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, label=r"$K_R$ (frozen-field transport)")

    fig.suptitle(r"Frozen-field radius transport $K_R$; total response requires $M_R$", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "07_SR_sensitivity_heatmap.png"), bbox_inches="tight")
    plt.close()
    print("\nFig 7: K_R transport heatmap saved")

    u_true_2d = y_true.reshape(n_x, n_y)
    x_stations = [0, n_x // 2, n_x - 1]
    fig, axes = plt.subplots(1, len(x_stations), figsize=(5 * len(x_stations), 5))
    color_map = {"A": "steelblue", "B": "coral"}

    for ax, ix in zip(np.atleast_1d(axes), x_stations):
        ax2 = ax.twinx()
        ax2.fill_between(y_vals, 0, u_true_2d[ix, :], alpha=0.15, color="gray")
        ax2.plot(y_vals, u_true_2d[ix, :], "k--", alpha=0.4, linewidth=1, label="u* (CFD)")
        ax2.set_ylabel("u*", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")

        for item in selected:
            ax.plot(
                y_vals,
                item["S_R_2d"][ix, :],
                linewidth=2,
                color=color_map.get(item["strategy"], "black"),
                label=f"Strategy {item['strategy']}",
            )
        ax.axhline(y=0, color="k", linestyle=":", alpha=0.3)
        ax.set_xlabel("y/D")
        ax.set_ylabel(r"$K_R$ (frozen-field transport)")
        ax.set_title(f"x/D = {x_vals[ix]:.2f}")
        ax.legend(loc="upper left", fontsize=9)

    fig.suptitle(r"$K_R$ Cross-Section Profiles (with $u^*$ background)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "08_SR_cross_section_profiles.png"), bbox_inches="tight")
    plt.close()
    print("Fig 8: K_R cross-section profiles saved")

    sens_a = sens_df[sens_df["strategy"] == "A"]
    sens_b = sens_df[sens_df["strategy"] == "B"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    if not sens_a.empty:
        ax1.plot(sens_a["complexity"], sens_a["S_R_integrated"], "o-", color="steelblue", label="Strategy A")
    if not sens_b.empty:
        ax1.plot(sens_b["complexity"], sens_b["S_R_integrated"], "s-", color="coral", label="Strategy B")
    ax1.set_xlabel("Formula Complexity")
    ax1.set_ylabel(r"$\bar{K}_R$ (mean absolute transport)")
    ax1.set_title(r"Integrated $|K_R|$ vs Complexity")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    sc_a = None
    sc_b = None
    if not sens_a.empty:
        sc_a = ax2.scatter(
            sens_a["mse"],
            sens_a["S_R_integrated"],
            c=sens_a["complexity"],
            cmap="Blues",
            s=80,
            edgecolors="k",
            label="Strategy A",
            marker="o",
        )
    if not sens_b.empty:
        sc_b = ax2.scatter(
            sens_b["mse"],
            sens_b["S_R_integrated"],
            c=sens_b["complexity"],
            cmap="Reds",
            s=80,
            edgecolors="k",
            label="Strategy B",
            marker="s",
        )
    ax2.set_xlabel("MSE")
    ax2.set_ylabel(r"$\bar{K}_R$")
    ax2.set_title(r"MSE vs Integrated $|K_R|$")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    if sc_a is not None:
        plt.colorbar(sc_a, ax=ax2, label="Complexity (A)", pad=0.12, shrink=0.6)
    if sc_b is not None:
        plt.colorbar(sc_b, ax=ax2, label="Complexity (B)", pad=0.02, shrink=0.6)

    fig.suptitle("Coordinate Transport Across Pareto Front", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "09_sensitivity_vs_complexity.png"), bbox_inches="tight")
    plt.close()
    print("Fig 9: Transport vs complexity saved")

    best_a_row_for_fig10 = None
    if best_a_idx is not None:
        best_a_row_for_fig10 = sens_df.loc[best_a_idx]
    else:
        best_a_row_for_fig10 = selected[0]["row"]

    best_expr = sp.sympify(best_a_row_for_fig10["sympy"])
    mid_ix = n_x // 2
    eta_profile = y_vals
    xi_profile = np.full_like(eta_profile, x_vals[mid_ix])
    terms = [xi_sym * sp.diff(best_expr, xi_sym), eta_sym * sp.diff(best_expr, eta_sym)]
    values = [ensure_numpy_array(sp.lambdify((xi_sym, eta_sym), term, "numpy")(xi_profile, eta_profile), len(eta_profile)) for term in terms]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, profile, title in zip(axes, [values[0], values[1], values[0] + values[1]],
                                    [r"$\xi\,\partial_\xi f$", r"$\eta\,\partial_\eta f$", r"$\mathcal{E}[f]=\xi\partial_\xi f+\eta\partial_\eta f$"]):
        axis.plot(eta_profile, profile, linewidth=2)
        axis.set(xlabel="eta", ylabel="Euler term", title=title)
        axis.axhline(0, color="k", linestyle=":", alpha=.3)
        axis.grid(alpha=.3)

    formula_latex = sp.latex(best_expr)
    fig.suptitle(
        rf"Euler Operator Decomposition for $u^* = {formula_latex}$"
        f"\n(x/D = {x_vals[mid_ix]:.2f})",
        fontsize=12,
        y=1.05,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "10_euler_operator_decomposition.png"), bbox_inches="tight")
    plt.close()
    print("Fig 10: Euler operator decomposition saved")

    fig, axes = plt.subplots(2, len(selected), figsize=(6 * len(selected), 9), squeeze=False)
    for col, item in enumerate(selected):
        row = item["row"]
        self_check = item["self_check"]
        delta_formula_2d = self_check["delta_formula"].reshape(n_x, n_y)
        delta_euler_2d = self_check["delta_euler"].reshape(n_x, n_y)
        ax_profile = axes[0, col]
        ax_scatter = axes[1, col]

        ax_profile.plot(
            y_vals,
            delta_formula_2d[mid_ix, :],
            "k-",
            linewidth=2,
            label=r"$f(\xi',\eta') - f(\xi,\eta)$",
        )
        ax_profile.plot(
            y_vals,
            delta_euler_2d[mid_ix, :],
            "r--",
            linewidth=2,
            label=r"$K_R\,\Delta R$",
        )
        ax_profile.axhline(y=0, color="k", linestyle=":", alpha=0.3)
        ax_profile.set_xlabel("y/D")
        ax_profile.set_ylabel(r"$\Delta u^*$")
        ax_profile.set_title(
            f"Strategy {item['strategy']} profile at x/D={x_vals[mid_ix]:.2f}\n"
            f"rel L2={self_check['rel_l2']:.2e}, max abs={self_check['max_abs']:.2e}"
        )
        ax_profile.legend(fontsize=9)
        ax_profile.grid(True, alpha=0.3)

        ax_scatter.scatter(
            self_check["delta_formula"],
            self_check["delta_euler"],
            s=14,
            alpha=0.35,
            edgecolors="none",
        )
        lim = max(
            np.max(np.abs(self_check["delta_formula"])),
            np.max(np.abs(self_check["delta_euler"])),
            1e-8,
        )
        ax_scatter.plot([-lim, lim], [-lim, lim], "k--", linewidth=1)
        ax_scatter.set_xlabel(r"Exact rescaled $\Delta u^*$")
        ax_scatter.set_ylabel(r"Euler $\Delta u^*$")
        ax_scatter.set_title(f"Strategy {item['strategy']} parity over all {len(X)} points")
        ax_scatter.grid(True, alpha=0.3)

    fig.suptitle(
        f"Frozen-formula Taylor Check for Coordinate Transport "
        f"(Delta D/D={SELF_CHECK_DELTA_FRAC:.0%})",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "11_self_consistency_validation.png"), bbox_inches="tight")
    plt.close()
    print("Fig 11: self-consistency validation saved")

    # ---- Fig 12: Three-panel heatmap (S_R, S_cx, S_cy) per strategy ----
    fig, axes = plt.subplots(
        len(selected), 3, figsize=(18, 5 * len(selected)), squeeze=False
    )
    for row_idx, item in enumerate(selected):
        row = item["row"]
        sr_lim = float(np.max(np.abs(item["S_R_2d"])))
        scx_lim = float(np.max(np.abs(item["S_cx_2d"])))
        scy_lim = float(np.max(np.abs(item["S_cy_2d"])))
        sr_lim = max(sr_lim, 1e-12)
        scx_lim = max(scx_lim, 1e-12)
        scy_lim = max(scy_lim, 1e-12)

        panels = [
            ("S_R_2d", sr_lim, r"$K_R = -\frac{2}{D}\,\mathcal{E}[f]$", "Radius R"),
            ("S_cx_2d", scx_lim, r"$K_{c_x} = -\frac{1}{D}\,\partial f/\partial \xi$", "Center c_x"),
            ("S_cy_2d", scy_lim, r"$K_{c_y} = -\frac{1}{D}\,\partial f/\partial \eta$", "Center c_y"),
        ]
        for col_idx, (key, lim, eq_label, param_label) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            im = ax.pcolormesh(
                x_vals,
                y_vals,
                item[key].T,
                cmap="RdBu_r",
                vmin=-lim,
                vmax=lim,
                shading="auto",
            )
            ax.set_xlabel("x/D")
            ax.set_ylabel("y/D")
            ax.set_aspect("equal")
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(eq_label, fontsize=10)
            mean_abs = float(np.mean(np.abs(item[key])))
            ax.set_title(
                f"Strategy {item['strategy']} (C={row['complexity']:.0f}) -- "
                f"{param_label}\n"
                rf"$\bar{{|K|}}={mean_abs:.4f}$, $|K|_\max={lim:.4f}$"
            )

    fig.suptitle(
        "Coordinate Transport K: Radius and Cylinder-Center Translations",
        fontsize=13,
        y=1.005,
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_dir, "12_three_param_sensitivity_heatmap.png"),
        bbox_inches="tight",
    )
    plt.close()
    print("Fig 12: three-parameter transport heatmap saved")

    # ---- Fig 13: Math identity parity S_R == 2*(xi*S_cx + eta*S_cy) ----
    fig, axes = plt.subplots(1, len(selected), figsize=(7 * len(selected), 6), squeeze=False)
    axes = axes[0]
    for ax, item in zip(axes, selected):
        S_R_flat = item["S_R_2d"].flatten()
        S_R_recon_flat = item["S_R_recon_2d"].flatten()
        ax.scatter(S_R_flat, S_R_recon_flat, s=14, alpha=0.35, edgecolors="none")
        lim = max(np.max(np.abs(S_R_flat)), np.max(np.abs(S_R_recon_flat)), 1e-8)
        ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1, label="y=x")
        ax.set_xlabel(r"$K_R$ (Euler operator)")
        ax.set_ylabel(r"$2(\xi K_{c_x} + \eta K_{c_y})$ (translation decomposition)")
        ax.set_title(
            f"Strategy {item['strategy']} (C={item['row']['complexity']:.0f})\n"
            f"identity rel L2 = {item['identity_rel_l2']:.2e}, "
            f"max abs = {item['identity_max_abs']:.2e}"
        )
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
    fig.suptitle(
        r"Math Identity: $K_R = 2(\xi\,K_{c_x} + \eta\,K_{c_y})$",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_dir, "13_euler_translation_identity.png"),
        bbox_inches="tight",
    )
    plt.close()
    print("Fig 13: Euler-translation identity parity saved")

    print("\n" + "=" * 60)
    print("COORDINATE TRANSPORT SUMMARY: K (TOTAL T=K+M)")
    print("=" * 60)

    for label, idx in [("A", best_a_idx), ("B", best_b_idx)]:
        if idx is None:
            print(f"\nStrategy {label}: no Euler-compatible formula available.")
            continue
        row = sens_df.loc[idx]
        expr = sp.sympify(row["sympy"])
        euler = xi_sym * sp.diff(expr, xi_sym) + eta_sym * sp.diff(expr, eta_sym)
        euler_simplified = sp.simplify(euler)

        print(
            f"\nStrategy {label} best "
            f"(complexity={row['complexity']:.0f}, MSE={row['mse']:.6f}):"
        )
        print(f"  u*  = {row['equation']}")
        print(f"  LaTeX: u^* = {sp.latex(expr)}")
        print(f"  Euler operator: E[f] = {euler_simplified}")
        print(f"  LaTeX: \\mathcal{{E}}[f] = {sp.latex(euler_simplified)}")
        print(f"  K_R = -(2/D) * E[f]")
        S_R_sym = sp.Rational(-2, 1) / sp.Symbol("D") * euler_simplified
        print(f"  LaTeX: K_R = {sp.latex(S_R_sym)}")

    print("\n--- Coordinate transport interpretation ---")
    print("K_R describes resampling a frozen normalized field. Total CFD response is T_R=K_R+M_R.")
    print("At eta=0, K_R=-(2/D)*xi*df/dxi; it is not generally zero.")
    print("The self-check tests the same formula's Taylor expansion, not a perturbed CFD solution.")
    for item in selected:
        expression = sp.sympify(item["row"]["sympy"])
        centerline = -(2.0 / D) * xi_sym * sp.diff(expression, xi_sym).subs(eta_sym, 0)
        print(f"  Strategy {item['strategy']} centerline K_R: {centerline}")

    print("\n--- Selected-model transport magnitudes ---")
    for label in ["A", "B"]:
        idx, _ = chosen_index(label)
        if idx is None:
            print(f"  Strategy {label}: no Euler-compatible formula")
            continue
        best = sens_df.loc[idx]
        print(
            f"  Strategy {label}: |K_R|_mean ={best['S_R_integrated']:.4f}, "
            f"|K_cx|_mean ={best['S_cx_integrated']:.4e}, "
            f"|K_cy|_mean ={best['S_cy_integrated']:.4f}"
        )

    print("\n--- xi-Degeneracy Audit (K_cx_mean across full Pareto front) ---")
    for label in ["A", "B"]:
        sub = sens_df[sens_df["strategy"] == label].sort_values("complexity")
        if sub.empty:
            continue
        n_total = len(sub)
        n_degen = int((sub["S_cx_integrated"] < 1e-10).sum())
        max_scx = float(sub["S_cx_integrated"].max())
        print(
            f"  Strategy {label}: {n_degen}/{n_total} formulas have |K_cx| < 1e-10 "
            f"(structurally xi-degenerate). Max |K_cx|_mean = {max_scx:.4e}."
        )
        if n_degen / max(n_total, 1) > 0.5:
            print(
                f"  WARNING: >50% of Strategy {label} Pareto front is xi-degenerate. "
                "This makes frozen-field c_x transport structurally zero. Consider re-running "
                "Step 2 with stream-function objective or Sobolev xi-grad penalty."
            )

    print("\n--- Math Identity Check: K_R == 2(xi*K_cx + eta*K_cy) ---")
    for item in selected:
        rel_l2 = item["identity_rel_l2"]
        max_abs = item["identity_max_abs"]
        status = "PASS" if rel_l2 < 1e-10 else ("WARN" if rel_l2 < 1e-6 else "FAIL")
        print(
            f"  Strategy {item['strategy']}: rel_L2 = {rel_l2:.4e}, "
            f"max_abs = {max_abs:.4e}  [{status}]"
        )

    print(f"\n--- Zero-Extra-Compute Self-Check (delta_D/D = {SELF_CHECK_DELTA_FRAC:.0%}) ---")
    for item in selected:
        self_check = item["self_check"]
        print(
            f"  Strategy {item['strategy']}: rel_L2 = {self_check['rel_l2']:.4e}, "
            f"RMSE = {self_check['rmse']:.4e}, "
            f"max_abs = {self_check['max_abs']:.4e}"
        )

    print(f"\nAll figures saved to: {fig_dir}")
    print(f"Sensitivity table saved to: {sens_path}")
    print(f"Self-check table saved to: {self_check_path}")
    if skipped_path:
        print(f"Skipped-formula table saved to: {skipped_path}")
    print("Done!")


if __name__ == "__main__":
    main()
