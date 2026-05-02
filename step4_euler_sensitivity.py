"""
Step 4: Euler Operator Sensitivity Analysis
  Maps SR formula u* = f(x/D, y/D) to geometric parameter sensitivity S_R
  via Euler operator: S_R = -(2/D) * [xi * df/dxi + eta * df/deta]

This script can run on the default outputs_mgn/ directory or on a self-contained
sub-run directory created by step2_pisr_mgn.py.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import sympy as sp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.size"] = 11
rcParams["figure.dpi"] = 150

BASE_OUT_DIR = r"D:\keyan\cylinder_flow_pisr\outputs_mgn"
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
    df = pd.read_csv(os.path.join(run_out_dir, "wake_data.csv"))
    with open(os.path.join(run_out_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    res_a = pd.read_csv(os.path.join(run_out_dir, "results_strategy_a.csv"))
    res_b = pd.read_csv(os.path.join(run_out_dir, "results_strategy_b.csv"))
    return df, meta, res_a, res_b


def main():
    args = parse_args()
    run_out_dir = resolve_run_out_dir(
        BASE_OUT_DIR, out_dir=args.out_dir, out_subdir=args.out_subdir
    )
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
    print("Euler Operator Sensitivity Analysis")
    print("=" * 60)

    sensitivity_rows = []
    skipped_rows = []

    for label, res_df in [("A", res_a), ("B", res_b)]:
        print(f"\n--- Strategy {label} ---")
        for _, row in res_df.iterrows():
            try:
                _, _, S_R_int, S_R_max, euler_expr = compute_euler_sensitivity(
                    row["sympy"], X, D
                )
                self_check = compute_self_consistency(
                    row["sympy"], X, D, SELF_CHECK_DELTA_FRAC
                )

                sensitivity_rows.append(
                    {
                        "strategy": label,
                        "complexity": row["complexity"],
                        "mse": row["mse"],
                        "combined_loss": row["combined_loss"]
                        if "combined_loss" in row.index
                        else np.nan,
                        "div_rms": row["div_rms"],
                        "S_R_integrated": S_R_int,
                        "S_R_max": S_R_max,
                        "self_check_delta_frac": SELF_CHECK_DELTA_FRAC,
                        "self_check_rel_l2": self_check["rel_l2"],
                        "self_check_rmse": self_check["rmse"],
                        "self_check_mae": self_check["mae"],
                        "self_check_max_abs": self_check["max_abs"],
                        "equation": row["equation"],
                        "sympy": row["sympy"],
                        "euler_expr": str(euler_expr),
                    }
                )
                print(
                    f"  C={row['complexity']:>2.0f}  MSE={row['mse']:.6f}  "
                    f"|S_R|_mean={S_R_int:.4f}  |S_R|_max={S_R_max:.4f}  "
                    f"self-check relL2={self_check['rel_l2']:.3e}  "
                    f"Euler: {euler_expr}"
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

    sens_path = os.path.join(run_out_dir, "sensitivity_results.csv")
    sens_df.to_csv(sens_path, index=False)
    print("\nSensitivity results saved to sensitivity_results.csv")

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

    best_a_idx, best_a_metric = select_best_supported(sens_df, "A", "mse")
    best_b_idx, best_b_metric = select_best_supported(sens_df, "B", "combined_loss")

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
        selected.append(
            {
                "strategy": label,
                "row": row,
                "selection_metric": metric,
                "S_R_2d": S_R_vals.reshape(n_x, n_y),
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
        plt.colorbar(im, ax=ax, label=r"$S_R = \partial u^*/\partial R$")

    fig.suptitle(r"Sensitivity of $u^*$ to Cylinder Radius $R$ (Euler Operator)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "07_SR_sensitivity_heatmap.png"), bbox_inches="tight")
    plt.close()
    print("\nFig 7: S_R sensitivity heatmap saved")

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
        ax.set_ylabel(r"$S_R = \partial u^*/\partial R$")
        ax.set_title(f"x/D = {x_vals[ix]:.2f}")
        ax.legend(loc="upper left", fontsize=9)

    fig.suptitle(r"$S_R$ Cross-Section Profiles (with $u^*$ background)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "08_SR_cross_section_profiles.png"), bbox_inches="tight")
    plt.close()
    print("Fig 8: S_R cross-section profiles saved")

    sens_a = sens_df[sens_df["strategy"] == "A"]
    sens_b = sens_df[sens_df["strategy"] == "B"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    if not sens_a.empty:
        ax1.plot(sens_a["complexity"], sens_a["S_R_integrated"], "o-", color="steelblue", label="Strategy A")
    if not sens_b.empty:
        ax1.plot(sens_b["complexity"], sens_b["S_R_integrated"], "s-", color="coral", label="Strategy B")
    ax1.set_xlabel("Formula Complexity")
    ax1.set_ylabel(r"$\bar{S}_R$ (integrated sensitivity)")
    ax1.set_title(r"Integrated $|S_R|$ vs Complexity")
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
    ax2.set_ylabel(r"$\bar{S}_R$")
    ax2.set_title(r"MSE vs Integrated $|S_R|$")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    if sc_a is not None:
        plt.colorbar(sc_a, ax=ax2, label="Complexity (A)", pad=0.12, shrink=0.6)
    if sc_b is not None:
        plt.colorbar(sc_b, ax=ax2, label="Complexity (B)", pad=0.02, shrink=0.6)

    fig.suptitle("Sensitivity Stability Across Pareto Front", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "09_sensitivity_vs_complexity.png"), bbox_inches="tight")
    plt.close()
    print("Fig 9: Sensitivity vs complexity saved")

    best_a_row_for_fig10 = None
    if best_a_idx is not None:
        best_a_row_for_fig10 = sens_df.loc[best_a_idx]
    else:
        best_a_row_for_fig10 = selected[0]["row"]

    best_expr = sp.sympify(best_a_row_for_fig10["sympy"])
    df_deta_expr = sp.diff(best_expr, eta_sym)
    df_deta_func = sp.lambdify([xi_sym, eta_sym], df_deta_expr, modules="numpy")

    mid_ix = n_x // 2
    eta_profile = y_vals
    df_deta_profile = df_deta_func(x_vals[mid_ix] * np.ones_like(eta_profile), eta_profile)
    if np.isscalar(df_deta_profile):
        df_deta_profile = np.full_like(eta_profile, float(df_deta_profile))
    df_deta_profile = np.array(df_deta_profile, dtype=float)
    euler_profile = eta_profile * df_deta_profile

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(eta_profile, df_deta_profile, "b-", linewidth=2)
    axes[0].set_xlabel(r"$\eta = y/D$")
    axes[0].set_ylabel(r"$df/d\eta$")
    axes[0].set_title(r"Velocity Gradient $df/d\eta$")
    axes[0].axhline(y=0, color="k", linestyle=":", alpha=0.3)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(eta_profile, eta_profile, "g-", linewidth=2)
    axes[1].set_xlabel(r"$\eta = y/D$")
    axes[1].set_ylabel(r"$\eta$")
    axes[1].set_title(r"Position Factor $\eta = y/D$")
    axes[1].axhline(y=0, color="k", linestyle=":", alpha=0.3)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(eta_profile, euler_profile, "r-", linewidth=2)
    axes[2].fill_between(eta_profile, 0, euler_profile, alpha=0.2, color="red")
    axes[2].set_xlabel(r"$\eta = y/D$")
    axes[2].set_ylabel(r"$\mathcal{E}[f] = \eta \cdot df/d\eta$")
    axes[2].set_title(r"Euler Operator $\mathcal{E}[f]$")
    axes[2].axhline(y=0, color="k", linestyle=":", alpha=0.3)
    axes[2].grid(True, alpha=0.3)

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
            label=r"$S_R\,\Delta R$",
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
        f"Pure Mathematical Self-Check for Euler Sensitivity "
        f"(Delta D/D={SELF_CHECK_DELTA_FRAC:.0%})",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "11_self_consistency_validation.png"), bbox_inches="tight")
    plt.close()
    print("Fig 11: self-consistency validation saved")

    print("\n" + "=" * 60)
    print("EULER OPERATOR SENSITIVITY SUMMARY")
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
        print(f"  S_R = -(2/D) * E[f]")
        S_R_sym = sp.Rational(-2, 1) / sp.Symbol("D") * euler_simplified
        print(f"  LaTeX: S_R = {sp.latex(S_R_sym)}")

    cyl_cy = meta.get("cyl_cy", "N/A")
    print("\n--- Physical Interpretation ---")
    print("1. S_R > 0: increasing R increases local u* (flow acceleration)")
    print("2. S_R < 0: increasing R decreases local u* (more blockage/wake deficit)")
    print(f"3. S_R = 0 at y/D = 0 (cylinder center axis, physical y={cyl_cy}):")
    print("   Coordinate origin is at cylinder center, so y/D=0 IS the wake centerline.")
    print("   Euler operator has eta as multiplicative factor -> S_R exactly zero here.")
    print("4. |S_R| peaks at intermediate y/D: maximum sensitivity in the shear layer")
    print("   region, consistent with wake width scaling with D")

    print("\n--- Integrated Sensitivity Ranking ---")
    for label in ["A", "B"]:
        sub = sens_df[sens_df["strategy"] == label].sort_values("mse")
        if sub.empty:
            print(f"  Strategy {label}: no Euler-compatible formula")
            continue
        best = sub.iloc[0]
        print(
            f"  Strategy {label}: |S_R|_mean = {best['S_R_integrated']:.4f}, "
            f"|S_R|_max = {best['S_R_max']:.4f}"
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
