"""
Step 3 (MeshGraphNets): Visualization and Validation
  1. Pareto front comparison (Strategy A vs B)
  2. CFD ground truth u* vs SR reconstructed u*
  3. Divergence heatmap
  4. Error distribution
  5. y/D cross-section profiles
  6. Best formula LaTeX output

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
x_sym, y_sym, dvdy_sym = sp.symbols("x_D y_D dvdy")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Step 3 validation on outputs_mgn/ or one of its sub-runs."
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


def evaluate_expression(expr, X_data, dvdy_data):
    free = expr.free_symbols
    unexpected = free - {x_sym, y_sym, dvdy_sym}
    if unexpected:
        raise ValueError(f"Unsupported symbols in expression: {sorted(str(s) for s in unexpected)}")

    if dvdy_sym in free:
        func = sp.lambdify([x_sym, y_sym, dvdy_sym], expr, modules="numpy")
        values = func(X_data[:, 0], X_data[:, 1], dvdy_data)
    else:
        func = sp.lambdify([x_sym, y_sym], expr, modules="numpy")
        values = func(X_data[:, 0], X_data[:, 1])
    return ensure_numpy_array(values, len(X_data))


def compute_dudx_from_expr(expr, X_data, dvdy_data):
    return evaluate_expression(sp.diff(expr, x_sym), X_data, dvdy_data)


def load_inputs(run_out_dir):
    df = pd.read_csv(os.path.join(run_out_dir, "wake_data.csv"))
    with open(os.path.join(run_out_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    res_a = pd.read_csv(os.path.join(run_out_dir, "results_strategy_a.csv"))
    res_b = pd.read_csv(os.path.join(run_out_dir, "results_strategy_b.csv"))
    v_wake = np.load(os.path.join(run_out_dir, "v_mean_wake.npy"))
    return df, meta, res_a, res_b, v_wake


def main():
    args = parse_args()
    run_out_dir = resolve_run_out_dir(
        BASE_OUT_DIR, out_dir=args.out_dir, out_subdir=args.out_subdir
    )
    fig_dir = os.path.join(run_out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print(f"Using run directory: {run_out_dir}")
    df, meta, res_a, res_b, v_wake = load_inputs(run_out_dir)

    X = df[["x_D", "y_D"]].values
    y_true = df["u_star"].values
    n_x = int(meta["n_x_wake"])
    n_y = int(meta["n_y"])
    D = float(meta["D"])
    dy_D = float(meta["dy"]) / D

    dvdy = np.zeros_like(v_wake)
    dvdy[:, 1:-1] = (v_wake[:, 2:] - v_wake[:, :-2]) / (2 * dy_D)
    dvdy[:, 0] = (v_wake[:, 1] - v_wake[:, 0]) / dy_D
    dvdy[:, -1] = (v_wake[:, -1] - v_wake[:, -2]) / dy_D
    dvdy_flat = dvdy.flatten()

    x_vals = np.unique(df["x_D"])
    y_vals = np.unique(df["y_D"])
    u_true_2d = y_true.reshape(n_x, n_y)

    best_a_idx = res_a["mse"].idxmin()
    best_b_idx = res_b["mse"].idxmin()
    best_a_row = res_a.loc[best_a_idx]
    best_b_row = res_b.loc[best_b_idx]
    best_a_expr = sp.sympify(best_a_row["sympy"])
    best_b_expr = sp.sympify(best_b_row["sympy"])

    print(
        f"\nBest A: complexity={best_a_row['complexity']}, "
        f"MSE={best_a_row['mse']:.6f}"
    )
    print(f"  Equation: {best_a_row['equation']}")
    print(
        f"\nBest B: complexity={best_b_row['complexity']}, "
        f"MSE={best_b_row['mse']:.6f}"
    )
    print(f"  Equation: {best_b_row['equation']}")

    pred_a = evaluate_expression(best_a_expr, X, dvdy_flat)
    pred_b = evaluate_expression(best_b_expr, X, dvdy_flat)
    pred_a_2d = pred_a.reshape(n_x, n_y)
    pred_b_2d = pred_b.reshape(n_x, n_y)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.semilogy(
        res_a["complexity"],
        res_a["mse"],
        "o-",
        label="Strategy A (standard)",
        color="steelblue",
    )
    if "mse" in res_b.columns:
        ax.semilogy(
            res_b["complexity"],
            res_b["mse"],
            "s-",
            label="Strategy B (div penalty)",
            color="coral",
        )
    ax.set_xlabel("Complexity")
    ax.set_ylabel("MSE")
    ax.set_title("Pareto Front: Complexity vs MSE (MeshGraphNets Data)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "01_pareto_front.png"))
    plt.close()
    print("Fig 1: Pareto front saved")

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(
        res_a["complexity"],
        res_a["div_rms"],
        "o-",
        label="Strategy A div_rms",
        color="steelblue",
    )
    ax.plot(
        res_b["complexity"],
        res_b["div_rms"],
        "s-",
        label="Strategy B div_rms",
        color="coral",
    )
    ax.axhline(y=0.5, color="green", linestyle="--", alpha=0.5, label="Threshold 0.5")
    ax.set_xlabel("Complexity")
    ax.set_ylabel("Divergence RMS")
    ax.set_title("Divergence Constraint: Strategy A vs B")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "02_divergence_comparison.png"))
    plt.close()
    print("Fig 2: Divergence comparison saved")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    vmin, vmax = y_true.min(), y_true.max()
    for ax, data, title in zip(
        axes,
        [u_true_2d, pred_a_2d, pred_b_2d],
        ["CFD (Ground Truth)", "Strategy A (Standard SR)", "Strategy B (Div Penalty)"],
    ):
        im = ax.pcolormesh(
            x_vals,
            y_vals,
            data.T,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )
        ax.set_xlabel("x/D")
        ax.set_ylabel("y/D")
        ax.set_title(title)
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, label="u*")

    fig.suptitle("Wake Region u* Field Comparison (MeshGraphNets)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(
        os.path.join(fig_dir, "03_velocity_field_comparison.png"),
        bbox_inches="tight",
    )
    plt.close()
    print("Fig 3: Velocity field comparison saved")

    dudx_a = compute_dudx_from_expr(best_a_expr, X, dvdy_flat)
    dudx_b = compute_dudx_from_expr(best_b_expr, X, dvdy_flat)
    div_a = (dudx_a + dvdy_flat).reshape(n_x, n_y)
    div_b = (dudx_b + dvdy_flat).reshape(n_x, n_y)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    div_max_abs = max(np.abs(div_a).max(), np.abs(div_b).max())
    for ax, data, title in zip(
        axes,
        [div_a, div_b],
        ["Strategy A: du_pred/dx + dv_data/dy", "Strategy B: du_pred/dx + dv_data/dy"],
    ):
        im = ax.pcolormesh(
            x_vals,
            y_vals,
            data.T,
            cmap="RdBu_r",
            vmin=-div_max_abs,
            vmax=div_max_abs,
            shading="auto",
        )
        ax.set_xlabel("x/D")
        ax.set_ylabel("y/D")
        ax.set_title(title)
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, label="Divergence")

    fig.suptitle("Divergence Check: Continuity Violation", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "04_divergence_heatmap.png"), bbox_inches="tight")
    plt.close()
    print("Fig 4: Divergence heatmap saved")

    err_a = (pred_a - y_true).reshape(n_x, n_y)
    err_b = (pred_b - y_true).reshape(n_x, n_y)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    err_max = max(np.abs(err_a).max(), np.abs(err_b).max())
    for ax, data, title in zip(axes, [err_a, err_b], ["Strategy A Error", "Strategy B Error"]):
        im = ax.pcolormesh(
            x_vals,
            y_vals,
            data.T,
            cmap="RdBu_r",
            vmin=-err_max,
            vmax=err_max,
            shading="auto",
        )
        ax.set_xlabel("x/D")
        ax.set_ylabel("y/D")
        ax.set_title(title)
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, label="Error (pred - true)")

    fig.suptitle("Spatial Error Distribution", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "05_error_distribution.png"), bbox_inches="tight")
    plt.close()
    print("Fig 5: Error distribution saved")

    mid_x_idx = n_x // 2
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(y_vals, u_true_2d[mid_x_idx, :], "k-", linewidth=2, label="CFD")
    ax.plot(y_vals, pred_a_2d[mid_x_idx, :], "b--", linewidth=1.5, label="Strategy A")
    ax.plot(y_vals, pred_b_2d[mid_x_idx, :], "r--", linewidth=1.5, label="Strategy B")
    ax.set_xlabel("y/D")
    ax.set_ylabel("u*")
    ax.set_title(f"u* Profile at x/D = {x_vals[mid_x_idx]:.2f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "06_profile_comparison.png"))
    plt.close()
    print("Fig 6: Profile comparison saved")

    print("\n" + "=" * 60)
    print("SUMMARY (MeshGraphNets Data)")
    print("=" * 60)

    r2_a = 1 - np.sum((pred_a - y_true) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    print(f"\nStrategy A best formula (complexity={best_a_row['complexity']}):")
    print(f"  u* = {best_a_row['equation']}")
    print(f"  MSE = {np.mean((pred_a - y_true) ** 2):.6f}")
    print(f"  R^2 = {r2_a:.4f}")
    print(f"  div_rms = {best_a_row['div_rms']:.4f}")
    print(f"  LaTeX: u^* = {sp.latex(best_a_expr)}")

    r2_b = 1 - np.sum((pred_b - y_true) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    print(f"\nStrategy B best formula (complexity={best_b_row['complexity']}):")
    print(f"  u* = {best_b_row['equation']}")
    print(f"  MSE = {np.mean((pred_b - y_true) ** 2):.6f}")
    print(f"  R^2 = {r2_b:.4f}")
    print(f"  div_rms = {best_b_row['div_rms']:.4f}")
    print(f"  LaTeX: u^* = {sp.latex(best_b_expr)}")

    print("\n--- Strategy A Full Pareto ---")
    for _, row in res_a.iterrows():
        print(
            f"  C={row['complexity']:>2.0f}  MSE={row['mse']:.6f}  "
            f"div_rms={row['div_rms']:.4f}  {row['equation']}"
        )

    print("\n--- Strategy B Full Pareto ---")
    for _, row in res_b.iterrows():
        combined = row["combined_loss"] if "combined_loss" in row.index else "N/A"
        print(
            f"  C={row['complexity']:>2.0f}  MSE={row['mse']:.6f}  "
            f"div_rms={row['div_rms']:.4f}  combined={combined}  {row['equation']}"
        )

    print(f"\nAll figures saved to: {fig_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
