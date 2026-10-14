"""
Step 3: Visualization and Validation
  1. Pareto front comparison (Strategy A vs B)
  2. CFD ground truth u* vs SR reconstructed u*
  3. Divergence heatmap
  4. Error distribution
  5. y/D cross-section profiles
  6. Best formula LaTeX output

This script can run on the default outputs_mgn/ directory or on a self-contained
sub-run directory created by step2_pisr.py.
"""
import argparse
import json
import os
from pathlib import Path
from pipeline_contracts import (resolve_run_dir, load_strategies, selected_models, dataset_manifest, atomic_json)

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
    run_out_dir = resolve_run_dir(run_out_dir)
    strategies = load_strategies(run_out_dir)
    df = pd.read_csv(run_out_dir / "wake_data.csv").sort_values(["x_D", "y_D"]).reset_index(drop=True)
    with (run_out_dir / "meta.json").open(encoding="utf-8") as handle:
        meta = json.load(handle)
    return df, meta, strategies, np.load(run_out_dir / "v_mean_wake.npy")


def divergence_fields(row, X, dvdy_data):
    """Keep model divergence distinct from residual against observed v."""
    u = sp.sympify(str(row["sympy"]))
    mixed = compute_dudx_from_expr(u, X, dvdy_data) + dvdy_data
    if "psi_sympy" in row and pd.notna(row["psi_sympy"]):
        psi = sp.sympify(str(row["psi_sympy"]))
        v = -sp.diff(psi, x_sym)
        own_u = sp.diff(psi, y_sym)
        expression = sp.diff(own_u, x_sym) + sp.diff(v, y_sym)
        model = evaluate_expression(expression, X, dvdy_data)
        return model, mixed, "model divergence: du_pred/dxi + dv_pred/deta"
    return None, mixed, "mixed residual: du_pred/dxi + dv_data/deta"


def plot_fields(path, x_vals, y_vals, panels, title, *, symmetric=False):
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.8), squeeze=False)
    minimum = min(float(np.min(values)) for _, values in panels)
    maximum = max(float(np.max(values)) for _, values in panels)
    if symmetric:
        maximum = max(abs(minimum), abs(maximum), 1e-12)
        minimum = -maximum
    for ax, (label, values) in zip(axes.ravel(), panels):
        im = ax.pcolormesh(x_vals, y_vals, values.T, shading="auto", cmap="RdBu_r" if symmetric else "viridis", vmin=minimum, vmax=maximum)
        ax.set(xlabel="xi", ylabel="eta", title=label)
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, shrink=.85)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    run_out_dir = resolve_run_dir(resolve_run_out_dir(BASE_OUT_DIR, args.out_dir, args.out_subdir))
    df, meta, strategies, v_wake = load_inputs(run_out_dir)
    manifest = selected_models(run_out_dir)
    data = dataset_manifest(run_out_dir)
    fig_dir = run_out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    X = df[["x_D", "y_D"]].values
    target = df.u_star.values
    shape = tuple(data["grid"]["shape"])
    x_vals, y_vals = np.unique(X[:, 0]), np.unique(X[:, 1])
    dvdy = np.gradient(v_wake, y_vals, axis=1, edge_order=2).ravel()
    source = str(meta.get("data_source", "snapshot"))
    selected = {}
    metrics = []
    pareto_rows = []
    for label, table in strategies.items():
        record = manifest["models"][label]
        pred = evaluate_expression(sp.sympify(record["sympy"]), X, dvdy)
        model_div, mixed, div_label = divergence_fields(record, X, dvdy)
        is_psi = model_div is not None
        strategy_name = f"{label}: psi " + ("u+v" if label == "B" else "u-only") if is_psi else f"{label}: u fit"
        selected[label] = dict(pred=pred.reshape(shape), divergence=(model_div if is_psi else mixed).reshape(shape),
                               mixed=mixed.reshape(shape), label=strategy_name, div_label=div_label)
        metrics.append({"strategy": label, "expression_sha256": record["expression_sha256"],
            "complexity": record["complexity"], "mse_u": float(np.mean((pred-target)**2)),
            "model_divergence_rms": float(np.sqrt(np.mean(model_div**2))) if is_psi else None,
            "mixed_residual_rms": float(np.sqrt(np.mean(mixed**2))),
            "divergence_kind": div_label, "training_grid_id": data["grid"]["sha256"],
            "evaluation_grid_id": data["grid"]["sha256"], "target_id": data["target_sha256"]})
        for _, row in table.iterrows():
            own_div, mixed_div, kind = divergence_fields(row, X, dvdy)
            pred_row = evaluate_expression(sp.sympify(row["sympy"]), X, dvdy)
            pareto_rows.append({"strategy": label, "complexity": row["complexity"],
                "mse": float(np.mean((pred_row-target)**2)),
                "div_rms": float(np.sqrt(np.mean((own_div if own_div is not None else mixed_div)**2))), "kind": kind})
    pareto = pd.DataFrame(pareto_rows)
    for filename, column, ylabel in (("01_pareto_front.png", "mse", "analytic u MSE"), ("02_divergence_comparison.png", "div_rms", "divergence / mixed residual RMS")):
        fig, ax = plt.subplots(figsize=(8, 5))
        for label in strategies:
            rows = pareto[pareto.strategy == label]
            ax.plot(rows.complexity, rows[column], "o-", label=selected[label]["label"])
        if column == "mse":
            ax.set_yscale("symlog", linthresh=1e-12)
        ax.set(xlabel="complexity", ylabel=ylabel, title=source)
        ax.legend()
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(fig_dir / filename)
        plt.close(fig)
    truth = target.reshape(shape)
    plot_fields(fig_dir / "03_velocity_field_comparison.png", x_vals, y_vals,
                [("CFD target", truth)] + [(v["label"], v["pred"]) for v in selected.values()], f"Velocity fit: {source}")
    plot_fields(fig_dir / "04_divergence_heatmap.png", x_vals, y_vals,
                [(v["label"] + "\n" + v["div_label"], v["divergence"]) for v in selected.values()], "Model divergence for psi; mixed residual for u-only models", symmetric=True)
    plot_fields(fig_dir / "04b_mixed_residual_heatmap.png", x_vals, y_vals,
                [(v["label"], v["mixed"]) for v in selected.values()], "Mixed diagnostic: du_pred/dxi + dv_data/deta", symmetric=True)
    plot_fields(fig_dir / "05_error_distribution.png", x_vals, y_vals,
                [(v["label"], v["pred"]-truth) for v in selected.values()], "Velocity residual", symmetric=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    mid = shape[0] // 2
    ax.plot(y_vals, truth[mid], "k-", label="CFD target")
    for value in selected.values():
        ax.plot(y_vals, value["pred"][mid], "--", label=value["label"])
    ax.set(xlabel="eta", ylabel="u*", title=f"xi={x_vals[mid]:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "06_profile_comparison.png")
    plt.close(fig)
    pd.DataFrame(metrics).to_csv(run_out_dir / "validation_metrics.csv", index=False)
    atomic_json(run_out_dir / "validation_manifest.json", {"schema_version": 1, "role": "in_sample_model_diagnostics",
        "training_data": manifest["training_data"], "evaluation_data": data, "models": manifest["models"]})
    print(pd.DataFrame(metrics).to_string(index=False))
    return metrics


if __name__ == "__main__":
    main()
