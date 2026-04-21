"""
Step 2 (MeshGraphNets): Physics-Informed Symbolic Regression (PISR)
  - Strategy A: Standard PySR + post-hoc divergence filtering
  - Strategy B: full_objective with embedded divergence penalty

This script is the formal Step 2 entry point. It can optionally crop the
regression window in eta = y/D and write a self-contained run snapshot to an
output subdirectory under outputs_mgn/.
"""
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import sympy as sp


BASE_OUT_DIR = r"D:\keyan\cylinder_flow_pisr\outputs_mgn"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Step 2 PISR on the full wake or an eta-windowed subset."
    )
    parser.add_argument(
        "--eta-min",
        type=float,
        default=None,
        help="Lower bound of the eta = y/D window to keep.",
    )
    parser.add_argument(
        "--eta-max",
        type=float,
        default=None,
        help="Upper bound of the eta = y/D window to keep.",
    )
    parser.add_argument(
        "--out-subdir",
        type=str,
        default=None,
        help="Relative subdirectory under outputs_mgn/ for this run.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only write the prepared run snapshot, without fitting PySR models.",
    )
    args = parser.parse_args()

    if (args.eta_min is None) != (args.eta_max is None):
        parser.error("Provide both --eta-min and --eta-max together.")
    if args.eta_min is not None and args.eta_min >= args.eta_max:
        parser.error("--eta-min must be smaller than --eta-max.")
    if args.eta_min is not None and not args.out_subdir:
        parser.error(
            "--out-subdir is required when using an eta window to avoid "
            "overwriting outputs_mgn/."
        )

    return args


def resolve_run_out_dir(base_out_dir, out_subdir):
    if not out_subdir:
        return base_out_dir, ""

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

    os.makedirs(run_out_dir, exist_ok=True)
    return run_out_dir, normalized.replace("\\", "/")


def load_base_inputs(base_out_dir):
    df_full = pd.read_csv(os.path.join(base_out_dir, "wake_data.csv"))
    with open(os.path.join(base_out_dir, "meta.json"), encoding="utf-8") as f:
        meta_full = json.load(f)
    v_wake_full = np.load(os.path.join(base_out_dir, "v_mean_wake.npy"))
    return df_full, meta_full, v_wake_full


def ensure_numpy_array(values, n_points):
    if np.isscalar(values):
        return np.full(n_points, float(values))
    arr = np.array(values, dtype=float)
    if arr.shape == ():
        return np.full(n_points, float(arr))
    return arr.reshape(-1)


def prepare_dataset(
    df_full,
    v_wake_full,
    meta_full,
    eta_min=None,
    eta_max=None,
    source_out_dir=BASE_OUT_DIR,
    out_subdir="",
):
    D = float(meta_full["D"])
    dx = float(meta_full["dx"])
    dy = float(meta_full["dy"])
    n_x_wake = int(meta_full["n_x_wake"])
    n_y_full = int(meta_full["n_y"])

    if v_wake_full.shape != (n_x_wake, n_y_full):
        raise ValueError(
            f"v_mean_wake.npy shape {v_wake_full.shape} does not match "
            f"(n_x_wake, n_y)=({n_x_wake}, {n_y_full})."
        )

    df_sorted = df_full.sort_values(["x_D", "y_D"]).reset_index(drop=True)
    y_unique_full = np.sort(df_sorted["y_D"].unique())
    if len(y_unique_full) != n_y_full:
        raise ValueError(
            f"wake_data.csv has {len(y_unique_full)} unique y/D rows but meta.json says {n_y_full}."
        )

    if eta_min is None:
        iy_keep = np.arange(n_y_full)
    else:
        iy_keep = np.where((y_unique_full >= eta_min) & (y_unique_full <= eta_max))[0]
        if len(iy_keep) == 0:
            raise ValueError(
                f"No y/D samples fall inside eta window [{eta_min}, {eta_max}]."
            )
        if len(iy_keep) > 1 and not np.all(np.diff(iy_keep) == 1):
            raise ValueError("Selected eta window is not contiguous on the grid.")

    y_keep = y_unique_full[iy_keep]
    df_run = (
        df_sorted[df_sorted["y_D"].isin(y_keep)]
        .sort_values(["x_D", "y_D"])
        .reset_index(drop=True)
    )
    v_wake_run = v_wake_full[:, iy_keep]
    n_y_run = len(y_keep)

    expected_rows = n_x_wake * n_y_run
    if len(df_run) != expected_rows:
        raise ValueError(
            f"Filtered wake_data.csv has {len(df_run)} rows, expected {expected_rows}."
        )

    X = df_run[["x_D", "y_D"]].values
    y = df_run["u_star"].values

    dx_D = dx / D
    dy_D = dy / D
    dvdy = np.zeros_like(v_wake_run)
    if n_y_run > 1:
        dvdy[:, 1:-1] = (v_wake_run[:, 2:] - v_wake_run[:, :-2]) / (2 * dy_D)
        dvdy[:, 0] = (v_wake_run[:, 1] - v_wake_run[:, 0]) / dy_D
        dvdy[:, -1] = (v_wake_run[:, -1] - v_wake_run[:, -2]) / dy_D
    dvdy_flat = dvdy.flatten()

    meta_run = dict(meta_full)
    meta_run["n_y_full"] = n_y_full
    meta_run["n_y"] = n_y_run
    meta_run["eta_min"] = None if eta_min is None else float(eta_min)
    meta_run["eta_max"] = None if eta_max is None else float(eta_max)
    meta_run["y_D_min"] = float(y_keep.min())
    meta_run["y_D_max"] = float(y_keep.max())
    meta_run["n_points"] = int(expected_rows)
    meta_run["is_eta_windowed"] = bool(eta_min is not None)
    meta_run["source_out_dir"] = source_out_dir
    meta_run["out_subdir"] = out_subdir

    print(
        f"Data loaded: {len(df_run)} points, x/D in [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}]"
    )
    print(f"y/D in [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")
    print(f"u* range: [{y.min():.4f}, {y.max():.4f}]")
    print(
        f"n_x_wake={n_x_wake}, n_y={n_y_run}, dx/D={dx_D:.6f}, dy/D={dy_D:.6f}"
    )
    print(f"dv/dy range: [{dvdy_flat.min():.4f}, {dvdy_flat.max():.4f}]")
    if eta_min is None:
        print("Eta window: full domain")
    else:
        print(
            f"Eta window: [{eta_min:.4f}, {eta_max:.4f}] -> kept {n_y_run}/{n_y_full} y rows"
        )

    return {
        "df": df_run,
        "X": X,
        "y": y,
        "dvdy_flat": dvdy_flat,
        "v_wake": v_wake_run,
        "meta": meta_run,
        "n_x_wake": n_x_wake,
        "n_y": n_y_run,
        "dx_D": dx_D,
    }


def save_run_snapshot(run_out_dir, df_run, v_wake_run, meta_run):
    df_run.to_csv(os.path.join(run_out_dir, "wake_data.csv"), index=False)
    np.save(os.path.join(run_out_dir, "v_mean_wake.npy"), v_wake_run)
    with open(os.path.join(run_out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_run, f, indent=2)
    print(f"Run snapshot saved to: {run_out_dir}")


def run_strategy_a(X, y, dvdy_flat, run_out_dir):
    from pysr import PySRRegressor

    print("\n" + "=" * 60)
    print("Strategy A: Standard PySR")
    print("=" * 60)

    model_a = PySRRegressor(
        niterations=30,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "square"],
        populations=20,
        population_size=40,
        maxsize=25,
        extra_sympy_mappings={"square": lambda x: x**2},
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        temp_equation_file=True,
        delete_tempfiles=False,
        verbosity=0,
        progress=False,
    )

    print("Fitting u*(x/D, y/D) ...")
    model_a.fit(X, y, variable_names=["x_D", "y_D"])

    print("\n--- Pareto Front (Strategy A) ---")
    eqs = model_a.equations_
    for _, row in eqs.iterrows():
        print(
            f"  complexity={row['complexity']:>2d}  loss={row['loss']:.6f}  "
            f"equation: {row['equation']}"
        )

    print("\n--- Divergence Post-hoc Filtering ---")
    x_sym, y_sym = sp.symbols("x_D y_D")
    results_a = []
    for eq_idx, row in eqs.iterrows():
        expr = row["sympy_format"]
        has_x = x_sym in expr.free_symbols
        dudx_expr = sp.diff(expr, x_sym)
        dudx_func = sp.lambdify([x_sym, y_sym], dudx_expr, modules="numpy")
        try:
            dudx_vals = ensure_numpy_array(dudx_func(X[:, 0], X[:, 1]), len(X))
            div = dudx_vals + dvdy_flat
            div_rms = float(np.sqrt(np.mean(div**2)))
            div_max = float(np.max(np.abs(div)))

            results_a.append(
                {
                    "equation_index": int(eq_idx),
                    "complexity": int(row["complexity"]),
                    "mse": float(row["loss"]),
                    "div_rms": div_rms,
                    "div_max": div_max,
                    "has_x_D": bool(has_x),
                    "equation": str(row["equation"]),
                    "sympy": str(expr),
                }
            )
            print(
                f"  complexity={row['complexity']:>2d}  MSE={row['loss']:.6f}  "
                f"div_rms={div_rms:.4f}  div_max={div_max:.4f}  "
                f"x_D={'X' if has_x else '.'}  "
                f"{'PASS' if div_rms < 0.5 else 'FAIL'}"
            )
        except Exception as exc:
            print(f"  complexity={row['complexity']:>2d}  ERROR in divergence eval: {exc}")

    results_a_df = pd.DataFrame(results_a)
    results_a_df.to_csv(os.path.join(run_out_dir, "results_strategy_a.csv"), index=False)
    with open(os.path.join(run_out_dir, "pysr_model_a.pkl"), "wb") as f:
        pickle.dump(model_a, f)
    print("Strategy A results saved")


def run_strategy_b(X, y, dvdy_flat, n_x_wake, n_y, dx_D, run_out_dir):
    from pysr import PySRRegressor

    print("\n" + "=" * 60)
    print("Strategy B: full_objective with divergence penalty")
    print("=" * 60)

    X_ext = np.column_stack([X, dvdy_flat])

    full_obj_julia = f"""
function my_custom_objective(tree, dataset::Dataset{{T,L}}, options) where {{T,L}}
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return L(Inf)
    end

    diffs = prediction .- dataset.y
    mse = sum(diffs .^ 2) / length(diffs)

    n_y = {n_y}
    n_x = {n_x_wake}
    dx_D = {dx_D:.10f}
    lambda_div = 0.1

    div_penalty = L(0.0)
    count = 0
    for ix in 2:(n_x-1)
        for iy in 1:n_y
            idx_c = (ix - 1) * n_y + iy
            idx_p = (ix - 2) * n_y + iy
            idx_n = ix * n_y + iy
            dudx = (prediction[idx_n] - prediction[idx_p]) / (2 * dx_D)
            dvdy_val = dataset.X[3, idx_c]
            div_val = dudx + dvdy_val
            div_penalty += div_val^2
            count += 1
        end
    end
    if count > 0
        div_penalty = div_penalty / count
    end

    return mse + lambda_div * div_penalty
end
"""

    model_b = PySRRegressor(
        niterations=30,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "square"],
        populations=20,
        population_size=40,
        maxsize=25,
        extra_sympy_mappings={"square": lambda x: x**2},
        loss_function=full_obj_julia,
        temp_equation_file=True,
        delete_tempfiles=False,
        verbosity=0,
        progress=False,
    )

    print("Fitting u*(x/D, y/D) with divergence penalty ...")
    model_b.fit(X_ext, y, variable_names=["x_D", "y_D", "dvdy"])

    print("\n--- Pareto Front (Strategy B) ---")
    eqs_b = model_b.equations_
    for _, row in eqs_b.iterrows():
        print(
            f"  complexity={row['complexity']:>2d}  loss={row['loss']:.6f}  "
            f"equation: {row['equation']}"
        )

    x_sym, y_sym, dvdy_sym = sp.symbols("x_D y_D dvdy")
    results_b = []
    for eq_idx, row in eqs_b.iterrows():
        expr = row["sympy_format"]
        free = expr.free_symbols
        has_x = x_sym in free
        has_dvdy = dvdy_sym in free

        try:
            if has_dvdy:
                pred_func = sp.lambdify([x_sym, y_sym, dvdy_sym], expr, modules="numpy")
                pred_vals = ensure_numpy_array(
                    pred_func(X[:, 0], X[:, 1], dvdy_flat), len(X)
                )
                dudx_expr = sp.diff(expr, x_sym)
                dudx_func = sp.lambdify(
                    [x_sym, y_sym, dvdy_sym], dudx_expr, modules="numpy"
                )
                dudx_vals = ensure_numpy_array(
                    dudx_func(X[:, 0], X[:, 1], dvdy_flat), len(X)
                )
            else:
                pred_func = sp.lambdify([x_sym, y_sym], expr, modules="numpy")
                pred_vals = ensure_numpy_array(pred_func(X[:, 0], X[:, 1]), len(X))
                dudx_expr = sp.diff(expr, x_sym)
                dudx_func = sp.lambdify([x_sym, y_sym], dudx_expr, modules="numpy")
                dudx_vals = ensure_numpy_array(dudx_func(X[:, 0], X[:, 1]), len(X))

            div = dudx_vals + dvdy_flat
            div_rms = float(np.sqrt(np.mean(div**2)))
            div_max = float(np.max(np.abs(div)))
            mse_pure = float(np.mean((pred_vals - y) ** 2))

            results_b.append(
                {
                    "equation_index": int(eq_idx),
                    "complexity": int(row["complexity"]),
                    "combined_loss": float(row["loss"]),
                    "mse": mse_pure,
                    "div_rms": div_rms,
                    "div_max": div_max,
                    "has_x_D": bool(has_x),
                    "has_dvdy": bool(has_dvdy),
                    "equation": str(row["equation"]),
                    "sympy": str(expr),
                }
            )
            print(
                f"  complexity={row['complexity']:>2d}  combined={row['loss']:.6f}  "
                f"MSE={mse_pure:.6f}  div_rms={div_rms:.4f}  div_max={div_max:.4f}  "
                f"x_D={'X' if has_x else '.'}  dvdy={'v' if has_dvdy else '.'}"
            )
        except Exception as exc:
            print(f"  complexity={row['complexity']:>2d}  ERROR: {exc}")

    results_b_df = pd.DataFrame(results_b)
    results_b_df.to_csv(os.path.join(run_out_dir, "results_strategy_b.csv"), index=False)
    with open(os.path.join(run_out_dir, "pysr_model_b.pkl"), "wb") as f:
        pickle.dump(model_b, f)
    print("Strategy B results saved")


def main():
    args = parse_args()
    run_out_dir, out_subdir = resolve_run_out_dir(BASE_OUT_DIR, args.out_subdir)
    df_full, meta_full, v_wake_full = load_base_inputs(BASE_OUT_DIR)

    dataset = prepare_dataset(
        df_full=df_full,
        v_wake_full=v_wake_full,
        meta_full=meta_full,
        eta_min=args.eta_min,
        eta_max=args.eta_max,
        source_out_dir=BASE_OUT_DIR,
        out_subdir=out_subdir,
    )
    save_run_snapshot(run_out_dir, dataset["df"], dataset["v_wake"], dataset["meta"])

    if args.prepare_only:
        print("\n" + "=" * 60)
        print(f"Step 2 prepare-only done! Snapshot saved to {run_out_dir}")
        print("=" * 60)
        return

    run_strategy_a(
        X=dataset["X"],
        y=dataset["y"],
        dvdy_flat=dataset["dvdy_flat"],
        run_out_dir=run_out_dir,
    )
    run_strategy_b(
        X=dataset["X"],
        y=dataset["y"],
        dvdy_flat=dataset["dvdy_flat"],
        n_x_wake=dataset["n_x_wake"],
        n_y=dataset["n_y"],
        dx_D=dataset["dx_D"],
        run_out_dir=run_out_dir,
    )

    print("\n" + "=" * 60)
    print(f"Step 2 Done! Outputs saved to {run_out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
