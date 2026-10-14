"""
Step 2 (Stream-Function PISR): fit psi(xi, eta) instead of u directly.

Mass conservation is structural under the stream-function ansatz:
    u* =   d psi*/d eta
    v* = - d psi*/d xi
where (u*, v*) are dimensionless (divided by U_in) and (xi, eta) are dimensionless
(divided by D). PySR searches for psi*; u* and v* are recovered analytically.

Two strategies (sibling pair to baseline u-fit):
  A_psi: u-loss only on derived u (no v-constraint; tests if joint fit matters)
  B_psi: joint loss u-MSE + lambda_v * v-MSE  (anchors xi-dependence via v != 0)

Outputs match the schema of step2_pisr.py so step4_euler_sensitivity.py runs
unchanged on the result directory.
"""
import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
from pipeline_contracts import resolve_run_dir, training_run, dataset_manifest


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_OUT_DIR = str(REPO_ROOT / "outputs_mgn")
DEFAULT_LAMBDA_V = 1.0
DEFAULT_NITERATIONS = 30
X_SYM, Y_SYM = sp.symbols("x_D y_D")


PSI_OBJECTIVE_U_ONLY = """
function psi_u_only_objective(tree, dataset::Dataset{T,L}, options) where {T,L}
    psi, flag = eval_tree_array(tree, dataset.X, options)
    flag || return L(Inf)

    u_loss = L(0.0)
    count = 0
    @inbounds for ix in 1:N_X
        for iy in 2:(N_Y - 1)
            idx_c = (ix - 1) * N_Y + iy
            dpsi_deta = (psi[idx_c + 1] - psi[idx_c - 1]) / (L(2.0) * L(DY_D))
            u_err = dpsi_deta - L(U_DATA[idx_c])
            u_loss += u_err^2
            count += 1
        end
    end
    count > 0 && (u_loss /= L(count))
    return u_loss
end
"""


PSI_OBJECTIVE_JOINT = """
function psi_joint_objective(tree, dataset::Dataset{T,L}, options) where {T,L}
    psi, flag = eval_tree_array(tree, dataset.X, options)
    flag || return L(Inf)

    u_loss = L(0.0)
    v_loss = L(0.0)
    count = 0
    @inbounds for ix in 2:(N_X - 1)
        for iy in 2:(N_Y - 1)
            idx_c = (ix - 1) * N_Y + iy
            idx_xi_p = (ix - 2) * N_Y + iy
            idx_xi_n = ix * N_Y + iy

            dpsi_deta = (psi[idx_c + 1] - psi[idx_c - 1]) / (L(2.0) * L(DY_D))
            dpsi_dxi  = (psi[idx_xi_n] - psi[idx_xi_p]) / (L(2.0) * L(DX_D))

            u_err = dpsi_deta - L(U_DATA[idx_c])
            v_err = -dpsi_dxi - L(V_DATA[idx_c])

            u_loss += u_err^2
            v_loss += v_err^2
            count += 1
        end
    end
    if count > 0
        u_loss /= L(count)
        v_loss /= L(count)
    end
    return u_loss + L(LAMBDA_V) * v_loss
end
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream-function PISR: fit psi(xi, eta), recover u, v analytically."
    )
    parser.add_argument("--base-out-dir", type=str, default=DEFAULT_BASE_OUT_DIR)
    parser.add_argument(
        "--input-subdir",
        type=str,
        required=True,
        help="Subdir under base-out-dir with wake_data.csv, meta.json, v_mean_wake.npy.",
    )
    parser.add_argument(
        "--out-subdir",
        type=str,
        required=True,
        help="Subdir under base-out-dir to write psi-fit results.",
    )
    parser.add_argument("--lambda-v", type=float, default=DEFAULT_LAMBDA_V)
    parser.add_argument("--niterations", type=int, default=DEFAULT_NITERATIONS)
    parser.add_argument("--maxsize", type=int, default=25)
    parser.add_argument("--skip-strategy-a", action="store_true",
                        help="Skip the u-only ablation; only fit joint u+v.")
    args = parser.parse_args()
    if args.lambda_v < 0:
        parser.error("--lambda-v must be non-negative.")
    return args


def safe_resolve(base, sub):
    """Validate sub stays under base."""
    base_abs = os.path.abspath(base)
    if os.path.isabs(sub):
        raise ValueError(f"--*-subdir must be relative: got {sub}")
    norm = os.path.normpath(sub)
    if norm in ("", ".") or norm.startswith(".."):
        raise ValueError(f"--*-subdir must stay inside base: got {sub}")
    full = os.path.abspath(os.path.join(base_abs, norm))
    if os.path.commonpath([base_abs, full]) != base_abs:
        raise ValueError(f"--*-subdir resolves outside base: got {sub}")
    return full


def load_inputs(input_dir):
    input_dir = resolve_run_dir(input_dir)
    dataset_manifest(input_dir)
    df = pd.read_csv(os.path.join(input_dir, "wake_data.csv"))
    with open(os.path.join(input_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    v_mean_wake = np.load(os.path.join(input_dir, "v_mean_wake.npy"))
    return df, meta, v_mean_wake


def prepare_psi_dataset(df, meta, v_mean_wake):
    df_sorted = df.sort_values(["x_D", "y_D"]).reset_index(drop=True)
    n_x = int(meta["n_x_wake"])
    n_y = int(meta["n_y"])
    D = float(meta["D"])
    U_in = float(meta["U_in"])
    dx_D = float(meta["dx"]) / D
    dy_D = float(meta["dy"]) / D

    if v_mean_wake.shape != (n_x, n_y):
        raise ValueError(
            f"v_mean_wake shape {v_mean_wake.shape} != (n_x={n_x}, n_y={n_y})."
        )
    expected_rows = n_x * n_y
    if len(df_sorted) != expected_rows:
        raise ValueError(
            f"wake_data has {len(df_sorted)} rows, expected {expected_rows}."
        )

    X = df_sorted[["x_D", "y_D"]].values
    u_star = df_sorted["u_star"].values.astype(np.float64)
    # v_mean_wake.npy follows the Step 1 contract and is already v/U_in.
    v_star = v_mean_wake.astype(np.float64).flatten()  # row-major like df_sorted

    # The wake_data.csv ordering must match v_mean_wake.npy reshape ordering.
    # Verify by matching x_D layout: n_x chunks of n_y same-x rows.
    x_arr = df_sorted["x_D"].values.reshape(n_x, n_y)
    if not np.allclose(x_arr.std(axis=1), 0.0):
        raise ValueError(
            "wake_data.csv ordering does not match (n_x, n_y) reshape; "
            "expected blocks of constant x_D per row."
        )

    return {
        "X": X,
        "u_star": u_star,
        "v_star": v_star,
        "n_x": n_x,
        "n_y": n_y,
        "D": D,
        "U_in": U_in,
        "dx_D": dx_D,
        "dy_D": dy_D,
        "df_sorted": df_sorted,
        "v_mean_wake": v_mean_wake,
    }


def inject_psi_globals(u_data, v_data, n_x, n_y, dx_D, dy_D, lambda_v):
    from pysr import jl

    expected_rows = n_x * n_y
    if len(u_data) != expected_rows or len(v_data) != expected_rows:
        raise ValueError(
            f"u_data len={len(u_data)}, v_data len={len(v_data)}, expected {expected_rows}."
        )

    jl.seval("global U_DATA, V_DATA, N_X, N_Y, DX_D, DY_D, LAMBDA_V")
    jl.U_DATA = np.ascontiguousarray(u_data, dtype=np.float64)
    jl.V_DATA = np.ascontiguousarray(v_data, dtype=np.float64)
    jl.N_X = int(n_x)
    jl.N_Y = int(n_y)
    jl.DX_D = float(dx_D)
    jl.DY_D = float(dy_D)
    jl.LAMBDA_V = float(lambda_v)


def derive_velocity_from_psi(psi_expr_str):
    psi_expr = sp.sympify(psi_expr_str)
    u_expr = sp.diff(psi_expr, Y_SYM)
    v_expr = -sp.diff(psi_expr, X_SYM)
    return psi_expr, u_expr, v_expr


def lambdify_safe(expr, n_points):
    """Lambdify expr and return a function that always emits a length-n_points array."""
    func = sp.lambdify([X_SYM, Y_SYM], expr, modules="numpy")

    def evaluate(xi_vals, eta_vals):
        out = func(xi_vals, eta_vals)
        out = np.array(out, dtype=float)
        if out.ndim == 0:
            return np.full(n_points, float(out))
        return out.reshape(-1)

    return evaluate


def evaluate_psi_pareto(eqs, X, u_data, v_data, *, strategy_label, include_combined_loss=True):
    n_points = len(X)
    rows = []
    print(f"\n--- Strategy {strategy_label}: derived u/v Pareto ---")
    for eq_idx, eq_row in eqs.iterrows():
        try:
            psi_expr_str = str(eq_row["sympy_format"])
            psi_expr, u_expr, v_expr = derive_velocity_from_psi(psi_expr_str)

            unexpected = (psi_expr.free_symbols | u_expr.free_symbols | v_expr.free_symbols) - {X_SYM, Y_SYM}
            if unexpected:
                raise ValueError(
                    f"closed-form check failed: extra symbols {sorted(str(s) for s in unexpected)}"
                )

            u_func = lambdify_safe(u_expr, n_points)
            v_func = lambdify_safe(v_expr, n_points)
            u_pred = u_func(X[:, 0], X[:, 1])
            v_pred = v_func(X[:, 0], X[:, 1])
            if not np.isfinite(u_pred).all() or not np.isfinite(v_pred).all():
                raise ValueError("Candidate has non-finite analytic velocity values")

            mse_u = float(np.mean((u_pred - u_data) ** 2))
            mse_v = float(np.mean((v_pred - v_data) ** 2))
            x_in_u = bool(X_SYM in u_expr.free_symbols)
            x_in_psi = bool(X_SYM in psi_expr.free_symbols)

            entry = {
                "equation_index": int(eq_idx),
                "complexity": int(eq_row["complexity"]),
                "mse": mse_u,
                "mse_v": mse_v,
                "div_rms": 0.0,
                "div_max": 0.0,
                "has_x_D": x_in_u,
                "has_x_D_in_psi": x_in_psi,
                "equation": str(u_expr),
                "sympy": str(u_expr),
                "psi_equation": str(psi_expr),
                "psi_sympy": psi_expr_str,
                "v_equation": str(v_expr),
                "v_sympy": str(v_expr),
            }
            if include_combined_loss:
                entry["combined_loss"] = float(eq_row["loss"])
                entry["has_dvdy"] = False

            print(
                f"  C={int(eq_row['complexity']):>2d}  MSE_u={mse_u:.5f}  MSE_v={mse_v:.5f}  "
                f"u_has_x_D={'X' if x_in_u else '.'}  psi_has_x_D={'X' if x_in_psi else '.'}"
            )
            rows.append(entry)
        except Exception as exc:
            print(f"  C={int(eq_row['complexity']):>2d}  SKIPPED: {exc}")

    return pd.DataFrame(rows)


def run_psi_strategy(
    label,
    *,
    X,
    target_dummy,
    objective_julia,
    u_data,
    v_data,
    niterations,
    maxsize,
    run_out_dir,
    label_long,
):
    from pysr import PySRRegressor

    print("\n" + "=" * 60)
    print(f"Strategy {label} ({label_long})")
    print("=" * 60)

    model = PySRRegressor(
        niterations=niterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "square"],
        populations=20,
        population_size=40,
        maxsize=maxsize,
        extra_sympy_mappings={"square": lambda x: x ** 2},
        loss_function=objective_julia,
        temp_equation_file=True,
        delete_tempfiles=False,
        verbosity=0,
        progress=False,
    )

    print(f"Fitting psi(x_D, y_D) for {label_long} ...")
    model.fit(X, target_dummy, variable_names=["x_D", "y_D"])

    eqs = model.equations_
    print(f"\n--- Pareto Front (psi formulas, Strategy {label}) ---")
    for _, eq_row in eqs.iterrows():
        print(
            f"  C={int(eq_row['complexity']):>2d}  loss={eq_row['loss']:.5f}  "
            f"psi: {eq_row['equation']}"
        )

    df_results = evaluate_psi_pareto(
        eqs, X, u_data, v_data, strategy_label=label, include_combined_loss=True
    )
    csv_path = os.path.join(run_out_dir, f"results_strategy_{label.lower()}.csv")
    df_results.to_csv(csv_path, index=False)
    pkl_path = os.path.join(run_out_dir, f"pysr_model_{label.lower()}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved {csv_path} and {pkl_path}")


def main():
    args = parse_args()
    base = os.path.abspath(args.base_out_dir)
    input_dir = resolve_run_dir(safe_resolve(base, args.input_subdir))
    output_root = safe_resolve(base, args.out_subdir)

    df, meta, v_mean_wake = load_inputs(input_dir)
    dataset = prepare_psi_dataset(df, meta, v_mean_wake)

    with training_run(
        output_root, input_dir, pipeline="stream_function_psi",
        strategies=["B"] if args.skip_strategy_a else ["A", "B"],
        settings={"lambda_v": args.lambda_v, "niterations": args.niterations, "maxsize": args.maxsize},
    ) as run_out_dir:
        train_snapshot(args, dataset, meta, v_mean_wake, run_out_dir)
    print(f"Completed run: {run_out_dir}")
    return run_out_dir


def train_snapshot(args, dataset, meta, v_mean_wake, run_out_dir):

    # Snapshot inputs into run dir so step4 finds wake_data.csv, meta.json, v_mean_wake.npy.
    dataset["df_sorted"].to_csv(
        os.path.join(run_out_dir, "wake_data.csv"), index=False
    )
    np.save(os.path.join(run_out_dir, "v_mean_wake.npy"), v_mean_wake)
    meta_new = dict(meta)
    meta_new["pipeline"] = "stream_function_psi"
    meta_new["lambda_v"] = float(args.lambda_v)
    meta_new["psi_input_subdir"] = args.input_subdir
    meta_new["psi_out_subdir"] = args.out_subdir
    with open(os.path.join(run_out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_new, f, indent=2)

    inject_psi_globals(
        u_data=dataset["u_star"],
        v_data=dataset["v_star"],
        n_x=dataset["n_x"],
        n_y=dataset["n_y"],
        dx_D=dataset["dx_D"],
        dy_D=dataset["dy_D"],
        lambda_v=args.lambda_v,
    )

    print(f"Run dir   : {run_out_dir}")
    print(f"Input     : {args.input_subdir}")
    print(f"Grid      : n_x={dataset['n_x']}, n_y={dataset['n_y']}, "
          f"dx/D={dataset['dx_D']:.5f}, dy/D={dataset['dy_D']:.5f}")
    print(f"||u*||_RMS = {np.sqrt(np.mean(dataset['u_star']**2)):.4e}")
    print(f"||v*||_RMS = {np.sqrt(np.mean(dataset['v_star']**2)):.4e}")
    print(f"lambda_v   = {args.lambda_v}")

    n_points = len(dataset["X"])
    target_dummy = np.zeros(n_points, dtype=np.float64)

    if not args.skip_strategy_a:
        run_psi_strategy(
            "A",
            X=dataset["X"],
            target_dummy=target_dummy,
            objective_julia=PSI_OBJECTIVE_U_ONLY,
            u_data=dataset["u_star"],
            v_data=dataset["v_star"],
            niterations=args.niterations,
            maxsize=args.maxsize,
            run_out_dir=run_out_dir,
            label_long="psi-fit, u-loss only (no v constraint)",
        )

    run_psi_strategy(
        "B",
        X=dataset["X"],
        target_dummy=target_dummy,
        objective_julia=PSI_OBJECTIVE_JOINT,
        u_data=dataset["u_star"],
        v_data=dataset["v_star"],
        niterations=args.niterations,
        maxsize=args.maxsize,
        run_out_dir=run_out_dir,
        label_long=f"psi-fit, joint u+v (lambda_v={args.lambda_v})",
    )

    print("\n" + "=" * 60)
    print(f"Stream-function PISR done! Outputs in {run_out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
