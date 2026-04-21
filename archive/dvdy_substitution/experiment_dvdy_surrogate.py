"""
Experiment: regress dvdy ~= g(x_D, y_D), then substitute g into Strategy B
formulas to test whether a closed-form surrogate can retain or improve u*
prediction accuracy.
"""
import argparse
import json
import os
import pickle

import matplotlib
import numpy as np
import pandas as pd
import sympy as sp

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.size"] = 11
rcParams["figure.dpi"] = 150

BASE_OUT_DIR = r"D:\keyan\cylinder_flow_pisr\outputs_mgn"
x_sym, y_sym, dvdy_sym = sp.symbols("x_D y_D dvdy")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regress dvdy as g(x_D, y_D) and substitute it into Strategy B formulas."
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
    parser.add_argument(
        "--niterations",
        type=int,
        default=30,
        help="PySR iterations for fitting dvdy ~= g(x_D, y_D).",
    )
    parser.add_argument(
        "--populations",
        type=int,
        default=20,
        help="PySR populations for the dvdy surrogate run.",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=40,
        help="PySR population size for the dvdy surrogate run.",
    )
    parser.add_argument(
        "--maxsize",
        type=int,
        default=20,
        help="Maximum symbolic complexity for the dvdy surrogate run.",
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


def compute_r2(y_true, y_pred):
    denom = np.sum((y_true - y_true.mean()) ** 2)
    if denom <= 1e-12:
        return np.nan
    return 1.0 - np.sum((y_pred - y_true) ** 2) / denom


def evaluate_expr(expr, X_data, dvdy_data=None):
    free = expr.free_symbols
    unexpected = free - {x_sym, y_sym, dvdy_sym}
    if unexpected:
        raise ValueError(f"Unsupported symbols in expression: {sorted(str(s) for s in unexpected)}")

    if dvdy_sym in free:
        if dvdy_data is None:
            raise ValueError("Expression requires dvdy but no dvdy_data was provided.")
        func = sp.lambdify([x_sym, y_sym, dvdy_sym], expr, modules="numpy")
        values = func(X_data[:, 0], X_data[:, 1], dvdy_data)
    else:
        func = sp.lambdify([x_sym, y_sym], expr, modules="numpy")
        values = func(X_data[:, 0], X_data[:, 1])
    return ensure_numpy_array(values, len(X_data))


def compute_dvdy(v_wake, dy_over_D):
    dvdy = np.zeros_like(v_wake)
    if v_wake.shape[1] > 1:
        dvdy[:, 1:-1] = (v_wake[:, 2:] - v_wake[:, :-2]) / (2 * dy_over_D)
        dvdy[:, 0] = (v_wake[:, 1] - v_wake[:, 0]) / dy_over_D
        dvdy[:, -1] = (v_wake[:, -1] - v_wake[:, -2]) / dy_over_D
    return dvdy


def fit_dvdy_surrogate(X, dvdy_target, args, run_out_dir):
    from pysr import PySRRegressor

    print("\n" + "=" * 60)
    print("Fitting dvdy surrogate g(x_D, y_D) with PySR")
    print("=" * 60)

    model = PySRRegressor(
        niterations=args.niterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "square"],
        populations=args.populations,
        population_size=args.population_size,
        maxsize=args.maxsize,
        extra_sympy_mappings={"square": lambda x: x**2},
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        temp_equation_file=True,
        delete_tempfiles=False,
        verbosity=0,
        progress=False,
    )
    model.fit(X, dvdy_target, variable_names=["x_D", "y_D"])

    rows = []
    for eq_idx, row in model.equations_.iterrows():
        expr = row["sympy_format"]
        pred = evaluate_expr(expr, X)
        mse = float(np.mean((pred - dvdy_target) ** 2))
        mae = float(np.mean(np.abs(pred - dvdy_target)))
        max_abs = float(np.max(np.abs(pred - dvdy_target)))
        r2 = float(compute_r2(dvdy_target, pred))
        has_x = bool(x_sym in expr.free_symbols)

        rows.append(
            {
                "equation_index": int(eq_idx),
                "complexity": int(row["complexity"]),
                "loss": float(row["loss"]),
                "mse": mse,
                "mae": mae,
                "max_abs": max_abs,
                "r2": r2,
                "has_x_D": has_x,
                "selected_terms": "",
                "equation": str(row["equation"]),
                "sympy": str(expr),
            }
        )
        print(
            f"  C={int(row['complexity']):>2d}  pysr_loss={float(row['loss']):.6f}  "
            f"mse={mse:.6f}  mae={mae:.6f}  r2={r2:.4f}  "
            f"x_D={'X' if has_x else '.'}  {row['equation']}"
        )

    results_df = pd.DataFrame(rows).sort_values(["mse", "complexity"]).reset_index(drop=True)
    best_expr = sp.sympify(results_df.iloc[0]["sympy"])
    best_pred = evaluate_expr(best_expr, X)
    model_path = os.path.join(run_out_dir, "pysr_model_dvdy.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return results_df, best_expr, best_pred, model_path


def run_substitution_experiment(X, u_true, dvdy_true, res_b, g_expr, g_pred):
    rows = []
    for _, row in res_b.iterrows():
        expr = sp.sympify(row["sympy"])
        pred_true_dvdy = evaluate_expr(expr, X, dvdy_true if dvdy_sym in expr.free_symbols else None)
        mse_true_dvdy = float(np.mean((pred_true_dvdy - u_true) ** 2))
        r2_true_dvdy = float(compute_r2(u_true, pred_true_dvdy))

        expr_sub = expr.subs(dvdy_sym, g_expr)
        pred_sub = evaluate_expr(expr_sub, X)
        mse_sub = float(np.mean((pred_sub - u_true) ** 2))
        r2_sub = float(compute_r2(u_true, pred_sub))

        rows.append(
            {
                "complexity": int(row["complexity"]),
                "has_dvdy": bool(dvdy_sym in expr.free_symbols),
                "original_mse": mse_true_dvdy,
                "original_r2": r2_true_dvdy,
                "substituted_mse": mse_sub,
                "substituted_r2": r2_sub,
                "delta_mse": mse_sub - mse_true_dvdy,
                "delta_r2": r2_sub - r2_true_dvdy,
                "equation": row["equation"],
                "sympy": row["sympy"],
                "substituted_sympy": str(expr_sub),
            }
        )

    results_df = pd.DataFrame(rows).sort_values(["substituted_mse", "complexity"]).reset_index(drop=True)
    return results_df


def make_figure(fig_path, x_vals, y_vals, dvdy_true_2d, g_pred_2d, sub_df, a_best_mse):
    err_2d = g_pred_2d - dvdy_true_2d
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    vmax = max(np.abs(dvdy_true_2d).max(), np.abs(g_pred_2d).max())
    for ax, data, title in zip(
        axes[0],
        [dvdy_true_2d, g_pred_2d],
        ["True dvdy", "Surrogate g(x_D, y_D)"],
    ):
        im = ax.pcolormesh(
            x_vals,
            y_vals,
            data.T,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            shading="auto",
        )
        ax.set_xlabel("x/D")
        ax.set_ylabel("y/D")
        ax.set_title(title)
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax)

    err_max = np.abs(err_2d).max()
    im = axes[1, 0].pcolormesh(
        x_vals,
        y_vals,
        err_2d.T,
        cmap="RdBu_r",
        vmin=-err_max,
        vmax=err_max,
        shading="auto",
    )
    axes[1, 0].set_xlabel("x/D")
    axes[1, 0].set_ylabel("y/D")
    axes[1, 0].set_title("Surrogate Error: g - dvdy")
    axes[1, 0].set_aspect("equal")
    plt.colorbar(im, ax=axes[1, 0])

    top = sub_df.sort_values("substituted_mse").head(6).sort_values("complexity")
    axes[1, 1].plot(
        top["complexity"],
        top["original_mse"],
        "o-",
        color="gray",
        label="B with true dvdy",
    )
    axes[1, 1].plot(
        top["complexity"],
        top["substituted_mse"],
        "s-",
        color="steelblue",
        label="B with surrogate g",
    )
    axes[1, 1].axhline(a_best_mse, color="coral", linestyle="--", label="A best closed form")
    axes[1, 1].set_xlabel("Strategy B Complexity")
    axes[1, 1].set_ylabel("u* MSE")
    axes[1, 1].set_title("Substitution Accuracy on Top B Formulas")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    run_out_dir = resolve_run_out_dir(
        BASE_OUT_DIR, out_dir=args.out_dir, out_subdir=args.out_subdir
    )
    fig_dir = os.path.join(run_out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print(f"Using run directory: {run_out_dir}")
    df = pd.read_csv(os.path.join(run_out_dir, "wake_data.csv"))
    with open(os.path.join(run_out_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    res_a = pd.read_csv(os.path.join(run_out_dir, "results_strategy_a.csv"))
    res_b = pd.read_csv(os.path.join(run_out_dir, "results_strategy_b.csv"))
    v_wake = np.load(os.path.join(run_out_dir, "v_mean_wake.npy"))

    X = df[["x_D", "y_D"]].values
    u_true = df["u_star"].values
    n_x = int(meta["n_x_wake"])
    n_y = int(meta["n_y"])
    D = float(meta["D"])
    dy_over_D = float(meta["dy"]) / D
    x_vals = np.unique(df["x_D"])
    y_vals = np.unique(df["y_D"])

    dvdy_true_2d = compute_dvdy(v_wake, dy_over_D)
    dvdy_true = dvdy_true_2d.flatten()
    print(
        f"Loaded {len(X)} points; dvdy range [{dvdy_true.min():.4f}, {dvdy_true.max():.4f}], "
        f"std={dvdy_true.std():.4f}"
    )

    g_results_df, g_expr, g_pred, model_path = fit_dvdy_surrogate(
        X, dvdy_true, args, run_out_dir
    )
    g_results_path = os.path.join(run_out_dir, "dvdy_regression_results.csv")
    g_results_df.to_csv(g_results_path, index=False)

    g_best_summary = {
        "equation": g_results_df.iloc[0]["equation"],
        "sympy": g_results_df.iloc[0]["sympy"],
        "complexity": int(g_results_df.iloc[0]["complexity"]),
        "mse": float(g_results_df.iloc[0]["mse"]),
        "mae": float(g_results_df.iloc[0]["mae"]),
        "r2": float(g_results_df.iloc[0]["r2"]),
    }
    with open(os.path.join(run_out_dir, "dvdy_regression_best.json"), "w", encoding="utf-8") as f:
        json.dump(g_best_summary, f, indent=2)

    sub_df = run_substitution_experiment(X, u_true, dvdy_true, res_b, g_expr, g_pred)
    sub_path = os.path.join(run_out_dir, "dvdy_substitution_results.csv")
    sub_df.to_csv(sub_path, index=False)

    a_best_mse = float(res_a["mse"].min())
    g_pred_2d = g_pred.reshape(n_x, n_y)
    fig_path = os.path.join(fig_dir, "12_dvdy_surrogate_substitution.png")
    make_figure(fig_path, x_vals, y_vals, dvdy_true_2d, g_pred_2d, sub_df, a_best_mse)

    best_sub = sub_df.iloc[0]
    best_original_b = res_b.loc[res_b["mse"].idxmin()]
    better_than_a = sub_df[sub_df["substituted_mse"] < a_best_mse]

    print("\n" + "=" * 60)
    print("dvdy SURROGATE SUMMARY")
    print("=" * 60)
    print(
        f"Best g complexity={g_best_summary['complexity']}  "
        f"mse={g_best_summary['mse']:.6f}  mae={g_best_summary['mae']:.6f}  "
        f"r2={g_best_summary['r2']:.4f}"
    )
    print(f"g(x_D, y_D) = {g_best_summary['equation']}")

    print("\n" + "=" * 60)
    print("SUBSTITUTION SUMMARY")
    print("=" * 60)
    print(f"Strategy A best closed-form mse = {a_best_mse:.6f}")
    print(
        f"Strategy B best with true dvdy: complexity={int(best_original_b['complexity'])}  "
        f"mse={float(best_original_b['mse']):.6f}"
    )
    print(
        f"Best substituted B formula: complexity={int(best_sub['complexity'])}  "
        f"substituted_mse={float(best_sub['substituted_mse']):.6f}  "
        f"original_mse={float(best_sub['original_mse']):.6f}  "
        f"delta={float(best_sub['delta_mse']):+.6f}"
    )
    print(
        f"Substituted formulas beating A best: {len(better_than_a)}/{len(sub_df)}"
    )
    if not better_than_a.empty:
        print("Top substituted formulas better than A best:")
        for _, row in better_than_a.head(5).iterrows():
            print(
                f"  B C={int(row['complexity']):>2d}  "
                f"sub_mse={float(row['substituted_mse']):.6f}  "
                f"orig_mse={float(row['original_mse']):.6f}"
            )
    else:
        print("No substituted B formula beats the current A best closed-form.")

    print(f"\nSaved: {g_results_path}")
    print(f"Saved: {model_path}")
    print(f"Saved: {sub_path}")
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
