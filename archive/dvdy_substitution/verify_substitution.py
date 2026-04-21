"""
Verify the dvdy-substitution experiment on two axes:
(1) expanded complexity (sp.count_ops on the substituted closed form)
(2) divergence RMS of the substituted closed form against the data dvdy

Reads outputs from outputs_mgn/eta_sym/, writes verification CSV + summary.
"""
import json
import os

import numpy as np
import pandas as pd
import sympy as sp

RUN_DIR = r"D:\keyan\cylinder_flow_pisr\outputs_mgn\eta_sym"
x_sym, y_sym, dvdy_sym = sp.symbols("x_D y_D dvdy")


def compute_dvdy(v_wake, dy_over_D):
    dvdy = np.zeros_like(v_wake)
    if v_wake.shape[1] > 1:
        dvdy[:, 1:-1] = (v_wake[:, 2:] - v_wake[:, :-2]) / (2 * dy_over_D)
        dvdy[:, 0] = (v_wake[:, 1] - v_wake[:, 0]) / dy_over_D
        dvdy[:, -1] = (v_wake[:, -1] - v_wake[:, -2]) / dy_over_D
    return dvdy


def as_array(vals, n):
    if np.isscalar(vals):
        return np.full(n, float(vals))
    arr = np.array(vals, dtype=float)
    if arr.shape == ():
        return np.full(n, float(arr))
    return arr.reshape(-1)


def eval_expr(expr, X):
    free = expr.free_symbols
    if not free:
        return np.full(len(X), float(expr))
    f = sp.lambdify([x_sym, y_sym], expr, modules="numpy")
    return as_array(f(X[:, 0], X[:, 1]), len(X))


df = pd.read_csv(os.path.join(RUN_DIR, "wake_data.csv"))
with open(os.path.join(RUN_DIR, "meta.json"), encoding="utf-8") as f:
    meta = json.load(f)
res_b = pd.read_csv(os.path.join(RUN_DIR, "results_strategy_b.csv"))
sub = pd.read_csv(os.path.join(RUN_DIR, "dvdy_substitution_results.csv"))
v_wake = np.load(os.path.join(RUN_DIR, "v_mean_wake.npy"))

X = df[["x_D", "y_D"]].values
u_true = df["u_star"].values
n_x = int(meta["n_x_wake"])
n_y = int(meta["n_y"])
D = float(meta["D"])
dy_over_D = float(meta["dy"]) / D
dvdy_data = compute_dvdy(v_wake, dy_over_D).flatten()

res_b_idx = res_b.set_index("complexity")["div_rms"].to_dict()
res_b_sympy = res_b.set_index("complexity")["sympy"].to_dict()

rows = []
for _, row in sub.iterrows():
    c_b = int(row["complexity"])
    nominal_c = c_b
    orig_mse = float(row["original_mse"])
    sub_mse = float(row["substituted_mse"])
    has_dvdy = bool(row["has_dvdy"])

    expr_sub = sp.sympify(row["substituted_sympy"])
    expr_orig = sp.sympify(row["sympy"])

    expanded_ops = int(sp.count_ops(expr_sub))
    orig_ops = int(sp.count_ops(expr_orig))
    dvdy_occurrences = str(expr_orig).count("dvdy")

    try:
        dudx_expr = sp.diff(expr_sub, x_sym)
        dudx_vals = eval_expr(dudx_expr, X)
        div_after = dudx_vals + dvdy_data
        div_rms_after = float(np.sqrt(np.mean(div_after**2)))
    except Exception as e:
        div_rms_after = float("nan")
        print(f"  c_B={c_b}: div eval error: {e}")

    orig_div_rms = res_b_idx.get(c_b, float("nan"))

    rows.append({
        "c_B": c_b,
        "nominal_total_c": nominal_c,
        "expanded_ops": expanded_ops,
        "orig_ops": orig_ops,
        "dvdy_count": dvdy_occurrences,
        "mse_orig_true_dvdy": orig_mse,
        "mse_substituted": sub_mse,
        "div_rms_orig": float(orig_div_rms),
        "div_rms_substituted": div_rms_after,
        "has_dvdy": has_dvdy,
    })

out = pd.DataFrame(rows).sort_values("c_B").reset_index(drop=True)
out_path = os.path.join(RUN_DIR, "substitution_verification.csv")
out.to_csv(out_path, index=False)

print("=" * 90)
print("Substitution verification (eta_sym)")
print("=" * 90)
print(f"{'c_B':>4} {'exp_ops':>8} {'dvdy#':>6} {'mse_orig':>10} {'mse_sub':>10} "
      f"{'div_orig':>10} {'div_sub':>10}  has_dvdy")
print("-" * 90)
for _, r in out.iterrows():
    print(f"{int(r['c_B']):>4d} {int(r['expanded_ops']):>8d} "
          f"{int(r['dvdy_count']):>6d} {r['mse_orig_true_dvdy']:>10.6f} "
          f"{r['mse_substituted']:>10.6f} {r['div_rms_orig']:>10.4f} "
          f"{r['div_rms_substituted']:>10.4f}  {bool(r['has_dvdy'])}")

print(f"\nSaved: {out_path}")

res_a = pd.read_csv(os.path.join(RUN_DIR, "results_strategy_a.csv"))
a_best = res_a.loc[res_a["mse"].idxmin()]
print(f"\nReference: A best complexity={int(a_best['complexity'])}  "
      f"mse={float(a_best['mse']):.6f}  "
      f"div_rms={float(a_best['div_rms']):.4f}")

print("\n--- Pareto by expanded complexity ---")
print("(lower-left is better)")
a_rows = res_a.assign(which="A", expanded=res_a["complexity"], mse_eff=res_a["mse"],
                     div_eff=res_a["div_rms"])[["which", "expanded", "mse_eff", "div_eff"]]
b_rows = out[out["has_dvdy"]].assign(
    which="B+g",
    expanded=out[out["has_dvdy"]]["expanded_ops"],
    mse_eff=out[out["has_dvdy"]]["mse_substituted"],
    div_eff=out[out["has_dvdy"]]["div_rms_substituted"],
)[["which", "expanded", "mse_eff", "div_eff"]]
combined = pd.concat([a_rows, b_rows], ignore_index=True).sort_values("expanded")
for _, r in combined.iterrows():
    print(f"  {r['which']:>4}  ops={int(r['expanded']):>3d}  "
          f"mse={float(r['mse_eff']):.6f}  div={float(r['div_eff']):.4f}")
