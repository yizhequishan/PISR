"""
Strategy A rerun on eta_sym data with maxsize=45 to get c~40 Pareto baseline.

Outputs to outputs_mgn/eta_sym/a_maxsize45/ to avoid overwriting the original
maxsize=25 run.
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
import sympy as sp


ETA_DIR = r"D:\keyan\cylinder_flow_pisr\outputs_mgn\eta_sym"
OUT_DIR = os.path.join(ETA_DIR, "a_maxsize45")
os.makedirs(OUT_DIR, exist_ok=True)


def as_array(vals, n):
    if np.isscalar(vals):
        return np.full(n, float(vals))
    arr = np.array(vals, dtype=float)
    if arr.shape == ():
        return np.full(n, float(arr))
    return arr.reshape(-1)


def compute_dvdy(v_wake, dy_over_D):
    dvdy = np.zeros_like(v_wake)
    if v_wake.shape[1] > 1:
        dvdy[:, 1:-1] = (v_wake[:, 2:] - v_wake[:, :-2]) / (2 * dy_over_D)
        dvdy[:, 0] = (v_wake[:, 1] - v_wake[:, 0]) / dy_over_D
        dvdy[:, -1] = (v_wake[:, -1] - v_wake[:, -2]) / dy_over_D
    return dvdy


df = pd.read_csv(os.path.join(ETA_DIR, "wake_data.csv"))
with open(os.path.join(ETA_DIR, "meta.json"), encoding="utf-8") as f:
    meta = json.load(f)
v_wake = np.load(os.path.join(ETA_DIR, "v_mean_wake.npy"))

X = df[["x_D", "y_D"]].values
y = df["u_star"].values
D = float(meta["D"])
dy_over_D = float(meta["dy"]) / D
dvdy_flat = compute_dvdy(v_wake, dy_over_D).flatten()

print(f"Data: {len(X)} points, eta in "
      f"[{X[:, 1].min():.3f}, {X[:, 1].max():.3f}]")

from pysr import PySRRegressor

print("=" * 60)
print("Strategy A, maxsize=45")
print("=" * 60)

model_a = PySRRegressor(
    niterations=30,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "cos", "exp", "square"],
    populations=20,
    population_size=40,
    maxsize=45,
    extra_sympy_mappings={"square": lambda x: x**2},
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    temp_equation_file=True,
    delete_tempfiles=False,
    verbosity=0,
    progress=False,
)
model_a.fit(X, y, variable_names=["x_D", "y_D"])

x_sym, y_sym = sp.symbols("x_D y_D")
rows = []
for eq_idx, row in model_a.equations_.iterrows():
    expr = row["sympy_format"]
    has_x = x_sym in expr.free_symbols
    ops = int(sp.count_ops(expr))

    try:
        dudx_expr = sp.diff(expr, x_sym)
        dudx_func = sp.lambdify([x_sym, y_sym], dudx_expr, modules="numpy")
        dudx_vals = as_array(dudx_func(X[:, 0], X[:, 1]), len(X))

        pred_func = sp.lambdify([x_sym, y_sym], expr, modules="numpy")
        pred = as_array(pred_func(X[:, 0], X[:, 1]), len(X))

        mse = float(np.mean((pred - y) ** 2))
        div = dudx_vals + dvdy_flat
        div_rms = float(np.sqrt(np.mean(div**2)))
        div_max = float(np.max(np.abs(div)))
    except Exception as e:
        print(f"  c={row['complexity']} EVAL ERROR: {e}")
        continue

    rows.append({
        "equation_index": int(eq_idx),
        "complexity": int(row["complexity"]),
        "ops": ops,
        "mse": mse,
        "div_rms": div_rms,
        "div_max": div_max,
        "has_x_D": bool(has_x),
        "equation": str(row["equation"]),
        "sympy": str(expr),
    })
    print(f"  c={int(row['complexity']):>2d}  ops={ops:>3d}  "
          f"mse={mse:.6f}  div_rms={div_rms:.4f}  "
          f"x_D={'X' if has_x else '.'}  {row['equation'][:60]}")

res_df = pd.DataFrame(rows)
res_df.to_csv(os.path.join(OUT_DIR, "results_strategy_a.csv"), index=False)
with open(os.path.join(OUT_DIR, "pysr_model_a.pkl"), "wb") as f:
    pickle.dump(model_a, f)

print(f"\nSaved: {OUT_DIR}")
print(f"Best: c={int(res_df.loc[res_df['mse'].idxmin(), 'complexity'])}  "
      f"mse={float(res_df['mse'].min()):.6f}")
