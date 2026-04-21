"""
Same-ops Pareto contour: A (maxsize=45) vs B+g substituted.

Produces:
  - outputs_mgn/eta_sym/pareto_compare.csv
  - outputs_mgn/eta_sym/figures/13_pareto_a45_vs_bg.png
"""
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.size"] = 11
rcParams["figure.dpi"] = 150

ETA_DIR = r"D:\keyan\cylinder_flow_pisr\outputs_mgn\eta_sym"
ARCHIVE_DIR = os.path.join(ETA_DIR, "archive_dvdy_substitution")
A45_DIR = os.path.join(ARCHIVE_DIR, "a_maxsize45")
FIG_DIR = ARCHIVE_DIR

a_old = pd.read_csv(os.path.join(ETA_DIR, "results_strategy_a.csv"))
a_new = pd.read_csv(os.path.join(A45_DIR, "results_strategy_a.csv"))
bg = pd.read_csv(os.path.join(ETA_DIR, "substitution_verification.csv"))

bg_keep = bg[bg["has_dvdy"]].copy()
bg_keep["ops"] = bg_keep["expanded_ops"]
bg_keep["mse"] = bg_keep["mse_substituted"]
bg_keep["div_rms"] = bg_keep["div_rms_substituted"]
bg_keep = bg_keep[["ops", "mse", "div_rms"]].assign(strategy="B+g(sub)")

a_old_sub = a_old[["complexity", "mse", "div_rms"]].rename(columns={"complexity": "ops"})
a_old_sub = a_old_sub.assign(strategy="A (max=25)")

a_new_sub = a_new[["ops", "mse", "div_rms"]].assign(strategy="A (max=45)")

all_rows = pd.concat([a_old_sub, a_new_sub, bg_keep], ignore_index=True)
all_rows = all_rows.sort_values(["strategy", "ops"]).reset_index(drop=True)
all_rows.to_csv(os.path.join(ETA_DIR, "pareto_compare.csv"), index=False)

print("=" * 80)
print("Pareto comparison (same-ops contour)")
print("=" * 80)
print(f"{'strategy':>12} {'ops':>4} {'mse':>12} {'div_rms':>10}")
print("-" * 80)
for _, r in all_rows.iterrows():
    print(f"{r['strategy']:>12} {int(r['ops']):>4d} {r['mse']:>12.6f} {r['div_rms']:>10.4f}")


def pareto_front(df, x_col="ops", y_col="mse"):
    s = df.sort_values(x_col).reset_index(drop=True)
    front = []
    best_y = np.inf
    for _, row in s.iterrows():
        if row[y_col] < best_y:
            front.append(row)
            best_y = row[y_col]
    return pd.DataFrame(front)


fig, axes = plt.subplots(1, 2, figsize=(13, 5))

styles = {
    "A (max=25)": {"color": "tab:orange", "marker": "o", "label": "A, maxsize=25 (baseline)"},
    "A (max=45)": {"color": "tab:red", "marker": "s", "label": "A, maxsize=45"},
    "B+g(sub)": {"color": "tab:blue", "marker": "^", "label": "B + g substituted"},
}

for strat, style in styles.items():
    sub = all_rows[all_rows["strategy"] == strat]
    axes[0].scatter(sub["ops"], sub["mse"], alpha=0.5, **{k: v for k, v in style.items() if k != "label"},
                    s=40)
    front = pareto_front(sub, "ops", "mse")
    axes[0].plot(front["ops"], front["mse"], "-", color=style["color"], linewidth=2,
                 label=style["label"])

axes[0].set_xlabel("Expanded complexity (sp.count_ops)")
axes[0].set_ylabel("MSE")
axes[0].set_yscale("log")
axes[0].set_title("Pareto: MSE vs expanded complexity")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

for strat, style in styles.items():
    sub = all_rows[all_rows["strategy"] == strat]
    axes[1].scatter(sub["ops"], sub["div_rms"], alpha=0.5,
                    **{k: v for k, v in style.items() if k != "label"}, s=40)

axes[1].set_xlabel("Expanded complexity (sp.count_ops)")
axes[1].set_ylabel("div_rms")
axes[1].set_title("Physics consistency: div_rms vs complexity")
axes[1].grid(True, alpha=0.3)
axes[1].legend([v["label"] for v in styles.values()])

fig.tight_layout()
fig_path = os.path.join(FIG_DIR, "13_pareto_a45_vs_bg.png")
fig.savefig(fig_path, bbox_inches="tight")
plt.close()

print(f"\nSaved: {os.path.join(ETA_DIR, 'pareto_compare.csv')}")
print(f"Saved: {fig_path}")

print("\n--- Key comparisons at matched ops ---")

def best_at_ops(df, ops_min, ops_max):
    sub = df[(df["ops"] >= ops_min) & (df["ops"] <= ops_max)]
    if len(sub) == 0:
        return None
    return sub.loc[sub["mse"].idxmin()]


for ops_window in [(12, 15), (20, 25), (38, 45)]:
    print(f"\nops window {ops_window}:")
    for strat in ["A (max=25)", "A (max=45)", "B+g(sub)"]:
        sub = all_rows[all_rows["strategy"] == strat]
        r = best_at_ops(sub, *ops_window)
        if r is not None:
            print(f"  {strat:>12}  ops={int(r['ops']):>3d}  mse={r['mse']:.6f}  div={r['div_rms']:.4f}")
        else:
            print(f"  {strat:>12}  (no entry)")
