"""
Step 5: Finite-difference sensitivity check against PISR predictions.

Loads 7 COMSOL runs (base + 6 single-parameter perturbations R-/R+, cx-/cx+, cy-/cy+),
time-averages u(x,y), interpolates onto a common normalized (x_D, y_D) grid using
each run's own (cx, cy, D) and inlet-estimated U_in, then computes central
finite-difference sensitivities:

    S_R   ~= (u*_R+  - u*_R-)  / (2*dR)
    S_cx  ~= (u*_cx+ - u*_cx-) / (2*dcx)
    S_cy  ~= (u*_cy+ - u*_cy-) / (2*dcy)

and compares |S|_mean / |S|_max with the PISR psi-B prediction at
outputs_mgn/comsol/eta_sym_psi/sensitivity_results.csv  (strategy=B, complexity=21).
"""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

REPO = Path(r"D:\keyan\cylinder_flow_pisr")
OUT_DIR = REPO / "outputs_mgn" / "fd_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Perturbation table (matches user's spec).
CASES = {
    "base": dict(file="cylinder_flow_result", D=0.100, cx=0.200, cy=0.200, Umax=1.5000),
    "R-":   dict(file="R-.txt",                D=0.098, cx=0.200, cy=0.200, Umax=1.5306),
    "R+":   dict(file="R+.txt",                D=0.102, cx=0.200, cy=0.200, Umax=1.4706),
    "cx-":  dict(file="cx-.txt",               D=0.100, cx=0.198, cy=0.200, Umax=1.5000),
    "cx+":  dict(file="cx+.txt",               D=0.100, cx=0.202, cy=0.200, Umax=1.5000),
    "cy-":  dict(file="cy-.txt",               D=0.100, cx=0.200, cy=0.198, Umax=1.5000),
    "cy+":  dict(file="cy+.txt",               D=0.100, cx=0.200, cy=0.202, Umax=1.5000),
}
# Finite-difference step sizes (m):
dR, dcx, dcy = 0.001, 0.002, 0.002

T_START_FRAC = 0.5         # use second half of time history
INLET_WIDTH  = 0.02

# Use the same (x_D, y_D) grid as the base wake_data.csv for direct comparison.
BASE_RUN_DIR = REPO / "outputs_mgn" / "comsol" / "eta_sym_psi"


def read_comsol(path):
    """Return (mesh_pos[N,2], u_time[N,T], v_time[N,T])."""
    # Parse header for n_time:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped.startswith("%"):
                first = stripped
                break
    n_cols = len(first.split())
    n_time = (n_cols - 2) // 6
    usecols = [0, 1]
    for i in range(n_time):
        usecols.extend([2 + 6 * i, 3 + 6 * i])
    data = np.loadtxt(path, comments="%", usecols=usecols, dtype=np.float64)
    coords = data[:, :2]
    u_time = data[:, 2::2]
    v_time = data[:, 3::2]
    return coords, u_time, v_time, n_time


def estimate_uin(coords, u_mean, x_min, inlet_width):
    mask = coords[:, 0] <= x_min + inlet_width
    if not np.any(mask):
        x0 = coords[:, 0].min()
        mask = np.isclose(coords[:, 0], x0)
    u_in = float(np.mean(u_mean[mask]))
    return u_in if abs(u_in) > 1e-8 else 1.0


def load_case(case_name, info):
    path = REPO / info["file"]
    print(f"[{case_name}] loading {path.name} ...")
    coords, u_time, v_time, n_time = read_comsol(path)
    t0 = int(n_time * T_START_FRAC)
    u_mean = u_time[:, t0:].mean(axis=1)
    v_mean = v_time[:, t0:].mean(axis=1)
    x_min = float(coords[:, 0].min())
    u_in = estimate_uin(coords, u_mean, x_min, INLET_WIDTH)
    print(f"    n_nodes={coords.shape[0]:5d}  n_time={n_time}  U_in_est={u_in:.6f}  Umax={info['Umax']:.4f}")
    return dict(coords=coords, u_mean=u_mean, v_mean=v_mean, U_in=u_in, **info)


def interp_to_normalized_grid(case, x_D, y_D):
    """Map normalized grid (x_D, y_D) -> physical (x,y) using *this run's* cx, cy, D,
    then linearly interpolate u_mean. Return u*/U_in array of shape (n_x*n_y,).
    """
    XX, YY = np.meshgrid(x_D, y_D, indexing="ij")
    phys_x = case["cx"] + XX.ravel() * case["D"]
    phys_y = case["cy"] + YY.ravel() * case["D"]
    pts = np.column_stack([phys_x, phys_y])
    u_lin = griddata(case["coords"], case["u_mean"], pts, method="linear")
    nan_mask = np.isnan(u_lin)
    if nan_mask.any():
        u_near = griddata(case["coords"], case["u_mean"], pts, method="nearest")
        u_lin[nan_mask] = u_near[nan_mask]
    return u_lin / case["U_in"]


def main():
    # --- shared normalized grid taken straight from psi-B run wake_data.csv ----
    df_base = pd.read_csv(BASE_RUN_DIR / "wake_data.csv")
    with open(BASE_RUN_DIR / "meta.json", "r", encoding="utf-8") as f:
        base_meta = json.load(f)
    n_x = int(base_meta["n_x_wake"])
    n_y = int(base_meta["n_y"])
    x_D = np.sort(df_base["x_D"].unique())
    y_D = np.sort(df_base["y_D"].unique())
    assert x_D.size == n_x and y_D.size == n_y, "grid mismatch with meta.json"
    print(f"Common (x_D, y_D) grid: n_x={n_x}, n_y={n_y}, "
          f"x_D in [{x_D[0]:.3f},{x_D[-1]:.3f}], y_D in [{y_D[0]:.3f},{y_D[-1]:.3f}]")

    # --- load all 7 cases ----
    cases = {name: load_case(name, info) for name, info in CASES.items()}

    # --- interpolate u* to common normalized grid for each case ----
    u_star = {}
    for name, c in cases.items():
        u_star[name] = interp_to_normalized_grid(c, x_D, y_D).reshape(n_x, n_y)

    # --- central finite-difference sensitivities -------------------------------
    S_R_fd  = (u_star["R+"]  - u_star["R-"])  / (2 * dR)
    S_cx_fd = (u_star["cx+"] - u_star["cx-"]) / (2 * dcx)
    S_cy_fd = (u_star["cy+"] - u_star["cy-"]) / (2 * dcy)

    def stats(name, S):
        return dict(
            param=name,
            mean_abs=float(np.mean(np.abs(S))),
            max_abs=float(np.max(np.abs(S))),
            rms=float(np.sqrt(np.mean(S ** 2))),
        )

    fd_rows = [stats("R", S_R_fd), stats("cx", S_cx_fd), stats("cy", S_cy_fd)]

    # --- PISR psi-B prediction at C=21 ----------------------------------------
    pred_df = pd.read_csv(BASE_RUN_DIR / "sensitivity_results.csv")
    pred = pred_df[(pred_df["strategy"] == "B") & (pred_df["complexity"] == 21)].iloc[0]
    pred_rows = [
        dict(param="R",  mean_abs=float(pred["S_R_integrated"]),  max_abs=float(pred["S_R_max"])),
        dict(param="cx", mean_abs=float(pred["S_cx_integrated"]), max_abs=float(pred["S_cx_max"])),
        dict(param="cy", mean_abs=float(pred["S_cy_integrated"]), max_abs=float(pred["S_cy_max"])),
    ]

    # --- print summary table ---------------------------------------------------
    summary = pd.DataFrame({
        "param":         ["R", "cx", "cy"],
        "FD |S|_mean":   [r["mean_abs"] for r in fd_rows],
        "FD |S|_max":    [r["max_abs"]  for r in fd_rows],
        "FD RMS":        [r["rms"]      for r in fd_rows],
        "psi-B |S|_mean":[r["mean_abs"] for r in pred_rows],
        "psi-B |S|_max": [r["max_abs"]  for r in pred_rows],
        "ratio mean":    [fd_rows[i]["mean_abs"] / pred_rows[i]["mean_abs"]
                          for i in range(3)],
        "ratio max":     [fd_rows[i]["max_abs"]  / pred_rows[i]["max_abs"]
                          for i in range(3)],
    })
    print("\n" + "=" * 70)
    print("Finite-difference vs PISR psi-B (C=21) sensitivity comparison")
    print("=" * 70)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))

    rank_fd   = sorted(fd_rows,   key=lambda r: -r["mean_abs"])
    rank_pred = sorted(pred_rows, key=lambda r: -r["mean_abs"])
    print("\nRanking by |S|_mean (largest -> smallest):")
    print(f"  Finite-difference (CFD):  {' > '.join(r['param'] for r in rank_fd)}")
    print(f"  PISR psi-B prediction:    {' > '.join(r['param'] for r in rank_pred)}")
    match = [r["param"] for r in rank_fd] == [r["param"] for r in rank_pred]
    print(f"  Ranking match? {'YES' if match else 'NO'}")

    summary.to_csv(OUT_DIR / "fd_vs_pisr_summary.csv", index=False)

    # --- save raw FD fields ----------------------------------------------------
    np.savez(OUT_DIR / "fd_sensitivity_fields.npz",
             x_D=x_D, y_D=y_D,
             u_base=u_star["base"],
             u_Rp=u_star["R+"], u_Rn=u_star["R-"],
             u_cxp=u_star["cx+"], u_cxn=u_star["cx-"],
             u_cyp=u_star["cy+"], u_cyn=u_star["cy-"],
             S_R=S_R_fd, S_cx=S_cx_fd, S_cy=S_cy_fd)

    # --- Figure 1: u* at base ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.pcolormesh(x_D, y_D, u_star["base"].T, cmap="viridis", shading="auto")
    ax.set_xlabel("x/D"); ax.set_ylabel("y/D"); ax.set_aspect("equal")
    ax.set_title("Base u* (time-averaged, normalized by inlet U_in)")
    plt.colorbar(im, ax=ax, label="u*")
    fig.tight_layout(); fig.savefig(OUT_DIR / "01_u_star_base.png", dpi=150); plt.close()

    # --- Figure 2: 3-panel FD sensitivity heatmaps -----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, S, name in zip(axes, [S_R_fd, S_cx_fd, S_cy_fd], ["S_R (FD)", "S_cx (FD)", "S_cy (FD)"]):
        lim = max(np.max(np.abs(S)), 1e-12)
        im = ax.pcolormesh(x_D, y_D, S.T, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
        ax.set_xlabel("x/D"); ax.set_ylabel("y/D"); ax.set_aspect("equal")
        ax.set_title(f"{name}  |S|_mean={np.mean(np.abs(S)):.3f}, |S|_max={lim:.3f}")
        plt.colorbar(im, ax=ax)
    fig.suptitle("Central finite-difference sensitivity from 6 perturbed COMSOL runs", y=1.02)
    fig.tight_layout(); fig.savefig(OUT_DIR / "02_fd_sensitivity_heatmaps.png", dpi=150,
                                    bbox_inches="tight"); plt.close()

    # --- Figure 3: bar chart of |S|_mean (FD vs PISR) --------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = ["R", "cx", "cy"]
    fd_mean   = [r["mean_abs"] for r in fd_rows]
    pred_mean = [r["mean_abs"] for r in pred_rows]
    fd_max    = [r["max_abs"]  for r in fd_rows]
    pred_max  = [r["max_abs"]  for r in pred_rows]
    xpos = np.arange(3)
    w = 0.38
    axes[0].bar(xpos - w/2, fd_mean,   w, label="FD (CFD)",    color="steelblue")
    axes[0].bar(xpos + w/2, pred_mean, w, label="PISR psi-B",  color="coral")
    axes[0].set_xticks(xpos); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("|S|_mean")
    axes[0].set_title("Mean |sensitivity|"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].bar(xpos - w/2, fd_max,   w, label="FD (CFD)",    color="steelblue")
    axes[1].bar(xpos + w/2, pred_max, w, label="PISR psi-B",  color="coral")
    axes[1].set_xticks(xpos); axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("|S|_max")
    axes[1].set_title("Max |sensitivity|"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.suptitle("Sensitivity magnitude: CFD finite difference vs PISR prediction", y=1.02)
    fig.tight_layout(); fig.savefig(OUT_DIR / "03_fd_vs_pisr_bar.png", dpi=150,
                                    bbox_inches="tight"); plt.close()

    print(f"\nAll outputs saved to: {OUT_DIR}")
    return summary, match


if __name__ == "__main__":
    main()