"""Render COMSOL solution samples and mesh-comparison results locally."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    folder = args.folder
    state = json.loads((folder / "build_state.json").read_text(encoding="utf-8"))
    field = np.load(folder / "flow_field.npz")
    x = np.unique(field["x_over_h"])
    y = np.unique(field["y_over_h"])
    ix = np.searchsorted(x, field["x_over_h"])
    iy = np.searchsorted(y, field["y_over_h"])
    arrays = {}
    for name in ("u", "v", "p"):
        arr = np.full((len(y), len(x)), np.nan)
        arr[iy, ix] = field[name]
        arrays[name] = np.ma.masked_invalid(arr)
    u = arrays["u"] / float(field["Umean"])
    v = arrays["v"] / float(field["Umean"])
    speed = np.sqrt(u ** 2 + v ** 2)
    xr = state["refined"]["reattachment_x_over_h"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(13, 8), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=[0.8, 1.2, 1.25], width_ratios=[1, 0.025])
    fig.suptitle("Backward-facing step | 2D steady laminar flow | Re_h = 100", fontsize=16, x=0.07, ha="left")

    ax = fig.add_subplot(grid[0, 0])
    im = ax.pcolormesh(x, y, speed, shading="auto", cmap="viridis", vmin=0)
    ax.streamplot(x, y, u, v, color="#ffffff", density=(2.5, 0.5), linewidth=0.45, arrowsize=0.5)
    ax.add_patch(Rectangle((-5, 0), 5, 1, facecolor="#d8d8d8", edgecolor="#555555"))
    ax.set(xlim=(-5, 30), ylim=(0, 2), xlabel="x / h", ylabel="y / h", aspect="equal")
    ax.set_title("Full channel | expansion ratio 2 | parabolic inlet | pressure outlet", loc="left", fontsize=11)
    fig.colorbar(im, cax=fig.add_subplot(grid[0, 1]), label="|U| / Umean")

    ax = fig.add_subplot(grid[1, 0])
    keep = (x >= -1.5) & (x <= max(8, xr + 3))
    im = ax.pcolormesh(x[keep], y, u[:, keep], shading="auto", cmap="RdBu_r",
                       norm=TwoSlopeNorm(vmin=min(-0.1, float(u.min())), vcenter=0, vmax=float(u.max())))
    ax.streamplot(x[keep], y, u[:, keep], v[:, keep], color="#303030", density=(2.2, 1.0), linewidth=0.6, arrowsize=0.6)
    seeds = np.column_stack([np.linspace(0.7, max(0.8, xr - 0.7), 3), np.full(3, 0.2)])
    ax.streamplot(x[keep], y, u[:, keep], v[:, keep], color="#103d59", start_points=seeds,
                  linewidth=0.7, arrowsize=0.8, density=3, broken_streamlines=False)
    ax.add_patch(Rectangle((-5, 0), 5, 1, facecolor="#d8d8d8", edgecolor="#555555"))
    ax.plot(xr, 0.02, "o", color="#151515", markersize=5, clip_on=False)
    ax.annotate(f"Near-wall reattachment x/h = {xr:.3f}", xy=(xr, 0.02),
                xytext=(xr + 0.7, 0.5), arrowprops={"arrowstyle": "->", "lw": 0.8}, fontsize=10,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 3})
    ax.set(xlim=(x[keep][0], x[keep][-1]), ylim=(0, 2), xlabel="x / h", ylabel="y / h", aspect="equal")
    ax.set_title("Step detail | blue identifies reverse streamwise flow", loc="left", fontsize=11)
    fig.colorbar(im, cax=fig.add_subplot(grid[1, 1]), label="u / Umean", ticks=[-0.15, 0, 0.3, 0.6, 0.9, 1.2, 1.5])

    ax = fig.add_subplot(grid[2, 0])
    for label, style, color in [("baseline", "--", "#9a6030"), ("refined", "-", "#174f76")]:
        curve = np.loadtxt(folder / f"nearwall_{label}.csv", delimiter=",", skiprows=1)
        mesh = state[label]["mesh"]["elements"]
        ax.plot(curve[:, 0], curve[:, 2], style, color=color, linewidth=1.5,
                label=f"{label.capitalize()}: {mesh:,} elements")
    ax.axhline(0, color="#555555", linewidth=0.7)
    ax.axvline(xr, color="#999999", linewidth=0.7, linestyle=":")
    ax.set(xlim=(0, max(10, xr + 3)), xlabel="x / h", ylabel="u(y = 0.001h) / Umean")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(alpha=0.15)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Mesh comparison | near-wall velocity zero crossing", loc="left", fontsize=11)
    fig.savefig(folder / "backward_facing_step_overview.png", dpi=180, facecolor="white")
    plt.close(fig)

    ref = state["refined"]
    report = f"""# Backward-facing-step COMSOL model

Saved model on phy: `{state['saved_file']}`

## Definition

- COMSOL 6.4; 2D steady incompressible Newtonian laminar flow.
- Step height h = 0.01 m; upstream height h; downstream height 2h.
- Upstream length 5h; downstream length 30h; step at x=0.
- Density 1000 kg/m^3; dynamic viscosity 0.001 Pa s.
- Inlet mean speed 0.01 m/s; parabolic peak speed 0.015 m/s.
- Re_h = rho Umean h / mu = 100; inlet hydraulic-diameter Reynolds number = 200.
- No-slip walls; outlet gauge pressure 0 Pa.
- Coordinate-based named selections and a graded triangular mesh.
- Stationary solver relative tolerance: 1e-6.
- Computed in an independent COMSOL process using 8 processor cores.

## Numerical checks

| Quantity | Value |
|---|---:|
| Final mesh elements | {ref['mesh']['elements']} |
| Inlet flow per unit span, m^2/s | {ref['inlet_flux_m2_s']:.9g} |
| Outlet flow per unit span, m^2/s | {ref['outlet_flux_m2_s']:.9g} |
| Relative flow imbalance | {ref['relative_flux_imbalance']:.3g} |
| Minimum streamwise velocity, m/s | {ref['u_min_m_s']:.8g} |
| Near-wall reattachment x/h | {ref['reattachment_x_over_h']:.6f} |
| Reattachment change between two meshes | {state['reattachment_relative_mesh_change']:.3%} |

Reattachment is estimated from the negative-to-positive streamwise-velocity crossing
at y/h=0.001, checked against y/h=0.002. This is a near-wall estimate, not a direct
wall-shear zero calculation. The two-grid comparison checks mesh sensitivity;
it is not a formal convergence-order or independent benchmark validation.

## Files

- `backward_facing_step_overview.png`: full channel, recirculation, mesh comparison.
- `flow_field.csv`: x/h, y/h, u, v, and p sampled from the final COMSOL solution.
- `flow_field.npz`: the same field data in a compact NumPy archive.
- `nearwall_baseline.csv`, `nearwall_refined.csv`: near-wall velocity samples.
- `build_state.json`: parameters, mesh counts, diagnostics, and saved-model metadata.

The existing `three_cylinder_flow_2d` model was left unchanged.
"""
    (folder / "README.md").write_text(report, encoding="utf-8")
    print(folder / "backward_facing_step_overview.png")


if __name__ == "__main__":
    main()
