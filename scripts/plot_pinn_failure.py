"""Plot an existing failure diagnosis in a process that does not import Torch."""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.diagnosis_dir.resolve()
    diagnosis = json.loads((out/"diagnosis.json").read_text(encoding="utf-8"))
    with (Path(diagnosis["run_dir"])/"history.csv").open(encoding="utf-8", newline="") as stream:
        history = list(csv.DictReader(stream))
    adam_steps = diagnosis["checkpoint_progress"]["adam_steps"]
    steps = [int(row["step"])+(adam_steps if row["phase"] == "lbfgs" else 0) for row in history]
    losses = [float(row["total"]) for row in history]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), constrained_layout=True)
    blue, red, green = "#24608A", "#C84938", "#237C64"

    ax = axes[0, 0]
    ax.semilogy(steps, losses, color=blue, linewidth=1.8, label="Training (current sampled points)")
    validation = diagnosis["original_validation"]["weighted_losses"]["total"]
    ax.scatter([steps[-1]], [validation], color=red, marker="D", zorder=4, label="Final independent validation")
    ax.axvline(adam_steps, color="0.55", linewidth=1, linestyle="--")
    ax.text(adam_steps-180, .3, "L-BFGS starts", rotation=90, va="bottom", ha="right", color="0.35", fontsize=9)
    ax.set(xlabel="Optimizer updates", ylabel="Weighted loss", title="A  Budget ended while loss was still decreasing")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.grid(alpha=.15)

    ax = axes[0, 1]
    sections = diagnosis["mass_conservation"][-1]["sections"]
    x, flux = [r["x"] for r in sections], [r["relative_to_target"] for r in sections]
    ax.axhline(1, color=green, linestyle="--", linewidth=1.5, label="Required flux")
    ax.axvspan(1.5, 2.5, color="0.88", label="Cylinder x-range")
    ax.plot(x, flux, color=red, linewidth=2, label="Checkpoint prediction")
    ax.scatter([x[-1]], [flux[-1]], color=red)
    ax.annotate(f"Outlet: {flux[-1]:.1%}", (x[-1], flux[-1]), xytext=(-5, 26),
                textcoords="offset points", ha="right", color=red)
    ax.set(xlabel="x / D", ylabel="Q(x) / prescribed inlet flux", ylim=(0, 1.14),
           title="B  Mass flux falls throughout the channel")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.grid(alpha=.15)

    ax = axes[1, 0]
    names = ("continuity", "momentum_x", "momentum_y")
    labels = ("Continuity", "x momentum", "y momentum")
    training = [np.sqrt(diagnosis["final_training"]["terms_mse"]["pde_"+name]) for name in names]
    independent = [diagnosis["fresh_residuals"][name]["rms"] for name in names]
    locations = np.arange(3)
    ax.bar(locations-.19, training, .36, color=blue, label="Final collocation set (2,048)")
    ax.bar(locations+.19, independent, .36, color=red,
           label=f"Fresh uniform points ({diagnosis['fresh_sample_size']:,})")
    for positions, values in ((locations-.19, training), (locations+.19, independent)):
        for position, value in zip(positions, values):
            ax.text(position, value+.002, f"{value:.3f}", ha="center", fontsize=9)
    ax.set(xticks=locations, xticklabels=labels, ylabel="Residual RMS", ylim=(0, .17),
           title="C  Training points understate the residual")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(axis="y", alpha=.15)

    ax = axes[1, 1]
    regions = diagnosis["spatial_regions"]
    positions = np.arange(len(regions))
    sample_share = np.array([r["test_points"] for r in regions])/diagnosis["fresh_sample_size"]
    residual_share = [r["continuity_squared_residual_share"] for r in regions]
    ax.bar(positions-.19, sample_share, .36, color="0.7", label="Fraction of uniform test points")
    ax.bar(positions+.19, residual_share, .36, color=red, label="Share of squared continuity residual")
    ax.text(.19, residual_share[0]+.015, f"{residual_share[0]:.1%}", ha="center", color=red, fontsize=10)
    ax.set(xticks=positions, xticklabels=[f"{r['x_min']:g}-{r['x_max']:g}" for r in regions],
           xlabel="Streamwise region (x / D)", ylabel="Fraction", ylim=(0, 1.07),
           title="D  Residual concentrates before the cylinder")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.grid(axis="y", alpha=.15)

    fig.suptitle("First full-budget PINN: Re = 20, steady cylinder flow", fontsize=15)
    fig.savefig(out/"failure_summary.png", dpi=180)
    fig.savefig(out/"failure_summary.svg")
    plt.close(fig)
    print(out/"failure_summary.png")


if __name__ == "__main__":
    main()
