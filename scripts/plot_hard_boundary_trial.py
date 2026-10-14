"""Plot saved hard-boundary results without importing Torch."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.trial_dir.resolve()
    summary = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
    runs = summary["runs"]
    names = [key for key in ("soft", "cylinder_poiseuille", "zero", "poiseuille") if key in runs]
    labels = {"soft": "Soft neural", "cylinder_poiseuille": "Corrected analytic",
              "zero": "Zero (invalid BC)", "poiseuille": "Raw Poiseuille (invalid BC)"}
    colors = {"soft": "#686c73", "cylinder_poiseuille": "#007e87", "zero": "#c48739", "poiseuille": "#ae4e62"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0, 0]
    for key in ("soft", "cylinder_poiseuille"):
        if key in runs:
            sections = runs[key]["split_flux"]["sections"]
            ax.plot([r["x"] for r in sections], [100*r["relative_error"] for r in sections],
                    "o-", markersize=3, color=colors[key], label=labels[key])
    ax.axhspan(-1, 1, color=colors["cylinder_poiseuille"], alpha=.12, label="1% target")
    ax.set(xlabel="x / diameter", ylabel="Signed flux error (%)", title="A  Admissible models: cross-section flux")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    positions = np.arange(len(names))
    momentum_values = [runs[name]["pde_rms"]["momentum_"+axis] for name in names for axis in ("x", "y")]
    wide_range = min(momentum_values) > 0 and max(momentum_values)/min(momentum_values) > 100
    for i, (component, color) in enumerate((("x", "#305d85"), ("y", "#83adb8"))):
        bars = ax.bar(positions+(i-.5)*.34, [runs[n]["pde_rms"]["momentum_"+component] for n in names],
                      width=.34, label="Momentum " + component, color=color)
        if not wide_range:
            ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    if wide_range:
        ax.set_yscale("log")
        ax.set_ylim(min(momentum_values)*.6, max(momentum_values)*2)
    else:
        ax.set_ylim(0, max(momentum_values)*1.25)
    ax.set_xticks(positions)
    ax.set_xticklabels([labels[n].replace(" (", "\n(") for n in names], fontsize=8)
    title = ("B  Momentum: invalid BC controls cannot qualify" if any(n in names for n in ("zero", "poiseuille"))
             else "B  Momentum: identical uniform audit for both models")
    ax.set(ylabel="RMS on 20,000 independent points", title=title)
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    keys = ("inlet_velocity_error", "bottom_velocity", "top_velocity", "cylinder_velocity")
    for name in names:
        values = [max(runs[name]["boundary_errors"][key]["rms"], 1e-16) for key in keys]
        ax.plot(np.arange(4), values, "o-", markersize=4, label=labels[name], color=colors[name])
    ax.set_yscale("log")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(["Inlet", "Bottom", "Top", "Cylinder"])
    ax.set(ylabel="Velocity boundary RMS (floor 1e-16)", title="C  Velocity boundary verification")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    for name in names:
        with (root / name / "history.csv").open(encoding="utf-8") as stream:
            history = list(csv.DictReader(stream))
        updates = [int(r["step"])+(summary["protocol"]["adam_steps"] if r["phase"] == "lbfgs" else 0) for r in history]
        ax.plot(updates, [float(r["total"]) for r in history], color=colors[name], label=labels[name])
    ax.axvline(summary["protocol"]["adam_steps"], color="#aaa", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set(xlabel="Optimizer updates (Adam then L-BFGS)", ylabel="Total training loss",
           title="D  Identical loss weights and update budget; no SR")
    ax.legend(fontsize=8)
    stage = "Matched refinement" if summary["protocol"].get("init_trial") else "Initial trial"
    fig.suptitle(stage + " | Hard velocity boundaries in a streamfunction PINN | Re=20, seed="+str(summary["seed"]), fontsize=13)
    for extension in ("png", "pdf"):
        fig.savefig(root / ("comparison."+extension))
    plt.close(fig)
    print(root / "comparison.png")


if __name__ == "__main__":
    main()
