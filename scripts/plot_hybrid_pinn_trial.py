"""Render a matched-budget PINN comparison without importing PyTorch."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    summary = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
    labels = {"uvp": "Direct u-v-p", "streamfunction": "Neural streamfunction",
              "hybrid_streamfunction": "SR + neural residual"}
    colors = {"uvp": "#3975A4", "streamfunction": "#23896F", "hybrid_streamfunction": "#D2693E"}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), constrained_layout=True)
    for index, (name, record) in enumerate(summary["runs"].items()):
        run = root / name
        if not (run / "dense_audit.json").exists():
            run = Path(record["run_directory"])
        dense = json.loads((run / "dense_audit.json").read_text(encoding="utf-8"))
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        sections = dense["cross_section_fluxes"]
        axes[0, 0].plot([r["x"] for r in sections],
                        [r["flux"]/dense["prescribed_inlet_flux"] for r in sections],
                        "o-", ms=4, color=colors[name], label=labels[name])
        for ax, values in ((axes[0, 1], record["pde_rms"]),
                           (axes[1, 0], record["wake_regions"]["taper"]["pde_rms"])):
            ax.bar(np.arange(2)+(index-1)*.24, [values["momentum_x"], values["momentum_y"]],
                   width=.23, color=colors[name], label=labels[name])
        with (run / "history.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        updates = [int(r["step"])+(config["adam_steps"] if r["phase"] == "lbfgs" else 0) for r in rows]
        axes[1, 1].semilogy(updates, [float(r["total"]) for r in rows], color=colors[name], label=labels[name])
    axes[0, 0].axhline(1, ls="--", color="0.35", lw=1)
    axes[0, 0].axvspan(5, 6.9, color="0.7", alpha=.15)
    axes[0, 0].set(title="A  Cross-section flow rate", xlabel="x / D", ylabel="Q / prescribed inlet flow")
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0, 0].legend(frameon=False, fontsize=9)
    for ax, title in ((axes[0, 1], "B  Independent full-domain momentum residuals"),
                      (axes[1, 0], "C  Momentum residuals inside the wake taper")):
        ax.set(yscale="log", title=title, ylabel="Residual RMS")
        ax.set_xticks([0, 1], ["x momentum", "y momentum"])
        ax.axhline(.01, ls="--", color="0.35", lw=1)
    axes[1, 1].set(title="D  Training objectives (different loss terms)",
                   xlabel="Optimizer updates", ylabel="Weighted loss")
    axes[1, 1].axvline(config["adam_steps"], ls="--", color="0.35", lw=1)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Re=20 cylinder PINN: one seed, matched update budget, no CFD reference", fontsize=13)
    fig.savefig(root / "comparison.png", dpi=180)
    fig.savefig(root / "comparison.pdf")
    plt.close(fig)
    print(root / "comparison.png")


if __name__ == "__main__":
    main()
