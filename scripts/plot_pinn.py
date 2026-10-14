"""Render PINN artifacts in a process that does not import PyTorch.

Separate plotting avoids competing Torch/MKL OpenMP runtimes on Windows.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(evaluation_dir):
    out = Path(evaluation_dir)
    c = json.loads((out / "plot_config.json").read_text(encoding="utf-8"))
    with np.load(out / "fields.npz", allow_pickle=False) as archive:
        coords, fields, residuals = archive["coords"], archive["uvp"], archive["residuals"]
    mask = np.isclose(coords[:, 2], c["t_end"]) if c["transient"] else np.ones(len(coords), dtype=bool)
    q, f, r = coords[mask], fields[mask], residuals[mask]
    x, y = np.unique(q[:, 0]), np.unique(q[:, 1])
    ix, iy = np.searchsorted(x, q[:, 0]), np.searchsorted(y, q[:, 1])
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), constrained_layout=True)
    for ax, values, title in zip(axes, (f[:, 0], f[:, 1], f[:, 2], np.sqrt(np.sum(r*r, axis=1))),
                                 ("u / Uref", "v / Uref", "p / (rho Uref^2)", "NS residual magnitude")):
        grid = np.full((len(y), len(x)), np.nan)
        grid[iy, ix] = values
        mesh = ax.pcolormesh(x, y, grid, shading="auto", cmap="viridis")
        if c["case"] == "cylinder":
            ax.add_patch(plt.Circle((c["cylinder_x"], c["cylinder_y"]), 0.5, color="white", ec="black", lw=0.7))
        ax.set(title=title, xlabel="x / Lref", ylabel="y / Lref", aspect="equal")
        fig.colorbar(mesh, ax=ax, pad=0.015)
    fig.suptitle(f"{c['case']}, Re={c['reynolds']:g}" + (f", t*={c['t_end']:g}" if c["transient"] else ""))
    fig.savefig(out / "fields.png", dpi=160)
    plt.close(fig)
    history = out.parent / "history.csv"
    if history.exists():
        with history.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        adam_end = max((int(r["step"]) for r in rows if r["phase"] == "adam"), default=0)
        updates = [int(r["step"]) + (adam_end if r["phase"] == "lbfgs" else 0) for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        axes[0].semilogy(updates, [max(float(r["total"]), 1e-30) for r in rows], label="weighted total")
        for group in ("pde", "boundary", "initial", "gauge", "data"):
            columns = [k for k in rows[0] if k.startswith(group + "_")]
            if columns:
                values = [max(sum(float(r[k]) for k in columns), 1e-30) for r in rows]
                axes[1].semilogy(updates, values, label=group)
        for ax in axes:
            ax.set(xlabel="Optimizer updates (Adam then L-BFGS)", ylabel="Loss")
            if any(r["phase"] == "lbfgs" for r in rows):
                ax.axvline(adam_end, color="gray", ls="--", lw=0.8)
            ax.grid(alpha=0.2)
            ax.legend()
        axes[0].set_title("Training objective on current sampled points")
        axes[1].set_title("Unweighted sums of component MSEs")
        fig.savefig(out.parent / "training_loss.png", dpi=160)
        plt.close(fig)
    forces = out / "forces.csv"
    if forces.exists() and c["transient"]:
        with forces.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True, constrained_layout=True)
        for ax, key in zip(axes, ("Cd", "Cl")):
            ax.plot([float(r["time"]) for r in rows], [float(r[key]) for r in rows], "o-", ms=3)
            ax.set(ylabel=key)
            ax.grid(alpha=0.2)
        axes[-1].set_xlabel("t Uref / D")
        fig.suptitle("PINN stress integrals (require independent flow validation)")
        fig.savefig(out / "forces.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    render(parser.parse_args().evaluation_dir)