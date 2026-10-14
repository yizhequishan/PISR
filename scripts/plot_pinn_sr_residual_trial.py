"""Plot archived SR residual-weight continuation results without running a model.

Compatible with Python 3.8.  Diagnosis uses only development observations and the
frozen selection; comparison reads final audit results only when requested.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Rectangle
import numpy as np
import sympy as sp


COLORS = {"baseline": "#858b92", "ordinary": "#3e6489",
          "inlet": "#c18b42", "sr": "#198678"}
LABELS = {"baseline": "Starting checkpoint", "ordinary": "Ordinary continuation",
          "inlet": "Simple inlet weighting", "sr": "Frozen SR weighting"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_figure(fig, folder, stem):
    paths = []
    for extension in ("png", "pdf"):
        path = folder / (stem + "." + extension)
        fig.savefig(str(path), dpi=220, bbox_inches="tight", facecolor="white")
        paths.append(str(path))
    plt.close(fig)
    return paths


def geometry(ax, config, zoom=False):
    length, height = config["channel_length"], config["channel_height"]
    ax.add_patch(Circle((config["cylinder_x"], config["cylinder_y"]), .5,
                        facecolor="white", edgecolor="#303840", linewidth=.9,
                        zorder=5))
    ax.set_xlim(0, min(3., length) if zoom else length)
    ax.set_ylim(0, height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / D")
    ax.set_ylabel("y / D")
    if zoom:
        ax.axvline(1., color="#687079", linestyle="--", linewidth=.8, zorder=6)
        ax.set_xticks([0, 1, 2, 3])
    else:
        ax.add_patch(Rectangle((0, 0), min(3., length), height, fill=False,
                               edgecolor="#687079", linewidth=.8, linestyle="--"))
        ax.set_xticks(np.arange(0, length + .1, 2))
    ax.tick_params(labelsize=8)


def symbolic_values(expression, points, height):
    """Evaluate the saved expression with the same physical feature map."""
    x_symbol, y_symbol = sp.symbols("x_D y_D")
    expression = sp.sympify(expression, locals={"x_D": x_symbol, "y_D": y_symbol,
                                               "square": lambda value: value**2})
    unknown = expression.free_symbols - {x_symbol, y_symbol}
    if unknown:
        raise ValueError("Unknown expression variables: " + str(unknown))
    evaluator = sp.lambdify((x_symbol, y_symbol), expression, modules="numpy")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = evaluator(points[:, 0], 2 * points[:, 1] / height - 1)
    result = np.broadcast_to(np.asarray(result, dtype=float), (len(points),)).copy()
    if not np.isfinite(result).all():
        raise ValueError("Frozen SR expression is non-finite on plotting points")
    return result


def raw_weights(selection, points, height):
    score = np.clip(symbolic_values(selection["expression"], points, height),
                    0., float(selection["score_cap"]))
    return .5 + .5 * np.clip(score / float(selection["score_mean"]), 0., 10.)


def plot_diagnosis(root, protocol, folder):
    config = protocol["baseline_config"]
    with np.load(str(root / "development.npz")) as archive:
        points = np.array(archive["selection_coords"])
        residuals = np.array(archive["selection_residuals"])
    if not np.isfinite(points).all() or not np.isfinite(residuals).all():
        raise ValueError("Development arrays contain non-finite values")
    if points.ndim != 2 or points.shape[1] != 2 or residuals.shape != (len(points), 3):
        raise ValueError("Unexpected development coordinates or residual shape")

    # This is the momentum residual norm, not the scaled SR fitting target.
    log_residual = np.log10(np.maximum(np.linalg.norm(residuals[:, 1:], axis=1), 1e-12))
    residual_norm = Normalize(vmin=np.floor(log_residual.min()),
                              vmax=np.ceil(log_residual.max()))
    selection_path = root / "selection.json"
    selection = read_json(selection_path) if selection_path.exists() else None
    nrows = 2 if selection else 1
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 3.55 * nrows), squeeze=False,
                             gridspec_kw={"width_ratios": [22, 3]},
                             constrained_layout=True)
    order = np.argsort(log_residual)
    for column, ax in enumerate(axes[0]):
        artists = ax.scatter(points[order, 0], points[order, 1], c=log_residual[order],
                             s=7 if column == 0 else 9, cmap="magma", norm=residual_norm,
                             edgecolors="none", rasterized=True)
        geometry(ax, config, zoom=bool(column))
        ax.set_title("A  Baseline momentum residual" if column == 0 else "B  Inlet detail",
                     loc="left", fontsize=10, pad=9)
    colorbar = fig.colorbar(artists, ax=list(axes[0]), fraction=.025, pad=.025)
    colorbar.set_label(r"$\log_{10}\sqrt{r_x^2+r_y^2}$", fontsize=10)
    colorbar.ax.tick_params(labelsize=8)

    normalization = None
    if selection:
        height, length = config["channel_height"], config["channel_length"]
        grid_x, grid_y = np.meshgrid(np.linspace(0, length, 661),
                                     np.linspace(0, height, 161))
        grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        fluid = ((grid_points[:, 0] - config["cylinder_x"])**2 +
                 (grid_points[:, 1] - config["cylinder_y"])**2 >= .25)
        normalization = float(raw_weights(selection, points, height).mean())
        weights = np.full(len(grid_points), np.nan)
        weights[fluid] = raw_weights(selection, grid_points[fluid], height) / normalization
        weights = weights.reshape(grid_x.shape)
        weight_norm = Normalize(vmin=float(np.nanmin(weights)), vmax=float(np.nanmax(weights)))
        for column, ax in enumerate(axes[1]):
            artist = ax.pcolormesh(grid_x, grid_y, np.ma.masked_invalid(weights),
                                   cmap="viridis", norm=weight_norm, shading="auto",
                                   rasterized=True)
            geometry(ax, config, zoom=bool(column))
            ax.set_title("C  Frozen symbolic weighting" if column == 0 else "D  Inlet detail",
                         loc="left", fontsize=10, pad=9)
        colorbar = fig.colorbar(artist, ax=list(axes[1]), fraction=.025, pad=.025)
        colorbar.set_label("Relative momentum-loss weight", fontsize=10)
        colorbar.ax.tick_params(labelsize=8)

    seed = protocol.get("seeds", {}).get("selection", "recorded")
    title = "Steady cylinder PINN | Re={:g} | development diagnosis".format(config["reynolds"])
    fig.suptitle(title, fontsize=13, fontweight="semibold")
    footnote = ("Baseline: {:,} new development points (seed {}), including inlet enrichment. "
                "Dashed line in detail: x/D=1.\n"
                "Residuals measure equation mismatch; no reference CFD field is used.").format(len(points), seed)
    if selection:
        footnote += ("  Display weights have mean 1 on these development points; "
                     "training normalizes each batch.")
    else:
        footnote += "  Frozen SR selection is not yet available."
    fig.supxlabel(footnote, fontsize=8, color="#4e555d", y=.002)
    paths = save_figure(fig, folder, "baseline_and_sr_weights")
    return paths, {"development_points": len(points), "selection_available": bool(selection),
                   "display_weight_normalizer": normalization,
                   "expression": selection["expression"] if selection else None}


def inlet_rms(regions):
    if "regions" in regions:
        regions = regions["regions"]
    for key in ("inlet_x_lt_1", "inlet"):
        if key in regions:
            return regions[key]["momentum_rms"]
    raise KeyError("No inlet region in archived diagnostics")


def annotate_bar(ax, rectangle):
    value = rectangle.get_height()
    ax.annotate("{:.3g}".format(value),
                (rectangle.get_x() + rectangle.get_width() / 2, value),
                xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                fontsize=7, rotation=0, color="#333b43")


def comparison_axis(ax, title):
    ax.set_title(title, loc="left", fontsize=11, pad=13)
    ax.set_ylabel("Residual RMS")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e0e4e8", linewidth=.7)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(bottom=0)


def plot_comparison(root, protocol, folder):
    comparison_path = root / "comparison.json"
    if not comparison_path.exists():
        return []
    comparison = read_json(comparison_path)
    names = ["baseline"] + [name for name in ("ordinary", "inlet", "sr")
                             if name in comparison.get("runs", {})]
    values = {"baseline": {"momentum_rms": comparison["baseline_momentum_rms"],
                            "outlet_traction_rms": comparison["baseline_outlet_traction_rms"]}}
    values.update(comparison.get("runs", {}))
    regional_path = root / "baseline_audit" / "regional_test.json"
    if not regional_path.exists():
        raise FileNotFoundError("Baseline inlet comparison requires " + str(regional_path))
    values["baseline"]["regions"] = read_json(regional_path)
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.6),
                             gridspec_kw={"width_ratios": [1.15, 1., 1.15]})
    # Reserve explicit figure bands for the legend and audit note.  The older
    # Matplotlib in the project environment does not reliably reserve room for
    # a manually positioned supxlabel under constrained_layout.
    fig.subplots_adjust(left=.065, right=.99, bottom=.20, top=.76, wspace=.29)
    width = .8 / len(names)
    for panel, metric in ((0, "momentum_rms"), (2, "inlet")):
        ax = axes[panel]
        for index, name in enumerate(names):
            data = values[name][metric] if metric != "inlet" else inlet_rms(values[name]["regions"])
            if len(data) != 2 or not np.isfinite(data).all():
                raise ValueError("Invalid momentum RMS for " + name)
            positions = np.arange(2) + (index - (len(names) - 1) / 2.) * width
            bars = ax.bar(positions, data, width=width * .91, color=COLORS[name],
                          label=LABELS[name], edgecolor="white", linewidth=.4)
            for bar in bars:
                annotate_bar(ax, bar)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["x momentum", "y momentum"])
        comparison_axis(ax, "A  Global momentum" if panel == 0 else "C  Inlet: x/D < 1")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.19)

    outlet = [values[name]["outlet_traction_rms"] for name in names]
    if not np.isfinite(outlet).all():
        raise ValueError("Invalid outlet traction RMS")
    bars = axes[1].bar(np.arange(len(names)), outlet, width=.65,
                       color=[COLORS[name] for name in names], edgecolor="white", linewidth=.4)
    for bar in bars:
        annotate_bar(axes[1], bar)
    axes[1].set_xticks(np.arange(len(names)))
    short_labels = {"baseline": "Start", "ordinary": "Ordinary", "inlet": "Inlet", "sr": "SR"}
    axes[1].set_xticklabels([short_labels[name] for name in names], fontsize=9)
    comparison_axis(axes[1], "B  Outlet traction")
    axes[1].set_ylim(0, axes[1].get_ylim()[1] * 1.19)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .90),
               ncol=4, frameon=False, fontsize=9)
    title = "Matched-time continuation from one checkpoint | Re={:g}".format(
        protocol["baseline_config"]["reynolds"])
    if not comparison.get("all_three_arms_complete", False):
        title += " | partial results"
    fig.suptitle(title, fontsize=13, fontweight="semibold", y=.975)
    seed = protocol.get("seeds", {}).get("test", "recorded")
    fig.text(.5, .045, "Fresh common audit: 20,000 uniform fluid points, seed {}. "
             "Lower is better; bars show one continuation per method.\n"
             "SR labeling and search count toward its time budget. "
             "Residual improvements do not establish CFD field accuracy.".format(seed),
             ha="center", va="bottom", fontsize=8, color="#4e555d", linespacing=1.5)
    return save_figure(fig, folder, "continuation_comparison")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", "--trial-dir", dest="trial_dir", required=True, type=Path,
                        help="Archived trial directory; figures are written below it")
    parser.add_argument("--diagnosis-only", action="store_true",
                        help="Read only development data and frozen selection, never final audits")
    args = parser.parse_args()
    root = args.trial_dir.resolve()
    protocol = read_json(root / "protocol.json")
    folder = root / "figures"
    folder.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#333b43", "axes.edgecolor": "#687079",
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    paths, diagnosis = plot_diagnosis(root, protocol, folder)
    if not args.diagnosis_only:
        paths.extend(plot_comparison(root, protocol, folder))
    manifest = {"figures": paths, "diagnosis": diagnosis,
                "comparison_requested": not args.diagnosis_only,
                "comparison_available": (root / "comparison.json").exists(),
                "source_checkpoint_sha256": protocol.get("checkpoint_sha256")}
    (folder / "plot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
