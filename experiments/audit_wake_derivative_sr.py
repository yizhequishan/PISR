"""Verify a completed real-wake SR run without importing PySR or retraining."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline_contracts import array_hash, atomic_json, content_hash, file_hash
from experiments.wake_derivative_aware_sr import metrics, predict


def audit(out, reference):
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    complete = json.loads((out / "complete.json").read_text(encoding="utf-8"))
    assert complete["protocol_id"] == protocol["protocol_id"]
    assert file_hash(ROOT / "experiments/wake_derivative_aware_sr.py") == protocol["source_sha256"]
    assert file_hash(ROOT / "experiments/wake_derivative_loss.jl") == protocol["julia_loss_sha256"]
    assert file_hash(ROOT / "experiments/p2_analytic_derivative_loss.py") == protocol["objective_source_sha256"]
    for name, digest in protocol["dataset"]["files"].items():
        assert file_hash(ROOT / "outputs_mgn/comsol/eta_sym" / name) == digest
    archive = np.load(out / "data_splits.npz")
    for c in protocol["settings"]["components"]:
        for split in ("train", "validation", "test"):
            assert array_hash(*[archive[f"{c}_{split}_{k}"] for k in ("points", "q", "dx", "dy")]) == protocol["split_sha256"][c][split]
    losses, equivalent = [], []
    for path in sorted(out.glob("*/selection.json")):
        fit = json.loads(path.read_text(encoding="utf-8"))
        assert fit["protocol_id"] == protocol["protocol_id"]
        c = fit["component"]
        train = {k: archive[f"{c}_train_{k}"] for k in ("features", "points", "q", "dx", "dy")}
        for rule, choice in fit["choices"].items():
            assert content_hash(choice["expression"]) == choice["expression_sha256"]
            result = metrics(choice["expression"], train, protocol["scales"][c], protocol["radius"])
            rebuilt = result["normalized_value"] + fit["lambda_grad"] * result["normalized_gradient"]
            delta = abs(rebuilt - choice["train_loss"])
            assert delta < 1e-8 * max(1, choice["train_loss"])
            losses.append({"fit": path.parent.name, "selection": rule, "julia_train_loss": choice["train_loss"],
                           "sympy_rebuilt_loss": rebuilt, "absolute_error": delta})
        old_path = reference / path.parent.name / "selection.json"
        if old_path.exists():
            old = json.loads(old_path.read_text(encoding="utf-8"))
            identical = old["choices"] == fit["choices"]
            assert identical
            equivalent.append({"fit": path.parent.name, "identical_choices_and_metrics": identical,
                               "reference_seconds": old["seconds"], "typed_seconds": fit["seconds"]})
    assert len(losses) == 2 * complete["fits"]
    pd.DataFrame(losses).to_csv(out / "training_loss_recheck.csv", index=False)
    main = pd.read_csv(out / "test_metrics.csv")
    overview = []
    for c in protocol["settings"]["components"]:
        for ref in ("central_difference", "savgol5_stability"):
            for control in ("own", "value", "h1"):
                table = main[(main.component == c) & (main.reference == ref)]
                a = table[(table.lambda_grad == 0) & (table.selection == ("value" if control == "own" else control))].sort_values("seed")
                b = table[(table.lambda_grad == 1) & (table.selection == ("h1" if control == "own" else control))].sort_values("seed")
                assert np.array_equal(a.seed, b.seed)
                for metric in ("value_mse", "dx_rel_l2", "dy_rel_l2", "gradient_rel_l2", "K_R_rel_l2"):
                    av, bv = a[metric].to_numpy(), b[metric].to_numpy()
                    overview.append({"component": c, "reference": ref, "selection_control": control, "metric": metric,
                                     "value_only_mean": float(av.mean()), "derivative_mean": float(bv.mean()),
                                     "ratio_of_means_change_pct": float(100 * (bv.mean() / av.mean() - 1)),
                                     "improved_seed_count": int(np.sum(bv < av)), "n_seeds": len(av)})
    pd.DataFrame(overview).to_csv(out / "comparison_overview.csv", index=False)
    result = {"input_and_source_hashes_match": True, "frozen_expression_hashes_match": True,
              "test_independent_of_selection": complete["all_test_choices_frozen_on_validation"],
              "completed_fits": complete["fits"], "checked_selected_train_losses": len(losses),
              "max_julia_vs_sympy_train_loss_error": max(r["absolute_error"] for r in losses),
              "reference_vs_typed_completed_fits": equivalent}
    atomic_json(out / "verification.json", result)
    profile_plot(out, archive, main, protocol)
    print(json.dumps(result, indent=2))


def profile_plot(out, archive, table, protocol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    components = protocol["settings"]["components"]
    seed = protocol["settings"]["seeds"][0]
    fig, axes = plt.subplots(len(components), 3, figsize=(13, 3.5 * len(components)), squeeze=False, constrained_layout=True)
    styles = [(0, "value", "Value search / value selection", "#2878b5"),
              (0, "h1", "Value search / H1 selection", "#55975b"),
              (1, "h1", "Derivative search / H1 selection", "#d95f02")]
    for i, c in enumerate(components):
        points = archive[f"{c}_test_points"]
        positions = np.unique(points[:, 0])
        xi = positions[len(positions) // 2]
        mask = points[:, 0] == xi
        eta = points[mask, 1]
        for j, key in enumerate(("q", "dx", "dy")):
            ax = axes[i, j]
            ax.plot(eta, archive[f"{c}_test_{key}"][mask], color="black", lw=2.2, label="CFD numerical reference")
            for weight, criterion, label, color in styles:
                row = table[(table.component == c) & (table.seed == seed) & (table.lambda_grad == weight)
                            & (table.selection == criterion) & (table.reference == "central_difference")].iloc[0]
                pred = predict(row.expression, archive[f"{c}_test_features"][mask])
                ax.plot(eta, pred[key], lw=1.3, color=color, label=label)
            ax.set_title(f"{c}*: {key}, xi={xi:.4g}")
            ax.set_xlabel("eta")
            ax.grid(alpha=0.2)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle(f"Fixed illustrative seed {seed}; middle test section; no test-set reselection")
    fig.savefig(out / "test_profiles_seed0.png", dpi=180)
    plt.close(fig)

    selected = table[(table.reference == "central_difference") & (table.selection == "h1")]
    fig, axes = plt.subplots(len(components), 3, figsize=(12, 3.5 * len(components)),
                             squeeze=False, constrained_layout=True)
    for i, c in enumerate(components):
        for j, (key, title, scale) in enumerate((("value_mse", "Velocity MSE", 1),
                                                ("gradient_rel_l2", "Gradient error [%]", 100),
                                                ("K_R_rel_l2", "Radius transport error [%]", 100))):
            ax = axes[i, j]
            for seed in protocol["settings"]["seeds"]:
                pair = selected[(selected.component == c) & (selected.seed == seed)].sort_values("lambda_grad")
                ax.plot([0, 1], scale * pair[key], "o-", label=f"seed {seed}")
            ax.set_xticks([0, 1], ["Value loss", "Derivative loss"])
            ax.set_title(f"{c}*: {title}")
            ax.grid(alpha=0.2)
            if j == 0:
                ax.set_yscale("log")
                ax.legend(fontsize=8)
    fig.suptitle("Both searches use H1 validation selection; fixed spatial test block")
    fig.savefig(out / "matched_h1_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/wake_derivative_aware_sr_2026-09-28_typed")
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "outputs/wake_derivative_aware_sr_2026-09-28")
    args = parser.parse_args()
    audit(args.out_dir, args.reference_dir)
