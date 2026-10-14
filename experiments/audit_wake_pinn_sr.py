"""Independently audit frozen PINN-inspired symbolic models using SymPy."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline_contracts import array_hash, atomic_json, content_hash, dataset_manifest, file_hash
from experiments.wake_derivative_aware_sr import partition_field
from experiments.wake_pinn_informed_sr import (
    BLOCKS, ETA_SCALE, KEYS, RIDGE, X, Y, dictionary_indices,
    gram_parts, objective_gram, predict_choice, quadrature, read_json,
    score_pair, symbolic_expressions, velocity_basis,
)


def evaluate_sympy(expressions, features):
    result = {}
    for component in ("u", "v"):
        expr = sp.sympify(expressions[component])
        fields = {}
        for key, (nx, ny) in zip(KEYS, ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))):
            f = sp.lambdify((X, Y), sp.diff(expr, X, nx, Y, ny), "numpy")
            fields[key] = np.broadcast_to(np.asarray(f(*features.T), dtype=float), (len(features),)).copy()
        result[component] = fields
    return result


def audit(args):
    protocol = read_json(args.out_dir / "protocol.json")
    assert content_hash({k: v for k, v in protocol.items() if k != "protocol_id"}) == protocol["protocol_id"]
    for name, fingerprint in protocol["sources"].items():
        assert file_hash(ROOT / name) == fingerprint, name
    assert dataset_manifest(args.input_dir) == protocol["dataset"]
    frozen_path = args.out_dir / "frozen_selection.json"
    frozen = read_json(frozen_path)
    assert frozen["protocol_id"] == protocol["protocol_id"]
    frozen_hash = file_hash(frozen_path)
    candidates = read_json(args.out_dir / "validation_candidates.json")
    by_id = {r["candidate_id"]: r for r in candidates}
    assert len(by_id) == len(candidates) == frozen["candidate_count"]
    for method, chosen in frozen["finalists"].items():
        best = min([r for r in candidates if r["method"] == method],
                   key=lambda r: (r["validation"]["score"], r["n_coefficients"], r["candidate_id"]))
        assert chosen == best
    best = min(frozen["finalists"].values(), key=lambda r: (r["validation"]["score"], r["n_coefficients"], r["candidate_id"]))
    assert best["method"] == frozen["selected_method"]
    for budget, choices in frozen["compact_choices"].items():
        for method, choice in choices.items():
            assert choice == min([r for r in candidates if r["method"] == method and r["n_coefficients"] <= int(budget)],
                                 key=lambda r: (r["validation"]["score"], r["n_coefficients"], r["candidate_id"]))
    for r in candidates:
        indices = dictionary_indices(r["kind"], r["degrees"])
        active = np.flatnonzero(r["coefficients"])
        assert len(indices) == len(r["coefficients"]) and len(active) == r["n_coefficients"] <= 48
        if r["kind"] == "direct":
            assert all(sum(indices[i][0] == c for i in active) <= 24 for c in ("u", "v"))
    raw = pd.read_csv(args.input_dir / "wake_data.csv").sort_values(["x_D", "y_D"])
    xi, eta = np.sort(raw.x_D.unique()), np.sort(raw.y_D.unique())
    fields = {"u": raw.u_star.to_numpy().reshape(len(xi), len(eta)),
              "v": np.load(args.input_dir / "v_mean_wake.npy", allow_pickle=False)}
    parts = {c: partition_field(q, xi, eta, BLOCKS) for c, q in fields.items()}
    with np.load(args.out_dir / "data_splits.npz", allow_pickle=False) as snapshot:
        for c, p in parts.items():
            for split, d in p.items():
                assert array_hash(d["points"], d["q"], d["dx"], d["dy"]) == protocol["split_hashes"][c][split]
                for k, v in d.items():
                    np.testing.assert_array_equal(v, snapshot[f"{c}_{split}_{k}"])
    # Poison all held-out labels, including those used by the smoothed targets.
    # Every training array and its normalization must remain identical.
    heldout = [i for split in ("validation", "test") for lo, hi in BLOCKS[split] for i in range(lo, hi)]
    for c, values in fields.items():
        changed = values.copy(); changed[heldout] += np.arange(len(eta))[None, :] * 123.456 + 987.0
        altered = partition_field(changed, xi, eta, BLOCKS)["train"]
        for k, v in parts[c]["train"].items():
            np.testing.assert_array_equal(v, altered[k])
    train = {c: p["train"] for c, p in parts.items()}
    validation = {c: p["validation"] for c, p in parts.items()}
    qpoints, weights = quadrature()
    # Use a different, denser quadrature for the independent physics calculation.
    dense_points, dense_weights = quadrature(21, 33)
    selections = list(frozen["finalists"].values()) + list(frozen["fixed_support_ablations"].values())
    for choices in frozen["compact_choices"].values():
        selections.extend(choices.values())
    unique = {r["candidate_id"]: r for r in selections}
    rows = []
    maximum_basis_error = maximum_score_error = maximum_loss_error = maximum_stationarity = 0.0
    for name, choice in unique.items():
        expressions = symbolic_expressions(choice)
        numerical = predict_choice(choice, train["u"]["features"])
        symbolic = evaluate_sympy(expressions, train["u"]["features"])
        error = max(float(np.max(np.abs(symbolic[c][k] - numerical[c][k]))) for c in ("u", "v") for k in KEYS)
        maximum_basis_error = max(maximum_basis_error, error)
        assert error < 2e-9, name
        val_metrics = score_pair(evaluate_sympy(expressions, validation["u"]["features"]), validation, protocol["scales"], protocol["radius"])
        score_error = abs(val_metrics["score"] - choice["validation"]["score"])
        maximum_score_error = max(maximum_score_error, score_error)
        assert score_error < 1e-10
        train_metrics = score_pair(symbolic, train, protocol["scales"], protocol["radius"])
        lv = sum(m["normalized_value"] for m in train_metrics["fields"].values())
        lg = sum(m["normalized_gradient"] for m in train_metrics["fields"].values())
        collocation = evaluate_sympy(expressions, dense_points)
        r = collocation["u"]["dx"] + collocation["v"]["dy"]
        rx = collocation["u"]["dxx"] + collocation["v"]["dxy"]
        ry = collocation["u"]["dxy"] + collocation["v"]["dyy"]
        lc = float(np.sum(dense_weights * r ** 2) / protocol["continuity_scale"])
        lcg = float(np.sum(dense_weights * (rx ** 2 + (ETA_SCALE * ry) ** 2)) / (2 * protocol["continuity_scale"]))
        loss = lv + choice["gradient_weight"] * lg + choice["physics_weight"] * (lc + choice["residual_gradient_weight"] * lcg)
        bt = velocity_basis(train["u"]["features"], choice["kind"], choice["degrees"])
        bc = velocity_basis(qpoints, choice["kind"], choice["degrees"])
        gram = gram_parts(bt, bc, train, protocol["scales"], protocol["continuity_scale"], weights)
        g, h = objective_gram(gram, choice["gradient_weight"], choice["physics_weight"], choice["residual_gradient_weight"])
        coeff = np.asarray(choice["coefficients"])
        target_square = sum(float(np.mean(train[c]["q"] ** 2) / protocol["scales"][c]["value"])
                            + choice["gradient_weight"] * 0.5 * sum(float(np.mean(train[c][k] ** 2) / protocol["scales"][c][k]) for k in ("dx", "dy")) for c in ("u", "v"))
        gram_loss = float(coeff @ g @ coeff - 2 * h @ coeff + target_square)
        loss_error = abs(loss - gram_loss)
        maximum_loss_error = max(maximum_loss_error, loss_error)
        assert loss_error < 2e-10
        support = np.flatnonzero(coeff)
        stationarity = g @ coeff - h + RIDGE * np.diag(g) * coeff
        relative_stationarity = float(np.linalg.norm(stationarity[support]) / max(np.linalg.norm(h[support]), 1e-30))
        maximum_stationarity = max(maximum_stationarity, relative_stationarity)
        assert relative_stationarity < 1e-10
        row = {"candidate_id": name, "method": choice["method"], "n_coefficients": choice["n_coefficients"],
               "L_value": lv, "L_gradient": lg, "L_continuity": lc, "L_continuity_gradient": lcg,
               "objective": loss, "basis_error": error, "objective_error": loss_error,
               "normal_equation_relative_residual": relative_stationarity,
               "domain_divergence_rms": float(np.sqrt(np.sum(dense_weights * r ** 2)))}
        if choice["kind"] == "streamfunction":
            # Separately expanded Float polynomials can differ by roundoff.
            # Verify exact cancellation with the same decimal coefficients as
            # rationals, and verify the actual Float export numerically too.
            exact_psi = sum(sp.Rational(str(c)) * sp.chebyshevt(i, X)
                            * sp.chebyshevt(j, Y / sp.Rational(19, 10))
                            for (_, i, j), c in zip(dictionary_indices(choice["kind"], choice["degrees"]), choice["coefficients"]) if c)
            assert sp.expand(sp.diff(exact_psi, Y, X) - sp.diff(exact_psi, X, Y)) == 0
            row["float_symbolic_divergence_max_roundoff"] = float(np.max(np.abs(r)))
            assert row["float_symbolic_divergence_max_roundoff"] < 1e-10
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.out_dir / "training_loss_audit.csv", index=False)
    exported = read_json(args.out_dir / "selected_velocity_models.json")
    assert exported["frozen_selection_sha256"] == frozen_hash
    assert exported["model"]["candidate_id"] == best["candidate_id"]
    assert content_hash(exported["model"]["expressions"]) == exported["model"]["expression_sha256"]
    xx, yy = np.meshgrid(np.linspace(-1, 1, 97), np.linspace(-1.9, 1.9, 193), indexing="ij")
    dense = np.column_stack([xx.ravel(), yy.ravel()])
    before, after = predict_choice(best, dense), evaluate_sympy(exported["model"]["expressions"], dense)
    export_error = max(float(np.max(np.abs(before[c][k] - after[c][k]))) for c in ("u", "v") for k in KEYS)
    assert export_error < 2e-9
    test_error = None
    if not args.pre_test:
        complete = read_json(args.out_dir / "complete.json")
        assert complete["frozen_selection_sha256"] == frozen_hash
        test = {c: p["test"] for c, p in parts.items()}
        saved = pd.read_csv(args.out_dir / "test_metrics.csv")
        test_error = 0.0
        for method, choice in {**frozen["finalists"], **frozen["fixed_support_ablations"]}.items():
            pred = evaluate_sympy(symbolic_expressions(choice), test["u"]["features"])
            for reference in ("raw", "smooth"):
                metrics = score_pair(pred, test, protocol["scales"], protocol["radius"], smooth=reference == "smooth")
                for c, m in metrics["fields"].items():
                    row = saved[(saved.method == method) & (saved.component == c) & (saved.reference == reference)].iloc[0]
                    test_error = max(test_error, max(abs(float(row[k]) - v) for k, v in m.items()))
        assert test_error < 2e-9
    assert file_hash(frozen_path) == frozen_hash
    result = {"protocol_id": protocol["protocol_id"], "audit_source_sha256": file_hash(Path(__file__)),
              "frozen_selection_sha256": frozen_hash, "all_sources_and_data_hashes_match": True,
              "all_selections_minimize_declared_validation_score": True,
              "all_candidate_coefficient_caps_verified": len(candidates),
              "heldout_label_poisoning_leaves_all_training_arrays_unchanged": True,
              "unique_frozen_models_audited": len(unique), "basis_vs_sympy_max_error_through_second_derivatives": maximum_basis_error,
              "validation_score_max_error": maximum_score_error, "training_objective_max_error": maximum_loss_error,
              "normal_equation_max_relative_residual": maximum_stationarity,
              "dense_horner_export_max_error_through_second_derivatives": export_error,
              "test_metrics_independent_max_error": test_error}
    atomic_json(args.out_dir / ("pre_test_verification.json" if args.pre_test else "verification.json"), result)
    print(result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/wake_pinn_informed_sr_v3")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "outputs_mgn/comsol/eta_sym")
    parser.add_argument("--pre-test", action="store_true")
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
