"""PINN-inspired sparse symbolic velocity fits on the time-averaged CFD wake.

Compare a data/Sobolev control, soft continuity, gradient-enhanced continuity,
and a hard streamfunction representation. Only incompressibility is imposed:
the mean field has no known Reynolds number or Reynolds-stress closure.
No PySR/Julia or neural-network training is required by this experiment.
"""
from __future__ import annotations

import argparse
import importlib.metadata as package_metadata
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
from numpy.polynomial.chebyshev import chebder, chebval
from numpy.polynomial.legendre import leggauss
from scipy.linalg import cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline_contracts import array_hash, atomic_json, content_hash, dataset_manifest, file_hash
from experiments.wake_derivative_aware_sr import partition_field, predict, scales_for
from experiments.wake_operator_aware_sr import (
    BLOCKS, basis_arrays, check_numerics as check_previous_numerics,
    design_system, field_metrics, sparse_path, transport,
)

X, Y = sp.symbols("x y")
ETA_SCALE = 1.9
RIDGE = 1e-8
TOTAL_BUDGETS = (16, 24, 32, 40, 48)
PER_COMPONENT_CAP = 24
GRAD_WEIGHTS = (0.0, 0.1, 1.0)
PHYSICS_WEIGHTS = (0.01, 0.1, 1.0, 10.0)
RESIDUAL_GRAD_WEIGHTS = (0.01, 0.1, 1.0)
# Each tuple is (degree_x_u, degree_eta_u, degree_x_v, degree_eta_v).
# Include the previously validated heterogeneous dictionary as a paired option.
DICTIONARIES = [(i, j, i, j) for i in (1, 2, 3) for j in (6, 8, 10, 12)] + [(2, 10, 2, 12)]
METHODS = ("data_control", "pinn_continuity", "gpinn_continuity", "hard_streamfunction")
KEYS = ("q", "dx", "dy", "dxx", "dxy", "dyy")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dictionary_indices(kind, degrees):
    ux, uy, vx, vy = degrees
    if kind == "direct":
        return [("u", i, j) for i in range(ux + 1) for j in range(uy + 1)] + [
            ("v", i, j) for i in range(vx + 1) for j in range(vy + 1)]
    # Intersect the two velocity degree envelopes. Omit the constant psi gauge.
    # This ensures the hard model's velocity polynomials belong to precisely
    # the same degree envelope as the direct model for this dictionary.
    return [("psi", i, j)
            for i in range(max(ux, vx + 1) + 1)
            for j in range(max(uy + 1, vy) + 1)
            if (i or j) and (j == 0 or (i <= ux and j - 1 <= uy))
            and (i == 0 or (i - 1 <= vx and j <= vy))]


def velocity_basis(features, kind, degrees):
    indices = dictionary_indices(kind, degrees)
    nx, ny = max(t[1] for t in indices), max(t[2] for t in indices)
    tables = []
    for axis, degree, scale in ((features[:, 0], nx, 1.0),
                                (features[:, 1] / ETA_SCALE, ny, ETA_SCALE)):
        t = {}
        for i in range(degree + 1):
            c = np.zeros(i + 1); c[-1] = 1.0
            for order in range(4):
                t[i, order] = chebval(axis, chebder(c, m=order)) / scale ** order
        tables.append(t)
    n, p = len(features), len(indices)
    result = {c: {k: np.zeros((n, p)) for k in KEYS} for c in ("u", "v")}
    for col, (component, i, j) in enumerate(indices):
        for key, (dx, dy) in zip(KEYS, ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))):
            if kind == "direct":
                result[component][key][:, col] = tables[0][i, dx] * tables[1][j, dy]
            else:
                result["u"][key][:, col] = tables[0][i, dx] * tables[1][j, dy + 1]
                result["v"][key][:, col] = -tables[0][i, dx + 1] * tables[1][j, dy]
    return result


def quadrature(nx=16, ny=24):
    """Mean-integral quadrature on x=xi-4 in [-1,1], eta in [-1.9,1.9]."""
    x, wx = leggauss(nx); y, wy = leggauss(ny)
    xx, yy = np.meshgrid(x, ETA_SCALE * y, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel()]), np.outer(wx, wy).ravel() / 4


def residual_basis(basis):
    return {"r": basis["u"]["dx"] + basis["v"]["dy"],
            "rx": basis["u"]["dxx"] + basis["v"]["dxy"],
            "ry": basis["u"]["dxy"] + basis["v"]["dyy"]}


def data_system(basis, data, scales, gradient_only=False):
    a, b = [], []
    for component in ("u", "v"):
        d = data[component]
        for key in (("dx", "dy") if gradient_only else ("q",)):
            scale = scales[component]["value" if key == "q" else key]
            weight = 1.0 if key == "q" else 0.5
            factor = np.sqrt(weight / (len(d["q"]) * scale))
            a.append(factor * basis[component][key]); b.append(factor * d[key])
    return np.vstack(a), np.concatenate(b)


def gram_parts(basis, collocation_basis, data, scales, continuity_scale, weights):
    a, b = data_system(basis, data, scales)
    ag, bg = data_system(basis, data, scales, gradient_only=True)
    residual = residual_basis(collocation_basis)
    factor = np.sqrt(weights / continuity_scale)[:, None]
    c = factor * residual["r"]
    cg = np.vstack([factor * residual["rx"] / np.sqrt(2),
                    factor * ETA_SCALE * residual["ry"] / np.sqrt(2)])
    return {"value": (a.T @ a, a.T @ b), "gradient": (ag.T @ ag, ag.T @ bg),
            "continuity": c.T @ c, "continuity_gradient": cg.T @ cg}


def objective_gram(parts, gradient_weight, physics_weight, residual_gradient_weight):
    g = parts["value"][0] + gradient_weight * parts["gradient"][0]
    g = g + physics_weight * (parts["continuity"] + residual_gradient_weight * parts["continuity_gradient"])
    h = parts["value"][1] + gradient_weight * parts["gradient"][1]
    return g, h


def sparse_gram_path(gram, rhs, indices, kind, budgets=TOTAL_BUDGETS):
    """Forward least squares with a total cap and at most 24 direct terms/field."""
    norms = np.sqrt(np.diag(gram))
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("Invalid or unobservable dictionary column")
    g, h = gram / norms[:, None] / norms[None, :], rhs / norms
    if kind == "direct":
        active = [indices.index(("u", 0, 0)), indices.index(("v", 0, 0))]
        groups = np.array([t[0] for t in indices])
        maximum = sum(min(PER_COMPONENT_CAP, int(np.sum(groups == c))) for c in ("u", "v"))
    else:
        active = [indices.index(("psi", 0, 1)), indices.index(("psi", 1, 0))]
        groups = None
        maximum = len(indices)
    maximum = min(max(budgets), maximum)
    sizes = sorted(set([b for b in budgets if b <= maximum] + [maximum]))
    while True:
        factor = cho_factor(g[np.ix_(active, active)] + RIDGE * np.eye(len(active)))
        beta = cho_solve(factor, h[active])
        if len(active) in sizes:
            coefficients = np.zeros(len(h))
            coefficients[active] = beta / norms[active]
            yield len(active), coefficients
        if len(active) == maximum:
            break
        corr = h - g[:, active] @ beta
        projected = cho_solve(factor, g[active, :])
        residual_norm = np.diag(g) - np.sum(g[:, active] * projected.T, axis=1)
        gain = corr ** 2 / np.maximum(residual_norm, 1e-12)
        gain[active] = -np.inf
        if groups is not None:
            for c in ("u", "v"):
                if np.sum(groups[active] == c) >= PER_COMPONENT_CAP:
                    gain[groups == c] = -np.inf
        if not np.isfinite(gain).any():
            raise RuntimeError("No eligible term before the requested budget")
        active.append(int(np.argmax(gain)))


def predict_basis(basis, coefficients):
    return {c: {k: a @ coefficients for k, a in d.items()} for c, d in basis.items()}


def predict_choice(choice, features):
    return predict_basis(velocity_basis(features, choice["kind"], choice["degrees"]),
                         np.asarray(choice["coefficients"]))


def score_pair(pred, data, scales, radius, smooth=False):
    fields = {c: field_metrics(pred[c], data[c], scales[c], radius, smooth=smooth) for c in ("u", "v")}
    r = pred["u"]["dx"] + pred["v"]["dy"]
    prefix = "smooth_" if smooth else ""
    reference = data["u"][prefix + "dx"] + data["v"][prefix + "dy"]
    return {"fields": fields, "score": sum(d["operator"] for d in fields.values()),
            "divergence_rms": float(np.sqrt(np.mean(r ** 2))),
            "reference_divergence_rms": float(np.sqrt(np.mean(reference ** 2))),
            "divergence_reference_error_rms": float(np.sqrt(np.mean((r - reference) ** 2)))}


def configs(kind):
    for gw in GRAD_WEIGHTS:
        if kind == "streamfunction":
            yield "hard_streamfunction", gw, 0.0, 0.0
            continue
        yield "data_control", gw, 0.0, 0.0
        for pw in PHYSICS_WEIGHTS:
            yield "pinn_continuity", gw, pw, 0.0
            for rgw in RESIDUAL_GRAD_WEIGHTS:
                yield "gpinn_continuity", gw, pw, rgw


def choose(records):
    valid = [r for r in records if np.isfinite(r["validation"]["score"])]
    if not valid:
        raise RuntimeError("No finite validation candidates")
    return min(valid, key=lambda r: (r["validation"]["score"], r["n_coefficients"], r["candidate_id"]))


def symbolic_expressions(choice):
    indices = dictionary_indices(choice["kind"], choice["degrees"])
    expr = {c: sp.S.Zero for c in ("u", "v", "psi")}
    for (component, i, j), coefficient in zip(indices, choice["coefficients"]):
        if coefficient:
            expr[component] += (sp.Float(coefficient, 17) * sp.chebyshevt(i, X)
                                * sp.chebyshevt(j, Y / sp.Rational(19, 10)))
    if choice["kind"] == "streamfunction":
        expr["psi"] -= expr["psi"].subs({X: 0, Y: 0})
        expr["u"], expr["v"] = sp.diff(expr["psi"], Y), -sp.diff(expr["psi"], X)
    else:
        del expr["psi"]
    return expr


def model_export(choice):
    expr = symbolic_expressions(choice)
    horner = {c: str(sp.horner(e, X, Y)) for c, e in expr.items()}
    # Differentiate the single psi expression when exact cancellation is needed.
    return {**choice, "expressions": horner, "expression_sha256": content_hash(horner),
            "sympy_ops": {c: int(sp.count_ops(sp.sympify(e))) for c, e in horner.items()},
            "normalization": {"x": "xi - 4", "y": "eta", "velocities": "u/U_in, v/U_in",
                              "psi": "psi/(U_in*D); gauge psi(x=0,y=0)=0"},
            "velocity_rule": "u=diff(psi,y), v=-diff(psi,x)" if choice["kind"] == "streamfunction" else "direct u,v",
            "derivative_rule": "Differentiate analytically in x,y; these equal derivatives in xi,eta."}


def check_numerics():
    previous = check_previous_numerics()
    rng = np.random.default_rng(924)
    points = rng.uniform([-0.95, -1.8], [0.95, 1.8], size=(27, 2))
    errors, hard_residuals = [], []
    for kind in ("direct", "streamfunction"):
        degrees = (2, 5, 3, 4)
        indices = dictionary_indices(kind, degrees)
        coeff = rng.normal(size=len(indices)) / 20
        choice = {"kind": kind, "degrees": degrees, "coefficients": coeff.tolist()}
        basis = velocity_basis(points, kind, degrees)
        numerical = predict_basis(basis, coeff)
        symbolic = symbolic_expressions(choice)
        for c in ("u", "v"):
            for key, (nx, ny) in zip(KEYS, ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))):
                f = sp.lambdify((X, Y), sp.diff(symbolic[c], X, nx, Y, ny), "numpy")
                errors.append(float(np.max(np.abs(numerical[c][key] - f(*points.T)))))
        if kind == "streamfunction":
            hard_residuals.extend(float(np.max(np.abs(a))) for a in residual_basis(basis).values())
            assert sp.expand(sp.diff(symbolic["u"], X) + sp.diff(symbolic["v"], Y)) == 0
    assert max(errors) < 1e-10 and max(hard_residuals) == 0
    # The quadrature must integrate every direct residual/gradient product
    # exactly at the largest allowed degree, not only penalize a coarse grid.
    grams = []
    for nx, ny in ((16, 24), (24, 40)):
        points, weights = quadrature(nx, ny)
        r = residual_basis(velocity_basis(points, "direct", (3, 12, 3, 12)))
        grams.append({k: a.T @ (weights[:, None] * a) for k, a in r.items()})
    quad_error = max(float(np.linalg.norm(grams[0][k] - grams[1][k]) / np.linalg.norm(grams[1][k])) for k in grams[0])
    assert quad_error < 1e-11
    # Exact incompressible manufactured field, fitted using values and gradients.
    points = rng.uniform([-0.9, -1.7], [0.9, 1.7], size=(200, 2))
    basis = velocity_basis(points, "streamfunction", (2, 3, 2, 3))
    choice = {"kind": "streamfunction", "degrees": [2, 3, 2, 3],
              "coefficients": rng.normal(size=len(dictionary_indices("streamfunction", (2, 3, 2, 3)))).tolist()}
    values = predict_choice(choice, points)
    scales = {c: {"value": max(float(np.var(d["q"])), 1e-12),
                   "dx": float(np.mean(d["dx"] ** 2)), "dy": float(np.mean(d["dy"] ** 2))} for c, d in values.items()}
    a, b = data_system(basis, values, scales)
    ag, bg = data_system(basis, values, scales, gradient_only=True)
    a, b = np.vstack([a, ag]), np.concatenate([b, bg])
    fit = np.linalg.lstsq(a, b, rcond=None)[0]
    manufactured_error = float(np.linalg.norm(a @ fit - b) / np.linalg.norm(b))
    assert manufactured_error < 1e-12
    return {"previous_stencil_and_derivative_checks": previous,
            "analytic_basis_through_second_velocity_derivatives_max_error": max(errors),
            "hard_continuity_and_gradient_basis_max_error": max(hard_residuals),
            "quadrature_refinement_max_relative_gram_error": quad_error,
            "manufactured_streamfunction_fit_relative_error": manufactured_error}


def prepare(args):
    manifest = dataset_manifest(args.input_dir)
    frame = pd.read_csv(args.input_dir / "wake_data.csv").sort_values(["x_D", "y_D"])
    xi, eta = np.sort(frame.x_D.unique()), np.sort(frame.y_D.unique())
    if (len(xi), len(eta)) != (33, 64):
        raise ValueError("Expected the dense 33 x 64 wake")
    fields = {"u": frame.u_star.to_numpy().reshape(33, 64), "v": np.load(args.input_dir / "v_mean_wake.npy", allow_pickle=False)}
    parts = {c: partition_field(q, xi, eta, BLOCKS) for c, q in fields.items()}
    radius = manifest["normalization"]["D"] / 2
    scales = {}
    for c, d in parts.items():
        scales[c] = scales_for(d["train"])
        scales[c]["K"] = float(np.mean(transport(d["train"], d["train"]["dx"], d["train"]["dy"], radius) ** 2))
    continuity_scale = scales["u"]["dx"] + scales["v"]["dy"]
    files = [Path(__file__).resolve(), ROOT / "experiments/wake_operator_aware_sr.py",
             ROOT / "experiments/wake_derivative_aware_sr.py", ROOT / "experiments/p2_analytic_derivative_loss.py",
             ROOT / "pipeline_contracts.py", ROOT / "experiments/wake_derivative_loss.jl", ROOT / "experiments/wake_operator_loss.jl"]
    prior = args.prior_dir / "selected_velocity_models.json"
    protocol = {"dataset": manifest, "sources": {str(p.relative_to(ROOT)): file_hash(p) for p in files},
                "prior_selected_models_sha256": file_hash(prior), "blocks": BLOCKS,
                "scales": scales, "radius": radius, "continuity_scale": continuity_scale,
                "settings": {"dictionaries": DICTIONARIES, "total_budgets": TOTAL_BUDGETS,
                             "per_direct_component_cap": PER_COMPONENT_CAP, "gradient_weights": GRAD_WEIGHTS,
                             "physics_weights": PHYSICS_WEIGHTS, "residual_gradient_weights": RESIDUAL_GRAD_WEIGHTS,
                             "ridge_on_normalized_columns": RIDGE, "quadrature": [16, 24],
                             "residual_gradient_length_scales": [1.0, ETA_SCALE],
                             "selection": "sum over u,v of validation L_value + L_gradient + L_K; no physics term",
                             "coordinate_shift": [4.0, 0.0], "eta_basis_scale": ETA_SCALE},
                "sizes": {s: len(d["q"]) for s, d in parts["u"].items()},
                "split_hashes": {c: {s: array_hash(d["points"], d["q"], d["dx"], d["dy"]) for s, d in p.items()} for c, p in parts.items()},
                "python": platform.python_version(), "packages": {k: package_metadata.version(k) for k in ("numpy", "scipy", "sympy", "pandas")},
                "checks": check_numerics(),
                "scope": "Continued development on the previously inspected same CFD field and v2 spatial split; not independent new-case validation.",
                "physics": "Only incompressibility of time-averaged velocity. No steady momentum residual, pressure assumption or Reynolds-stress closure.",
                "collocation": "384 label-free Gauss points across the full prescribed wake rectangle, including held-out coordinates. No held-out values or derivatives in training."}
    protocol["protocol_id"] = content_hash(protocol)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pp = args.out_dir / "protocol.json"
    if pp.exists() and read_json(pp)["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("Code, inputs or protocol changed: use a fresh --out-dir")
    atomic_json(pp, protocol)
    np.savez_compressed(args.out_dir / "data_splits.npz", xi=xi, eta=eta,
                        **{f"{c}_{s}_{k}": v for c, p in parts.items() for s, d in p.items() for k, v in d.items()})
    return parts, protocol


def previous_baseline(args, parts, protocol):
    old = read_json(args.prior_dir / "selected_velocity_models.json")
    models = old["models"]
    degrees = [models[c][k] for c in ("u", "v") for k in ("degree_x", "degree_eta")]
    coefficients, errors = [], []
    for c in ("u", "v"):
        if models[c]["method"] != "sparse_value":
            raise ValueError("This comparison expects the frozen v2 value baseline")
        basis, _ = basis_arrays(parts[c]["train"]["features"], *degrees[(0 if c == "u" else 2):(2 if c == "u" else 4)])
        a, b = design_system(basis, parts[c]["train"], protocol["scales"][c], protocol["radius"], "value")
        _, fit = list(sparse_path(a, b, [models[c]["n_terms"]]))[-1]
        error = float(np.max(np.abs(fit - models[c]["coefficients"])))
        assert error < 1e-11, "Prior model not reproducible"
        errors.append(error); coefficients.extend(fit.tolist())
    record = {"method": "previous_v2", "candidate_id": "previous_v2", "kind": "direct", "degrees": degrees,
              "gradient_weight": 0.0, "physics_weight": 0.0, "residual_gradient_weight": 0.0,
              "n_coefficients": int(np.count_nonzero(coefficients)), "coefficients": coefficients,
              "refit_max_coefficient_error": max(errors)}
    validation = {c: d["validation"] for c, d in parts.items()}
    record["validation"] = score_pair(predict_choice(record, validation["u"]["features"]), validation,
                                       protocol["scales"], protocol["radius"])
    return record


def candidate_table(records):
    rows = []
    for r in records:
        row = {k: r[k] for k in ("candidate_id", "method", "kind", "degrees", "gradient_weight", "physics_weight", "residual_gradient_weight", "n_coefficients")}
        row.update({k: v for k, v in r["validation"].items() if k != "fields"})
        row.update({f"{c}_{k}": v for c, metrics in r["validation"]["fields"].items() for k, v in metrics.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def fit_all(args, parts, protocol):
    frozen_path = args.out_dir / "frozen_selection.json"
    if frozen_path.exists():
        frozen = read_json(frozen_path)
        assert frozen["protocol_id"] == protocol["protocol_id"]
        return frozen
    start = time.perf_counter()
    train = {c: d["train"] for c, d in parts.items()}
    validation = {c: d["validation"] for c, d in parts.items()}
    qpoints, weights = quadrature()
    records = []
    path_count = 0
    for di, degrees in enumerate(DICTIONARIES):
        for kind in ("direct", "streamfunction"):
            indices = dictionary_indices(kind, degrees)
            bt = velocity_basis(train["u"]["features"], kind, degrees)
            bv = velocity_basis(validation["u"]["features"], kind, degrees)
            bc = velocity_basis(qpoints, kind, degrees)
            gram = gram_parts(bt, bc, train, protocol["scales"], protocol["continuity_scale"], weights)
            for method, gw, pw, rgw in configs(kind):
                g, h = objective_gram(gram, gw, pw, rgw)
                for count, coeff in sparse_gram_path(g, h, indices, kind):
                    pred = predict_basis(bv, coeff)
                    validation_score = score_pair(pred, validation, protocol["scales"], protocol["radius"])
                    records.append({"candidate_id": f"d{di}_{method}_g{gw:g}_p{pw:g}_r{rgw:g}_n{count}",
                                    "method": method, "kind": kind, "degrees": list(degrees),
                                    "gradient_weight": gw, "physics_weight": pw, "residual_gradient_weight": rgw,
                                    "n_coefficients": count, "coefficients": coeff.tolist(), "validation": validation_score})
                path_count += 1
        print(f"DICTIONARY {di+1}/{len(DICTIONARIES)} {degrees}: {len(records)} candidates; {time.perf_counter()-start:.1f}s", flush=True)
    baseline = previous_baseline(args, parts, protocol)
    records.append(baseline)
    finalists = {m: choose([r for r in records if r["method"] == m]) for m in METHODS}
    finalists["previous_v2"] = baseline
    # Mechanistic ablation: remove physics from the chosen soft model while
    # keeping its dictionary, selected support and derivative weight fixed.
    # These fits are frozen before test evaluation and are diagnostic only.
    ablations = {}
    for name in ("pinn_continuity", "gpinn_continuity"):
        chosen = finalists[name]
        bt = velocity_basis(train["u"]["features"], "direct", chosen["degrees"])
        bc = velocity_basis(qpoints, "direct", chosen["degrees"])
        gram = gram_parts(bt, bc, train, protocol["scales"], protocol["continuity_scale"], weights)
        g, h = objective_gram(gram, chosen["gradient_weight"], 0.0, 0.0)
        support = np.flatnonzero(chosen["coefficients"])
        norms = np.sqrt(np.diag(g))[support]
        z = g[np.ix_(support, support)] / norms[:, None] / norms[None, :]
        coeff = np.zeros(len(h))
        coeff[support] = np.linalg.solve(z + RIDGE * np.eye(len(support)), h[support] / norms) / norms
        ablation = {**chosen, "method": name + "_fixed_support_no_physics",
                    "candidate_id": chosen["candidate_id"] + "_no_physics", "physics_weight": 0.0,
                    "residual_gradient_weight": 0.0, "coefficients": coeff.tolist()}
        ablation["validation"] = score_pair(predict_choice(ablation, validation["u"]["features"]), validation, protocol["scales"], protocol["radius"])
        ablations[ablation["method"]] = ablation
    best = choose(list(finalists.values()))
    compact = {}
    for budget in TOTAL_BUDGETS:
        compact[str(budget)] = {method: choose([r for r in records if r["method"] == method and r["n_coefficients"] <= budget]) for method in METHODS}
    atomic_json(args.out_dir / "validation_candidates.json", records)
    candidate_table(records).to_csv(args.out_dir / "validation_candidates.csv", index=False)
    frozen = {"protocol_id": protocol["protocol_id"], "created_at": datetime.now().astimezone().isoformat(),
              "selection": protocol["settings"]["selection"], "selected_method": best["method"],
              "candidate_count": len(records), "sparse_path_count": path_count,
              "fit_seconds": time.perf_counter() - start, "finalists": finalists,
              "fixed_support_ablations": ablations, "compact_choices": compact}
    atomic_json(frozen_path, frozen)
    atomic_json(args.out_dir / "selected_velocity_models.json",
                {"protocol_id": protocol["protocol_id"], "frozen_selection_sha256": file_hash(frozen_path),
                 "domain": {"xi": [3.0, 5.0], "eta": [-1.9, 1.9]}, "radius": protocol["radius"],
                 "physical_normalization": protocol["dataset"]["normalization"], "model": model_export(best)})
    atomic_json(args.out_dir / "finalist_formulas.json", {m: model_export(r) for m, r in finalists.items()})
    print(candidate_table(list(finalists.values()))[["method", "degrees", "gradient_weight", "physics_weight", "residual_gradient_weight", "n_coefficients", "score"]].to_string(index=False), flush=True)
    print(f"FROZEN: {best['method']}; {len(records)} candidates / {path_count} paths", flush=True)
    return frozen


def evaluate(args, parts, protocol, frozen):
    # Called only after every main, compact and mechanistic choice is on disk.
    frozen_hash = file_hash(args.out_dir / "frozen_selection.json")
    test = {c: p["test"] for c, p in parts.items()}
    rows = []
    candidates = {**frozen["finalists"], **frozen["fixed_support_ablations"]}
    for method, choice in candidates.items():
        pred = predict_choice(choice, test["u"]["features"])
        for reference in ("raw", "smooth"):
            metrics = score_pair(pred, test, protocol["scales"], protocol["radius"], smooth=reference == "smooth")
            for c, values in metrics["fields"].items():
                rows.append({"method": method, "component": c, "reference": reference,
                             "selected": method == frozen["selected_method"], "n_coefficients_total": choice["n_coefficients"],
                             "pair_score": metrics["score"], **{k: v for k, v in metrics.items() if k not in ("fields", "score")}, **values})
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out_dir / "test_metrics.csv", index=False)
    frame[frame.selected].to_csv(args.out_dir / "recommended_test_metrics.csv", index=False)
    compact = []
    for budget, choices in frozen["compact_choices"].items():
        for method, choice in choices.items():
            metrics = score_pair(predict_choice(choice, test["u"]["features"]), test, protocol["scales"], protocol["radius"])
            for c, values in metrics["fields"].items():
                compact.append({"budget_total": int(budget), "method": method, "component": c,
                                "n_coefficients_total": choice["n_coefficients"], **values})
    pd.DataFrame(compact).to_csv(args.out_dir / "compact_test_metrics.csv", index=False)
    plots(args, parts, protocol, frozen, frame)
    assert file_hash(args.out_dir / "frozen_selection.json") == frozen_hash
    atomic_json(args.out_dir / "complete.json", {"protocol_id": protocol["protocol_id"], "frozen_selection_sha256": frozen_hash,
                "created_at": datetime.now().astimezone().isoformat(), "selected_method": frozen["selected_method"],
                "candidate_count": frozen["candidate_count"], "sparse_path_count": frozen["sparse_path_count"],
                "fit_seconds": frozen["fit_seconds"]})
    print(frame[(frame.reference == "raw") & ~frame.method.str.contains("fixed_support")][
        ["method", "component", "value_rel_l2", "dx_rel_l2", "gradient_rel_l2", "K_R_rel_l2", "divergence_rms"]].to_string(index=False), flush=True)


def plots(args, parts, protocol, frozen, frame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["previous_v2", *METHODS]
    labels = ["Previous v2", "Data/Sobolev", "PINN continuity", "gPINN continuity", "Hard streamfunction"]
    colors = ["#7e858c", "#e9b458", "#51a7a0", "#4575b4", "#ba6582"]
    raw = frame[(frame.reference == "raw") & frame.method.isin(names)]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.9), constrained_layout=True)
    for row, c in enumerate(("u", "v")):
        part = raw[raw.component == c].set_index("method").loc[names]
        for col, (key, title) in enumerate((("dx_rel_l2", "Streamwise derivative"), ("gradient_rel_l2", "Combined gradient"), ("K_R_rel_l2", "Coordinate transport K_R"))):
            ax = axes[row, col]
            values = 100 * part[key].to_numpy()
            ax.bar(np.arange(len(names)), values, color=colors)
            for i, value in enumerate(values):
                ax.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
            ax.set_xticks(np.arange(len(names))); ax.set_xticklabels(labels, rotation=27, ha="right", fontsize=8)
            ax.set_ylabel(f"{c}: relative L2 error (%)"); ax.set_title(title)
            ax.set_ylim(0, max(values) * 1.22); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Same CFD field and spatial holdout; all models selected on validation", fontsize=13)
    fig.savefig(args.out_dir / "comparison.png", dpi=180); plt.close(fig)
    test = {c: p["test"] for c, p in parts.items()}
    xi = np.unique(test["u"]["points"][:, 0])[1]
    keep = np.isclose(test["u"]["points"][:, 0], xi)
    eta = test["u"]["points"][keep, 1]
    selected = frozen["selected_method"]
    pair = ["previous_v2", selected] if selected != "previous_v2" else ["previous_v2", "hard_streamfunction"]
    preds = {m: predict_choice(frozen["finalists"][m], test["u"]["features"]) for m in pair}
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), constrained_layout=True)
    for row, c in enumerate(("u", "v")):
        for col, key in enumerate(("q", "dx", "dy", "K")):
            ax = axes[row, col]
            target = transport(test[c], test[c]["dx"], test[c]["dy"], protocol["radius"]) if key == "K" else test[c][key]
            ax.plot(eta, target[keep], ".", color="#333333", ms=3, label="CFD / raw FD")
            if key != "q":
                smooth = transport(test[c], test[c]["smooth_dx"], test[c]["smooth_dy"], protocol["radius"]) if key == "K" else test[c]["smooth_" + key]
                ax.plot(eta, smooth[keep], color="#aaaaaa", lw=1.0, label="Smoothed FD")
            for m, p in preds.items():
                values = transport(test[c], p[c]["dx"], p[c]["dy"], protocol["radius"]) if key == "K" else p[c][key]
                ax.plot(eta, values[keep], lw=1.7, label=m, color=colors[names.index(m)])
            ax.set_title(f"{c}: {key}"); ax.set_xlabel("eta"); ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Frozen test profile at xi={xi:.4f}; selection: {selected}")
    fig.savefig(args.out_dir / "selected_test_profiles.png", dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "outputs_mgn/comsol/eta_sym")
    parser.add_argument("--prior-dir", type=Path, default=ROOT / "outputs/wake_operator_aware_sr_v2")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/wake_pinn_informed_sr_v3")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--fit-only", action="store_true")
    modes.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    parts, protocol = prepare(args)
    if args.prepare_only:
        print(json.dumps(protocol["checks"], indent=2)); return
    if args.evaluate_only:
        frozen = read_json(args.out_dir / "frozen_selection.json")
        assert frozen["protocol_id"] == protocol["protocol_id"]
    else:
        frozen = fit_all(args, parts, protocol)
    if not args.fit_only:
        evaluate(args, parts, protocol, frozen)


if __name__ == "__main__":
    main()
