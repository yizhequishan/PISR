"""Read-only checkpoint diagnosis: replay samples, localize residuals, audit mass.

Writes new diagnostic artifacts only; does not train or modify the source run.
The replay is verified against the recorded final training objective.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

# Keep NumPy's MKL from loading a second threaded OpenMP runtime next to Torch.
# This affects this diagnostic process only; Torch's thread count is set below.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pinn.evaluation import predict
from pinn.losses import loss_terms, weighted_loss
from pinn.physics import gradient, navier_stokes
from pinn.train import load_checkpoint

NAMES = ("continuity", "momentum_x", "momentum_y")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    return {"mean": float(np.mean(values)), "rms": float(np.sqrt(np.mean(values**2))),
            "p99_abs": float(np.quantile(np.abs(values), .99)),
            "max_abs": float(np.max(np.abs(values)))}


def residuals(model, points, re, batch_size=512):
    parameter = next(model.parameters())
    chunks = []
    for start in range(0, len(points), batch_size):
        q = torch.as_tensor(points[start:start+batch_size], dtype=parameter.dtype,
                            device=parameter.device)
        result = navier_stokes(model, q, re)
        chunks.append(torch.cat([result[name] for name in NAMES], 1).detach().cpu().numpy())
    return np.concatenate(chunks)


def cosine(a, b):
    denominator = float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
    return float(torch.dot(a, b)) / denominator if denominator else None


def loss_diagnostics(model, problem, batch):
    terms = loss_terms(model, problem, batch)
    config = problem.config
    groups = {
        "continuity": config.weight_pde * terms["pde_continuity"],
        "momentum": config.weight_pde * (terms["pde_momentum_x"] + terms["pde_momentum_y"]),
        "pde": config.weight_pde * sum(value for key, value in terms.items() if key.startswith("pde_")),
        "boundary": config.weight_boundary * sum(value for key, value in terms.items() if key.startswith("boundary_")),
        "total": weighted_loss(terms, config),
    }
    parameters = list(model.parameters())
    gradients = {}
    for name, value in groups.items():
        derivatives = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
        gradients[name] = torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                                     for p, g in zip(parameters, derivatives)])
    result = {
        "terms_mse": {name: float(value.detach()) for name, value in terms.items()},
        "weighted_losses": {name: float(value.detach()) for name, value in groups.items()},
        "gradient_l2": {name: float(torch.linalg.vector_norm(g)) for name, g in gradients.items()},
        "gradient_max_abs": {name: float(g.abs().max()) for name, g in gradients.items()},
        "gradient_cosines": {a + "__" + b: cosine(gradients[a], gradients[b])
                             for a, b in (("pde", "boundary"), ("continuity", "momentum"))},
        "interpretation": "Single final-checkpoint gradient snapshot; not a causal diagnosis of the training trajectory.",
    }
    return result


def gauss_interval(a, b, n):
    z, w = np.polynomial.legendre.leggauss(n)
    return (a+b)/2 + (b-a)*z/2, (b-a)*w/2


def fluid_quadrature(config, nx, ny):
    """Gaussian quadrature over the channel excluding the circular obstacle.

    x=cx+r*sin(theta) smooths the two circle-intersecting vertical strips.
    No quadrature nodes are evaluated inside the solid.
    """
    cx, cy, r = config.cylinder_x, config.cylinder_y, .5
    points, weights = [], []
    ys, yw = gauss_interval(0, config.channel_height, ny)
    for a, b in ((0, cx-r), (cx+r, config.channel_length)):
        xs, xw = gauss_interval(a, b, nx)
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        points.append(np.column_stack((xx.ravel(), yy.ravel())))
        weights.append((xw[:, None] * yw[None, :]).ravel())
    theta, tw = gauss_interval(-math.pi/2, math.pi/2, nx)
    xs = cx + r*np.sin(theta)
    xw = tw*r*np.cos(theta)
    half_height = r*np.cos(theta)
    unit_y, unit_w = gauss_interval(0, 1, ny)
    for lower, upper in ((np.zeros(nx), cy-half_height),
                         (cy+half_height, np.full(nx, config.channel_height))):
        yy = lower[:, None] + (upper-lower)[:, None]*unit_y[None, :]
        xx = np.broadcast_to(xs[:, None], yy.shape)
        points.append(np.column_stack((xx.ravel(), yy.ravel())))
        weights.append((xw[:, None]*(upper-lower)[:, None]*unit_w[None, :]).ravel())
    return np.concatenate(points), np.concatenate(weights)


def mass_audit(model, problem, n):
    config = problem.config
    sections = []
    for x in sorted(set(np.linspace(0, config.channel_length, 45).tolist() +
                        [config.cylinder_x-.5, config.cylinder_x, config.cylinder_x+.5])):
        intervals = [(0., config.channel_height)]
        if abs(x-config.cylinder_x) < .5:
            half = math.sqrt(.25-(x-config.cylinder_x)**2)
            intervals = [(0., config.cylinder_y-half), (config.cylinder_y+half, config.channel_height)]
        flux = 0.
        for a, b in intervals:
            y, w = gauss_interval(a, b, n)
            q = np.column_stack((np.full_like(y, x), y))
            flux += float(np.dot(w, predict(model, q)[:, 0]))
        sections.append({"x": x, "flux": flux, "relative_to_target": flux/config.channel_height})
    wall_flux = 0.
    for a, b in ((0, config.cylinder_x-.5), (config.cylinder_x-.5, config.cylinder_x+.5),
                 (config.cylinder_x+.5, config.channel_length)):
        x, w = gauss_interval(a, b, n)
        top = predict(model, np.column_stack((x, np.full_like(x, config.channel_height))))[:, 1]
        bottom = predict(model, np.column_stack((x, np.zeros_like(x))))[:, 1]
        wall_flux += float(np.dot(w, top-bottom))
    theta = (np.arange(8*n)+.5)*2*math.pi/(8*n)
    normals = np.column_stack((np.cos(theta), np.sin(theta)))
    circle = np.array([config.cylinder_x, config.cylinder_y]) + .5*normals
    cylinder_flux = float(-np.sum(predict(model, circle)[:, :2]*normals)*math.pi/(8*n))
    boundary_flux = sections[-1]["flux"] - sections[0]["flux"] + wall_flux + cylinder_flux
    q, w = fluid_quadrature(config, n, n//2)
    assert problem.contains(q).all()
    expected_area = config.channel_length*config.channel_height-math.pi*.25
    if abs(w.sum()-expected_area) > 1e-9:
        raise ValueError("Invalid fluid-domain quadrature area")
    divergence_chunks = []
    parameter = next(model.parameters())
    for start in range(0, len(q), 1024):
        coords = torch.tensor(q[start:start+1024], dtype=parameter.dtype, requires_grad=True)
        u, v, _ = model(coords).split(1, 1)
        divergence = gradient(u, coords)[:, :1] + gradient(v, coords)[:, 1:2]
        divergence_chunks.append(divergence.detach().numpy().ravel())
    divergence = np.concatenate(divergence_chunks)
    volume_flux = float(np.dot(w, divergence))
    return {"quadrature_order": n, "volume_points": len(q), "fluid_area": float(w.sum()),
            "sections": sections, "wall_outflow": wall_flux, "cylinder_outflow": cylinder_flux,
            "net_boundary_outflow": boundary_flux, "integrated_divergence": volume_flux,
            "divergence_theorem_absolute_error": abs(boundary_flux-volume_flux),
            "divergence_mean": volume_flux/expected_area,
            "divergence_rms": float(np.sqrt(np.dot(w, divergence**2)/expected_area))}


def finite_difference_check(model, points, re, exact):
    center = predict(model, points)
    checks = []
    for h in (1e-4, 5e-5):
        first, second = [], []
        for axis in (0, 1):
            offset = np.zeros_like(points)
            offset[:, axis] = h
            plus, minus = predict(model, points+offset), predict(model, points-offset)
            first.append((plus-minus)/(2*h))
            second.append((plus-2*center+minus)/h**2)
        dx, dy = first
        lap = second[0]+second[1]
        fd = np.column_stack((dx[:, 0]+dy[:, 1],
                              center[:, 0]*dx[:, 0]+center[:, 1]*dy[:, 0]+dx[:, 2]-lap[:, 0]/re,
                              center[:, 0]*dx[:, 1]+center[:, 1]*dy[:, 1]+dy[:, 2]-lap[:, 1]/re))
        checks.append({"step": h, "max_absolute_difference": dict(zip(NAMES, np.max(np.abs(fd-exact), axis=0).tolist()))})
    return {"points": len(points), "checks": checks,
            "interpretation": "Finite differences of the same network check derivative implementation, not CFD accuracy."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-points", type=int, default=32768)
    args = parser.parse_args()
    run, out = args.run_dir.resolve(), args.out_dir.resolve()
    if args.n_points < 1024:
        raise ValueError("Use at least 1024 independent points")
    if out == run or (out.exists() and any(out.iterdir())):
        raise ValueError("Choose a new or empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    recorded_environment = json.loads((run/"environment.json").read_text(encoding="utf-8"))
    source_check = {name: digest(ROOT/Path(name)) == sha
                    for name, sha in recorded_environment["source_sha256"].items()}
    if not all(source_check.values()):
        raise ValueError("Source changed since training; exact sample replay is not certified")
    config, problem, model, saved = load_checkpoint(run/"checkpoint_last.pt", "cpu")
    torch.set_num_threads(2)
    if config.case != "cylinder" or config.transient or recorded_environment.get("training_data"):
        raise ValueError("This diagnosis requires an unobserved steady cylinder run")
    if saved["progress"]["status"] != "training_finished":
        raise ValueError("Diagnose a completed, immutable checkpoint")
    model.eval()
    with (run/"history.csv").open(encoding="utf-8", newline="") as stream:
        history = list(csv.DictReader(stream))
    recorded_metrics = json.loads((run/"evaluation"/"metrics.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(config.seed)
    batch = problem.sample(rng, "cpu", torch.float64)
    for step in range(1, saved["progress"]["adam_steps"]+1):
        if step > 1 and (step-1) % config.resample_every == 0:
            batch = problem.sample(rng, "cpu", torch.float64)
    if saved["progress"]["lbfgs_steps"]:
        batch = problem.sample(rng, "cpu", torch.float64)
    training = loss_diagnostics(model, problem, batch)
    training_difference = abs(training["weighted_losses"]["total"]-float(history[-1]["total"]))
    if training_difference > 1e-9:
        raise ValueError("Final loss did not reproduce; check sampling/source provenance")
    validation = problem.sample(np.random.default_rng(config.seed+1_000_003), "cpu", torch.float64)
    independent = loss_diagnostics(model, problem, validation)
    validation_difference = abs(independent["weighted_losses"]["total"]-recorded_metrics["validation_loss"])
    if validation_difference > 1e-9:
        raise ValueError("Stored validation metric did not reproduce")
    print("Replayed final collocation batch and reproduced validation loss.", flush=True)
    seed = 20260924
    points = problem.interior(np.random.default_rng(seed), args.n_points)
    test_residual = residuals(model, points, config.reynolds)
    train_points = batch["interior"].numpy()
    train_residual = residuals(model, train_points, config.reynolds)
    regions = []
    limits = (0., 1.5, 2.5, 4., 8., 16., config.channel_length)
    for a, b in zip(limits[:-1], limits[1:]):
        region = (points[:, 0] >= a) & (points[:, 0] < b)
        train_region = (train_points[:, 0] >= a) & (train_points[:, 0] < b)
        row = {"x_min": a, "x_max": b, "test_points": int(region.sum()), "train_points": int(train_region.sum())}
        for i, name in enumerate(NAMES):
            row[name+"_rms"] = float(np.sqrt(np.mean(test_residual[region, i]**2)))
            row[name+"_squared_residual_share"] = float(np.sum(test_residual[region, i]**2)/np.sum(test_residual[:, i]**2))
        regions.append(row)
    coverage = []
    for width in (.1, .25, .5):
        radius = np.linalg.norm(points-np.array([config.cylinder_x, config.cylinder_y]), axis=1)
        train_radius = np.linalg.norm(train_points-np.array([config.cylinder_x, config.cylinder_y]), axis=1)
        test_mask, train_mask = radius < .5+width, train_radius < .5+width
        fraction = math.pi*((.5+width)**2-.25)/(config.channel_length*config.channel_height-math.pi*.25)
        coverage.append({"annulus_width": width, "area_fraction": fraction,
                         "expected_train_points": config.n_interior*fraction,
                         "actual_train_points": int(train_mask.sum()), "test_points": int(test_mask.sum()),
                         "test_residual_rms": dict(zip(NAMES, np.sqrt(np.mean(test_residual[test_mask]**2, axis=0)).tolist())),
                         "train_residual_rms": dict(zip(NAMES, np.sqrt(np.mean(train_residual[train_mask]**2, axis=0)).tolist()))})
    safe = (points[:, 0] > .001) & (points[:, 0] < config.channel_length-.001)
    safe &= (points[:, 1] > .001) & (points[:, 1] < config.channel_height-.001)
    safe &= np.linalg.norm(points-np.array([config.cylinder_x, config.cylinder_y]), axis=1) > .501
    safe_indices = np.flatnonzero(safe)
    worst_indices = safe_indices[np.argsort(np.linalg.norm(test_residual[safe], axis=1))[-16:]]
    fd_indices = np.concatenate((safe_indices[:32], worst_indices))
    fd = finite_difference_check(model, points[fd_indices], config.reynolds, test_residual[fd_indices])
    print("Independent spatial residuals and finite-difference checks complete.", flush=True)
    mass = [mass_audit(model, problem, n) for n in (64, 128)]
    lbfgs = [row for row in history if row["phase"] == "lbfgs"]
    last_improvement = (float(lbfgs[-2]["total"])-float(lbfgs[-1]["total"]))/float(lbfgs[-2]["total"]) if len(lbfgs)>1 else None
    result = {
        "run_dir": str(run), "checkpoint_sha256": digest(run/"checkpoint_last.pt"),
        "source_hash_matches": source_check, "checkpoint_progress": saved["progress"],
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "history_initial_loss": float(history[0]["total"]),
        "history_final_loss": float(history[-1]["total"]),
        "history_final_record_interval_relative_improvement": last_improvement,
        "history_final_record_interval_steps": int(lbfgs[-1]["step"])-int(lbfgs[-2]["step"]) if len(lbfgs)>1 else None,
        "final_training": training, "original_validation": independent,
        "training_reproduction_absolute_error": training_difference,
        "validation_reproduction_absolute_error": validation_difference,
        "validation_to_training_loss_ratio": independent["weighted_losses"]["total"]/training["weighted_losses"]["total"],
        "fresh_sample_seed": seed, "fresh_sample_size": len(points),
        "fresh_residuals": {name: stats(test_residual[:, i]) for i, name in enumerate(NAMES)},
        "spatial_regions": regions, "near_cylinder_sampling": coverage,
        "finite_difference_check": fd, "mass_conservation": mass,
        "reference_field_supplied": bool(json.loads((run/"evaluation_input.json").read_text(encoding="utf-8")).get("reference")),
        "limitations": ["No independent CFD accuracy comparison.",
                        "Final checkpoint only; loss history does not contain historical validation or flux.",
                        "Residual sampling bias and gradient conflicts are observations, not proof of a unique failure cause.",
                        "No training or hyperparameter changes were made by this diagnostic."],
    }
    (out/"diagnosis.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    with (out/"residual_regions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(regions[0]))
        writer.writeheader()
        writer.writerows(regions)
    np.savez_compressed(out/"diagnostic_samples.npz", coords=points, residuals=test_residual,
                        training_coords=train_points, training_residuals=train_residual)
    print(json.dumps({"output": str(out), "validation_train_ratio": result["validation_to_training_loss_ratio"],
                      "fresh_residual_rms": {k: v["rms"] for k, v in result["fresh_residuals"].items()},
                      "mass": {k: v for k, v in mass[-1].items() if k != "sections"},
                      "gradient_cosines": training["gradient_cosines"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
