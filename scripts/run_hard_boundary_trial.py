"""Matched-budget soft streamfunction and analytic hard-boundary ablations."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn.config import Config
from pinn.evaluation import evaluate, predict
from pinn.train import new_output, setup, train, write_json
from scripts.audit_pinn_cylinder import audit


VARIANTS = ("soft", "zero", "poiseuille", "cylinder_poiseuille")


def section_intervals(x, config):
    """Integrate only fluid, with exact cylinder intersections as endpoints."""
    offset = x - config.cylinder_x
    if abs(offset) < .5:
        half = np.sqrt(.25 - offset**2)
        return [(0., config.cylinder_y-half), (config.cylinder_y+half, config.channel_height)]
    return [(0., config.channel_height)]


def flux_diagnostics(model, problem, orders=(64, 128, 256)):
    c = problem.config
    parameter = next(model.parameters())
    positions = sorted(set(x for x in (0., 1., c.cylinder_x-.5, c.cylinder_x-.25,
                       c.cylinder_x, c.cylinder_x+.25, c.cylinder_x+.5, 3., 4., 6.,
                       10., 16., c.channel_length) if 0 <= x <= c.channel_length))
    result = {"prescribed_flux": c.channel_height, "quadrature": "split-fluid Gauss-Legendre",
              "orders": list(orders), "sections": []}
    for x in positions:
        intervals = section_intervals(x, c)
        flows = []
        for order in orders:
            nodes, weights = np.polynomial.legendre.leggauss(order)
            flow = 0.
            for a, b in intervals:
                y = (a+b)/2 + (b-a)/2*nodes
                coords = np.column_stack((np.full_like(y, x), y))
                flow += (b-a)/2 * np.dot(weights, predict(model, coords)[:, 0])
            flows.append(float(flow))
        endpoints = np.array([[x, y] for interval in intervals for y in interval])
        with torch.no_grad():
            psi, _ = model.streamfunction_pressure(torch.as_tensor(
                endpoints, dtype=parameter.dtype, device=parameter.device))
        endpoint_flux = float(np.diff(psi.cpu().numpy().reshape(-1, 2), axis=1).sum())
        result["sections"].append({"x": x, "fluid_intervals": intervals,
            "flux_by_order": flows, "endpoint_flux": endpoint_flux,
            "relative_error": (flows[-1]-c.channel_height)/c.channel_height,
            "quadrature_endpoint_difference": flows[-1]-endpoint_flux})
    result["max_relative_error"] = max(abs(r["relative_error"]) for r in result["sections"])
    result["max_quadrature_endpoint_difference"] = max(abs(r["quadrature_endpoint_difference"]) for r in result["sections"])
    return result


def boundary_diagnostics(model, problem):
    parameter = next(model.parameters())
    batch = problem.sample(np.random.default_rng(20261013), parameter.device, parameter.dtype)
    result = {}
    with torch.no_grad():
        for name in ("left", "bottom", "top", "obstacle"):
            q = batch[name]
            error = model(q)[:, :2]
            if name == "left":
                error = error - problem.inlet_velocity(q)
            result[name] = {"rms": float(error.square().mean().sqrt().cpu()),
                            "max_abs": float(error.abs().max().cpu())}
    return result


def assess(runs):
    if "soft" not in runs or "cylinder_poiseuille" not in runs:
        return {"status": "incomplete_comparison", "success": None}
    base, hard = runs["soft"], runs["cylinder_poiseuille"]
    checks = {"flux_below_one_percent": max(hard["max_flux_relative_error"],
                                          hard["split_flux"]["max_relative_error"]) < .01}
    for axis in ("x", "y"):
        key = "momentum_" + axis
        checks[key + "_no_worse"] = hard["pde_rms"][key] <= base["pde_rms"][key]
    checks["velocity_boundaries_exact"] = all(
        hard["boundary_errors"][key]["max_abs"] < 1e-10 for key in
        ("inlet_velocity_error", "bottom_velocity", "top_velocity", "cylinder_velocity"))
    return {"status": "complete_comparison", "success": all(checks.values()), "checks": checks,
            "criterion": "Both momentum component RMS values <= matched pure neural streamfunction; max cross-section flux error <1%; exact velocity boundaries.",
            "sr_used": False}


def run_variant(config_values, variant, out, init_trial=None, extra_audit_seed=None):
    config_values = dict(config_values)
    config_values.update(architecture="streamfunction" if variant == "soft" else "hard_streamfunction",
                         hard_background="cylinder_poiseuille" if variant == "soft" else variant)
    config = Config.from_dict(config_values)
    run = new_output(Path(out) / variant, "cylinder")
    problem, initial, _, _ = setup(config)
    init_from = Path(init_trial) / variant / "checkpoint_last.pt" if init_trial else None
    if init_from:
        initial.load_state_dict(torch.load(init_from, map_location="cpu", weights_only=True)["model"])
    initial_diagnostics = {"boundary_errors": boundary_diagnostics(initial, problem),
                           "split_flux": flux_diagnostics(initial, problem)}
    write_json(run / "initial_diagnostics.json", initial_diagnostics)
    del initial
    print("\nVariant: " + variant, flush=True)
    problem, model = train(config, run, init_from=init_from)
    metrics = evaluate(model, problem, run / "evaluation", make_plots=False)
    dense = audit(run, config.device)
    split_flux = flux_diagnostics(model, problem)
    write_json(run / "split_flux.json", split_flux)
    with (run / "history.csv").open(encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    record = {"run_directory": str(run), "parameters": sum(p.numel() for p in model.parameters()),
        "admissible_boundary_problem": variant in ("soft", "cylinder_poiseuille"),
        "initial_loss": float(history[0]["total"]), "final_loss": float(history[-1]["total"]),
        "validation_loss": metrics["validation_loss"],
        "pde_rms": {key: value["rms"] for key, value in dense["pde_residuals"].items()},
        "max_flux_relative_error": dense["max_flux_relative_error"], "split_flux": split_flux,
        "boundary_errors": dense["boundary_errors"], "wall_normal_leakage": dense["wall_normal_leakage"],
        "cylinder_normal_leakage": dense["cylinder_normal_leakage"],
        "screening_passes": dense["screening_passes"], "progress": dense["checkpoint_progress"],
        "checkpoint_sha256": hashlib.sha256((run / "checkpoint_last.pt").read_bytes()).hexdigest()}
    if variant == "cylinder_poiseuille":
        record["cylinder_streamfunction_constant"] = float(model.cylinder_constant.detach().cpu())
    if init_from:
        source = torch.load(init_from, map_location="cpu", weights_only=True)
        record["warm_start"] = {"checkpoint": str(init_from.resolve()),
            "sha256": hashlib.sha256(init_from.read_bytes()).hexdigest(),
            "source_progress": source["progress"], "optimizer_state": "reset"}
    if extra_audit_seed is not None:
        extra = audit(run, config.device, seed=extra_audit_seed)
        record["fresh_audit"] = {"seed": extra_audit_seed,
            "pde_rms": {key: value["rms"] for key, value in extra["pde_residuals"].items()},
            "max_flux_relative_error": extra["max_flux_relative_error"]}
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, help="Torch CPU threads per process")
    parser.add_argument("--workers", type=int, default=1, help="Independent CPU trial processes (default: serial)")
    parser.add_argument("--init-trial", type=Path, help="Warm start each variant from a completed matched trial")
    parser.add_argument("--reference-trial", type=Path, help="Also require hard momentum <= this original soft baseline")
    parser.add_argument("--cylinder-refine-fraction", type=float, default=0.)
    parser.add_argument("--inlet-refine-fraction", type=float, default=0.)
    parser.add_argument("--inlet-refine-width", type=float, default=1.)
    parser.add_argument("--extra-audit-seed", type=int, help="Additional independent uniform audit")
    parser.add_argument("--adam-steps", type=int, default=1500)
    parser.add_argument("--lbfgs-steps", type=int, default=300)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--fixed-cylinder-constant", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.variants)) != len(args.variants):
        parser.error("Duplicate variants are not allowed")
    if args.workers < 1 or (args.workers > 1 and args.device != "cpu"):
        parser.error("--workers must be positive; parallel trials require --device cpu")
    base = Config.load(ROOT / "configs/pinn/cylinder_steady_hard.json")
    values = base.to_dict()
    values.update(seed=args.seed, device=args.device, adam_steps=args.adam_steps,
                  lbfgs_steps=args.lbfgs_steps, hard_train_cylinder_constant=not args.fixed_cylinder_constant,
                  cylinder_refine_fraction=args.cylinder_refine_fraction,
                  inlet_refine_fraction=args.inlet_refine_fraction, inlet_refine_width=args.inlet_refine_width)
    if args.threads is not None:
        values["threads"] = args.threads
    base = Config.from_dict(values)
    if args.init_trial:
        for variant in args.variants:
            checkpoint = args.init_trial / variant / "checkpoint_last.pt"
            source = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if source["progress"]["status"] != "training_finished":
                parser.error("Warm-start trials must have completed: " + str(checkpoint))
    reference = None
    if args.reference_trial:
        reference_summary = json.loads((args.reference_trial / "comparison.json").read_text(encoding="utf-8"))
        reference_run = reference_summary["runs"]["soft"]
        if reference_run["progress"]["status"] != "training_finished":
            parser.error("Reference soft baseline must be complete")
        reference = {"trial": str(args.reference_trial.resolve()),
                     "checkpoint_sha256": reference_run["checkpoint_sha256"],
                     "pde_rms": reference_run["pde_rms"]}
        if args.extra_audit_seed is not None:
            reference_audit = audit(args.reference_trial / "soft", args.device, seed=args.extra_audit_seed)
            reference["fresh_pde_rms"] = {key: value["rms"] for key, value in reference_audit["pde_residuals"].items()}
    out = new_output(args.out_dir, "cylinder")
    sources = [*sorted((ROOT / "pinn").glob("*.py")), Path(__file__).resolve(),
               ROOT / "scripts/audit_pinn_cylinder.py", ROOT / "configs/pinn/cylinder_steady_hard.json"]
    for source in sources:
        destination = out / "source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    summary = {"kind": "single_seed_matched_update_budget_hard_boundary_trial", "seed": args.seed,
        "reference_data": None, "sr_used": False, "runs": {}, "original_soft_reference": reference,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "protocol": {"adam_steps": args.adam_steps, "lbfgs_steps": args.lbfgs_steps,
            "device": args.device, "threads_per_worker": base.threads, "workers": args.workers,
            "init_trial": str(args.init_trial.resolve()) if args.init_trial else None,
            "cylinder_refine_fraction": base.cylinder_refine_fraction,
            "cylinder_refine_radius": base.cylinder_refine_radius,
            "inlet_refine_fraction": base.inlet_refine_fraction,
            "inlet_refine_width": base.inlet_refine_width,
            "extra_audit_seed": args.extra_audit_seed,
            "requested_variants": args.variants, "n_interior": base.n_interior,
            "n_boundary_per_segment": base.n_boundary, "dtype": base.dtype,
            "hard_inlet_length": base.hard_inlet_length, "hard_cylinder_length": base.hard_cylinder_length,
            "train_cylinder_constant": base.hard_train_cylinder_constant,
            "momentum_acceptance": "Each component RMS no worse than the matched soft model on the same 20000 independent points",
            "flux_acceptance": "Maximum relative error against prescribed H is <0.01",
            "note": "Identical neural initialization, samples, weights and update counts; not equal time/FLOPs. Zero and raw Poiseuille are incompatible negative controls."}}
    write_json(out / "comparison.json", summary)
    def collect(variant, record):
        summary["runs"][variant] = record
        summary["assessment"] = assess(summary["runs"])
        if ("soft" in summary["runs"] and "cylinder_poiseuille" in summary["runs"]
                and args.extra_audit_seed is not None):
            baseline = summary["runs"]["soft"]["fresh_audit"]["pde_rms"]
            hard = summary["runs"]["cylinder_poiseuille"]["fresh_audit"]["pde_rms"]
            checks = {key: hard[key] <= baseline[key] for key in ("momentum_x", "momentum_y")}
            summary["fresh_audit_assessment"] = {"seed": args.extra_audit_seed,
                                                "momentum_no_worse": checks, "passed": all(checks.values())}
            summary["assessment"]["checks"].update({"fresh_"+key+"_no_worse": value
                                                      for key, value in checks.items()})
            summary["assessment"]["success"] = all(summary["assessment"]["checks"].values())
        if reference is not None and summary["assessment"]["status"] == "complete_comparison":
            hard_run = summary["runs"]["cylinder_poiseuille"]
            checks = summary["assessment"]["checks"]
            for key in ("momentum_x", "momentum_y"):
                checks["original_"+key+"_no_worse"] = hard_run["pde_rms"][key] <= reference["pde_rms"][key]
                if args.extra_audit_seed is not None:
                    checks["original_fresh_"+key+"_no_worse"] = (
                        hard_run["fresh_audit"]["pde_rms"][key] <= reference["fresh_pde_rms"][key])
            summary["assessment"]["success"] = all(checks.values())
            summary["assessment"]["criterion"] += " Also no worse than the original soft baseline on both uniform audits."
        summary["requested_runs_complete"] = set(summary["runs"]) == set(args.variants)
        write_json(out / "comparison.json", summary)
        print(json.dumps({"variant": variant, "pde_rms": record["pde_rms"],
                          "flux_relative_error": record["split_flux"]["max_relative_error"],
                          "assessment": summary["assessment"]}, indent=2), flush=True)
    if args.workers == 1:
        for variant in args.variants:
            collect(variant, run_variant(base.to_dict(), variant, out, args.init_trial, args.extra_audit_seed))
    else:
        # Each process owns one disjoint run directory and reseeds setup().
        # Only the parent writes the shared comparison; optimizer batches and
        # initialization are independent of task completion order.
        with ProcessPoolExecutor(max_workers=min(args.workers, len(args.variants))) as executor:
            pending = {executor.submit(run_variant, base.to_dict(), variant, out, args.init_trial, args.extra_audit_seed): variant
                       for variant in args.variants}
            for future in as_completed(pending):
                collect(pending[future], future.result())
    return out


if __name__ == "__main__":
    main()
