"""Frozen symbolic residual weights for a matched-time steady PINN continuation.

Subcommands prepare, run, summarize keep fitting, optimization and final audit
separate. No reference CFD is assumed; reported improvements concern residuals.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import subprocess

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("PYTHONUTF8", "1")
import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pinn.config import Config
from pinn.evaluation import evaluate
from pinn.model import SympyTorch
from pinn.physics import navier_stokes
from pinn.train import load_checkpoint, new_output, train, write_json
from scripts.audit_pinn_cylinder import audit
from scripts.run_hard_boundary_trial import flux_diagnostics

SEEDS = {"fit": 2026101311, "selection": 2026101312,
         "continuation": 2026101313, "test": 2026101314}
KINDS = ("ordinary", "inlet", "sr")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def features(points, config):
    return np.column_stack((points[:, 0], 2 * points[:, 1] / config.channel_height - 1))


def expression_values(expression, points, config):
    evaluator = SympyTorch(expression).double()
    with torch.no_grad():
        values = evaluator(torch.as_tensor(features(points, config), dtype=torch.float64))
    result = values.numpy().reshape(-1)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite symbolic score")
    return result


def residual_values(model, points, reynolds):
    parameter = next(model.parameters())
    chunks = []
    for start in range(0, len(points), 512):
        q = torch.as_tensor(points[start:start+512], device=parameter.device, dtype=parameter.dtype)
        r = navier_stokes(model, q, reynolds)
        chunks.append(torch.cat([r[k] for k in ("continuity", "momentum_x", "momentum_y")], 1)
                      .detach().cpu().numpy())
    result = np.concatenate(chunks)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite checkpoint residual")
    return result


def region_diagnostics(points, residual, config):
    x, y = points[:, 0], points[:, 1]
    masks = {"inlet_x_lt_1": x < 1,
             "near_cylinder": (x-config.cylinder_x)**2 + (y-config.cylinder_y)**2 < 1.5**2,
             "near_wake": (x >= config.cylinder_x+.5) & (x <= 6) & (abs(y-config.cylinder_y) < 1.5),
             "remaining": np.ones(len(points), dtype=bool)}
    unused = np.ones(len(points), dtype=bool)
    total = (residual[:, 1:]**2).sum(axis=0)
    result = {}
    for name, raw_mask in masks.items():
        mask = raw_mask & unused
        unused &= ~mask
        if not mask.any():
            continue
        r = residual[mask, 1:]
        result[name] = {"n": int(mask.sum()), "fraction": float(mask.mean()),
                        "momentum_rms": np.sqrt((r*r).mean(axis=0)).tolist(),
                        "squared_residual_share": ((r*r).sum(axis=0)/total).tolist()}
    return result


def development_points(problem, seed, n_uniform, n_inlet):
    rng = np.random.default_rng(seed)
    uniform = problem.interior(rng, n_uniform)
    # The cylinder lies outside this inlet slab for the selected standard case.
    slab = problem.interior(rng, n_inlet)
    slab[:, 0] = rng.uniform(0, 1, n_inlet)
    if not problem.contains(slab).all():
        raise ValueError("Inlet development slab intersects geometry")
    return np.concatenate((uniform, slab))


def prepare(args):
    out = new_output(args.out_dir, "cylinder")
    start = time.perf_counter()
    config, problem, model, saved = load_checkpoint(args.checkpoint, "cpu")
    torch.set_num_threads(1)
    if (config.case != "cylinder" or config.transient or config.architecture != "hard_streamfunction"
            or config.hard_background != "cylinder_poiseuille"):
        raise ValueError("This protocol requires the valid steady hard-boundary cylinder checkpoint")
    if saved["progress"]["status"] != "training_finished":
        raise ValueError("Source baseline must have completed training")
    sources = [*sorted((ROOT / "pinn").glob("*.py")), Path(__file__).resolve(),
               ROOT / "scripts/pinn_sr_weight_search.py",
               ROOT / "scripts/audit_pinn_cylinder.py", ROOT / "scripts/run_hard_boundary_trial.py"]
    for source in sources:
        destination = out / "source" / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    protocol = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha(args.checkpoint),
                "source_progress": saved["progress"], "baseline_config": config.to_dict(),
                "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
                "seeds": SEEDS, "reference_data": None,
                "method": "Frozen symbolic spatial weighting of the original momentum residuals",
                "target": "log1p((r_x**2+r_y**2)/median_fit_residual_energy)",
                "input_features": {"x_D": "x", "y_D": "2*y/channel_height-1"},
                "fit_distribution": "4000 uniform fluid points plus 1500 uniform inlet-slab points",
                "selection_distribution": "6000 independent uniform fluid points plus 1500 inlet-slab points",
                "test_distribution": "20000 fresh uniform fluid points, used only after expression freeze",
                "success_criterion": "Both global momentum RMS components lower than ordinary; outlet no worse; exact velocity boundaries and flux retained",
                "gate": {"relative_selection_mse_below": .95, "spearman_above": .3},
                "budget": "Same per-arm wall-time cap including SR labeling/import/search/selection for SR; equal common audit excluded",
                "inference_limit": "Single checkpoint and continuation seed; no independent CFD accuracy or generalization claim"}
    write_json(out / "protocol.json", protocol)
    q_fit = development_points(problem, SEEDS["fit"], 4000, 1500)
    q_val = development_points(problem, SEEDS["selection"], 6000, 1500)
    r_fit = residual_values(model, q_fit, config.reynolds)
    r_val = residual_values(model, q_val, config.reynolds)
    e_fit, e_val = [(r[:, 1:]**2).sum(axis=1) for r in (r_fit, r_val)]
    scale = max(float(np.median(e_fit)), 1e-12)
    target_fit, target_val = np.log1p(e_fit/scale), np.log1p(e_val/scale)
    np.savez_compressed(out / "development.npz", fit_coords=q_fit, fit_residuals=r_fit,
                        selection_coords=q_val, selection_residuals=r_val,
                        target_fit=target_fit, target_selection=target_val)
    write_json(out / "baseline_development_diagnosis.json", {
        "kind": "Unweighted new uniform interior development sample; exclusive spatial regions",
        "points": 6000, "regions": region_diagnostics(q_val[:6000], r_val[:6000], config),
        "momentum_rms": np.sqrt((r_val[:6000, 1:]**2).mean(axis=0)).tolist()})
    print("Labeled baseline residuals; starting symbolic search", flush=True)
    # Julia runs in a separate interpreter so its runtime never shares a
    # process with Torch. Startup and compilation count toward the SR budget.
    subprocess.run([sys.executable, str(ROOT / "scripts/pinn_sr_weight_search.py"),
                    "--out-dir", str(out), "--iterations", str(args.iterations),
                    "--search-seconds", str(args.search_seconds)], check=True)
    constant_mse = float(np.mean((target_val-target_fit.mean())**2))
    candidates = []
    for candidate in read_json(out / "search_candidates.json"):
        try:
            pred = expression_values(candidate["expression"], q_val, config)
            candidate["mse"] = float(np.mean((pred-target_val)**2))
            candidate["relative_mse"] = candidate["mse"]/constant_mse
            correlation = float(spearmanr(pred, target_val).statistic) if np.std(pred) > 1e-12 else 0.
            candidate["spearman"] = correlation if np.isfinite(correlation) else 0.
            candidate["status"] = "finite"
        except (ValueError, RuntimeError, OverflowError) as exc:
            candidate["status"] = str(exc)
        candidates.append(candidate)
    valid = [c for c in candidates if c["status"] == "finite"]
    if not valid:
        raise RuntimeError("No finite symbolic candidates on expression-selection points")
    best = min(valid, key=lambda c: c["mse"])
    # Prefer the simplest candidate within 2% of the best held-out MSE.
    selected = min([c for c in valid if c["mse"] <= 1.02*best["mse"]], key=lambda c: c["complexity"])
    cap = float(np.quantile(target_fit, .99))
    fit_score = np.clip(expression_values(selected["expression"], q_fit, config), 0, cap)
    selected.update(score_cap=cap, score_mean=max(float(fit_score.mean()), 1e-12),
                    residual_energy_scale=scale, fit_target_mean=float(target_fit.mean()),
                    gate_passed=selected["relative_mse"] < .95 and selected["spearman"] > .3,
                    fit_total_seconds=time.perf_counter()-start,
                    weight_rule="normalize_batch(0.5 + 0.5*clip(score/score_mean,0,10)); score=clip(expression,0,score_cap)")
    write_json(out / "candidates.json", candidates)
    write_json(out / "selection.json", selected)
    print(json.dumps(selected, indent=2), flush=True)
    return out


class FrozenWeightTransform:
    def __init__(self, kind, config, selection):
        self.kind, self.config, self.selection = kind, config, selection
        self.batch_hashes, self.weight_stats = [], []
        self.evaluator = SympyTorch(selection["expression"]).double() if kind == "sr" else None

    def __call__(self, batch):
        points = batch["interior"]
        self.batch_hashes.append({key: array_sha(value.detach().cpu().numpy()) for key, value in batch.items()})
        if self.kind == "ordinary":
            return batch
        with torch.no_grad():
            if self.kind == "inlet":
                raw = 1 + 4*torch.exp(-points[:, :1].square())
            else:
                self.evaluator.to(device=points.device, dtype=points.dtype)
                q = torch.cat((points[:, :1], 2*points[:, 1:2]/self.config.channel_height-1), 1)
                score = self.evaluator(q).clamp(0, self.selection["score_cap"])
                raw = .5 + .5*(score/self.selection["score_mean"]).clamp(0, 10)
            weights = raw/raw.mean()
        self.weight_stats.append({"min": float(weights.min()), "max": float(weights.max()), "mean": float(weights.mean())})
        return {**batch, "interior_weights": weights.detach()}


def run(args):
    out = args.out_dir.resolve()
    protocol, selection = read_json(out / "protocol.json"), read_json(out / "selection.json")
    checkpoint = Path(protocol["checkpoint"])
    if sha(checkpoint) != protocol["checkpoint_sha256"]:
        raise ValueError("Baseline checkpoint changed after preparation")
    if args.kind == "sr" and not selection["gate_passed"]:
        raise ValueError("SR failed its held-out predictive gate; refusing to claim a useful SR constraint")
    setup_start = time.perf_counter()
    for relative, expected in protocol["source_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError("Experiment source changed after freeze: " + relative)
    config = Config.from_dict(protocol["baseline_config"])
    config.device, config.threads, config.seed = "cpu", 1, SEEDS["continuation"]
    # Same modest learning rate and optimizer schedule for all arms. Retain the
    # baseline collocation distribution, geometry, lifting and all loss weights.
    config.learning_rate = .0001
    config.adam_steps, config.lbfgs_steps = 100, 1000000
    config.log_every, config.checkpoint_every, config.eval_grid = 50, 100, 81
    config.validate()
    run_dir = new_output(out / args.kind, "cylinder")
    transform = FrozenWeightTransform(args.kind, config, selection)
    fit_cost = selection["fit_total_seconds"] if args.kind == "sr" else 0.
    runner_setup_seconds = time.perf_counter()-setup_start
    available = args.total_seconds - fit_cost - runner_setup_seconds
    if available <= 0:
        raise ValueError("SR preparation consumed the whole end-to-end time budget")
    arm_protocol = {"kind": args.kind, "checkpoint_sha256": protocol["checkpoint_sha256"],
                    "optimizer_reset": True, "total_seconds_cap": args.total_seconds,
                    "preparation_seconds_charged": fit_cost,
                    "runner_setup_seconds": runner_setup_seconds,
                    "training_seconds_cap": available,
                    "inlet_weight_rule": "normalize_batch(1+4*exp(-x**2))" if args.kind == "inlet" else None,
                    "time_limit_note": "Stop at an optimizer-step boundary; final serialization and audit are reported separately"}
    write_json(run_dir / "arm_protocol.json", arm_protocol)
    train_start = time.perf_counter()
    problem, model = train(config, run_dir, init_from=checkpoint,
                           batch_transform=transform, max_seconds=available)
    train_wall_seconds = time.perf_counter()-train_start
    write_json(run_dir / "sampling_trace.json", {"batches": transform.batch_hashes, "weights": transform.weight_stats})
    progress = read_json(run_dir / "training_status.json")
    arm_protocol.update(actual_training_seconds=progress["elapsed_seconds"],
                        train_call_wall_seconds=train_wall_seconds,
                        total_charged_seconds=fit_cost+runner_setup_seconds+train_wall_seconds)
    write_json(run_dir / "arm_protocol.json", arm_protocol)
    # Unweighted tests never call transform. The fitted score only affects training.
    evaluate(model, problem, run_dir / "evaluation", make_plots=False)
    dense = audit(run_dir, "cpu", seed=SEEDS["test"])
    write_json(run_dir / "split_flux.json", flux_diagnostics(model, problem))
    q_test = problem.interior(np.random.default_rng(SEEDS["test"]), 20000)
    r_test = residual_values(model, q_test, config.reynolds)
    write_json(run_dir / "regional_test.json", region_diagnostics(q_test, r_test, config))
    return dense


def summarize(args):
    out = args.out_dir.resolve()
    protocol = read_json(out / "protocol.json")
    runs = {}
    for kind in KINDS:
        folder = out / kind
        if not (folder / "regional_test.json").exists():
            continue
        dense = read_json(folder / f"dense_audit_seed_{SEEDS['test']}.json")
        runs[kind] = {"momentum_rms": [dense["pde_residuals"][k]["rms"] for k in ("momentum_x", "momentum_y")],
                      "outlet_traction_rms": dense["boundary_errors"]["outlet_traction"]["rms"],
                      "maximum_velocity_boundary_error": max(dense["boundary_errors"][k]["max_abs"] for k in (
                          "inlet_velocity_error", "bottom_velocity", "top_velocity", "cylinder_velocity")),
                      "flux_relative_error": read_json(folder / "split_flux.json")["max_relative_error"],
                      "progress": read_json(folder / "training_status.json"),
                      "budget": read_json(folder / "arm_protocol.json"),
                      "regions": read_json(folder / "regional_test.json"),
                      "forces_uncertified": dense["force_quadrature_refinement"][-1]}
    baseline_dir = out / "baseline_audit"
    baseline_dir.mkdir(exist_ok=True)
    if not (baseline_dir / "checkpoint_last.pt").exists():
        # Copy the checkpoint into the NEW output so auditing never writes into
        # the source baseline directory.
        (baseline_dir / "checkpoint_last.pt").write_bytes(Path(protocol["checkpoint"]).read_bytes())
    if not (baseline_dir / f"dense_audit_seed_{SEEDS['test']}.json").exists():
        audit(baseline_dir, "cpu", seed=SEEDS["test"])
    if not (baseline_dir / "regional_test.json").exists():
        config, problem, model, _ = load_checkpoint(baseline_dir / "checkpoint_last.pt", "cpu")
        torch.set_num_threads(1)
        q_test = problem.interior(np.random.default_rng(SEEDS["test"]), 20000)
        write_json(baseline_dir / "regional_test.json", region_diagnostics(q_test, residual_values(model, q_test, config.reynolds), config))
    baseline = read_json(baseline_dir / f"dense_audit_seed_{SEEDS['test']}.json")
    checks = {}
    if "ordinary" in runs and "sr" in runs:
        base, sr = runs["ordinary"], runs["sr"]
        checks = {"momentum_x_better": sr["momentum_rms"][0] < base["momentum_rms"][0],
                  "momentum_y_better": sr["momentum_rms"][1] < base["momentum_rms"][1],
                  "outlet_no_worse": sr["outlet_traction_rms"] <= base["outlet_traction_rms"],
                  "exact_velocity_boundaries": sr["maximum_velocity_boundary_error"] < 1e-10,
                  "flux_below_one_percent": sr["flux_relative_error"] < .01}
    traces = {k: read_json(out / k / "sampling_trace.json")["batches"] for k in runs}
    # Same optimizer phase/sample number must have identical coordinate hashes.
    sampling_match = all(traces[k][:min(len(traces[k]), len(traces["ordinary"]))]
                         == traces["ordinary"][:min(len(traces[k]), len(traces["ordinary"]))]
                         for k in traces) if "ordinary" in traces else False
    result = {"protocol": protocol, "selection": read_json(out / "selection.json"),
              "baseline_momentum_rms": [baseline["pde_residuals"][k]["rms"] for k in ("momentum_x", "momentum_y")],
              "baseline_outlet_traction_rms": baseline["boundary_errors"]["outlet_traction"]["rms"],
              "runs": runs, "sampling_sequence_prefix_matches": sampling_match,
              "checks": checks, "sr_passes_predeclared_criterion": bool(checks) and all(checks.values()),
              "all_three_arms_complete": set(runs) == set(KINDS)}
    if "inlet" in runs and "sr" in runs:
        result["sr_vs_simple_inlet"] = {
            "momentum_rms_ratio": (np.array(runs["sr"]["momentum_rms"])/runs["inlet"]["momentum_rms"]).tolist(),
            "outlet_traction_ratio": runs["sr"]["outlet_traction_rms"]/runs["inlet"]["outlet_traction_rms"]}
    write_json(out / "comparison.json", result)
    print(json.dumps({"runs": {k: {"rms": v["momentum_rms"], "outlet": v["outlet_traction_rms"]} for k, v in runs.items()},
                      "checks": checks, "sampling_match": sampling_match}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--iterations", type=int, default=80)
    p.add_argument("--search-seconds", type=float, default=60)
    p = sub.add_parser("run")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--kind", choices=KINDS, required=True)
    p.add_argument("--total-seconds", type=float, default=300)
    p = sub.add_parser("summarize")
    p.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    return {"prepare": prepare, "run": run, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
