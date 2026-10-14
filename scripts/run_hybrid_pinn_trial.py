"""Matched-budget UVP / streamfunction / symbolic-hybrid cylinder trial."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn.config import Config
from pinn.evaluation import evaluate
from pinn.physics import gradient, navier_stokes
from pinn.train import new_output, train, write_json
from scripts.audit_pinn_cylinder import audit


def wake_diagnostics(model, config):
    parameter = next(model.parameters())
    bounds = config.symbolic_window
    x = np.linspace(bounds[0], bounds[1], 81)
    y = np.linspace(bounds[2], bounds[3], 97)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    local = np.column_stack((xx.ravel(), yy.ravel()))
    coordinates = local + [config.cylinder_x, config.cylinder_y]
    parts, symbolic_uv, neural_uv, total_uv = [], [], [], []
    for start in range(0, len(coordinates), 512):
        q = torch.tensor(coordinates[start:start+512], dtype=parameter.dtype,
                         device=parameter.device, requires_grad=True)
        parts.append(torch.cat(list(navier_stokes(model, q, config.reynolds).values()), dim=1).detach().cpu().numpy())
        if getattr(model, "symbolic", None) is not None:
            dpsi = gradient(model.symbolic_streamfunction(q), q)
            s = torch.cat((dpsi[:, 1:2], -dpsi[:, :1]), dim=1)
            uv = model(q)[:, :2]
            symbolic_uv.append(s.detach().cpu().numpy())
            total_uv.append(uv.detach().cpu().numpy())
            neural_uv.append((uv-s).detach().cpu().numpy())
    residuals = np.concatenate(parts)
    sx, sy = config.symbolic_taper
    plateau = ((local[:, 0] >= bounds[0]+sx) & (local[:, 0] <= bounds[1]-sx)
               & (local[:, 1] >= bounds[2]+sy) & (local[:, 1] <= bounds[3]-sy))
    result = {"grid_shape": [len(x), len(y)], "regions": {}}
    for name, mask in (("plateau", plateau), ("taper", ~plateau)):
        region = {"points": int(mask.sum()), "pde_rms": dict(zip(
            ("continuity", "momentum_x", "momentum_y"), np.sqrt(np.mean(residuals[mask]**2, axis=0)).tolist()))}
        if symbolic_uv:
            for label, chunks in (("symbolic", symbolic_uv), ("neural", neural_uv), ("total", total_uv)):
                region[label+"_velocity_rms"] = float(np.sqrt(np.mean(np.concatenate(chunks)[mask]**2)))
            a, b = np.concatenate(symbolic_uv)[mask], np.concatenate(neural_uv)[mask]
            denominator = float(np.linalg.norm(a)*np.linalg.norm(b))
            region["symbolic_neural_velocity_cosine"] = float(np.sum(a*b)/denominator) if denominator else None
        result["regions"][name] = region
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--adam-steps", type=int, default=1500)
    parser.add_argument("--lbfgs-steps", type=int, default=300)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--architectures", nargs="+", default=["uvp", "streamfunction", "hybrid_streamfunction"],
                        choices=["uvp", "streamfunction", "hybrid_streamfunction"])
    args = parser.parse_args()
    out = new_output(args.out_dir, "cylinder")
    base = Config.load(ROOT / "configs/pinn/cylinder_steady_hybrid.json")
    summary = {"kind": "single_seed_matched_update_budget_trial", "seed": base.seed,
               "reference_data": None, "runs": {},
               "note": "Equal optimizer updates and collocation counts, not equal time/FLOPs. No CFD accuracy claim; symbolic prior comes from unknown-Re time-mean data."}
    for architecture in args.architectures:
        values = base.to_dict()
        values.update(architecture=architecture, device=args.device, adam_steps=args.adam_steps,
                      lbfgs_steps=args.lbfgs_steps, eval_grid=121)
        if architecture != "hybrid_streamfunction":
            values.update(symbolic_expression="", symbolic_coefficients={}, symbolic_source="")
        config = Config.from_dict(values)
        run = new_output(out / architecture, "cylinder")
        print("\nArchitecture: " + architecture, flush=True)
        problem, model = train(config, run)
        metrics = evaluate(model, problem, run / "evaluation", make_plots=False)
        dense = audit(run, args.device)
        wake = wake_diagnostics(model, config)
        write_json(run / "wake_audit.json", wake)
        with (run / "history.csv").open(encoding="utf-8") as stream:
            history = list(csv.DictReader(stream))
        record = {"run_directory": str(run), "parameters": sum(p.numel() for p in model.parameters()),
                  "initial_loss": float(history[0]["total"]), "final_loss": float(history[-1]["total"]),
                  "validation_loss": metrics["validation_loss"], "pde_rms": {k: v["rms"] for k, v in dense["pde_residuals"].items()},
                  "max_flux_relative_error": dense["max_flux_relative_error"], "boundary_errors": dense["boundary_errors"],
                  "screening_passes": dense["screening_passes"], "progress": dense["checkpoint_progress"],
                  "wake_regions": wake["regions"]}
        if getattr(model, "symbolic", None) is not None:
            record["refined_coefficients"] = model.symbolic.coefficient_record()
        summary["runs"][architecture] = record
        write_json(out / "comparison.json", summary)
        print(json.dumps(record, indent=2, allow_nan=False), flush=True)
    return out


if __name__ == "__main__":
    main()
