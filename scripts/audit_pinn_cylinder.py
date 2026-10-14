"""Independent dense diagnostics for a completed steady cylinder PINN run."""
import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pinn.train import load_checkpoint
from pinn.evaluation import predict, force_coefficients
from pinn.physics import navier_stokes, outlet_traction


def summarize(values):
    return {"rms": float(np.sqrt(np.mean(values**2))),
            "max_abs": float(np.max(np.abs(values))),
            "p99_abs": float(np.quantile(np.abs(values), .99))}


def audit(run_dir, device="cpu", *, seed=20260923):
    run = Path(run_dir).resolve()
    config, problem, model, saved = load_checkpoint(run / "checkpoint_last.pt", device)
    if config.case != "cylinder" or config.transient:
        raise ValueError("This audit is for steady cylinder flow")
    model.eval()
    parameter = next(model.parameters())
    rng = np.random.default_rng(seed)
    interior = problem.interior(rng, 20000)
    residual_parts = []
    for start in range(0, len(interior), 512):
        q = torch.as_tensor(interior[start:start+512], dtype=parameter.dtype, device=parameter.device)
        residual_parts.append(torch.cat(list(navier_stokes(model, q, config.reynolds).values()), dim=1).detach().cpu().numpy())
    residual = np.concatenate(residual_parts)
    y = np.linspace(0, config.channel_height, 4097)
    prescribed_flux = config.channel_height
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    sections = []
    for x in [0., 1., 1.5, 2., 2.5, 3., 4., 6., 10., 16., config.channel_length]:
        if not 0 <= x <= config.channel_length:
            continue
        q = np.column_stack((np.full_like(y, x), y))
        valid = problem.contains(q)
        u = np.zeros_like(y)
        u[valid] = predict(model, q[valid])[:, 0]
        flow = float(trapz(u, y))
        sections.append({"x": x, "flux": flow, "relative_error_vs_prescribed_inlet": (flow - prescribed_flux)/prescribed_flux})
    inlet = np.column_stack((y*0, y))
    inlet_q = torch.as_tensor(inlet, device=parameter.device, dtype=parameter.dtype)
    inlet_error = predict(model, inlet)[:, :2] - problem.inlet_velocity(inlet_q).detach().cpu().numpy()
    x = np.linspace(0, config.channel_length, 4097)
    bottom, top = np.column_stack((x, x*0)), np.column_stack((x, x*0+config.channel_height))
    theta = (np.arange(4096)+.5)*2*math.pi/4096
    obstacle = np.column_stack((config.cylinder_x+.5*np.cos(theta), config.cylinder_y+.5*np.sin(theta)))
    obstacle_uv = predict(model, obstacle)[:, :2]
    bottom_uv, top_uv = predict(model, bottom)[:, :2], predict(model, top)[:, :2]
    outlet = np.column_stack((y*0+config.channel_length, y))
    traction = []
    for start in range(0, len(outlet), 512):
        q = torch.as_tensor(outlet[start:start+512], device=parameter.device, dtype=parameter.dtype)
        traction.append(outlet_traction(model, q, config.reynolds).detach().cpu().numpy())
    traction = np.concatenate(traction)
    pressure_points = np.array([[config.cylinder_x-.5, config.cylinder_y],
                                [config.cylinder_x+.5, config.cylinder_y]])
    pressures = predict(model, pressure_points)[:, 2]
    force_refinement = []
    for n in (256, 512, 1024, 2048):
        force = force_coefficients(model, problem, [0], n_surface=n)[0]
        force_refinement.append({"surface_points": n, **force})
    wall_leakage = float(trapz(top_uv[:, 1], x) - trapz(bottom_uv[:, 1], x))
    body_leakage = float(np.sum(-obstacle_uv[:, 0]*np.cos(theta)-obstacle_uv[:, 1]*np.sin(theta))*math.pi/len(theta))
    thresholds = {"maximum_cross_section_flux_relative_error": .01,
                  "noslip_velocity_component_rms": .01, "pde_component_rms": .01,
                  "inlet_velocity_component_rms": .01, "outlet_traction_component_rms": .01}
    pde_stats = {name: summarize(residual[:, i]) for i, name in enumerate(("continuity", "momentum_x", "momentum_y"))}
    boundaries = {"inlet_velocity_error": summarize(inlet_error), "bottom_velocity": summarize(bottom_uv),
                  "top_velocity": summarize(top_uv), "cylinder_velocity": summarize(obstacle_uv),
                  "outlet_traction": summarize(traction)}
    max_flux_error = max(abs(row["relative_error_vs_prescribed_inlet"]) for row in sections)
    checks = {"flux": max_flux_error < thresholds["maximum_cross_section_flux_relative_error"],
              "pde": all(s["rms"] < thresholds["pde_component_rms"] for s in pde_stats.values()),
              "noslip": all(boundaries[k]["rms"] < thresholds["noslip_velocity_component_rms"] for k in ("bottom_velocity", "top_velocity", "cylinder_velocity")),
              "inlet": boundaries["inlet_velocity_error"]["rms"] < thresholds["inlet_velocity_component_rms"],
              "outlet": boundaries["outlet_traction"]["rms"] < thresholds["outlet_traction_component_rms"]}
    result = {"checkpoint_progress": saved["progress"], "independent_sampling_seed": seed,
              "interior_points": len(interior), "pde_residuals": pde_stats, "boundary_errors": boundaries,
              "prescribed_inlet_flux": prescribed_flux, "cross_section_fluxes": sections,
              "max_flux_relative_error": max_flux_error,
              "wall_normal_leakage": wall_leakage, "cylinder_normal_leakage": body_leakage,
              "net_boundary_outflow": sections[-1]["flux"]-sections[0]["flux"]+wall_leakage+body_leakage,
              "pressure_drop_front_minus_back": float(pressures[0]-pressures[1]),
              "force_quadrature_refinement": force_refinement,
              "screening_thresholds": thresholds, "screening_passes": checks,
              "all_screening_passed": all(checks.values()),
              "interpretation": "Necessary numerical screening only, not an independent CFD accuracy certificate. Force quadrature convergence does not establish wall-gradient accuracy."}
    out = run / ("dense_audit.json" if seed == 20260923 else f"dense_audit_seed_{seed}.json")
    out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"run": str(run), "max_flux_relative_error": max_flux_error,
                      "pde_rms": {k:v["rms"] for k,v in pde_stats.items()},
                      "passes": checks, "Cd": force_refinement[-1]["Cd"], "Cl": force_refinement[-1]["Cl"]}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    audit(args.run_dir, args.device, seed=args.seed)
