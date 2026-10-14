"""Field, equation, boundary and force diagnostics for trained checkpoints."""

import csv
import json
import math
from pathlib import Path
import numpy as np
import torch
from .losses import loss_terms, weighted_loss
from .physics import gradient, navier_stokes


def predict(model, coordinates, batch_size=1024):
    parameter = next(model.parameters())
    chunks = []
    with torch.no_grad():
        for start in range(0, len(coordinates), batch_size):
            q = torch.as_tensor(coordinates[start:start+batch_size], dtype=parameter.dtype, device=parameter.device)
            chunks.append(model(q).cpu().numpy())
    return np.concatenate(chunks)


def force_coefficients(model, problem, times, n_surface=512):
    """Cd, Cl from fluid Cauchy stress acting on the body; Uref=mean inlet speed.

    Dimensionless D=1 and force per unit span: C = 2 integral(sigma*n_body ds).
    Quadrature is midpoint periodic with no duplicated angular endpoint.
    """
    parameter = next(model.parameters())
    theta = (np.arange(n_surface) + 0.5) * 2 * math.pi / n_surface
    nx, ny = np.cos(theta), np.sin(theta)
    c = problem.config
    xy = np.column_stack((c.cylinder_x + 0.5 * nx, c.cylinder_y + 0.5 * ny))
    rows = []
    for time in times:
        coords = np.column_stack((xy, np.full(n_surface, time))) if c.transient else xy
        q = torch.tensor(coords, device=parameter.device, dtype=parameter.dtype, requires_grad=True)
        u, v, p = model(q).split(1, dim=1)
        du, dv = gradient(u, q), gradient(v, q)
        normal_x = q.new_tensor(nx[:, None])
        normal_y = q.new_tensor(ny[:, None])
        shear = (du[:, 1:2] + dv[:, :1]) / c.reynolds
        pressure_x, pressure_y = -p * normal_x, -p * normal_y
        viscous_x = 2 * du[:, :1] / c.reynolds * normal_x + shear * normal_y
        viscous_y = shear * normal_x + 2 * dv[:, 1:2] / c.reynolds * normal_y
        factor = 2 * math.pi / n_surface  # 2 (coefficient) * radius .5 * dtheta
        cd_p, cl_p, cd_v, cl_v = [float(f.detach().sum().cpu()) * factor
                                  for f in (pressure_x, pressure_y, viscous_x, viscous_y)]
        rows.append({"time": float(time), "Cd": cd_p + cd_v, "Cl": cl_p + cl_v,
                     "Cd_pressure": cd_p, "Cd_viscous": cd_v,
                     "Cl_pressure": cl_p, "Cl_viscous": cl_v})
    return rows


def error_metrics(predicted, reference):
    result = {}
    for name, columns in (("velocity", slice(0, 2)), ("u", slice(0, 1)),
                          ("v", slice(1, 2)), ("pressure", slice(2, 3))):
        if reference.shape[1] < columns.stop:
            continue
        target, error = reference[:, columns], predicted[:, columns] - reference[:, columns]
        norm = float(np.sqrt(np.sum(target**2)))
        result[name + "_rmse"] = float(np.sqrt(np.mean(error**2)))
        result[name + "_relative_l2"] = float(np.sqrt(np.sum(error**2)) / norm) if norm > 1e-12 else None
    return result


def render_plots(out):
    # Keep Torch and matplotlib's numerical libraries in separate processes.
    import os
    import subprocess
    import sys
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", str(Path(out) / ".matplotlib_cache"))
    script = Path(__file__).resolve().parents[1] / "scripts" / "plot_pinn.py"
    subprocess.run([sys.executable, str(script), "--evaluation-dir", str(out)],
                   check=True, env=environment)


def evaluate(model, problem, out_dir, reference=None, make_plots=True):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    c, parameter = problem.config, next(model.parameters())
    model.eval()
    coords = problem.evaluation_grid()
    values = predict(model, coords, c.eval_batch_size)
    residual_chunks, vorticity_chunks = [], []
    for start in range(0, len(coords), c.eval_batch_size):
        q = torch.tensor(coords[start:start+c.eval_batch_size], dtype=parameter.dtype, device=parameter.device, requires_grad=True)
        residual = navier_stokes(model, q, c.reynolds)
        residual_chunks.append(torch.cat(list(residual.values()), dim=1).detach().cpu().numpy())
        uvp = model(q)
        omega = gradient(uvp[:, 1:2], q)[:, :1] - gradient(uvp[:, :1], q)[:, 1:2]
        vorticity_chunks.append(omega.detach().cpu().numpy())
    residuals, vorticity = np.concatenate(residual_chunks), np.concatenate(vorticity_chunks)
    np.savez_compressed(out / "fields.npz", coords=coords, uvp=values, vorticity=vorticity,
                        residuals=residuals, residual_names=np.array(["continuity", "momentum_x", "momentum_y"]),
                        field_kind="instantaneous" if c.transient else "steady", coordinate_system="dimensionless")
    validation = problem.sample(np.random.default_rng(c.seed + 1_000_003), parameter.device, parameter.dtype)
    terms = loss_terms(model, problem, validation)
    metrics = {"case": c.case, "reynolds": c.reynolds, "grid_points": len(coords),
               "validation_seed": c.seed + 1_000_003,
               "validation_loss": float(weighted_loss(terms, c).detach().cpu()),
               "validation_terms_mse": {k: float(v.detach().cpu()) for k, v in terms.items()},
               "grid_residual_rms": {name: float(np.sqrt(np.mean(residuals[:, i]**2)))
                                     for i, name in enumerate(("continuity", "momentum_x", "momentum_y"))}}
    if c.case != "cylinder":
        q = torch.as_tensor(coords, device=parameter.device, dtype=parameter.dtype)
        metrics["analytic_error"] = error_metrics(values, problem.exact(q).detach().cpu().numpy())
    if reference is not None:
        predicted = predict(model, reference["coords"], c.eval_batch_size)
        truth = np.column_stack((reference["velocity"], reference["pressure"])) if "pressure" in reference else reference["velocity"]
        metrics["reference_error"] = error_metrics(predicted, truth)
        metrics["reference_note"] = "Reference independence must be established by the data split; pressure uses the supplied gauge."
    if c.case == "cylinder":
        times = np.linspace(0, c.t_end, max(21, c.eval_times)) if c.transient else [0.0]
        forces = force_coefficients(model, problem, times)
        with (out / "forces.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(forces[0]))
            writer.writeheader()
            writer.writerows(forces)
        metrics["final_force_coefficients"] = forces[-1]
        metrics["force_note"] = "Unvalidated PINN stress integrals; assess wall derivatives and compare with independent CFD before using Cd/Cl."
        flux = []
        y = np.linspace(0, c.channel_height, 513)
        for time in times:
            flow_rates = []
            for x in (0, c.channel_length):
                q = np.column_stack((np.full_like(y, x), y))
                if c.transient:
                    q = np.column_stack((q, np.full_like(y, time)))
                flow_rates.append(float((np.trapezoid if hasattr(np, "trapezoid") else np.trapz)(predict(model, q, c.eval_batch_size)[:, 0], y)))
            flux.append({"time": float(time), "inlet_flux": flow_rates[0], "outlet_flux": flow_rates[1],
                         "outlet_minus_inlet": flow_rates[1] - flow_rates[0]})
        metrics["flux_balance"] = flux
        if reference is None:
            metrics["accuracy_status"] = "No reference field: residuals and boundary losses do not establish flow accuracy."
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    (out / "plot_config.json").write_text(json.dumps(c.to_dict(), indent=2), encoding="utf-8")
    if make_plots:
        render_plots(out)
    return metrics