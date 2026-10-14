"""Autograd residuals of dimensionless incompressible Navier--Stokes."""

import torch


def gradient(value, coordinates):
    """Pointwise derivative; retain differentiability for PDE-loss backprop.

    Networks must act independently on each row (no batch-dependent layers).
    The zero branch also supports constant/linear analytic verification fields.
    """
    if not value.requires_grad:
        return coordinates * 0
    result = torch.autograd.grad(value, coordinates, torch.ones_like(value),
                                 create_graph=True, retain_graph=True,
                                 allow_unused=True)[0]
    return coordinates * 0 if result is None else result + coordinates * 0


def navier_stokes(model, coordinates, reynolds):
    """Return continuity, x/y momentum columns; inputs are (x,y) or (x,y,t)."""
    q = coordinates.detach().clone().requires_grad_(True)
    values = model(q)
    u, v, p = values.split(1, dim=1)
    du, dv, dp = (gradient(f, q) for f in (u, v, p))
    lap_u = gradient(du[:, :1], q)[:, :1] + gradient(du[:, 1:2], q)[:, 1:2]
    lap_v = gradient(dv[:, :1], q)[:, :1] + gradient(dv[:, 1:2], q)[:, 1:2]
    u_t = du[:, 2:3] if q.shape[1] == 3 else 0
    v_t = dv[:, 2:3] if q.shape[1] == 3 else 0
    return {
        "continuity": du[:, :1] + dv[:, 1:2],
        "momentum_x": u_t + u * du[:, :1] + v * du[:, 1:2] + dp[:, :1] - lap_u / reynolds,
        "momentum_y": v_t + u * dv[:, :1] + v * dv[:, 1:2] + dp[:, 1:2] - lap_v / reynolds,
    }


def outlet_traction(model, coordinates, reynolds):
    """Zero Cauchy traction at a right outlet, outward normal n=(1,0)."""
    q = coordinates.detach().clone().requires_grad_(True)
    u, v, p = model(q).split(1, dim=1)
    du, dv = gradient(u, q), gradient(v, q)
    return torch.cat((-p + 2 * du[:, :1] / reynolds,
                      (du[:, 1:2] + dv[:, :1]) / reynolds), dim=1)
