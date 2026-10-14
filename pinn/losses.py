"""Named loss terms: extension point for weighting and additional physics."""

import torch
from .physics import gradient, navier_stokes, outlet_traction


def mse(value):
    return value.square().mean()


def loss_terms(model, problem, batch, observations=None):
    c = problem.config
    weights = batch.get("interior_weights")
    if weights is not None:
        if not isinstance(weights, torch.Tensor):
            raise ValueError("interior_weights must be a Torch tensor")
        if weights.shape != (len(batch["interior"]), 1):
            raise ValueError("interior_weights must have shape (N, 1)")
        if weights.requires_grad:
            raise ValueError("interior_weights must be detached")
        if not torch.isfinite(weights).all() or not (weights > 0).all():
            raise ValueError("interior_weights must be finite and positive")
    terms = {}
    for key, value in navier_stokes(model, batch["interior"], c.reynolds).items():
        if key == "continuity" and getattr(model, "divergence_free", False):
            continue
        terms["pde_" + key] = ((weights * value.square()).mean()
                              if weights is not None and key in ("momentum_x", "momentum_y")
                              else mse(value))
    if getattr(model, "symbolic", None) is not None:
        terms["regularization_residual"] = model.residual_penalty(batch["interior"])
    if c.case == "cylinder":
        terms["boundary_inlet"] = mse(model(batch["left"])[:, :2] - problem.inlet_velocity(batch["left"]))
        for name in ("bottom", "top", "obstacle"):
            terms["boundary_" + name] = mse(model(batch[name])[:, :2])
        terms["boundary_outlet"] = mse(outlet_traction(model, batch["right"], c.reynolds))
    elif c.case == "kovasznay":
        for name in ("left", "right", "bottom", "top"):
            terms["boundary_" + name] = mse(model(batch[name])[:, :2] - problem.exact(batch[name])[:, :2])
    else:
        # Periodic velocity/pressure values and normal derivatives of velocity.
        for a, b, axis in (("left", "right", 0), ("bottom", "top", 1)):
            qa, qb = (batch[key].detach().clone().requires_grad_(True) for key in (a, b))
            fa, fb = model(qa), model(qb)
            terms["boundary_periodic_" + a] = mse(fa - fb)
            for component in range(2):
                da = gradient(fa[:, component:component+1], qa)[:, axis:axis+1]
                db = gradient(fb[:, component:component+1], qb)[:, axis:axis+1]
                terms[f"boundary_derivative_{a}_{component}"] = mse(da - db)
    if "gauge" in batch:
        q = batch["gauge"]
        terms["gauge_pressure"] = mse(model(q)[:, 2:3] - problem.exact(q)[:, 2:3])
    if "initial" in batch:
        q = batch["initial"]
        target = torch.zeros_like(q[:, :2]) if c.case == "cylinder" else problem.exact(q)[:, :2]
        terms["initial_velocity"] = mse(model(q)[:, :2] - target)
    if observations is not None:
        predicted = model(observations["coords"])
        terms["data_velocity"] = mse(predicted[:, :2] - observations["velocity"])
        if "pressure" in observations:
            terms["data_pressure"] = mse(predicted[:, 2:3] - observations["pressure"])
    return terms


def weighted_loss(terms, config):
    return sum(getattr(config, "weight_" + name.split("_", 1)[0]) * value
               for name, value in terms.items())
