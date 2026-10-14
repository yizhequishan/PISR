"""Explicit instantaneous/steady nondimensional NPZ data contract."""

import hashlib
from pathlib import Path
import numpy as np
import torch


def load_observations(path, problem):
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as archive:
        required = {"coords", "velocity", "field_kind", "coordinate_system"}
        if not required.issubset(archive.files):
            raise ValueError(f"{path}: required NPZ keys are {sorted(required)}")
        expected = "instantaneous" if problem.config.transient else "steady"
        if str(archive["field_kind"].item()) != expected:
            raise ValueError(f"Expected field_kind={expected}; time-averaged flow is not a steady NS solution")
        if str(archive["coordinate_system"].item()) != "dimensionless":
            raise ValueError("Convert x/L, y/L, t*U/L, velocity/U and pressure/(rho*U^2) first")
        if "reynolds" in archive and not np.isclose(float(archive["reynolds"].item()), problem.config.reynolds, rtol=1e-8, atol=0):
            raise ValueError("Observation Reynolds number differs from the PINN configuration")
        arrays = {k: np.asarray(archive[k], dtype=np.float64) for k in ("coords", "velocity", "pressure") if k in archive}
    n = len(arrays["coords"])
    dimensions = {"coords": len(problem.lower), "velocity": 2, "pressure": 1}
    for key, value in arrays.items():
        if key == "pressure" and value.shape == (n,):
            value = arrays[key] = value[:, None]
        if value.shape != (n, dimensions[key]) or n == 0 or not np.isfinite(value).all():
            raise ValueError(f"Invalid {key} shape/values in {path}; expected ({n}, {dimensions[key]})")
    if not problem.contains(arrays["coords"]).all():
        raise ValueError(f"Observation coordinates lie outside the space/time domain or inside the cylinder: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return arrays, {"path": str(path), "sha256": digest.hexdigest(), "rows": n, "field_kind": expected}


def observation_batch(arrays, rng, size, device, dtype):
    if arrays is None:
        return None
    indices = rng.choice(len(arrays["coords"]), min(size, len(arrays["coords"])), replace=False)
    return {key: torch.as_tensor(value[indices], device=device, dtype=dtype) for key, value in arrays.items()}