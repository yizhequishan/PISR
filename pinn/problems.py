"""Problem definitions and seeded sampling, independent of the network."""

import math
import numpy as np
import torch


class FlowProblem:
    def __init__(self, config):
        self.config = config.validate()
        if config.case == "kovasznay":
            self.lower, self.upper = [-0.5, -0.5], [1.0, 1.5]
        elif config.case == "taylor_green":
            self.lower, self.upper = [-math.pi, -math.pi], [math.pi, math.pi]
        else:
            self.lower, self.upper = [0.0, 0.0], [config.channel_length, config.channel_height]
        if config.transient:
            self.lower += [0.0]
            self.upper += [config.t_end]
        self.lower, self.upper = np.array(self.lower), np.array(self.upper)

    def contains(self, q, tolerance=1e-9):
        valid = np.all((q >= self.lower - tolerance) & (q <= self.upper + tolerance), axis=1)
        if self.config.case == "cylinder":
            radius2 = (q[:, 0] - self.config.cylinder_x)**2 + (q[:, 1] - self.config.cylinder_y)**2
            valid &= radius2 >= 0.25 - tolerance
        return valid

    def uniform(self, rng, n):
        return rng.uniform(self.lower, self.upper, (n, len(self.lower)))

    def interior(self, rng, n):
        chunks, remaining = [], n
        while remaining:
            q = self.uniform(rng, max(remaining * 2, 32))
            accepted = q[self.contains(q)][:remaining]
            chunks.append(accepted)
            remaining -= len(accepted)
        return np.concatenate(chunks)

    def exact(self, q):
        """Analytic fields for operator verification; cylinder has no exact field."""
        x, y = q[:, :1], q[:, 1:2]
        re = self.config.reynolds
        if self.config.case == "kovasznay":
            lam = -4 * math.pi**2 / (math.sqrt(re**2 / 4 + 4 * math.pi**2) + re / 2)
            e = torch.exp(lam * x)
            return torch.cat((1 - e * torch.cos(2 * math.pi * y),
                              lam / (2 * math.pi) * e * torch.sin(2 * math.pi * y),
                              (1 - e**2) / 2), dim=1)
        if self.config.case == "taylor_green":
            decay = torch.exp(-2 * q[:, 2:3] / re)
            return torch.cat((-torch.cos(x) * torch.sin(y) * decay,
                              torch.sin(x) * torch.cos(y) * decay,
                              -(torch.cos(2*x) + torch.cos(2*y)) * decay**2 / 4), dim=1)
        raise ValueError("Cylinder flow needs an independent reference solution")

    def collocation(self, rng, n):
        """Optional training-only body/inlet refinement; interior() stays uniform.

        A radius-uniform annulus resolves near-body derivative peaks. The
        inlet slab can independently resolve entrance derivatives. Remaining
        points keep whole-domain coverage. The unweighted PDE loss
        therefore uses a deliberately biased training distribution; the dense
        audit calls interior() directly and never inherits this refinement.
        """
        c = self.config
        focused = int(n*c.cylinder_refine_fraction)
        inlet_count = int(n*c.inlet_refine_fraction)
        if focused + inlet_count == 0:
            return self.interior(rng, n)
        chunks = [self.interior(rng, n-focused-inlet_count)]
        remaining = focused
        while remaining:
            count = max(2*remaining, 32)
            q = self.uniform(rng, count)
            radius = rng.uniform(.5, c.cylinder_refine_radius, count)
            theta = rng.uniform(0, 2*math.pi, count)
            q[:, 0] = c.cylinder_x + radius*np.cos(theta)
            q[:, 1] = c.cylinder_y + radius*np.sin(theta)
            accepted = q[self.contains(q)][:remaining]
            chunks.append(accepted)
            remaining -= len(accepted)
        remaining = inlet_count
        while remaining:
            count = max(2*remaining, 32)
            q = self.uniform(rng, count)
            q[:, 0] = rng.uniform(0, c.inlet_refine_width, count)
            accepted = q[self.contains(q)][:remaining]
            chunks.append(accepted)
            remaining -= len(accepted)
        return np.concatenate(chunks)

    def inlet_velocity(self, q):
        h = self.config.channel_height
        u = 6 * q[:, 1:2] * (h - q[:, 1:2]) / h**2
        if self.config.transient:
            # Compatible with a fluid initially at rest, including the t=0 inlet.
            u = u * (1 - torch.exp(-(q[:, 2:3] / self.config.ramp_time)**2))
        return torch.cat((u, torch.zeros_like(u)), dim=1)

    def sample(self, rng, device, dtype):
        """Each boundary segment has equal sample count; do not sample solid points."""
        c = self.config
        batch = {"interior": self.collocation(rng, c.n_interior)}
        for name, axis, value in (("left", 0, self.lower[0]), ("right", 0, self.upper[0]),
                                  ("bottom", 1, self.lower[1]), ("top", 1, self.upper[1])):
            q = self.uniform(rng, c.n_boundary)
            q[:, axis] = value
            batch[name] = q
        if c.case == "taylor_green":
            batch["right"] = batch["left"].copy()
            batch["right"][:, 0] = self.upper[0]
            batch["top"] = batch["bottom"].copy()
            batch["top"][:, 1] = self.upper[1]
        if c.case == "cylinder":
            q = self.uniform(rng, c.n_boundary)
            theta = rng.uniform(0, 2 * math.pi, c.n_boundary)
            q[:, 0] = c.cylinder_x + 0.5 * np.cos(theta)
            q[:, 1] = c.cylinder_y + 0.5 * np.sin(theta)
            batch["obstacle"] = q
        else:
            gauge = self.uniform(rng, c.n_gauge)
            gauge[:, :2] = self.lower[:2]
            batch["gauge"] = gauge
        if c.transient:
            initial = self.interior(rng, c.n_initial)
            initial[:, 2] = 0
            batch["initial"] = initial
        return {key: torch.as_tensor(value, dtype=dtype, device=device) for key, value in batch.items()}

    def evaluation_grid(self):
        c = self.config
        x, y = np.linspace(self.lower[0], self.upper[0], c.eval_grid), np.linspace(self.lower[1], self.upper[1], c.eval_grid)
        xx, yy = np.meshgrid(x, y, indexing="xy")
        xy = np.column_stack((xx.ravel(), yy.ravel()))
        times = np.linspace(0, c.t_end, c.eval_times) if c.transient else [None]
        points = [xy if t is None else np.column_stack((xy, np.full(len(xy), t))) for t in times]
        return np.concatenate([q[self.contains(q)] for q in points])
