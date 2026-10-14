"""Strict experiment configuration; no training takes place during import."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path


@dataclass
class Config:
    case: str = "kovasznay"
    reynolds: float = 40.0
    transient: bool = False
    t_end: float = 1.0
    # Cylinder geometry in diameter units (D = 1).
    channel_length: float = 22.0
    channel_height: float = 4.1
    cylinder_x: float = 2.0
    cylinder_y: float = 2.0
    ramp_time: float = 2.0
    hidden_layers: tuple[int, ...] = (64, 64, 64, 64)
    architecture: str = "uvp"
    hard_background: str = "cylinder_poiseuille"
    hard_inlet_length: float = 1.0
    hard_cylinder_length: float = 1.0
    hard_train_cylinder_constant: bool = True
    symbolic_expression: str = ""
    symbolic_coefficients: dict[str, float] = field(default_factory=dict)
    symbolic_coefficient_radius: float = 0.1
    symbolic_window: tuple[float, ...] = (3.0, 4.9, -1.9, 1.9)
    symbolic_taper: tuple[float, float] = (0.4, 0.4)
    symbolic_source: str = ""
    seed: int = 42
    dtype: str = "float64"
    device: str = "auto"
    threads: int = 4
    n_interior: int = 2048
    cylinder_refine_fraction: float = 0.0
    cylinder_refine_radius: float = 1.5
    inlet_refine_fraction: float = 0.0
    inlet_refine_width: float = 1.0
    n_boundary: int = 256  # per boundary segment / periodic pair
    n_initial: int = 512
    n_gauge: int = 128
    data_batch_size: int = 512
    adam_steps: int = 5000
    learning_rate: float = 0.001
    lbfgs_steps: int = 500  # optimizer steps; line searches may call closure repeatedly
    lbfgs_history: int = 50
    resample_every: int = 100
    log_every: int = 100
    checkpoint_every: int = 1000
    weight_pde: float = 1.0
    weight_boundary: float = 10.0
    weight_initial: float = 10.0
    weight_gauge: float = 1.0
    weight_data: float = 1.0
    weight_regularization: float = 0.0001
    eval_grid: int = 81
    eval_times: int = 3
    eval_batch_size: int = 1024

    def validate(self):
        if self.architecture not in {"uvp", "streamfunction", "hybrid_streamfunction", "hard_streamfunction"}:
            raise ValueError("Unknown architecture: " + str(self.architecture))
        if self.architecture == "hard_streamfunction":
            if self.case != "cylinder" or self.transient:
                raise ValueError("hard_streamfunction currently requires steady case=cylinder")
            if self.hard_background not in {"zero", "poiseuille", "cylinder_poiseuille"}:
                raise ValueError("Unknown hard_background: " + str(self.hard_background))
            for name in ("hard_inlet_length", "hard_cylinder_length"):
                if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                    raise ValueError(name + " must be finite and positive")
            if type(self.hard_train_cylinder_constant) is not bool:
                raise ValueError("hard_train_cylinder_constant must be boolean")
        if self.architecture == "hybrid_streamfunction":
            if self.case != "cylinder":
                raise ValueError("The wake hybrid currently requires case=cylinder")
            if not isinstance(self.symbolic_expression, str) or not self.symbolic_expression.strip():
                raise ValueError("hybrid_streamfunction requires symbolic_expression")
            if (not math.isfinite(self.symbolic_coefficient_radius)
                    or self.symbolic_coefficient_radius <= 0):
                raise ValueError("symbolic_coefficient_radius must be finite and positive")
            if (not isinstance(self.symbolic_coefficients, dict)
                    or any(not isinstance(k, str) or not isinstance(v, (float, int)) or not math.isfinite(v)
                           for k, v in self.symbolic_coefficients.items())):
                raise ValueError("symbolic_coefficients must map names to finite numbers")
            bounds, taper = self.symbolic_window, self.symbolic_taper
            if (len(bounds) != 4 or len(taper) != 2
                    or any(not math.isfinite(v) for v in (*bounds, *taper))
                    or any(taper[i] <= 0 or 2*taper[i] >= bounds[2*i+1]-bounds[2*i] for i in range(2))):
                raise ValueError("Invalid symbolic_window or symbolic_taper")
            if not (0 <= self.cylinder_x+bounds[0] < self.cylinder_x+bounds[1] <= self.channel_length
                    and 0 <= self.cylinder_y+bounds[2] < self.cylinder_y+bounds[3] <= self.channel_height):
                raise ValueError("The symbolic window must lie inside the channel")
        if self.case not in {"kovasznay", "taylor_green", "cylinder"}:
            raise ValueError(f"Unknown case: {self.case}")
        if (not math.isfinite(self.cylinder_refine_fraction)
                or not 0 <= self.cylinder_refine_fraction < 1):
            raise ValueError("cylinder_refine_fraction must be in [0,1)")
        if not math.isfinite(self.cylinder_refine_radius) or self.cylinder_refine_radius <= .5:
            raise ValueError("cylinder_refine_radius must exceed the cylinder radius .5")
        if (not math.isfinite(self.inlet_refine_fraction) or self.inlet_refine_fraction < 0
                or self.inlet_refine_fraction + self.cylinder_refine_fraction >= 1):
            raise ValueError("Refinement fractions must be nonnegative and sum to less than one")
        if not math.isfinite(self.inlet_refine_width) or not 0 < self.inlet_refine_width <= self.channel_length:
            raise ValueError("inlet_refine_width must lie in (0,channel_length]")
        if (self.cylinder_refine_fraction or self.inlet_refine_fraction) and self.case != "cylinder":
            raise ValueError("Cylinder refinement requires case=cylinder")
        if self.case == "kovasznay" and self.transient:
            raise ValueError("Kovasznay is a steady benchmark")
        if self.case == "taylor_green" and not self.transient:
            raise ValueError("Taylor--Green requires transient=true")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        for name in ("reynolds", "t_end", "ramp_time", "learning_rate",
                     "channel_length", "channel_height"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("threads", "n_interior", "n_boundary", "n_initial", "n_gauge",
                     "data_batch_size", "lbfgs_history", "resample_every", "log_every",
                     "checkpoint_every", "eval_grid", "eval_times", "eval_batch_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.eval_grid < 3 or (self.transient and self.eval_times < 2):
            raise ValueError("Evaluation needs >=3 spatial points and >=2 transient times")
        for name in ("adam_steps", "lbfgs_steps", "seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not self.hidden_layers or any(type(n) is not int or n < 1 for n in self.hidden_layers):
            raise ValueError("hidden_layers must contain positive integers")
        weights = [getattr(self, f"weight_{k}") for k in ("pde", "boundary", "initial", "gauge", "data", "regularization")]
        if any(not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError("Loss weights must be finite and nonnegative")
        if self.weight_pde == 0 or self.weight_boundary == 0:
            raise ValueError("A baseline requires positive PDE and boundary weights")
        if self.transient and self.weight_initial == 0:
            raise ValueError("Transient baselines require a positive initial weight")
        if self.case != "cylinder" and self.weight_gauge == 0:
            raise ValueError("Analytic benchmarks require a positive pressure gauge weight")
        if self.case == "cylinder":
            if not (0.5 < self.cylinder_x < self.channel_length - 0.5
                    and 0.5 < self.cylinder_y < self.channel_height - 0.5):
                raise ValueError("The radius-0.5 cylinder must lie strictly inside the channel")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        values = dict(values)
        for name in ("hidden_layers", "symbolic_window", "symbolic_taper"):
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values).validate()

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
