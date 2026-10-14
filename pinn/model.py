"""Pointwise velocity and divergence-free streamfunction architectures."""

import ast
import math
import torch
from torch import nn


class FlowMLP(nn.Module):
    """Pointwise tanh MLP: (x,y[,t]) -> (u,v,p), with input scaling in graph.

    p is kinematic dimensionless pressure, p_dim / (rho U_ref**2).
    All derivatives are taken against the original dimensionless coordinates.
    """

    def __init__(self, lower, upper, hidden_layers=(64, 64, 64, 64)):
        super().__init__()
        self.register_buffer("lower", torch.as_tensor(lower).clone())
        self.register_buffer("upper", torch.as_tensor(upper).clone())
        widths = [len(lower), *hidden_layers, 3]
        layers = []
        for i, (n_in, n_out) in enumerate(zip(widths[:-1], widths[1:])):
            layer = nn.Linear(n_in, n_out)
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            if i < len(widths) - 2:
                layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(self, coordinates):
        scaled = 2 * (coordinates - self.lower) / (self.upper - self.lower) - 1
        return self.network(scaled)


class SympyTorch(nn.Module):
    """Compile a smooth SymPy expression to differentiable Torch operations.

    Explicit coefficient symbols are trainable; numeric literals (including
    powers) stay fixed. Coefficients have a bounded refinement interval
    ``initial +/- radius * max(1, abs(initial))``. This is a trust region, not
    a general guarantee against singular expressions. No NumPy evaluation or
    detached coordinate derivatives are used. String parsing never uses eval.
    """

    def __init__(self, expression, coefficients=None, coefficient_radius=0.1):
        super().__init__()
        import sympy as sp

        coefficients = dict(coefficients or {})
        self.coefficient_names = tuple(sorted(coefficients))
        variables = ("x_D", "y_D", "t")
        if any(not name.isidentifier() or name in variables or name in ("pi", "E")
               for name in coefficients):
            raise ValueError("Invalid symbolic coefficient name")
        if not math.isfinite(coefficient_radius) or coefficient_radius <= 0:
            raise ValueError("coefficient_radius must be finite and positive")
        initial = [float(coefficients[name]) for name in self.coefficient_names]
        if any(not math.isfinite(value) for value in initial):
            raise ValueError("Symbolic coefficients must be finite")
        self.register_buffer("initial_coefficients", torch.tensor(initial, dtype=torch.float64))
        self.register_buffer("coefficient_span", coefficient_radius * self.initial_coefficients.abs().clamp_min(1))
        self.raw_coefficients = nn.Parameter(torch.zeros(len(initial), dtype=torch.float64))
        functions = {name: getattr(sp, name) for name in ("sin", "cos", "tanh", "exp", "log", "sqrt", "sinh", "cosh", "atan")}
        symbols = {name: sp.Symbol(name) for name in (*variables, *self.coefficient_names)}

        def parse(node):
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                if not math.isfinite(node.value):
                    raise ValueError("Non-finite symbolic constant")
                return sp.sympify(node.value)
            if isinstance(node, ast.Name) and node.id in symbols:
                return symbols[node.id]
            if isinstance(node, ast.Name) and node.id in ("pi", "E"):
                return getattr(sp, node.id)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                return parse(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
            if isinstance(node, ast.BinOp):
                a, b = parse(node.left), parse(node.right)
                operations = {ast.Add: lambda: a+b, ast.Sub: lambda: a-b,
                              ast.Mult: lambda: a*b, ast.Div: lambda: a/b,
                              ast.Pow: lambda: a**b}
                if type(node.op) in operations:
                    return operations[type(node.op)]()
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in functions and len(node.args) == 1 and not node.keywords):
                return functions[node.func.id](parse(node.args[0]))
            raise ValueError("Unsupported symbolic syntax: " + ast.dump(node))

        try:
            expr = parse(ast.parse(expression, mode="eval").body) if isinstance(expression, str) else expression
        except SyntaxError as error:
            raise ValueError("Invalid symbolic expression") from error
        if not isinstance(expr, sp.Expr) or expr.free_symbols - set(symbols.values()):
            raise ValueError("Unsupported symbols in streamfunction")
        unused = set(self.coefficient_names) - {str(s) for s in expr.free_symbols}
        if unused:
            raise ValueError("Unused symbolic coefficients: " + str(sorted(unused)))
        self.expression = str(expr)
        torch_functions = {getattr(sp, name): getattr(torch, name) for name in functions if name != "sqrt"}

        def compile_node(node):
            if node.is_Number or node in (sp.pi, sp.E):
                number = float(node)
                if not math.isfinite(number):
                    raise ValueError("Non-finite symbolic constant")
                return lambda q, c: q.new_tensor(number)
            if node in symbols.values():
                name = str(node)
                if name in variables:
                    index = variables.index(name)
                    return lambda q, c: q[:, index:index+1]
                index = self.coefficient_names.index(name)
                return lambda q, c: c[index]
            children = [compile_node(arg) for arg in node.args]
            if node.func == sp.Add:
                return lambda q, c: sum(child(q, c) for child in children)
            if node.func == sp.Mul:
                def product(q, c):
                    result = children[0](q, c)
                    for child in children[1:]:
                        result = result * child(q, c)
                    return result
                return product
            if node.func == sp.Pow:
                if not node.args[1].is_Number:
                    raise ValueError("Only fixed numeric powers are supported")
                # Python integer exponents remain real on negative bases.
                exponent = int(node.args[1]) if node.args[1].is_Integer else float(node.args[1])
                return lambda q, c: children[0](q, c).pow(exponent)
            if node.func in torch_functions:
                function = torch_functions[node.func]
                return lambda q, c: function(children[0](q, c))
            raise ValueError("Unsupported or nonsmooth symbolic operation: " + str(node.func))

        self._evaluate = compile_node(expr)
        self.uses_time = symbols["t"] in expr.free_symbols

    @property
    def coefficients(self):
        return self.initial_coefficients + self.coefficient_span * self.raw_coefficients.tanh()

    def forward(self, local_coordinates):
        if self.uses_time and local_coordinates.shape[1] != 3:
            raise ValueError("A time-dependent expression requires (x_D,y_D,t)")
        return self._evaluate(local_coordinates, self.coefficients) + local_coordinates[:, :1] * 0

    def coefficient_record(self):
        return dict(zip(self.coefficient_names, self.coefficients.detach().cpu().tolist()))


class WakeWindow(nn.Module):
    """Compact C3 gate; the entire taper stays inside the trusted SR window.

    The first three derivatives vanish at both ends of the seventh-degree
    smoothstep. Third spatial derivatives are needed by the momentum loss.
    """

    def __init__(self, bounds=(3.0, 4.9, -1.9, 1.9), taper=(0.4, 0.4)):
        super().__init__()
        if (len(bounds) != 4 or len(taper) != 2
                or any(not math.isfinite(v) for v in (*bounds, *taper))
                or any(taper[i] <= 0 or 2*taper[i] >= bounds[2*i+1]-bounds[2*i] for i in range(2))):
            raise ValueError("Invalid wake bounds/taper; a nonempty plateau is required")
        self.register_buffer("bounds", torch.tensor(bounds, dtype=torch.float64))
        self.register_buffer("taper", torch.tensor(taper, dtype=torch.float64))

    @staticmethod
    def smoothstep(t):
        t = t.clamp(0, 1)
        return t.pow(4) * (35 + t * (-84 + t * (70 - 20*t)))

    def forward(self, local_coordinates):
        x, y = local_coordinates[:, :1], local_coordinates[:, 1:2]
        a, b, c, d = self.bounds.unbind()
        sx, sy = self.taper.unbind()
        # Tapers cannot overlap. Only the nearer edge can have a factor
        # below one, so this equals the four-edge product with half as many
        # high-order derivative graphs. The min switches on the flat plateau.
        return (self.smoothstep(torch.minimum(x-a, b-x)/sx)
                * self.smoothstep(torch.minimum(y-c, d-y)/sy))


class StreamFunctionMLP(FlowMLP):
    """(x,y[,t]) -> (psi_y,-psi_x,p), optionally psi = g*psi_SR + N.

    Outside the trusted window g and its first three derivatives are zero,
    leaving the neural streamfunction. No blending of velocity fields occurs.
    Evaluation under no_grad/inference_mode still enables the one derivative
    needed for velocity, then returns detached values to existing consumers.
    """

    divergence_free = True

    def __init__(self, lower, upper, hidden_layers=(64, 64, 64, 64), *,
                 symbolic_expression=None, symbolic_coefficients=None,
                 symbolic_coefficient_radius=0.1, symbolic_origin=(0.0, 0.0),
                 symbolic_window=(3.0, 4.9, -1.9, 1.9), symbolic_taper=(0.4, 0.4)):
        super().__init__(lower, upper, hidden_layers)
        # Keep the hidden architecture; the final outputs now mean (N, P).
        head = nn.Linear(hidden_layers[-1], 2)
        nn.init.xavier_normal_(head.weight)
        nn.init.zeros_(head.bias)
        self.network[-1] = head
        self.register_buffer("symbolic_origin", torch.tensor(symbolic_origin, dtype=torch.float64))
        self.symbolic = None
        self.window = None
        if symbolic_expression is not None:
            self.symbolic = SympyTorch(symbolic_expression, symbolic_coefficients, symbolic_coefficient_radius)
            self.window = WakeWindow(symbolic_window, symbolic_taper)

    def local_coordinates(self, coordinates):
        xy = coordinates[:, :2] - self.symbolic_origin
        return torch.cat((xy, coordinates[:, 2:]), dim=1)

    def symbolic_streamfunction(self, coordinates):
        if self.symbolic is None:
            return coordinates[:, :1] * 0
        local = self.local_coordinates(coordinates)
        bounds = self.window.bounds
        # Only compile an autograd graph for supported points. Apart from
        # avoiding 0*NaN at extrapolated poles, this skips expensive symbolic
        # third derivatives at every distant boundary point. Index/scatter
        # stays differentiable; the C3 gate makes support edges smooth.
        inside = ((local[:, 0] > bounds[0]) & (local[:, 0] < bounds[1])
                  & (local[:, 1] > bounds[2]) & (local[:, 1] < bounds[3]))
        indices = inside.nonzero(as_tuple=True)[0]
        if indices.numel() == 0:
            return coordinates[:, :1] * 0
        active = local[indices]
        gate = self.window(active)
        anchor_xy = torch.stack(((bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2)).expand(len(active), 2)
        anchor = torch.cat((anchor_xy, active[:, 2:]), dim=1)
        # A constant streamfunction gauge becomes physical after gating;
        # remove it before applying the window (also at each transient time).
        values = gate * (self.symbolic(active) - self.symbolic(anchor))
        return (coordinates[:, :1] * 0).index_copy(0, indices, values)

    def streamfunction_pressure(self, coordinates):
        neural, pressure = super().forward(coordinates).split(1, dim=1)
        return neural + self.symbolic_streamfunction(coordinates), pressure

    def forward(self, coordinates):
        differentiable = torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
        with torch.inference_mode(False), torch.enable_grad():
            q = coordinates if differentiable and coordinates.requires_grad else coordinates.detach().clone().requires_grad_(True)
            psi, pressure = self.streamfunction_pressure(q)
            dpsi = torch.autograd.grad(psi, q, torch.ones_like(psi),
                                      create_graph=differentiable, retain_graph=differentiable)[0]
            result = torch.cat((dpsi[:, 1:2], -dpsi[:, :1], pressure), dim=1)
        return result if differentiable else result.detach()

    def residual_penalty(self, coordinates):
        """L2 of the neural psi residual on a fixed grid in the SR plateau.

        Penalizing outside the window would suppress the whole flow there.
        A dedicated grid avoids a mostly-empty penalty with global sampling.
        Spatial coordinates are deterministic; transient times follow the
        current batch so L-BFGS still sees a fixed objective.
        """
        if self.symbolic is None:
            return coordinates.sum() * 0
        a, b, c, d = self.window.bounds.unbind()
        sx, sy = self.window.taper.unbind()
        unit = torch.linspace(0, 1, 12, device=coordinates.device, dtype=coordinates.dtype)
        xx, yy = torch.meshgrid(a+sx+(b-a-2*sx)*unit, c+sy+(d-c-2*sy)*unit, indexing="ij")
        xy = torch.stack((xx.flatten(), yy.flatten()), dim=1) + self.symbolic_origin
        if coordinates.shape[1] == 3:
            indices = torch.arange(len(xy), device=coordinates.device) % len(coordinates)
            xy = torch.cat((xy, coordinates[indices, 2:3]), dim=1)
        return super().forward(xy)[:, :1].square().mean()


class HardBoundaryStreamFunctionMLP(StreamFunctionMLP):
    """Steady cylinder lifting: psi = psi_bg(C) + D*N, u=psi_y, v=-psi_x.

    A vanishes quadratically on inlet/channel walls and B on the cylinder.
    w=A/(A+B), D=A*B/(A+B). Thus D and grad(D) vanish on all velocity
    boundaries; w=1, grad(w)=0 on the cylinder and w=grad(w)=0 on the
    inlet/walls. The denominator is strictly positive on the fluid closure
    because the cylinder does not touch the channel boundaries. These are
    smooth defining functions, not the nonsmooth nearest-boundary distance.

    The corrected Poiseuille lifting is P(y)+w*(C-P(y)). The free constant
    C is a homogeneous velocity mode: no-slip does not prescribe the split
    of flux above/below an internal obstacle. Fixing C is supported explicitly.
    Zero and raw Poiseuille backgrounds are deliberately incompatible negative
    controls, not solutions of the prescribed cylinder-flow boundary problem.
    """

    def __init__(self, lower, upper, hidden_layers=(64, 64, 64, 64), *,
                 cylinder_origin=(2.0, 2.0), background="cylinder_poiseuille",
                 inlet_length=1.0, cylinder_length=1.0, train_cylinder_constant=True):
        super().__init__(lower, upper, hidden_layers)
        if background not in {"zero", "poiseuille", "cylinder_poiseuille"}:
            raise ValueError("Unknown hard background")
        self.background = background
        self.register_buffer("cylinder_origin", torch.tensor(cylinder_origin, dtype=torch.float64))
        self.register_buffer("inlet_length", torch.tensor(inlet_length, dtype=torch.float64))
        self.register_buffer("cylinder_length", torch.tensor(cylinder_length, dtype=torch.float64))
        center_value = self.poiseuille(self.cylinder_origin[1])
        if background == "cylinder_poiseuille" and train_cylinder_constant:
            self.cylinder_constant = nn.Parameter(center_value.clone())
        else:
            self.register_buffer("cylinder_constant", center_value.clone())

    def poiseuille(self, y):
        h = self.upper[1]
        s = y / h
        return h * s.square() * (3 - 2*s)

    def boundary_factors(self, coordinates):
        x, y = coordinates[:, :1], coordinates[:, 1:2]
        s = y / self.upper[1]
        a = x.square() / (x.square() + self.inlet_length.square()) * (4*s*(1-s)).square()
        # (r^2-R^2)/(2R), R=.5: a smooth first-order defining distance.
        radius2 = (coordinates[:, :2] - self.cylinder_origin).square().sum(dim=1, keepdim=True)
        b = ((radius2 - .25) / self.cylinder_length).square()
        return a*b/(a+b), a/(a+b)

    def background_streamfunction(self, coordinates, cylinder_weight=None):
        if self.background == "zero":
            return coordinates[:, :1] * 0
        base = self.poiseuille(coordinates[:, 1:2])
        if self.background == "poiseuille":
            return base
        if cylinder_weight is None:
            _, cylinder_weight = self.boundary_factors(coordinates)
        return base + cylinder_weight * (self.cylinder_constant - base)

    def streamfunction_pressure(self, coordinates):
        neural, pressure = FlowMLP.forward(self, coordinates).split(1, dim=1)
        distance, weight = self.boundary_factors(coordinates)
        return self.background_streamfunction(coordinates, weight) + distance*neural, pressure


def build_model(config, lower, upper):
    """Single construction path for training and checkpoint reloading."""
    if config.architecture == "uvp":
        return FlowMLP(lower, upper, config.hidden_layers)
    if config.architecture == "hard_streamfunction":
        return HardBoundaryStreamFunctionMLP(
            lower, upper, config.hidden_layers, cylinder_origin=(config.cylinder_x, config.cylinder_y),
            background=config.hard_background, inlet_length=config.hard_inlet_length,
            cylinder_length=config.hard_cylinder_length,
            train_cylinder_constant=config.hard_train_cylinder_constant)
    options = {}
    if config.architecture == "hybrid_streamfunction":
        options = dict(symbolic_expression=config.symbolic_expression,
                       symbolic_coefficients=config.symbolic_coefficients,
                       symbolic_coefficient_radius=config.symbolic_coefficient_radius,
                       symbolic_origin=(config.cylinder_x, config.cylinder_y),
                       symbolic_window=config.symbolic_window, symbolic_taper=config.symbolic_taper)
    return StreamFunctionMLP(lower, upper, config.hidden_layers, **options)
