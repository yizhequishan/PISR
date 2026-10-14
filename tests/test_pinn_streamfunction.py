"""Hard divergence, higher derivatives, symbolic refinement, and real training."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sp = pytest.importorskip("sympy")

from pinn.config import Config
from pinn.evaluation import predict
from pinn.losses import loss_terms, weighted_loss
from pinn.model import StreamFunctionMLP, SympyTorch, WakeWindow
from pinn.physics import gradient, navier_stokes
from pinn.train import load_checkpoint, main, setup

ROOT = Path(__file__).resolve().parents[1]


def hybrid_config(**overrides):
    values = Config.load(ROOT / "configs/pinn/cylinder_steady_hybrid.json").to_dict()
    values.update(hidden_layers=(8, 8), device="cpu", threads=1, n_interior=24,
                  n_boundary=8, n_initial=16, eval_grid=9, eval_batch_size=64)
    values.update(overrides)
    return Config.from_dict(values)


def test_compiler_matches_sympy_through_third_order_and_coefficient_gradient():
    module = SympyTorch("a*sin(x_D*y_D) + cos(y_D)/x_D + y_D**3", {"a": 0.7}).double()
    q = torch.tensor([[3.7, -0.8], [4.1, 0.5]], dtype=torch.float64, requires_grad=True)
    x, y, a = sp.symbols("x_D y_D a")
    expr = a*sp.sin(x*y) + sp.cos(y)/x + y**3
    value = module(q)
    derivative_expr = expr
    for axis in (None, 0, 1, 0):
        if axis is not None:
            value = gradient(value, q)[:, axis:axis+1]
            derivative_expr = sp.diff(derivative_expr, (x, y)[axis])
        expected = sp.lambdify((x, y, a), derivative_expr, "numpy")(q.detach().numpy()[:, 0], q.detach().numpy()[:, 1], .7)
        np.testing.assert_allclose(value.detach().numpy().ravel(), expected, rtol=1e-12, atol=1e-12)
    grad = torch.autograd.grad(value.sum(), module.raw_coefficients)[0].item()
    eps = 1e-5
    def third_sum():
        f = module(q)
        for axis in (0, 1, 0):
            f = gradient(f, q)[:, axis:axis+1]
        return f.sum().item()
    with torch.no_grad():
        module.raw_coefficients[0] = eps
    plus = third_sum()
    with torch.no_grad():
        module.raw_coefficients[0] = -eps
    minus = third_sum()
    assert grad == pytest.approx((plus-minus)/(2*eps), rel=1e-7, abs=1e-9)


@pytest.mark.parametrize("expression", ["abs(y_D)", "z + x_D", "__import__('os')", "x_D**a"])
def test_compiler_rejects_unsupported_operations(expression):
    with pytest.raises(ValueError):
        SympyTorch(expression, {"a": .3} if expression == "x_D**a" else None)


def test_compact_gate_value_and_first_three_derivatives_at_edges():
    gate = WakeWindow().double()
    q = torch.tensor([[3., .2], [4.9, .2], [4., -1.9], [4., 1.9]], dtype=torch.float64, requires_grad=True)
    f = gate(q)
    assert torch.count_nonzero(f) == 0
    for axis in (0, 1):
        derivative = f
        for order in range(3):
            derivative = gradient(derivative, q)[:, axis:axis+1]
            assert derivative.abs().max().item() < 1e-10
    center = torch.tensor([[4., 0.]], dtype=torch.float64)
    assert gate(center).item() == pytest.approx(1, abs=1e-13)


@pytest.mark.parametrize("transient", [False, True])
def test_hybrid_divergence_and_momentum_backward_including_taper(transient):
    config = hybrid_config(transient=transient)
    problem, model, device, dtype = setup(config)
    q = torch.tensor([[5., 2.], [5.15, 2.3], [5.7, .3], [6.8, 3.75], [10., 1.]], dtype=dtype)
    if transient:
        q = torch.column_stack((q, torch.linspace(.1, .9, len(q), dtype=dtype)))
    residuals = navier_stokes(model, q, 20)
    assert residuals["continuity"].abs().max().item() < 1e-10
    sum(v.square().mean() for k, v in residuals.items() if k != "continuity").backward()
    coeff_grad = model.symbolic.raw_coefficients.grad
    assert torch.isfinite(coeff_grad).all() and torch.count_nonzero(coeff_grad) == 5
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    expected = model(q).detach()
    with torch.no_grad():
        result = model(q)
        assert not result.requires_grad
        torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)
    with torch.inference_mode():
        torch.testing.assert_close(model(q.clone()), expected, rtol=1e-12, atol=1e-12)
    assert np.isfinite(predict(model, q.numpy())).all()


def test_no_extrapolation_at_external_pole_and_exact_neural_fallback():
    bounds = [torch.tensor(v, dtype=torch.float64) for v in ([0., 0.], [22., 4.1])]
    model = StreamFunctionMLP(*bounds, (8,), symbolic_expression="y_D/(x_D-1)").double()
    q = torch.tensor([[1., 0.], [2., 1.], [10., 2.]], dtype=torch.float64, requires_grad=True)
    symbolic = model.symbolic_streamfunction(q)
    assert torch.count_nonzero(symbolic) == 0
    uvp = model(q)
    neural = model.network(2*(q-model.lower)/(model.upper-model.lower)-1)
    dpsi = gradient(neural[:, :1], q)
    torch.testing.assert_close(uvp, torch.cat((dpsi[:, 1:2], -dpsi[:, :1], neural[:, 1:2]), dim=1))
    sum(v.square().mean() for v in navier_stokes(model, q, 20).values()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_loss_removes_only_hard_continuity_and_regularization_reaches_network():
    config = hybrid_config()
    problem, model, device, dtype = setup(config)
    batch = problem.sample(np.random.default_rng(1), device, dtype)
    observations = {"coords": batch["interior"][:3], "velocity": torch.zeros(3, 2, dtype=dtype)}
    terms = loss_terms(model, problem, batch, observations)
    assert "pde_continuity" not in terms
    assert "regularization_residual" in terms and "data_velocity" in terms
    terms["regularization_residual"].backward()
    assert model.network[0].weight.grad.abs().sum().item() > 0
    expected = sum(getattr(config, "weight_"+k.split("_", 1)[0])*v for k, v in terms.items())
    torch.testing.assert_close(weighted_loss(terms, config), expected)
    config.architecture = "uvp"
    _, baseline, _, _ = setup(config)
    baseline_terms = loss_terms(baseline, problem, batch)
    assert "pde_continuity" in baseline_terms and "regularization_residual" not in baseline_terms


def test_initial_expression_matches_saved_sr_and_bounded_refinement_has_no_pole():
    config = hybrid_config()
    _, model, _, dtype = setup(config)
    with (ROOT / "outputs_mgn/comsol/eta_sym_psi/results_strategy_b.csv").open(encoding="utf-8") as stream:
        record = next(r for r in csv.DictReader(stream) if r["equation_index"] == "14")
    x, y = sp.symbols("x_D y_D")
    expression = sp.sympify(record["psi_sympy"])
    q = torch.tensor([[3., -1.8], [4., -.1], [4.9, 1.8]], dtype=dtype, requires_grad=True)
    expected = sp.lambdify((x, y), expression, "numpy")(*q.detach().numpy().T)
    np.testing.assert_allclose(model.symbolic(q).detach().numpy().ravel(), expected, rtol=1e-12, atol=1e-12)
    # Certified worst-case bound for this particular cancelled rational form.
    coef, radius = config.symbolic_coefficients, config.symbolic_coefficient_radius
    upper = lambda name: coef[name] + radius*max(1, abs(coef[name]))
    lower = lambda name: coef[name] - radius*max(1, abs(coef[name]))
    assert config.symbolic_window[0] - upper("b") - upper("d")*(1-lower("a")) > .4


@pytest.mark.parametrize("architecture", ["streamfunction", "hybrid_streamfunction"])
def test_training_and_checkpoint_round_trip(tmp_path, architecture):
    config = hybrid_config(architecture=architecture, adam_steps=4, lbfgs_steps=2, log_every=2)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    out = main(["--config", str(path), "--out-dir", str(tmp_path / "run"), "--no-plots"])
    loaded, _, model, saved = load_checkpoint(out / "checkpoint_last.pt", "cpu")
    assert loaded.architecture == architecture and saved["progress"]["status"] == "training_finished"
    with np.load(out / "evaluation/fields.npz") as fields:
        np.testing.assert_allclose(predict(model, fields["coords"]), fields["uvp"], rtol=1e-12, atol=1e-12)
        assert np.isfinite(fields["residuals"]).all()
        assert abs(fields["residuals"][:, 0]).max() < 1e-9
    if architecture == "hybrid_streamfunction":
        assert not torch.equal(model.symbolic.raw_coefficients, torch.zeros_like(model.symbolic.raw_coefficients))
        assert (out / "symbolic_coefficients.json").exists()
    evaluated = main(["--evaluate", str(out / "checkpoint_last.pt"), "--device", "cpu",
                      "--out-dir", str(tmp_path / "reloaded"), "--no-plots"])
    original = json.loads((out / "evaluation/metrics.json").read_text())["validation_loss"]
    reloaded = json.loads((evaluated / "evaluation/metrics.json").read_text())["validation_loss"]
    assert reloaded == pytest.approx(original, rel=1e-12)
