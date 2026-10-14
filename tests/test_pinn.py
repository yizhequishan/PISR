"""Physics identities, geometry/data contracts and a real optimizer/checkpoint cycle."""
from __future__ import annotations

import csv
import json
import math
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from pinn.config import Config
from pinn.data import load_observations
from pinn.evaluation import force_coefficients, predict
from pinn.model import FlowMLP
from pinn.physics import gradient, navier_stokes, outlet_traction
from pinn.problems import FlowProblem
from pinn.train import load_checkpoint, main
from scripts.prepare_pinn_comsol import convert


@pytest.mark.parametrize("case,transient,re", [("kovasznay", False, 40), ("taylor_green", True, 100)])
def test_analytic_ns_solutions(case, transient, re):
    problem = FlowProblem(Config(case=case, transient=transient, reynolds=re))
    coords = torch.tensor(problem.interior(np.random.default_rng(2), 71), dtype=torch.float64)
    residuals = navier_stokes(problem.exact, coords, re)
    for values in residuals.values():
        assert values.abs().max().item() < 2e-13


def test_poiseuille_pressure_and_diffusion_sign():
    h, re = 4.1, 20
    def exact(q):
        x, y = q[:, :1], q[:, 1:2]
        return torch.cat((6*y*(h-y)/h**2, y*0, -12*x/(re*h**2)), dim=1)
    q = torch.rand(43, 2, dtype=torch.float64)
    for residual in navier_stokes(exact, q, re).values():
        assert residual.abs().max().item() < 1e-14


def test_input_normalization_chain_rule_and_backward():
    torch.manual_seed(3)
    model = FlowMLP(torch.tensor([0., 0.], dtype=torch.float64),
                    torch.tensor([22., 4.1], dtype=torch.float64), (8, 8)).double()
    q = torch.tensor([[7., 2.]], dtype=torch.float64, requires_grad=True)
    derivative = gradient(model(q)[:, :1], q)[0, 0].item()
    step = torch.tensor([[1e-5, 0.]], dtype=torch.float64)
    numeric = ((model(q + step)[0, 0] - model(q - step)[0, 0]) / 2e-5).item()
    assert derivative == pytest.approx(numeric, rel=1e-7, abs=1e-9)
    sum(v.square().mean() for v in navier_stokes(model, q, 20).values()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("transient", [False, True])
def test_cylinder_sampling_and_startup(transient):
    problem = FlowProblem(Config(case="cylinder", transient=transient, n_interior=2000))
    a = problem.sample(np.random.default_rng(9), "cpu", torch.float64)
    b = problem.sample(np.random.default_rng(9), "cpu", torch.float64)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert problem.contains(a["interior"].numpy()).all()
    c = problem.config
    radius2 = (a["obstacle"][:, 0] - c.cylinder_x)**2 + (a["obstacle"][:, 1] - c.cylinder_y)**2
    assert torch.allclose(radius2, torch.full_like(radius2, 0.25))
    assert "gauge" not in a  # outlet traction fixes the pressure reference
    if transient:
        q = a["left"].clone()
        q[:, 2] = 0
        assert torch.count_nonzero(problem.inlet_velocity(q)) == 0
        assert torch.count_nonzero(a["initial"][:, 2]) == 0
    y = torch.linspace(0, c.channel_height, 10001, dtype=torch.float64)
    q = torch.column_stack((y*0, y))
    if transient:
        q = torch.column_stack((q, torch.full_like(y, 100*c.ramp_time)))
    integral = torch.trapz(problem.inlet_velocity(q)[:, 0], y) / c.channel_height
    assert integral.item() == pytest.approx(1, abs=2e-8)


def test_periodic_samples_pair_space_and_time():
    problem = FlowProblem(Config(case="taylor_green", transient=True))
    batch = problem.sample(np.random.default_rng(7), "cpu", torch.float64)
    assert torch.equal(batch["left"][:, 1:], batch["right"][:, 1:])
    assert torch.equal(batch["top"][:, [0, 2]], batch["bottom"][:, [0, 2]])


class AnalyticStress(torch.nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.kind = kind

    def forward(self, q):
        x, y = q[:, :1], q[:, 1:2]
        if self.kind == "pressure":
            return torch.cat((x*0, x*0, 2*x + 7), dim=1) + self.anchor*0
        return torch.cat((x*x + y*y, x*0, x*0), dim=1) + self.anchor*0


@pytest.mark.parametrize("kind,expected", [("pressure", -math.pi), ("viscous", 3*math.pi/20)])
def test_force_stress_sign_and_nondimensional_coefficient(kind, expected):
    problem = FlowProblem(Config(case="cylinder", reynolds=20))
    result = force_coefficients(AnalyticStress(kind), problem, [0], n_surface=128)[0]
    assert result["Cd"] == pytest.approx(expected, abs=1e-12)
    assert result["Cl"] == pytest.approx(0, abs=1e-12)


def test_outlet_traction_components():
    def linear(q):
        x, y = q[:, :1], q[:, 1:2]
        return torch.cat((3*x + 2*y, 5*x - 3*y, x*0 + 0.3), dim=1)
    traction = outlet_traction(linear, torch.rand(9, 2, dtype=torch.float64), 20)
    assert torch.allclose(traction[:, 0], torch.zeros(9, dtype=torch.float64), atol=1e-14)
    assert torch.allclose(traction[:, 1], torch.full((9,), 0.35, dtype=torch.float64))


def test_data_rejects_mean_flow_and_wrong_reynolds(tmp_path):
    problem = FlowProblem(Config(case="cylinder", reynolds=20))
    path = tmp_path / "observations.npz"
    data = dict(coords=np.array([[3., 2.]]), velocity=np.zeros((1, 2)), coordinate_system="dimensionless")
    np.savez(path, **data, field_kind="time_mean")
    with pytest.raises(ValueError, match="time-averaged"):
        load_observations(path, problem)
    np.savez(path, **data, field_kind="steady", reynolds=100)
    with pytest.raises(ValueError, match="Reynolds"):
        load_observations(path, problem)
    np.savez(path, **data, field_kind="steady", reynolds=20)
    assert load_observations(path, problem)[1]["rows"] == 1
    data["coords"] = np.array([[2., 2.]])
    np.savez(path, **data, field_kind="steady")
    with pytest.raises(ValueError, match="inside the cylinder"):
        load_observations(path, problem)


def test_comsol_conversion_uses_instantaneous_fields_and_units(tmp_path):
    source, output = tmp_path / "wide.txt", tmp_path / "samples.npz"
    header = "% x y " + " ".join(f"{field} @ t={t}" for t in (0, 2) for field in
                                     ("u (m/s)", "v (m/s)", "p (Pa)", "spf.U (m/s)", "x (m)", "y (m)"))
    source.write_text(header + "\n0.3 0.2 2 4 108 0 0.3 0.2 6 8 116 0 0.3 0.2\n", encoding="utf-8")
    result = convert(source, output, length_ref=.1, velocity_ref=2, density=2,
                     reynolds=100, pressure_offset=100)
    assert result["physical_times"] == [0, 2]
    with np.load(output) as data:
        np.testing.assert_allclose(data["coords"], [[3, 2, 0], [3, 2, 40]])
        np.testing.assert_allclose(data["velocity"], [[1, 2], [3, 4]])
        np.testing.assert_allclose(data["pressure"], [[1], [2]])
        assert data["field_kind"].item() == "instantaneous"
    with pytest.raises(FileExistsError):
        convert(source, output, .1, 2, 2, 100)


@pytest.mark.parametrize("transient", [False, True])
def test_real_training_evaluation_and_reload(tmp_path, transient):
    config = Config(case="cylinder", transient=transient, reynolds=20, hidden_layers=(8, 8),
                    n_interior=24, n_boundary=8, n_initial=16, adam_steps=10, lbfgs_steps=2,
                    log_every=5, eval_grid=9, eval_batch_size=64, device="cpu", threads=1)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    out = main(["--config", str(config_path), "--out-dir", str(tmp_path / "run"), "--no-plots"])
    _, problem, model, saved = load_checkpoint(out / "checkpoint_last.pt", "cpu")
    assert saved["progress"]["status"] == "training_finished"
    with (out / "history.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert float(rows[-1]["total"]) < float(rows[0]["total"])
    with np.load(out / "evaluation/fields.npz") as fields:
        np.testing.assert_allclose(predict(model, fields["coords"]), fields["uvp"], rtol=1e-12, atol=1e-12)
        assert problem.contains(fields["coords"]).all()
        assert np.isfinite(fields["residuals"]).all()
    metrics = json.loads((out / "evaluation/metrics.json").read_text())
    assert "boundary_outlet" in metrics["validation_terms_mse"]
    assert "final_force_coefficients" in metrics
    evaluation = main(["--evaluate", str(out / "checkpoint_last.pt"), "--device", "cpu",
                       "--out-dir", str(tmp_path / "eval"), "--no-plots"])
    new_metrics = json.loads((evaluation / "evaluation/metrics.json").read_text())
    assert metrics["validation_loss"] == pytest.approx(new_metrics["validation_loss"], rel=1e-12)
    with pytest.raises(ValueError, match="not empty"):
        main(["--config", str(config_path), "--out-dir", str(out), "--no-plots"])


@pytest.mark.parametrize("values", [{"reynolds": 0}, {"learning_rate": float("nan")},
                                     {"n_boundary": 0}, {"case": "taylor_green"},
                                     {"case": "cylinder", "cylinder_x": .1}, {"hidden_layers": []},
                                     {"adam_stpes": 1}, {"weight_boundary": 0}])
def test_invalid_configuration_fails_early(values):
    with pytest.raises(ValueError):
        Config.from_dict(values)