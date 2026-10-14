"""Analytic boundary identities, obstacle flux, third derivatives and reload."""
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinn.config import Config
from pinn.evaluation import predict
from pinn.physics import gradient, navier_stokes
from pinn.train import load_checkpoint, main, new_output, setup, train
from pinn.problems import FlowProblem
from scripts.run_hard_boundary_trial import assess, flux_diagnostics, run_variant


def hard_config(**overrides):
    values = dict(case="cylinder", reynolds=20., architecture="hard_streamfunction",
                  hidden_layers=(8, 8), device="cpu", threads=1, n_interior=24,
                  n_boundary=16, eval_grid=9, eval_batch_size=64)
    values.update(overrides)
    return Config.from_dict(values)


@pytest.mark.parametrize("train_constant", [False, True])
def test_exact_velocity_boundaries_and_second_order_distance(train_constant):
    config = hard_config(hard_train_cylinder_constant=train_constant)
    problem, model, _, dtype = setup(config)
    # Non-default free constant must preserve every prescribed velocity.
    with torch.no_grad():
        model.cylinder_constant.add_(.41)
        for parameter in model.network.parameters():
            parameter.mul_(1.7)
    batch = problem.sample(np.random.default_rng(17), "cpu", dtype)
    batch["corners"] = torch.tensor([[0., 0.], [0., config.channel_height],
                                    [config.channel_length, 0.],
                                    [config.channel_length, config.channel_height]], dtype=dtype)
    for name in ("left", "bottom", "top", "obstacle", "corners"):
        q = batch[name].clone().requires_grad_(True)
        distance, _ = model.boundary_factors(q)
        assert distance.abs().max() < 1e-27
        assert gradient(distance, q).abs().max() < 2e-13
        target = problem.inlet_velocity(q) if name == "left" else torch.zeros_like(q)
        torch.testing.assert_close(model(q)[:, :2], target, rtol=0, atol=3e-13)
        psi, _ = model.streamfunction_pressure(q)
        if name == "bottom":
            torch.testing.assert_close(psi, psi*0, rtol=0, atol=1e-13)
        elif name == "top":
            torch.testing.assert_close(psi, psi*0+config.channel_height, rtol=0, atol=1e-13)
        elif name == "obstacle":
            torch.testing.assert_close(psi, psi*0+model.cylinder_constant, rtol=0, atol=1e-13)
    diagnostic = flux_diagnostics(model, problem, orders=(64, 128))
    assert diagnostic["max_relative_error"] < 1e-11
    assert diagnostic["max_quadrature_endpoint_difference"] < 1e-11


@pytest.mark.parametrize("background", ["zero", "poiseuille"])
def test_negative_controls_cannot_learn_missing_boundary_values(background):
    problem, model, _, dtype = setup(hard_config(hard_background=background))
    batch = problem.sample(np.random.default_rng(18), "cpu", dtype)
    for name in ("left", "obstacle"):
        expected = (torch.zeros_like(batch[name]) if background == "zero"
                    else problem.inlet_velocity(batch[name]))
        torch.testing.assert_close(model(batch[name])[:, :2], expected, rtol=0, atol=1e-13)
    diagnostic = flux_diagnostics(model, problem, orders=(128,))
    if background == "zero":
        assert diagnostic["max_relative_error"] == pytest.approx(1., abs=1e-12)
    else:
        assert diagnostic["max_relative_error"] > .3
        assert model(batch["obstacle"])[:, 0].min().item() > 1.


def test_hard_divergence_high_derivatives_and_homogeneous_mode_gradient():
    _, model, _, dtype = setup(hard_config())
    q = torch.tensor([[1.48, 2.01], [2.05, 2.53], [0., 1.], [8., 3.2]], dtype=dtype)
    residuals = navier_stokes(model, q, 20.)
    assert residuals["continuity"].abs().max().item() < 1e-12
    loss = sum(r.square().mean() for r in residuals.values())
    loss.backward()
    derivative = model.cylinder_constant.grad.item()
    assert np.isfinite(derivative) and abs(derivative) > 1e-6
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.network[0].weight.grad.abs().sum().item() > 0
    center = model.cylinder_constant.item()
    values = []
    for shift in (1e-5, -1e-5):
        with torch.no_grad():
            model.cylinder_constant.fill_(center+shift)
        values.append(sum(r.square().mean() for r in navier_stokes(model, q, 20.).values()).item())
    assert derivative == pytest.approx((values[0]-values[1])/2e-5, rel=1e-7, abs=1e-8)
    with torch.no_grad():
        model.cylinder_constant.fill_(center)
    expected = model(q).detach()
    for context in (torch.no_grad(), torch.inference_mode()):
        with context:
            actual = model(q.clone())
            assert not actual.requires_grad
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_matched_neural_initialization():
    _, soft, _, _ = setup(hard_config(architecture="streamfunction"))
    for background in ("zero", "poiseuille", "cylinder_poiseuille"):
        _, hard, _, _ = setup(hard_config(hard_background=background))
        for key, value in soft.network.state_dict().items():
            assert torch.equal(value, hard.network.state_dict()[key])


@pytest.mark.parametrize("background", ["zero", "poiseuille", "cylinder_poiseuille"])
def test_hard_training_checkpoint_round_trip(tmp_path, background):
    config = hard_config(hard_background=background, adam_steps=3, lbfgs_steps=2, log_every=2)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    out = main(["--config", str(path), "--out-dir", str(tmp_path / "run"), "--no-plots"])
    loaded, problem, model, saved = load_checkpoint(out / "checkpoint_last.pt", "cpu")
    assert loaded.hard_background == background
    assert saved["progress"]["status"] == "training_finished"
    with np.load(out / "evaluation/fields.npz") as fields:
        np.testing.assert_allclose(predict(model, fields["coords"]), fields["uvp"], rtol=1e-12, atol=1e-12)
    if background == "cylinder_poiseuille":
        assert flux_diagnostics(model, problem, orders=(128,))["max_relative_error"] < 1e-11
    # A changed distance factor cannot silently reuse old weights.
    config.hard_cylinder_length = 1.5
    path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
    with pytest.raises(ValueError, match="hard_cylinder_length"):
        main(["--config", str(path), "--out-dir", str(tmp_path / "mismatch"),
              "--init-from", str(out / "checkpoint_last.pt"), "--no-plots"])


@pytest.mark.parametrize("overrides", [dict(transient=True), dict(case="kovasznay"),
    dict(hard_background="sr"), dict(hard_cylinder_length=0), dict(hard_inlet_length=float("nan")),
    dict(hard_train_cylinder_constant=1)])
def test_invalid_hard_config(overrides):
    with pytest.raises(ValueError):
        hard_config(**overrides)


def test_acceptance_requires_each_momentum_component_and_both_flux_checks():
    boundaries = {key: {"max_abs": 1e-14} for key in
        ("inlet_velocity_error", "bottom_velocity", "top_velocity", "cylinder_velocity")}
    runs = {"soft": {"pde_rms": {"momentum_x": .2, "momentum_y": .1}},
            "cylinder_poiseuille": {"pde_rms": {"momentum_x": .1, "momentum_y": .05},
                "max_flux_relative_error": 1e-6, "split_flux": {"max_relative_error": 1e-12},
                "boundary_errors": boundaries}}
    assert assess(runs)["success"]
    runs["cylinder_poiseuille"]["pde_rms"]["momentum_y"] = .11
    assert not assess(runs)["success"]
    assert assess({"soft": runs["soft"]})["success"] is None


def test_cpu_trial_processes_keep_runs_and_configs_separate(tmp_path):
    config = hard_config(adam_steps=2, lbfgs_steps=1, log_every=2)
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {name: executor.submit(run_variant, config.to_dict(), name, tmp_path)
                   for name in ("zero", "cylinder_poiseuille")}
        records = {name: future.result(timeout=120) for name, future in futures.items()}
    for name, record in records.items():
        assert record["progress"]["status"] == "training_finished"
        saved = Config.load(tmp_path / name / "config.json")
        assert saved.hard_background == name and saved.device == "cpu"
    assert records["zero"]["split_flux"]["max_relative_error"] == pytest.approx(1., abs=1e-11)
    assert records["cylinder_poiseuille"]["split_flux"]["max_relative_error"] < 1e-11


def test_refinement_only_changes_collocation_not_uniform_audit():
    plain = FlowProblem(hard_config(n_interior=2048))
    focused = FlowProblem(hard_config(n_interior=2048, cylinder_refine_fraction=.5))
    np.testing.assert_array_equal(plain.interior(np.random.default_rng(4), 2000),
                                  focused.interior(np.random.default_rng(4), 2000))
    a = focused.sample(np.random.default_rng(42), "cpu", torch.float64)
    b = focused.sample(np.random.default_rng(42), "cpu", torch.float64)
    assert all(torch.equal(a[key], b[key]) for key in a)
    assert len(a["interior"]) == 2048 and focused.contains(a["interior"].numpy()).all()
    radius = np.linalg.norm(a["interior"].numpy()-[2., 2.], axis=1)
    assert np.sum(radius <= 1.5) >= 1024
    assert np.sum(radius > 1.5) > 800
    np.testing.assert_array_equal(plain.collocation(np.random.default_rng(4), 2000),
                                  plain.interior(np.random.default_rng(4), 2000))


@pytest.mark.parametrize("overrides", [dict(cylinder_refine_fraction=-.1),
    dict(cylinder_refine_fraction=1.), dict(cylinder_refine_fraction=float("nan")),
    dict(cylinder_refine_radius=.5), dict(cylinder_refine_radius=float("inf")),
    dict(case="kovasznay", architecture="streamfunction", cylinder_refine_fraction=.5)])
def test_invalid_collocation_refinement(overrides):
    with pytest.raises(ValueError):
        hard_config(**overrides)


def test_warm_start_restores_requested_threads_and_supports_sampling_refinement(tmp_path):
    original = hard_config(threads=2, adam_steps=0, lbfgs_steps=0)
    source = new_output(tmp_path / "source", "cylinder")
    train(original, source)
    refined = hard_config(threads=1, adam_steps=1, lbfgs_steps=1, cylinder_refine_fraction=.5)
    destination = new_output(tmp_path / "refined", "cylinder")
    _, model = train(refined, destination, init_from=source / "checkpoint_last.pt")
    assert torch.get_num_threads() == 1
    assert json.loads((destination / "environment.json").read_text())["torch_num_threads"] == 1
    assert json.loads((destination / "config.json").read_text())["cylinder_refine_fraction"] == .5
    assert torch.isfinite(model.cylinder_constant)


def test_body_and_inlet_refinement_preserves_count_and_uniform_audit():
    config = hard_config(n_interior=2048, cylinder_refine_fraction=.25, inlet_refine_fraction=.25)
    problem = FlowProblem(config)
    points = problem.collocation(np.random.default_rng(42), 2048)
    assert points.shape == (2048, 2) and problem.contains(points).all()
    radius = np.linalg.norm(points[1024:1536]-[2., 2.], axis=1)
    assert np.all((radius >= .5) & (radius <= 1.5))
    assert np.all((points[1536:, 0] >= 0) & (points[1536:, 0] <= config.inlet_refine_width))
    plain = FlowProblem(hard_config())
    np.testing.assert_array_equal(problem.interior(np.random.default_rng(5), 99),
                                  plain.interior(np.random.default_rng(5), 99))


@pytest.mark.parametrize("overrides", [dict(inlet_refine_fraction=-.1), dict(inlet_refine_fraction=float("nan")),
    dict(cylinder_refine_fraction=.5, inlet_refine_fraction=.5), dict(inlet_refine_width=0),
    dict(inlet_refine_width=23.), dict(inlet_refine_width=float("inf"))])
def test_invalid_inlet_refinement(overrides):
    with pytest.raises(ValueError):
        hard_config(**overrides)
