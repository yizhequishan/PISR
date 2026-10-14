"""Frozen spatial weighting and matched-checkpoint continuation contracts."""
import importlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinn.config import Config
from pinn.losses import loss_terms, weighted_loss
from pinn.physics import navier_stokes
from pinn.train import load_checkpoint, new_output, setup, train


def small_config(**overrides):
    values = dict(case="cylinder", reynolds=20., architecture="hard_streamfunction",
                  hidden_layers=(8, 8), device="cpu", threads=1, n_interior=12,
                  n_boundary=8, adam_steps=2, lbfgs_steps=1, log_every=1,
                  resample_every=1, eval_grid=9)
    values.update(overrides)
    return Config.from_dict(values)


@pytest.mark.parametrize("architecture", ["uvp", "hard_streamfunction"])
def test_spatial_weights_match_manual_momentum_loss_and_gradients(architecture):
    config = small_config(architecture=architecture)
    problem, model, device, dtype = setup(config)
    batch = problem.sample(np.random.default_rng(29), device, dtype)
    plain = loss_terms(model, problem, batch)
    weights = torch.linspace(.5, 1.5, config.n_interior, dtype=dtype).reshape(-1, 1)
    weighted = loss_terms(model, problem, {**batch, "interior_weights": weights})
    residuals = navier_stokes(model, batch["interior"], config.reynolds)
    for name in ("momentum_x", "momentum_y"):
        torch.testing.assert_close(weighted["pde_" + name], (weights * residuals[name].square()).mean())
    for name in plain.keys() - {"pde_momentum_x", "pde_momentum_y"}:
        torch.testing.assert_close(weighted[name], plain[name], rtol=0, atol=0)
    actual = torch.autograd.grad(weighted["pde_momentum_x"] + weighted["pde_momentum_y"],
                                 tuple(model.parameters()), allow_unused=True)
    expected = torch.autograd.grad(sum((weights * residuals[key].square()).mean()
                                      for key in ("momentum_x", "momentum_y")),
                                   tuple(model.parameters()), allow_unused=True)
    for a, b in zip(actual, expected):
        if a is None or b is None:
            assert a is None and b is None
        else:
            torch.testing.assert_close(a, b)
    uniform = loss_terms(model, problem, {**batch, "interior_weights": torch.ones_like(weights)})
    torch.testing.assert_close(weighted_loss(uniform, config), weighted_loss(plain, config), rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["wrong_shape", "zero", "negative", "nan", "inf", "trainable", "not_tensor"])
def test_invalid_spatial_weights_rejected(kind):
    config = small_config()
    problem, model, device, dtype = setup(config)
    batch = problem.sample(np.random.default_rng(19), device, dtype)
    weights = torch.ones((config.n_interior, 1), dtype=dtype)
    if kind == "wrong_shape":
        weights = weights[:, 0]
    elif kind == "trainable":
        weights.requires_grad_(True)
    elif kind == "not_tensor":
        weights = weights.numpy()
    else:
        weights[0, 0] = {"zero": 0., "negative": -1., "nan": float("nan"), "inf": float("inf")}[kind]
    with pytest.raises(ValueError, match="interior_weights"):
        loss_terms(model, problem, {**batch, "interior_weights": weights})


def test_uniform_weights_keep_same_checkpoint_and_sampling_trajectory(tmp_path):
    source = new_output(tmp_path / "source", "cylinder")
    train(small_config(adam_steps=0, lbfgs_steps=0), source)
    config = small_config()
    recorded = {"plain": [], "weighted": []}
    states = {}
    for name in recorded:
        def transform(batch, arm=name):
            recorded[arm].append({key: value.clone() for key, value in batch.items()})
            return {**batch, "interior_weights": torch.ones_like(batch["interior"][:, :1])} if arm == "weighted" else batch

        out = new_output(tmp_path / name, "cylinder")
        _, model = train(config, out, init_from=source / "checkpoint_last.pt", batch_transform=transform)
        _, _, loaded, saved = load_checkpoint(out / "checkpoint_last.pt", "cpu")
        states[name] = model.state_dict()
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)
        assert saved["progress"]["status"] == "training_finished"
        assert saved["progress"]["adam_steps"] == 2
        assert saved["progress"]["lbfgs_steps"] == 1
        assert saved["progress"]["closure_evaluations"] >= 1
        assert len(recorded[name]) == 3  # Initial, Adam resample, frozen L-BFGS batch.
    for a, b in zip(recorded["plain"], recorded["weighted"]):
        assert a.keys() == b.keys()
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    for key in states["plain"]:
        torch.testing.assert_close(states["plain"][key], states["weighted"][key], rtol=0, atol=0)


def test_walltime_budget_stops_between_real_optimizer_steps_and_saves(tmp_path, monkeypatch):
    training_module = importlib.import_module("pinn.train")
    clock = [0.]
    original_step = torch.optim.Adam.step

    def timed_step(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        clock[0] += 1.
        return result

    monkeypatch.setattr(training_module.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(torch.optim.Adam, "step", timed_step)
    out = new_output(tmp_path / "limited", "cylinder")
    train(small_config(adam_steps=5, lbfgs_steps=3), out, max_seconds=.5)
    _, _, _, saved = load_checkpoint(out / "checkpoint_last.pt", "cpu")
    progress = saved["progress"]
    assert progress["status"] == "budget_finished"
    assert progress["adam_steps"] == 1
    assert progress["lbfgs_steps"] == progress["closure_evaluations"] == 0
    assert progress["elapsed_seconds"] == 1.
    assert json.loads((out / "training_status.json").read_text()) == progress


def test_walltime_budget_includes_model_setup(tmp_path, monkeypatch):
    training_module = importlib.import_module("pinn.train")
    clock = [0.]
    original_setup = training_module.setup

    def timed_setup(config):
        result = original_setup(config)
        clock[0] += .75
        return result

    monkeypatch.setattr(training_module.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(training_module, "setup", timed_setup)
    out = new_output(tmp_path / "setup_limited", "cylinder")
    train(small_config(), out, max_seconds=.5)
    progress = json.loads((out / "training_status.json").read_text())
    assert progress["status"] == "budget_finished"
    assert progress["elapsed_seconds"] == .75
    assert progress["adam_steps"] == progress["lbfgs_steps"] == 0
    assert progress["closure_evaluations"] == 0


@pytest.mark.parametrize("budget", [0., -1., float("nan"), float("inf"), True])
def test_invalid_time_budget_rejected(tmp_path, budget):
    with pytest.raises(ValueError, match="max_seconds"):
        train(small_config(), tmp_path, max_seconds=budget)
