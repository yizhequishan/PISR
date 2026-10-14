"""Data isolation and frozen-weight contracts for the residual-SR trial."""
import itertools

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinn.config import Config
from pinn.problems import FlowProblem
from scripts.pinn_sr_residual_trial import (
    SEEDS, FrozenWeightTransform, development_points, features, region_diagnostics,
)


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config():
    return Config(case="cylinder", reynolds=20., architecture="hard_streamfunction",
                  device="cpu", threads=1).validate()


def batch():
    return {
        "interior": torch.tensor([[.1, .1], [.8, 2.], [2.8, 2.], [10., 3.9]],
                                 dtype=torch.float64, requires_grad=True),
        "left": torch.tensor([[0., 1.], [0., 3.]], dtype=torch.float64),
        "obstacle": torch.tensor([[1.5, 2.], [2.5, 2.]], dtype=torch.float64),
    }


def selection(expression="10*x_D + 20*y_D"):
    return {"expression": expression, "score_cap": 3., "score_mean": 2.}


def test_ordinary_transform_is_identity_and_only_records_provenance():
    original = batch()
    before = {name: tensor.detach().clone() for name, tensor in original.items()}
    transform = FrozenWeightTransform("ordinary", config(), selection())
    result = transform(original)
    assert result is original
    assert result.keys() == before.keys()
    for name in original:
        torch.testing.assert_close(original[name], before[name], rtol=0, atol=0)
    assert len(transform.batch_hashes) == 1
    assert transform.weight_stats == []


@pytest.mark.parametrize("kind", ["inlet", "sr"])
def test_frozen_weights_preserve_coordinates_and_boundaries(kind):
    original = batch()
    before = {name: tensor.detach().clone() for name, tensor in original.items()}
    transform = FrozenWeightTransform(kind, config(), selection())
    result = transform(original)
    assert "interior_weights" not in original
    assert result.keys() == original.keys() | {"interior_weights"}
    for name, tensor in original.items():
        assert result[name] is tensor
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
    weights = result["interior_weights"]
    assert weights.shape == (len(original["interior"]), 1)
    assert torch.isfinite(weights).all() and (weights > 0).all()
    assert not weights.requires_grad and weights.grad_fn is None
    assert weights.mean().item() == pytest.approx(1., abs=1e-14)
    assert original["interior"].requires_grad  # The callback cannot detach model inputs.
    repeated = FrozenWeightTransform(kind, config(), selection())(original)
    torch.testing.assert_close(weights, repeated["interior_weights"], rtol=0, atol=0)
    ordinary = FrozenWeightTransform("ordinary", config(), selection())
    ordinary(original)
    assert transform.batch_hashes == ordinary.batch_hashes


@pytest.mark.parametrize("expression", ["2.0", "0.0", "-3.0"])
def test_constant_symbolic_scores_reduce_to_uniform_weights(expression):
    result = FrozenWeightTransform("sr", config(), selection(expression))(batch())
    torch.testing.assert_close(result["interior_weights"],
                               torch.ones_like(result["interior_weights"]), rtol=0, atol=0)


def test_features_preserve_diameter_units_and_normalize_channel_height():
    c = config()
    points = np.array([[0., 0.], [7., c.channel_height/2], [c.channel_length, c.channel_height]])
    before = points.copy()
    np.testing.assert_array_equal(features(points, c), [[0., -1.], [7., 0.], [c.channel_length, 1.]])
    np.testing.assert_array_equal(points, before)


def test_overlapping_physical_regions_form_an_exclusive_diagnostic_partition():
    c = config()
    # First point belongs to inlet and near-body masks; second belongs to
    # near-body and wake masks. Each must contribute to only its first region.
    points = np.array([[.8, 2.], [2.8, 2.], [4., 2.], [10., 2.]])
    residuals = np.array([[0., 1., 4.], [0., 2., 3.], [0., 3., 2.], [0., 4., 1.]])
    result = region_diagnostics(points, residuals, c)
    names = ["inlet_x_lt_1", "near_cylinder", "near_wake", "remaining"]
    assert set(result) == set(names)
    assert sum(row["n"] for row in result.values()) == len(points)
    assert sum(row["fraction"] for row in result.values()) == pytest.approx(1.)
    np.testing.assert_allclose(np.sum([row["squared_residual_share"] for row in result.values()], axis=0), [1., 1.])
    for index, name in enumerate(names):
        assert result[name]["n"] == 1
        np.testing.assert_allclose(result[name]["momentum_rms"], residuals[index, 1:])


def test_development_and_test_samples_have_distinct_reproducible_seeds():
    assert len(set(SEEDS.values())) == len(SEEDS)
    problem = FlowProblem(config())
    samples = {name: development_points(problem, SEEDS[name], 8, 4) for name in ("fit", "selection")}
    samples["test"] = problem.interior(np.random.default_rng(SEEDS["test"]), 12)
    for name in ("fit", "selection"):
        np.testing.assert_array_equal(samples[name], development_points(problem, SEEDS[name], 8, 4))
    for points in samples.values():
        assert problem.contains(points).all()
    for a, b in itertools.combinations(samples.values(), 2):
        assert not np.any(np.all(a[:, None, :] == b[None, :, :], axis=2))
