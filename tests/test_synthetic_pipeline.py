"""Exercise real file writers/readers using tiny COMSOL tables and analytic candidates.

Only symbolic search is replaced; normalization, interpolation, publication,
selection, differentiation, diagnostics and result serialization are real.
"""
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import pipeline_contracts as contracts
import step1_extract_wake_comsol as step1
import step3_visualize as step3
import step4_euler_sensitivity as step4
import step5_fd_sensitivity_check as step5
import step6_phi_R_analysis as step6
from experiments import p0_derivative_closure as p0
from experiments import p2_analytic_derivative_loss as p2
from experiments import psi_pareto_gradient_audit as psi_audit
from experiments import mlp_psi_gradient_audit as mlp_audit
try:  # Optional local experiment; not part of the tracked baseline repository.
    from experiments import a0_m_reconstructability as a0
except ImportError:
    a0 = None
from test_pipeline_contracts import run_fixture_training


def invoke(monkeypatch, module, *arguments):
    monkeypatch.setattr(sys, "argv", [module.__name__, *map(str, arguments)])
    return module.main()


def write_comsol_cases(directory):
    directory.mkdir()
    x, y = np.meshgrid(np.linspace(0, 1.2, 49), np.linspace(0, .4, 17), indexing="ij")
    x, y = x.ravel(), y.ravel()
    for name, case in step5.CASES.items():
        xi, eta = (x-case["cx"])/case["D"], (y-case["cy"])/case["D"]
        amplitude = .05 + .2*(case["D"]-.1) + .3*(case["cx"]-.2) + .4*(case["cy"]-.2)
        u = case["Umax"] / 1.5 * (1 + amplitude * xi)
        v = -case["Umax"] / 1.5 * amplitude * eta
        block = np.column_stack([u, v, np.zeros_like(u), np.hypot(u, v), x, y])
        table = np.column_stack([x, y, *[block for _ in range(4)]])
        np.savetxt(directory / case["file"], table, header="synthetic COMSOL wide table", comments="% ")


def test_complete_synthetic_file_chain(tmp_path, monkeypatch):
    raw, data, psi_root, fd = [tmp_path / name for name in ("raw", "data", "psi", "fd")]
    write_comsol_cases(raw)
    invoke(monkeypatch, step1, "--source", raw / "cylinder_flow_result", "--out-dir", data,
           "--symmetric-y-window", "--n-x-grid", 25, "--n-y-grid", 9, "--min-wake-x-points", 5)
    data_info = contracts.dataset_manifest(data)
    run = run_fixture_training(monkeypatch, data, psi_root)
    base_model = contracts.selected_model(psi_root)
    assert base_model["complexity"] != 21

    invoke(monkeypatch, step3, "--out-dir", psi_root)
    invoke(monkeypatch, step4, "--out-dir", psi_root)
    assert (run / "figures/04b_mixed_residual_heatmap.png").exists()
    metrics = pd.read_csv(run / "validation_metrics.csv")
    assert metrics.strategy.tolist() == ["B"]
    np.testing.assert_allclose(metrics.model_divergence_rms, 0.)
    transport = pd.read_csv(run / "sensitivity_results.csv")
    np.testing.assert_allclose(transport.K_R_integrated, transport.S_R_integrated)

    invoke(monkeypatch, step5, "--source-dir", raw, "--base-grid-dir", data,
           "--psi-dir", psi_root, "--out-dir", fd, "--no-plots")
    with np.load(fd / "fd_sensitivity_fields.npz") as loaded:
        fields = {key: loaded[key].copy() for key in loaded.files}
    provenance = contracts.verify_fd_model(fields, base_model)
    assert provenance["evaluation_grid"] == data_info["grid"]
    assert {"u_Rp", "u_Rn", "u_cxp", "u_cxn", "u_cyp", "u_cyn"}.issubset(fields)
    invoke(monkeypatch, p0, "--input", fd / "fd_sensitivity_fields.npz", "--psi-dir", psi_root, "--out-dir", tmp_path / "p0")
    closure = pd.read_csv(tmp_path / "p0/closure_metrics.csv")
    assert closure[closure.k_source == "exact_discrete_path"].rel_l2.max() < 1e-10

    phi_root = fd / "phi_R_pysr"
    run_fixture_training(monkeypatch, fd / "phi_R_input", phi_root)
    result = invoke(monkeypatch, step6, "--fd-dir", fd, "--phi-dir", phi_root,
                    "--psi-dir", psi_root, "--out-dir", tmp_path / "phi_eval", "--no-plots")
    assert result["evaluation"]["target_id"] == provenance["fd_target_id"]
    with np.load(tmp_path / "phi_eval/phi_R_symbolic_fields.npz") as loaded:
        xx, yy = np.meshgrid(fields["x_D"], fields["y_D"], indexing="ij")
        for component in ("u", "v"):
            expected = contracts.evaluate(result["dimensional_total_sympy"][component], xx, yy)
            np.testing.assert_allclose(loaded[f"dimensional_{component}_symbolic"], expected)

    invoke(monkeypatch, psi_audit, "--psi-dir", psi_root, "--fd-dir", fd, "--out-dir", tmp_path / "psi_audit")
    summary = contracts.read_json(tmp_path / "psi_audit/summary.json")
    assert summary["selected_model"]["expression_sha256"] == base_model["expression_sha256"]
    assert summary["candidate_count"] == 2  # B-only table.

    # MLP producer is outside this task; these fixtures check its existing file interface.
    mlp = tmp_path / "mlp"
    shutil.copytree(data, mlp)
    xx, yy = np.meshgrid(fields["x_D"], fields["y_D"], indexing="ij")
    for seed in (0, 1, 2):
        seed_dir = mlp / f"seed_{seed}"
        seed_dir.mkdir()
        np.savez(seed_dir / "sensitivity_fields.npz", S_R_vals=fields["K_R_u_psi"])
        np.savez(seed_dir / "predictions.npz", u_pred=contracts.evaluate(base_model["sympy"], xx, yy))
    result_mlp = invoke(monkeypatch, mlp_audit, "--mlp-dir", mlp, "--psi-dir", psi_root,
                        "--fd-dir", fd, "--out-dir", tmp_path / "mlp_audit")
    assert result_mlp["target_id"] == provenance["fd_target_id"]
    assert result_mlp["base_model"]["expression_sha256"] == base_model["expression_sha256"]
    # New NPZ metadata must remain compatible with the user's existing A0 loader.
    if a0 is not None:
        _, _, _, targets, difference = a0.load_fields(fd / "fd_sensitivity_fields.npz", fd / "data_gradient_closure_fields.npz")
        assert len(targets) == 6 and difference == 0

    bad_model = {**base_model, "expression_sha256": "different"}
    with pytest.raises(ValueError, match="different runs"):
        contracts.verify_fd_model(fields, bad_model)
    changed = dict(fields)
    changed["M_R_u"] = changed["M_R_u"] + 1
    with pytest.raises(ValueError, match="fingerprint"):
        contracts.verify_fd_model(changed, base_model)


class AnalyticSearchFixture:
    def __init__(self, derivative):
        self.equations_ = pd.DataFrame({"complexity": [1, 5, 11], "loss": [1., .2, .1],
            "sympy_format": ["0", "sin(x)*cos(y)", "sin(x)*cos(y) + 0.05*sin(6*x)" if derivative else "sin(x)*cos(y) + 0.02*x"]})

    def fit(self, *args, **kwargs):
        return self


def test_p2_full_reporting_chain_uses_separate_splits(tmp_path, monkeypatch):
    monkeypatch.setattr(p2, "inject_derivative_globals", lambda *args, **kwargs: None)
    monkeypatch.setattr(p2, "make_model", lambda args, objective, seed: AnalyticSearchFixture(objective == p2.DERIVATIVE_OBJECTIVE))
    invoke(monkeypatch, p2, "--out-dir", tmp_path, "--n-train", 41, "--n-validation", 43, "--test-grid", 11)
    metrics = contracts.read_json(tmp_path / "metrics.json")
    protocol = metrics["evaluation_protocol"]
    assert protocol["protocol"] == "train_validate_select_then_test_v2"
    assert len({protocol[k]["sha256"] for k in ("train", "validation", "test")}) == 3
    assert all(protocol[k]["target_sha256"] for k in ("train", "validation", "test"))
    with np.load(tmp_path / "evaluation_splits.npz") as loaded:
        sets = [set(map(tuple, loaded[key])) for key in ("train", "validation", "test")]
    assert all(not sets[i].intersection(sets[j]) for i in range(3) for j in range(i+1, 3))
    for selected in metrics["selected"].values():
        assert "validation_value_mse" in selected and "expression_sha256" in selected
