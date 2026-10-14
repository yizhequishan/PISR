import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import sympy as sp

import pipeline_contracts as contracts
import step2_pisr_psi as step2
import step3_visualize as step3
import step4_euler_sensitivity as step4
import step6_phi_R_analysis as step6
from experiments import p1_observability_nullspace as p1
from experiments import p2_analytic_derivative_loss as p2
from experiments import summarize_preliminary_experiments as summary


def write_dataset(directory, *, n_x=7, n_y=9):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    x, y = np.linspace(3, 5, n_x), np.linspace(-1, 1, n_y)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    pd.DataFrame({"x_D": xx.ravel(), "y_D": yy.ravel(), "u_star": (1 + .05 * xx).ravel()}).to_csv(directory / "wake_data.csv", index=False)
    np.save(directory / "v_mean_wake.npy", -.05 * yy)
    contracts.atomic_json(directory / "meta.json", {"D": .1, "U_in": 1., "cyl_cx": .2, "cyl_cy": .2,
        "dx": .1 * (x[1]-x[0]), "dy": .1 * (y[1]-y[0]), "n_x_wake": n_x, "n_y": n_y,
        "lambda_v": 1., "pipeline": "stream_function_psi", "data_source": "synthetic"})
    contracts.write_data_manifest(directory)
    return directory


def candidate_fixture(label, *, X, u_data, v_data, run_out_dir, **kwargs):
    equations = pd.DataFrame({"complexity": [1, 7], "sympy_format": ["0", "y_D*(1 + 0.05*x_D)"], "loss": [0., 100.]})
    results = step2.evaluate_psi_pareto(equations, X, u_data, v_data, strategy_label=label)
    results.to_csv(Path(run_out_dir) / f"results_strategy_{label.lower()}.csv", index=False)


def run_fixture_training(monkeypatch, source, output, *, labels=("B",)):
    # Exercise the production transaction and snapshot writer, replacing only Julia search.
    monkeypatch.setattr(step2, "inject_psi_globals", lambda **kwargs: None)
    monkeypatch.setattr(step2, "run_psi_strategy", candidate_fixture)
    args = SimpleNamespace(base_out_dir=str(source.parent), input_subdir=source.name,
        out_subdir=output.name, lambda_v=1., niterations=1, maxsize=10, skip_strategy_a=labels == ("B",))
    monkeypatch.setattr(step2, "parse_args", lambda: args)
    return step2.main()


def test_training_publication_isolates_old_results_and_failure(tmp_path, monkeypatch):
    source = write_dataset(tmp_path / "input")
    root = tmp_path / "psi"
    root.mkdir()
    old = root / "results_strategy_a.csv"
    old.write_text("legacy A from another grid", encoding="utf-8")
    first = run_fixture_training(monkeypatch, source, root)
    assert contracts.resolve_run_dir(root) == first
    assert not (first / "results_strategy_a.csv").exists()
    assert old.read_text(encoding="utf-8") == "legacy A from another grid"
    selected = contracts.selected_model(root)
    assert selected["complexity"] == 7  # Deliberately not C21, even though raw loss prefers C1.
    pointer = (root / "current_run.json").read_bytes()

    def fail(*args, **kwargs):
        raise RuntimeError("simulated Julia failure")
    monkeypatch.setattr(step2, "run_psi_strategy", fail)
    with pytest.raises(RuntimeError, match="simulated"):
        step2.main()
    assert (root / "current_run.json").read_bytes() == pointer
    failed = [p for p in (root / "runs").iterdir() if p != first][0]
    assert contracts.read_json(failed / "run_manifest.json")["status"] == "failed"
    with pytest.raises(ValueError, match="not complete"):
        contracts.resolve_run_dir(failed)
    assert contracts.selected_model(root)["expression_sha256"] == selected["expression_sha256"]


def test_selected_expression_fields_are_verified(tmp_path, monkeypatch):
    source = write_dataset(tmp_path / "input")
    run = run_fixture_training(monkeypatch, source, tmp_path / "psi")
    path = run / "selected_models.json"
    selection = contracts.read_json(path)
    selection["models"]["B"]["sympy"] = "123"
    contracts.atomic_json(path, selection)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        contracts.selected_model(run)


def test_data_and_candidate_tampering_is_rejected(tmp_path, monkeypatch):
    source = write_dataset(tmp_path / "input")
    run = run_fixture_training(monkeypatch, source, tmp_path / "psi")
    table_path = run / "results_strategy_b.csv"
    frame = pd.read_csv(table_path)
    frame.loc[0, "psi_sympy"] = "x_D"
    frame.to_csv(table_path, index=False)
    with pytest.raises(ValueError, match="Pareto table changed"):
        contracts.selected_model(run)
    data = pd.read_csv(source / "wake_data.csv")
    data.loc[0, "u_star"] += 1.
    data.to_csv(source / "wake_data.csv", index=False)
    with pytest.raises(ValueError, match="snapshot changed"):
        contracts.dataset_manifest(source)


@pytest.mark.parametrize("fault", ["duplicate", "spacing", "nonfinite"])
def test_dataset_validation(tmp_path, fault):
    source = write_dataset(tmp_path / "input")
    if fault == "spacing":
        meta = contracts.read_json(source / "meta.json")
        meta["dx"] *= 2
        contracts.atomic_json(source / "meta.json", meta)
    else:
        frame = pd.read_csv(source / "wake_data.csv")
        if fault == "duplicate":
            frame.iloc[0] = frame.iloc[1]
        else:
            frame.loc[0, "u_star"] = np.nan
        frame.to_csv(source / "wake_data.csv", index=False)
    with pytest.raises(ValueError):
        contracts.dataset_manifest(source, verify=False)


@pytest.mark.parametrize("labels", [("A",), ("B",), ("A", "B")])
def test_available_strategies_and_consistent_selection(tmp_path, labels):
    run = write_dataset(tmp_path / "run")
    frame = pd.read_csv(run / "wake_data.csv")
    for label in labels:
        candidate_fixture(label, X=frame[["x_D", "y_D"]].values, u_data=frame.u_star.values,
                          v_data=np.load(run / "v_mean_wake.npy").ravel(), run_out_dir=run)
    _, _, loaded, _ = step3.load_inputs(run)
    assert set(loaded) == set(labels)
    _, _, a, b = step4.load_inputs(run)
    assert a.empty == ("A" not in labels)
    assert b.empty == ("B" not in labels)
    chosen = contracts.selected_models(run)
    assert set(chosen["models"]) == set(labels)


def test_stream_function_divergence_is_separate_from_mixed_residual():
    points = np.array([[3., -1.], [4., 0.], [5., 1.]])
    row = contracts.expression_record({"psi_sympy": "x_D*y_D + y_D**3"})
    model, mixed, kind = step3.divergence_fields(row, points, np.zeros(3))
    np.testing.assert_allclose(model, 0.)
    np.testing.assert_allclose(mixed, 1.)
    assert kind.startswith("model divergence")
    assert step4.compute_euler_sensitivity("x_D+y_D**2", np.array([[4., 0.]]), .1)[1][0] == -80.


def test_phi_candidates_are_recomputed_on_current_targets():
    x, y = np.linspace(3., 5., 5), np.linspace(-1., 1., 7)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    results = pd.DataFrame({"complexity": [3, 7, 5], "equation_index": [0, 1, 2],
        "psi_sympy": ["y_D", "x_D*y_D", "y_D/(x_D-4)"],
        "mse": [0., 100., 0.], "mse_v": [0., 100., 0.]})
    table = step6.candidate_table(results, xx, -yy, x, y)
    chosen = table[table.selection != ""].iloc[0]
    assert chosen.equation_index == 1
    assert chosen.balanced_score == 0.
    assert chosen.training_mse == 100.
    assert table.iloc[2].status.startswith("rejected")


def test_dimensional_metrics_use_exported_formula():
    xi, eta = sp.symbols("x_D y_D")
    xx, yy = np.meshgrid(np.linspace(3, 5, 5), np.linspace(-1, 1, 7), indexing="ij")
    phi, psi = xi*eta**2, eta*(1 + xi**2)
    phi_dim, expressions, values = step6.dimensional_response(phi, psi, xx, yy, .1, .7, -14.)
    assert sp.expand(phi_dim - (.7*phi - 14*psi)) == 0
    for component in ("u", "v"):
        np.testing.assert_allclose(values[component], contracts.evaluate(str(expressions[component]), xx, yy))
    assert sp.simplify(sp.diff(expressions["u"], xi) + sp.diff(expressions["v"], eta)) == 0


def test_p2_freezes_validation_choice_even_if_test_prefers_another():
    validation = pd.DataFrame({"source": ["value_only"]*2, "status": ["ok"]*2,
        "equation_index": [0, 1], "expression": ["x", "y"], "value_mse": [.1, .2],
        "gradient_mse": [.1, .2], "normalized_h1_score": [.2, .4]})
    choice = p2.select_row(validation, "value_only", "value_mse")
    test = validation.copy()
    test["value_mse"] = [9., 0.]
    row = p2.test_row_for_selection(choice, test)
    assert row.equation_index == 0 and row.value_mse == 9.
    test.loc[0, "status"] = "skipped: singular"
    with pytest.raises(ValueError, match="do not reselect"):
        p2.test_row_for_selection(choice, test)


def test_uniform_sparse_observations_cover_domain():
    axis = np.linspace(-1., 1., 41)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    _, u, v = p1.basis_and_observations(xx.ravel(), yy.ravel())
    chosen = p1.uniform_rows(xx.size, 8)
    assert len(set(chosen)) == 8
    assert xx.ravel()[chosen].min() == -1 and xx.ravel()[chosen].max() == 1
    assert p1.numerical_rank(np.vstack([u[:, 1:], v[chosen, 1:]]))[0] == 6


@pytest.mark.parametrize("versioned", [False, True])
def test_summary_preserves_protocol_and_never_claims_legacy_test(tmp_path, monkeypatch, versioned):
    p0_dir, p1_dir, p2_dir = [tmp_path / name for name in ("p0", "p1", "p2")]
    for directory in (p0_dir, p1_dir, p2_dir):
        directory.mkdir()
    pd.DataFrame([{"parameter": parameter, "k_source": source, "rel_l2": error}
                  for parameter in ("R", "cx", "cy")
                  for source, error in (("exact_discrete_path", 1e-16), ("baseline_data_gradient", .1), ("symbolic_analytic_gradient", .2))]
                 ).to_csv(p0_dir / "closure_metrics.csv", index=False)
    p1_metrics = {"u_only": {"rank": 3, "nullity": 4}, "u_plus_v": {"rank": 6, "nullity": 1}}
    if versioned:
        p1_metrics["sampling_protocol"] = "full_domain_uniform_per_budget_v2"
    contracts.atomic_json(p1_dir / "metrics.json", p1_metrics)
    pd.DataFrame([{"method": name, "n_v": 3, "rank_without_gauge": 6, "coefficient_error_mean": .01}
                  for name in ("uniform", "random", "d_optimal")]).to_csv(p1_dir / "sparse_v_budget.csv", index=False)
    selected = {"complexity": 5, "value_mse": .01, "gradient_mse": .02, "gradient_rel_l2": .03}
    metrics = {"selected": {key: dict(selected) for key in ("value_only_by_value_mse", "analytic_derivative_by_h1")}}
    if versioned:
        metrics["evaluation_protocol"] = {"protocol": "train_validate_select_then_test_v2",
                                         **{key: {"sha256": key} for key in ("train", "validation", "test")}}
        for row in metrics["selected"].values():
            row.update(expression_sha256="frozen_expression", validation_value_mse=.01)
    contracts.atomic_json(p2_dir / "metrics.json", metrics)
    # No common complexity: report unavailability without inventing a comparison or failing.
    pd.DataFrame([{"source": name, "status": "ok", "complexity": complexity}
                  for name, complexity in (("value_only", 3), ("analytic_derivative", 5))]
                 ).to_csv(p2_dir / "candidate_metrics.csv", index=False)
    out = tmp_path / "summary"
    monkeypatch.setattr(summary, "parse_args", lambda: SimpleNamespace(p0_dir=p0_dir, p1_dir=p1_dir, p2_dir=p2_dir, out_dir=out))
    result = summary.main()
    assert result["P2"]["matched_complexity"] is None
    assert result["sources"]["P2"]["files_sha256"]["metrics.json"] == contracts.file_hash(p2_dir / "metrics.json")
    report = (out / "SUMMARY.md").read_text(encoding="utf-8")
    if versioned:
        assert result["P2"]["evaluation_regime"] == "independent_test_after_validation_selection"
        assert "frozen using an independent validation set" in report
    else:
        assert "not independent test evidence" in report
        assert "Rerun P1" in report
