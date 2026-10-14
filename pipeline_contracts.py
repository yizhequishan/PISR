"""Versioned data, run and selected-model contracts shared by pipeline scripts.

Training writes a fresh runs/<id> directory. Only a completed run is published
through current_run.json; historical flat snapshots remain readable in place.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import sympy as sp

SCHEMA_VERSION = 1
DATA_FILES = ("wake_data.csv", "v_mean_wake.npy", "meta.json")
XI, ETA = sp.symbols("x_D y_D")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def array_hash(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array, dtype="<f8")
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def grid_manifest(xi, eta):
    xi, eta = np.asarray(xi, float), np.asarray(eta, float)
    for name, axis in (("xi", xi), ("eta", eta)):
        if axis.ndim != 1 or len(axis) < 3 or not np.isfinite(axis).all() or not (np.diff(axis) > 0).all():
            raise ValueError(f"{name} must be a finite increasing axis with at least 3 points")
    return {"shape": [len(xi), len(eta)], "sha256": array_hash(xi, eta),
            "xi_range": [float(xi[0]), float(xi[-1])], "eta_range": [float(eta[0]), float(eta[-1])]}


def resolve_run_dir(directory):
    directory = Path(directory).resolve()
    pointer = directory / "current_run.json"
    if pointer.exists():
        entry = read_json(pointer)
        run = (directory / entry["relative_path"]).resolve()
        if not run.is_relative_to(directory / "runs"):
            raise ValueError(f"Run pointer escapes its runs directory: {pointer}")
        state = read_json(run / "run_manifest.json")
        if state.get("status") != "complete" or state.get("run_id") != entry["run_id"]:
            raise ValueError(f"Pointer does not identify a completed run: {pointer}")
        return run
    state_path = directory / "run_manifest.json"
    if state_path.exists() and read_json(state_path).get("status") != "complete":
        raise ValueError(f"Run is not complete: {directory}")
    return directory


def dataset_manifest(directory, *, verify=True):
    """Validate the Cartesian grid and hash both targets and their normalization."""
    directory = Path(directory)
    frame = pd.read_csv(directory / "wake_data.csv").sort_values(["x_D", "y_D"]).reset_index(drop=True)
    meta = read_json(directory / "meta.json")
    v = np.load(directory / "v_mean_wake.npy", allow_pickle=False)
    xi, eta = np.sort(frame.x_D.unique()), np.sort(frame.y_D.unique())
    grid = grid_manifest(xi, eta)
    shape = tuple(grid["shape"])
    if shape != (int(meta["n_x_wake"]), int(meta["n_y"])) or v.shape != shape:
        raise ValueError("Dataset grid/metadata/v shape mismatch")
    xx, yy = np.meshgrid(xi, eta, indexing="ij")
    if len(frame) != xx.size or not np.array_equal(frame[["x_D", "y_D"]].values, np.column_stack([xx.ravel(), yy.ravel()])):
        raise ValueError("wake_data must contain every Cartesian grid point exactly once")
    d, uin = float(meta["D"]), float(meta["U_in"])
    if not np.isfinite(d) or d <= 0 or not np.isfinite(uin) or uin == 0:
        raise ValueError("D must be positive and U_in finite and nonzero")
    for axis, spacing in ((xi, float(meta["dx"]) / d), (eta, float(meta["dy"]) / d)):
        if not np.allclose(np.diff(axis), spacing, rtol=1e-7, atol=1e-12):
            raise ValueError("Grid spacing does not match dx/dy metadata")
    u = frame.u_star.to_numpy(float).reshape(shape)
    if not np.isfinite(u).all() or not np.isfinite(v).all():
        raise ValueError("Dataset targets must be finite")
    normalization = {key: meta.get(key) for key in ("D", "U_in", "cyl_cx", "cyl_cy", "response_parameter", "response_units", "fd_target_id")}
    value = {"schema_version": SCHEMA_VERSION, "grid": grid,
             "target_sha256": array_hash(u, v), "normalization": normalization}
    value["dataset_id"] = content_hash(value)
    value["files"] = {name: file_hash(directory / name) for name in DATA_FILES}
    existing = directory / "data_manifest.json"
    if verify and existing.exists() and read_json(existing) != value:
        raise ValueError(f"Data snapshot changed since its manifest was written: {directory}")
    return value


def write_data_manifest(directory):
    value = dataset_manifest(directory, verify=False)
    atomic_json(Path(directory) / "data_manifest.json", value)
    return value


def load_strategies(directory):
    directory = resolve_run_dir(directory)
    dataset_manifest(directory)
    state_path = directory / "run_manifest.json"
    declared = read_json(state_path).get("strategies") if state_path.exists() else None
    result = {}
    for label in ("A", "B"):
        path = directory / f"results_strategy_{label.lower()}.csv"
        if declared is not None and label not in declared:
            if path.exists():
                raise ValueError(f"Undeclared strategy {label} in run {directory}")
            continue
        if path.exists():
            frame = pd.read_csv(path)
            if frame.empty:
                raise ValueError(f"Empty Pareto table: {path}")
            result[label] = frame
        elif declared is not None:
            raise FileNotFoundError(path)
    if not result:
        raise FileNotFoundError(f"No strategy results in {directory}")
    return result


def expression_record(row):
    if "psi_sympy" in row and pd.notna(row["psi_sympy"]):
        psi = sp.sympify(str(row["psi_sympy"]))
        expressions = {"psi_sympy": str(psi), "sympy": str(sp.diff(psi, ETA)), "v_sympy": str(-sp.diff(psi, XI))}
    else:
        expressions = {"sympy": str(sp.sympify(str(row["sympy"])))}
    if any(sp.sympify(text).free_symbols - {XI, ETA} for text in expressions.values()):
        raise ValueError("Selected model must be a closed expression in x_D/y_D")
    return {**expressions, "expression_sha256": content_hash(expressions)}


def evaluate(expr, xx, yy):
    expression = sp.sympify(expr)
    if expression.free_symbols - {XI, ETA}:
        raise ValueError("Expression has unsupported symbols")
    with np.errstate(all="ignore"):
        values = sp.lambdify((XI, ETA), expression, modules="numpy")(xx, yy)
        array = np.broadcast_to(np.asarray(values, dtype=float), np.shape(xx)).copy()
    if not np.isfinite(array).all():
        raise ValueError("Expression produced non-finite values")
    return array


def build_selected_models(directory):
    directory = Path(directory)
    data = dataset_manifest(directory)
    meta = read_json(directory / "meta.json")
    frame = pd.read_csv(directory / "wake_data.csv").sort_values(["x_D", "y_D"])
    xx, yy = frame.x_D.values, frame.y_D.values
    truth_u = frame.u_star.values
    truth_v = np.load(directory / "v_mean_wake.npy").ravel()
    # Called before publication too; no completed-run resolver is used here.
    state_path = directory / "run_manifest.json"
    state = read_json(state_path) if state_path.exists() else {}
    labels = state.get("strategies", [label for label in ("A", "B") if (directory / f"results_strategy_{label.lower()}.csv").exists()])
    models = {}
    for label in labels:
        path = directory / f"results_strategy_{label.lower()}.csv"
        results = pd.read_csv(path)
        candidates = []
        for index, row in results.iterrows():
            try:
                record = expression_record(row)
                mse = float(np.mean((evaluate(record["sympy"], xx, yy) - truth_u)**2))
                mse_v = float(np.mean((evaluate(record["v_sympy"], xx, yy) - truth_v)**2)) if "v_sympy" in record else None
                if label == "B" and mse_v is not None:
                    score, metric = mse + float(meta.get("lambda_v", 1.0)) * mse_v, "analytic_u_mse_plus_lambda_v_mse"
                elif label == "B" and np.isfinite(row.get("combined_loss", np.nan)):
                    score, metric = float(row["combined_loss"]), "training_combined_loss"
                else:
                    score, metric = mse, "analytic_u_mse"
                candidates.append({**record, "strategy": label, "equation_index": int(row.get("equation_index", index)),
                    "complexity": int(row["complexity"]), "selection_metric": metric, "selection_score": score,
                    "mse": mse, "mse_v": mse_v})
            except (ValueError, TypeError, OverflowError, ZeroDivisionError):
                continue
        if not candidates:
            raise ValueError(f"No finite supported candidates for strategy {label}")
        models[label] = min(candidates, key=lambda item: (item["selection_score"], item["complexity"], item["equation_index"]))
        models[label]["results_sha256"] = file_hash(path)
    if not models:
        raise ValueError("No model candidates to select")
    return {"schema_version": SCHEMA_VERSION, "run_id": state.get("run_id", "legacy_unverified"),
            "training_data": data, "selection_role": "training_snapshot_selection", "models": models}


def selected_models(directory):
    directory = resolve_run_dir(directory)
    path = directory / "selected_models.json"
    if not path.exists():
        atomic_json(path, build_selected_models(directory))
    value = read_json(path)
    if value["training_data"] != dataset_manifest(directory):
        raise ValueError("Selected model training snapshot changed; create a new run")
    strategies = load_strategies(directory)
    if set(value["models"]) != set(strategies):
        raise ValueError("Selected model strategies differ from the run")
    for label, model in value["models"].items():
        if model["results_sha256"] != file_hash(directory / f"results_strategy_{label.lower()}.csv"):
            raise ValueError("Pareto table changed after selection; create a new run")
        canonical = expression_record(model)
        if any(model.get(key) != text for key, text in canonical.items()):
            raise ValueError("Selected expression fingerprint mismatch")
    return value


def selected_model(directory, strategy="B"):
    manifest = selected_models(directory)
    if strategy not in manifest["models"]:
        raise ValueError(f"Strategy {strategy} is unavailable")
    return {**manifest["models"][strategy], "run_id": manifest["run_id"], "training_data": manifest["training_data"]}


@contextmanager
def training_run(output_root, input_dir, *, pipeline, strategies, settings):
    output_root, input_dir = Path(output_root).resolve(), resolve_run_dir(input_dir)
    if output_root == input_dir or input_dir.is_relative_to(output_root / "runs"):
        raise ValueError("Training input and output must be separate snapshots")
    source = dataset_manifest(input_dir)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run = output_root / "runs" / run_id
    run.mkdir(parents=True, exist_ok=False)
    state = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "status": "running", "pipeline": pipeline,
             "strategies": list(strategies), "settings": settings, "source": {"directory": str(input_dir), "data": source}}
    atomic_json(run / "run_manifest.json", state)
    try:
        yield run
        data = write_data_manifest(run)
        for label in ("A", "B"):
            exists = (run / f"results_strategy_{label.lower()}.csv").exists()
            if exists != (label in strategies):
                raise ValueError(f"Unexpected/missing strategy {label} in new run")
        atomic_json(run / "selected_models.json", build_selected_models(run))
        state.update(status="complete", data=data)
        atomic_json(run / "run_manifest.json", state)
        atomic_json(output_root / "current_run.json", {"schema_version": SCHEMA_VERSION, "run_id": run_id,
                    "relative_path": run.relative_to(output_root).as_posix()})
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        atomic_json(run / "run_manifest.json", state)
        raise


def archive_provenance(fields):
    if "provenance_json" not in fields:
        raise ValueError("FD archive lacks provenance; rerun Step 5 with the current code")
    provenance = json.loads(str(np.asarray(fields["provenance_json"]).item()))
    if provenance["evaluation_grid"] != grid_manifest(fields["x_D"], fields["y_D"]):
        raise ValueError("FD archive grid fingerprint mismatch")
    actual = content_hash({key: array_hash(value) for key, value in sorted(fields.items()) if key != "provenance_json"})
    if actual != provenance.get("archive_fields_sha256", provenance["fd_target_id"]):
        raise ValueError("FD archive fields differ from their recorded fingerprint")
    return provenance


def verify_fd_model(fields, model):
    provenance = archive_provenance(fields)
    recorded = provenance["base_model"]
    if (recorded["expression_sha256"], recorded["run_id"], recorded["training_data"]["dataset_id"]) != (
        model["expression_sha256"], model["run_id"], model["training_data"]["dataset_id"]):
        raise ValueError("FD transport and selected base model belong to different runs; rerun Step 5")
    return provenance
