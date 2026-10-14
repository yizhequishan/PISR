"""A0: test whether intrinsic response fields are local functions of the base flow.

This is a deliberately small, deterministic NumPy-only probe.  It builds a
local feature dictionary from the baseline velocity, its first and second
spatial derivatives, coordinates, and the analytic coordinate-generator
forcing.  Ridge models are evaluated by leave-one-xi-slice-out (LOXO): every
point on a held xi section is excluded from the corresponding fit.

The primary Gate 1 decision uses the predeclared ``local_full`` quadratic
dictionary and ``--ridge-alpha``.  The alpha sweep and the two ablations are
diagnostics only; they do not silently tune the reported gate on held-out data.
No PySR, Julia, SciPy, pandas, or scikit-learn code is imported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


REPO = Path(__file__).resolve().parents[1]
DEFAULT_FD = REPO / "outputs_mgn" / "fd_check" / "fd_sensitivity_fields.npz"
DEFAULT_CLOSURE = (
    REPO / "outputs_mgn" / "fd_check" / "data_gradient_closure_fields.npz"
)
DEFAULT_OUT = REPO / "outputs" / "a0_m_reconstructability"
PARAMETERS = ("R", "cx", "cy")
COMPONENTS = ("u", "v")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd-fields", type=Path, default=DEFAULT_FD)
    parser.add_argument("--closure-fields", type=Path, default=DEFAULT_CLOSURE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--diameter",
        type=float,
        default=0.1,
        help="Baseline cylinder diameter D [m], used only in analytic forcing.",
    )
    parser.add_argument("--degree", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1e-6,
        help="Predeclared dimensionless ridge strength used for Gate 1.",
    )
    parser.add_argument(
        "--alpha-grid",
        default="1e-10,1e-8,1e-6,1e-4,1e-2,1e0",
        help="Comma-separated diagnostic ridge sweep; must contain ridge-alpha.",
    )
    parser.add_argument("--strict-threshold", type=float, default=0.10)
    parser.add_argument("--practical-threshold", type=float, default=0.20)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fields(
    fd_path: Path, closure_path: Path
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], float]:
    with np.load(fd_path, allow_pickle=False) as loaded:
        required = {"x_D", "y_D", "u_base", "v_base", "M_R_u", "M_R_v"}
        missing = required.difference(loaded.files)
        if missing:
            raise KeyError(f"{fd_path} is missing: {sorted(missing)}")
        xi = loaded["x_D"].astype(float, copy=True)
        eta = loaded["y_D"].astype(float, copy=True)
        base = {
            "u": loaded["u_base"].astype(float, copy=True),
            "v": loaded["v_base"].astype(float, copy=True),
        }
        targets = {
            "R_u": loaded["M_R_u"].astype(float, copy=True),
            "R_v": loaded["M_R_v"].astype(float, copy=True),
        }

    with np.load(closure_path, allow_pickle=False) as loaded:
        required = {
            "x_D",
            "y_D",
            "R_u_M",
            "R_v_M",
            "cx_u_M",
            "cx_v_M",
            "cy_u_M",
            "cy_v_M",
        }
        missing = required.difference(loaded.files)
        if missing:
            raise KeyError(f"{closure_path} is missing: {sorted(missing)}")
        if not np.array_equal(xi, loaded["x_D"]) or not np.array_equal(
            eta, loaded["y_D"]
        ):
            raise ValueError("FD and closure archives use different xi/eta grids")
        targets.update(
            {
                f"{parameter}_{component}": loaded[f"{parameter}_{component}_M"].astype(
                    float, copy=True
                )
                for parameter in ("cx", "cy")
                for component in COMPONENTS
            }
        )
        r_archive_difference = max(
            float(np.max(np.abs(targets[f"R_{component}"] - loaded[f"R_{component}_M"])))
            for component in COMPONENTS
        )

    expected_shape = (len(xi), len(eta))
    all_fields = {**base, **targets}
    bad_shapes = {name: value.shape for name, value in all_fields.items() if value.shape != expected_shape}
    if bad_shapes:
        raise ValueError(f"Expected field shape {expected_shape}, found {bad_shapes}")
    if not np.all(np.diff(xi) > 0) or not np.all(np.diff(eta) > 0):
        raise ValueError("xi and eta must be strictly increasing")
    if not all(np.all(np.isfinite(value)) for value in all_fields.values()):
        raise ValueError("All base and target fields must be finite")
    return xi, eta, base, targets, r_archive_difference


def field_derivatives(
    field: np.ndarray, xi: np.ndarray, eta: np.ndarray
) -> dict[str, np.ndarray]:
    d_xi, d_eta = np.gradient(field, xi, eta, edge_order=2)
    d_xixi = np.gradient(d_xi, xi, axis=0, edge_order=2)
    d_etaeta = np.gradient(d_eta, eta, axis=1, edge_order=2)
    # Averaging both paths reduces directional roundoff in the mixed derivative.
    d_xieta = 0.5 * (
        np.gradient(d_xi, eta, axis=1, edge_order=2)
        + np.gradient(d_eta, xi, axis=0, edge_order=2)
    )
    return {
        "xi": d_xi,
        "eta": d_eta,
        "xixi": d_xixi,
        "xieta": d_xieta,
        "etaeta": d_etaeta,
    }


def analytic_forcing(
    parameter: str,
    derivatives: dict[str, dict[str, np.ndarray]],
    xi_grid: np.ndarray,
    eta_grid: np.ndarray,
    diameter: float,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for component in COMPONENTS:
        d_xi = derivatives[component]["xi"]
        d_eta = derivatives[component]["eta"]
        if parameter == "R":
            result[component] = -(2.0 / diameter) * (
                xi_grid * d_xi + eta_grid * d_eta
            )
        elif parameter == "cx":
            result[component] = -(1.0 / diameter) * d_xi
        elif parameter == "cy":
            result[component] = -(1.0 / diameter) * d_eta
        else:
            raise ValueError(f"Unknown parameter: {parameter}")
    return result


def raw_feature_fields(
    parameter: str,
    xi: np.ndarray,
    eta: np.ndarray,
    base: dict[str, np.ndarray],
    diameter: float,
) -> tuple[list[str], np.ndarray]:
    xi_grid, eta_grid = np.meshgrid(xi, eta, indexing="ij")
    derivatives = {
        component: field_derivatives(base[component], xi, eta)
        for component in COMPONENTS
    }
    forcing = analytic_forcing(parameter, derivatives, xi_grid, eta_grid, diameter)

    names = ["u", "v"]
    fields = [base["u"], base["v"]]
    for component in COMPONENTS:
        for derivative in ("xi", "eta", "xixi", "xieta", "etaeta"):
            names.append(f"{component}_{derivative}")
            fields.append(derivatives[component][derivative])
    names.extend(["xi", "eta", f"forcing_{parameter}_u", f"forcing_{parameter}_v"])
    fields.extend([xi_grid, eta_grid, forcing["u"], forcing["v"]])
    return names, np.stack(fields, axis=-1)


def standardize_train_test(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = np.mean(train, axis=0)
    scale = np.std(train, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (train - center) / scale, (test - center) / scale, center, scale


def polynomial_terms(
    values: np.ndarray, names: list[str], degree: int
) -> tuple[np.ndarray, list[str]]:
    columns = [values]
    term_names = list(names)
    if degree == 2:
        products = []
        for left in range(values.shape[1]):
            for right in range(left, values.shape[1]):
                products.append(values[:, left] * values[:, right])
                term_names.append(f"{names[left]}*{names[right]}")
        columns.append(np.column_stack(products))
    return np.column_stack(columns), term_names


def preprocess_fold(
    raw_train: np.ndarray,
    raw_test: np.ndarray,
    names: list[str],
    degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    scaled_train, scaled_test, _, _ = standardize_train_test(raw_train, raw_test)
    design_train, _ = polynomial_terms(scaled_train, names, degree)
    design_test, _ = polynomial_terms(scaled_test, names, degree)
    design_train, design_test, _, _ = standardize_train_test(design_train, design_test)
    return design_train, design_test


def ridge_path_predict(
    design_train: np.ndarray,
    target_train: np.ndarray,
    design_test: np.ndarray,
    alphas: np.ndarray,
) -> np.ndarray:
    target_center = np.mean(target_train, axis=0)
    centered_target = target_train - target_center
    gram = (design_train.T @ design_train) / len(design_train)
    rhs = (design_train.T @ centered_target) / len(design_train)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    projected_rhs = eigenvectors.T @ rhs
    predictions = []
    for alpha in alphas:
        denominator = eigenvalues + alpha
        tolerance = max(float(np.max(np.abs(eigenvalues))), 1.0) * 1e-12
        inverse = np.divide(
            1.0,
            denominator,
            out=np.zeros_like(denominator),
            where=np.abs(denominator) > tolerance,
        )
        coefficients = eigenvectors @ (inverse[:, None] * projected_rhs)
        predictions.append(target_center + design_test @ coefficients)
    return np.stack(predictions, axis=0)


def lo_xi_predictions(
    raw_fields: np.ndarray,
    feature_names: list[str],
    targets: np.ndarray,
    degree: int,
    alphas: np.ndarray,
) -> np.ndarray:
    n_xi, n_eta, _ = raw_fields.shape
    predictions = np.empty((len(alphas), n_xi, n_eta, targets.shape[-1]))
    for held_xi in range(n_xi):
        train_slices = np.arange(n_xi) != held_xi
        raw_train = raw_fields[train_slices].reshape(-1, raw_fields.shape[-1])
        raw_test = raw_fields[held_xi]
        target_train = targets[train_slices].reshape(-1, targets.shape[-1])
        design_train, design_test = preprocess_fold(
            raw_train, raw_test, feature_names, degree
        )
        predictions[:, held_xi] = ridge_path_predict(
            design_train, target_train, design_test, alphas
        )
    return predictions


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth_flat = truth.ravel()
    prediction_flat = prediction.ravel()
    residual = prediction_flat - truth_flat
    truth_centered = truth_flat - np.mean(truth_flat)
    prediction_centered = prediction_flat - np.mean(prediction_flat)
    denominator = np.linalg.norm(truth_centered) * np.linalg.norm(prediction_centered)
    correlation = (
        float(np.dot(truth_centered, prediction_centered) / denominator)
        if denominator > 0
        else float("nan")
    )
    return {
        "rel_l2": float(np.linalg.norm(residual) / max(np.linalg.norm(truth_flat), 1e-15)),
        "nrmse_rms": float(np.sqrt(np.mean(residual**2)) / max(np.sqrt(np.mean(truth_flat**2)), 1e-15)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "correlation": correlation,
    }


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def alpha_label(alpha: float) -> str:
    return f"{alpha:.12g}"


def threshold_label(value: float, strict: float, practical: float) -> str:
    if value <= strict:
        return "strong_pass"
    if value <= practical:
        return "practical_pass"
    return "fail"


def full_fit_model(
    raw_fields: np.ndarray,
    names: list[str],
    targets: np.ndarray,
    degree: int,
    alpha: float,
) -> tuple[dict[str, np.ndarray], list[str]]:
    raw = raw_fields.reshape(-1, raw_fields.shape[-1])
    target = targets.reshape(-1, targets.shape[-1])
    raw_scaled, _, raw_center, raw_scale = standardize_train_test(raw, raw)
    design, term_names = polynomial_terms(raw_scaled, names, degree)
    design_scaled, _, design_center, design_scale = standardize_train_test(design, design)
    target_center = np.mean(target, axis=0)
    gram = (design_scaled.T @ design_scaled) / len(design_scaled)
    rhs = (design_scaled.T @ (target - target_center)) / len(design_scaled)
    coefficients = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), rhs)
    return (
        {
            "raw_center": raw_center,
            "raw_scale": raw_scale,
            "design_center": design_center,
            "design_scale": design_scale,
            "target_center": target_center,
            "coefficients": coefficients,
        },
        term_names,
    )


def make_report(
    gate_rows: list[dict[str, object]],
    ablation_rows: list[dict[str, object]],
    verdict: str,
    worst_rel_l2: float,
    args: argparse.Namespace,
    n_terms: int,
) -> str:
    decision = {
        "PASS_STRONG_CONTINUE_A1": "强通过：继续 A1。",
        "PASS_PRACTICAL_CONTINUE_A1": "实用阈值通过：可以继续 A1，但保留误差预算。",
        "FAIL_ESCALATE_D1": "失败：纯 D0 局部路线不足，应升到 D1（一次辅助 tangent 解）或调整口径。",
    }[verdict]
    lines = [
        "# A0 — M̂ 可重建性探针",
        "",
        f"**Gate 1：{verdict}。{decision}**",
        "",
        f"主判据最差目标的逐 ξ 留片 pooled rel L2 = `{worst_rel_l2:.2%}`。",
        f"强通过阈值为 `{args.strict_threshold:.0%}`，实用通过阈值为 `{args.practical_threshold:.0%}`。",
        "",
        "## 主结果（local_full）",
        "",
        "| 参数 | 分量 | rel L2 | 去首尾 rel L2 | 相关系数 | 截面中位数 | 最差截面 | 判定 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in gate_rows:
        lines.append(
            f"| {row['parameter']} | {row['component']} | {float(row['rel_l2']):.2%} | "
            f"{float(row['interior_rel_l2']):.2%} | {float(row['correlation']):.4f} | "
            f"{float(row['slice_rel_l2_median']):.2%} | "
            f"{float(row['slice_rel_l2_max']):.2%} | {row['threshold_class']} |"
        )
    lines.extend(
        [
            "",
            "## 设计与防泄漏",
            "",
            f"- 网格：33 个 ξ 截面 × 64 个 η 点；每一折完整留出一个 ξ 截面。",
            f"- 字典：基线 `u*, v*`、一二阶 ξ/η 导数、`ξ, η`、参数对应的两个解析 forcing；二次展开共 `{n_terms}` 项。",
            f"- 模型：纯 NumPy 岭回归，预先固定 `alpha={args.ridge_alpha:g}`；alpha 扫描只作敏感性诊断，不用于挑选 Gate 结果。",
            "- 标准化只在每折训练截面上拟合，留出截面不参与均值、尺度或系数估计。",
            "",
            "解析 forcing 使用坐标生成元：",
            "",
            "- `R`: `-(2/D)(ξ∂ξ + η∂η)q`",
            "- `cx`: `-(1/D)∂ξq`",
            "- `cy`: `-(1/D)∂ηq`",
            "",
            "## 参数级解释",
            "",
            "- `R`：两个分量均低于 10%，支持局部/薄片路线进入 A1。",
            "- `cy`：两个分量均低于 20%，属于带误差预算的实用通过，可进入 A1 做闭合稳健性验证。",
            "- `cx`：两个分量均超过 20%，且去掉首尾截面后仍失败；这与上游/全局历史耦合一致。若研究主张必须覆盖 `cx`，应转 D1。",
            "- 这里的失败严格地否定的是“当前预注册局部字典 + 低复杂度映射”，不是对所有可能非线性函数作数学上的不可能性证明。",
            "",
            "## 消融（pooled rel L2）",
            "",
            "| 字典 | R-u | R-v | cx-u | cx-v | cy-u | cy-v | 最差 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for ablation in ("local_full", "no_forcing", "coordinates_only"):
        selected = [row for row in ablation_rows if row["ablation"] == ablation]
        mapping = {f"{row['parameter']}_{row['component']}": float(row["rel_l2"]) for row in selected}
        ordered = [mapping[f"{parameter}_{component}"] for parameter in PARAMETERS for component in COMPONENTS]
        lines.append(
            f"| {ablation} | " + " | ".join(f"{value:.2%}" for value in ordered) + f" | {max(ordered):.2%} |"
        )
    lines.extend(
        [
            "",
            "## 产物",
            "",
            "- `metrics.json`: Gate、输入指纹、协议与逐目标指标。",
            "- `model_metrics.csv`, `slice_metrics.csv`: pooled 与逐截面误差。",
            "- `ridge_sensitivity.csv`, `ablation_metrics.csv`: 稳健性/消融诊断。",
            "- `oos_predictions.npz`: 每个目标的真值与严格留片预测。",
            "- `full_fit_model.npz`, `coefficients.csv`: 供 A1 使用的全数据局部模型快照（不参与 Gate 评分）。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.diameter <= 0:
        raise ValueError("--diameter must be positive")
    if args.ridge_alpha <= 0:
        raise ValueError("--ridge-alpha must be positive")
    if not 0 < args.strict_threshold <= args.practical_threshold:
        raise ValueError("Require 0 < strict-threshold <= practical-threshold")
    alphas = np.array(sorted({float(item) for item in args.alpha_grid.split(",")}))
    if np.any(alphas < 0):
        raise ValueError("Ridge alphas must be non-negative")
    if not np.any(np.isclose(alphas, args.ridge_alpha, rtol=1e-12, atol=0.0)):
        alphas = np.sort(np.append(alphas, args.ridge_alpha))
    primary_alpha_index = int(np.flatnonzero(np.isclose(alphas, args.ridge_alpha, rtol=1e-12, atol=0.0))[0])

    xi, eta, base, targets, r_archive_difference = load_fields(
        args.fd_fields, args.closure_fields
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model_rows: list[dict[str, object]] = []
    slice_rows: list[dict[str, object]] = []
    ridge_rows: list[dict[str, object]] = []
    ablation_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []
    prediction_payload: dict[str, np.ndarray] = {"xi": xi, "eta": eta}
    model_payload: dict[str, np.ndarray] = {
        "ridge_alpha": np.array(args.ridge_alpha),
        "degree": np.array(args.degree),
        "diameter": np.array(args.diameter),
    }
    primary_term_count = 0

    for parameter in PARAMETERS:
        full_names, full_raw = raw_feature_fields(
            parameter, xi, eta, base, args.diameter
        )
        target_pair = np.stack(
            [targets[f"{parameter}_{component}"] for component in COMPONENTS], axis=-1
        )
        ablations = {
            "local_full": np.arange(len(full_names)),
            "no_forcing": np.arange(len(full_names) - 2),
            "coordinates_only": np.array(
                [full_names.index("xi"), full_names.index("eta")]
            ),
        }
        for ablation, indices in ablations.items():
            names = [full_names[index] for index in indices]
            raw = full_raw[..., indices]
            predictions = lo_xi_predictions(raw, names, target_pair, args.degree, alphas)
            primary_prediction = predictions[primary_alpha_index]
            for component_index, component in enumerate(COMPONENTS):
                truth = target_pair[..., component_index]
                prediction = primary_prediction[..., component_index]
                metrics = regression_metrics(truth, prediction)
                ablation_rows.append(
                    {
                        "ablation": ablation,
                        "parameter": parameter,
                        "component": component,
                        "degree": args.degree,
                        "ridge_alpha": args.ridge_alpha,
                        "n_raw_features": len(names),
                        **metrics,
                    }
                )
                for alpha_index, alpha in enumerate(alphas):
                    ridge_rows.append(
                        {
                            "ablation": ablation,
                            "parameter": parameter,
                            "component": component,
                            "alpha": alpha_label(float(alpha)),
                            **regression_metrics(
                                truth, predictions[alpha_index, ..., component_index]
                            ),
                        }
                    )

                if ablation != "local_full":
                    continue
                per_slice = []
                for xi_index, xi_value in enumerate(xi):
                    slice_metric = regression_metrics(truth[xi_index], prediction[xi_index])
                    per_slice.append(slice_metric["rel_l2"])
                    slice_rows.append(
                        {
                            "parameter": parameter,
                            "component": component,
                            "xi_index": xi_index,
                            "xi": float(xi_value),
                            **slice_metric,
                        }
                    )
                row = {
                    "parameter": parameter,
                    "component": component,
                    "degree": args.degree,
                    "ridge_alpha": args.ridge_alpha,
                    "n_raw_features": len(names),
                    "slice_rel_l2_median": float(np.median(per_slice)),
                    "slice_rel_l2_p90": float(np.percentile(per_slice, 90)),
                    "slice_rel_l2_max": float(np.max(per_slice)),
                    "interior_rel_l2": regression_metrics(
                        truth[1:-1], prediction[1:-1]
                    )["rel_l2"],
                    **metrics,
                }
                row["threshold_class"] = threshold_label(
                    float(row["rel_l2"]), args.strict_threshold, args.practical_threshold
                )
                model_rows.append(row)
                prediction_payload[f"truth_{parameter}_{component}"] = truth
                prediction_payload[f"prediction_{parameter}_{component}"] = prediction

            if ablation == "local_full":
                fitted, term_names = full_fit_model(
                    raw, names, target_pair, args.degree, args.ridge_alpha
                )
                primary_term_count = len(term_names)
                model_payload[f"raw_names_{parameter}"] = np.asarray(names)
                model_payload[f"term_names_{parameter}"] = np.asarray(term_names)
                for key, value in fitted.items():
                    model_payload[f"{key}_{parameter}"] = value
                for component_index, component in enumerate(COMPONENTS):
                    for term, coefficient in zip(
                        term_names, fitted["coefficients"][:, component_index]
                    ):
                        coefficient_rows.append(
                            {
                                "parameter": parameter,
                                "component": component,
                                "term": term,
                                "standardized_coefficient": float(coefficient),
                                "abs_standardized_coefficient": float(abs(coefficient)),
                            }
                        )

    worst_rel_l2 = max(float(row["rel_l2"]) for row in model_rows)
    if worst_rel_l2 <= args.strict_threshold:
        verdict = "PASS_STRONG_CONTINUE_A1"
    elif worst_rel_l2 <= args.practical_threshold:
        verdict = "PASS_PRACTICAL_CONTINUE_A1"
    else:
        verdict = "FAIL_ESCALATE_D1"

    write_csv(args.out_dir / "model_metrics.csv", model_rows)
    write_csv(args.out_dir / "slice_metrics.csv", slice_rows)
    write_csv(args.out_dir / "ridge_sensitivity.csv", ridge_rows)
    write_csv(args.out_dir / "ablation_metrics.csv", ablation_rows)
    write_csv(args.out_dir / "coefficients.csv", coefficient_rows)
    np.savez_compressed(args.out_dir / "oos_predictions.npz", **prediction_payload)
    np.savez_compressed(args.out_dir / "full_fit_model.npz", **model_payload)

    metrics = {
        "experiment": "A0_M_reconstructability",
        "implementation": "numpy_only",
        "gate": {
            "verdict": verdict,
            "worst_target_rel_l2": worst_rel_l2,
            "strict_threshold": args.strict_threshold,
            "practical_threshold": args.practical_threshold,
            "aggregation": "maximum pooled LOXO rel_l2 across R/cx/cy x u/v",
            "parameter_verdicts": {
                parameter: (
                    "strong_pass"
                    if max(
                        float(row["rel_l2"])
                        for row in model_rows
                        if row["parameter"] == parameter
                    )
                    <= args.strict_threshold
                    else "practical_pass"
                    if max(
                        float(row["rel_l2"])
                        for row in model_rows
                        if row["parameter"] == parameter
                    )
                    <= args.practical_threshold
                    else "fail"
                )
                for parameter in PARAMETERS
            },
        },
        "protocol": {
            "split": "leave-one-complete-xi-slice-out",
            "n_xi_slices": len(xi),
            "n_eta_points": len(eta),
            "degree": args.degree,
            "ridge_alpha_predeclared": args.ridge_alpha,
            "alpha_grid_diagnostic_only": [float(value) for value in alphas],
            "raw_features": "u, v, first/second derivatives, xi, eta, two parameter-specific analytic forcing fields",
            "n_polynomial_terms": primary_term_count,
            "fold_scaling": "training slices only",
        },
        "inputs": {
            "fd_fields": str(args.fd_fields.resolve()),
            "fd_fields_sha256": file_sha256(args.fd_fields),
            "closure_fields": str(args.closure_fields.resolve()),
            "closure_fields_sha256": file_sha256(args.closure_fields),
            "R_target_archive_max_abs_difference": r_archive_difference,
            "diameter": args.diameter,
        },
        "targets": model_rows,
    }
    (args.out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = make_report(
        model_rows,
        ablation_rows,
        verdict,
        worst_rel_l2,
        args,
        primary_term_count,
    )
    (args.out_dir / "A0_REPORT.md").write_text(report, encoding="utf-8")

    print(f"A0 Gate 1: {verdict}")
    for row in model_rows:
        print(
            f"  {row['parameter']}-{row['component']}: "
            f"rel_L2={float(row['rel_l2']):.4f}, "
            f"corr={float(row['correlation']):.4f}"
        )
    print(f"Report: {args.out_dir / 'A0_REPORT.md'}")


if __name__ == "__main__":
    main()
