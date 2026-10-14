"""Paired Sobolev-loss PySR experiment on the real, dense CFD wake.

Fits u* and v* separately. This is an isolated test of first spatial derivatives,
not a stream-function fit or an identification of total parameter responses.
Run with scripts/Invoke-DeeplearningPython.ps1 outside the sandbox.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline_contracts import array_hash, atomic_json, content_hash, dataset_manifest, file_hash
from experiments.p2_analytic_derivative_loss import (
    DERIVATIVE_OBJECTIVE, VALUE_ONLY_OBJECTIVE, ensure_array, inject_derivative_globals,
)

# All CFD observations used in any derivative stencil belong to the same split.
# End indices are exclusive. The primary labels use centered 3-point differences.
BLOCKS = {"train": [(0, 8), (13, 20), (26, 33)],
          "validation": [(8, 13)], "test": [(20, 26)]}
X, Y = sp.symbols("x y")


def partition_field(q, xi, eta, blocks=BLOCKS):
    result = {}
    for split, intervals in blocks.items():
        pieces = []
        for lo, hi in intervals:
            z = q[lo:hi]
            gx, gy = np.gradient(z, xi[lo:hi], eta, edge_order=2)
            smooth = savgol_filter(savgol_filter(z, 5, 3, axis=0), 5, 3, axis=1)
            sx, sy = np.gradient(smooth, xi[lo:hi], eta, edge_order=2)
            ii, jj = np.meshgrid(np.arange(lo + 1, hi - 1),
                                 np.arange(1, len(eta) - 1), indexing="ij")
            pieces.append({"points": np.column_stack([xi[ii.ravel()], eta[jj.ravel()]]),
                           "indices": np.column_stack([ii.ravel(), jj.ravel()]),
                           "q": z[1:-1, 1:-1].ravel(),
                           "dx": gx[1:-1, 1:-1].ravel(),
                           "dy": gy[1:-1, 1:-1].ravel(),
                           "smooth_dx": sx[1:-1, 1:-1].ravel(),
                           "smooth_dy": sy[1:-1, 1:-1].ravel()})
        result[split] = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
        result[split]["features"] = result[split]["points"] - np.array([4.0, 0.0])
    return result


def scales_for(train):
    return {"value": max(float(np.var(train["q"])), 1e-12),
            "dx": max(float(np.mean(train["dx"] ** 2)), 1e-12),
            "dy": max(float(np.mean(train["dy"] ** 2)), 1e-12)}


def predict(expression, features):
    expr = sp.sympify(expression)
    if expr.free_symbols - {X, Y}:
        raise ValueError("Unexpected variables in expression")
    functions = [sp.lambdify((X, Y), e, "numpy")
                 for e in (expr, sp.diff(expr, X), sp.diff(expr, Y))]
    with np.errstate(all="ignore"):
        fields = [ensure_array(f(*features.T), len(features)) for f in functions]
    if not all(np.isfinite(f).all() for f in fields):
        raise FloatingPointError("Non-finite symbolic field or derivative")
    return dict(zip(("q", "dx", "dy"), fields))


def rel_l2(pred, ref):
    return float(np.linalg.norm(pred - ref) / max(np.linalg.norm(ref), 1e-30))


def metrics(expression, data, scales, radius, smoothed=False):
    pred = predict(expression, data["features"])
    prefix = "smooth_" if smoothed else ""
    dx, dy = data[prefix + "dx"], data[prefix + "dy"]
    mse = {"value_mse": float(np.mean((pred["q"] - data["q"]) ** 2)),
           "dx_mse": float(np.mean((pred["dx"] - dx) ** 2)),
           "dy_mse": float(np.mean((pred["dy"] - dy) ** 2))}
    val = mse["value_mse"] / scales["value"]
    grad = 0.5 * (mse["dx_mse"] / scales["dx"] + mse["dy_mse"] / scales["dy"])
    xi, eta = data["points"].T
    transport = -(xi * pred["dx"] + eta * pred["dy"]) / radius
    reference = -(xi * dx + eta * dy) / radius
    return {**mse, "value_rel_l2": rel_l2(pred["q"], data["q"]),
            "dx_rel_l2": rel_l2(pred["dx"], dx), "dy_rel_l2": rel_l2(pred["dy"], dy),
            "gradient_rel_l2": rel_l2(np.stack([pred["dx"], pred["dy"]]), np.stack([dx, dy])),
            "gradient_mse": 0.5 * (mse["dx_mse"] + mse["dy_mse"]),
            "K_R_rel_l2": rel_l2(transport, reference),
            "normalized_value": val, "normalized_gradient": grad, "h1": val + grad}


def verify_protocol():
    xi, eta = np.linspace(3, 5, 33), np.linspace(-1.9, 1.9, 64)
    xx, yy = np.meshgrid(xi, eta, indexing="ij")
    q = xx ** 2 + 3 * xx * yy + 2 * yy ** 2
    parts = partition_field(q, xi, eta)
    supports = {k: {i for lo, hi in ranges for i in range(lo, hi)}
                for k, ranges in BLOCKS.items()}
    assert not (supports["train"] & (supports["validation"] | supports["test"]))
    assert not supports["validation"] & supports["test"]
    errors = []
    for name, data in parts.items():
        px, py = data["points"].T
        errors.extend([float(np.max(np.abs(data["dx"] - (2 * px + 3 * py)))),
                       float(np.max(np.abs(data["dy"] - (3 * px + 4 * py))))])
        for i, j in data["indices"]:
            assert {i - 1, i, i + 1} <= supports[name]
        m = metrics("(x+4)**2+3*(x+4)*y+2*y**2", data, scales_for(parts["train"]), 0.05)
        assert max(m[k] for k in ("value_rel_l2", "dx_rel_l2", "dy_rel_l2", "K_R_rel_l2")) < 1e-12
    changed = q.copy()
    changed[sorted(supports["validation"] | supports["test"])] += 1234
    other = partition_field(changed, xi, eta)["train"]
    assert all(np.array_equal(other[k], parts["train"][k]) for k in other)
    return {"quadratic_max_abs_derivative_error": max(errors),
            "split_stencil_supports_disjoint": True,
            "heldout_value_mutation_leaves_training_unchanged": True}


def julia_objective_check():
    """Compare the actual Julia loss and tree derivatives to independent SymPy."""
    from pysr import jl
    pts = np.array([[-0.8, -0.6], [-0.2, 0.4], [0.3, -0.5], [0.7, 0.8]])
    expr = "sin(x)*cos(y)+0.17*x/(2+y*y)"
    exact = predict(expr, pts)
    targets = {"q": exact["q"] + 0.1, "dx": exact["dx"] - 0.2, "dy": exact["dy"] + 0.3}
    inject_derivative_globals(targets["dx"], targets["dy"], value_scale=0.4,
                              dx_scale=0.6, dy_scale=0.8, lambda_grad=1.0)
    jl.seval("using SymbolicRegression; global WAKE_CHECK_X, WAKE_CHECK_Y")
    jl.WAKE_CHECK_X = pts.T.copy()
    jl.WAKE_CHECK_Y = targets["q"].copy()
    jl.seval(VALUE_ONLY_OBJECTIVE)
    jl.seval(DERIVATIVE_OBJECTIVE)
    result = jl.seval('''begin
        using SymbolicRegression
        check_options = SymbolicRegression.Options(binary_operators=[+, -, *, /], unary_operators=[sin, cos])
        check_tree = SymbolicRegression.parse_expression(:(sin(x)*cos(y)+0.17*x/(2+y*y));
            operators=check_options.operators, variable_names=["x", "y"])
        check_dataset = SymbolicRegression.Dataset(convert(Matrix{Float64}, WAKE_CHECK_X), convert(Vector{Float64}, WAKE_CHECK_Y))
        [oasis_value_objective(check_tree, check_dataset, check_options),
         oasis_derivative_objective(check_tree, check_dataset, check_options)]
    end''')
    expected = np.array([0.1 ** 2 / 0.4, 0.1 ** 2 / 0.4 + 0.5 * (0.2 ** 2 / 0.6 + 0.3 ** 2 / 0.8)])
    actual = np.array(result)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    return {"julia": actual.tolist(), "sympy_expected": expected.tolist(),
            "max_absolute_error": float(np.max(np.abs(actual - expected)))}


def prepare(args):
    manifest = dataset_manifest(args.input_dir)
    frame = pd.read_csv(args.input_dir / "wake_data.csv").sort_values(["x_D", "y_D"])
    xi, eta = np.sort(frame.x_D.unique()), np.sort(frame.y_D.unique())
    if len(xi) != 33 or len(eta) != 64:
        raise ValueError("This preregistered blocked experiment requires the 33 x 64 dense wake")
    meta = json.loads((args.input_dir / "meta.json").read_text(encoding="utf-8"))
    fields = {"u": frame.u_star.to_numpy().reshape(33, 64),
              "v": np.load(args.input_dir / "v_mean_wake.npy")}
    parts = {c: partition_field(fields[c], xi, eta) for c in args.components}
    settings = {k: getattr(args, k) for k in ("seeds", "components", "niterations", "populations",
                "population_size", "maxsize", "ncycles_per_iteration")}
    settings.update({"lambdas": [0.0, 1.0], "precision": 64, "parallelism": "serial",
                     "operators": ["+", "-", "*", "/", "sin", "cos"],
                     "selection": ["normalized_value", "h1"], "coordinate_shift": [4.0, 0.0]})
    protocol = {"dataset": manifest, "settings": settings, "raw_x_blocks": BLOCKS,
                "radius": float(meta["D"]) / 2, "checks": verify_protocol(),
                "python": platform.python_version(),
                "source_sha256": file_hash(Path(__file__)),
                "objective_source_sha256": file_hash(ROOT / "experiments/p2_analytic_derivative_loss.py"),
                "scales": {c: scales_for(parts[c]["train"]) for c in parts},
                "split_sizes": {s: len(d["q"]) for s, d in parts[args.components[0]].items()},
                "split_sha256": {c: {s: array_hash(d["points"], d["q"], d["dx"], d["dy"])
                                      for s, d in p.items()} for c, p in parts.items()},
                "scope": "Separate scalar velocity SR; within-case spatial holdout; numerical derivative references; no hard incompressibility or total parameter-response claim"}
    protocol["protocol_id"] = content_hash(protocol)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8"))["protocol_id"] != protocol["protocol_id"]:
        raise ValueError("Output directory belongs to a different protocol; use a new --out-dir")
    atomic_json(path, protocol)
    np.savez_compressed(args.out_dir / "data_splits.npz", xi=xi, eta=eta,
                        **{f"{c}_{s}_{k}": v for c, p in parts.items() for s, d in p.items() for k, v in d.items()})
    return parts, protocol


def run_fit(args, component, seed, weight, data, protocol):
    from pysr import PySRRegressor
    label = f"{component}_seed{seed}_lambda{weight:g}"
    out = args.out_dir / label
    out.mkdir(exist_ok=True)
    completed = out / "selection.json"
    if completed.exists():
        saved = json.loads(completed.read_text(encoding="utf-8"))
        if saved["protocol_id"] != protocol["protocol_id"]:
            raise ValueError("Saved fit protocol mismatch")
        print("Using completed " + label, flush=True)
        return saved
    scales = protocol["scales"][component]
    train = data["train"]
    inject_derivative_globals(train["dx"], train["dy"], value_scale=scales["value"],
                              dx_scale=scales["dx"], dy_scale=scales["dy"], lambda_grad=weight)
    model = PySRRegressor(niterations=args.niterations, populations=args.populations,
                         population_size=args.population_size, maxsize=args.maxsize,
                         ncycles_per_iteration=args.ncycles_per_iteration,
                         binary_operators=["+", "-", "*", "/"], unary_operators=["sin", "cos"],
                         loss_function=VALUE_ONLY_OBJECTIVE if weight == 0 else DERIVATIVE_OBJECTIVE,
                         random_state=seed, deterministic=True, parallelism="serial",
                         precision=64, batching=False, model_selection="accuracy",
                         progress=False, verbosity=0, output_directory=str(out / "search"), run_id="fit")
    print(f"START {label}", flush=True)
    start = time.perf_counter()
    model.fit(train["features"], train["q"], variable_names=["x", "y"])
    elapsed = time.perf_counter() - start
    model.equations_.to_csv(out / "equations.csv", index=False)
    candidates = []
    for i, row in model.equations_.iterrows():
        entry = {"equation_index": int(i), "complexity": int(row["complexity"]),
                 "train_loss": float(row["loss"]), "expression": str(row["sympy_format"])}
        try:
            entry.update(metrics(entry["expression"], data["validation"], scales, protocol["radius"]))
            entry["status"] = "ok"
        except (ValueError, TypeError, FloatingPointError, OverflowError, ZeroDivisionError) as exc:
            entry["status"] = str(exc)
        candidates.append(entry)
    table = pd.DataFrame(candidates)
    table.to_csv(out / "validation_candidates.csv", index=False)
    valid = table[table.status == "ok"]
    if valid.empty:
        raise RuntimeError("No finite validation candidates")
    choices = {name: valid.sort_values([metric, "complexity"]).iloc[0].to_dict()
               for name, metric in (("value", "normalized_value"), ("h1", "h1"))}
    for choice in choices.values():
        choice["expression_sha256"] = content_hash(choice["expression"])
    saved = {"protocol_id": protocol["protocol_id"], "component": component,
             "seed": seed, "lambda_grad": weight, "seconds": elapsed, "choices": choices}
    atomic_json(completed, saved)  # Freeze validation choices before any test evaluation.
    print(f"DONE {label}: {elapsed:.1f}s; validation H1={choices['h1']['h1']:.6g}", flush=True)
    return saved


def evaluate_selections(args, parts, protocol, selections):
    rows, saved_fields = [], {}
    for fit in selections:
        c, seed, weight = fit["component"], fit["seed"], fit["lambda_grad"]
        for selection, choice in fit["choices"].items():
            data = parts[c]["test"]
            base = {"component": c, "seed": seed, "lambda_grad": weight, "selection": selection,
                    "complexity": choice["complexity"], "expression": choice["expression"],
                    "seconds": fit["seconds"], "validation_score": choice["h1" if selection == "h1" else "normalized_value"]}
            # Failure on test propagates; never replace the frozen expression.
            for ref, smooth in (("central_difference", False), ("savgol5_stability", True)):
                rows.append({**base, "reference": ref,
                             **metrics(choice["expression"], data, protocol["scales"][c], protocol["radius"], smooth)})
            saved_fields.update({f"{c}_{seed}_{weight:g}_{selection}_{k}": v
                                 for k, v in predict(choice["expression"], data["features"]).items()})
    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "test_metrics.csv", index=False)
    np.savez_compressed(args.out_dir / "test_predictions.npz", **saved_fields)
    keys = ["value_mse", "value_rel_l2", "dx_rel_l2", "dy_rel_l2", "gradient_mse", "gradient_rel_l2", "K_R_rel_l2", "complexity", "seconds"]
    summary = table.groupby(["component", "lambda_grad", "selection", "reference"])[keys].agg(["mean", "std"])
    summary.to_csv(args.out_dir / "summary.csv")
    comparisons = []
    for c in args.components:
        for ref in table.reference.unique():
            for criterion in ("own", "value", "h1"):
                for seed in args.seeds:
                    subset = table[(table.component == c) & (table.seed == seed) & (table.reference == ref)]
                    a = subset[(subset.lambda_grad == 0) & (subset.selection == ("value" if criterion == "own" else criterion))].iloc[0]
                    b = subset[(subset.lambda_grad == 1) & (subset.selection == ("h1" if criterion == "own" else criterion))].iloc[0]
                    comparisons.append({"component": c, "seed": seed, "reference": ref, "selection_control": criterion,
                                        **{k + "_change_pct": 100 * (b[k] / max(a[k], 1e-30) - 1) for k in keys[:-2]}})
    pd.DataFrame(comparisons).to_csv(args.out_dir / "paired_changes.csv", index=False)
    write_report(args, protocol, table, comparisons)
    plot_results(args, parts, table)
    atomic_json(args.out_dir / "complete.json", {"protocol_id": protocol["protocol_id"], "fits": len(selections),
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "all_test_choices_frozen_on_validation": True})


def write_report(args, protocol, table, comparisons):
    main = table[(table.reference == "central_difference") &
                 (((table.lambda_grad == 0) & (table.selection == "value")) |
                  ((table.lambda_grad == 1) & (table.selection == "h1")))]
    lines = ["# 真实尾流导数感知符号回归对照", "",
             "分别拟合 u*、v*；模型的导数由表达式树解析求值，测试时独立用 SymPy 求导。",
             "导数监督是数值 CFD 参考，不是真实连续解的精确导数。", "",
             "## 固定协议", "", "```json", json.dumps(protocol["settings"], indent=2), "```", "",
             f"训练/验证/测试点数：{protocol['split_sizes']}。原始列分区：{BLOCKS}（左闭右开）。",
             "各分区端点各去掉一列/行；三点中心差分模板不跨分区，归一化尺度仅来自训练集。",
             "连续测试块检验同一 CFD 算例内的空间插值；不代表跨几何泛化。",
             "损失为 L=MSE(q)/Var_train(q)+λ/2·[MSE(qξ)/mean_train(qξ²)+MSE(qη)/mean_train(qη²)]，λ=0 或 1。",
             "两组算子、复杂度、迭代数、种群和种子相同；解析导数增加每次求值成本，故不等同于相同耗时。",
             "在独立验证块分别按值误差、H1选式并保存，再一次性评价测试块。主对照为 λ=0 按值选式与 λ=1 按 H1 选式。", "",
             "## 测试结果（种子均值 ± 样本标准差）", "",
             "|分量|训练|速度 MSE|ξ 导数相对 L2|η 导数相对 L2|合并梯度相对 L2|K_R 相对 L2|",
             "|---|---|---:|---:|---:|---:|---:|"]
    for (c, w), group in main.groupby(["component", "lambda_grad"]):
        def cell(k, percent=True):
            factor = 100 if percent else 1
            suffix = "%" if percent else ""
            return f"{factor*group[k].mean():.4g} ± {factor*group[k].std():.3g}{suffix}"
        lines.append(f"|{c}|{'仅值' if w == 0 else '值+导数'}|{cell('value_mse', False)}|{cell('dx_rel_l2')}|{cell('dy_rel_l2')}|{cell('gradient_rel_l2')}|{cell('K_R_rel_l2')}|")
    lines += ["", "## 配对变化（负值表示误差下降）", "",
              "|分量|种子|速度 MSE 变化|梯度相对 L2 变化|K_R 相对 L2 变化|", "|---|---:|---:|---:|---:|"]
    for row in comparisons:
        if row["reference"] == "central_difference" and row["selection_control"] == "own":
            lines.append(f"|{row['component']}|{row['seed']}|{row['value_mse_change_pct']:+.2f}%|{row['gradient_rel_l2_change_pct']:+.2f}%|{row['K_R_rel_l2_change_pct']:+.2f}%|")
    lines += ["", "`paired_changes.csv` 同时保存相同选式准则的交叉对照，区分搜索与选式的贡献。",
              "`test_metrics.csv` 的 savgol5_stability 是分区内 5 点三阶 Savitzky–Golay 平滑后的导数参考敏感性检查，不参与训练、选式或调参。", "",
              "K_R=−(ξ qξ+η qη)/R，此处评价坐标输运项相对数值梯度参考的误差；没有用它替代总参数响应 T=K+M。",
              "这是直接速度符号回归，没有施加流函数无散硬约束。只有一份 CFD 快照和三个搜索种子，结果属探索性证据。",
              "未把历史流函数模型与本轮不同训练协议直接作因果比较。", "",
              "![测试对照](comparison.png)", "", "## 复现", "", "```powershell",
              ".\\scripts\\Invoke-DeeplearningPython.ps1 -ScriptPath .\\experiments\\wake_derivative_aware_sr.py --out-dir " + str(args.out_dir.relative_to(ROOT)).replace('\\', '/'),
              "```", "", "数据、代码和协议指纹见 protocol.json；完整公式见各 fit 的 selection.json。"]
    (args.out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(args, parts, table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    main = table[(table.reference == "central_difference") &
                 (((table.lambda_grad == 0) & (table.selection == "value")) |
                  ((table.lambda_grad == 1) & (table.selection == "h1")))]
    fig, axes = plt.subplots(len(args.components), 3, figsize=(12, 3.6 * len(args.components)), squeeze=False, constrained_layout=True)
    for i, c in enumerate(args.components):
        for j, (key, label) in enumerate((("value_mse", "Velocity MSE"), ("gradient_rel_l2", "Gradient relative L2"), ("K_R_rel_l2", "Radius transport relative L2"))):
            ax = axes[i, j]
            for seed in args.seeds:
                group = main[(main.component == c) & (main.seed == seed)].sort_values("lambda_grad")
                ax.plot([0, 1], group[key], "o-", label=f"seed {seed}", alpha=0.85)
            ax.set_xticks([0, 1], ["Value only", "Value + derivative"])
            ax.set_title(f"{c}*: {label}")
            ax.set_yscale("log")
            ax.grid(alpha=0.2)
            if j == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Paired searches; blocked spatial test; validation-frozen expressions")
    fig.savefig(args.out_dir / "comparison.png", dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=ROOT / "outputs_mgn/comsol/eta_sym")
    p.add_argument("--out-dir", type=Path, default=ROOT / "outputs/wake_derivative_aware_sr_2026-09-28")
    p.add_argument("--components", nargs="+", choices=["u", "v"], default=["u", "v"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--niterations", type=int, default=30)
    p.add_argument("--populations", type=int, default=8)
    p.add_argument("--population-size", type=int, default=40)
    p.add_argument("--maxsize", type=int, default=25)
    p.add_argument("--ncycles-per-iteration", type=int, default=150)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--evaluate-only", action="store_true")
    args = p.parse_args()
    args.out_dir = args.out_dir.resolve()
    parts, protocol = prepare(args)
    print(json.dumps({"protocol_id": protocol["protocol_id"], "split_sizes": protocol["split_sizes"], "checks": protocol["checks"]}), flush=True)
    if args.prepare_only:
        return
    if not args.evaluate_only:
        atomic_json(args.out_dir / "julia_objective_check.json", julia_objective_check())
    selections = []
    for c in args.components:
        for seed in args.seeds:
            for weight in (0.0, 1.0):
                if args.evaluate_only:
                    selections.append(json.loads((args.out_dir / f"{c}_seed{seed}_lambda{weight:g}/selection.json").read_text(encoding="utf-8")))
                else:
                    selections.append(run_fit(args, c, seed, weight, parts[c], protocol))
    evaluate_selections(args, parts, protocol, selections)
    print("COMPLETE " + str(args.out_dir), flush=True)


if __name__ == "__main__":
    main()
