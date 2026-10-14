"""Optimize real-wake derivatives with a transport loss and a sparse grammar.

Free-tree PySR H1/operator losses are paired. Sparse tensor Chebyshev models
provide value/H1/operator ablations with identical dictionaries and term caps.
All choices are frozen on a new validation block before the new test is read.
This is within-case development on a previously inspected CFD field, not a new
CFD case. Run PySR through scripts/Invoke-DeeplearningPython.ps1.
"""
from __future__ import annotations

import argparse
import importlib.metadata as package_metadata
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
from numpy.polynomial.chebyshev import chebder, chebval

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline_contracts import array_hash, atomic_json, content_hash, dataset_manifest, file_hash
from experiments.wake_derivative_aware_sr import (
    inject_derivative_globals, partition_field, predict, rel_l2, scales_for,
)

BLOCKS = {"train": [(0, 6), (12, 21), (27, 33)],
          "validation": [(21, 27)], "test": [(6, 12)]}
DEGREES = [(i, j) for i in (1, 2, 3) for j in (6, 8, 10, 12)]
TERM_BUDGETS = [8, 12, 16, 20, 24]
LOSS_PATH = Path(__file__).resolve().with_name("wake_operator_loss.jl")
LOSS = LOSS_PATH.read_text(encoding="utf-8")
X, Y = sp.symbols("x y")


def transport(data, dx, dy, radius):
    return -(data["points"][:, 0] * dx + data["points"][:, 1] * dy) / radius


def field_metrics(pred, data, scales, radius, smooth=False):
    prefix = "smooth_" if smooth else ""
    tx, ty = data[prefix + "dx"], data[prefix + "dy"]
    kr = transport(data, pred["dx"], pred["dy"], radius)
    kt = transport(data, tx, ty, radius)
    raw = {"value_mse": float(np.mean((pred["q"] - data["q"]) ** 2)),
           "dx_mse": float(np.mean((pred["dx"] - tx) ** 2)),
           "dy_mse": float(np.mean((pred["dy"] - ty) ** 2)),
           "K_mse": float(np.mean((kr - kt) ** 2))}
    nv = raw["value_mse"] / scales["value"]
    ng = 0.5 * (raw["dx_mse"] / scales["dx"] + raw["dy_mse"] / scales["dy"])
    nk = raw["K_mse"] / scales["K"]
    return {**raw, "value_rel_l2": rel_l2(pred["q"], data["q"]),
            "dx_rel_l2": rel_l2(pred["dx"], tx), "dy_rel_l2": rel_l2(pred["dy"], ty),
            "gradient_rel_l2": rel_l2(np.stack([pred["dx"], pred["dy"]]), np.stack([tx, ty])),
            "K_R_rel_l2": rel_l2(kr, kt), "normalized_value": nv, "normalized_gradient": ng,
            "normalized_K": nk, "h1": nv + ng, "operator": nv + ng + nk}


def basis_arrays(features, degree_x, degree_eta):
    x, y = features[:, 0], features[:, 1] / 1.9
    bx, by, dx, dy = [], [], [], []
    for axis, n, vals, deriv, scale in ((x, degree_x, bx, dx, 1.0), (y, degree_eta, by, dy, 1.9)):
        for degree in range(n + 1):
            coeff = np.zeros(degree + 1); coeff[-1] = 1
            vals.append(chebval(axis, coeff))
            deriv.append(chebval(axis, chebder(coeff)) / scale)
    indices = [(i, j) for i in range(degree_x + 1) for j in range(degree_eta + 1)]
    return {"q": np.column_stack([bx[i] * by[j] for i, j in indices]),
            "dx": np.column_stack([dx[i] * by[j] for i, j in indices]),
            "dy": np.column_stack([bx[i] * dy[j] for i, j in indices])}, indices


def symbolic_basis(indices, coefficients):
    terms = [sp.Float(float(c), 17) * sp.chebyshevt(i, X) * sp.chebyshevt(j, Y / sp.Rational(19, 10))
             for (i, j), c in zip(indices, coefficients) if c != 0]
    return str(sp.Add(*terms))


def sparse_predict(choice, data):
    basis, indices = basis_arrays(data["features"], choice["degree_x"], choice["degree_eta"])
    coeff = np.asarray(choice["coefficients"])
    return {k: b @ coeff for k, b in basis.items()}


def design_system(basis, train, scales, radius, objective):
    n = len(train["q"])
    blocks = [basis["q"] / np.sqrt(n * scales["value"])]
    labels = [train["q"] / np.sqrt(n * scales["value"])]
    if objective != "value":
        for key in ("dx", "dy"):
            factor = np.sqrt(0.5 / (n * scales[key]))
            blocks.append(factor * basis[key]); labels.append(factor * train[key])
    if objective == "operator":
        factor = 1 / np.sqrt(n * scales["K"])
        kb = -(train["points"][:, 0, None] * basis["dx"] + train["points"][:, 1, None] * basis["dy"]) / radius
        blocks.append(factor * kb)
        labels.append(factor * transport(train, train["dx"], train["dy"], radius))
    return np.vstack(blocks), np.concatenate(labels)


def sparse_path(design, target, budgets, ridge=1e-8):
    """Forward least-squares term selection, with column scaling and tiny ridge."""
    norms = np.linalg.norm(design, axis=0)
    z = design / norms
    gram, rhs = z.T @ z, z.T @ target
    active = [0]  # Always permit a constant/background value.
    for size in range(1, min(max(budgets), len(rhs)) + 1):
        if len(active) < size:
            g = gram[np.ix_(active, active)] + ridge * np.eye(len(active))
            coeff = np.linalg.solve(g, rhs[active])
            residual_correlation = rhs - gram[:, active] @ coeff
            projection = np.linalg.solve(g, gram[active, :])
            residual_norm = np.diag(gram) - np.sum(gram[:, active] * projection.T, axis=1)
            gain = residual_correlation ** 2 / np.maximum(residual_norm, 1e-12)
            gain[active] = -np.inf
            active.append(int(np.argmax(gain)))
        g = gram[np.ix_(active, active)] + ridge * np.eye(len(active))
        beta = np.zeros(len(rhs))
        beta[active] = np.linalg.solve(g, rhs[active]) / norms[active]
        if size in budgets:
            yield size, beta


def choose_candidates(records):
    valid = [r for r in records if all(np.isfinite(r[k]) for k in ("normalized_value", "h1", "operator"))]
    if not valid:
        raise RuntimeError("No valid candidates")
    return {name: min(valid, key=lambda r: (r[metric], r["complexity"]))
            for name, metric in (("value", "normalized_value"), ("h1", "h1"), ("operator", "operator"))}


def fit_sparse(component, objective, data, scales, radius, out, protocol_id):
    path = out / f"{component}_sparse_{objective}"
    path.mkdir(exist_ok=True)
    selection = path / "selection.json"
    if selection.exists():
        fit = json.loads(selection.read_text(encoding="utf-8"))
        assert fit["protocol_id"] == protocol_id
        return fit
    start, records = time.perf_counter(), []
    for nx, ny in DEGREES:
        train_basis, indices = basis_arrays(data["train"]["features"], nx, ny)
        val_basis, _ = basis_arrays(data["validation"]["features"], nx, ny)
        a, b = design_system(train_basis, data["train"], scales, radius, objective)
        for terms, coefficients in sparse_path(a, b, TERM_BUDGETS):
            pred = {k: v @ coefficients for k, v in val_basis.items()}
            score = field_metrics(pred, data["validation"], scales, radius)
            records.append({"degree_x": nx, "degree_eta": ny, "complexity": terms, "n_terms": terms,
                            "coefficients": coefficients.tolist(), "train_loss": float(np.sum((a @ coefficients - b) ** 2)),
                            **score})
    choices = choose_candidates(records)
    for name, choice in choices.items():
        choice = choices[name] = dict(choice)
        indices = [(i, j) for i in range(choice["degree_x"] + 1) for j in range(choice["degree_eta"] + 1)]
        choice["expression"] = symbolic_basis(indices, choice["coefficients"])
        choice["expression_sha256"] = content_hash(choice["expression"])
        choice["sympy_ops"] = int(sp.count_ops(sp.sympify(choice["expression"])))
    atomic_json(path / "validation_candidates.json", records)
    pd.DataFrame([{k: v for k, v in r.items() if k != "coefficients"} for r in records]).to_csv(path / "validation_candidates.csv", index=False)
    fit = {"protocol_id": protocol_id, "component": component, "method": "sparse_" + objective,
           "seed": -1, "family": "sparse", "objective": objective, "seconds": time.perf_counter() - start,
           "choices": choices, "candidate_count": len(records)}
    atomic_json(selection, fit)
    print(f"DONE {component}_sparse_{objective}: val operator={choices['operator']['operator']:.6g}, terms={choices['operator']['n_terms']}", flush=True)
    return fit


def inject_operator(train, scales, radius, weight):
    from pysr import jl
    inject_derivative_globals(train["dx"], train["dy"], value_scale=scales["value"],
                              dx_scale=scales["dx"], dy_scale=scales["dy"], lambda_grad=1.0)
    jl.seval("global WAKE_K_TARGET, WAKE_K_SCALE, WAKE_RADIUS, WAKE_K_WEIGHT")
    jl.WAKE_K_TARGET = np.ascontiguousarray(transport(train, train["dx"], train["dy"], radius))
    jl.WAKE_K_SCALE = float(scales["K"]); jl.WAKE_RADIUS = float(radius); jl.WAKE_K_WEIGHT = float(weight)
    jl.seval("WAKE_K_TARGET=convert(Vector{Float64}, WAKE_K_TARGET)")


def fit_pysr(args, component, seed, objective, data, scales, radius, protocol_id):
    from pysr import PySRRegressor
    path = args.out_dir / f"{component}_pysr_{objective}_seed{seed}"
    path.mkdir(exist_ok=True)
    if (path / "selection.json").exists():
        fit = json.loads((path / "selection.json").read_text(encoding="utf-8"))
        assert fit["protocol_id"] == protocol_id
        return fit
    inject_operator(data["train"], scales, radius, float(objective == "operator"))
    model = PySRRegressor(niterations=args.niterations, populations=8, population_size=40,
                         ncycles_per_iteration=150, maxsize=25, binary_operators=["+", "-", "*", "/"],
                         unary_operators=["sin", "cos"], loss_function=LOSS, random_state=seed,
                         deterministic=True, parallelism="serial", precision=64, batching=False,
                         model_selection="accuracy", progress=False, verbosity=0,
                         output_directory=str(path / "search"), run_id="fit")
    print(f"START {component}_pysr_{objective}_seed{seed}", flush=True)
    start = time.perf_counter()
    model.fit(data["train"]["features"], data["train"]["q"], variable_names=["x", "y"])
    elapsed, records = time.perf_counter() - start, []
    model.equations_.to_csv(path / "equations.csv", index=False)
    for i, row in model.equations_.iterrows():
        expression = str(row.sympy_format)
        try:
            score = field_metrics(predict(expression, data["validation"]["features"]), data["validation"], scales, radius)
            if not all(np.isfinite(v) for v in score.values()):
                continue
            records.append({"equation_index": int(i), "complexity": int(row.complexity),
                            "train_loss": float(row.loss), "expression": expression,
                            "expression_sha256": content_hash(expression),
                            "sympy_ops": int(sp.count_ops(sp.sympify(expression))), "n_terms": None, **score})
        except (ValueError, FloatingPointError, ZeroDivisionError, OverflowError):
            continue
    pd.DataFrame(records).to_csv(path / "validation_candidates.csv", index=False)
    fit = {"protocol_id": protocol_id, "component": component, "method": "pysr_" + objective,
           "seed": seed, "family": "pysr", "objective": objective, "seconds": elapsed,
           "choices": choose_candidates(records), "candidate_count": len(records)}
    atomic_json(path / "selection.json", fit)
    print(f"DONE {component}_pysr_{objective}_seed{seed}: {elapsed:.1f}s, val operator={fit['choices']['operator']['operator']:.6g}", flush=True)
    return fit


def check_numerics():
    xi, eta = np.linspace(3, 5, 33), np.linspace(-1.9, 1.9, 64)
    xx, yy = np.meshgrid(xi, eta, indexing="ij")
    q = xx ** 2 + 3 * xx * yy + 2 * yy ** 2
    parts = partition_field(q, xi, eta, BLOCKS)
    supports = {s: {i for a, b in ranges for i in range(a, b)} for s, ranges in BLOCKS.items()}
    assert all(not supports[a] & supports[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")))
    changed = q.copy(); changed[sorted(supports["validation"] | supports["test"])] += 777
    other = partition_field(changed, xi, eta, BLOCKS)["train"]
    assert all(np.array_equal(other[k], parts["train"][k]) for k in other)
    basis, indices = basis_arrays(parts["train"]["features"], 2, 4)
    coeff = np.random.default_rng(173).normal(size=len(indices)) / 10
    symbolic = predict(symbolic_basis(indices, coeff), parts["train"]["features"])
    error = max(float(np.max(np.abs(basis[k] @ coeff - symbolic[k]))) for k in basis)
    assert error < 1e-11
    for s, data in parts.items():
        for i, j in data["indices"]:
            assert {i - 1, i, i + 1} <= supports[s]
        px, py = data["points"].T
        np.testing.assert_allclose(data["dx"], 2 * px + 3 * py, atol=2e-12)
        np.testing.assert_allclose(data["dy"], 3 * px + 4 * py, atol=2e-12)
    train = parts["train"]
    scales = scales_for(train); scales["K"] = float(np.mean(transport(train, train["dx"], train["dy"], .05) ** 2))
    b, _ = basis_arrays(train["features"], 2, 2)
    a, target = design_system(b, train, scales, .05, "operator")
    _, fit = list(sparse_path(a, target, [9]))[-1]
    fit_error = float(np.linalg.norm(a @ fit - target) / np.linalg.norm(target))
    assert fit_error < 1e-6
    return {"stencil_supports_disjoint": True, "heldout_mutation_does_not_change_train": True,
            "basis_vs_sympy_max_error": error, "manufactured_operator_fit_rel_error": fit_error}


def check_julia():
    from pysr import jl
    points = np.array([[3.2, -.6], [3.8, .4], [4.3, -.5], [4.7, .8]])
    features = points - np.array([4., 0.])
    expression = "sin(x)*cos(y)+0.17*x/(2+y*y)"
    exact = predict(expression, features)
    data = {"points": points, "features": features, "q": exact["q"]+.1,
            "dx": exact["dx"]-.2, "dy": exact["dy"]+.3}
    scales = {"value": .4, "dx": .6, "dy": .8, "K": 100.}
    inject_operator(data, scales, .05, 1.)
    jl.seval("using SymbolicRegression; global OP_CHECK_X, OP_CHECK_Y")
    jl.OP_CHECK_X = features.T.copy(); jl.OP_CHECK_Y = data["q"].copy()
    jl.seval(LOSS)
    actual = float(jl.seval('''begin
        op_options=SymbolicRegression.Options(binary_operators=[+,-,*,/], unary_operators=[sin,cos])
        op_tree=SymbolicRegression.parse_expression(:(sin(x)*cos(y)+0.17*x/(2+y*y)); operators=op_options.operators, variable_names=["x","y"])
        op_data=SymbolicRegression.Dataset(convert(Matrix{Float64},OP_CHECK_X),convert(Vector{Float64},OP_CHECK_Y))
        wake_operator_objective(op_tree,op_data,op_options)
    end'''))
    expected = field_metrics(exact, data, scales, .05)["operator"]
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    return {"julia": actual, "independent_python": expected, "absolute_error": abs(actual-expected)}


def prepare(args):
    manifest = dataset_manifest(args.input_dir)
    frame = pd.read_csv(args.input_dir / "wake_data.csv").sort_values(["x_D", "y_D"])
    xi, eta = np.sort(frame.x_D.unique()), np.sort(frame.y_D.unique())
    if (len(xi), len(eta)) != (33, 64):
        raise ValueError("Expected the dense 33 x 64 wake")
    meta = json.loads((args.input_dir / "meta.json").read_text(encoding="utf-8"))
    fields = {"u": frame.u_star.to_numpy().reshape(33, 64), "v": np.load(args.input_dir / "v_mean_wake.npy")}
    parts = {c: partition_field(q, xi, eta, BLOCKS) for c, q in fields.items()}
    radius, scales = float(meta["D"])/2, {}
    for c, p in parts.items():
        scales[c] = scales_for(p["train"])
        scales[c]["K"] = float(np.mean(transport(p["train"], p["train"]["dx"], p["train"]["dy"], radius) ** 2))
    files = [Path(__file__).resolve(), LOSS_PATH, ROOT / "experiments/wake_derivative_aware_sr.py",
             ROOT / "experiments/p2_analytic_derivative_loss.py", ROOT / "experiments/wake_derivative_loss.jl", ROOT / "pipeline_contracts.py"]
    protocol = {"dataset": manifest, "sources": {str(p.relative_to(ROOT)): file_hash(p) for p in files},
                "blocks": BLOCKS, "scales": scales, "radius": radius, "checks": check_numerics(),
                "settings": {"seeds": args.seeds, "niterations": args.niterations, "populations": 8,
                "population_size": 40, "ncycles_per_iteration": 150, "maxsize_pysr": 25,
                "sparse_degrees": DEGREES, "sparse_term_budgets": TERM_BUDGETS, "ridge": 1e-8,
                "selection_scores": ["value", "h1", "operator"], "coordinate_shift": [4.,0.], "eta_basis_scale": 1.9},
                "sizes": {s: len(p["q"]) for s,p in parts["u"].items()},
                "split_hashes": {c: {s: array_hash(p["points"],p["q"],p["dx"],p["dy"]) for s,p in d.items()} for c,d in parts.items()},
                "python": platform.python_version(), "packages": {k: package_metadata.version(k) for k in ("numpy","scipy","sympy","pysr","pandas")},
                "scope": "New within-run spatial holdout of previously inspected CFD; not a new CFD case. Sparse term count is not PySR tree complexity."}
    protocol["protocol_id"] = content_hash(protocol)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pp = args.out_dir / "protocol.json"
    if pp.exists():
        assert json.loads(pp.read_text(encoding="utf-8"))["protocol_id"] == protocol["protocol_id"], "Use a fresh output directory after code/settings changes"
    atomic_json(pp, protocol)
    np.savez_compressed(args.out_dir / "data_splits.npz", xi=xi, eta=eta,
                        **{f"{c}_{s}_{k}": v for c,d in parts.items() for s,p in d.items() for k,v in p.items()})
    return parts, protocol


def evaluate(args, parts, protocol, fits):
    # Called only after ALL validation choices (including dictionary settings) are frozen.
    recommended = {}
    for c in ("u", "v"):
        eligible = [f for f in fits if f["component"] == c]
        best = min(eligible, key=lambda f: f["choices"]["operator"]["operator"])
        recommended[c] = {"method": best["method"], "seed": best["seed"], "family": best["family"],
                          "objective": best["objective"], **best["choices"]["operator"]}
    atomic_json(args.out_dir / "selected_velocity_models.json",
                {"protocol_id": protocol["protocol_id"], "selection": "minimum validation operator score across all frozen fits",
                 "normalization": protocol["dataset"]["normalization"], "radius": protocol["radius"],
                 "coordinate_shift": [4.0, 0.0], "models": recommended})
    rows, checks, coeff_rows = [], [], []
    for fit in fits:
        c = fit["component"]; scales = protocol["scales"][c]
        for rule, choice in fit["choices"].items():
            assert content_hash(choice["expression"]) == choice["expression_sha256"]
            pred = predict(choice["expression"], parts[c]["test"]["features"])
            train_pred = predict(choice["expression"], parts[c]["train"]["features"])
            train_score = field_metrics(train_pred, parts[c]["train"], scales, protocol["radius"])
            key = "normalized_value" if fit["objective"] == "value" else fit["objective"]
            delta = abs(train_score[key] - choice["train_loss"])
            assert delta < 1e-7 * max(1, choice["train_loss"]), (fit["method"], delta)
            checks.append({"component": c, "method": fit["method"], "seed": fit["seed"], "selection": rule, "train_loss_error": delta})
            if fit["family"] == "sparse":
                bp = sparse_predict(choice, parts[c]["test"])
                for k in pred:
                    np.testing.assert_allclose(pred[k], bp[k], rtol=1e-8, atol=1e-8)
                if rule == "operator":
                    indices = [(i,j) for i in range(choice["degree_x"]+1) for j in range(choice["degree_eta"]+1)]
                    coeff_rows.extend({"component":c, "method":fit["method"], "i":i, "j":j, "coefficient":a}
                                      for (i,j),a in zip(indices,choice["coefficients"]) if a != 0)
            for reference, smooth in (("central_difference",False),("savgol5_stability",True)):
                rows.append({"component":c,"method":fit["method"],"seed":fit["seed"],"selection":rule,
                             "reference":reference,"family":fit["family"],"complexity":choice["complexity"],
                             "n_terms":choice["n_terms"],"sympy_ops":choice["sympy_ops"],"seconds":fit["seconds"],
                             "expression":choice["expression"], **field_metrics(pred,parts[c]["test"],scales,protocol["radius"],smooth)})
    table = pd.DataFrame(rows)
    table.to_csv(args.out_dir / "test_metrics.csv",index=False)
    pd.DataFrame(checks).to_csv(args.out_dir / "training_loss_recheck.csv",index=False)
    pd.DataFrame(coeff_rows).to_csv(args.out_dir / "selected_sparse_coefficients.csv",index=False)
    keys = ["value_mse","dx_rel_l2","dy_rel_l2","gradient_rel_l2","K_R_rel_l2","operator","complexity","sympy_ops"]
    table.groupby(["component","method","selection","reference"])[keys].agg(["mean","std"]).to_csv(args.out_dir / "summary.csv")
    atomic_json(args.out_dir / "verification.json", {"frozen_choices":len(checks),"max_rebuilt_train_loss_error":max(r["train_loss_error"] for r in checks),
                "all_expression_hashes_match":True,"all_sparse_formulas_match_matrix_predictions":True})
    atomic_json(args.out_dir / "complete.json", {"protocol_id":protocol["protocol_id"],"fits":len(fits),
                "pysr_searches":sum(f["family"]=="pysr" for f in fits),"all_selections_frozen_before_test":True,
                "completed_at":time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    write_summary(args, protocol, table)


def write_summary(args, protocol, table):
    primary=table[(table.reference=="central_difference") & (table.selection=="operator")]
    order=["pysr_h1","pysr_operator","sparse_value","sparse_h1","sparse_operator"]
    lines=["# 算子感知与稀疏结构符号回归", "", "所有主表方案均按同一验证算子分数选式：值误差＋归一化 H1＋归一化 K_R。",
           "PySR 是三个搜索种子均值；稀疏模型是确定性结果，未复制成伪三种子。",
           "本轮为新划分下的同算例空间留出，历史上已分析过此 CFD 场；不属于全新算例盲测。", "",
           "|分量|方法|速度 MSE|ξ 导数误差|η 导数误差|梯度误差|K_R 误差|",
           "|---|---|---:|---:|---:|---:|---:|"]
    for c in ("u","v"):
        for method in order:
            d=primary[(primary.component==c)&(primary.method==method)]
            lines.append(f"|{c}|{method}|{d.value_mse.mean():.6g}|{100*d.dx_rel_l2.mean():.2f}%|{100*d.dy_rel_l2.mean():.2f}%|{100*d.gradient_rel_l2.mean():.2f}%|{100*d.K_R_rel_l2.mean():.2f}%|")
    lines += ["",f"训练/验证/测试点数：{protocol['sizes']}。", "",
              "稀疏字典 q=Σ a_ij T_i(ξ−4)T_j(η/1.9)，ξ 阶数在 1/2/3、η 阶数在 6/8/10/12 中用验证集选择；8/12/16/20/24 项预算同样只用验证集选择。",
              "项数与 PySR 节点复杂度不可直接等同；实际 SymPy 操作数及各组验证候选均已保存。改结构、词典和系数求解器的收益不能全部归因于 K_R 损失。",
              "K_R=−(ξqξ+ηqη)/R，只是坐标输运项，不是总参数响应。参考来自数值 CFD 差分，savgol5_stability 为额外参考稳定性诊断。",
              "", "![测试误差](comparison.png)", "", "完整固定设置与指纹见 protocol.json，全部测试表见 test_metrics.csv，表达式与系数见各目录 selection.json。"]
    (args.out_dir/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
    colors=["#4278a8","#df8846","#979c9f","#67a483","#7563a4"]
    labels=["Tree H1","Tree + K","Sparse value","Sparse H1","Sparse + K"]
    for i,c in enumerate(("u","v")):
        for j,(key,title,scale) in enumerate((("value_mse","Velocity MSE",1),("dx_rel_l2","Xi derivative error [%]",100),("K_R_rel_l2","Radius transport error [%]",100))):
            ax=axes[i,j]
            for pos,method in enumerate(order):
                d=primary[(primary.component==c)&(primary.method==method)][key].to_numpy()*scale
                ax.bar(pos,np.mean(d),color=colors[pos],alpha=.8)
                ax.scatter(np.full(len(d),pos),d,color="black",s=15,zorder=3)
            ax.set_xticks(range(len(order)),labels,rotation=20,ha="right")
            ax.set_yscale("log"); ax.set_title(f"{c}*: {title}"); ax.grid(axis="y",alpha=.2)
    fig.suptitle("New spatial holdout; common operator-aware validation selection")
    fig.savefig(args.out_dir/"comparison.png",dpi=180); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir",type=Path,default=ROOT/"outputs_mgn/comsol/eta_sym")
    parser.add_argument("--out-dir",type=Path,default=ROOT/"outputs/wake_operator_aware_sr_v2")
    parser.add_argument("--seeds",nargs="+",type=int,default=[0,1,2])
    parser.add_argument("--niterations",type=int,default=30)
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--sparse-only",action="store_true",help="Fit dictionary candidates and freeze validation selections; do NOT test")
    parser.add_argument("--evaluate-only",action="store_true")
    args=parser.parse_args(); args.out_dir=args.out_dir.resolve()
    parts,protocol=prepare(args)
    print(json.dumps({"protocol_id":protocol["protocol_id"],"sizes":protocol["sizes"],"checks":protocol["checks"]}),flush=True)
    if args.prepare_only: return
    fits=[]
    for c in ("u","v"):
        for objective in ("value","h1","operator"):
            fits.append(fit_sparse(c,objective,parts[c],protocol["scales"][c],protocol["radius"],args.out_dir,protocol["protocol_id"]))
    if args.sparse_only: return
    if not args.evaluate_only:
        atomic_json(args.out_dir/"julia_objective_check.json",check_julia())
    for c in ("u","v"):
        for seed in args.seeds:
            for objective in ("h1","operator"):
                if args.evaluate_only:
                    fits.append(json.loads((args.out_dir/f"{c}_pysr_{objective}_seed{seed}/selection.json").read_text(encoding="utf-8")))
                else:
                    fits.append(fit_pysr(args,c,seed,objective,parts[c],protocol["scales"][c],protocol["radius"],protocol["protocol_id"]))
    evaluate(args,parts,protocol,fits)
    print("COMPLETE "+str(args.out_dir),flush=True)


if __name__=="__main__":
    main()
