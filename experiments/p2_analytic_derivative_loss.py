"""P2: compare value-only and analytic expression-tree derivative losses in PySR.

The synthetic field contains a low-amplitude, high-frequency component.  It is
cheap to ignore in value MSE but expensive to ignore in spatial-gradient MSE:

    f(x, y) = sin(x) cos(y) + 0.05 sin(6 x).

Two PySR searches use the same operators and search budget.  The first optimizes
normalized value MSE.  The second additionally evaluates exact expression-tree
derivatives through SymbolicRegression.eval_diff_tree_array and optimizes a
normalized H1-like loss.  All discovered candidates are then differentiated
symbolically with SymPy and evaluated on a dense held-out grid.

This script imports PySR and must be run with scripts/Invoke-DeeplearningPython.ps1.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
from scipy.stats import spearmanr


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "outputs" / "p2_analytic_derivative_loss"


VALUE_ONLY_OBJECTIVE = r"""
function oasis_value_objective(tree, dataset::Dataset{T,L}, options) where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    flag || return L(Inf)
    value_loss = L(0.0)
    @inbounds for i in eachindex(prediction)
        err = prediction[i] - dataset.y[i]
        value_loss += err^2
    end
    value_loss /= L(length(prediction))
    return value_loss / L(OASIS_VALUE_SCALE)
end
"""


DERIVATIVE_OBJECTIVE = r"""
function oasis_derivative_objective(tree, dataset::Dataset{T,L}, options) where {T,L}
    prediction, dx, flag_x = eval_diff_tree_array(tree, dataset.X, options, 1)
    flag_x || return L(Inf)
    _, dy, flag_y = eval_diff_tree_array(tree, dataset.X, options, 2)
    flag_y || return L(Inf)

    value_loss = L(0.0)
    dx_loss = L(0.0)
    dy_loss = L(0.0)
    @inbounds for i in eachindex(prediction)
        value_err = prediction[i] - dataset.y[i]
        dx_err = dx[i] - L(OASIS_DX_TARGET[i])
        dy_err = dy[i] - L(OASIS_DY_TARGET[i])
        value_loss += value_err^2
        dx_loss += dx_err^2
        dy_loss += dy_err^2
    end
    count = L(length(prediction))
    value_loss = (value_loss / count) / L(OASIS_VALUE_SCALE)
    dx_loss = (dx_loss / count) / L(OASIS_DX_SCALE)
    dy_loss = (dy_loss / count) / L(OASIS_DY_SCALE)
    gradient_loss = L(0.5) * (dx_loss + dy_loss)
    return value_loss + L(OASIS_LAMBDA_GRAD) * gradient_loss
end
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-train", type=int, default=512)
    parser.add_argument("--test-grid", type=int, default=151)
    parser.add_argument("--niterations", type=int, default=35)
    parser.add_argument("--populations", type=int, default=12)
    parser.add_argument("--population-size", type=int, default=40)
    parser.add_argument("--maxsize", type=int, default=25)
    parser.add_argument("--lambda-grad", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def truth(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sin(x) * np.cos(y) + 0.05 * np.sin(6.0 * x)


def truth_dx(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.cos(x) * np.cos(y) + 0.30 * np.cos(6.0 * x)


def truth_dy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.sin(x) * np.sin(y)


def inject_derivative_globals(
    dx_target: np.ndarray,
    dy_target: np.ndarray,
    *,
    value_scale: float,
    dx_scale: float,
    dy_scale: float,
    lambda_grad: float,
) -> None:
    from pysr import jl

    # DynamicExpressions uses its Zygote extension for derivatives of generic
    # operators (notably division).  The package is part of the local Julia
    # project but its extension is activated only after an explicit load.
    jl.seval("using Zygote")
    jl.seval(
        "global OASIS_DX_TARGET, OASIS_DY_TARGET, OASIS_VALUE_SCALE, "
        "OASIS_DX_SCALE, OASIS_DY_SCALE, OASIS_LAMBDA_GRAD"
    )
    jl.OASIS_DX_TARGET = np.ascontiguousarray(dx_target, dtype=np.float64)
    jl.OASIS_DY_TARGET = np.ascontiguousarray(dy_target, dtype=np.float64)
    jl.OASIS_VALUE_SCALE = float(value_scale)
    jl.OASIS_DX_SCALE = float(dx_scale)
    jl.OASIS_DY_SCALE = float(dy_scale)
    jl.OASIS_LAMBDA_GRAD = float(lambda_grad)


def make_model(args: argparse.Namespace, objective: str, random_state: int):
    from pysr import PySRRegressor

    return PySRRegressor(
        niterations=args.niterations,
        populations=args.populations,
        population_size=args.population_size,
        tournament_selection_n=min(10, args.population_size - 1),
        maxsize=args.maxsize,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos"],
        loss_function=objective,
        model_selection="accuracy",
        random_state=random_state,
        deterministic=True,
        parallelism="serial",
        progress=False,
        verbosity=0,
        temp_equation_file=True,
        delete_tempfiles=False,
    )


def ensure_array(value: object, n_points: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(n_points, float(array))
    return np.broadcast_to(array, (n_points,)).astype(float, copy=False)


def candidate_metrics(
    expression_text: str,
    x_test: np.ndarray,
    y_test: np.ndarray,
    target: np.ndarray,
    dx_target: np.ndarray,
    dy_target: np.ndarray,
) -> dict[str, float]:
    x_symbol, y_symbol = sp.symbols("x y")
    expression = sp.sympify(expression_text)
    unsupported = expression.free_symbols - {x_symbol, y_symbol}
    if unsupported:
        raise ValueError(f"Unsupported symbols: {unsupported}")
    prediction_function = sp.lambdify((x_symbol, y_symbol), expression, modules="numpy")
    dx_function = sp.lambdify(
        (x_symbol, y_symbol), sp.diff(expression, x_symbol), modules="numpy"
    )
    dy_function = sp.lambdify(
        (x_symbol, y_symbol), sp.diff(expression, y_symbol), modules="numpy"
    )
    prediction = ensure_array(prediction_function(x_test, y_test), len(x_test))
    prediction_dx = ensure_array(dx_function(x_test, y_test), len(x_test))
    prediction_dy = ensure_array(dy_function(x_test, y_test), len(x_test))
    if not (
        np.all(np.isfinite(prediction))
        and np.all(np.isfinite(prediction_dx))
        and np.all(np.isfinite(prediction_dy))
    ):
        raise FloatingPointError("candidate produced non-finite test values")

    value_error = prediction - target
    dx_error = prediction_dx - dx_target
    dy_error = prediction_dy - dy_target
    gradient_numerator = np.sum(dx_error**2 + dy_error**2)
    gradient_denominator = np.sum(dx_target**2 + dy_target**2)
    return {
        "value_mse": float(np.mean(value_error**2)),
        "dx_mse": float(np.mean(dx_error**2)),
        "dy_mse": float(np.mean(dy_error**2)),
        "gradient_mse": float(0.5 * np.mean(dx_error**2 + dy_error**2)),
        "gradient_rel_l2": float(
            np.sqrt(gradient_numerator / max(gradient_denominator, 1e-30))
        ),
    }


def expression_column(equations: pd.DataFrame) -> str:
    for candidate in ("sympy_format", "sympy", "equation"):
        if candidate in equations.columns:
            return candidate
    raise KeyError(f"No expression column in {equations.columns.tolist()}")


def evaluate_pareto(
    model,
    source: str,
    x_test: np.ndarray,
    y_test: np.ndarray,
    target: np.ndarray,
    dx_target: np.ndarray,
    dy_target: np.ndarray,
    value_scale: float,
    dx_scale: float,
    dy_scale: float,
    lambda_grad: float,
) -> pd.DataFrame:
    equations = model.equations_.copy()
    column = expression_column(equations)
    rows: list[dict[str, object]] = []
    for equation_index, row in equations.iterrows():
        expression = str(row[column])
        try:
            metrics = candidate_metrics(
                expression, x_test, y_test, target, dx_target, dy_target
            )
        except (ValueError, TypeError, FloatingPointError, OverflowError) as error:
            rows.append(
                {
                    "source": source,
                    "equation_index": int(equation_index),
                    "complexity": int(row["complexity"]),
                    "train_loss": float(row["loss"]),
                    "expression": expression,
                    "status": f"skipped: {error}",
                }
            )
            continue
        normalized_value = metrics["value_mse"] / value_scale
        normalized_gradient = 0.5 * (
            metrics["dx_mse"] / dx_scale + metrics["dy_mse"] / dy_scale
        )
        rows.append(
            {
                "source": source,
                "equation_index": int(equation_index),
                "complexity": int(row["complexity"]),
                "train_loss": float(row["loss"]),
                "expression": expression,
                "status": "ok",
                **metrics,
                "normalized_value": normalized_value,
                "normalized_gradient": normalized_gradient,
                "normalized_h1_score": normalized_value
                + lambda_grad * normalized_gradient,
            }
        )
    return pd.DataFrame(rows)


def controlled_candidates(
    x_test: np.ndarray,
    y_test: np.ndarray,
    target: np.ndarray,
    dx_target: np.ndarray,
    dy_target: np.ndarray,
    value_scale: float,
    dx_scale: float,
    dy_scale: float,
    lambda_grad: float,
) -> pd.DataFrame:
    candidates = {
        "drop_low_amplitude_ripple": "sin(x)*cos(y)",
        "retain_ripple_add_bias": "sin(x)*cos(y) + 0.05*sin(6*x) + 0.04",
        "ground_truth": "sin(x)*cos(y) + 0.05*sin(6*x)",
    }
    rows: list[dict[str, object]] = []
    for name, expression in candidates.items():
        metrics = candidate_metrics(
            expression, x_test, y_test, target, dx_target, dy_target
        )
        normalized_value = metrics["value_mse"] / value_scale
        normalized_gradient = 0.5 * (
            metrics["dx_mse"] / dx_scale + metrics["dy_mse"] / dy_scale
        )
        rows.append(
            {
                "source": "controlled",
                "name": name,
                "equation_index": -1,
                "complexity": int(sp.count_ops(sp.sympify(expression)) + 1),
                "train_loss": float("nan"),
                "expression": expression,
                "status": "ok",
                **metrics,
                "normalized_value": normalized_value,
                "normalized_gradient": normalized_gradient,
                "normalized_h1_score": normalized_value
                + lambda_grad * normalized_gradient,
            }
        )
    return pd.DataFrame(rows)


def strongest_ranking_inversion(frame: pd.DataFrame) -> dict[str, object] | None:
    valid = frame[(frame["status"] == "ok") & (frame["source"] != "controlled")]
    records = valid.to_dict("records")
    best: tuple[float, dict[str, object], dict[str, object]] | None = None
    for lower_value in records:
        for lower_gradient in records:
            if (
                lower_value["value_mse"] < lower_gradient["value_mse"]
                and lower_value["gradient_mse"] > lower_gradient["gradient_mse"]
            ):
                score = np.log10(
                    max(lower_gradient["value_mse"], 1e-30)
                    / max(lower_value["value_mse"], 1e-30)
                ) + np.log10(
                    max(lower_value["gradient_mse"], 1e-30)
                    / max(lower_gradient["gradient_mse"], 1e-30)
                )
                if best is None or score > best[0]:
                    best = (float(score), lower_value, lower_gradient)
    if best is None:
        return None
    return {
        "score": best[0],
        "field_preferred": {
            key: best[1][key]
            for key in (
                "source",
                "complexity",
                "expression",
                "value_mse",
                "gradient_mse",
            )
        },
        "gradient_preferred": {
            key: best[2][key]
            for key in (
                "source",
                "complexity",
                "expression",
                "value_mse",
                "gradient_mse",
            )
        },
    }


def select_row(frame: pd.DataFrame, source: str, metric: str) -> pd.Series:
    subset = frame[(frame["source"] == source) & (frame["status"] == "ok")]
    if subset.empty:
        raise RuntimeError(f"No valid candidates for {source}")
    return subset.loc[subset[metric].idxmin()]


def plot_results(
    frame: pd.DataFrame,
    out_dir: Path,
    x_axis: np.ndarray,
    truth_cross_section: np.ndarray,
    selected_rows: list[tuple[str, pd.Series]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    markers = {
        "value_only": ("o", "#2878b5"),
        "analytic_derivative": ("s", "#d95f02"),
        "controlled": ("^", "#4daf4a"),
    }
    for source, (marker, color) in markers.items():
        subset = frame[(frame["source"] == source) & (frame["status"] == "ok")]
        if subset.empty:
            continue
        axes[0].scatter(
            subset["value_mse"],
            subset["gradient_mse"],
            s=30 + 2 * subset["complexity"],
            marker=marker,
            color=color,
            alpha=0.8,
            label=source,
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("held-out value MSE")
    axes[0].set_ylabel("held-out analytic-gradient MSE")
    axes[0].set_title("Value and derivative errors are separate objectives")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(x_axis, truth_cross_section, color="black", lw=2.2, label="truth")
    x_symbol, y_symbol = sp.symbols("x y")
    for label, row in selected_rows:
        expression = sp.sympify(str(row["expression"]))
        function = sp.lambdify((x_symbol, y_symbol), expression, modules="numpy")
        values = ensure_array(function(x_axis, np.zeros_like(x_axis)), len(x_axis))
        axes[1].plot(x_axis, values, lw=1.6, label=label)
    axes[1].set_xlabel("x at y=0")
    axes[1].set_ylabel("f(x, 0)")
    axes[1].set_title("Selected symbolic fields")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.savefig(out_dir / "p2_value_vs_gradient.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    x_train = rng.uniform(-1.0, 1.0, size=args.n_train)
    y_train = rng.uniform(-1.0, 1.0, size=args.n_train)
    features = np.column_stack([x_train, y_train])
    target_train = truth(x_train, y_train)
    dx_train = truth_dx(x_train, y_train)
    dy_train = truth_dy(x_train, y_train)

    value_scale = max(float(np.var(target_train)), 1e-12)
    dx_scale = max(float(np.mean(dx_train**2)), 1e-12)
    dy_scale = max(float(np.mean(dy_train**2)), 1e-12)
    inject_derivative_globals(
        dx_train,
        dy_train,
        value_scale=value_scale,
        dx_scale=dx_scale,
        dy_scale=dy_scale,
        lambda_grad=args.lambda_grad,
    )

    print("Fitting value-only PySR model ...")
    value_model = make_model(args, VALUE_ONLY_OBJECTIVE, args.seed)
    value_model.fit(features, target_train, variable_names=["x", "y"])

    print("Fitting analytic-derivative PySR model ...")
    derivative_model = make_model(args, DERIVATIVE_OBJECTIVE, args.seed + 1)
    derivative_model.fit(features, target_train, variable_names=["x", "y"])

    value_model.equations_.to_csv(args.out_dir / "raw_value_equations.csv", index=False)
    derivative_model.equations_.to_csv(
        args.out_dir / "raw_derivative_equations.csv", index=False
    )
    with (args.out_dir / "value_model.pkl").open("wb") as handle:
        pickle.dump(value_model, handle)
    with (args.out_dir / "derivative_model.pkl").open("wb") as handle:
        pickle.dump(derivative_model, handle)

    axis = np.linspace(-1.0, 1.0, args.test_grid)
    x_grid, y_grid = np.meshgrid(axis, axis, indexing="ij")
    x_test = x_grid.ravel()
    y_test = y_grid.ravel()
    target_test = truth(x_test, y_test)
    dx_test = truth_dx(x_test, y_test)
    dy_test = truth_dy(x_test, y_test)

    value_frame = evaluate_pareto(
        value_model,
        "value_only",
        x_test,
        y_test,
        target_test,
        dx_test,
        dy_test,
        value_scale,
        dx_scale,
        dy_scale,
        args.lambda_grad,
    )
    derivative_frame = evaluate_pareto(
        derivative_model,
        "analytic_derivative",
        x_test,
        y_test,
        target_test,
        dx_test,
        dy_test,
        value_scale,
        dx_scale,
        dy_scale,
        args.lambda_grad,
    )
    controlled_frame = controlled_candidates(
        x_test,
        y_test,
        target_test,
        dx_test,
        dy_test,
        value_scale,
        dx_scale,
        dy_scale,
        args.lambda_grad,
    )
    all_candidates = pd.concat(
        [value_frame, derivative_frame, controlled_frame], ignore_index=True
    )
    all_candidates.to_csv(args.out_dir / "candidate_metrics.csv", index=False)

    value_selected = select_row(all_candidates, "value_only", "value_mse")
    derivative_selected = select_row(
        all_candidates, "analytic_derivative", "normalized_h1_score"
    )
    controlled_drop = all_candidates[
        all_candidates.get("name", pd.Series(index=all_candidates.index, dtype=object))
        == "drop_low_amplitude_ripple"
    ].iloc[0]
    controlled_bias = all_candidates[
        all_candidates.get("name", pd.Series(index=all_candidates.index, dtype=object))
        == "retain_ripple_add_bias"
    ].iloc[0]

    valid_search = all_candidates[
        (all_candidates["status"] == "ok")
        & (all_candidates["source"] != "controlled")
    ]
    rank_correlation = spearmanr(
        valid_search["value_mse"], valid_search["gradient_mse"]
    )
    inversion = strongest_ranking_inversion(all_candidates)

    metrics = {
        "truth": "sin(x)*cos(y) + 0.05*sin(6*x)",
        "settings": {
            "n_train": args.n_train,
            "test_grid": args.test_grid,
            "niterations": args.niterations,
            "populations": args.populations,
            "population_size": args.population_size,
            "maxsize": args.maxsize,
            "lambda_grad": args.lambda_grad,
            "seed": args.seed,
            "value_scale": value_scale,
            "dx_scale": dx_scale,
            "dy_scale": dy_scale,
        },
        "selected": {
            "value_only_by_value_mse": {
                key: value_selected[key]
                for key in (
                    "complexity",
                    "expression",
                    "value_mse",
                    "gradient_mse",
                    "gradient_rel_l2",
                    "normalized_h1_score",
                )
            },
            "analytic_derivative_by_h1": {
                key: derivative_selected[key]
                for key in (
                    "complexity",
                    "expression",
                    "value_mse",
                    "gradient_mse",
                    "gradient_rel_l2",
                    "normalized_h1_score",
                )
            },
        },
        "controlled_ranking_inversion": {
            "field_preferred": {
                "name": controlled_drop.get("name"),
                "value_mse": controlled_drop["value_mse"],
                "gradient_mse": controlled_drop["gradient_mse"],
            },
            "gradient_preferred": {
                "name": controlled_bias.get("name"),
                "value_mse": controlled_bias["value_mse"],
                "gradient_mse": controlled_bias["gradient_mse"],
            },
        },
        "search_candidates_value_gradient_spearman": {
            "statistic": float(rank_correlation.statistic),
            "pvalue": float(rank_correlation.pvalue),
        },
        "strongest_search_ranking_inversion": inversion,
    }

    def json_default(value: object):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)

    (args.out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )

    plot_results(
        all_candidates,
        args.out_dir,
        axis,
        truth(axis, np.zeros_like(axis)),
        [
            ("value-only selection", value_selected),
            ("analytic-derivative selection", derivative_selected),
            ("controlled field-preferred", controlled_drop),
            ("controlled gradient-preferred", controlled_bias),
        ],
    )

    report = [
        "# P2 analytic derivative-loss experiment",
        "",
        "Both searches used the same operator set and search budget. Candidate "
        "derivatives were evaluated analytically on a dense held-out grid.",
        "",
        "| selection | complexity | value MSE | gradient MSE | gradient rel L2 |",
        "|---|---:|---:|---:|---:|",
        (
            f"| value-only / min value | {int(value_selected['complexity'])} | "
            f"{value_selected['value_mse']:.6g} | {value_selected['gradient_mse']:.6g} | "
            f"{value_selected['gradient_rel_l2']:.6g} |"
        ),
        (
            f"| analytic derivative / min H1 | {int(derivative_selected['complexity'])} | "
            f"{derivative_selected['value_mse']:.6g} | "
            f"{derivative_selected['gradient_mse']:.6g} | "
            f"{derivative_selected['gradient_rel_l2']:.6g} |"
        ),
        "",
        "## Controlled ranking inversion",
        "",
        (
            f"The field-MSE-preferred candidate has value MSE "
            f"{controlled_drop['value_mse']:.6g} and gradient MSE "
            f"{controlled_drop['gradient_mse']:.6g}."
        ),
        (
            f"The gradient-preferred candidate has the *higher* value MSE "
            f"{controlled_bias['value_mse']:.6g} but gradient MSE "
            f"{controlled_bias['gradient_mse']:.6g}."
        ),
        "",
        f"Across discovered candidates, value-MSE versus gradient-MSE Spearman "
        f"correlation is {rank_correlation.statistic:.4f}.",
    ]
    if inversion is not None:
        report.extend(
            [
                "",
                "A ranking inversion also exists within the discovered PySR candidates; "
                "see `metrics.json` for the two expressions.",
            ]
        )
    (args.out_dir / "P2_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=json_default))
    print(f"P2 outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
