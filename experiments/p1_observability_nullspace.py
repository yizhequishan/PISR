"""P1: demonstrate differential-observation nullspaces independently of PySR.

The experiment uses a fixed symbolic dictionary for a stream function.  It
computes the exact observation matrices for u = dpsi/deta and
v = -dpsi/dxi, verifies their ranks, reconstructs coefficients with ordinary
least squares, and compares random versus D-optimal sparse v measurements.

Existing PySR Pareto fronts are read only as an empirical companion; the rank
deficiency and coefficient non-identifiability do not depend on PySR.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "outputs" / "p1_observability_nullspace"

BASIS_NAMES = [
    "1",
    "sin(xi) sin(eta)",
    "xi^2",
    "exp(0.7 xi)",
    "cos(2 xi)",
    "xi eta",
    "eta^2",
]
TRUE_COEFFICIENTS = np.array([0.70, 1.20, 0.35, -0.20, 0.15, 0.25, -0.10])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--noise-relative", type=float, default=1e-3)
    parser.add_argument("--random-repeats", type=int, default=50)
    return parser.parse_args()


def basis_and_observations(
    xi: np.ndarray, eta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return basis values, d/deta basis, and -d/dxi basis."""
    values = np.column_stack(
        [
            np.ones_like(xi),
            np.sin(xi) * np.sin(eta),
            xi**2,
            np.exp(0.7 * xi),
            np.cos(2.0 * xi),
            xi * eta,
            eta**2,
        ]
    )
    obs_u = np.column_stack(
        [
            np.zeros_like(xi),
            np.sin(xi) * np.cos(eta),
            np.zeros_like(xi),
            np.zeros_like(xi),
            np.zeros_like(xi),
            xi,
            2.0 * eta,
        ]
    )
    obs_v = np.column_stack(
        [
            np.zeros_like(xi),
            -np.cos(xi) * np.sin(eta),
            -2.0 * xi,
            -0.7 * np.exp(0.7 * xi),
            2.0 * np.sin(2.0 * xi),
            -eta,
            np.zeros_like(xi),
        ]
    )
    return values, obs_u, obs_v


def numerical_rank(matrix: np.ndarray) -> tuple[int, np.ndarray, float]:
    singular = np.linalg.svd(matrix, compute_uv=False)
    tolerance = singular[0] * max(matrix.shape) * np.finfo(float).eps
    rank = int(np.sum(singular > tolerance))
    nonzero = singular[singular > tolerance]
    condition = float(nonzero[0] / nonzero[-1]) if len(nonzero) else float("inf")
    return rank, singular, condition


def fit_coefficients(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(matrix, target, rcond=None)[0]


def coefficient_error(estimate_without_gauge: np.ndarray) -> float:
    truth = TRUE_COEFFICIENTS[1:]
    return float(np.linalg.norm(estimate_without_gauge - truth) / np.linalg.norm(truth))


def greedy_d_optimal_rows(
    base_matrix: np.ndarray, candidates: np.ndarray, count: int, ridge: float = 1e-10
) -> list[int]:
    """Greedy log-det design using the matrix determinant lemma."""
    gram = base_matrix.T @ base_matrix + ridge * np.eye(base_matrix.shape[1])
    available = np.ones(len(candidates), dtype=bool)
    selected: list[int] = []
    for _ in range(count):
        inverse = np.linalg.inv(gram)
        leverage = np.einsum("ij,jk,ik->i", candidates, inverse, candidates)
        leverage[~available] = -np.inf
        index = int(np.argmax(leverage))
        selected.append(index)
        available[index] = False
        row = candidates[index]
        gram += np.outer(row, row)
    return selected


def uniform_rows(n_candidates: int, count: int) -> list[int]:
    """For each budget, cover the entire candidate order instead of its prefix."""
    if not 0 <= count <= n_candidates:
        raise ValueError("Uniform budget must lie between zero and candidate count")
    if count == 0:
        return []
    if count == 1:
        return [n_candidates // 2]
    return np.linspace(0, n_candidates - 1, count, dtype=int).tolist()


def existing_pysr_degeneracy() -> list[dict[str, object]]:
    specifications = [
        (
            "u-fit A",
            REPO / "outputs_mgn" / "comsol" / "eta_sym" / "results_strategy_a.csv",
            "has_x_D",
        ),
        (
            "u-fit B",
            REPO / "outputs_mgn" / "comsol" / "eta_sym" / "results_strategy_b.csv",
            "has_x_D",
        ),
        (
            "psi-fit u-only",
            REPO
            / "outputs_mgn"
            / "comsol"
            / "eta_sym_psi"
            / "results_strategy_a.csv",
            "has_x_D_in_psi",
        ),
        (
            "psi-fit u+v",
            REPO
            / "outputs_mgn"
            / "comsol"
            / "eta_sym_psi"
            / "results_strategy_b.csv",
            "has_x_D_in_psi",
        ),
    ]
    rows: list[dict[str, object]] = []
    for label, path, column in specifications:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        present = frame[column].astype(bool)
        rows.append(
            {
                "model": label,
                "n_formulas": int(len(frame)),
                "xi_present": int(present.sum()),
                "xi_degenerate": int((~present).sum()),
                "degeneracy_rate": float((~present).mean()),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    axis = np.linspace(-1.0, 1.0, args.grid_size)
    xi_grid, eta_grid = np.meshgrid(axis, axis, indexing="ij")
    xi = xi_grid.ravel()
    eta = eta_grid.ravel()
    basis, observation_u, observation_v = basis_and_observations(xi, eta)
    target_u = observation_u @ TRUE_COEFFICIENTS
    target_v = observation_v @ TRUE_COEFFICIENTS

    rank_u, singular_u, condition_u = numerical_rank(observation_u)
    observation_uv = np.vstack([observation_u, observation_v])
    target_uv = np.concatenate([target_u, target_v])
    rank_uv, singular_uv, condition_uv = numerical_rank(observation_uv)

    coefficients_u = fit_coefficients(observation_u, target_u)
    coefficients_uv = fit_coefficients(observation_uv, target_uv)

    x_only_columns = [2, 3, 4]
    h_coefficients = np.zeros_like(TRUE_COEFFICIENTS)
    h_coefficients[x_only_columns] = TRUE_COEFFICIENTS[x_only_columns]
    invisible_u = observation_u @ h_coefficients
    visible_v = observation_v @ h_coefficients

    # Sparse-v design excludes the constant gauge column, which no velocity
    # observation can identify.
    matrix_u = observation_u[:, 1:]
    matrix_v = observation_v[:, 1:]
    truth = TRUE_COEFFICIENTS[1:]
    scale = max(float(np.sqrt(np.mean(target_uv**2))), 1e-12)
    noise_sigma = args.noise_relative * scale
    budget_fractions = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]
    budget_counts = sorted({int(round(frac * len(matrix_v))) for frac in budget_fractions})
    max_count = max(budget_counts)
    greedy_order = greedy_d_optimal_rows(matrix_u, matrix_v, max_count)

    budget_rows: list[dict[str, object]] = []
    rng_master = np.random.default_rng(20260716)
    for count in budget_counts:
        selections = {
            "d_optimal": greedy_order[:count],
            "uniform": uniform_rows(len(matrix_v), count),
        }
        for method, selected in selections.items():
            matrix = np.vstack([matrix_u, matrix_v[selected]])
            target = np.concatenate([target_u, target_v[selected]])
            errors = []
            for _ in range(args.random_repeats):
                noisy = target + rng_master.normal(0.0, noise_sigma, size=len(target))
                errors.append(coefficient_error(fit_coefficients(matrix, noisy)))
            rank, singular, condition = numerical_rank(matrix)
            budget_rows.append(
                {
                    "method": method,
                    "n_v": count,
                    "fraction_v": count / len(matrix_v),
                    "rank_without_gauge": rank,
                    "min_singular": float(singular[-1]),
                    "condition_nonzero": condition,
                    "coefficient_error_mean": float(np.mean(errors)),
                    "coefficient_error_std": float(np.std(errors)),
                }
            )

        random_errors: list[float] = []
        random_ranks: list[int] = []
        random_min_singular: list[float] = []
        for repeat in range(args.random_repeats):
            rng = np.random.default_rng(1000 + repeat + count * 17)
            selected = (
                rng.choice(len(matrix_v), size=count, replace=False)
                if count
                else np.array([], dtype=int)
            )
            matrix = np.vstack([matrix_u, matrix_v[selected]])
            target = np.concatenate([target_u, target_v[selected]])
            noisy = target + rng.normal(0.0, noise_sigma, size=len(target))
            random_errors.append(coefficient_error(fit_coefficients(matrix, noisy)))
            rank, singular, _ = numerical_rank(matrix)
            random_ranks.append(rank)
            random_min_singular.append(float(singular[-1]))
        budget_rows.append(
            {
                "method": "random",
                "n_v": count,
                "fraction_v": count / len(matrix_v),
                "rank_without_gauge": float(np.mean(random_ranks)),
                "min_singular": float(np.mean(random_min_singular)),
                "condition_nonzero": float("nan"),
                "coefficient_error_mean": float(np.mean(random_errors)),
                "coefficient_error_std": float(np.std(random_errors)),
            }
        )

    budget_frame = pd.DataFrame(budget_rows)
    budget_frame.to_csv(args.out_dir / "sparse_v_budget.csv", index=False)

    coefficient_frame = pd.DataFrame(
        {
            "basis": BASIS_NAMES,
            "truth": TRUE_COEFFICIENTS,
            "u_only_min_norm": coefficients_u,
            "u_plus_v": coefficients_uv,
        }
    )
    coefficient_frame.to_csv(args.out_dir / "coefficient_recovery.csv", index=False)

    pysr_rows = existing_pysr_degeneracy()
    pd.DataFrame(pysr_rows).to_csv(
        args.out_dir / "existing_pysr_degeneracy.csv", index=False
    )

    metrics = {
        "sampling_protocol": "full_domain_uniform_per_budget_v2",
        "grid_size": args.grid_size,
        "n_observations_per_component": len(xi),
        "dictionary_size": len(BASIS_NAMES),
        "u_only": {
            "rank": rank_u,
            "nullity": len(BASIS_NAMES) - rank_u,
            "condition_nonzero": condition_u,
            "coefficient_error_without_gauge": coefficient_error(coefficients_u[1:]),
        },
        "u_plus_v": {
            "rank": rank_uv,
            "nullity": len(BASIS_NAMES) - rank_uv,
            "condition_nonzero": condition_uv,
            "coefficient_error_without_gauge": coefficient_error(coefficients_uv[1:]),
        },
        "x_only_perturbation": {
            "max_abs_change_in_u": float(np.max(np.abs(invisible_u))),
            "rms_change_in_v": float(np.sqrt(np.mean(visible_v**2))),
        },
        "noise_relative": args.noise_relative,
        "existing_pysr": pysr_rows,
    }
    (args.out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    max_singular = max(len(singular_u), len(singular_uv))
    indices = np.arange(1, max_singular + 1)
    axes[0].semilogy(indices[: len(singular_u)], singular_u, "o-", label="u only")
    axes[0].semilogy(indices[: len(singular_uv)], singular_uv, "s-", label="u + v")
    axes[0].set_xlabel("singular-value index")
    axes[0].set_ylabel("singular value")
    axes[0].set_title("Observation-matrix spectrum")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    positions = np.arange(len(BASIS_NAMES))
    width = 0.25
    axes[1].bar(positions - width, TRUE_COEFFICIENTS, width, label="truth")
    axes[1].bar(positions, coefficients_u, width, label="u only")
    axes[1].bar(positions + width, coefficients_uv, width, label="u + v")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(BASIS_NAMES, rotation=55, ha="right")
    axes[1].set_title("Minimum-norm coefficient recovery")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    for method, style in [("random", "o-"), ("uniform", "^-"), ("d_optimal", "s-")]:
        subset = budget_frame[budget_frame["method"] == method].sort_values("n_v")
        axes[2].errorbar(
            100.0 * subset["fraction_v"],
            subset["coefficient_error_mean"],
            yerr=subset["coefficient_error_std"],
            fmt=style,
            capsize=2,
            label=method,
        )
    axes[2].set_yscale("log")
    axes[2].set_xlabel("observed v points (%)")
    axes[2].set_ylabel("relative coefficient error")
    axes[2].set_title("Sparse complementary observations")
    axes[2].legend()
    axes[2].grid(alpha=0.25)
    fig.savefig(args.out_dir / "p1_observability.png", dpi=180)
    plt.close(fig)

    report_lines = [
        "# P1 observability/nullspace experiment",
        "",
        f"- u-only observation matrix: rank {rank_u}/{len(BASIS_NAMES)}, "
        f"nullity {len(BASIS_NAMES) - rank_u}.",
        f"- u+v observation matrix: rank {rank_uv}/{len(BASIS_NAMES)}, "
        f"nullity {len(BASIS_NAMES) - rank_uv}; the remaining direction is the "
        "constant stream-function gauge.",
        f"- Adding the chosen x-only h(xi) changes u by at most "
        f"{np.max(np.abs(invisible_u)):.3e}, but changes v with RMS "
        f"{np.sqrt(np.mean(visible_v**2)):.3e}.",
        f"- Minimum-norm coefficient error (excluding gauge): u-only "
        f"{coefficient_error(coefficients_u[1:]):.3e}, u+v "
        f"{coefficient_error(coefficients_uv[1:]):.3e}.",
        "",
        "These statements follow from the observation matrices and ordinary least "
        "squares, so they are independent of PySR's search behavior.",
        "",
        "## Existing PySR companion results",
        "",
        "| model | formulas | xi-degenerate | rate |",
        "|---|---:|---:|---:|",
    ]
    for row in pysr_rows:
        report_lines.append(
            f"| {row['model']} | {row['n_formulas']} | {row['xi_degenerate']} | "
            f"{row['degeneracy_rate']:.3f} |"
        )
    (args.out_dir / "P1_REPORT.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"P1 outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
