"""Audit MLP-psi transport gradients against CFD data-gradient closure.

This script consumes already-trained MLP-psi sensitivity fields; it never
retrains the networks.  It compares each seed's radius transport field with
both

    K_emp  = S_total_FD - M_FD
    K_data = matched-step frozen-base CFD gradient

and verifies the conditioning identity

    closure_rel_L2 = epsilon_K_vs_empirical * kappa.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pipeline_contracts import (resolve_run_dir, selected_model, verify_fd_model, archive_provenance, dataset_manifest, evaluate, file_hash)
DEFAULT_MAIN_MLP = REPO / "outputs_mgn" / "comsol" / "eta_sym_psi_mlp"
DEFAULT_WORKTREE_MLP = (
    REPO
    / ".claude"
    / "worktrees"
    / "pinn-baseline"
    / "outputs_mgn"
    / "comsol"
    / "eta_sym_psi_mlp"
)
DEFAULT_FD = REPO / "outputs_mgn" / "fd_check"
DEFAULT_OUT = DEFAULT_FD / "mlp_psi_gradient_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlp-dir", type=Path, default=None)
    parser.add_argument("--fd-dir", type=Path, default=DEFAULT_FD)
    parser.add_argument("--psi-dir", type=Path, default=REPO / "outputs_mgn/comsol/eta_sym_psi")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    return parser.parse_args()


def resolve_mlp_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        candidate = explicit.resolve()
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        return candidate
    for candidate in (DEFAULT_MAIN_MLP, DEFAULT_WORKTREE_MLP):
        if (candidate / "sensitivity_fields_agg.npz").exists():
            return candidate
    raise FileNotFoundError(
        "No trained MLP-psi aggregate found in the main output or pinn-baseline worktree"
    )


def relative_l2(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(
        np.linalg.norm(estimate - reference)
        / max(np.linalg.norm(reference), 1e-12)
    )


def correlation(reference: np.ndarray, estimate: np.ndarray) -> float:
    a = reference.ravel() - reference.mean()
    b = estimate.ravel() - estimate.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 0 else float("nan")


def audit_row(
    label: str,
    k_field: np.ndarray,
    *,
    s_total: np.ndarray,
    m_response: np.ndarray,
    k_empirical: np.ndarray,
    k_data: np.ndarray,
    field_mse: float,
    kappa: float,
    seed: int | str,
) -> dict:
    epsilon_empirical = relative_l2(k_empirical, k_field)
    epsilon_data = relative_l2(k_data, k_field)
    reconstructed = k_field + m_response
    closure = relative_l2(s_total, reconstructed)
    return {
        "model": label,
        "seed": seed,
        "field_mse_u": field_mse,
        "epsilon_K_vs_empirical": epsilon_empirical,
        "epsilon_K_vs_data_gradient": epsilon_data,
        "kappa": kappa,
        "predicted_closure_kappa_epsilon": kappa * epsilon_empirical,
        "actual_closure_rel_l2": closure,
        "prediction_identity_abs_error": abs(kappa * epsilon_empirical - closure),
        "K_correlation_vs_empirical": correlation(k_empirical, k_field),
        "K_mean_abs": float(np.mean(np.abs(k_field))),
    }


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    order = ["psi-B", "MLP-psi", "base-CFD data gradient"]
    selected = summary.set_index("model").loc[order].reset_index()
    colors = ["#C06C84", "#355C7D", "#7FB069"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    axes[0].bar(
        ["psi-B", "MLP-psi"],
        selected.iloc[:2]["field_mse_u"],
        color=colors[:2],
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("u* field MSE (log scale)")
    axes[0].set_title("Field-value accuracy")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(
        order,
        100.0 * selected["epsilon_K_vs_empirical"],
        color=colors,
    )
    axes[1].set_ylabel("epsilon_K vs S-M [%]")
    axes[1].set_title("Transport-gradient error")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(
        order,
        100.0 * selected["actual_closure_rel_l2"],
        color=colors,
    )
    axes[2].axhspan(10, 15, color="#7FB069", alpha=0.15)
    axes[2].set_ylabel("total closure rel L2 [%]")
    axes[2].set_title("After kappa amplification")
    axes[2].tick_params(axis="x", rotation=18)
    axes[2].grid(axis="y", alpha=0.25)

    fig.suptitle("MLP-psi gradient audit: field MSE does not control sensitivity error")
    fig.savefig(out_dir / "10_mlp_gradient_error_kappa_summary.png", dpi=180)
    plt.close(fig)


def plot_fields(
    x_d: np.ndarray,
    y_d: np.ndarray,
    k_empirical: np.ndarray,
    k_data: np.ndarray,
    k_psi: np.ndarray,
    k_mlp: np.ndarray,
    out_dir: Path,
) -> None:
    fields = [
        ("K empirical = S-M", k_empirical),
        ("K base-CFD data gradient", k_data),
        ("K psi-B", k_psi),
        ("K MLP-psi (seed mean)", k_mlp),
    ]
    limit = max(float(np.max(np.abs(field))) for _, field in fields)
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5), constrained_layout=True)
    for column, (label, field) in enumerate(fields):
        image = axes[0, column].pcolormesh(
            x_d,
            y_d,
            field.T,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            shading="auto",
        )
        axes[0, column].set_title(label)
        axes[0, column].set_aspect("equal")
        axes[0, column].set_xlabel("xi")
        axes[0, column].set_ylabel("eta")
        fig.colorbar(image, ax=axes[0, column], shrink=0.82)

        residual = field - k_empirical
        residual_limit = max(float(np.max(np.abs(residual))), 1e-12)
        residual_image = axes[1, column].pcolormesh(
            x_d,
            y_d,
            residual.T,
            cmap="RdBu_r",
            vmin=-residual_limit,
            vmax=residual_limit,
            shading="auto",
        )
        axes[1, column].set_title(
            f"residual vs empirical; rel L2={relative_l2(k_empirical, field):.3f}"
        )
        axes[1, column].set_aspect("equal")
        axes[1, column].set_xlabel("xi")
        axes[1, column].set_ylabel("eta")
        fig.colorbar(residual_image, ax=axes[1, column], shrink=0.82)
    fig.suptitle("Radius transport fields and gradient residuals")
    fig.savefig(out_dir / "11_mlp_transport_field_comparison.png", dpi=180)
    plt.close(fig)


def main() -> dict:
    args = parse_args()
    args.fd_dir = resolve_run_dir(args.fd_dir)
    mlp_dir = resolve_mlp_dir(args.mlp_dir)
    mlp_data = dataset_manifest(mlp_dir)
    base_model = selected_model(args.psi_dir, "B")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (mlp_dir / "meta.json").open("r", encoding="utf-8") as handle:
        mlp_meta = json.load(handle)

    with np.load(args.fd_dir / "data_gradient_closure_fields.npz") as loaded:
        fd = {key: loaded[key].copy() for key in loaded.files}
    with np.load(args.fd_dir / "fd_sensitivity_fields.npz") as loaded:
        legacy = {key: loaded[key].copy() for key in loaded.files}

    provenance = verify_fd_model(legacy, base_model)
    closure_provenance = archive_provenance(fd)
    if provenance["fd_target_id"] != closure_provenance["fd_target_id"]:
        raise ValueError("FD and closure archives are from different runs")
    if mlp_data["grid"] != provenance["evaluation_grid"]:
        raise ValueError("MLP fields and FD evaluation grids differ; regenerate MLP evaluation fields")
    x_d, y_d = fd["x_D"], fd["y_D"]
    s_total = fd["R_u_S_total"]
    m_response = fd["R_u_M"]
    k_empirical = fd["R_u_K_empirical"]
    k_data = fd["R_u_K_data"]
    k_psi = legacy["K_R_u_psi"]
    kappa = float(np.linalg.norm(k_empirical) / np.linalg.norm(s_total))

    xx, yy = np.meshgrid(x_d, y_d, indexing="ij")
    psi_mse = float(np.mean((evaluate(base_model["sympy"], xx, yy) - legacy["u_base"])**2))
    seed_provenance = {}

    seed_rows = []
    seed_fields = []
    seed_mses = []
    for seed in args.seeds:
        seed_dir = mlp_dir / f"seed_{seed}"
        with np.load(seed_dir / "sensitivity_fields.npz") as loaded:
            k_seed = loaded["S_R_vals"].copy()
        with np.load(seed_dir / "predictions.npz") as loaded:
            prediction = loaded["u_pred"].copy()
        if k_seed.shape != s_total.shape or prediction.shape != s_total.shape:
            raise ValueError(f"Seed {seed} fields do not match the evaluation grid")
        if not np.isfinite(k_seed).all() or not np.isfinite(prediction).all():
            raise ValueError(f"Seed {seed} contains non-finite fields")
        seed_mse = float(np.mean((prediction - legacy["u_base"])**2))
        seed_provenance[str(seed)] = {"sensitivity_fields_sha256": file_hash(seed_dir / "sensitivity_fields.npz"),
                                      "predictions_sha256": file_hash(seed_dir / "predictions.npz")}
        seed_mses.append(seed_mse)
        seed_fields.append(k_seed)
        seed_rows.append(
            audit_row(
                "MLP-psi",
                k_seed,
                s_total=s_total,
                m_response=m_response,
                k_empirical=k_empirical,
                k_data=k_data,
                field_mse=seed_mse,
                kappa=kappa,
                seed=seed,
            )
        )

    k_mlp_mean = np.mean(np.stack(seed_fields, axis=0), axis=0)
    mlp_mse_mean = float(np.mean(seed_mses))
    mlp_mean_row = audit_row(
        "MLP-psi",
        k_mlp_mean,
        s_total=s_total,
        m_response=m_response,
        k_empirical=k_empirical,
        k_data=k_data,
        field_mse=mlp_mse_mean,
        kappa=kappa,
        seed="mean_field",
    )
    psi_row = audit_row(
        "psi-B",
        k_psi,
        s_total=s_total,
        m_response=m_response,
        k_empirical=k_empirical,
        k_data=k_data,
        field_mse=psi_mse,
        kappa=kappa,
        seed=base_model["expression_sha256"],
    )
    data_row = audit_row(
        "base-CFD data gradient",
        k_data,
        s_total=s_total,
        m_response=m_response,
        k_empirical=k_empirical,
        k_data=k_data,
        field_mse=float("nan"),
        kappa=kappa,
        seed="matched_step",
    )

    per_seed = pd.DataFrame(seed_rows)
    per_seed.to_csv(args.out_dir / "mlp_gradient_audit_per_seed.csv", index=False)
    summary = pd.DataFrame([psi_row, mlp_mean_row, data_row])
    summary.to_csv(args.out_dir / "mlp_gradient_audit_summary.csv", index=False)

    np.savez_compressed(
        args.out_dir / "mlp_gradient_audit_fields.npz",
        x_D=x_d,
        y_D=y_d,
        K_empirical=k_empirical,
        K_data=k_data,
        K_psi=k_psi,
        K_mlp_mean=k_mlp_mean,
        K_mlp_std=np.std(np.stack(seed_fields, axis=0), axis=0, ddof=0),
    )
    plot_summary(summary, args.out_dir)
    plot_fields(x_d, y_d, k_empirical, k_data, k_psi, k_mlp_mean, args.out_dir)

    mse_improvement = psi_mse / mlp_mse_mean
    epsilon_improvement = (
        psi_row["epsilon_K_vs_empirical"]
        / mlp_mean_row["epsilon_K_vs_empirical"]
    )
    result = {
        "base_model": base_model,
        "mlp_training_data": mlp_data,
        "evaluation_grid": provenance["evaluation_grid"],
        "target_id": provenance["fd_target_id"],
        "seed_files": seed_provenance,
        "source_mlp_dir": str(mlp_dir),
        "source_grid_revision": mlp_meta.get("grid_revision", "legacy_or_unknown"),
        "source_grid_shape": [
            int(mlp_meta["n_x_wake"]),
            int(mlp_meta["n_y"]),
        ],
        "source_training_steps": int(mlp_meta.get("steps", 0)),
        "n_seeds": len(args.seeds),
        "kappa": kappa,
        "psi_B": psi_row,
        "mlp_mean_field": mlp_mean_row,
        "base_CFD_data_gradient": data_row,
        "mlp_seed_epsilon_mean": float(per_seed["epsilon_K_vs_empirical"].mean()),
        "mlp_seed_epsilon_std": float(per_seed["epsilon_K_vs_empirical"].std(ddof=0)),
        "mlp_seed_closure_mean": float(per_seed["actual_closure_rel_l2"].mean()),
        "mlp_seed_closure_std": float(per_seed["actual_closure_rel_l2"].std(ddof=0)),
        "field_mse_improvement_psi_over_mlp": mse_improvement,
        "field_mse_improvement_orders": float(np.log10(mse_improvement)),
        "gradient_error_improvement_psi_over_mlp": epsilon_improvement,
    }
    with (args.out_dir / "mlp_gradient_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    report = [
        "# MLP-psi gradient audit",
        "",
        "The audit itself does not train networks; it consumes the sensitivity fields from "
        f"grid `{result['source_grid_revision']}` with shape "
        f"`{result['source_grid_shape'][0]} x {result['source_grid_shape'][1]}`.",
        "",
        "| model | u field MSE | epsilon_K vs S-M | epsilon_K vs data-K | kappa | predicted closure | actual closure |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in (psi_row, mlp_mean_row, data_row):
        mse_text = "-" if np.isnan(row["field_mse_u"]) else f"{row['field_mse_u']:.6g}"
        report.append(
            f"| {row['model']} | {mse_text} | {row['epsilon_K_vs_empirical']:.4f} | "
            f"{row['epsilon_K_vs_data_gradient']:.4f} | {row['kappa']:.4f} | "
            f"{row['predicted_closure_kappa_epsilon']:.4f} | "
            f"{row['actual_closure_rel_l2']:.4f} |"
        )
    report.extend(
        [
            "",
            f"MLP field MSE is {mse_improvement:.1f}x ({np.log10(mse_improvement):.2f} orders) "
            f"lower than psi-B, but epsilon_K improves only {epsilon_improvement:.2f}x.",
            f"Across seeds, epsilon_K = {per_seed['epsilon_K_vs_empirical'].mean():.4f} +/- "
            f"{per_seed['epsilon_K_vs_empirical'].std(ddof=0):.4f}; closure = "
            f"{per_seed['actual_closure_rel_l2'].mean():.4f} +/- "
            f"{per_seed['actual_closure_rel_l2'].std(ddof=0):.4f}.",
            "",
            "These errors refer to the recorded evaluation grid and FD target version. "
            "The relation closure=kappa*epsilon is an algebraic conditioning check.",
        ]
    )
    (args.out_dir / "MLP_GRADIENT_AUDIT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Outputs: {args.out_dir}")
    return result


if __name__ == "__main__":
    main()
