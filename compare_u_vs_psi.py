"""Compare u-fit baseline vs psi-fit results.

Reads sensitivity_results.csv from two run dirs (after step4 has been run on each)
and prints a side-by-side metric table covering xi-degeneracy rate, S_cx magnitude,
and identity check residual.
"""
import argparse
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-dir",
        default=r"D:\keyan\cylinder_flow_pisr\outputs_mgn\comsol\eta_sym",
        help="Run dir with u-fit results.",
    )
    parser.add_argument(
        "--psi-dir",
        default=r"D:\keyan\cylinder_flow_pisr\outputs_mgn\comsol\eta_sym_psi",
        help="Run dir with psi-fit results.",
    )
    return parser.parse_args()


def summarize(sens_df, label):
    rows = []
    for strategy in sorted(sens_df["strategy"].unique()):
        sub = sens_df[sens_df["strategy"] == strategy]
        n_total = len(sub)
        n_degen = int((sub["S_cx_integrated"] < 1e-10).sum())
        max_cx = float(sub["S_cx_integrated"].max())
        # best by mse
        best = sub.sort_values("mse").iloc[0]
        rows.append(
            {
                "label": label,
                "strategy": strategy,
                "n_total": n_total,
                "n_xi_degen": n_degen,
                "degen_rate": n_degen / max(n_total, 1),
                "max_S_cx_mean": max_cx,
                "best_complexity": int(best["complexity"]),
                "best_mse": float(best["mse"]),
                "best_S_R_mean": float(best["S_R_integrated"]),
                "best_S_cx_mean": float(best["S_cx_integrated"]),
                "best_S_cy_mean": float(best["S_cy_integrated"]),
                "best_identity_relL2": float(best.get("identity_rel_l2", float("nan"))),
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    base_path = os.path.join(args.baseline_dir, "sensitivity_results.csv")
    psi_path = os.path.join(args.psi_dir, "sensitivity_results.csv")

    if not os.path.exists(base_path):
        raise SystemExit(f"Baseline sensitivity file missing: {base_path}")
    if not os.path.exists(psi_path):
        raise SystemExit(
            f"Psi sensitivity file missing: {psi_path}\n"
            "Run step4 on the psi run first: "
            f"python step4_euler_sensitivity.py --out-dir {args.psi_dir}"
        )

    base_df = pd.read_csv(base_path)
    psi_df = pd.read_csv(psi_path)

    base_summary = summarize(base_df, "u-fit baseline")
    psi_summary = summarize(psi_df, "psi-fit")

    combined = pd.concat([base_summary, psi_summary], ignore_index=True)

    print("\n" + "=" * 100)
    print("u-fit baseline  vs  psi-fit  (key metrics across full Pareto front)")
    print("=" * 100)
    print(
        combined.to_string(
            index=False,
            formatters={
                "degen_rate": "{:.1%}".format,
                "max_S_cx_mean": "{:.4e}".format,
                "best_mse": "{:.5f}".format,
                "best_S_R_mean": "{:.4f}".format,
                "best_S_cx_mean": "{:.4e}".format,
                "best_S_cy_mean": "{:.4f}".format,
                "best_identity_relL2": "{:.2e}".format,
            },
        )
    )

    print("\n--- Verdict ---")
    psi_a = psi_summary[psi_summary["strategy"] == "A"]
    psi_b = psi_summary[psi_summary["strategy"] == "B"]
    if len(psi_b) > 0:
        rate_b = float(psi_b.iloc[0]["degen_rate"])
        scx_b = float(psi_b.iloc[0]["best_S_cx_mean"])
        if rate_b < 0.5 and scx_b > 1e-3:
            print("  SUCCESS: psi-fit Strategy B (joint u+v) broke the xi-degeneracy.")
        elif rate_b < 0.9:
            print("  PARTIAL: psi-fit Strategy B improved degeneracy rate but not fully.")
        else:
            print("  FAILED: psi-fit Strategy B did not break the xi-degeneracy.")
    print("=" * 100)


if __name__ == "__main__":
    main()
