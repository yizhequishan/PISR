"""
Diagnostic: how strong is the v_mean signal and its xi-variation?

Decides whether the stream-function PISR route is viable. Key ratios reported:
  R1 = ||v||_RMS / ||u||_RMS                 (overall transverse signal strength)
  R2 = ||dv/dxi||_RMS / ||du/deta||_RMS      (xi-variation of v, normalized by main u gradient)
  R3 = ||dv/dxi||_RMS / ||dv/deta||_RMS      (anisotropy of v gradients, mirror of rho test on u)

Decision rule (per the paper-discussion thread):
  R1 > 5%  --> v signal large enough to anchor a stream-function objective
  R1 < 1%  --> v ~ 0, stream-function route cannot force xi dependence
  1% <= R1 <= 5% --> ambiguous, run both routes
"""
import json
import os

import numpy as np
import pandas as pd

RUN_DIR = r"D:\keyan\cylinder_flow_pisr\outputs_mgn\comsol\eta_sym"


def rms(arr):
    return float(np.sqrt(np.mean(arr ** 2)))


def central_diff(arr2d, axis, h):
    """Central difference along `axis` with one-sided at boundaries."""
    out = np.zeros_like(arr2d, dtype=float)
    if axis == 0:
        out[1:-1, :] = (arr2d[2:, :] - arr2d[:-2, :]) / (2.0 * h)
        out[0, :] = (arr2d[1, :] - arr2d[0, :]) / h
        out[-1, :] = (arr2d[-1, :] - arr2d[-2, :]) / h
    elif axis == 1:
        out[:, 1:-1] = (arr2d[:, 2:] - arr2d[:, :-2]) / (2.0 * h)
        out[:, 0] = (arr2d[:, 1] - arr2d[:, 0]) / h
        out[:, -1] = (arr2d[:, -1] - arr2d[:, -2]) / h
    else:
        raise ValueError("axis must be 0 or 1")
    return out


def main():
    with open(os.path.join(RUN_DIR, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)

    D = float(meta["D"])
    dx = float(meta["dx"])
    dy = float(meta["dy"])
    dx_D = dx / D
    dy_D = dy / D
    n_x = int(meta["n_x_wake"])
    n_y = int(meta["n_y"])

    df = pd.read_csv(os.path.join(RUN_DIR, "wake_data.csv"))
    df = df.sort_values(["x_D", "y_D"]).reset_index(drop=True)

    u_star_2d = df["u_star"].values.reshape(n_x, n_y)
    v_wake_2d = np.load(os.path.join(RUN_DIR, "v_mean_wake.npy"))

    if v_wake_2d.shape != u_star_2d.shape:
        raise ValueError(
            f"v_mean_wake shape {v_wake_2d.shape} != u_star_2d shape {u_star_2d.shape}"
        )

    U_in = float(meta["U_in"])
    # v_mean_wake.npy follows the Step 1 contract and is already v/U_in.
    v_star_2d = v_wake_2d.astype(float)

    print("=" * 64)
    print("Stream-function viability diagnostic")
    print(f"Run dir: {RUN_DIR}")
    print(f"Grid: n_x={n_x}, n_y={n_y}, dx/D={dx_D:.5f}, dy/D={dy_D:.5f}, U_in={U_in:.4f}")
    print("=" * 64)

    print("\n--- Field magnitudes (dimensionless) ---")
    u_rms = rms(u_star_2d)
    v_rms = rms(v_star_2d)
    u_max = float(np.max(np.abs(u_star_2d)))
    v_max = float(np.max(np.abs(v_star_2d)))
    print(f"  ||u*||_RMS  = {u_rms:.4e}    ||u*||_max  = {u_max:.4e}")
    print(f"  ||v*||_RMS  = {v_rms:.4e}    ||v*||_max  = {v_max:.4e}")
    R1 = v_rms / u_rms
    R1_max = v_max / u_max
    print(f"  R1 = ||v*||_RMS / ||u*||_RMS = {R1:.4f}  ({R1*100:.2f}%)")
    print(f"  R1_max ratio                 = {R1_max:.4f}  ({R1_max*100:.2f}%)")

    print("\n--- Gradients ---")
    du_dxi = central_diff(u_star_2d, axis=0, h=dx_D)
    du_deta = central_diff(u_star_2d, axis=1, h=dy_D)
    dv_dxi = central_diff(v_star_2d, axis=0, h=dx_D)
    dv_deta = central_diff(v_star_2d, axis=1, h=dy_D)

    du_dxi_rms = rms(du_dxi)
    du_deta_rms = rms(du_deta)
    dv_dxi_rms = rms(dv_dxi)
    dv_deta_rms = rms(dv_deta)
    print(f"  ||du*/dxi||_RMS  = {du_dxi_rms:.4e}")
    print(f"  ||du*/deta||_RMS = {du_deta_rms:.4e}")
    print(f"  ||dv*/dxi||_RMS  = {dv_dxi_rms:.4e}")
    print(f"  ||dv*/deta||_RMS = {dv_deta_rms:.4e}")
    rho_u = du_dxi_rms / du_deta_rms
    print(f"  rho_u = ||du*/dxi|| / ||du*/deta|| = {rho_u:.4f}  ({rho_u*100:.2f}%)  [matches earlier 6% finding]")

    print("\n--- Stream-function-relevant ratios ---")
    R2 = dv_dxi_rms / du_deta_rms
    print(f"  R2 = ||dv*/dxi|| / ||du*/deta|| = {R2:.4f}  ({R2*100:.2f}%)")
    print("       (xi-variation of v vs main u gradient)")
    R3 = dv_dxi_rms / max(dv_deta_rms, 1e-12)
    print(f"  R3 = ||dv*/dxi|| / ||dv*/deta|| = {R3:.4f}  ({R3*100:.2f}%)")
    print("       (anisotropy of v gradients - the analog of rho for v)")

    print("\n--- Divergence residual (sanity) ---")
    div_resid = du_dxi + dv_deta
    div_rms = rms(div_resid)
    print(f"  ||du/dxi + dv/deta||_RMS = {div_rms:.4e}")
    print(f"  vs ||du/dxi||_RMS = {du_dxi_rms:.4e}")
    print(f"  ratio (residual / signal) = {div_rms / max(du_dxi_rms, 1e-12):.3f}")
    print("  (small = data is divergence-free; large = numerical noise / wrong scaling)")

    print("\n--- v field structure (slices at typical xi stations) ---")
    x_vals = np.unique(df["x_D"])
    y_vals = np.unique(df["y_D"])
    for ix_pick in [0, n_x // 2, n_x - 1]:
        v_slice = v_star_2d[ix_pick, :]
        print(
            f"  xi={x_vals[ix_pick]:5.2f}: v* in [{v_slice.min():+.4e}, {v_slice.max():+.4e}], "
            f"RMS={rms(v_slice):.4e}, mean={v_slice.mean():+.4e}"
        )

    print("\n--- Antisymmetry check (mean v should be ~0 by reflection) ---")
    mid = n_y // 2
    if n_y % 2 == 0:
        v_upper = v_star_2d[:, mid:]
        v_lower = v_star_2d[:, :mid][:, ::-1]
    else:
        v_upper = v_star_2d[:, mid + 1:]
        v_lower = v_star_2d[:, :mid][:, ::-1]
    asym = v_upper + v_lower  # should be ~0 if perfectly antisymmetric
    print(f"  ||v_upper + v_lower||_RMS = {rms(asym):.4e}  (should be small if antisymmetric)")
    print(f"  ||v_upper||_RMS           = {rms(v_upper):.4e}")
    print(f"  asymmetry ratio           = {rms(asym) / max(rms(v_upper), 1e-12):.3f}")

    print("\n" + "=" * 64)
    print("DECISION")
    print("=" * 64)
    if R1 > 0.05:
        verdict = "VIABLE"
        rec = "Stream-function route can anchor xi dependence via v fitting."
    elif R1 < 0.01:
        verdict = "NOT VIABLE"
        rec = "v signal too small; fall back to Sobolev xi-grad penalty."
    else:
        verdict = "AMBIGUOUS"
        rec = "Run both Sobolev and stream-function and compare."

    print(f"  R1 (v vs u magnitude)         = {R1*100:.2f}%   --> {verdict}")
    print(f"  R2 (dv/dxi vs du/deta)        = {R2*100:.2f}%")
    print(f"  R3 (dv/dxi vs dv/deta)        = {R3*100:.2f}%")
    print(f"  Recommendation: {rec}")
    print("=" * 64)


if __name__ == "__main__":
    main()
