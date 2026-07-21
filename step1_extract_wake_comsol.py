"""
Step 1 (COMSOL): convert an exported cylinder-flow result table into the
standard PISR inputs consumed by step2_pisr.py.

The expected source is the wide COMSOL text export currently named
``cylinder_flow_result``:

    x y [u, v, p, spf.U, x, y] for each saved time step

The script time-averages u and v, interpolates them onto the same regular wake
grid used by the MeshGraphNets pipeline, and writes.  The default x grid is
deliberately dense enough that the SR window has 33 xi columns (rather than
the legacy 16), with dimensionless x/y spacings that are approximately equal:

    wake_data.csv
    v_mean_wake.npy
    wake_x.npy / wake_y.npy
    u_mean_full_grid.npy / v_mean_full_grid.npy
    x_grid.npy / y_grid.npy
    meta.json
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import griddata


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = REPO_ROOT / "cylinder_flow_result"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs_mgn" / "comsol"
DEFAULT_N_X_GRID = 97
DEFAULT_N_Y_GRID = 64
DEFAULT_MIN_WAKE_X_POINTS = 32
DENSE_XI_GRID_REVISION = "dense_xi_v1"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare COMSOL cylinder-flow output for the PISR pipeline."
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help="Path to the COMSOL wide text export.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Directory where standard PISR input files will be written.",
    )
    parser.add_argument(
        "--diameter",
        type=float,
        default=0.1,
        help="Cylinder diameter D in source length units.",
    )
    parser.add_argument(
        "--cyl-cx",
        type=float,
        default=0.2,
        help="Cylinder center x coordinate.",
    )
    parser.add_argument(
        "--cyl-cy",
        type=float,
        default=0.2,
        help="Cylinder center y coordinate.",
    )
    parser.add_argument(
        "--u-in",
        type=float,
        default=None,
        help="Reference inlet velocity. If omitted, it is estimated near x_min.",
    )
    parser.add_argument(
        "--inlet-width",
        type=float,
        default=0.02,
        help="Width from x_min used to estimate inlet velocity.",
    )
    parser.add_argument(
        "--time-start-index",
        type=int,
        default=None,
        help="First time index included in the time average.",
    )
    parser.add_argument(
        "--time-end-index",
        type=int,
        default=None,
        help="Exclusive final time index included in the time average.",
    )
    parser.add_argument(
        "--time-start-frac",
        type=float,
        default=0.5,
        help="Fallback start fraction for time averaging when no index is given.",
    )
    parser.add_argument(
        "--n-x-grid",
        type=int,
        default=DEFAULT_N_X_GRID,
        help=(
            "Number of x points in the full regular interpolation grid. "
            "The default 97 gives 33 points in xi=[3, 5] and Delta xi=0.0625 "
            "for the standard xi=[2, 8] domain."
        ),
    )
    parser.add_argument(
        "--n-y-grid",
        type=int,
        default=DEFAULT_N_Y_GRID,
        help="Number of y points in the regular interpolation grid.",
    )
    parser.add_argument(
        "--min-wake-x-points",
        type=int,
        default=DEFAULT_MIN_WAKE_X_POINTS,
        help=(
            "Reject an under-resolved SR wake grid with fewer x points than "
            "this value. Lower explicitly only for smoke tests."
        ),
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.01,
        help="Domain margin excluded from the interpolation window.",
    )
    parser.add_argument(
        "--symmetric-y-window",
        action="store_true",
        help="Use the largest y-window symmetric about --cyl-cy after margins.",
    )
    args = parser.parse_args()

    if args.diameter <= 0:
        parser.error("--diameter must be positive.")
    if args.n_x_grid < 3 or args.n_y_grid < 3:
        parser.error("--n-x-grid and --n-y-grid must both be at least 3.")
    if args.min_wake_x_points < 3:
        parser.error("--min-wake-x-points must be at least 3.")
    if not 0 <= args.time_start_frac < 1:
        parser.error("--time-start-frac must be in [0, 1).")

    return args


def read_comsol_metadata(path):
    metadata = {}
    first_data_line = None

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("%"):
                content = stripped[1:].strip()
                if ":" in content:
                    key, value = content.split(":", 1)
                    metadata[key.strip().lower()] = value.strip()
                continue
            first_data_line = stripped
            break

    if first_data_line is None:
        raise ValueError(f"No numeric data rows found in {path}.")

    n_cols = len(first_data_line.split())
    if n_cols < 8 or (n_cols - 2) % 6 != 0:
        raise ValueError(
            "Expected columns: x y plus repeated [u, v, p, spf.U, x, y] blocks; "
            f"found {n_cols} numeric columns."
        )

    n_time = (n_cols - 2) // 6
    return metadata, n_cols, n_time


def load_comsol_uv(path, n_time):
    usecols = [0, 1]
    for i in range(n_time):
        usecols.extend([2 + 6 * i, 3 + 6 * i])

    print(f"Loading COMSOL table: {path}")
    print(f"  time steps: {n_time}")
    data = np.loadtxt(path, comments="%", usecols=usecols, dtype=np.float64)

    coords = data[:, :2]
    uv = data[:, 2:]
    u_time = uv[:, 0::2]
    v_time = uv[:, 1::2]
    return coords, u_time, v_time


def select_time_window(n_time, start_index, end_index, start_frac):
    if start_index is None:
        start_index = int(n_time * start_frac)
    if end_index is None:
        end_index = n_time

    start_index = max(0, start_index)
    end_index = min(n_time, end_index)
    if start_index >= end_index:
        raise ValueError(
            f"Invalid time window [{start_index}, {end_index}) for {n_time} time steps."
        )
    return start_index, end_index


def estimate_u_in(mesh_pos, u_mean_nodes, x_min, inlet_width):
    inlet_mask = mesh_pos[:, 0] <= x_min + inlet_width
    if not np.any(inlet_mask):
        closest_x = np.min(mesh_pos[:, 0])
        inlet_mask = np.isclose(mesh_pos[:, 0], closest_x)

    u_in = float(np.mean(u_mean_nodes[inlet_mask]))
    return u_in if abs(u_in) > 1e-8 else 1.0


def interpolate_wake(mesh_pos, u_mean_nodes, v_mean_nodes, args):
    D = float(args.diameter)
    cyl_cx = float(args.cyl_cx)
    cyl_cy = float(args.cyl_cy)

    x_min, x_max = float(mesh_pos[:, 0].min()), float(mesh_pos[:, 0].max())
    y_min, y_max = float(mesh_pos[:, 1].min()), float(mesh_pos[:, 1].max())

    wake_x_lo = max(cyl_cx + 2 * D, x_min + args.margin)
    wake_x_hi = min(cyl_cx + 8 * D, x_max - args.margin)
    if args.symmetric_y_window:
        y_half_width = min(
            3 * D,
            cyl_cy - (y_min + args.margin),
            (y_max - args.margin) - cyl_cy,
        )
        if y_half_width <= 0:
            raise ValueError(
                "Symmetric y-window is empty. Check --cyl-cy and --margin."
            )
        wake_y_lo = cyl_cy - y_half_width
        wake_y_hi = cyl_cy + y_half_width
    else:
        wake_y_lo = max(cyl_cy - 3 * D, y_min + args.margin)
        wake_y_hi = min(cyl_cy + 3 * D, y_max - args.margin)

    if wake_x_lo >= wake_x_hi or wake_y_lo >= wake_y_hi:
        raise ValueError(
            "Wake interpolation window is empty. Check --diameter, --cyl-cx, "
            "--cyl-cy, and --margin."
        )

    x_grid = np.linspace(wake_x_lo, wake_x_hi, args.n_x_grid)
    y_grid = np.linspace(wake_y_lo, wake_y_hi, args.n_y_grid)
    dx = float(x_grid[1] - x_grid[0])
    dy = float(y_grid[1] - y_grid[0])

    XX, YY = np.meshgrid(x_grid, y_grid, indexing="ij")
    grid_points = np.column_stack([XX.ravel(), YY.ravel()])

    print("\nWake interpolation region:")
    print(f"  x: [{wake_x_lo:.6f}, {wake_x_hi:.6f}]")
    print(f"  y: [{wake_y_lo:.6f}, {wake_y_hi:.6f}]")
    print(f"  x/D: [{(wake_x_lo - cyl_cx) / D:.2f}, {(wake_x_hi - cyl_cx) / D:.2f}]")
    print(f"  y/D: [{(wake_y_lo - cyl_cy) / D:.2f}, {(wake_y_hi - cyl_cy) / D:.2f}]")
    print(f"Regular grid: {args.n_x_grid} x {args.n_y_grid}")
    print(f"  Delta xi={dx / D:.6f}, Delta eta={dy / D:.6f}")

    u_grid = griddata(mesh_pos, u_mean_nodes, grid_points, method="linear").reshape(
        args.n_x_grid, args.n_y_grid
    )
    v_grid = griddata(mesh_pos, v_mean_nodes, grid_points, method="linear").reshape(
        args.n_x_grid, args.n_y_grid
    )

    nan_mask = np.isnan(u_grid) | np.isnan(v_grid)
    if np.any(nan_mask):
        print(f"WARNING: filling {int(nan_mask.sum())} NaN grid points with nearest data.")
        u_near = griddata(mesh_pos, u_mean_nodes, grid_points, method="nearest").reshape(
            args.n_x_grid, args.n_y_grid
        )
        v_near = griddata(mesh_pos, v_mean_nodes, grid_points, method="nearest").reshape(
            args.n_x_grid, args.n_y_grid
        )
        u_grid[np.isnan(u_grid)] = u_near[np.isnan(u_grid)]
        v_grid[np.isnan(v_grid)] = v_near[np.isnan(v_grid)]

    wake_sr_x_lo = cyl_cx + 3 * D
    wake_sr_x_hi = cyl_cx + 5 * D
    ix_sr = np.where((x_grid >= wake_sr_x_lo) & (x_grid <= wake_sr_x_hi))[0]
    if len(ix_sr) < 3:
        print("WARNING: SR region too small, expanding to 2D..6D.")
        ix_sr = np.where((x_grid >= cyl_cx + 2 * D) & (x_grid <= cyl_cx + 6 * D))[0]
    if len(ix_sr) < 3:
        print("WARNING: SR region still too small, using the middle half of x-grid.")
        ix_sr = np.arange(args.n_x_grid // 4, 3 * args.n_x_grid // 4)

    if len(ix_sr) < args.min_wake_x_points:
        raise ValueError(
            "Under-resolved SR wake grid: "
            f"xi=[3, 5] contains only {len(ix_sr)} x points, but "
            f"--min-wake-x-points={args.min_wake_x_points}. Increase "
            "--n-x-grid (97 is the standard dense-xi setting) or lower the "
            "minimum explicitly for a smoke test."
        )

    print(f"\nSR wake sub-region: {len(ix_sr)} x-positions, indices {ix_sr[0]}..{ix_sr[-1]}")
    print(
        f"  x/D range: [{(x_grid[ix_sr[0]] - cyl_cx) / D:.2f}, "
        f"{(x_grid[ix_sr[-1]] - cyl_cx) / D:.2f}]"
    )

    return x_grid, y_grid, u_grid, v_grid, ix_sr, dx, dy


def write_outputs(
    out_dir,
    source_path,
    metadata,
    mesh_pos,
    x_grid,
    y_grid,
    u_grid,
    v_grid,
    ix_sr,
    dx,
    dy,
    u_in,
    time_start,
    time_end,
    n_time,
    args,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    D = float(args.diameter)
    cyl_cx = float(args.cyl_cx)
    cyl_cy = float(args.cyl_cy)

    rows = []
    for i in ix_sr:
        for j in range(len(y_grid)):
            rows.append(
                {
                    "x_D": (x_grid[i] - cyl_cx) / D,
                    "y_D": (y_grid[j] - cyl_cy) / D,
                    "u_star": u_grid[i, j] / u_in,
                }
            )

    df = pd.DataFrame(rows)
    v_wake = v_grid[np.ix_(ix_sr, np.arange(len(y_grid)))] / u_in

    df.to_csv(out_dir / "wake_data.csv", index=False)
    np.save(out_dir / "v_mean_wake.npy", v_wake)
    np.save(out_dir / "wake_x.npy", x_grid[ix_sr])
    np.save(out_dir / "wake_y.npy", y_grid)
    np.save(out_dir / "u_mean_full_grid.npy", u_grid / u_in)
    np.save(out_dir / "v_mean_full_grid.npy", v_grid / u_in)
    np.save(out_dir / "x_grid.npy", x_grid)
    np.save(out_dir / "y_grid.npy", y_grid)

    meta_out = {
        "D": D,
        "U_in": float(u_in),
        "Re": "not provided in COMSOL export",
        "cyl_cx": cyl_cx,
        "cyl_cy": cyl_cy,
        "dx": float(dx),
        "dy": float(dy),
        "dx_over_D": float(dx / D),
        "dy_over_D": float(dy / D),
        "xi_eta_spacing_ratio": float(dx / dy),
        "wake_ix": ix_sr.tolist(),
        "n_x_wake": int(len(ix_sr)),
        "n_y": int(len(y_grid)),
        "n_x_full": int(len(x_grid)),
        "n_y_full": int(len(y_grid)),
        "wake_x_D_min": float((x_grid[ix_sr[0]] - cyl_cx) / D),
        "wake_x_D_max": float((x_grid[ix_sr[-1]] - cyl_cx) / D),
        "min_wake_x_points": int(args.min_wake_x_points),
        "grid_revision": DENSE_XI_GRID_REVISION,
        "data_source": "comsol_export",
        "source_file": str(source_path),
        "source_metadata": metadata,
        "t_avg_start_index": int(time_start),
        "t_avg_end_index": int(time_end),
        "n_time_steps": int(n_time),
        "n_nodes_original": int(mesh_pos.shape[0]),
        "interp_method": "linear",
        "symmetric_y_window": bool(args.symmetric_y_window),
        "n_points": int(len(df)),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2, ensure_ascii=False)

    print(f"\nDataFrame: {len(df)} samples")
    print(df.describe())
    print(f"\nAll outputs saved to {out_dir}")


def main():
    args = parse_args()
    source_path = Path(args.source).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    metadata, n_cols, n_time = read_comsol_metadata(source_path)
    print(f"COMSOL metadata: nodes={metadata.get('nodes', 'unknown')}, columns={n_cols}")

    mesh_pos, u_time, v_time = load_comsol_uv(source_path, n_time)
    t_start, t_end = select_time_window(
        n_time, args.time_start_index, args.time_end_index, args.time_start_frac
    )

    print(f"Time-averaging over indices [{t_start}, {t_end})")
    u_mean_nodes = u_time[:, t_start:t_end].mean(axis=1)
    v_mean_nodes = v_time[:, t_start:t_end].mean(axis=1)

    x_min, x_max = float(mesh_pos[:, 0].min()), float(mesh_pos[:, 0].max())
    y_min, y_max = float(mesh_pos[:, 1].min()), float(mesh_pos[:, 1].max())
    print(f"\nNodes: {mesh_pos.shape[0]}")
    print(f"Domain x=[{x_min:.6f}, {x_max:.6f}], y=[{y_min:.6f}, {y_max:.6f}]")
    print(f"u_mean range: [{u_mean_nodes.min():.6f}, {u_mean_nodes.max():.6f}]")
    print(f"v_mean range: [{v_mean_nodes.min():.6f}, {v_mean_nodes.max():.6f}]")

    if args.u_in is None:
        u_in = estimate_u_in(mesh_pos, u_mean_nodes, x_min, args.inlet_width)
        print(f"\nEstimated inlet velocity U_in={u_in:.6f}")
    else:
        u_in = float(args.u_in)
        print(f"\nUsing provided inlet velocity U_in={u_in:.6f}")

    x_grid, y_grid, u_grid, v_grid, ix_sr, dx, dy = interpolate_wake(
        mesh_pos, u_mean_nodes, v_mean_nodes, args
    )

    write_outputs(
        out_dir=out_dir,
        source_path=source_path,
        metadata=metadata,
        mesh_pos=mesh_pos,
        x_grid=x_grid,
        y_grid=y_grid,
        u_grid=u_grid,
        v_grid=v_grid,
        ix_sr=ix_sr,
        dx=dx,
        dy=dy,
        u_in=u_in,
        time_start=t_start,
        time_end=t_end,
        n_time=n_time,
        args=args,
    )
    print("Done.")


if __name__ == "__main__":
    main()
