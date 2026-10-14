"""Convert selected snapshots of this project's wide COMSOL export to PINN NPZ.

Reads x y [u,v,p,spf.U,x,y] blocks without time averaging. No COMSOL server is
needed. Geometry, boundary conditions and Re must still match the PINN config.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_times(path):
    pattern = re.compile(r"(?<![\w.])u\s*\([^)]*\)\s*@\s*t=([+\-\d.eE]+)")
    with Path(path).open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if not line.strip():
                continue
            if not line.lstrip().startswith("%"):
                break
            matches = pattern.findall(line)
            if matches:
                times = np.array([float(value) for value in matches])
                if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                    raise ValueError("COMSOL time coordinates must be finite and strictly increasing")
                return times
    raise ValueError("No 'u (unit) @ t=...' time header found; do not guess time coordinates")


def convert(source, output, length_ref, velocity_ref, density, reynolds,
            time_start=None, time_end=None, time_stride=1, node_stride=1,
            origin=(0.0, 0.0), pressure_offset=0.0):
    source, output = Path(source), Path(output)
    for value in (length_ref, velocity_ref, density, reynolds):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Length, velocity, density and Reynolds number must be finite and positive")
    if time_stride < 1 or node_stride < 1:
        raise ValueError("Strides must be positive integers")
    if len(origin) != 2 or not np.isfinite(origin).all() or not np.isfinite(pressure_offset):
        raise ValueError("Origin and pressure offset must be finite")
    if output.suffix.lower() != ".npz":
        raise ValueError("Output must have .npz extension")
    metadata_path = output.with_suffix(".json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or its metadata")
    metadata, n_time = {}, None
    with source.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("%"):
                if ":" in stripped:
                    key, value = stripped[1:].split(":", 1)
                    metadata[key.strip().lower()] = value.strip()
                continue
            n_columns = len(stripped.split())
            if n_columns < 8 or (n_columns - 2) % 6:
                raise ValueError("Expected x y plus repeated [u,v,p,spf.U,x,y] blocks")
            n_time = (n_columns - 2) // 6
            break
    if n_time is None:
        raise ValueError("No numeric rows in COMSOL export")
    times = read_times(source)
    if len(times) != n_time:
        raise ValueError("Time header does not match the six-column block layout")
    mask = np.ones(n_time, dtype=bool)
    if time_start is not None:
        mask &= times >= time_start
    if time_end is not None:
        mask &= times <= time_end
    indices = np.flatnonzero(mask)[::time_stride]
    if not len(indices):
        raise ValueError("No time snapshots selected")
    usecols = [0, 1] + [2 + 6 * int(i) + component for i in indices for component in range(3)]
    table = np.loadtxt(source, comments="%", usecols=usecols, ndmin=2)[::node_stride]
    xy = (table[:, :2] - np.asarray(origin)) / length_ref
    fields = table[:, 2:].reshape(len(table), len(indices), 3)
    coords = np.column_stack((np.repeat(xy, len(indices), axis=0),
                              np.tile(times[indices] * velocity_ref / length_ref, len(table))))
    velocity = fields[:, :, :2].reshape(-1, 2) / velocity_ref
    pressure = (fields[:, :, 2].reshape(-1, 1) - pressure_offset) / (density * velocity_ref**2)
    if not all(np.isfinite(a).all() for a in (coords, velocity, pressure)):
        raise ValueError("Non-finite COMSOL coordinates or fields")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, coords=coords, velocity=velocity, pressure=pressure,
                        field_kind="instantaneous", coordinate_system="dimensionless", reynolds=reynolds)
    info = {"source": str(source.resolve()), "source_size_bytes": source.stat().st_size,
            "source_metadata": metadata, "length_ref": length_ref, "velocity_ref": velocity_ref,
            "density": density, "reynolds": reynolds, "origin": list(origin),
            "pressure_offset": pressure_offset, "time_indices": indices.tolist(),
            "physical_times": times[indices].tolist(), "node_stride": node_stride,
            "rows": len(coords), "note": "Instantaneous snapshots, no averaging. Verify geometry/BC/IC and reference pressure before training."}
    metadata_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "cylinder_flow_result")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length-ref", type=float, required=True, help="Cylinder diameter in source units")
    parser.add_argument("--velocity-ref", type=float, required=True, help="Reference mean inlet speed")
    parser.add_argument("--density", type=float, required=True)
    parser.add_argument("--reynolds", type=float, required=True, help="From physical simulation settings; not inferred from the export")
    parser.add_argument("--time-start", type=float)
    parser.add_argument("--time-end", type=float)
    parser.add_argument("--time-stride", type=int, default=1)
    parser.add_argument("--node-stride", type=int, default=1)
    parser.add_argument("--origin", type=float, nargs=2, default=(0.0, 0.0))
    parser.add_argument("--pressure-offset", type=float, default=0.0)
    args = parser.parse_args(argv)
    info = convert(**vars(args))
    print(f"Wrote {info['rows']} instantaneous observations to {args.output}")


if __name__ == "__main__":
    main()