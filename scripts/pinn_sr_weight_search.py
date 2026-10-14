"""Isolated Julia/PySR worker; deliberately does not import Torch."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--search-seconds", type=float, default=60)
    args = parser.parse_args()
    start = time.perf_counter()
    from pysr import PySRRegressor
    import numpy as np
    out = args.out_dir
    protocol = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    data = np.load(out / "development.npz")
    points = data["fit_coords"]
    height = protocol["baseline_config"]["channel_height"]
    inputs = np.column_stack((points[:, 0], 2*points[:, 1]/height-1))
    search = PySRRegressor(niterations=args.iterations, populations=6, population_size=32,
                          ncycles_per_iteration=100, maxsize=23,
                          binary_operators=["+", "-", "*"], unary_operators=["square", "exp", "tanh"],
                          nested_constraints={"exp": {"exp": 0}, "tanh": {"tanh": 0}},
                          random_state=protocol["seeds"]["fit"], deterministic=True, parallelism="serial",
                          precision=64, batching=False, model_selection="accuracy",
                          progress=False, verbosity=0, timeout_in_seconds=args.search_seconds,
                          output_directory=str(out / "search"), run_id="residual_weights")
    search.fit(inputs, data["target_fit"], variable_names=["x_D", "y_D"])
    search.equations_.drop(columns=[c for c in ("lambda_format",) if c in search.equations_]).to_csv(
        out / "equations.csv", index=False)
    candidates = [{"index": int(i), "complexity": int(row["complexity"]),
                   "expression": str(row["sympy_format"])} for i, row in search.equations_.iterrows()]
    (out / "search_candidates.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    print("Symbolic worker finished in %.2f s" % (time.perf_counter()-start), flush=True)


if __name__ == "__main__":
    main()
