"""Train/evaluate a standard PINN with Adam followed by fixed-batch L-BFGS."""

import argparse
import csv
from datetime import datetime, timezone
import json
import hashlib
import math
from pathlib import Path
import platform
import random
import time
import uuid
import numpy as np
import torch
from .config import Config
from .data import load_observations, observation_batch
from .evaluation import evaluate
from .losses import loss_terms, weighted_loss
from .model import build_model
from .problems import FlowProblem

ROOT = Path(__file__).resolve().parents[1]


def setup(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available; use --device cpu")
    dtype = getattr(torch, config.dtype)
    problem = FlowProblem(config)
    # Construct buffers in float64 first so domain bounds are not rounded to float32.
    model = build_model(config, torch.tensor(problem.lower, dtype=dtype), torch.tensor(problem.upper, dtype=dtype))
    model = model.to(device=device, dtype=dtype)
    return problem, model, device, dtype


def load_checkpoint(path, device=None):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format_version") != 1:
        raise ValueError("Unsupported PINN checkpoint format")
    config = Config.from_dict(saved["config"])
    if device is not None:
        config.device = device
    problem, model, _, _ = setup(config)
    model.load_state_dict(saved["model"])
    return config, problem, model, saved


def save_checkpoint(path, model, config, progress):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save({"format_version": 1, "config": config.to_dict(), "model": model.state_dict(),
                "progress": progress, "restart_mode": "weights_only_warm_start"}, temporary)
    temporary.replace(path)


def new_output(explicit, case, suffix=""):
    path = Path(explicit) if explicit else ROOT / "outputs_pinn" / case / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6] + suffix)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory is not empty: {path}. Choose a new run directory.")
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def write_json(path, values):
    path.write_text(json.dumps(values, indent=2, allow_nan=False), encoding="utf-8")


def train(config, out, data=None, init_from=None, smoke=False, *,
          batch_transform=None, max_seconds=None):
    """Train from weights, optionally transform batches or bound training time.

    A batch transform may attach frozen ``interior_weights`` for momentum
    residuals. It receives no sampler RNG; deterministic transforms preserve
    the same sampled coordinates across matched runs. L-BFGS transforms its
    fixed batch once, never within a closure. The wall-clock budget includes
    model/checkpoint setup, batch preparation and logging. It is checked
    between optimizer steps; one in-flight step may finish after the limit.
    """
    if batch_transform is not None and not callable(batch_transform):
        raise ValueError("batch_transform must be callable")
    if max_seconds is not None and (isinstance(max_seconds, bool)
                                   or not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be finite and positive")
    config.validate()
    start = time.perf_counter()
    problem, model, device, dtype = setup(config)
    data_arrays, data_meta = load_observations(data, problem) if data else (None, None)
    if init_from:
        old_config, _, old_model, _ = load_checkpoint(init_from, str(device))
        for name in ("case", "reynolds", "transient", "t_end", "channel_length", "channel_height",
                     "cylinder_x", "cylinder_y", "ramp_time", "hidden_layers", "architecture",
                     "symbolic_expression", "symbolic_coefficients", "symbolic_coefficient_radius",
                     "symbolic_window", "symbolic_taper", "hard_background",
                     "hard_inlet_length", "hard_cylinder_length", "hard_train_cylinder_constant"):
            if getattr(old_config, name) != getattr(config, name):
                raise ValueError(f"Warm-start configuration mismatch: {name}")
        model.load_state_dict(old_model.state_dict())
        # setup() inside checkpoint loading reseeds Torch; reset to this run's seed.
        torch.manual_seed(config.seed)
        torch.set_num_threads(config.threads)
    write_json(out / "config.json", config.to_dict())
    write_json(out / "environment.json", {"python": platform.python_version(), "torch": str(torch.__version__),
               "numpy": np.__version__, "device": str(device), "dtype": config.dtype,
               "architecture": config.architecture, "torch_num_threads": torch.get_num_threads(),
               "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in sorted((ROOT / "pinn").glob("*.py"))},
               "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "run_kind": "smoke_test" if smoke else "training",
               "training_data": data_meta, "init_from": str(Path(init_from).resolve()) if init_from else None,
               "max_training_seconds": max_seconds,
               "batch_transform": batch_transform is not None,
               "reproducibility": "Seeded sampling/initialization; identical results across hardware are not guaranteed."})
    rng, data_rng = np.random.default_rng(config.seed), np.random.default_rng(config.seed + 17)

    def sample_batch():
        sampled = problem.sample(rng, device, dtype)
        if batch_transform is not None:
            sampled = batch_transform(sampled)
            if not isinstance(sampled, dict):
                raise ValueError("batch_transform must return a batch dictionary")
        return sampled

    def budget_reached():
        return max_seconds is not None and time.perf_counter() - start >= max_seconds

    batch = sample_batch()
    observations = observation_batch(data_arrays, data_rng, config.data_batch_size, device, dtype)
    adam = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    progress = {"adam_steps": 0, "lbfgs_steps": 0, "closure_evaluations": 0, "status": "running"}
    history_path = out / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = None

        def record(phase, step):
            nonlocal writer
            terms = loss_terms(model, problem, batch, observations)
            total = weighted_loss(terms, config)
            row = {"phase": phase, "step": step, "elapsed_seconds": time.perf_counter() - start,
                   "closure_evaluations": progress["closure_evaluations"], "total": float(total.detach().cpu()),
                   **{name: float(value.detach().cpu()) for name, value in terms.items()}}
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            print(f"{phase:7s} {step:6d} loss={row['total']:.6e} elapsed={row['elapsed_seconds']:.1f}s", flush=True)

        def backward(optimizer):
            optimizer.zero_grad(set_to_none=True)
            total = weighted_loss(loss_terms(model, problem, batch, observations), config)
            if not torch.isfinite(total):
                raise FloatingPointError("Non-finite PINN loss; check scaling, learning rate and data")
            total.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Non-finite PINN gradient")
            return total

        try:
            record("initial", 0)
            budget_finished = False
            for step in range(1, config.adam_steps + 1):
                if budget_reached():
                    budget_finished = True
                    break
                if step > 1 and (step - 1) % config.resample_every == 0:
                    batch = sample_batch()
                    observations = observation_batch(data_arrays, data_rng, config.data_batch_size, device, dtype)
                if budget_reached():
                    budget_finished = True
                    break
                backward(adam)
                adam.step()
                progress["adam_steps"] = step
                if step % config.log_every == 0 or step == config.adam_steps:
                    record("adam", step)
                if step % config.checkpoint_every == 0:
                    save_checkpoint(out / "checkpoint_last.pt", model, config, progress)
            if config.lbfgs_steps and not budget_finished and budget_reached():
                budget_finished = True
            if config.lbfgs_steps and not budget_finished:
                # Freeze BOTH collocation and observation points for the entire phase.
                batch = sample_batch()
                observations = observation_batch(data_arrays, data_rng, config.data_batch_size, device, dtype)
                optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=1, max_eval=25,
                                             history_size=config.lbfgs_history, line_search_fn="strong_wolfe",
                                             tolerance_grad=1e-9, tolerance_change=1e-12)

                def closure():
                    progress["closure_evaluations"] += 1
                    return backward(optimizer)

                record("lbfgs", 0)
                for step in range(1, config.lbfgs_steps + 1):
                    if budget_reached():
                        budget_finished = True
                        break
                    optimizer.step(closure)
                    progress["lbfgs_steps"] = step
                    if step % config.log_every == 0 or step == config.lbfgs_steps:
                        record("lbfgs", step)
                    if step % config.checkpoint_every == 0:
                        save_checkpoint(out / "checkpoint_last.pt", model, config, progress)
            progress["status"] = "budget_finished" if budget_finished else "training_finished"
        except BaseException:
            progress["status"] = "interrupted_or_failed"
            raise
        finally:
            progress["elapsed_seconds"] = time.perf_counter() - start
            save_checkpoint(out / "checkpoint_last.pt", model, config, progress)
            write_json(out / "training_status.json", progress)
            if getattr(model, "symbolic", None) is not None:
                write_json(out / "symbolic_coefficients.json", {
                    "expression": model.symbolic.expression,
                    "initial": config.symbolic_coefficients,
                    "refined": model.symbolic.coefficient_record(),
                    "coefficient_radius": config.symbolic_coefficient_radius,
                    "source": config.symbolic_source})
    return problem, model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON configuration; defaults to cylinder_steady.json")
    parser.add_argument("--out-dir", type=Path, help="New or empty directory; existing results are never overwritten")
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--adam-steps", type=int)
    parser.add_argument("--lbfgs-steps", type=int)
    parser.add_argument("--data", type=Path, help="Optional dimensionless training observations NPZ")
    parser.add_argument("--reference", type=Path, help="Evaluation-only reference NPZ")
    parser.add_argument("--init-from", type=Path, help="Warm start weights; optimizer/sampler state is reset")
    parser.add_argument("--evaluate", type=Path, metavar="CHECKPOINT", help="Evaluate saved weights without training")
    parser.add_argument("--smoke", action="store_true", help="Tiny 3 Adam + 2 L-BFGS run to check execution, not accuracy")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.evaluate:
        if args.config or args.data or args.init_from or args.smoke or any(v is not None for v in (args.seed, args.adam_steps, args.lbfgs_steps)):
            parser.error("--evaluate only accepts --device, --out-dir, --reference and --no-plots")
        config, problem, model, _ = load_checkpoint(args.evaluate, args.device or "auto")
        out = new_output(args.out_dir, config.case, "_eval")
        write_json(out / "config.json", config.to_dict())
    else:
        config = Config.load(args.config or ROOT / "configs/pinn/cylinder_steady.json")
        for name in ("device", "seed", "adam_steps", "lbfgs_steps"):
            if getattr(args, name) is not None:
                setattr(config, name, getattr(args, name))
        if args.smoke:
            config.hidden_layers = (16, 16)
            config.adam_steps, config.lbfgs_steps = 3, 2
            config.n_interior, config.n_boundary, config.n_initial, config.n_gauge = 64, 16, 32, 8
            config.log_every, config.eval_grid, config.eval_batch_size = 1, 17, 128
        config.validate()
        # Validate files before creating a run or doing expensive training.
        problem = FlowProblem(config)
        if args.data:
            load_observations(args.data, problem)
        if args.reference:
            load_observations(args.reference, problem)
        out = new_output(args.out_dir, config.case, "_smoke" if args.smoke else "")
        print(f"Run directory: {out}", flush=True)
        problem, model = train(config, out, args.data, args.init_from, args.smoke)
    reference, meta = load_observations(args.reference, problem) if args.reference else (None, None)
    write_json(out / "evaluation_input.json", {"reference": meta, "checkpoint": str(args.evaluate.resolve()) if args.evaluate else "checkpoint_last.pt"})
    metrics = evaluate(model, problem, out / "evaluation", reference, not args.no_plots)
    print(f"Validation loss: {metrics['validation_loss']:.6e}\nResults: {out}", flush=True)
    return out


if __name__ == "__main__":
    main()
