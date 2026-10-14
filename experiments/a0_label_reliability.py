"""Audit raw time averaging and derivative stability; does not run CFD or training.

Batch-mean SEs are conditional diagnostics, not certified confidence intervals.
The independent-run and synchronous-pair alternatives are both reported because
the exports do not specify a joint sampling experiment for the two runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.interpolate import LinearNDInterpolator, RectBivariateSpline
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.a0_m_reconstructability import field_derivatives

CASES = {
    "base": ("cylinder_flow_result", .100, .200, .200),
    "cx+": ("cx+.txt", .100, .202, .200),
    "cx-": ("cx-.txt", .100, .198, .200),
    "cy+": ("cy+.txt", .100, .200, .202),
    "cy-": ("cy-.txt", .100, .200, .198),
    "R+": ("R+.txt", .102, .200, .200),
    "R-": ("R-.txt", .098, .200, .200),
}
STEPS = {"R": .001, "cx": .002, "cy": .002}


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def table(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a)**2)))


def rel(truth, estimate):
    return rms(np.asarray(estimate)-truth) / max(rms(truth), 1e-30)


def read_case(name, xx, yy, out, rebuild=False):
    filename, diameter, cx, cy = CASES[name]
    source = ROOT / filename
    stat = source.stat()
    cache_file = out / "time_series" / (name.replace("+", "p").replace("-", "m") + ".npz")
    if cache_file.exists() and not rebuild:
        with np.load(cache_file, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"].item()))
            if (meta["source_bytes"] == stat.st_size and meta["source_mtime_ns"] == stat.st_mtime_ns
                    and np.array_equal(z["xi"], xx[:, 0]) and np.array_equal(z["eta"], yy[0])):
                print(f"{name}: reusing audited time-series cache", flush=True)
                return {"uv": z["uv"], "inlet": z["inlet"], "times": z["times"], "metadata": meta}
    import re
    headers, coords, velocities, digest = [], [], [], hashlib.sha256()
    max_mesh_motion = 0.
    with source.open("rb") as f:
        for line in f:
            digest.update(line)
            if not line.strip():
                continue
            if line.startswith(b"%"):
                headers.append(line.decode("gb18030", errors="replace").strip())
                continue
            values = np.fromstring(line.decode("ascii"), sep=" ")
            if values.size < 8 or (values.size-2) % 6:
                raise ValueError(f"Invalid row in {filename}")
            coords.append(values[:2])
            velocities.append(np.column_stack([values[2::6], values[3::6]]))
            max_mesh_motion = max(max_mesh_motion, float(np.max(abs(values[6::6]-values[0]))),
                                  float(np.max(abs(values[7::6]-values[1]))))
            if len(coords) % 2000 == 0:
                print(f"{name}: parsed {len(coords)} nodes", flush=True)
    if source.stat().st_size != stat.st_size or source.stat().st_mtime_ns != stat.st_mtime_ns:
        raise RuntimeError("Source changed while reading")
    coords, velocities = np.asarray(coords), np.asarray(velocities)
    times = np.asarray(list(dict.fromkeys(float(t) for t in re.findall(
        r"\bt\s*=\s*([-+0-9.eE]+)", " ".join(headers)))))
    if times.size != velocities.shape[1] or not np.allclose(np.diff(times), .01):
        raise ValueError("Unexpected export timing")
    if not np.isfinite(velocities).all() or max_mesh_motion > 1e-10:
        raise ValueError("Nonfinite velocity or moving export grid")
    inlet_mask = coords[:, 0] <= coords[:, 0].min()+.02
    inlet = velocities[inlet_mask, :, 0].mean(axis=0)
    uv = LinearNDInterpolator(coords, velocities)(cx+diameter*xx, cy+diameter*yy)
    # Grid axes, time, component -> point, time, component.
    uv = uv.reshape(-1, times.size, 2)
    if not np.isfinite(uv).all():
        raise ValueError("Reference grid leaves interpolation hull; no fallback allowed")
    old_cache = ROOT / "outputs_mgn/fd_check/case_cache" / cache_file.name
    with np.load(old_cache, allow_pickle=False) as z:
        if not np.array_equal(coords, z["coords"]):
            raise ValueError("Raw node positions differ from historical cache")
        old_uin = float(z["U_in"])
        mean_errors = {c: rel(z[c+"_mean"], velocities[:, 400:801, k].mean(axis=1))
                       for k, c in enumerate(("u", "v"))}
    metadata = {"case": name, "file": str(source), "source_bytes": stat.st_size,
                "source_mtime_ns": stat.st_mtime_ns, "source_sha256": digest.hexdigest(),
                "nodes": len(coords), "n_time": times.size, "dt": float(times[1]-times[0]),
                "geometry": {"D": diameter, "cx": cx, "cy": cy},
                "time_start": float(times[0]), "time_end": float(times[-1]),
                "inlet_nodes": int(inlet_mask.sum()), "reference_U_in": float(inlet[400:801].mean()),
                "old_U_in": old_uin, "raw_vs_cached_node_mean_rel_l2": mean_errors,
                "max_export_mesh_motion": max_mesh_motion,
                "inlet_temporal_std_4_8": float(inlet[400:801].std(ddof=1))}
    assert max(mean_errors.values()) < 1e-10
    assert abs(metadata["reference_U_in"]-old_uin) < 1e-10
    np.savez_compressed(cache_file, uv=uv, inlet=inlet, times=times, xi=xx[:, 0], eta=yy[0],
                        metadata=np.array(json.dumps(metadata)))
    print(f"{name}: {len(coords)} nodes, reference inlet {metadata['reference_U_in']:.9f}, verified raw means", flush=True)
    return {"uv": uv, "inlet": inlet, "times": times, "metadata": metadata}


def mean_field(case, start=400, stop=801, renormalize=True):
    denominator = case["inlet"][start:stop].mean() if renormalize else case["metadata"]["reference_U_in"]
    return case["uv"][:, start:stop].mean(axis=1)/denominator


def response(cases, parameter, start=400, stop=801, renormalize=True):
    return (mean_field(cases[parameter+"+"], start, stop, renormalize)
            -mean_field(cases[parameter+"-"], start, stop, renormalize))/(2*STEPS[parameter])


def batch_diagnostics(cases, reference, out):
    rows, arrays, identity_errors = [], {}, []
    for parameter, h in STEPS.items():
        plus, minus = cases[parameter+"+"], cases[parameter+"-"]
        for length in (25, 50, 100, 200):
            starts = range(400, 800, length)
            for normalize in (False, True):
                p = np.stack([mean_field(plus, j, j+length, normalize) for j in starts])
                m = np.stack([mean_field(minus, j, j+length, normalize) for j in starts])
                count = p.shape[0]
                var_p, var_m = p.var(axis=0, ddof=1), m.var(axis=0, ddof=1)
                cov = ((p-p.mean(axis=0))*(m-m.mean(axis=0))).sum(axis=0)/(count-1)
                independent = (var_p+var_m)/(4*h*h*count)
                paired = ((p-m)/(2*h)).var(axis=0, ddof=1)/count
                formula = (var_p+var_m-2*cov)/(4*h*h*count)
                identity_errors.append(float(np.max(abs(paired-formula)))/max(float(np.max(paired)), 1e-30))
                norm_name = "window_inlet" if normalize else "frozen_inlet"
                arrays[f"{parameter}_{length}_{norm_name}_independent_se"] = np.sqrt(independent)
                arrays[f"{parameter}_{length}_{norm_name}_paired_se"] = np.sqrt(paired)
                arrays[f"{parameter}_{length}_{norm_name}_M_blocks"] = (p-m)/(2*h)
                for k, c in enumerate(("u", "v")):
                    truth = reference[parameter][:, k]
                    se_ind, se_pair = np.sqrt(np.mean(independent[:, k])), np.sqrt(np.mean(paired[:, k]))
                    rows.append({"parameter": parameter, "component": c, "normalization": norm_name,
                                 "block_samples": length, "block_duration_s": length*.01, "n_blocks": count,
                                 "reference_M_rms": rms(truth), "independent_SE_rms": float(se_ind),
                                 "independent_SE_over_M_rms": float(se_ind/max(rms(truth), 1e-30)),
                                 "synchronous_paired_SE_rms": float(se_pair),
                                 "synchronous_paired_SE_over_M_rms": float(se_pair/max(rms(truth), 1e-30)),
                                 "ensemble_covariance_identified": False,
                                 "recentered_block_mean_vs_reference_rel_l2": rel(truth, ((p-m)/(2*h)).mean(axis=0)[:, k])})
    table(out/"batch_mean_diagnostics.csv", rows)
    np.savez_compressed(out/"batch_mean_fields.npz", **arrays)
    assert max(identity_errors) < 1e-10
    return rows, max(identity_errors)


def temporal_diagnostics(cases, reference, out):
    rows, spectra, acfs, labels = [], [], [], []
    fields = {name: case["uv"][:, 400:800]/case["metadata"]["reference_U_in"] for name, case in cases.items()}
    fields.update({"M_"+p: (fields[p+"+"]-fields[p+"-"])/(2*h) for p, h in STEPS.items()})
    for name, field in fields.items():
        for k, component in enumerate(("u", "v")):
            a = field[:, :, k]-field[:, :, k].mean(axis=1, keepdims=True)
            n = a.shape[1]
            fft = np.fft.rfft(a, n=2*n, axis=1)
            correlation = np.fft.irfft(np.sum(abs(fft)**2, axis=0), n=2*n)[:n]/np.arange(n, 0, -1)
            acf = correlation/max(correlation[0], 1e-30)
            power = np.sum(abs(np.fft.rfft(a, axis=1))**2, axis=0)
            frequencies = np.fft.rfftfreq(n, d=.01)
            peak = int(np.argmax(power[1:])+1)
            nonpositive = np.flatnonzero(acf[1:] <= 0)
            zero = int(nonpositive[0]+1) if len(nonpositive) else n
            tau = .01*(.5+float(np.sum(acf[1:zero])))
            rows.append({"series": name, "component": component, "frequency_resolution_hz": .25,
                         "dominant_fft_frequency_hz": float(frequencies[peak]),
                         "dominant_period_s": float(1/frequencies[peak]),
                         "cycles_in_4s": float(4*frequencies[peak]), "first_nonpositive_acf_s": zero*.01,
                         "positive_lobe_integral_time_s": tau,
                         "max_abs_acf_1_to_2s": float(np.max(abs(acf[100:201]))),
                         "fluctuation_rms": rms(a)})
            labels.append(name+"_"+component)
            spectra.append(power)
            acfs.append(acf[:201])
    table(out/"temporal_scales.csv", rows)
    np.savez_compressed(out/"temporal_spectra.npz", labels=np.array(labels), power=np.array(spectra),
                        frequency=frequencies, autocorrelation=np.array(acfs), lag=np.arange(201)*.01)
    windows = [(2, 4), (4, 6), (6, 8), (3, 7), (3, 8), (4, 8), (5, 8), (7, 8)]
    window_rows = []
    for start, end in windows:
        for normalize in (False, True):
            for parameter in STEPS:
                m = response(cases, parameter, int(start*100), int(end*100)+1, normalize)
                for k, c in enumerate(("u", "v")):
                    window_rows.append({"parameter": parameter, "component": c, "start_s": start, "end_s": end,
                                        "n_frames": int((end-start)*100)+1,
                                        "normalization": "window_inlet" if normalize else "frozen_inlet",
                                        "relative_change_from_4_8": rel(reference[parameter][:, k], m[:, k]),
                                        "absolute_change_rms": rms(m[:, k]-reference[parameter][:, k]),
                                        "M_rms": rms(m[:, k])})
    table(out/"mean_window_sensitivity.csv", window_rows)
    return rows, window_rows


def derivative_variants(q, xi, eta, failures=None, context="manufactured"):
    methods = {"numpy_gradient": field_derivatives(q, xi, eta)}
    for window in (5, 9):
        smooth = savgol_filter(savgol_filter(q, window, 3, axis=0, mode="interp"), window, 3, axis=1, mode="interp")
        methods[f"savgol{window}_gradient"] = field_derivatives(smooth, xi, eta)
        methods[f"savgol{window}_gradient"]["q"] = smooth
    for relative_residual in (0., 1e-4, 1e-3, 1e-2):
        s = q.size*(relative_residual*float(np.std(q)))**2
        key = f"spline_residual_{relative_residual:g}"
        try:
            spline = RectBivariateSpline(xi, eta, q, kx=3, ky=3, s=s)
        except ValueError as error:
            if failures is None:
                raise
            failures.append({"context": context, "method": key, "requested_s": s,
                             "error": " ".join(str(error).split())})
            continue
        methods[key] = {d: spline(xi, eta, dx=i, dy=j) for d, i, j in
                        (("xi", 1, 0), ("eta", 0, 1), ("xixi", 2, 0), ("xieta", 1, 1), ("etaeta", 0, 2))}
        methods[key]["q"] = spline(xi, eta)
    methods["numpy_gradient"]["q"] = q
    return methods


def gradient_diagnostics(base, xi, eta, closure, out):
    rows, coarse_rows, transport_rows = [], [], []
    failures = []
    xx, yy = np.meshgrid(xi, eta, indexing="ij")
    saved = {}
    base_mean = mean_field(base).reshape(len(xi), len(eta), 2)
    ix, iy = np.arange(0, len(xi), 2), np.arange(0, len(eta), 2)
    for k, c in enumerate(("u", "v")):
        q = base_mean[:, :, k]
        methods = derivative_variants(q, xi, eta, failures, c+"_fine")
        # This is resampling of one interpolated field, not CFD mesh refinement.
        coarse = derivative_variants(q[np.ix_(ix, iy)], xi[ix], eta[iy], failures, c+"_coarse")
        for method, derivatives in methods.items():
            saved[f"{c}_{method}_q"] = derivatives["q"]
            for d in ("xi", "eta", "xixi", "xieta", "etaeta"):
                saved[f"{c}_{method}_{d}"] = derivatives[d]
                for trim in (0, 4):
                    mask = np.ones(q.shape, dtype=bool)
                    if trim:
                        mask[:] = False
                        mask[trim:-trim, trim:-trim] = True
                    rows.append({"component": c, "method": method, "derivative": d, "trim_cells": trim,
                                 "value_fit_relative_rms": rel(q[mask], derivatives["q"][mask]),
                                 "change_from_numpy_rel_l2": rel(methods["numpy_gradient"][d][mask], derivatives[d][mask]),
                                 "derivative_rms": rms(derivatives[d][mask])})
                if method in coarse:
                    fine = derivatives[d][np.ix_(ix, iy)][2:-2, 2:-2]
                    coarsened = coarse[method][d][2:-2, 2:-2]
                    coarse_rows.append({"component": c, "method": method, "derivative": d,
                                        "coarse_vs_fine_rel_l2": rel(fine, coarsened),
                                        "coarse_vs_fine_absolute_rms": rms(coarsened-fine),
                                        "meaning": "subsampling_same_interpolated_field_not_CFD_grid_convergence"})
            transports = {"R": -20*(xx*derivatives["xi"]+yy*derivatives["eta"]),
                          "cx": -10*derivatives["xi"], "cy": -10*derivatives["eta"]}
            for p, transport in transports.items():
                for trim in (0, 4):
                    mask = np.ones(q.shape, dtype=bool)
                    if trim:
                        mask[:] = False
                        mask[trim:-trim, trim:-trim] = True
                    empirical = closure[f"{p}_{c}_S_total"]-closure[f"{p}_{c}_M"]
                    total = closure[f"{p}_{c}_S_total"]
                    transport_rows.append({"parameter": p, "component": c, "method": method, "trim_cells": trim,
                                           "K_vs_TminusM_rel_l2": rel(empirical[mask], transport[mask]),
                                           "closure_over_T": rms(transport[mask]-empirical[mask])/max(rms(total[mask]), 1e-30)})
    table(out/"gradient_method_sensitivity.csv", rows)
    table(out/"gradient_resampling_sensitivity.csv", coarse_rows)
    table(out/"transport_reference_diagnostics.csv", transport_rows)
    dump(out/"spline_fit_failures.json", failures)
    np.savez_compressed(out/"gradient_fields.npz", xi=xi, eta=eta, **saved)
    return rows, coarse_rows


def checks():
    rng = np.random.default_rng(84)
    a, b = rng.normal(size=(16, 5)), rng.normal(size=(16, 5))
    h = .002
    covariance = ((a-a.mean(axis=0))*(b-b.mean(axis=0))).sum(axis=0)/15
    direct = ((a-b)/(2*h)).var(axis=0, ddof=1)/16
    expanded = (a.var(axis=0, ddof=1)+b.var(axis=0, ddof=1)-2*covariance)/(4*h*h*16)
    assert np.allclose(direct, expanded, rtol=1e-12)
    x, y = np.linspace(3, 5, 33), np.linspace(-2, 2, 64)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    exact = {"xi": 2*xx+3*yy, "eta": 3*xx+4*yy, "xixi": 2+0*xx,
             "xieta": 3+0*xx, "etaeta": 4+0*xx}
    methods = derivative_variants(xx**2+3*xx*yy+2*yy**2, x, y)
    errors = {name: max(rel(exact[d], value[d]) for d in exact) for name, value in methods.items()}
    assert max(errors.values()) < 1e-8
    # Independent affine interpolation test with a trailing time/component shape.
    points = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    target = np.array([[.2, .3], [.7, .4]])
    coeff = np.arange(12).reshape(2, 3, 2)+1.
    values = 2+points[:, 0, None, None]*coeff[0]+points[:, 1, None, None]*coeff[1]
    interp = LinearNDInterpolator(points, values)(target)
    truth = 2+target[:, 0, None, None]*coeff[0]+target[:, 1, None, None]*coeff[1]
    assert np.allclose(interp, truth, atol=1e-12)
    return {"batch_variance_identity": True, "affine_vector_interpolation": True,
            "manufactured_quadratic_derivative_max_rel_errors": errors}


def plots(out, batch, windows):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for j, p in enumerate(STEPS):
        for k, c in enumerate(("u", "v")):
            ax = axes[k, j]
            rows = [r for r in batch if r["parameter"] == p and r["component"] == c and r["normalization"] == "frozen_inlet"]
            ax.plot([r["block_duration_s"] for r in rows], [100*r["independent_SE_over_M_rms"] for r in rows], "o-", label="Independent runs (assumed)")
            ax.plot([r["block_duration_s"] for r in rows], [100*r["synchronous_paired_SE_over_M_rms"] for r in rows], "s--", label="Synchronous pairing (assumed)")
            ax.axhline(10, color="gray", ls=":", lw=1)
            ax.axhline(20, color="gray", ls="--", lw=1)
            ax.set(title=f"{p}/{c}", xlabel="Block duration [s]", ylabel="Conditional SE / reference M RMS [%]")
            ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("4-second window: 16, 8, 4, 2 batches; independence and stationarity unverified", fontsize=12)
    fig.savefig(out/"batch_uncertainty.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, c in zip(axes, ("u", "v")):
        for p in STEPS:
            rows = [r for r in windows if r["parameter"] == p and r["component"] == c and r["normalization"] == "frozen_inlet" and r["end_s"] == 8]
            rows.sort(key=lambda r: r["start_s"])
            ax.plot([r["start_s"] for r in rows], [100*r["relative_change_from_4_8"] for r in rows], "o-", label=p)
        ax.set(title=f"M, {c} component", xlabel="Averaging start [s], end fixed at 8 s", ylabel="Change from [4,8] reference [%]")
        ax.legend()
        ax.grid(alpha=.15)
    fig.savefig(out/"mean_window_sensitivity.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT/"outputs/a0_label_reliability_2026-09-09")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    self_checks = checks()
    print("Manufactured derivative, interpolation and variance checks passed", flush=True)
    if args.self_test_only:
        print(json.dumps(self_checks, indent=2))
        return
    out = args.out_dir
    (out/"time_series").mkdir(parents=True, exist_ok=True)
    with np.load(ROOT/"outputs_mgn/fd_check/data_gradient_closure_fields.npz", allow_pickle=False) as z:
        closure = {key: z[key] for key in z.files}
    xi, eta = closure["x_D"], closure["y_D"]
    xx, yy = np.meshgrid(xi, eta, indexing="ij")
    cases = {name: read_case(name, xx, yy, out, args.rebuild) for name in CASES}
    reference = {p: response(cases, p) for p in STEPS}
    reproduce = []
    for p in STEPS:
        for k, c in enumerate(("u", "v")):
            archived = closure[f"{p}_{c}_M"].ravel()
            error = rel(archived, reference[p][:, k])
            assert error < 1e-10, (p, c, error)
            reproduce.append({"parameter": p, "component": c, "M_rms": rms(archived),
                              "raw_vs_archived_M_rel_l2": error,
                              "omit_last_frame_relative_change": rel(archived, response(cases, p, 400, 800)[:, k])})
    print("All six M fields reproduced from raw time sequences", flush=True)
    table(out/"raw_reproduction.csv", reproduce)
    scales, windows = temporal_diagnostics(cases, reference, out)
    print("Temporal scales and averaging-window sensitivity complete", flush=True)
    batch, variance_identity = batch_diagnostics(cases, reference, out)
    gradients, resampling = gradient_diagnostics(cases["base"], xi, eta, closure, out)
    print("Batch-mean and baseline derivative audits complete", flush=True)
    plots(out, batch, windows)
    summary = {"protocol": {"reference_window": "4.00 through 8.00 s inclusive, 401 samples",
                             "batch_window": "4.00 through 7.99 s inclusive, 400 samples",
                             "grid_shape": list(xx.shape), "parameter_steps_m": STEPS,
                             "cx_step_convergence_identifiable": False,
                             "covariance_of_sampling_errors_identifiable": False,
                             "SE_is_conditional_not_certified_CI": True,
                             "gradient_comparisons_are_stability_not_truth_accuracy": True},
               "self_checks": self_checks, "batch_covariance_formula_max_rel_discrepancy": variance_identity,
               "raw_reproduction": reproduce, "cases": [c["metadata"] for c in cases.values()],
               "temporal_scales": scales,
               "runtime": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__},
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    dump(out/"audit_manifest.json", summary)
    print(json.dumps({"output": str(out), "reproduction": reproduce,
                      "cx_temporal_scales": [r for r in scales if r["series"] in ("cx+", "cx-", "M_cx")]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
