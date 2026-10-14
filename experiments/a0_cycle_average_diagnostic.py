"""Cycle-window diagnostics for the time-series cache from a0_label_reliability.

This is a different averaging functional, not a replacement FD ground truth.
Cycle-to-cycle variation is reported without assuming independent cycles.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.a0_label_reliability import CASES, STEPS, dump, rel, rms, table


def upward_crossings(times, signal, start=4., end=8., direction=1, level_std=0.):
    keep = (times >= start) & (times <= end)
    t, a = times[keep], signal[keep]
    a = direction*(a-a.mean()-level_std*a.std())
    indices = np.flatnonzero((a[:-1] < 0) & (a[1:] >= 0))
    return t[indices]-a[indices]*(t[indices+1]-t[indices])/(a[indices+1]-a[indices])


def time_mean(times, values, start, end):
    """Integrate a piecewise-linear sampled signal over exact fractional endpoints."""
    inside = np.flatnonzero((times > start) & (times < end))
    def interpolate(t):
        i = int(np.clip(np.searchsorted(times, t)-1, 0, len(times)-2))
        w = (t-times[i])/(times[i+1]-times[i])
        return (1-w)*values[i]+w*values[i+1]
    sample_t = np.r_[start, times[inside], end]
    sample_v = np.concatenate([interpolate(start)[None], values[inside], interpolate(end)[None]])
    return np.trapz(sample_v, sample_t, axis=0)/(end-start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=ROOT/"outputs/a0_label_reliability_2026-09-09")
    args = parser.parse_args()
    out = args.audit_dir
    # Independent integration and endpoint-crossing checks.
    t = np.linspace(0, 10, 1001)
    mean = time_mean(t, np.stack([3+2*t, -4+.5*t], axis=1), 1.234, 7.891)
    assert np.allclose(mean, [3+(1.234+7.891), -4+.25*(1.234+7.891)], atol=1e-12)
    events = upward_crossings(t, np.sin(2*np.pi*2*t+.3))
    assert np.allclose(np.diff(events), .5, atol=1e-10)
    with np.load(out/"time_series/base.npz", allow_pickle=False) as z:
        xi, eta = z["xi"], z["eta"]
        # One baseline-defined reference position, frozen across all cases.
        reference_point = int(np.argmax(z["uv"][:, 400:801, 1].var(axis=1)))
        point_xi, point_eta = np.unravel_index(reference_point, (len(xi), len(eta)))
    manifest = {"reference_point": {"point_index": reference_point,
                                   "xi": float(xi[point_xi]), "eta": float(eta[point_eta]),
                                   "selection": "maximum baseline v temporal variance on [4,8]"},
                "event": "upward crossing of each case's [4,8] mean v at the same reference position",
                "integration": "piecewise-linear trapezoidal integration with fractional endpoints",
                "caveat": "different averaging functional; neither FD step convergence nor independent-cycle uncertainty is certified",
                "self_checks": {"linear_signal_fractional_endpoint_mean": True, "sinusoidal_event_period": True},
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    means, raw_means, summaries, traces, mean_arrays, event_dict = {}, {}, [], {}, {}, {}
    alternate_means = {}
    manifest["raw_source_sha256"] = {}
    for name in CASES:
        path = out/"time_series"/(name.replace("+", "p").replace("-", "m")+".npz")
        with np.load(path, allow_pickle=False) as z:
            uv, inlet, times = z["uv"], z["inlet"], z["times"]
            manifest["raw_source_sha256"][name] = json.loads(str(z["metadata"].item()))["source_sha256"]
        trace = uv[reference_point, :, 1]
        events = upward_crossings(times, trace)
        if len(events) < 5:
            raise ValueError(f"Too few cycles in {name}")
        periods = np.diff(events)
        if periods.std()/periods.mean() > .1:
            raise ValueError(f"Crossing periods too irregular in {name}; inspect before cycle averaging")
        values = np.moveaxis(uv, 1, 0)
        def avg(start, end):
            return time_mean(times, values, start, end)/float(time_mean(times, inlet, start, end))
        raw = uv[:, 400:801].mean(axis=1)/inlet[400:801].mean()
        full = avg(events[0], events[-1])
        split = (len(events)-1)//2
        first, last = avg(events[0], events[split]), avg(events[split], events[-1])
        cycle_means = np.stack([avg(a, b) for a, b in zip(events[:-1], events[1:])])
        alternate_means[name] = {}
        for direction in (1, -1):
            for level in (-.5, 0., .5):
                alt_events = upward_crossings(times, trace, direction=direction, level_std=level)
                if len(alt_events) < 5:
                    raise ValueError("Too few events for phase robustness check")
                alternate_means[name][f"direction_{direction}_level_{level:g}"] = avg(alt_events[0], alt_events[-1])
        coarse_times = times[::2]
        coarse_events = upward_crossings(coarse_times, trace[::2])
        alternate_means[name]["temporal_subsample_dt_0.02"] = (
            time_mean(coarse_times, values[::2], coarse_events[0], coarse_events[-1]) /
            float(time_mean(coarse_times, inlet[::2], coarse_events[0], coarse_events[-1])))
        means[name] = {"full": full, "first_half": first, "last_half": last}
        raw_means[name] = raw
        traces[name] = trace
        event_dict[name] = events.tolist()
        mean_arrays[name+"_cycle_means"] = cycle_means
        mean_arrays[name+"_events"] = events
        mean_arrays[name+"_full"] = full
        mean_arrays[name+"_raw"] = raw
        for k, c in enumerate(("u", "v")):
            summaries.append({"case": name, "component": c, "n_cycles": len(periods),
                              "start_s": float(events[0]), "end_s": float(events[-1]),
                              "period_mean_s": float(periods.mean()), "period_std_s": float(periods.std(ddof=1)),
                              "raw_mean_rms": rms(raw[:, k]),
                              "cycle_vs_raw_field_rel_l2": rel(raw[:, k], full[:, k]),
                              "cycle_halves_difference_over_full_rms": rms(last[:, k]-first[:, k])/max(rms(full[:, k]), 1e-30),
                              "cycle_to_cycle_std_field_rms": rms(cycle_means[:, :, k].std(axis=0, ddof=1))})
        print(f"{name}: {len(periods)} full cycles, mean period {periods.mean():.6f} s", flush=True)
    targets, robustness = [], []
    for p, h in STEPS.items():
        raw = (raw_means[p+"+"]-raw_means[p+"-"])/(2*h)
        ms = {key: (means[p+"+"][key]-means[p+"-"][key])/(2*h) for key in means[p+"+"]}
        for key, field in ms.items():
            mean_arrays[f"M_{p}_{key}"] = field
        for k, c in enumerate(("u", "v")):
            full, first, last = (ms[key][:, k] for key in ("full", "first_half", "last_half"))
            targets.append({"parameter": p, "component": c, "raw_M_rms": rms(raw[:, k]), "cycle_M_rms": rms(full),
                            "cycle_vs_raw_M_rel_l2": rel(raw[:, k], full),
                            "cycle_halves_difference_over_cycle_M_rms": rms(last-first)/max(rms(full), 1e-30),
                            "cycle_first_half_vs_full_rel_l2": rel(full, first),
                            "cycle_last_half_vs_full_rel_l2": rel(full, last)})
        for name in alternate_means[p+"+"]:
            alternate = (alternate_means[p+"+"][name]-alternate_means[p+"-"][name])/(2*h)
            for k, c in enumerate(("u", "v")):
                robustness.append({"parameter": p, "component": c, "alternative": name,
                                   "change_from_primary_cycle_M_rel_l2": rel(ms["full"][:, k], alternate[:, k])})
    table(out/"cycle_mean_cases.csv", summaries)
    table(out/"cycle_mean_responses.csv", targets)
    table(out/"cycle_mean_phase_robustness.csv", robustness)
    np.savez_compressed(out/"cycle_mean_fields.npz", **mean_arrays)
    manifest["events"] = event_dict
    manifest["response_metrics"] = targets
    manifest["phase_robustness"] = robustness
    dump(out/"cycle_mean_manifest.json", manifest)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), constrained_layout=True)
    keep = (times >= 4) & (times <= 8)
    for name in ("base", "cx+", "cx-"):
        axes[0].plot(times[keep], traces[name][keep], label=name)
    axes[0].set(xlabel="Time [s]", ylabel="v at the fixed reference position [m/s]",
                title=f"Reference position: xi={xi[point_xi]:.3f}, eta={eta[point_eta]:.3f}")
    axes[0].legend(ncol=3)
    for name in ("cx+", "cx-"):
        es = np.array(event_dict[name])
        axes[1].plot(.5*(es[:-1]+es[1:]), np.diff(es), "o-", label=name)
    axes[1].set(xlabel="Cycle midpoint [s]", ylabel="Upward-crossing period [s]",
                title="Period variation within the available record; cycles are not assumed independent")
    axes[1].legend()
    fig.savefig(out/"cycle_phase_diagnostic.png", dpi=180)
    plt.close(fig)
    print(json.dumps(targets, indent=2), flush=True)


if __name__ == "__main__":
    main()
