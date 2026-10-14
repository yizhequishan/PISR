"""Create and validate a separate COMSOL 6.4 backward-facing-step model on phy.

Run with phy's E:\\CY\\comsol\\.venv\\Scripts\\python.exe. Existing models are
only listed; all writes target this newly created model and its own output files.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import time
import traceback
from pathlib import Path

import jpype
import mph
import numpy as np


ROOT = Path(r"E:\CY\comsol")
NAME = "backward_facing_step_2d_Re100"
OUTPUT = ROOT / "models" / "working" / f"{NAME}.mph"
RESULTS = ROOT / "results" / NAME
STATE_FILE = RESULTS / "build_state.json"

PARAMETERS = {
    "h_step": ("0.01[m]", "Step height; Reynolds-number reference length"),
    "Hin": ("h_step", "Upstream channel height"),
    "Hout": ("2*h_step", "Downstream channel height; expansion ratio 2"),
    "Lu": ("5*h_step", "Upstream length measured from the step"),
    "Ld": ("30*h_step", "Downstream length measured from the step"),
    "rho0": ("1000[kg/m^3]", "Constant Newtonian-fluid density"),
    "mu0": ("0.001[Pa*s]", "Constant dynamic viscosity"),
    "Re_h": ("100", "Reynolds number based on inlet MEAN speed and step height"),
    "Umean": ("Re_h*mu0/(rho0*h_step)", "Inlet cross-sectional mean velocity"),
    "Umax": ("1.5*Umean", "Peak of the fully developed inlet parabola"),
    "Re_Dh": ("rho0*Umean*(2*Hin)/mu0", "Reynolds number based on inlet hydraulic diameter"),
    "sel_tol": ("1e-3*h_step", "Tolerance for geometry-coordinate named selections"),
    "mesh_scale": ("1", "1: baseline mesh; 1.5: refined mesh"),
}


def sa(items):
    return jpype.JArray(jpype.JString)(list(items))


def da(items):
    return jpype.JArray(jpype.JDouble, 2)(np.asarray(items, dtype=float).tolist())


def tags(container):
    return [str(tag) for tag in container.tags()]


def record(state, stage, **values):
    state.update(stage=stage, updated=dt.datetime.now().isoformat(), **values)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print("STAGE", stage, json.dumps(values), flush=True)


def box(component, tag, label, xmin, xmax, ymin, ymax):
    selection = component.selection().create(tag, "Box")
    selection.label(label)
    selection.set("entitydim", jpype.JInt(1))
    selection.set("condition", "inside")
    for key, value in zip(("xmin", "xmax", "ymin", "ymax"), (xmin, xmax, ymin, ymax)):
        selection.set(key, value)
    return selection


def create_geometry(model, state):
    java = model.java
    java.modelPath(str(OUTPUT.parent))
    for name, (expression, description) in PARAMETERS.items():
        java.param().set(name, expression, description)
    # A reversible round trip is confined to the newly created model.
    original = str(java.param().get("mesh_scale"))
    java.param().set("mesh_scale", "1.01")
    assert abs(float(java.param().evaluate("mesh_scale")) - 1.01) < 1e-12
    java.param().set("mesh_scale", original)
    assert abs(float(java.param().evaluate("Re_h")) - 100) < 1e-12

    java.component().create("comp1", True)
    component = java.component("comp1")
    component.label("2D backward-facing step")
    component.geom().create("geom1", 2)
    geometry = component.geom("geom1")
    geometry.lengthUnit("m")
    polygon = geometry.create("fluid", "Polygon")
    polygon.label("Step channel: upstream 5h, downstream 30h")
    polygon.set("x", sa(["-Lu", "0", "0", "Ld", "Ld", "-Lu"]))
    polygon.set("y", sa(["h_step", "h_step", "0", "0", "Hout", "Hout"]))
    geometry.run()

    box(component, "sel_inlet", "Inlet", "-Lu-sel_tol", "-Lu+sel_tol", "h_step-sel_tol", "Hout+sel_tol")
    box(component, "sel_outlet", "Outlet", "Ld-sel_tol", "Ld+sel_tol", "-sel_tol", "Hout+sel_tol")
    box(component, "sel_top", "Upper wall", "-Lu-sel_tol", "Ld+sel_tol", "Hout-sel_tol", "Hout+sel_tol")
    box(component, "sel_upwall", "Upstream lower wall", "-Lu-sel_tol", "sel_tol", "h_step-sel_tol", "h_step+sel_tol")
    box(component, "sel_step", "Vertical step wall", "-sel_tol", "sel_tol", "-sel_tol", "h_step+sel_tol")
    box(component, "sel_downwall", "Downstream lower wall", "-sel_tol", "Ld+sel_tol", "-sel_tol", "sel_tol")
    wall_tags = ["sel_top", "sel_upwall", "sel_step", "sel_downwall"]
    walls = component.selection().create("sel_walls", "Union")
    walls.label("All no-slip walls")
    walls.set("entitydim", jpype.JInt(1))
    walls.set("input", sa(wall_tags))
    validate_geometry(model, state)


def validate_geometry(model, state):
    component = model.java.component("comp1")
    wall_tags = ["sel_top", "sel_upwall", "sel_step", "sel_downwall"]
    selections = {}
    for tag in ["sel_inlet", "sel_outlet"] + wall_tags + ["sel_walls"]:
        entities = [int(i) for i in component.selection(tag).entities()]
        assert len(entities) == (4 if tag == "sel_walls" else 1), (tag, entities)
        selections[tag] = entities
    separate = [selections[tag][0] for tag in ["sel_inlet", "sel_outlet"] + wall_tags]
    assert len(set(separate)) == 6
    record(state, "geometry_verified", selections=selections,
           parameters={k: {"expression": v[0], "description": v[1]} for k, v in PARAMETERS.items()})


def setup_physics(model, state):
    java = model.java
    component = java.component("comp1")
    material = component.material().create("mat1", "Common")
    material.label("Incompressible Newtonian fluid: rho=1000, mu=0.001")
    material.selection().all()
    material.propertyGroup("def").set("density", "rho0")
    material.propertyGroup("def").set("dynamicviscosity", "mu0")
    component.physics().create("spf", "LaminarFlow", "geom1")
    physics = component.physics("spf")
    physics.label("Steady incompressible Navier-Stokes")
    physics.prop("PhysicalModelProperty").set("Compressibility", "Incompressible")
    inlet = physics.create("inl1", "Inlet", 1)
    inlet.label("Fully developed parabolic inlet; mean Re_h=100")
    inlet.selection().named("sel_inlet")
    inlet.set("BoundaryCondition", "Velocity")
    inlet.set("U0in", "6*Umean*((y-h_step)/Hin)*(1-(y-h_step)/Hin)")
    outlet = physics.create("out1", "Outlet", 1)
    outlet.label("Pressure outlet: 0 Pa gauge")
    outlet.selection().named("sel_outlet")
    outlet.set("p0", "0[Pa]")
    wall = physics.create("wall_bfs", "Wall", 1)
    wall.label("No slip on all solid walls")
    wall.selection().named("sel_walls")

    for tag, selection in [("intin", "sel_inlet"), ("intout", "sel_outlet")]:
        operator = component.cpl().create(tag, "Integration")
        operator.selection().geom("geom1", 1)
        operator.selection().named(selection)
        operator.set("opname", tag)
    integral = component.cpl().create("intdom", "Integration")
    integral.selection().geom("geom1", 2)
    integral.selection().all()
    integral.set("opname", "intdom")
    java.study().create("std1")
    java.study("std1").label("Stationary laminar flow, Re_h=100")
    java.study("std1").create("stat", "Stationary")

    component.mesh().create("mesh1")
    mesh = component.mesh("mesh1")
    mesh.label("Graded triangular mesh with wall and step refinement")
    for tag, label, selection, hmax, hmin in [
        ("size_global", "Global channel size", None, "h_step/(5*mesh_scale)", "h_step/(80*mesh_scale)"),
        ("size_walls", "Wall resolution", "sel_walls", "h_step/(20*mesh_scale)", "h_step/(80*mesh_scale)"),
        ("size_step", "Step-corner resolution", "sel_step", "h_step/(40*mesh_scale)", "h_step/(100*mesh_scale)"),
    ]:
        size = mesh.create(tag, "Size")
        size.label(label)
        if selection:
            size.selection().named(selection)
        size.set("custom", "on")
        size.set("hmax", hmax)
        size.set("hmin", hmin)
        size.set("hgrad", 1.18)
        size.set("hcurve", 0.25)
        size.set("hnarrow", jpype.JInt(1))
    mesh.create("ftri1", "FreeTri")
    record(state, "physics_configured", physics=tags(component.physics()), studies=tags(java.study()))


def build_mesh(model, scale):
    model.java.param().set("mesh_scale", str(scale))
    mesh = model.java.component("comp1").mesh("mesh1")
    if "size_inlet" not in tags(mesh.feature()):
        inlet_size = mesh.create("size_inlet", "Size")
        inlet_size.label("Resolve the prescribed parabolic inlet profile")
        inlet_size.selection().named("sel_inlet")
        inlet_size.set("custom", "on")
        inlet_size.set("hmax", "h_step/(40*mesh_scale)")
        inlet_size.set("hmin", "h_step/(80*mesh_scale)")
        inlet_size.set("hgrad", 1.18)
        mesh.feature().move("size_inlet", jpype.JInt(tags(mesh.feature()).index("ftri1")))
    mesh.run()
    stats = {"scale": scale, "vertices": int(mesh.getNumVertex()), "elements": int(mesh.getNumElem())}
    assert stats["elements"] > 1000, stats
    print("MESH", json.dumps(stats), flush=True)
    return stats


def migrate_step_parameter(model, state):
    """Repair the reserved-name conflict in this script's own initial build."""
    java = model.java
    if "h" not in [str(p) for p in java.param().varnames()]:
        return
    component = java.component("comp1")
    for name, (expression, description) in PARAMETERS.items():
        java.param().set(name, expression, description)
    component.geom("geom1").feature("fluid").set("y", sa(["h_step", "h_step", "0", "0", "Hout", "Hout"]))
    for tag in tags(component.selection()):
        selection = component.selection(tag)
        if str(selection.getType()) == "Box":
            for key in ["xmin", "xmax", "ymin", "ymax"]:
                selection.set(key, re.sub(r"\bh\b", "h_step", str(selection.getString(key))))
    if "spf" in tags(component.physics()):
        component.physics("spf").feature("inl1").set("U0in", "6*Umean*((y-h_step)/Hin)*(1-(y-h_step)/Hin)")
    if "mesh1" in tags(component.mesh()):
        for tag in ["size_global", "size_walls", "size_step"]:
            feature = component.mesh("mesh1").feature(tag)
            for key in ["hmax", "hmin"]:
                feature.set(key, re.sub(r"\bh\b", "h_step", str(feature.getString(key))))
    java.param().remove("h")
    component.geom("geom1").run()
    validate_geometry(model, state)
    record(state, "parameter_name_corrected")


def solve(model):
    java = model.java
    if not tags(java.sol()):
        java.study("std1").createAutoSequences("all")
    for sol_tag in tags(java.sol()):
        solver = java.sol(sol_tag)
        for tag in tags(solver.feature()):
            feature = solver.feature(tag)
            if str(feature.getType()) == "Stationary":
                feature.set("stol", "1e-6")
    started = time.monotonic()
    java.study("std1").run()
    print("SOLVED_SECONDS", time.monotonic() - started, flush=True)


def interpolate(model, expressions, coordinates):
    numerical = model.java.result().numerical()
    if "interp_bfs" not in tags(numerical):
        numerical.create("interp_bfs", "Interp")
    interp = model.java.result().numerical("interp_bfs")
    interp.set("expr", sa(expressions))
    interp.set("coord", da(coordinates))
    raw = np.asarray(interp.getData(), dtype=float)
    print("INTERP_SHAPE", list(raw.shape), flush=True)
    return raw.reshape(len(expressions), -1)


def scalar(model, expression):
    value = np.asarray(model.evaluate(expression), dtype=float).reshape(-1)
    assert value.size == 1 and np.isfinite(value[0]), (expression, value)
    return float(value[0])


def verify(model, mesh_stats, label):
    h = float(model.java.param().evaluate("h_step"))
    mean = float(model.java.param().evaluate("Umean"))
    qin = scalar(model, "comp1.intin(u)")
    qout = scalar(model, "comp1.intout(u)")
    area = scalar(model, "comp1.intdom(1)")
    flux_error = abs(qout - qin) / abs(qin)
    assert abs(area / (65 * h * h) - 1) < 1e-8, area
    assert abs(qin / (mean * h) - 1) < 0.002, qin
    assert flux_error < 0.002, flux_error
    velocity = np.asarray(model.evaluate("u"), dtype=float)
    pressure = np.asarray(model.evaluate("p"), dtype=float)
    assert np.isfinite(velocity).all() and np.isfinite(pressure).all()
    assert float(velocity.min()) < -0.001 * mean, "No reverse streamwise flow found"

    # Approach the wall from inside the fluid; compare two offsets to quantify
    # the difference between the near-wall velocity zero and wall reattachment.
    x_h = np.linspace(0.03, 20, 1998)
    roots = []
    nearwall_data = [x_h]
    for offset in [0.002, 0.001]:
        u = interpolate(model, ["u"], np.vstack([x_h * h, np.full_like(x_h, offset * h)]))[0]
        assert np.isfinite(u).all(), "Nonfinite near-wall velocity"
        crossings = np.flatnonzero((u[:-1] < 0) & (u[1:] >= 0))
        assert len(crossings), "No reattachment detected in first 20 step heights"
        # Ignore any small secondary corner bubble; choose the end of the main
        # negative interval containing the minimum sampled near-wall velocity.
        minimum = int(np.argmin(u))
        valid = [int(i) for i in crossings if i >= minimum]
        assert valid, "No positive crossing downstream of recirculation minimum"
        i = valid[0]
        root = x_h[i] - u[i] * (x_h[i + 1] - x_h[i]) / (u[i + 1] - u[i])
        roots.append(float(root))
        nearwall_data.append(u / mean)
    np.savetxt(RESULTS / f"nearwall_{label}.csv", np.column_stack(nearwall_data), delimiter=",",
               header="x_over_h,u_over_Umean_at_y_0p002h,u_over_Umean_at_y_0p001h", comments="")
    assert abs(roots[0] - roots[1]) < 0.03, roots
    result = {
        "mesh": mesh_stats, "inlet_flux_m2_s": qin, "outlet_flux_m2_s": qout,
        "relative_flux_imbalance": flux_error, "fluid_area_m2": area,
        "inlet_profile_relative_discretization_error": abs(qin / (mean * h) - 1),
        "u_min_m_s": float(velocity.min()), "u_max_m_s": float(velocity.max()),
        "p_min_Pa": float(pressure.min()), "p_max_Pa": float(pressure.max()),
        "mean_inlet_pressure_Pa": scalar(model, "comp1.intin(p)/Hin"),
        "reattachment_x_over_h": roots[-1], "reattachment_offsets_y_over_h": [0.002, 0.001],
        "reattachment_roots_x_over_h": roots,
        "reattachment_method": "negative-to-positive near-wall streamwise-velocity crossing; two fluid-side offsets",
    }
    print("VALIDATION", label, json.dumps(result), flush=True)
    return result


def create_results(model, state):
    java = model.java
    for tag, label, expression in [
        ("pg_bfs_speed", "Velocity magnitude and streamlines", "spf.U"),
        ("pg_bfs_u", "Streamwise velocity: negative values identify recirculation", "u"),
        ("pg_bfs_p", "Gauge pressure", "p"),
    ]:
        if tag in tags(java.result()):
            continue
        group = java.result().create(tag, "PlotGroup2D")
        group.label(label)
        group.create("surf1", "Surface").set("expr", expression)
        if tag == "pg_bfs_speed":
            stream = group.create("str1", "Streamline")
            stream.set("expr", sa(["u", "v"]))
        group.run()

    # Export a uniform rectangular grid, omitting the solid lower-left block.
    x_h = np.linspace(-5, 30, 701)
    y_h = np.linspace(0, 2, 81)
    xx, yy = np.meshgrid(x_h, y_h)
    fluid = (xx >= 0) | (yy >= 1)
    # Nudge points on walls into the fluid for unambiguous interpolation.
    eps = 1e-7
    xc = np.clip(xx[fluid], -5 + eps, 30 - eps)
    yc = np.clip(yy[fluid], eps, 2 - eps)
    yc[(xc < 0) & (yc <= 1)] = 1 + eps
    xc[(np.abs(xc) < eps) & (yc < 1)] = eps
    h = float(java.param().evaluate("h_step"))
    values = interpolate(model, ["u", "v", "p"], np.vstack([xc * h, yc * h]))
    assert values.shape == (3, int(fluid.sum())), values.shape
    assert np.isfinite(values).all(), "Nonfinite exported solution samples"
    np.savez_compressed(RESULTS / "flow_field.npz", x_over_h=xx[fluid], y_over_h=yy[fluid],
                        u=values[0], v=values[1], p=values[2], h=h,
                        Umean=float(java.param().evaluate("Umean")))
    np.savetxt(RESULTS / "flow_field.csv", np.column_stack([xx[fluid], yy[fluid], values.T]),
               delimiter=",", header="x_over_h,y_over_h,u_m_s,v_m_s,p_Pa", comments="")
    model.save(OUTPUT)
    assert OUTPUT.is_file() and OUTPUT.stat().st_size > 100000
    state["saved_file"] = str(OUTPUT)
    state["saved_bytes"] = OUTPUT.stat().st_size
    state["result_plot_tags"] = tags(java.result())
    state["solver_tags"] = tags(java.sol())
    state["dataset_tags"] = tags(java.result().dataset())
    state["parameters"] = {name: {"expression": str(java.param().get(name)), "description": description}
                           for name, (_, description) in PARAMETERS.items()}
    record(state, "complete")
    print("SAVED", OUTPUT, flush=True)


def main():
    global OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", choices=["setup", "mesh", "solve", "refine", "finalize"])
    parser.add_argument("--standalone", action="store_true", help="Continue this build in an isolated eight-core COMSOL process")
    args = parser.parse_args()
    print("CONNECTING", flush=True)
    client = (mph.Client(version="6.4", cores=8) if args.standalone
              else mph.Client(version="6.4", host="127.0.0.1", port=2036))
    existing = [str(m.name()) for m in client.models()]
    print("EXISTING_MODELS", existing, flush=True)
    if args.standalone:
        if not args.resume:
            raise RuntimeError("Standalone continuation requires --resume")
        checkpoint = OUTPUT
        OUTPUT = OUTPUT.with_name(OUTPUT.stem + "_solved.mph")
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        model = client.load(OUTPUT if OUTPUT.exists() else checkpoint)
        model.rename(OUTPUT.stem)
        state.update(model_name=OUTPUT.stem, model_tag=str(model.java.tag()),
                     execution_mode="standalone", cores=8, source_checkpoint=str(checkpoint))
        client.java.showProgress(str(ROOT / "logs" / "bfs_standalone_progress.log"))
        record(state, "standalone_continuation")
    elif args.resume:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        matches = [m for m in client.models() if str(m.java.tag()) == state["model_tag"] and str(m.name()) == NAME]
        if len(matches) != 1:
            raise RuntimeError("The exact new model from build_state.json is not present")
        model = matches[0]
        migrate_step_parameter(model, state)
        if args.resume == "setup":
            model.java.param().set("sel_tol", PARAMETERS["sel_tol"][0])
            validate_geometry(model, state)
            setup_physics(model, state)
    else:
        if OUTPUT.exists() or NAME in existing or STATE_FILE.exists():
            raise RuntimeError("Output or model already exists; use --resume only for this build")
        RESULTS.mkdir(parents=True, exist_ok=False)
        model = client.create(NAME)
        state = {"model_name": NAME, "model_tag": str(model.java.tag()),
                 "existing_models_preserved": existing, "created": dt.datetime.now().isoformat()}
        record(state, "created")
        create_geometry(model, state)
        setup_physics(model, state)

    if args.resume in (None, "setup", "mesh", "solve"):
        stats = build_mesh(model, 1)
        record(state, "baseline_mesh_verified", baseline_mesh=stats)
        model.save(OUTPUT)
        solve(model)
        model.save(OUTPUT)
        baseline = verify(model, stats, "baseline")
        record(state, "baseline_verified", baseline=baseline)
    if args.resume in (None, "setup", "mesh", "solve", "refine"):
        stats = build_mesh(model, 1.5)
        record(state, "refined_mesh_verified", refined_mesh=stats)
        solve(model)
        model.save(OUTPUT)
        refined = verify(model, stats, "refined")
        change = abs(refined["reattachment_x_over_h"] / state["baseline"]["reattachment_x_over_h"] - 1)
        record(state, "refined_verified", refined=refined, reattachment_relative_mesh_change=change)
        assert change < 0.02, f"Reattachment mesh sensitivity is too high: {change}"
    create_results(model, state)
    print("PRESERVED_MODELS", [str(m.name()) for m in client.models()], flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
