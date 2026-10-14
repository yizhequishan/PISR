"""Independent numerical checks of the saved research claims; no CFD/PySR training.

Inputs are read-only. New evidence is written under --out-dir. Legacy archives
are deliberately read as historical evidence, never upgraded to verified runs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import sympy as sp
from scipy.interpolate import interp1d, LinearNDInterpolator

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pipeline_contracts import atomic_json, array_hash, file_hash, read_json, expression_record, evaluate
from experiments import a0_m_reconstructability as a0
from experiments import p1_observability_nullspace as p1
from experiments import p2_analytic_derivative_loss as p2
from step6_phi_R_analysis import dimensional_response


def rel(truth, estimate):
    return float(np.linalg.norm(estimate-truth) / max(np.linalg.norm(truth), 1e-15))


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def theoretical_checks(out):
    xi, eta, radius, cx, cy = sp.symbols("xi eta radius cx cy")
    xp, yp = sp.symbols("xp yp")
    psi = sp.sin(xi)*sp.sin(eta) + radius*eta*sp.exp(xi/10) + cx*xi*eta + cy*eta**2
    velocity = [sp.diff(psi, eta), -sp.diff(psi, xi)]
    uin = sp.Rational(7, 200)/radius
    base = {radius: .05, cx: .2, cy: .2}
    xx, yy = np.meshgrid(np.linspace(3, 5, 21), np.linspace(-1, 1, 23), indexing="ij")
    physical = [base[cx]+.1*xx, base[cy]+.1*yy]
    rows, exact_identities = [], []
    for parameter in (radius, cx, cy):
        for component, q in zip(("u", "v"), velocity):
            qphysical = (uin*q).subs({xi: (xp-cx)/(2*radius), eta: (yp-cy)/(2*radius)})
            exact_physical = sp.diff(qphysical, parameter)
            k = (-1/radius*(xi*sp.diff(q, xi)+eta*sp.diff(q, eta)) if parameter == radius
                 else -sp.diff(q, xi if parameter == cx else eta)/(2*radius))
            expected = uin*(k+sp.diff(q, parameter)) + sp.diff(uin, parameter)*q
            expected_physical = expected.subs({xi: (xp-cx)/(2*radius), eta: (yp-cy)/(2*radius)})
            identity = sp.simplify(exact_physical-expected_physical) == 0
            exact_identities.append(identity)
            exact = np.asarray(sp.lambdify((xi, eta), expected.subs(base), "numpy")(xx, yy), float)
            func = sp.lambdify((xp, yp, radius, cx, cy), qphysical, "numpy")
            previous = None
            for h in (.001, .0005, .00025):
                plus, minus = dict(base), dict(base)
                plus[parameter] += h
                minus[parameter] -= h
                derivative = (func(*physical, *[plus[p] for p in (radius,cx,cy)])
                              -func(*physical, *[minus[p] for p in (radius,cx,cy)]))/(2*h)
                error = rel(exact, derivative)
                rows.append({"parameter": str(parameter), "component": component, "h": h,
                             "relative_error": error, "previous_error_ratio": previous/error if previous else None,
                             "symbolic_chain_rule_exact": identity})
                previous = error
    pd.DataFrame(rows).to_csv(out / "manufactured_chain_rule.csv", index=False)
    assert all(exact_identities)
    ratios=[row['previous_error_ratio'] for row in rows if row['previous_error_ratio'] is not None]
    assert all(3.9 < value < 4.1 for value in ratios)
    rng = np.random.default_rng(20260908)
    mapped_plus, mapped_minus, own_plus, own_minus = rng.normal(size=(4, 1000))
    total, intrinsic = mapped_plus-mapped_minus, own_plus-own_minus
    path_k = mapped_plus-own_plus+own_minus-mapped_minus
    psi0 = sp.sin(xi)*sp.sin(eta)
    family_a, family_b = psi0, psi0+(radius-sp.Rational(1,20))*xi*eta
    same_baseline = sp.simplify((family_b-family_a).subs(radius, sp.Rational(1,20))) == 0
    result = {"manufactured_chain_rule_all_exact": all(exact_identities),
              "finite_difference_halving_error_ratio_range":[min(ratios),max(ratios)],
              "arbitrary_random_fields_path_closure": rel(total, intrinsic+path_k),
              "identical_base_divergence_free_families": same_baseline,
              "different_intrinsic_u_response": str(sp.diff(family_b-family_a, radius, eta)),
              "different_intrinsic_v_response": str(-sp.diff(family_b-family_a, radius, xi)),
              "counterexample_scope": "Kinematics only; these families need not solve the same PDE/boundary-value problem."}
    atomic_json(out / "theory_checks.json", result)
    return result


def p0_and_model_checks(out):
    old = load_npz(REPO / "outputs/p0_derivative_closure/closure_fields.npz")
    rows = []
    for parameter in ("R", "cx", "cy"):
        mask = old[parameter+"_mask"]
        t, m = old[parameter+"_T"][mask], old[parameter+"_M"][mask]
        for source in ("path", "data", "symbolic"):
            rows.append({"parameter": parameter, "grid": "16x64", "K_source": source,
                         "closure_rel_l2": rel(t, m+old[parameter+"_K_"+source][mask])})
    pd.DataFrame(rows).to_csv(out / "p0_archive_recomputed.csv", index=False)
    fd = load_npz(REPO / "outputs_mgn/fd_check/fd_sensitivity_fields.npz")
    closure = load_npz(REPO / "outputs_mgn/fd_check/data_gradient_closure_fields.npz")
    xx, yy = np.meshgrid(fd["x_D"], fd["y_D"], indexing="ij")
    xi, eta = sp.symbols("x_D y_D")
    run = read_json(REPO / "outputs_mgn/fd_check/decomposition_run_info.json")
    psi = sp.sympify(run["psi_sympy"])
    base_expr = {"u": sp.diff(psi, eta), "v": -sp.diff(psi, xi)}
    candidates, gradients, dimensional = [], [], []
    base_mse = {}
    for component in ("u", "v"):
        t, m = fd["S_total_R_"+component], fd["M_R_"+component]
        empirical = t-m
        kappa = float(np.linalg.norm(empirical)/np.linalg.norm(t))
        qexpr = base_expr[component]
        recalculated_k = evaluate(-20*(xi*sp.diff(qexpr,xi)+eta*sp.diff(qexpr,eta)), xx, yy)
        base_mse[component] = float(np.mean((evaluate(qexpr, xx, yy)-fd[component+"_base"])**2))
        for label, k in (("saved_psi", fd["K_R_"+component+"_psi"]),
                         ("analytic_psi", recalculated_k), ("data", fd["K_R_"+component+"_data"])):
            error = rel(empirical, k)
            gradients.append({"component": component, "model": label,
                              "epsilon_K": error, "kappa": kappa, "closure": rel(t,m+k),
                              "identity_gap": abs(rel(t,m+k)-kappa*error),
                              "K_alone_vs_T": rel(t,k), "mean_abs_T": float(np.mean(abs(t))),
                              "mean_abs_M": float(np.mean(abs(m))), "mean_abs_K": float(np.mean(abs(k)))})
    # Recompute each old expression on exactly the current FD target, without claiming a new run.
    table = pd.read_csv(REPO / "outputs_mgn/fd_check/phi_R_pysr/results_strategy_b.csv")
    u0, up = float(fd["U0"]), float(fd["Uprime"])
    for _, row in table.iterrows():
        try:
            record = expression_record(row)
            pred = {c: evaluate(record["sympy" if c == "u" else "v_sympy"], xx, yy) for c in ("u", "v")}
            scores = {c: rel(fd["M_R_"+c], pred[c]) for c in ("u", "v")}
            candidates.append({"equation_index": int(row.equation_index), "complexity": int(row.complexity),
                               "u_rel_l2": scores["u"], "v_rel_l2": scores["v"],
                               "score": sum(v*v for v in scores.values()), "psi_sympy": record["psi_sympy"],
                               "expression_sha256": record["expression_sha256"]})
        except (ValueError, FloatingPointError, TypeError):
            continue
    chosen = min(candidates, key=lambda row: (row["score"],row["complexity"]))
    phi = sp.sympify(chosen["psi_sympy"])
    _, dim_expr, dim_values = dimensional_response(phi, psi, xx, yy, .1, u0, up)
    for c in ("u", "v"):
        m_symbolic = evaluate(sp.diff(phi,eta) if c == "u" else -sp.diff(phi,xi),xx,yy)
        t_symbolic = fd["K_R_"+c+"_psi"]+m_symbolic
        hybrid_dim = u0*t_symbolic + up*fd[c+"_base"]
        dimensional.append({"component": c, "M_rel_l2": rel(fd["M_R_"+c],m_symbolic),
                            "total_symbolic_rel_l2": rel(fd["S_total_R_"+c],t_symbolic),
                            "dimensional_fully_symbolic_rel_l2": rel(fd["S_total_R_"+c+"_dim"],dim_values[c]),
                            "dimensional_CFD_base_hybrid_rel_l2": rel(fd["S_total_R_"+c+"_dim"],hybrid_dim),
                            "dimensional_formula_hybrid_gap_rms": float(np.sqrt(np.mean((dim_values[c]-hybrid_dim)**2)))})
    pd.DataFrame(candidates).to_csv(out / "phi_candidates_current_target.csv",index=False)
    pd.DataFrame(dimensional).to_csv(out / "phi_formula_metrics.csv",index=False)
    pd.DataFrame(gradients).to_csv(out / "gradient_closure_recomputed.csv",index=False)
    # MLP mean-field and per-seed metrics on the same target as psi.
    mlp_rows, predictions, transports = [], [], []
    for seed in (0,1,2):
        directory = REPO / f"outputs_mgn/comsol/eta_sym_psi_mlp/seed_{seed}"
        pred = load_npz(directory / "predictions.npz")["u_pred"]
        k = load_npz(directory / "sensitivity_fields.npz")["S_R_vals"]
        assert pred.shape == xx.shape == k.shape and np.isfinite(pred).all() and np.isfinite(k).all()
        predictions.append(pred)
        transports.append(k)
        mlp_rows.append({"model":"MLP","seed":str(seed),"u_mse_current":float(np.mean((pred-fd["u_base"])**2)),
                         "epsilon_K":rel(closure["R_u_K_empirical"],k),
                         "total_closure":rel(closure["R_u_S_total"],closure["R_u_M"]+k)})
    for label, pred, k in (("ensemble_mean",np.mean(predictions,axis=0),np.mean(transports,axis=0)),
                           ("psi",evaluate(base_expr["u"],xx,yy),fd["K_R_u_psi"])):
        mlp_rows.append({"model":label,"seed":"mean" if label == "ensemble_mean" else "historical",
                         "u_mse_current":float(np.mean((pred-fd["u_base"])**2)),
                         "epsilon_K":rel(closure["R_u_K_empirical"],k),
                         "total_closure":rel(closure["R_u_S_total"],closure["R_u_M"]+k)})
    pd.DataFrame(mlp_rows).to_csv(out / "mlp_current_target.csv",index=False)
    return {"old_P0_grid":[len(old["x_D"]),len(old["y_D"])], "FD_grid":list(xx.shape),
            "base_psi_current_mse":base_mse,"selected_phi":chosen,"dimensional":dimensional,
            "source_status":"legacy arrays; recalculated metrics are conditional on archived CFD targets"}


def fd_cache_reconstruction(out):
    from step5_fd_sensitivity_check import CASES
    fd=load_npz(REPO / "outputs_mgn/fd_check/fd_sensitivity_fields.npz")
    closure=load_npz(REPO / "outputs_mgn/fd_check/data_gradient_closure_fields.npz")
    xx,yy=np.meshgrid(fd["x_D"],fd["y_D"],indexing="ij")
    own,fixed,metadata={},{},[]
    for name,info in CASES.items():
        path=REPO/"outputs_mgn/fd_check/case_cache"/(name.replace('+','p').replace('-','m')+'.npz')
        cache=load_npz(path)
        source=REPO/info["file"]
        stat=source.stat()
        cache_source_matches=str(cache['cache_key'].item()).split(':')[:2]==[str(stat.st_size),str(stat.st_mtime_ns)]
        coords=cache['coords']
        inlet=coords[:,0]<=coords[:,0].min()+.02
        uin=float(cache['U_in'])
        reestimated=float(cache['u_mean'][inlet].mean())
        metadata.append({'case':name,'cached_mean_file_sha256':file_hash(path),
                         'raw_size_mtime_match':cache_source_matches,'U_in':uin,
                         'inlet_reestimate_abs_difference':abs(reestimated-uin),
                         'n_time':int(cache['n_time'])})
        interpolate=LinearNDInterpolator(coords,np.column_stack([cache['u_mean'],cache['v_mean']])/uin)
        own[name]=interpolate(info['cx']+info['D']*xx,info['cy']+info['D']*yy)
        fixed[name]=interpolate(.2+.1*xx,.2+.1*yy)
        assert np.isfinite(own[name]).all() and np.isfinite(fixed[name]).all()
    rows=[]
    for parameter,key,factor in (('R','D',4.),('cx','cx',2.),('cy','cy',2.)):
        step=(CASES[parameter+'+'][key]-CASES[parameter+'-'][key])/factor
        t=(fixed[parameter+'+']-fixed[parameter+'-'])/(2*step)
        m=(own[parameter+'+']-own[parameter+'-'])/(2*step)
        for index,component in enumerate(('u','v')):
            for quantity,value in (('S_total',t[...,index]),('M',m[...,index])):
                target=closure[f'{parameter}_{component}_{quantity}']
                rows.append({'parameter':parameter,'component':component,'quantity':quantity,
                             'relative_difference':rel(target,value),
                             'max_abs_difference':float(np.max(abs(target-value)))})
    for index,c in enumerate(('u','v')):
        rows.append({'parameter':'base','component':c,'quantity':'q_star',
                     'relative_difference':rel(fd[c+'_base'],own['base'][...,index]),
                     'max_abs_difference':float(np.max(abs(fd[c+'_base']-own['base'][...,index])))})
    pd.DataFrame(rows).to_csv(out/'fd_rebuilt_from_cached_means.csv',index=False)
    atomic_json(out/'fd_cached_mean_sources.json',metadata)
    maximum_difference=max(r['relative_difference'] for r in rows)
    assert maximum_difference<1e-10
    return {'maximum_field_relative_difference':maximum_difference,
            'all_cache_raw_stat_matches':all(r['raw_size_mtime_match'] for r in metadata),
            'limitation':'Nodal time means were reused; original time sequences were not re-averaged.'}


def p2_frozen_candidate_check(out):
    directory = REPO / "outputs/p2_analytic_derivative_loss"
    historical = read_json(directory / "metrics.json")
    settings = historical["settings"]
    functions = []
    x,y = sp.symbols("x y")
    for source, filename in (("value_only","raw_value_equations.csv"),("analytic_derivative","raw_derivative_equations.csv")):
        for index,row in pd.read_csv(directory/filename).iterrows():
            expr=sp.sympify(row.sympy_format)
            functions.append({"source":source,"equation_index":int(index),"complexity":int(row.complexity),
                              "expression":str(expr),"function":sp.lambdify((x,y),(expr,sp.diff(expr,x),sp.diff(expr,y)),"numpy")})
    def errors(points):
        vx,vy=points.T
        truth=[p2.truth(vx,vy),p2.truth_dx(vx,vy),p2.truth_dy(vx,vy)]
        rows=[]
        for f in functions:
            vals=[np.broadcast_to(np.asarray(v,float),(len(points),)) for v in f["function"](vx,vy)]
            mse=[float(np.mean((v-t)**2)) for v,t in zip(vals,truth)]
            rows.append({**{k:v for k,v in f.items() if k!="function"},"value_mse":mse[0],
                         "gradient_mse":(mse[1]+mse[2])/2,
                         "h1":mse[0]/settings["value_scale"]+.5*(mse[1]/settings["dx_scale"]+mse[2]/settings["dy_scale"])})
        return pd.DataFrame(rows)
    validations=[np.random.default_rng(2026090800+i).uniform(-1,1,(512,2)) for i in range(20)]
    selections=[]
    # Freeze every validation decision before evaluating the new test points.
    for repeat, points in enumerate(validations):
        frame=errors(points)
        for source in ("value_only","analytic_derivative"):
            subset=frame[frame.source==source]
            for criterion in ("value_mse","h1"):
                row=subset.loc[subset[criterion].idxmin()]
                selections.append({"repeat":repeat,"source":source,"criterion":criterion,
                                   "equation_index":int(row.equation_index),"complexity":int(row.complexity),
                                   "validation_score":float(row[criterion]),"expression":row.expression})
    test=np.random.default_rng(73019268).uniform(-1,1,(50000,2))
    test_frame=errors(test)
    rows=[]
    for selected in selections:
        row=test_frame[(test_frame.source==selected["source"])&(test_frame.equation_index==selected["equation_index"])].iloc[0]
        rows.append({**selected,"test_value_mse":float(row.value_mse),"test_gradient_mse":float(row.gradient_mse),"test_h1":float(row.h1)})
    frame=pd.DataFrame(rows)
    frame.to_csv(out / "p2_factorial_holdout.csv",index=False)
    test_frame.to_csv(out / "p2_all_candidates_test_diagnostic.csv",index=False)
    protocol={"role":"retrospective_fixed_candidate_revalidation_not_new_search",
              "validation_seeds":[2026090800+i for i in range(20)],"validation_size":512,
              "test_seed":73019268,"test_size":len(test),"test_sha256":array_hash(test),
              "validation_sha256":[array_hash(p) for p in validations],
              "search_seed_replicates":1,"historical_searches_used_different_seeds":True,
              "note":"20 selection repetitions are not 20 PySR training repetitions; no causal training-loss estimate."}
    atomic_json(out / "p2_protocol.json",protocol)
    return frame.groupby(["source","criterion"]).agg(
        test_value_mse_mean=("test_value_mse","mean"), test_gradient_mse_mean=("test_gradient_mse","mean"),
        test_gradient_mse_min=("test_gradient_mse","min"),test_gradient_mse_max=("test_gradient_mse","max")).reset_index()


def a0_stress_checks(out):
    xi, eta, base, targets, _ = a0.load_fields(a0.DEFAULT_FD,a0.DEFAULT_CLOSURE)
    n=len(xi)
    rows=[]
    for parameter in ("R","cx","cy"):
        names,raw=a0.raw_feature_fields(parameter,xi,eta,base,.1)
        truth=np.stack([targets[parameter+"_u"],targets[parameter+"_v"]],axis=-1)
        for scheme, groups, gap in (("LOXO",[np.array([i]) for i in range(n)],0),
                                    ("blocked5",np.array_split(np.arange(n),5),0),
                                    ("blocked5_gap2",np.array_split(np.arange(n),5),2)):
            predicted={key:np.empty_like(truth) for key in ("ridge_local_full","linear_M_interpolation","cubic_M_interpolation")}
            for held in groups:
                train=np.ones(n,bool)
                train[max(0,held.min()-gap):min(n,held.max()+gap+1)]=False
                design_train,design_test=a0.preprocess_fold(raw[train].reshape(-1,raw.shape[-1]),raw[held].reshape(-1,raw.shape[-1]),names,2)
                pred=a0.ridge_path_predict(design_train,truth[train].reshape(-1,2),design_test,np.array([1e-6]))[0]
                predicted["ridge_local_full"][held]=pred.reshape(len(held),len(eta),2)
                for method in ("linear","cubic"):
                    predicted[method+"_M_interpolation"][held]=interp1d(xi[train],truth[train],axis=0,kind=method,fill_value="extrapolate")(xi[held])
            for method,pred in predicted.items():
                for j,c in enumerate(("u","v")):
                    interior=np.arange(7,n-7)
                    rows.append({"parameter":parameter,"component":c,"scheme":scheme,"method":method,
                                 "pooled_rel_l2":rel(truth[...,j],pred[...,j]),
                                 "middle_19_slices_rel_l2":rel(truth[interior,:,j],pred[interior,:,j])})
    pd.DataFrame(rows).to_csv(out / "a0_spatial_stress.csv",index=False)
    return rows


def p3_reconstruction(out):
    saved=load_npz(REPO / "outputs_hd_p3/point_time_means.npz")
    points=saved["points"]
    axes=load_npz(REPO / "outputs_hd_p3/p3_core_fields.npz")
    xi,eta=axes["xi"],axes["eta"]
    xx,yy=np.meshgrid(xi,eta,indexing="ij")
    masks={"inclusive_81x81":(xx>=3-1e-10)&(xx<=5+1e-10)&(abs(yy)<=1+1e-10),
           "legacy_strict_80x80":(xx>=3)&(xx<=5)&(abs(yy)<=1)}
    metrics=[]
    for h_channel in (4,8):
        cy=.05*h_channel
        own,fixed={},{}
        for sr in (-1.,-.5,0.,.5,1.):
            p=points[(points[:,2]==h_channel)&(points[:,3]==sr)]
            radius=.05+.001*sr
            q=p[:,4]/(.075/radius)
            interpolation=LinearNDInterpolator(p[:,:2],q)
            own[sr]=interpolation(.2+2*radius*xx,cy+2*radius*yy)
            fixed[sr]=interpolation(.2+.1*xx,cy+.1*yy)
        dxi,deta=np.gradient(own[0.],xi,eta,edge_order=2)
        k=-20*(xx*dxi+yy*deta)
        schemes={}
        for label,sr,step in (("inner_1pct",.5,.0005),("outer_2pct",1.,.001)):
            schemes[label]=((fixed[sr]-fixed[-sr])/(2*step),(own[sr]-own[-sr])/(2*step))
        schemes["five_point"]=tuple((4*a-b)/3 for a,b in zip(schemes["inner_1pct"],schemes["outer_2pct"]))
        for label,(total,m) in schemes.items():
            for mask_name,core in masks.items():
                assert np.isfinite(total[core]).all() and np.isfinite(m[core]).all() and np.isfinite(k[core]).all()
                symmetric_denominator=sum(np.linalg.norm(q[core]) for q in (total,m,k))
                metrics.append({"H_over_D":h_channel,"scheme":label,"mask":mask_name,"n_points":int(core.sum()),
                                "rho_M_over_K":float(np.linalg.norm(m[core])/np.linalg.norm(k[core])),
                                "closure_relative_to_T":rel(total[core],(m+k)[core]),
                                "closure_symmetric":float(np.linalg.norm((total-m-k)[core])/symmetric_denominator),
                                "K_vs_T":rel(total[core],k[core]),
                                "E_step_disagreement":rel(schemes["inner_1pct"][0][core],schemes["outer_2pct"][0][core]),
                                "M_step_disagreement":rel(schemes["inner_1pct"][1][core],schemes["outer_2pct"][1][core])})
    pd.DataFrame(metrics).to_csv(out / "p3_independent_reconstruction.csv",index=False)
    variants=read_json(REPO/'outputs_hd_p3/step_variant_metrics.json')
    complete=read_json(REPO/'outputs_hd_p3/p3_metrics.json')
    differences=[]
    for row in metrics:
        if row['mask']=='legacy_strict_80x80':
            recorded=variants[str(row['H_over_D'])][row['scheme']]
            differences.extend([abs(row['rho_M_over_K']-recorded['rho']),abs(row['closure_symmetric']-recorded['closure'])])
        elif row['scheme']=='five_point':
            recorded=complete[str(row['H_over_D'])]['full']
            differences.extend([abs(row['rho_M_over_K']-recorded['rho_core']),abs(row['closure_symmetric']-recorded['closure'])])
    maximum_difference=max(differences)
    assert maximum_difference<1e-10
    weighted=np.average(saved['block_uv'],axis=1,weights=[81,80,80,80,80])
    block_mean_gap=float(np.max(abs(weighted-points[:,4:6])))
    atomic_json(out/'p3_reproduction_checks.json',{'maximum_metric_absolute_difference':maximum_difference,
                'block_weighted_mean_max_abs_difference':block_mean_gap,
                'closure_definition':'norm(T-M-K)/(norm(T)+norm(M)+norm(K))',
                'mask_definitions':{'p3_metrics_full':'inclusive_81x81','step_variant_metrics':'legacy_strict_80x80'}})
    lobo=read_json(REPO / "outputs_hd_p3/leave_one_block_out.json")["replicates"]
    lobo_rows=[]
    for scheme in ("inner","outer","five"):
        differences=[b["rho"]-a["rho"] for a,b in zip(lobo["4"][scheme],lobo["8"][scheme])]
        lobo_rows.append({"scheme":scheme,"minimum_delta":min(differences),"maximum_delta":max(differences),
                          "both_signs":min(differences)<0<max(differences)})
    atomic_json(out / "p3_time_block_signs.json",lobo_rows)
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir",type=Path,default=REPO/"outputs/logic_validation_2026-09-08")
    args=parser.parse_args()
    out=args.out_dir
    out.mkdir(parents=True,exist_ok=True)
    source_paths=[REPO/name for name in (
        "outputs/p0_derivative_closure/closure_fields.npz",
        "outputs_mgn/fd_check/fd_sensitivity_fields.npz",
        "outputs_mgn/fd_check/data_gradient_closure_fields.npz",
        "outputs_mgn/fd_check/decomposition_run_info.json",
        "outputs_mgn/fd_check/phi_R_pysr/results_strategy_b.csv",
        "outputs/p2_analytic_derivative_loss/raw_value_equations.csv",
        "outputs/p2_analytic_derivative_loss/raw_derivative_equations.csv",
        "outputs/p2_analytic_derivative_loss/metrics.json",
        "outputs_hd_p3/point_time_means.npz", "outputs_hd_p3/p3_core_fields.npz",
        "outputs_hd_p3/leave_one_block_out.json", "outputs_hd_p3/p3_metrics.json",
        "outputs_hd_p3/step_variant_metrics.json")]
    source_paths += list((REPO/"outputs_mgn/comsol/eta_sym_psi_mlp").glob("seed_*/predictions.npz"))
    source_paths += list((REPO/"outputs_mgn/comsol/eta_sym_psi_mlp").glob("seed_*/sensitivity_fields.npz"))
    source_paths += list((REPO/"outputs_mgn/fd_check/case_cache").glob("*.npz"))
    fingerprints={str(p.relative_to(REPO)):file_hash(p) for p in source_paths}
    results={"theory":theoretical_checks(out)}
    print("Theory: exact symbolic chain rule and finite-difference convergence checked",flush=True)
    results["P0_models"]=p0_and_model_checks(out)
    results["FD_cache_reconstruction"]=fd_cache_reconstruction(out)
    print("P0 and archived symbolic/MLP/phi fields recomputed",flush=True)
    p2_table=p2_frozen_candidate_check(out)
    p2_table.to_csv(out / "p2_factorial_summary.csv",index=False)
    print("P2: 20 validation selections and one fresh 50,000-point test evaluated",flush=True)
    a0_stress_checks(out)
    print("A0: blocked folds, buffers and label-interpolation controls evaluated",flush=True)
    p3_reconstruction(out)
    print("P3: reconstructed independently from saved nodal time means",flush=True)
    assert fingerprints=={str(p.relative_to(REPO)):file_hash(p) for p in source_paths}
    atomic_json(out / "evidence.json",results)
    import scipy
    helpers=[Path(a0.__file__),Path(p1.__file__),Path(p2.__file__),REPO/'step5_fd_sensitivity_check.py',
             REPO/'step6_phi_R_analysis.py',REPO/'pipeline_contracts.py']
    atomic_json(out / "validation_manifest.json",{"source_files_sha256":fingerprints,"script_sha256":file_hash(Path(__file__)),
                "helper_files_sha256":{str(p.relative_to(REPO)):file_hash(p) for p in helpers},
                "runtime":{"python":sys.version,"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"sympy":sp.__version__},
                "source_files_unchanged":True,"historical_outputs_overwritten":False,
                "scope":"Theory, saved CFD fields/candidates and new spatial validation; no raw CFD solve or new PySR search"})


if __name__=="__main__":
    main()
