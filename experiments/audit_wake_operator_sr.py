"""Freeze compact validation choices, then audit completed operator-aware runs."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline_contracts import array_hash, atomic_json, content_hash, file_hash
from experiments.wake_operator_aware_sr import X, Y, field_metrics, sparse_predict, symbolic_basis
from experiments.wake_derivative_aware_sr import predict


def freeze_compact(out):
    target = out / "compact_validation_choices.json"
    if target.exists():
        return json.loads(target.read_text(encoding="utf-8"))
    if (out / "complete.json").exists():
        raise RuntimeError("Compact choices must be fixed before the first test evaluation")
    choices = []
    for c in ("u", "v"):
        records = json.loads((out / f"{c}_sparse_value/validation_candidates.json").read_text(encoding="utf-8"))
        for cap in (8, 12, 16, 20, 24):
            chosen = dict(min((r for r in records if r["n_terms"] <= cap), key=lambda r: (r["operator"], r["n_terms"])))
            indices = [(i,j) for i in range(chosen["degree_x"]+1) for j in range(chosen["degree_eta"]+1)]
            chosen["expression"] = symbolic_basis(indices,chosen["coefficients"])
            chosen["expression_sha256"] = content_hash(chosen["expression"])
            choices.append({"component": c, "term_cap": cap, **chosen})
    bundle = {"frozen_at":time.strftime("%Y-%m-%dT%H:%M:%S%z"), "test_not_yet_evaluated":True,
              "rule":"minimum validation operator score subject to term cap, sparse value-loss candidate pool only", "choices":choices}
    atomic_json(target,bundle)
    print("Frozen compact validation choices; no test metrics computed",flush=True)
    return bundle


def audit(out):
    protocol=json.loads((out/"protocol.json").read_text(encoding="utf-8"))
    completed=json.loads((out/"complete.json").read_text(encoding="utf-8"))
    assert completed["protocol_id"]==protocol["protocol_id"]
    for path,digest in protocol["sources"].items():
        assert file_hash(ROOT/path)==digest,path
    for name,digest in protocol["dataset"]["files"].items():
        assert file_hash(ROOT/"outputs_mgn/comsol/eta_sym"/name)==digest,name
    archive=np.load(out/"data_splits.npz")
    for c in ("u","v"):
        for s in ("train","validation","test"):
            assert array_hash(*[archive[f"{c}_{s}_{k}"] for k in ("points","q","dx","dy")])==protocol["split_hashes"][c][s]
    data={c:{k:archive[f"{c}_test_{k}"] for k in ("points","features","q","dx","dy","smooth_dx","smooth_dy")} for c in ("u","v")}
    frozen=json.loads((out/"compact_validation_choices.json").read_text(encoding="utf-8"))
    assert frozen["frozen_at"]<completed["completed_at"]
    rows=[]
    for chosen in frozen["choices"]:
        c=chosen["component"]
        assert content_hash(chosen["expression"])==chosen["expression_sha256"]
        pred=sparse_predict(chosen,data[c])
        symbolic=predict(chosen["expression"],data[c]["features"])
        for k in pred:
            np.testing.assert_allclose(pred[k],symbolic[k],atol=1e-8,rtol=1e-8)
        rows.append({"component":c,"term_cap":chosen["term_cap"],"n_terms":chosen["n_terms"],
                     "degree_x":chosen["degree_x"],"degree_eta":chosen["degree_eta"],
                     "validation_operator":chosen["operator"],
                     **field_metrics(pred,data[c],protocol["scales"][c],protocol["radius"])})
    pd.DataFrame(rows).to_csv(out/"compact_test_metrics.csv",index=False)
    selected=json.loads((out/"selected_velocity_models.json").read_text(encoding="utf-8"))
    assert selected["protocol_id"]==protocol["protocol_id"]
    predictions={}; selected_rows=[]
    for c,model in selected["models"].items():
        assert content_hash(model["expression"])==model["expression_sha256"]
        predictions[c]=predict(model["expression"],data[c]["features"])
        for ref,smooth in (("central_difference",False),("savgol5_stability",True)):
            selected_rows.append({"component":c,"method":model["method"],"seed":model["seed"],"reference":ref,
                                  **field_metrics(predictions[c],data[c],protocol["scales"][c],protocol["radius"],smooth)})
    pd.DataFrame(selected_rows).to_csv(out/"recommended_test_metrics.csv",index=False)
    divergence=predictions["u"]["dx"]+predictions["v"]["dy"]
    reference=data["u"]["dx"]+data["v"]["dy"]
    diagnostic={"model_divergence_rms":float(np.sqrt(np.mean(divergence**2))),
                "numerical_reference_divergence_rms":float(np.sqrt(np.mean(reference**2))),
                "model_minus_numerical_reference_rms":float(np.sqrt(np.mean((divergence-reference)**2))),
                "hard_incompressibility":False}
    # Algebraic compression changes neither model selection nor coefficients.
    gx,gy=np.meshgrid(np.linspace(-1,1,97),np.linspace(-1.9,1.9,193),indexing="ij")
    dense=np.column_stack([gx.ravel(),gy.ravel()])
    compact={}; compression=[]
    for c,model in selected["models"].items():
        expression=sp.sympify(model["expression"])
        simplified=sp.horner(expression,X,Y) if expression.is_polynomial(X,Y) else expression
        before=predict(str(expression),dense); after=predict(str(simplified),dense)
        delta=max(float(np.max(np.abs(before[k]-after[k]))) for k in before)
        assert delta<1e-8
        compact[c]={**model,"original_expression_sha256":model["expression_sha256"],
                    "expression":str(simplified),"expression_sha256":content_hash(str(simplified)),
                    "sympy_ops":int(sp.count_ops(simplified))}
        compression.append({"component":c,"original_operations":int(sp.count_ops(expression)),
                            "horner_operations":int(sp.count_ops(simplified)),"dense_value_and_gradient_max_abs_difference":delta})
    atomic_json(out/"compact_velocity_models.json",{**selected,"evaluation_form":"multivariate Horner; same fitted coefficients", "models":compact})
    atomic_json(out/"additional_verification.json",{"input_source_split_hashes_match":True,
                "compact_choices_frozen_before_test":True,"compact_matrix_and_symbolic_predictions_match":True,
                "selected_expression_hashes_match":True,"divergence_diagnostic":diagnostic,
                "algebraic_compression_checks":compression})
    profile_plot(out,protocol,data,predictions)
    print(pd.DataFrame(selected_rows).to_string(index=False),flush=True)
    print(json.dumps(diagnostic),flush=True)


def profile_plot(out,protocol,data,predictions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    baseline={}
    for c in ("u","v"):
        model=json.loads((out/f"{c}_pysr_h1_seed0/selection.json").read_text(encoding="utf-8"))["choices"]["operator"]
        baseline[c]=predict(model["expression"],data[c]["features"])
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for i,c in enumerate(("u","v")):
        points=data[c]["points"]; positions=np.unique(points[:,0]); xi=positions[len(positions)//2]
        mask=points[:,0]==xi; eta=points[mask,1]
        for j,(key,title) in enumerate((("q","value"),("dx","xi derivative"),("dy","eta derivative"))):
            ax=axes[i,j]
            ax.plot(eta,data[c][key][mask],color="black",lw=2,label="CFD numerical reference")
            ax.plot(eta,baseline[c][key][mask],color="#4278a8",label="Tree H1, fixed seed 0")
            ax.plot(eta,predictions[c][key][mask],color="#c76336",label="Validation-selected model")
            ax.set_title(f"{c}*: {title}, xi={xi:.4g}"); ax.set_xlabel("eta"); ax.grid(alpha=.2)
            if i==0 and j==0: ax.legend(fontsize=8)
    fig.suptitle("New test section; model choice fixed on validation data")
    fig.savefig(out/"selected_test_profiles.png",dpi=180); plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir",type=Path,default=ROOT/"outputs/wake_operator_aware_sr_v2")
    parser.add_argument("--freeze-compact-only",action="store_true")
    args=parser.parse_args()
    if args.freeze_compact_only: freeze_compact(args.out_dir)
    else: audit(args.out_dir)
