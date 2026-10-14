"""Replot archived numerical evidence; no CFD, model training, or result mutation."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / '成果汇总附件_2026-09-20'
OUT.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.codex_tmp' / 'summary_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False, 'font.size': 11, 'axes.titlesize': 12,
    'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.facecolor': 'white', 'axes.labelcolor': '#222222',
})
BLUE, TEAL, RED, GREY = '#315574', '#408C89', '#B8634D', '#8B9299'
sources = set()
figure_data = {}

def rows(rel):
    sources.add(rel)
    with (ROOT / rel).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def save(fig, name, data):
    fig.savefig(OUT / name, dpi=210, bbox_inches='tight', pad_inches=.15)
    plt.close(fig)
    figure_data[name] = data

logic = 'outputs/logic_validation_2026-09-08/'
labels = 'outputs/a0_label_reliability_2026-09-09/'
grad = rows(logic + 'gradient_closure_recomputed.csv')
mlp = rows(logic + 'mlp_current_target.csv')
g = next(r for r in grad if r['component']=='u' and r['model']=='saved_psi')
d = next(r for r in grad if r['component']=='u' and r['model']=='data')
m = next(r for r in mlp if r['model']=='ensemble_mean')
kerr = np.array([float(g['epsilon_K']), float(m['epsilon_K']), float(d['epsilon_K'])])*100
cerr = np.array([float(g['closure']), float(m['total_closure']), float(d['closure'])])*100
fig, ax = plt.subplots(figsize=(8.5,3.5), layout='constrained')
x = np.arange(3)
b1 = ax.bar(x-.18,kerr,.34,label='K 相对误差',color=BLUE)
b2 = ax.bar(x+.18,cerr,.34,label='K 加已知 M 的闭合误差',color=TEAL)
ax.bar_label(b1,fmt='%.2f%%',padding=3,fontsize=10)
ax.bar_label(b2,fmt='%.2f%%',padding=3,fontsize=10)
ax.set_xticks(x,['历史符号 ψ','三种子 MLP 均值场','基准 CFD 数据梯度'])
ax.set_ylabel('相对误差 (%)'); ax.set_ylim(0,56)
ax.legend(frameon=False,loc='upper right');ax.yaxis.grid(alpha=.17);ax.set_axisbelow(True)
save(fig,'fig1_gradient_closure.png',{'K_percent':kerr.tolist(),'closure_percent':cerr.tolist(),'target':'R/u; 33x64; old 4–8 s mean; known FD M'})

p1=rows(logic+'p1_rerun/sparse_v_budget.csv')
fig,axes=plt.subplots(1,2,figsize=(8.6,3.4),layout='constrained',gridspec_kw={'width_ratios':[1.7,1]})
for meth,color,label in [('d_optimal',BLUE,'D-optimal'),('uniform',TEAL,'全域均匀'),('random',GREY,'随机')]:
    rr=[r for r in p1 if r['method']==meth and int(r['n_v'])>0]
    n=[int(r['n_v']) for r in rr]; e=[float(r['coefficient_error_mean']) for r in rr]
    axes[0].plot(n,e,'o-',label=label,color=color,ms=4)
axes[0].set_xscale('log');axes[0].set_yscale('log');axes[0].set_xlabel('补充 v 的点数');axes[0].set_ylabel('系数相对误差均值')
axes[0].grid(alpha=.16);axes[0].legend(frameon=False,fontsize=9)
for off,meth,color,label in [(-.17,'d_optimal',BLUE,'D-optimal'),(.17,'uniform',TEAL,'均匀')]:
    vals=[float(next(r for r in p1 if r['method']==meth and int(r['n_v'])==n)['rank_without_gauge']) for n in [3,8]]
    bb=axes[1].bar(np.arange(2)+off,vals,.32,color=color,label=label)
    axes[1].bar_label(bb,fmt='%.0f',padding=3)
axes[1].set_xticks([0,1],['3 点','8 点']);axes[1].set_ylabel('去除常数列后的秩');axes[1].set_ylim(0,7.3)
axes[1].axhline(6,color=GREY,ls=':',lw=1)
save(fig,'fig2_p1_budget.png',p1)

p2=rows(logic+'p2_factorial_summary.csv')
order=[('value_only','value_mse'),('value_only','h1'),('analytic_derivative','value_mse'),('analytic_derivative','h1')]
p2s=[next(r for r in p2 if (r['source'],r['criterion'])==key) for key in order]
fig,axes=plt.subplots(1,2,figsize=(8.6,3.4),layout='constrained')
names=['值搜索\n值选式','值搜索\nH1 选式','导数搜索\n值选式','导数搜索\nH1 选式']
for ax,key,scale,ylabel in [(axes[0],'test_value_mse_mean',1e5,r'函数值 MSE ($\times 10^{-5}$)'),(axes[1],'test_gradient_mse_mean',1e3,r'梯度 MSE ($\times 10^{-3}$)')]:
    yy=[float(r[key])*scale for r in p2s]
    bb=ax.bar(range(4),yy,color=[BLUE,BLUE,TEAL,TEAL],width=.65)
    ax.bar_label(bb,fmt='%.3f',padding=4,fontsize=9)
    ax.set_xticks(range(4),names,fontsize=9);ax.set_ylabel(ylabel);ax.set_ylim(0,max(yy)*1.24)
    ax.yaxis.grid(alpha=.16);ax.set_axisbelow(True)
save(fig,'fig3_p2_factorial.png',p2s)

a0=rows(logic+'a0_spatial_stress.csv')
target=[(p,c) for p in ['R','cx','cy'] for c in ['u','v']]
fig,ax=plt.subplots(figsize=(8.6,3.5),layout='constrained')
for off,scheme,meth,color,label in [(-.27,'LOXO','ridge_local_full',BLUE,'留单片岭回归'),(-.09,'LOXO','linear_M_interpolation',GREY,'留单片标签插值'),(.09,'blocked5','ridge_local_full',TEAL,'连续五块岭回归'),(.27,'blocked5_gap2','ridge_local_full',RED,'五块加缓冲岭回归')]:
    yy=[float(next(r for r in a0 if (r['parameter'],r['component'],r['scheme'],r['method'])==(p,c,scheme,meth))['pooled_rel_l2'])*100 for p,c in target]
    ax.bar(np.arange(6)+off,yy,.17,color=color,label=label)
ax.set_xticks(range(6),[p+'/'+c for p,c in target]);ax.set_ylabel('M 的 pooled 相对 L2 (%)');ax.set_ylim(0,111)
ax.axhline(20,color='#777777',ls=':',lw=1);ax.text(-.45,21,'20% 门槛',ha='left',fontsize=8,color='#555555')
ax.legend(frameon=False,ncol=2,fontsize=9,loc='upper right');ax.yaxis.grid(alpha=.16);ax.set_axisbelow(True)
save(fig,'fig4_a0_spatial.png',a0)

cyc=rows(labels+'cycle_mean_responses.csv')
fig,axes=plt.subplots(1,2,figsize=(8.6,3.5),layout='constrained')
names=[r['parameter']+'/'+r['component'] for r in cyc]
for ax,key,color,title in [(axes[0],'cycle_vs_raw_M_rel_l2',BLUE,'周期平均相对旧标签的变化'),(axes[1],'cycle_halves_difference_over_cycle_M_rms',TEAL,'前后周期组的标签差异')]:
    vv=np.array([float(r[key]) for r in cyc])*100
    bb=ax.barh(np.arange(6),vv,color=color,height=.62);ax.set_yticks(range(6),names);ax.invert_yaxis()
    ax.bar_label(bb,labels=[f'{v:.2f}%' for v in vv],padding=3,fontsize=9)
    ax.set_title(title);ax.set_xlim(0,max(vv)*1.3);ax.set_xlabel('相对变化 (%)');ax.xaxis.grid(alpha=.16);ax.set_axisbelow(True)
save(fig,'fig5_cycle_protocol.png',cyc)

p3=rows(logic+'p3_independent_reconstruction.csv')
p3=[r for r in p3 if r['mask']=='legacy_strict_80x80']
fig,ax=plt.subplots(figsize=(8.6,3.3),layout='constrained')
for scheme,color,label in [('inner_1pct',BLUE,'±1%R'),('outer_2pct',TEAL,'±2%R'),('five_point',RED,'五点差分')]:
    vals=[float(next(r for r in p3 if r['scheme']==scheme and int(r['H_over_D'])==h)['rho_M_over_K']) for h in [4,8]]
    ax.plot([4,8],vals,'o-',color=color,lw=2,label=label)
    for xx,vv in zip([4,8],vals):ax.annotate(f'{vv:.4f}',(xx,vv),xytext=(8,7),textcoords='offset points',fontsize=10,color=color)
ax.set_xlim(3.65,8.9);ax.set_ylim(0,1.58);ax.set_xticks([4,8]);ax.set_xlabel('H/D');ax.set_ylabel('ρ = ‖M‖ / ‖K‖');ax.legend(frameon=False,loc='upper left',ncol=3)
ax.yaxis.grid(alpha=.16)
save(fig,'fig6_p3_steps.png',p3)

# Check the rounded results used in the narrative directly against archived CSV.
phi=rows(logic+'phi_formula_metrics.csv')
checks={
    'psi_K_percent': [float(g['epsilon_K'])*100,16.87],
    'mlp_K_percent':[float(m['epsilon_K'])*100,5.58],
    'phi_T_u_percent':[float(phi[0]['total_symbolic_rel_l2'])*100,82.92],
    'phi_T_v_percent':[float(phi[1]['total_symbolic_rel_l2'])*100,155.71],
    'P2_gradient_improvement_percent':[(1-float(p2s[3]['test_gradient_mse_mean'])/float(p2s[0]['test_gradient_mse_mean']))*100,66.17],
    'cycle_R_v_change_percent':[float(cyc[1]['cycle_vs_raw_M_rel_l2'])*100,79.89],
}
for name,(actual,expected) in checks.items():
    assert abs(actual-expected)<.011, (name,actual,expected)
sources.update([
    'README.md','docs/pipeline_contracts.md','reports/logic_validation_2026-09-08.md',
    'reports/label_reliability_audit_2026-09-09.md','reports/research_assessment_2026-09-20.md',
    'reports/ai_paper_direction_2026-09-20.md','reports/a0_literature_action_plan_2026-09-09.md',
    'reports/sensitivity_ranking_feasibility_2026-09-20.md',
    'reports/sensitivity_ranking_mathematical_conditions_2026-09-20.md',
    'reports/project_audit_2026-09-08.md','archive/dvdy_substitution/README.md',
    'outputs_mgn/fd_check/psi_pareto_gradient_audit/REPORT.md',
    'outputs_mgn/fd_check/mlp_psi_gradient_audit/DENSE_XI_VS_LEGACY.md',
    'outputs_mgn/fd_check/decomposition_term_magnitudes.csv',
    'outputs_mgn/fd_check/phi_R_pysr/PHI_R_REPORT.md',
    'outputs_mgn/fd_check/phi_R_pysr/results_strategy_b.csv',
    'outputs_mgn/comsol/eta_sym_psi/results_strategy_b.csv',
    'outputs_mgn/comsol/eta_sym/meta.json','outputs_mgn/comsol/eta_sym_psi/meta.json',
    'outputs/p0_derivative_closure/metrics.json','outputs/p2_analytic_derivative_loss/metrics.json',
    'outputs/a0_m_reconstructability/A0_REPORT.md','outputs_hd_p3/P3_assessment.md',
    logic+'p1_rerun/metrics.json',logic+'p1_rerun/existing_pysr_degeneracy.csv',
    labels+'raw_reproduction.csv',labels+'mean_window_sensitivity.csv',
    labels+'cycle_mean_phase_robustness.csv',labels+'gradient_method_sensitivity.csv',
    labels+'gradient_resampling_sensitivity.csv',labels+'audit_manifest.json',
])
manifest={'date':'2026-09-20','operation':'Archive synthesis and plotting only; no CFD or model retraining',
          'sources':[{'path':p,'sha256':hashlib.sha256((ROOT/p).read_bytes()).hexdigest()} for p in sorted(sources)],
          'numeric_checks':checks,'figure_data':figure_data}
(OUT/'evidence_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'figures':list(figure_data),'sources':len(sources),'numeric_checks':'passed'},ensure_ascii=False))
