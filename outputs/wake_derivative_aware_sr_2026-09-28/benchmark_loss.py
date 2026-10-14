import json
import sys
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from experiments.p2_analytic_derivative_loss import DERIVATIVE_OBJECTIVE, inject_derivative_globals
from pysr import jl

out = root / 'outputs/wake_derivative_aware_sr_2026-09-28'
z = np.load(out / 'data_splits.npz')
p = json.loads((out / 'protocol.json').read_text(encoding='utf-8'))
s = p['scales']['u']
inject_derivative_globals(z['u_train_dx'], z['u_train_dy'], value_scale=s['value'], dx_scale=s['dx'], dy_scale=s['dy'], lambda_grad=1.)
jl.seval('using SymbolicRegression; global BENCH_X, BENCH_Y')
jl.BENCH_X = z['u_train_features'].T.copy()
jl.BENCH_Y = z['u_train_q'].copy()
jl.seval(DERIVATIVE_OBJECTIVE)
jl.seval((root / 'experiments/wake_derivative_loss.jl').read_text(encoding='utf-8'))
jl.seval('''
bench_options = SymbolicRegression.Options(binary_operators=[+, -, *, /], unary_operators=[sin, cos])
bench_dataset = SymbolicRegression.Dataset(convert(Matrix{Float64}, BENCH_X), convert(Vector{Float64}, BENCH_Y))
bench_trees = [SymbolicRegression.parse_expression(e; operators=bench_options.operators, variable_names=["x", "y"])
 for e in [:(sin(x)*cos(y)+0.17*x/(2+y*y)), :(cos(y)*x/3.2 + sin(y)), :(1.0*x*x+y*y), :(cos(cos(y)-0.63)*3.49-1.72)]]
function bench_objective(f, trees, data, options)
    for t in trees
        f(t, data, options)
    end
    return @elapsed for i in 1:30, t in trees
        f(t, data, options)
    end
end
''')
old = np.array(jl.seval('[oasis_derivative_objective(t, bench_dataset, bench_options) for t in bench_trees]'))
old_seconds = float(jl.seval('bench_objective(oasis_derivative_objective, bench_trees, bench_dataset, bench_options)'))
jl.seval('OASIS_DX_TARGET=convert(Vector{Float64}, OASIS_DX_TARGET); OASIS_DY_TARGET=convert(Vector{Float64}, OASIS_DY_TARGET)')
new = np.array(jl.seval('[wake_derivative_objective(t, bench_dataset, bench_options) for t in bench_trees]'))
new_seconds = float(jl.seval('bench_objective(wake_derivative_objective, bench_trees, bench_dataset, bench_options)'))
np.testing.assert_array_equal(old, new)
result={'reference_loss':old.tolist(),'typed_loss':new.tolist(),'bitwise_equal':True,
        'reference_seconds':old_seconds,'typed_seconds':new_seconds,'speedup':old_seconds/new_seconds,'calls':120}
(out/'loss_performance_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result),flush=True)
