# Preliminary OASIS-SR experiments

- `p0_derivative_closure.py`: separates fixed-physical total derivatives, fixed-reference material responses, and coordinate transport terms on the existing CFD archive.
- `p1_observability_nullspace.py`: proves the differential-observation nullspace with an explicit symbolic dictionary and compares sparse complementary observations.
- `p2_analytic_derivative_loss.py`: compares value-only and analytic expression-tree derivative losses in PySR (added and run separately because it requires the local Julia runtime).
- `wake_derivative_aware_sr.py`: paired value-only versus analytic first-derivative loss on the real 33 x 64 wake, with three seeds, separate u/v fits, disjoint spatial blocks and derivative stencils, validation-frozen selection, and a smoothed-reference stability check. This is a direct velocity experiment, not a stream-function model or total parameter-response identification.
- `wake_operator_aware_sr.py`: follow-up with a new spatial split, paired H1 versus radius-transport PySR losses, and sparse tensor Chebyshev value/H1/operator ablations. `audit_wake_operator_sr.py` checks the frozen models and the accuracy/term-count tradeoff. Sparse term counts and PySR tree sizes are different complexity measures.
- `wake_pinn_informed_sr.py`: joint sparse velocity fits with PINN continuity, gradient-enhanced continuity, and a hard streamfunction, compared with a matched data/Sobolev control and the reproduced v2 baseline. `audit_wake_pinn_sr.py` independently checks symbolic derivatives, integral losses, selections, exports and test metrics. This is a NumPy/SciPy experiment; it does not import PySR or train a neural network.
- `a0_m_reconstructability.py`: NumPy-only Gate 1 probe for reconstructing the six intrinsic `M` response fields from local baseline-flow features, evaluated by leave-one-`xi`-slice-out prediction.

Deterministic experiments can be run with the default Python environment:

```powershell
python .\experiments\p0_derivative_closure.py
python .\experiments\p1_observability_nullspace.py
python .\experiments\a0_m_reconstructability.py
```

The PySR experiment must use the repository launcher:

```powershell
.\scripts\Invoke-DeeplearningPython.ps1 -ScriptPath .\scripts\setup_pysr_derivatives.py
.\scripts\Invoke-DeeplearningPython.ps1 -ScriptPath .\experiments\p2_analytic_derivative_loss.py
```

The setup command is needed once for PySR 1.5.9 because analytic expression-tree
derivatives activate the Julia `DynamicExpressionsZygoteExt` extension.

After P0-P2 have run, create the combined report with:

```powershell
python .\experiments\summarize_preliminary_experiments.py
```

Run the real-wake derivative-aware experiment independently (does not replace pipeline models):

```powershell
.\scripts\Invoke-DeeplearningPython.ps1 -ScriptPath .\experiments\wake_derivative_aware_sr.py
.\scripts\Invoke-DeeplearningPython.ps1 -ScriptPath .\experiments\audit_wake_derivative_sr.py
```

The default output is `outputs/wake_derivative_aware_sr_2026-09-28_typed/`. It records
the input and source hashes, all validation candidates, frozen equations, paired
test metrics and `REPORT.md`. `--prepare-only` checks stencil separation and a
manufactured polynomial without importing PySR. `--evaluate-only` re-evaluates
completed selections without importing PySR. Use a fresh `--out-dir` if changing
the data, code or search settings; identical completed fits can be resumed.
The audit rebuilds selected Julia training losses with SymPy, verifies file and
expression fingerprints, and reports both ordinary and matched-selection
comparisons. `wake_derivative_loss.jl` preserves the P2 arithmetic while using
native Julia target vectors in its inner loop.
Results and interpretation: [real-wake derivative-aware trial](../reports/wake_derivative_aware_sr_2026-09-28.md).

For the operator-aware follow-up, freeze all compact validation choices before
the first test evaluation:

```powershell
.\scripts\Invoke-DeeplearningPython.ps1 -ScriptPath .\experiments\wake_operator_aware_sr.py --sparse-only
.\scripts\Invoke-DeeplearningPython.ps1 -ScriptPath .\experiments\audit_wake_operator_sr.py --freeze-compact-only
.\scripts\Invoke-DeeplearningPython.ps1 -ScriptPath .\experiments\wake_operator_aware_sr.py
.\scripts\Invoke-DeeplearningPython.ps1 -ScriptPath .\experiments\audit_wake_operator_sr.py
```

The default directory is `outputs/wake_operator_aware_sr_v2/`. Its
`selected_velocity_models.json` contains complete formulas, coefficients and
normalization for the model chosen across all methods using validation only.
This remains a direct velocity surrogate on the observed wake window; it does
not replace the stream-function pipeline or identify the full parameter response.
The audit also exports algebraically equivalent Horner formulas in
`compact_velocity_models.json`. Results: [operator and sparse-structure follow-up](../reports/wake_operator_aware_sr_v2_2026-09-28.md).

For the PINN-inspired continuation, use the local `deeplearning` interpreter
(the base environment on this machine has incompatible NumPy/pandas binaries):

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& 'D:\keyan\Anaconda\envs\deeplearning\python.exe' experiments/wake_pinn_informed_sr.py --fit-only
& 'D:\keyan\Anaconda\envs\deeplearning\python.exe' experiments/audit_wake_pinn_sr.py --pre-test
& 'D:\keyan\Anaconda\envs\deeplearning\python.exe' experiments/wake_pinn_informed_sr.py --evaluate-only
& 'D:\keyan\Anaconda\envs\deeplearning\python.exe' experiments/audit_wake_pinn_sr.py
```

The default output is `outputs/wake_pinn_informed_sr_v3/`. Physics losses use
analytic derivatives at 384 label-free Gauss points across the wake rectangle;
the momentum equations are not imposed on the unclosed time-averaged field.
Every candidate has at most 48 free coefficients, with at most 24 per component
for direct velocity fits. Validation selects all loss weights, polynomial
degrees, term counts and the final method before test evaluation. The v2 split
is reused for a paired comparison on the same previously studied CFD case.
`selected_velocity_models.json` contains the chosen pair of Horner formulas,
coefficients and coordinate normalization; `finalist_formulas.json` also retains
the hard streamfunction candidate. The formulas use `x=xi-4, y=eta`.
The experiment improves/evaluates spatial derivatives and coordinate transport
`K_R`, not the full parameter response `T=K+M`.
Results: [PINN-inspired symbolic derivative study](../reports/wake_pinn_informed_sr_v3_2026-09-29.md).
