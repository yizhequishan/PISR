# Preliminary OASIS-SR experiments

- `p0_derivative_closure.py`: separates fixed-physical total derivatives, fixed-reference material responses, and coordinate transport terms on the existing CFD archive.
- `p1_observability_nullspace.py`: proves the differential-observation nullspace with an explicit symbolic dictionary and compares sparse complementary observations.
- `p2_analytic_derivative_loss.py`: compares value-only and analytic expression-tree derivative losses in PySR (added and run separately because it requires the local Julia runtime).

Deterministic experiments can be run with the default Python environment:

```powershell
python .\experiments\p0_derivative_closure.py
python .\experiments\p1_observability_nullspace.py
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
