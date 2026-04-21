---
name: pysr-local-deeplearning
description: Run or troubleshoot PySR, SymbolicRegression, juliacall, or Julia-backed symbolic regression on this Windows machine using the local conda environment `deeplearning` instead of the sandbox. Use when a task needs to import `pysr`, fit a `PySRRegressor`, initialize Julia/juliacall, or execute local launchers that wrap PySR jobs.
---

# PySR Local Deeplearning

Use the local `deeplearning` interpreter for any task that imports or executes `pysr`. Do not try to run PySR inside the sandbox.

## Workflow

1. Prefer repo-provided PowerShell launchers when they exist. They already point at the correct interpreter and Julia project.
2. If no launcher exists, run `D:\keyan\Anaconda\envs\deeplearning\python.exe` directly and request non-sandbox execution before import or fit.
3. Let the first Julia-backed run finish. `juliapkg` and Julia precompilation can take minutes and may look idle while still working.
4. After the run, inspect stdout/stderr and the repo artifacts that the invoked script produced.

## Local Runtime

- Python: `D:\keyan\Anaconda\envs\deeplearning\python.exe`
- Julia project: `D:\keyan\Anaconda\julia_env`
- Julia depot/cache: `C:\Users\yizhe\.julia`
- Set:
  - `PYTHON_JULIACALL_HANDLE_SIGNALS=yes`
  - `PYTHON_JULIACALL_THREADS=auto`
  - `PYTHON_JULIACALL_OPTLEVEL=3`
  - `JULIA_PROJECT=D:\keyan\Anaconda\julia_env`

## Guardrails

- Request non-sandbox execution before running any command that imports or executes `pysr`.
- Do not rely on `conda activate`; call the environment's `python.exe` by absolute path.
- Treat first-run Julia setup and package precompilation as expected behavior, not an immediate failure.
- Keep repo-specific output locations out of the skill. Read the project scripts or the user request for where results should go.

## References

- For exact command patterns and local runtime details, read [references/runtime.md](references/runtime.md).
