# Runtime

- Interpreter: `D:\keyan\Anaconda\envs\deeplearning\python.exe`
- Julia project: `D:\keyan\Anaconda\julia_env`
- Julia cache/depot: `C:\Users\yizhe\.julia`

# Command Pattern

1. Use the environment Python by absolute path.
2. Set:
   - `PYTHON_JULIACALL_HANDLE_SIGNALS=yes`
   - `PYTHON_JULIACALL_THREADS=auto`
   - `PYTHON_JULIACALL_OPTLEVEL=3`
   - `JULIA_PROJECT=D:\keyan\Anaconda\julia_env`
3. Run the project script or PowerShell launcher outside the sandbox.

# Notes

- `conda env list` is not required. The absolute interpreter path is the source of truth.
- The first successful PySR run may install or precompile Julia packages. Expect long startup time.
- If Julia setup fails, verify that the interpreter path and `D:\keyan\Anaconda\julia_env` still exist.
