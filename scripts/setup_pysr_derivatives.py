"""Install the Julia extension required by analytic tree derivatives in PySR."""

from pysr import jl


jl.seval('import Pkg; Pkg.add("Zygote"); Pkg.precompile()')
print("Julia Zygote dependency is installed and precompiled.")
