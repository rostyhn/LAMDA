using Pkg
using PackageCompiler

ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python3")
ENV["PYTHON"] = Sys.which("python3")
ENV["JULIA_DEBUG"] = ""

Pkg.activate(".")
Pkg.instantiate()

println("Updating Python environment...")
Pkg.build("PyCall")
Pkg.precompile()

using TransVis
create_app(".", "build")
