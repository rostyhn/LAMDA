using Pkg

ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python3")

Pkg.activate(".")
Pkg.instantiate()
println("Updating Python environment...")
Pkg.build("PyCall")
Pkg.precompile()

using Revise, TransVis
