using Pkg

Pkg.activate(".")
Pkg.instantiate()
Pkg.precompile()

using Revise, TransVis
