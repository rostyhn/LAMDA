# can ignore the patch for now, can't get it to work consistently
using Pkg

ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python3")

Pkg.activate(".")
Pkg.instantiate()

Pkg.build("PyCall")
Pkg.precompile()

using GLMakie

makiedir = dirname(pathof(GLMakie))
screen = joinpath(makiedir, "screen.jl")
println("patching $(screen)")
patchcmd = `patch $(screen) screen.patch`
println(patchcmd)
run(patchcmd)
exit()
