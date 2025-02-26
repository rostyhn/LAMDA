using Distributed
using Pkg

Pkg.activate(".")
Pkg.instantiate()
Pkg.precompile()

# uses HUGE amount of memory
@everywhere begin
    using Pkg
    Pkg.activate(@__DIR__)
    #Pkg.instantiate()
    #Pkg.precompile()
end

@everywhere using Revise, TransVis

# experiment - run without -p
# using Revise, TransVis
# addprocs(15)

# @everywhere using Revise
# almost works, but is considered to be part of TransVis instead of its own module
# @everywhere include("src/multiprocess.jl")
