module TransVis

using PyCall
using Base.Threads
using ImageIO
using Pickle
using JLD2
using CodecZlib
using FileIO
using ColorTypes

using ProgressMeter
using LinearAlgebra
using Statistics
using StatsBase
using SparseArrays
using Distances
using GeometryBasics
using FixedPointNumbers

using MathTeXEngine
using NetworkLayout
using Makie: ray_at_cursor, position_on_plot, mouse_in_scene, shift_project, update_tooltip_alignment!, parent_scene, show_data, clear_temporary_plots!, Orthographic, apply_transform_and_model, Makie
using CairoMakie # for saving plots w/ SVG
using GLMakie: Screen, ScreenConfig
using GLMakie
using Observables

using Clustering: Clustering, hclust, cutree, kmedoids
using NearestNeighbors
using MultivariateStats
using GilbertCurves

include("constants.jl")
include("data_types.jl")
include("io.jl")
include("multiprocess.jl")
include("processing.jl")
include("utils.jl")
include("math.jl")
include("ui.jl")

include("vis/Dendrogram.jl")
include("vis/EmbeddingView.jl")
include("vis/Scratchpad.jl")

include("windows/ReductionWindow.jl")
include("windows/SelectionWindow.jl")
include("windows/ClusterWindow.jl")
include("windows/NoteWindow.jl")
include("windows/SettingsWindow.jl")

function __init__()
    py"""
    import pickle

    def load_pickle(fpath):
        with open(fpath, "rb") as f:
            data = pickle.load(f)
        return data
    """o

    @pyinclude(joinpath(dirname(@__FILE__), "ase_processing.py"))
    GLMakie.activate!()
end

export go

function julia_main()::Cint
    go(ARGS[1])
    return 0
end

function go(trajectory_name::String; cachePath::String=default_cache(), kwargs...)::Int
    init_memory = Sys.free_memory() / 2^20
    dataPath = abspath(trajectory_name)

    active_trajectory = get_data_alt(dataPath, cachePath)
    set_theme!(UI_THEME)

    @time window, final_cleanup = build_reduction_window(active_trajectory, dataPath; kwargs...)
    screen = GLMakie.Screen(title="LAMDA - Reduction Window")
    display(screen, window)
    wait(screen)
    @debug "Final cleanup"
    final_cleanup()
    final_cleanup = nothing
    finalize(active_trajectory)
    empty!(window)
    GLMakie.closeall()
    GC.gc(true)
    final_memory = Sys.free_memory() / 2^20
    @debug init_memory, final_memory
    return 0
end

end # close module
