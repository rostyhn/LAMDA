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

using Clustering: Clustering, hclust, cutree, kmedoids, clustering_quality
using NearestNeighbors
using MultivariateStats
using GilbertCurves
using Random
using UUIDs

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

export go, compare_matrices

function julia_main()::Cint
    go(ARGS[1])
    return 0
end

function compare_matrices(trajectory_name::String; id::String=random_string(), cachePath::String=default_cache())
    # script that compares clusters
    dataPath = abspath(trajectory_name)

    active_trajectory = get_data_alt(dataPath, cachePath)
    set_theme!(UI_THEME)

    dms::Dict{String,Matrix{Float32}} = active_trajectory.dms

    # https://strehl.com/diss/node80.html
    clusterings = Dict()
    for (x, xm) in dms
        clustering = hclust(to_ranked_mat(xm); linkage=:ward, branchorder=:barjoseph)
        clusterings[x] = clustering
    end
    n = length(active_trajectory.transitions)
    m = zeros(Float32, n, n)
    for (x, c) in clusterings
        for k in collect(2:50)
            m += labels_to_similarity_mat(cutree(c, k=k))
        end
    end
    m = m ./ (length(keys(clusterings)) * length(collect(2:50)))

    dms["consensus"] = 1 .- m
    clusterings["consensus"] = hclust(1 .- m; linkage=:ward, branchorder=:barjoseph)

    fms = Dict()
    for (x, c) in clusterings
        for (y, cc) in clusterings
            if x != y && !((x, y) in keys(fms)) && !((y, x) in keys(fms))
                fm_vals = []
                for k in collect(2:50)
                    fm = fowlkes_mallows_index(cutree(c, k=k), cutree(cc, k=k))
                    push!(fm_vals, fm)
                end
                fms[(x, y)] = fm_vals
            end
        end
    end

    f = Figure(size=(1920, 1080))
    order = sort(collect(keys(clusterings)))
    for (j, name) in enumerate(order)
        Label(f[1, j], text=name, tellwidth=false, tellheight=false)
        for (i, k) in enumerate(order)
            ax = Axis(f[i+1, j], title=k)
            if k != name
                v = get(fms, (name, k), nothing)
                if isnothing(v)
                    v = fms[(k, name)]
                end
                lines!(ax, collect(2:length(v)+1), v, color=(k == "consensus") ? :red : :blue)
                ylims!(ax, 0.0, 1.0)
            end
        end
    end
    GLMakie.save("FM_$(id).png", f)
    @info "Saved FM charts as FM_$(id).png"

    for (x, xm) in dms
        xmr = to_ranked(xm)
        for (y, ym) in dms
            if x != y
                ymr = to_ranked(ym)
                @show x, y, corspearman(xmr, ymr)
            end
        end
    end
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
