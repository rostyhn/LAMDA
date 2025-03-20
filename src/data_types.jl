# contents of colorbuffer returned by show(), used for rendering scenes to images 
const ColorMatrix = Matrix{ColorTypes.RGB{FixedPointNumbers.N0f8}} # stored as mat of ints from 0 to 255
const Maybe{T} = Union{Nothing,T}
const MaybeObservable{T} = Observable{Maybe{T}}

@kwdef mutable struct ClusterInfo
    groups
    assignments::Vector{Int}
    representatives
    # dendrogram info
    lines
    clusters
    cutoff
end

@kwdef mutable struct ClusterData
    clustering
    matrix
    idx_to_mtx::Vector{Int}
    m_extrema
    t_to_mtx::Dict{Tuple{Int,Int},Int}
    mtx_to_t::Dict{Int,Tuple{Int,Int}}
end

@kwdef mutable struct SingleClusterData
    ts::Vector{Tuple{Int,Int}}
    ref_t::Tuple{Int,Int}
    mat::Matrix{Float32}
    idx_to_mtx_idx::Vector{Int}# transition index to matrix index
    t_to_mtx::Dict{Tuple{Int,Int},Int}
end

function buildSingleClusterData(; ts, ref_t, mat, idx_to_mtx_idx)
    sortperm!(idx_to_mtx_idx, ts)

    # transition to matrix index dict
    t_to_mtx = Dict{Tuple{Int,Int},Int}()
    for (t, i) in zip(ts, idx_to_mtx_idx)
        t_to_mtx[t] = i
    end

    return SingleClusterData(ts=ts,
        ref_t=ref_t,
        mat=mat,
        idx_to_mtx_idx=idx_to_mtx_idx,
        t_to_mtx=t_to_mtx)
end
