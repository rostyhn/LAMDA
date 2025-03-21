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
    c_to_parent::Dict{Set{Int},Set{Int}}
    parent_to_c::Dict{Set{Int},Tuple{Set{Int},Set{Int}}}
end

function get_parent(ci::ClusterInfo, cluster::Set{Int})::Set{Int}
    return get(ci.c_to_parent, cluster, cluster)
end

function get_children(ci::ClusterInfo, cluster::Set{Int})::Union{Nothing,Tuple{Set{Int},Set{Int}}}
    return get(ci.parent_to_c, cluster, nothing)
end

function get_neighbor(ci::ClusterInfo, cluster::Set{Int}, idx)
    parent = ci.c_to_parent[cluster]
    children = ci.parent_to_c[parent]
    return children[idx]
end

@kwdef mutable struct ClusterData
    clustering
    matrix
    idx_to_mtx::Vector{Int}
    m_extrema
    t_to_mtx::Dict{Tuple{Int,Int},Int}
    mtx_to_t::Dict{Int,Tuple{Int,Int}}
end

@kwdef struct SingleClusterData
    ts::Vector{Tuple{Int,Int}}
    ref_t::Tuple{Int,Int}
    mat::Matrix{Float32}
    colors
    idx_to_mtx_idx::Vector{Int}# transition index to matrix index
    t_to_mtx::Dict{Tuple{Int,Int},Int}
end
function buildSingleClusterData(; ts, ref_t, mat, idx_to_mtx_idx, cluster_info, rel_t_to_idx)
    sortperm!(idx_to_mtx_idx, ts)
    cmap = to_colormap(CLUSTER_COLORS)

    rel_ts = map(x -> rel_t_to_idx[x], ts)
    colors = map(x -> cycle_colormap(cluster_info.assignments[x], cmap), rel_ts)

    # transition to matrix index dict
    t_to_mtx = Dict{Tuple{Int,Int},Int}()
    for (t, i) in zip(ts, idx_to_mtx_idx)
        t_to_mtx[t] = i
    end

    return SingleClusterData(ts=ts,
        ref_t=ref_t,
        mat=mat,
        colors=colors,
        idx_to_mtx_idx=idx_to_mtx_idx,
        t_to_mtx=t_to_mtx)
end
