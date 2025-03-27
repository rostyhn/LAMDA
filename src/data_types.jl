# contents of colorbuffer returned by show(), used for rendering scenes to images 
const ColorMatrix = Matrix{ColorTypes.RGB{FixedPointNumbers.N0f8}} # stored as mat of ints from 0 to 255
const Maybe{T} = Union{Nothing,T}
const MaybeObservable{T} = Observable{Maybe{T}}
const Transition = Tuple{Int16,Int16}

@kwdef mutable struct ClusterInfo
    groups::Dict{Int,Vector{Transition}} # dict of cluster idx to transition idx
    assignments::Vector{Int}
    representatives
    # dendrogram info
    lines
    clusters
    c2lx
    a2c
    cutoff
    cc2cidx
    h_range
    rel_t_to_idx
end

function get_cluster_of_transition(ci::ClusterInfo, t::Transition)
    t_idx = ci.rel_t_to_idx[t]
    return Set(t_idx)
end

function get_parents_of_transition(ci::ClusterInfo, t::Transition)
    c = get_cluster_of_transition(ci, t)
    # TODO: get all clusters transition belongs to
    return c
end

@kwdef mutable struct ClusterData
    clustering
    matrix
    m_extrema
    c2idx
    lines
    clusters
    c2lx
    c_to_parent::Dict{Set{Int},Set{Int}}
    parent_to_c::Dict{Set{Int},Tuple{Set{Int},Set{Int}}}
    t_to_mtx::Dict{Transition,Int}
    mtx_to_t::Dict{Int,Transition}
end

function get_local_matrix(cd::ClusterData, ts::Vector{Transition})
    mtx_idx = map(x -> cd.t_to_mtx[x], ts)
    # sortperm! doesn't mutate the arguments, spent a long time to figure this out
    s = sortperm(mtx_idx)
    t_to_mtx = Dict(reverse.(enumerate(ts[s])))
    return cd.matrix[mtx_idx[s], mtx_idx[s]], t_to_mtx
end

# we want to color transitions by their currently assigned cluster determined by the cutoff
function cluster_color(ci::ClusterInfo, t::Transition)
    return cycle_colormap(ci.cc2cidx[ci.assignments[ci.rel_t_to_idx[t]]], CLUSTER_COLORMAP)
end

function cluster_color(cd::ClusterData, c::Set{Int})
    return cycle_colormap(cd.c2idx[c], CLUSTER_COLORMAP)
end

function get_transitions(t_list, cluster::Set{Int})::Vector{Transition}
    return t_list[collect(cluster)]
end

function get_parent(cd::ClusterData, cluster::Set{Int})::Set{Int}
    return get(cd.c_to_parent, cluster, cluster)
end

function get_children(cd::ClusterData, cluster::Set{Int})::Union{Nothing,Tuple{Set{Int},Set{Int}}}
    return get(cd.parent_to_c, cluster, nothing)
end

function get_root(cd::ClusterData)
    c = first(keys(cd.c2idx))
    p = cd.c_to_parent[c]
    while length(intersect(p, c)) != length(p)
        c = p
        p = get_parent(cd, c)
    end
    return p
end

function dfs(cd::ClusterData, cluster::Set{Int}, acc=Ref([]))
    push!(acc[], cluster)
    children = get_children(cd, cluster)
    if isnothing(children)
        return
    end
    lc, rc = children
    dfs(cd, lc, acc)
    dfs(cd, rc, acc)
end

function get_neighbor(cd::ClusterData, cluster::Set{Int}, idx)
    parent = cd.c_to_parent[cluster]
    children = cd.parent_to_c[parent]
    return children[idx]
end

@kwdef struct SingleClusterData
    cluster
    ts::Vector{Transition}
    ref_t::Transition
    mat::Matrix{Float32}
    colors
    t_to_mtx::Dict{Transition,Int}
    heights::Dict{Set{Int},Float64}
    h_range
    lines
    assignments
    alignment
    clusters
end

function ClusterAnnotations()
    d = Dict()
    d["titles"] = Dict{Set{Int},String}()
    d["notes"] = Dict{Set{Int},String}()
    return d
end

function get_val(ca, property::String, s::Set{Int})
    dv = (property == "titles") ? string(s) : "..."
    return get(ca[property], s, dv)
end

function set_val(ca, s::Set{Int}, property::String, val::String)
    ca[property][s] = val
end

function buildSingleClusterData(; cluster, ts, ref_t, mat, t_to_mtx, cluster_data, cluster_info, rel_t_to_idx, alignment)
    rel_ts = map(x -> rel_t_to_idx[x], ts)
    colors = map(x -> cluster_color(cluster_data, Set(x)), rel_ts)
    clusters, lines, heights = branch(cluster_data, cluster)

    return SingleClusterData(cluster=cluster,
        ts=ts,
        ref_t=ref_t,
        mat=mat,
        colors=colors,
        t_to_mtx=t_to_mtx,
        lines=lines,
        heights=heights,
        clusters=clusters,
        alignment=alignment,
        assignments=assignments,
        h_range=cluster_info.h_range)
end

function clusters_above_cutoff(cc::Set{Int}, cd::ClusterData, scd::SingleClusterData, cutoff::Float64, acc=Ref([]))
    children = get_children(cd, cc)
    if isnothing(children)
        if scd.heights[cc] > cutoff
            push!(acc[], cc)
        end
        return
    end

    lc, rc = children
    if scd.heights[lc] > cutoff && scd.heights[rc] > cutoff
        clusters_above_cutoff(lc, cd, scd, cutoff, acc)
        clusters_above_cutoff(rc, cd, scd, cutoff, acc)
    else
        push!(acc[], cc)
    end
end
