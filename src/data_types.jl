# contents of colorbuffer returned by show(), used for rendering scenes to images 
const ColorMatrix = Matrix{ColorTypes.RGB{FixedPointNumbers.N0f8}} # stored as mat of ints from 0 to 255
const Maybe{T} = Union{Nothing,T}
const MaybeObservable{T} = Observable{Maybe{T}}
const State = UInt16
const Transition = Tuple{UInt16,UInt16}

@kwdef struct Trajectory
    name::String
    transitions::Vector{Transition}
    alignedPositionsMatrices::Dict{Transition,Tuple{Matrix{Float32},Matrix{Float32}}}
    kdTrees::Dict{Transition,Tuple{KDTree,KDTree}}
    t1::Dict{Transition,Vector{Float32}}
    t2::Dict{Transition,Vector{Float32}}
    t3::Dict{Transition,Vector{Float32}}
    stretchedPrincipalAxes::Dict{Transition,Vector{Vector{Vec3f}}}
    dms::Dict{String,Matrix{Float32}}
    scalars::Dict{String,Dict{Transition,Array{Float32}}}
    scalar_ranges::Dict{String,Tuple{Float32,Float32}}
    alignments::Dict{String,Dict{State,Matrix{Float32}}}
    t_to_idx::Dict{Transition,UInt16}
end



@kwdef struct ClusterInfo
    groups::Dict{UInt16,Vector{Transition}} # dict of cluster idx to transition idx
    assignments::Vector{UInt16}
    representatives::Dict{UInt16,Transition}
    # dendrogram info
    lines::Vector{Tuple{Point2f,Point2f}}
    clusters::Vector{Set{UInt16}}
    c2lx::Dict{Set{UInt16},Vector{UInt16}}
    a2c::Dict{Int,Set{UInt16}}
    cutoff::Float64
    cc2cidx::Dict{UInt16,UInt16}
    h_range::Tuple{Float32,Float32}
    rel_t_to_idx::Dict{Transition,UInt16}
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

@kwdef struct ClusterData
    clustering::Clustering.Hclust{Float32}
    matrix
    m_extrema::Tuple{Float32,Float32}
    c2idx::Dict{Set{UInt16},UInt16}
    lines::Vector{Tuple{Point2f,Point2f}}
    clusters::Vector{Set{UInt16}}
    c2lx::Dict{Set{UInt16},Vector{UInt16}}
    c_to_parent::Dict{Set{UInt16},Set{UInt16}}
    parent_to_c::Dict{Set{UInt16},Tuple{Set{UInt16},Set{UInt16}}}
    t_to_mtx::Dict{Transition,UInt16}
    mtx_to_t::Dict{UInt16,Transition}
end

function get_local_matrix(cd::ClusterData, ts::Vector{Transition})
    mtx_idx = map(x -> cd.t_to_mtx[x], ts)
    # sortperm! doesn't mutate the arguments, spent a long time to figure this out
    s = sortperm(mtx_idx)
    t_to_mtx::Dict{Transition,UInt16} = Dict(reverse.(enumerate(ts[s])))
    return view(cd.matrix, view(mtx_idx, s), view(mtx_idx, s)), t_to_mtx
end

# we want to color transitions by their currently assigned cluster determined by the cutoff
function cluster_color(ci::ClusterInfo, t::Transition)::RGBAf
    return cycle_colormap(ci.cc2cidx[ci.assignments[ci.rel_t_to_idx[t]]], CLUSTER_COLORMAP)
end

function cluster_color(cd::ClusterData, c::Set{UInt16})::RGBAf
    return cycle_colormap(cd.c2idx[c], CLUSTER_COLORMAP)
end

function get_transitions(t_list, cluster::Set{UInt16})::Vector{Transition}
    return view(t_list, collect(cluster))
end

function get_parent(cd::ClusterData, cluster::Set{UInt16})::Set{UInt16}
    return get(cd.c_to_parent, cluster, cluster)
end

function get_children(cd::ClusterData, cluster::Set{UInt16})::Union{Nothing,Tuple{Set{UInt16},Set{UInt16}}}
    return get(cd.parent_to_c, cluster, nothing)
end

function get_root(cd::ClusterData)::Set{UInt16}
    c = first(keys(cd.c2idx))
    p = cd.c_to_parent[c]
    while length(intersect(p, c)) != length(p)
        c = p
        p = get_parent(cd, c)
    end
    return p
end

function dfs(cd::ClusterData, cluster::Set{UInt16}, acc=Ref([]))
    push!(acc[], cluster)
    children = get_children(cd, cluster)
    if isnothing(children)
        return
    end
    lc, rc = children
    dfs(cd, lc, acc)
    dfs(cd, rc, acc)
end

function get_neighbor(cd::ClusterData, cluster::Set{UInt16}, idx)::Set{UInt16}
    parent = cd.c_to_parent[cluster]
    children = cd.parent_to_c[parent]
    return children[idx]
end

# can be more clever - no need to form a separate datastructure from cluster data
@kwdef struct SingleClusterData
    cluster::Set{UInt16}
    ts::Vector{Transition}
    ref_t::Transition
    mat::Matrix{Float32}
    colors::Vector{RGBAf}
    t_to_mtx::Dict{Transition,UInt16}
    heights::Dict{Set{UInt16},Float64}
    h_range::Tuple{Float32,Float32}
    lines::Vector{Tuple{Point2f,Point2f}}
    assignments::Vector{UInt16}
    alignment::Dict{Transition,Tuple{Matrix{Float32},Bool}}
    clusters::Vector{Set{UInt16}}
end

const ClusterAnnotation = Dict{String,Dict{Set{UInt16},String}}
function ClusterAnnotations()
    d = Dict{String,Dict{Set{UInt16},String}}()
    d["titles"] = Dict{Set{UInt16},String}()
    d["notes"] = Dict{Set{UInt16},String}()
    return d
end

function get_val(ca, property::String, s::Set{UInt16})
    dv = (property == "titles") ? string(s) : "..."
    return get(ca[property], s, dv)
end

function set_val(ca, s::Set{UInt16}, property::String, val::String)
    ca[property][s] = val
end

function buildSingleClusterData(; cluster::Set{UInt16},
    ts::Vector{Transition},
    ref_t::Transition,
    mat,
    t_to_mtx::Dict{Transition,UInt16},
    cluster_data::ClusterData,
    cluster_info::ClusterInfo,
    rel_t_to_idx::Dict{Transition,UInt16},
    alignment::Dict{Transition,Tuple{Matrix{Float32},Bool}})

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
        assignments=cluster_info.assignments,
        h_range=cluster_info.h_range)
end

function clusters_above_cutoff(cc::Set{UInt16}, cd::ClusterData, scd::SingleClusterData, cutoff::AbstractFloat, acc=Ref([]))
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
