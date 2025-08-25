# contents of colorbuffer returned by show(), used for rendering scenes to images 
const ColorMatrix = Matrix{ColorTypes.RGB{FixedPointNumbers.N0f8}} # stored as mat of ints from 0 to 255
const Maybe{T} = Union{Nothing,T}
const MaybeObservable{T} = Observable{Maybe{T}}
const State = UInt16
const Transition = Tuple{UInt16,UInt16}
const ClusterSet = Set{UInt16}

const ClusterAnnotation = Dict{String,Dict{ClusterSet,String}}

function get_st_clusters(merge::Matrix{Int}, i::Int, clusterIdx::Vector{Int})::ClusterSet
    if i < 0
        return Set(clusterIdx[-i])
    end

    lt = merge[i, 1]
    rt = merge[i, 2]

    c_lt = get_st_clusters(merge, lt, clusterIdx)
    c_rt = get_st_clusters(merge, rt, clusterIdx)

    return union(c_lt, c_rt)
end

# assigns each cluster a unique id
function get_hierarchy(hc::Clustering.Hclust{Float32})::Tuple{
    Dict{ClusterSet,UInt16},
    Dict{ClusterSet,ClusterSet},
    Dict{ClusterSet,Tuple{ClusterSet,ClusterSet}},
    Dict{ClusterSet,Float64},
    ClusterSet
}

    c2idx = Dict{ClusterSet,UInt16}()
    clusterIdx = collect(eachindex(hc.order))
    c_to_parent = Dict{ClusterSet,ClusterSet}()
    parent_to_c = Dict{ClusterSet,Tuple{ClusterSet,ClusterSet}}()
    heights = Dict{ClusterSet,Float64}()

    root::ClusterSet = Set()
    for i in 1:size(hc.merges, 1)
        pg = get_st_clusters(hc.merges, i, clusterIdx)
        c2idx[pg] = i

        lt = hc.merges[i, 1]
        rt = hc.merges[i, 2]

        lg = get_st_clusters(hc.merges, lt, clusterIdx)
        rg = get_st_clusters(hc.merges, rt, clusterIdx)

        if lt < 0
            c2idx[lg] = i
            heights[lg] = 0.0
        end

        if rt < 0
            c2idx[rg] = i
            heights[rg] = 0.0
        end

        c_to_parent[lg] = pg
        c_to_parent[rg] = pg

        heights[pg] = hc.heights[i]

        parent_to_c[pg] = (lg, rg)
        root = pg
    end
    return c2idx, c_to_parent, parent_to_c, heights, root
end

# based on https://vis.cs.ucdavis.edu/vis2014papers/TVCG/papers/2072_20tvcg12-tennekes-2346277.pdf
function assign_colors_LCHab(parent_to_c::Dict{ClusterSet,Tuple{ClusterSet,ClusterSet}},
    root::ClusterSet;
    range::Tuple{Float64,Float64}=(0.0, 360.0),
    kwargs...)

    colors = Dict{ClusterSet,RGBAf}()
    _assign_colors_LCHab(parent_to_c, root, colors, range; kwargs...)

    return colors
end

# note that this is implemented for a binary tree only!
function _assign_colors_LCHab(parent_to_c::Dict{ClusterSet,Tuple{ClusterSet,ClusterSet}},
    c::ClusterSet,
    colors::Dict{ClusterSet,RGBAf},
    hues::Tuple{Float64,Float64},
    depth::Int=1;
    luminance::Int=70, # root luminance, defined as L1 in paper above
    beta_l::Int=-10, # luminance slope
    chroma::Int=60, # root chroma, defined as C1
    beta_c::Int=5, # chroma slope
    f::Float64=0.50, # hue fraction
)
    children = get(parent_to_c, c, nothing)
    c_hue = (hues[1] + hues[2]) / 2
    if depth > 1
        color = LCHab(
            (depth - 1) * beta_l + luminance,
            (depth - 1) * beta_c + chroma,
            c_hue
        )
        colors[c] = convert(RGBAf, color) # may be lossy
    else
        colors[c] = RGBAf(0.5, 0.5, 0.5, 1.0)
    end

    if isnothing(children)
        return
    end

    hf = ((1.0 - f) / 2) # inverse f to keep call semantics same as the paper

    r = abs(hues[2] - hues[1])
    lc, rc = children
    # split range proportionately 
    l_n = length(lc)
    r_n = length(rc)

    total = r_n + l_n
    lp = l_n / total
    rp = r_n / total

    lr = lp * r
    rr = rp * r

    l_start = hues[1]
    l_end = hues[1] + lr

    r_start = l_end
    r_end = r_start + rr

    _assign_colors_LCHab(parent_to_c, lc, colors, (l_start + lr * hf, l_end - lr * hf), depth + 1;
        luminance=luminance, beta_l=beta_l, f=f, chroma=chroma, beta_c=beta_c)

    _assign_colors_LCHab(parent_to_c, rc, colors, (r_start + rr * hf, r_end - rr * hf), depth + 1;
        luminance=luminance, beta_l=beta_l, f=f, chroma=chroma, beta_c=beta_c)
end

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
    # dendrogram info
    a2c::Dict{Int,ClusterSet}
    cutoff::Float64
    cc2cidx::Dict{UInt16,UInt16}
    h_range::Tuple{Float32,Float32}
    rel_t_to_idx::Dict{Transition,UInt16}
end

@kwdef struct ClusterData
    clustering::Clustering.Hclust{Float32}
    matrix::AbstractArray{Float32}
    m_extrema::Tuple{Float32,Float32}
    c2idx::Dict{ClusterSet,UInt16}
    c_to_parent::Dict{ClusterSet,ClusterSet}
    parent_to_c::Dict{ClusterSet,Tuple{ClusterSet,Set{UInt16}}}
    t_to_mtx::Dict{Transition,UInt16}
    mtx_to_t::Dict{UInt16,Transition}
    colors::Dict{ClusterSet,RGBAf}
    heights::Dict{ClusterSet,<:AbstractFloat}
end

function ClusterData(clustering::Clustering.Hclust{Float32},
    transitionSequence::AbstractArray{Transition},
    matrix::AbstractArray{Float32})

    c2idx, c_to_parent, parent_to_c, heights, root = get_hierarchy(clustering)
    fl = vec(matrix)

    colors = assign_colors_LCHab(parent_to_c, root;
        range=(-360.0, 360.0),
        beta_l=-5,
        f=0.75)

    t_to_mtx = Dict{Transition,UInt16}()
    mtx_to_t = Dict{UInt16,Transition}()
    for (i, r) in enumerate(clustering.order)
        t_to_mtx[transitionSequence[r]] = i
        mtx_to_t[i] = transitionSequence[r]
    end

    return ClusterData(clustering=clustering,
        matrix=matrix,
        m_extrema=extrema(fl),
        c2idx=c2idx,
        c_to_parent=c_to_parent,
        parent_to_c=parent_to_c,
        t_to_mtx=t_to_mtx,
        mtx_to_t=mtx_to_t,
        colors=colors,
        heights=heights)
end


@kwdef struct SingleClusterData
    cluster::ClusterSet
    ts::Vector{Transition}
    ref_t::Transition
    mat::AbstractArray{Float32}
    colors::Vector{RGBAf}
    t_to_mtx::Dict{Transition,UInt16}
    h_range::Tuple{Float32,Float32}
    assignments::Vector{UInt16}
    alignment::Dict{Transition,Tuple{Matrix{Float32},Bool}}
end

function buildSingleClusterData(; cluster::ClusterSet,
    ts::Vector{Transition},
    ref_t::Transition,
    mat::AbstractArray{Float32},
    t_to_mtx::Dict{Transition,UInt16},
    cluster_data::ClusterData,
    cluster_info::ClusterInfo,
    rel_t_to_idx::Dict{Transition,UInt16},
    alignment::Dict{Transition,Tuple{Matrix{Float32},Bool}})

    rel_ts = map(x -> rel_t_to_idx[x], ts)
    colors = map(x -> cluster_data.colors[Set(x)], rel_ts)

    return SingleClusterData(cluster=cluster,
        ts=ts,
        ref_t=ref_t,
        mat=mat,
        colors=colors,
        t_to_mtx=t_to_mtx,
        alignment=alignment,
        assignments=cluster_info.assignments,
        h_range=cluster_info.h_range)
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

# we want to color transitions by their currently assigned cluster determined by the cutoff
function cluster_color(ci::ClusterInfo, t::Transition)::RGBAf
    return cycle_colormap(ci.cc2cidx[ci.assignments[ci.rel_t_to_idx[t]]], CLUSTER_COLORMAP)
end

function get_local_matrix(cd::ClusterData, ts::Vector{Transition})
    mtx_idx = map(x -> cd.t_to_mtx[x], ts)
    # sortperm! doesn't mutate the arguments, spent a long time to figure this out
    s = sortperm(mtx_idx)
    t_to_mtx::Dict{Transition,UInt16} = Dict(reverse.(enumerate(ts[s])))
    return view(cd.matrix, view(mtx_idx, s), view(mtx_idx, s)), t_to_mtx
end

function cluster_color(cd::ClusterData, c::ClusterSet)::RGBAf
    return cycle_colormap(cd.c2idx[c], CLUSTER_COLORMAP)
end

function get_transitions(t_list::AbstractArray{Transition}, cluster::ClusterSet)::Vector{Transition}
    return view(t_list, collect(cluster))
end

function get_parent(cd::ClusterData, cluster::ClusterSet)::ClusterSet
    return get(cd.c_to_parent, cluster, cluster)
end

function get_children(cd::ClusterData, cluster::ClusterSet)::Maybe{Tuple{ClusterSet,ClusterSet}}
    return get(cd.parent_to_c, cluster, nothing)
end

function get_root(cd::ClusterData)::ClusterSet
    c = first(keys(cd.c2idx))
    p = cd.c_to_parent[c]
    while length(intersect(p, c)) != length(p)
        c = p
        p = get_parent(cd, c)
    end
    return p
end

function dfs(cd::ClusterData, cluster::ClusterSet, acc=Ref([]))
    push!(acc[], cluster)
    children = get_children(cd, cluster)
    if isnothing(children)
        return
    end
    lc, rc = children
    dfs(cd, lc, acc)
    dfs(cd, rc, acc)
end

function get_neighbor(cd::ClusterData, cluster::ClusterSet, idx)::Set{UInt16}
    parent = cd.c_to_parent[cluster]
    children = cd.parent_to_c[parent]
    return children[idx]
end

function ClusterAnnotations()
    d = Dict{String,Dict{ClusterSet,String}}()
    d["titles"] = Dict{ClusterSet,String}()
    d["notes"] = Dict{ClusterSet,String}()
    return d
end

function get_val(ca, property::String, s::ClusterSet)
    dv = (property == "titles") ? join(string.(s, base=10), ",") : "..."

    return get(ca[property], s, dv)
end

function set_val(ca, s::ClusterSet, property::String, val::String)
    ca[property][s] = val
end


function clusters_above_cutoff(cc::ClusterSet,
    cd::ClusterData,
    cutoff::AbstractFloat)

    cut_clusters = []
    _clusters_above_cutoff(cc, cd, cutoff, cut_clusters)

    return cut_clusters
end

# can be better - just collect all children and check in heights dict
function _clusters_above_cutoff(cc::ClusterSet,
    cd::ClusterData,
    cutoff::AbstractFloat,
    acc)

    children = get_children(cd, cc)
    if isnothing(children)
        if cd.heights[cc] > cutoff
            push!(acc, cc)
        end
        return
    end

    lc, rc = children
    if cd.heights[lc] > cutoff && cd.heights[rc] > cutoff
        _clusters_above_cutoff(lc, cd, cutoff, acc)
        _clusters_above_cutoff(rc, cd, cutoff, acc)
    else
        push!(acc, cc)
    end
end
