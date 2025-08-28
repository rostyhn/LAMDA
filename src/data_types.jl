# contents of colorbuffer returned by show(), used for rendering scenes to images 
const ColorMatrix = Matrix{ColorTypes.RGB{FixedPointNumbers.N0f8}} # stored as mat of ints from 0 to 255
const Maybe{T} = Union{Nothing,T}
const MaybeObservable{T} = Observable{Maybe{T}}
const State = UInt16
# TODO: change all indexes to Index to make it easier to identify and change type
const Index = Int
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
    Dict{ClusterSet,Int},
    Dict{ClusterSet,ClusterSet},
    Dict{ClusterSet,Tuple{ClusterSet,ClusterSet}},
    Dict{ClusterSet,Float64},
    ClusterSet
}
    c2idx = Dict{ClusterSet,Int}()
    # the indexes in the cluster are indexes into the cluster order array
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
# did not implement permutations because the hierarchy is a binary tree
# could probably tweak a bit and release as a package
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
    reverseHues::Bool=true
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

    # split range proportionately - here we diverge from the original implementation
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

    l_hues = (l_start + lr * hf, l_end - lr * hf)

    _assign_colors_LCHab(parent_to_c, lc, colors, l_hues, depth + 1;
        luminance=luminance, beta_l=beta_l, f=f, chroma=chroma, beta_c=beta_c)

    r_hues = (r_start + rr * hf, r_end - rr * hf)

    if reverseHues
        r_hues = (r_hues[2], r_hues[1])
    end

    _assign_colors_LCHab(parent_to_c, rc, colors, r_hues, depth + 1;
        luminance=luminance, beta_l=beta_l, f=f, chroma=chroma, beta_c=beta_c)
end

@kwdef struct Trajectory
    name::String
    transitions::Vector{Transition}
    alignedPositionsMatrices::Dict{Transition,Tuple{Matrix{Float32},Matrix{Float32}}}
    t1::Dict{Transition,Vector{Float32}}
    t2::Dict{Transition,Vector{Float32}}
    t3::Dict{Transition,Vector{Float32}}
    stretchedPrincipalAxes::Dict{Transition,Vector{Vector{Vec3f}}}
    dms::Dict{String,Matrix{Float32}}
    scalars::Dict{String,Dict{Transition,Array{Float32}}}
    scalar_ranges::Dict{String,Tuple{Float32,Float32}}
    alignments::Dict{String,Dict{State,Matrix{Float32}}}
    t_to_idx::Dict{Transition,Int}
end

@kwdef struct ClusterData
    clustering::Clustering.Hclust{Float32}
    matrix::AbstractArray{Float32}
    m_extrema::Tuple{Float32,Float32}
    c2idx::Dict{ClusterSet,Index}
    c_to_parent::Dict{ClusterSet,ClusterSet}
    parent_to_c::Dict{ClusterSet,Tuple{ClusterSet,ClusterSet}}
    t_to_mtx::Dict{Transition,Index}
    mtx_to_t::Dict{Index,Transition}
    colors::Dict{ClusterSet,RGBAf}
    heights::Dict{ClusterSet,<:AbstractFloat}
    ts::Base.RefValue{<:AbstractArray{Transition}} # reference to list of all transition before ordering
    root::ClusterSet
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

    t_to_mtx = Dict{Transition,Index}()
    mtx_to_t = Dict{Index,Transition}()
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
        heights=heights,
        ts=Ref(transitionSequence),
        root=root)
end

@kwdef struct ClusterInfo
    groups::Dict{Int,Vector{Transition}} # dict of cluster idx to transition idx
    assignments::Vector{Int}
    # dendrogram info
    a2c::Dict{Int,ClusterSet}
    cutoff::Float64
end

function ClusterInfo(cluster_data::ClusterData, transitionSequence::Vector{Transition}, cutoff::Float32)
    assignments::Vector{Index} = cutree(cluster_data.clustering, h=cutoff)

    groups = Dict{Index,Vector{Transition}}()
    igroups = Dict{Index,Vector{Index}}()

    a2c = Dict{Index,ClusterSet}() # assignment to cluster     

    for (i, c) in enumerate(assignments)
        g = get(groups, c, [])
        ig = get(igroups, c, [])
        push!(g, transitionSequence[i])
        push!(ig, i)

        groups[c] = g
        igroups[c] = ig
    end

    for (idx, ig) in igroups
        c = Set(ig)
        a2c[idx] = c
    end

    return ClusterInfo(groups=groups,
        assignments=assignments,
        a2c=a2c,
        cutoff=cutoff)

end

@kwdef struct SingleClusterData
    cluster::ClusterSet
    ts::AbstractArray{Transition}
    ref_t::Transition
    mat::AbstractArray{Float32}
    colors::Vector{RGBAf}
    t_to_mtx::Dict{Transition,Index}
    mtx_to_t::Dict{Index,Transition}
    rel_ts::Vector{Index} # absolute indices into matrix
    alignment::Dict{Transition,Tuple{Matrix{Float32},Bool}}
end

function buildSingleClusterData(; cluster::ClusterSet,
    ts::AbstractArray{Transition},
    ref_t::Transition,
    cluster_data::ClusterData,
    rel_t_to_idx::Dict{Transition,Index},
    alignment::Dict{Transition,Tuple{Matrix{Float32},Bool}})

    # guaranteed to be in ts order, so other fns can just index into it and get the transition's absolute index
    rel_ts = map(x -> rel_t_to_idx[x], ts)
    colors = map(x -> cluster_data.colors[Set(UInt16(x))], rel_ts)
    mat, t_to_mtx, mtx_to_t = get_local_matrix(cluster_data, ts)

    return SingleClusterData(cluster=cluster,
        ts=ts,
        ref_t=ref_t,
        mat=mat,
        colors=colors,
        t_to_mtx=t_to_mtx,
        mtx_to_t=mtx_to_t,
        alignment=alignment,
        rel_ts=rel_ts)
end

function get_cluster_of_transition(cd::SingleClusterData, i::Index)::Set{UInt16}
    if i > length(cd.rel_ts)
        @warn "tried to get non-existent index to get cluster for transition!"
        return nothing
    end

    t_idx = cd.rel_ts[i]
    return Set(UInt16(t_idx))
end

function get_cluster_of_transition(ci::ClusterInfo, rel_t_to_idx::Dict{Transition,Index}, t::Transition)
    a_idx = rel_t_to_idx[t]
    a = ci.assignments[a_idx]
    return ci.a2c[a]
end

function get_parents_of_transition(ci::ClusterInfo, t::Transition)
    c = get_cluster_of_transition(ci, t)
    # TODO: get all clusters transition belongs to
    return c
end

# we want to color transitions by their currently assigned cluster determined by the cutoff
function cluster_color(ci::ClusterInfo, cd::ClusterData, rel_t_to_idx::Dict{Transition,Index}, t::Transition)::RGBAf
    return cd.colors[get_cluster_of_transition(ci, rel_t_to_idx, t)]
end

function get_local_matrix(cd::ClusterData, ts::AbstractArray{Transition})::Tuple{AbstractArray{Float32},Dict{Transition,Index},Dict{Index,Transition}}
    mtx_idx = map(x -> cd.t_to_mtx[x], ts)
    # sortperm! doesn't mutate the arguments, spent a long time to figure this out
    s = sortperm(mtx_idx)
    t_to_mtx::Dict{Transition,Index} = Dict(reverse.(enumerate(ts[s])))
    mtx_to_t::Dict{Index,Transition} = Dict(enumerate(ts[s]))
    return view(cd.matrix, view(mtx_idx, s), view(mtx_idx, s)), t_to_mtx,
    mtx_to_t
end

function cluster_color(cd::ClusterData, c::ClusterSet)::RGBAf
    return cycle_colormap(cd.c2idx[c], CLUSTER_COLORMAP)
end

function get_transitions(cd::ClusterData, cluster::ClusterSet)::AbstractArray{Transition}
    return map(x -> cd.ts[][x], collect(cluster))
end

function get_parent(cd::ClusterData, cluster::ClusterSet)::ClusterSet
    return get(cd.c_to_parent, cluster, cluster)
end

function get_children(cd::ClusterData, cluster::ClusterSet)::Maybe{Tuple{ClusterSet,ClusterSet}}
    return get(cd.parent_to_c, cluster, nothing)
end

function descend_tree(cd::ClusterData, cluster::ClusterSet)
    nodes = ClusterSet[]
    _descend_tree(cd, cluster, nodes)
    return nodes
end

# dfs but include parents along the way
function _descend_tree(cd::ClusterData, cluster::ClusterSet, acc::Vector{ClusterSet})
    push!(acc, cluster)
    children = get_children(cd, cluster)
    if isnothing(children)
        return
    end
    lc, rc = children
    _descend_tree(cd, lc, acc)
    _descend_tree(cd, rc, acc)
end

function dfs_leaves(cd::ClusterData,
    cluster::ClusterSet)::Vector{ClusterSet}

    leaves = ClusterSet[]
    _dfs_leaves(cd, cluster, leaves)
    return leaves
end

function _dfs_leaves(cd::ClusterData,
    cluster::ClusterSet,
    leaves::Vector{ClusterSet})

    children = get_children(cd, cluster)
    if isnothing(children)
        push!(leaves, cluster)
        return
    end
    lc, rc = children
    _dfs_leaves(cd, lc, leaves)
    _dfs_leaves(cd, rc, leaves)
end

function get_neighbor(cd::ClusterData, cluster::ClusterSet, idx::Integer)::ClusterSet
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
    cutoff::AbstractFloat)::AbstractArray{ClusterSet}

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

    # gets clusters right above cutoff
    lc, rc = children
    if cd.heights[lc] > cutoff && cd.heights[rc] > cutoff
        _clusters_above_cutoff(lc, cd, cutoff, acc)
        _clusters_above_cutoff(rc, cd, cutoff, acc)
    else
        push!(acc, cc)
    end
end
