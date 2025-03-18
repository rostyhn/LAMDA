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

