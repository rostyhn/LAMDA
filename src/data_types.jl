@kwdef mutable struct ClusterInfo
    groups
    assignments::Vector{Int}
    representatives
end

@kwdef mutable struct ClusterData
    clustering
    matrix
    idx_to_mtx::Vector{Int}
    m_extrema
    t_to_mtx::Dict{Tuple{Int,Int},Int}
end
