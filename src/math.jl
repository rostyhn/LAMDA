function getIndexFromSQMesh(i::Integer, resolution::AbstractFloat)
    numberY = trunc(Int, pi / resolution) + 2
    numberX = trunc(Int, 2 * pi / resolution) + 1
    number = numberX * numberY
    return trunc(Int, i / number) + 1
end

function signPow(base, exponent)::AbstractFloat
    return sign(base) * abs(base)^exponent
end

function qz(phi::AbstractFloat, theta::AbstractFloat, alpha::AbstractFloat, beta::AbstractFloat)
    return Point3f(signPow(cos(theta), alpha) * signPow(sin(phi), beta),
        signPow(sin(theta), alpha) * signPow(sin(phi), beta),
        signPow(cos(phi), beta))
end

function qx(phi::AbstractFloat, theta::AbstractFloat, alpha::AbstractFloat, beta::AbstractFloat)
    return Point3f(signPow(cos(phi), beta),
        -signPow(sin(theta), alpha) * signPow(sin(phi), beta),
        signPow(cos(theta), alpha) * signPow(sin(phi), beta))

end

function sq_triangle_indices(x::Integer, y::Integer, nTheta::Integer)::Matrix{UInt16}
    p11 = x + nTheta * (y - 1)
    p21 = x < nTheta ? (x + 1) + nTheta * (y - 1) : 1 + nTheta * (y - 1)
    p31 = x + nTheta * (y)

    p12 = x < nTheta ? (x + 1) + nTheta * (y - 1) : 1 + nTheta * (y - 1)
    p22 = x < nTheta ? (x + 1) + nTheta * y : 1 + nTheta * y # index 
    p32 = x + nTheta * (y)

    return hcat([p11, p31, p21], [p32, p22, p12])
end

function sq_triangle_indices_t(x::Integer, y::Integer, nTheta::Integer)::Tuple{Tuple{UInt16,UInt16,UInt16},
    Tuple{UInt16,UInt16,UInt16}}
    p11 = x + nTheta * (y - 1)
    p21 = x < nTheta ? (x + 1) + nTheta * (y - 1) : 1 + nTheta * (y - 1)
    p31 = x + nTheta * (y)

    p12 = x < nTheta ? (x + 1) + nTheta * (y - 1) : 1 + nTheta * (y - 1)
    p22 = x < nTheta ? (x + 1) + nTheta * y : 1 + nTheta * y # index 
    p32 = x + nTheta * (y)

    return (p11, p31, p21), (p32, p22, p12)
end

function call_trifaces(x)
    return TriangleFace((x[1], x[2], x[3]))
end

function superquadric(scale::AbstractFloat,
    position::Point3f,
    principalStretches::Vector{Vec3f},
    sharpness::AbstractFloat,
    resolution::AbstractFloat=0.2)::Tuple{Vector{Point3f},Vector{TriangleFace{UInt16}}}

    stretchRatio1 = norm(view(principalStretches, 3))
    stretchRatio2 = norm(view(principalStretches, 2))
    stretchRatio3 = norm(view(principalStretches, 1))

    stretchDirection1 = principalStretches[3] / stretchRatio1
    stretchDirection2 = principalStretches[2] / stretchRatio2
    stretchDirection3 = principalStretches[1] / stretchRatio3

    cl = (stretchRatio1 - stretchRatio2) / (stretchRatio1 + stretchRatio2 + stretchRatio3)   #linear anisotopy
    cp = 2 * (stretchRatio2 - stretchRatio3) / (stretchRatio1 + stretchRatio2 + stretchRatio3) # planar anisotropy

    phiRange = [0:resolution:pi;]  #vertical: south -> north
    push!(phiRange, pi) #ass pi to close the hole at the end introduced by resolution
    thetaRange = [0:resolution:2*pi;] #horizontal: west -> east
    nPhi = length(phiRange)
    nTheta = length(thetaRange)

    points = Vector{Point3f}(undef, nPhi * nTheta)
    if cl >= cp
        alpha = signPow((1 - cp), sharpness)
        beta = signPow((1 - cl), sharpness)

        i = 1
        for phi in phiRange
            row = qx.(Ref(phi), thetaRange, Ref(alpha), Ref(beta))
            points[i:i+nTheta-1] = row
            i += nTheta
        end
    else
        alpha = (1 - cl)^sharpness
        beta = (1 - cp)^sharpness

        i = 1
        for phi in phiRange
            row = qz.(Ref(phi), thetaRange, Ref(alpha), Ref(beta))
            points[i:i+nTheta-1] = row
            i += nTheta
        end
    end

    scaleMatrix = diagm([stretchRatio1 * scale, stretchRatio2 * scale, stretchRatio3 * scale])

    rotationMatrix = Matrix{Float32}(undef, 3, 3)
    rotationMatrix[:, 1] .= stretchDirection1
    rotationMatrix[:, 2] .= stretchDirection2
    rotationMatrix[:, 3] .= stretchDirection3

    if det(rotationMatrix) < 0
        rotationMatrix[:, 1] *= -1
    end

    transform = rotationMatrix * scaleMatrix

    broadcast!(*, points, Ref(transform), points)
    broadcast!(+, points, points, Ref(position))

    indices = Vector{Tuple{UInt16,UInt16,UInt16}}(undef, (nPhi - 1) * nTheta * 2)
    i = 1
    for y in 1:(nPhi-1)
        for x in 1:nTheta
            r1, r2 = sq_triangle_indices_t(x, y, nTheta)
            @inbounds indices[i] = r1
            @inbounds indices[i+1] = r2
            i += 2
        end
    end

    # broadcasting version - sadly can't find a way to avoid the hcat
    #indices = Matrix{UInt16}(undef, (3, (nPhi - 1) * nTheta * 2)) # triangles over the points

    #=xs = UInt16.([1:nTheta;])
        idx = 1
        for y in 1:(nPhi-1)
            r = hcat(sq_triangle_indices.(xs, Ref(y), Ref(nTheta)))
            ncol = size(r)[2]
            indices[:, idx:idx+ncol-1] = r
            idx += ncol
        end
        triFaces = call_trifaces.(eachcol(indices))
     =#
    f = TriangleFace.(indices)
    return points, f
end


function apply_alignment(rot_tuple::Tuple{Matrix{Float32},Bool}, ap_tuple::Tuple{Matrix{Float32},Matrix{Float32}})
    rot, flip = rot_tuple
    s1, s2 = ap_tuple

    s1a = rot * s1'
    s2a = rot * s2'

    init = flip ? s2a : s1a
    final = flip ? s1a : s2a

    # can we do this without a copy?
    return (Matrix(init'), Matrix(final'))
end

function angle(a, b)
    return acosd(clamp(a ⋅ b / (norm(a) * norm(b)), -1, 1))
end

function fractionalAnisotropy(ev::Vector{Float64})
    meanEV = (ev[1] + ev[2] + ev[3]) / 3.0
    a =
        sqrt((ev[1] - meanEV)^2 + (ev[2] - meanEV)^2 + (ev[3] - meanEV)^2) /
        sqrt(ev[1]^2 + ev[2]^2 + ev[3]^2)
    return sqrt(3.0 / 2.0) * a
end

function center_atom_positions(p)
    cm = mean(p, dims=1)
    return (p .- cm), cm
end

function com(p, weights)
    # assume weights to be positive
    return sum(p .* weights, dims=1) ./ sum(weights)
end

function pure_align(P, Q)
    # finds transformation from Q to P
    H = P' * Q
    #R = sqrt(H' * H) * inv(H) #alternate formulation, not always stable
    F = svd(H, full=true, alg=LinearAlgebra.QRIteration())
    R = F.U * Diagonal([1, 1, det(F.U) * det(F.Vt)]) * F.Vt
    # seems like flip step causes volumes to fail
    return R, norm(P - Q * R)
end

# finds the centroid between a group of clusters
function find_group_centroid(clusters::Set{Int}, cd::ClusterData, t_list::Vector{Transition})
    ts = get_transitions(cd, clusters)
    mtx_idx = map(x -> cd.t_to_mtx[x], ts)

    dist_sum = map(x -> sum(view(cd.matrix, x, mtx_idx)), mtx_idx)
    ref_t_idx = mtx_idx[argmin(dist_sum)]

    return t_list[cd.clustering.order[ref_t_idx]]
end

function split_delta(d)
    pos = ifelse.(d .> Float32(0.0), d .^ 1, Float32(0.0))
    neg = ifelse.(d .< Float32(0.0), d .^ 1, Float32(0.0))

    return (hcat(pos, neg), hcat(-neg, -pos))
end

function calculate_alignment(ref_t::Transition,
    ts::AbstractArray{Transition},
    posMats::Dict{Transition,Tuple{Matrix{Float32},Matrix{Float32}}},
    features::Dict{State,Matrix{Float32}})::Dict{Transition,Tuple{Matrix{Float32},Bool}}

    rot = Dict{Transition,Tuple{Matrix{Float32},Bool}}()
    ref_s1_pos = posMats[ref_t][1]

    ref_s1, ref_s2 = ref_t
    #TODO: calculate beforehand and save 
    ref_diff = features[ref_s2] - features[ref_s1]
    split_ref = split_delta(ref_diff)
    ref_s1_com = reduce(vcat, map(x -> com(ref_s1_pos, x), eachcol(split_ref[1])))

    ref_s1_shift = mean(ref_s1_com, dims=1)
    ref_s1_com = ref_s1_com .- ref_s1_shift

    for t in ts
        if t != ref_t
            s1, s2 = t
            t_diff = features[s2] - features[s1]
            split_diff = split_delta(t_diff)

            t_s1_pos = posMats[t][1]
            t_s1_com = reduce(vcat, map(x -> com(t_s1_pos, x), eachcol(split_diff[1])))
            t_s1_shift = mean(t_s1_com, dims=1)
            t_s1_com = t_s1_com .- t_s1_shift

            t_s2_pos = posMats[t][2]
            t_s2_com = reduce(vcat, map(x -> com(t_s2_pos, x), eachcol(split_diff[2])))
            t_s2_shift = mean(t_s2_com, dims=1)
            t_s2_com = t_s2_com .- t_s2_shift

            R1, res1 = pure_align(ref_s1_com, t_s1_com)
            R2, res2 = pure_align(ref_s1_com, t_s2_com)

            R = (res1 < res2) ? R1 : R2
            rot[t] = (R, res1 > res2)
        end
    end

    rot[ref_t] = (Matrix(1.0I, 3, 3), false)
    return rot
end

function int_sqrt(x)
    n = Int(floor(sqrt(x)))
    if (n * n) == x
        return n, n
    end
    while mod(x, n) != 0
        n -= 1
    end
    m = div(x, n)
    return m, n
end

function labels_to_similarity_mat(labels)
    m = zeros(length(labels), length(labels))
    u = get_indices_of_unique_elements(labels)
    for (i, x) in enumerate(labels)
        m[i, u[x]] .= 1
    end
    return m
end

function get_indices_of_unique_elements(arr)
    unique_indices = Dict{eltype(arr),Vector{Int}}()
    for (index, value) in enumerate(arr)
        if haskey(unique_indices, value)
            push!(unique_indices[value], index)
        else
            unique_indices[value] = [index]
        end
    end
    return unique_indices
end


function fowlkes_mallows_index(labels1::Vector{T}, labels2::Vector{T}) where {T}
    # Ensure label vectors have the same length
    if length(labels1) != length(labels2)
        throw(ArgumentError("Label vectors must have the same length"))
    end

    n = length(labels1)
    if n < 2
        return 1.0 # Or throw an error, depending on desired behavior for trivial cases
    end

    # Create a contingency matrix
    contingency_matrix = StatsBase.counts(labels1, labels2)

    # Calculate agreements
    # TP: Number of pairs of points that are in the same cluster in both labels.
    # This is the sum of n_ij * (n_ij - 1) / 2 for all cells in the contingency matrix.
    sum_contingency = sum(n * (n - 1) / 2 for n in contingency_matrix)

    # Sums for each label vector
    sum1 = sum(c * (c - 1) / 2 for c in sum(contingency_matrix, dims=2))
    sum2 = sum(c * (c - 1) / 2 for c in sum(contingency_matrix, dims=1))

    # Calculate the index components
    TP = sum_contingency
    FP = sum1 - TP
    FN = sum2 - TP

    # Compute the Fowlkes-Mallows Index
    # Handle the case of zero denominator
    denominator = sqrt(TP + FP) * sqrt(TP + FN)
    if denominator == 0
        return 0.0
    else
        return TP / denominator
    end
end

function to_ranked(m)
    return denserank(vec(m))
end


function get_optimal_k(clustering, dm)
    ss = []
    dd = []
    for k in collect(2:50)
        labels = cutree(clustering, k=k)
        s = clustering_quality(labels, dm; quality_index=:silhouettes)
        d = clustering_quality(labels, dm; quality_index=:dunn)
        push!(ss, s)
        push!(dd, d)
    end

    return argmax(dd) + 1
end
