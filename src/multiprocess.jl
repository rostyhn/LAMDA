function kernelFunction(point::Point3f, atomPosition::AbstractArray{Float32}, width::Float64)::Float32
    scale = 1 / ((2pi)^(3 / 2) * width^3)
    return scale * exp(-1 * (squaredNorm(point - atomPosition)) / (2 * width^2))
end

function squaredNorm(a::Point3f)::Float32
    return a[1] * a[1] + a[2] * a[2] + a[3] * a[3]
end

function calc_vols(sr::Tuple{Vector{Float64},Vector{Float64},Vector{Float64}},
    nn::Int,
    kw::Float64,
    points::Vector{Tuple{Tuple{Int,Int,Int},Point3f}},
    ts::AbstractArray{Int},
    kd::AbstractArray{<:KDTree},
    ap::AbstractArray{Matrix{Float32}},
    iv::AbstractArray{Vector{Float32}})

    sample_range = (length(sr[1]), length(sr[2]), length(sr[3]))

    volMax = floatmin(Float32)
    volMin = floatmax(Float32)
    absVolMin = floatmax(Float32)

    volData = Array{Array{Float32}}(undef, length(ts))
    for (rel_idx, t_idx) in enumerate(ts)
        vd = Array{Float32,3}(zeros(sample_range))

        pos1 = ap[rel_idx]
        kdTree1 = kd[rel_idx]
        iv1 = iv[rel_idx]
        for ((i, j, k), point) in points
            knn, dists = NearestNeighbors.knn(kdTree1, point, nn)
            kValue = sum(kernelFunction.(Ref(point), eachrow(view(pos1, knn, :)), kw) .* view(iv1, knn))
            vd[i, j, k] = kValue
            volMax = max(volMax, kValue)
            volMin = min(volMin, kValue)
            absVolMin = min(absVolMin, abs(kValue))
        end
        # should be returning something else?
        volData[rel_idx] = vec(vd)
    end
    return volData, volMin, volMax, absVolMin
end

