using GeometryBasics, NearestNeighbors

function kernelFunction(point::Point3f, atomPosition::Point3f, width::Float64)::Float32
    scale = 1 / ((2pi)^(3 / 2) * width^3)
    return scale * exp(-1 * (squaredNorm(point - atomPosition)) / (2 * width^2))
end

function squaredNorm(a::Point3f)::Float32
    return a[1] * a[1] + a[2] * a[2] + a[3] * a[3]
end

function calc_vols(sr, nn, kw, points, ts, kd, ap, iv)
    sample_range = (length(sr[1]), length(sr[2]), length(sr[3]))
    return calculateVolumes(ts, sample_range, ap, kd, points, nn, kw, iv)
end

function calculateVolumes(transitions, sampleRange, alignedPositions, stateKDTree, points, num_neighbors, kernelWidth, invariant)
    volMax = floatmin(Float32)
    volMin = floatmax(Float32)
    absVolMin = floatmax(Float32)

    volData = Array{Array{Float32}}(undef, length(transitions))
    for (rel_idx, t_idx) in enumerate(transitions)
        vd = Array{Float32,3}(zeros(sampleRange))

        pos1 = alignedPositions[rel_idx]
        kdTree1 = stateKDTree[rel_idx]
        for ((i, j, k), point) in points
            knn, dists = NearestNeighbors.knn(kdTree1, point, num_neighbors)
            kValue = sum(kernelFunction.(Ref(point), pos1[knn], kernelWidth) .* invariant[rel_idx][knn])
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

