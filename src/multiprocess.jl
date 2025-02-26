using GeometryBasics, NearestNeighbors

function kernelFunction(point::Point3f, atomPosition::Point3f, width::Float64)::Float32
    scale = 1 / ((2pi)^(3 / 2) * width^3)
    return scale * exp(-1 * (squaredNorm(point - atomPosition)) / (2 * width^2))
end

function squaredNorm(a::Point3f)::Float32
    return a[1] * a[1] + a[2] * a[2] + a[3] * a[3]
end

function calc_vols(d_ch, sr, nn, kw, ts, kd, ap, iv)
    sample_range = (length(sr[1]), length(sr[2]), length(sr[3]))

    points = Vector{Tuple{Tuple{Int,Int,Int},Point3f}}()
    for i in eachindex(sr[1]) # x
        for j in eachindex(sr[2]) # y
            for k in eachindex(sr[3]) # z
                point = Point3f(sr[1][i], sr[2][j], sr[3][k])
                push!(points, ((i, j, k), point))
            end
        end
    end

    sc_size = 25
    for sc in zip(Iterators.partition(ts, sc_size), Iterators.partition(kd, sc_size), Iterators.partition(ap, sc_size), Iterators.partition(iv, sc_size))
        # does not take up memory
        s_ts = sc[1]
        s_kd = sc[2]
        s_ap = sc[3]
        s_iv = sc[4]

        vd, volmin, volmax, absvolmin = calculateVolumes(s_ts, sample_range, s_ap, s_kd, points, nn, kw, s_iv)
        put!(d_ch, (vd, volmin, volmax, absvolmin))
        vd = nothing
        GC.gc()
    end
    println("finished processing")
end

function calculateVolumes(transitions, sampleRange, alignedPositions, stateKDTree, points, num_neighbors, kernelWidth, invariant)
    volMax = floatmin(Float32)
    volMin = floatmax(Float32)
    absVolMin = floatmax(Float32)

    volData = Array{Tuple{Int,Array{Float32}}}(undef, length(transitions))
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
        volData[rel_idx] = (t_idx, vec(vd))
    end
    return volData, volMin, volMax, absVolMin
end

