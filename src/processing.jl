using StatsBase, SparseArrays, Distances, LinearAlgebra
using ProgressMeter

function buildBonds(positions, bondDelta, connectivity)
    cartesians = findall(isone, connectivity)
    indices = Tuple.(cartesians)
    weights = bondDelta[cartesians]
    points = map(((i, j),) -> (Point3f(positions[i, :]), Point3f(positions[j, :])), indices)

    return (points, weights, indices)
end

function calc_bonds(connectivity)
    return Tuple.(findall(isone, connectivity))
end


# Moment feature map
function moment_map(diagram, max_level, H::Int64)
    # For H0, just compute lifetime moments
    if H == 0
        numMoments = max_level
        mu = zeros(numMoments)
        #y = persistenceDiagram[:,2] - persistenceDiagram[:,1]
        y = persistence.(diagram[H+1])
        pop!(y)
        #@show last(y)
        for i = 1:max_level
            mu[i] = sum((y .^ i)) / sqrt(factorial(i))
        end
        return mu
    else
        numMoments = Int(max_level * (max_level + 1) / 2)
        mu = zeros(numMoments)

        mcount = 1
        x = birth.(diagram[H+1])
        y = persistence.(diagram[H+1])

        for i = 1:max_level
            for j = 1:i
                mu[mcount] = sum((x .^ (i - j)) .* (y .^ j)) * sqrt(binomial(i, j) / factorial(i))
                mcount += 1
            end
        end

        return mu
    end
end

function invLerp(a, b, v)
    return (v - a) / (b - a)
end

function moment_map(diagram, max_level)
    M0 = moment_map(diagram, max_level, 0)
    M1 = moment_map(diagram, max_level, 1)
    M2 = moment_map(diagram, max_level, 2)
    return vcat(M0, M1, M2)
end

function computeMoment(values::Vector{Float64}, moment::Int64)
    meanValue = mean(values)

    if moment == 1
        return meanValue
    end

    valAboutMean = values .- meanValue
    return sum(valAboutMean .^ moment) / length(values)
end

function computeInvariantDistributionInNeighborhood(data::Vector{Float64}, positions::Vector{Point3f}, binEdges::Vector{Float64}, neighborCount::Int64, atomKDTree::NearestNeighbors.KDTree)::Vector{SparseVector{Float64}}
    distributionsAtPositions = Vector{SparseVector{Float64}}()
    for position in positions
        nns, dists = knn(atomKDTree, position, neighborCount)
        #add minimum distance
        vals = data[nns]
        histo = fit(Histogram, vals, binEdges; closed=:right)
        pd = StatsBase.normalize(histo; mode=:probability)
        push!(distributionsAtPositions, sparse(pd.weights))
    end
    return distributionsAtPositions
end

function computeLNCD(distributions::Dict{Tuple{Int,Int},Vector{SparseVector{Float64}}}, a::Tuple{Int,Int}, b::Tuple{Int,Int}, selected_atoms)::Float64 #local neighborhood cummulative diverge score

    informationScore = 0.0
    distA = distributions[a]
    distB = distributions[b]

    vals = Dict()
    for i in selected_atoms
        informationScore = informationScore + JSDivergence()(distA[i], distB[i])
        minDiv = floatmax(Float64)
        minIdx = i
        for j in eachindex(distB)
            div = JSDivergence()(distA[i], distB[j])
            if minDiv > div
                minDiv = div
                minIdx = j
            end
        end
        vals[i] = (minIdx, minDiv)
    end

    return informationScore
end

function computeDistances(invariants::Vector{Float64})::Matrix{Float64}
    out = zeros(length(invariants), length(invariants))
    Threads.@threads for k in 1:length(invariants)
        @inbounds out[k, k] = 0.0
        for j in 1:(k-1)
            @inbounds out[j, k] = abs(invariants[j] - invariants[k])
        end
    end
    return Symmetric(out)
end

function computeTransitionInvariants(
    transitions::Vector{Tuple{Int16,Int16}},
    alignedPositions::Dict{Tuple{Int16,Int16},Tuple{Matrix{Float32},Matrix{Float32}}},
    distances::Dict{Int16,Matrix{Float32}}
)::Tuple{Dict{Tuple{Int16,Int16},Vector{Float32}},Dict{Tuple{Int16,Int16},Vector{Float32}},
    Dict{Tuple{Int16,Int16},Vector{Float32}},Dict{Tuple{Int16,Int16},Vector{Vector{Vec3f}}}}

    transitionInvariants1 = Dict{Tuple{Int16,Int16},Vector}()
    transitionInvariants2 = Dict{Tuple{Int16,Int16},Vector}()
    transitionInvariants3 = Dict{Tuple{Int16,Int16},Vector}()
    stretchedPrincipalAxes = Dict{Tuple{Int16,Int16},Vector{Vector{Vec3f}}}()

    @showprogress for t in transitions
        s1, s2 = t
        aPos1, aPos2 = alignedPositions[t]
        # creates matrices with INF values if distance sum has 0s
        weights = 1 ./ ((distances[s1] + distances[s2]) ./ 2)

        replace!(weights, Inf => 0)
        F = Vector{Matrix{Float32}}(undef, length(aPos1[:, 1]))

        for m = 1:length(aPos1[:, 1])

            D = zeros(3, 3)
            A = zeros(3, 3)

            for n = 1:length(aPos1[:, 1]) #nnIndices[m] 
                if m == n || weights[m, n] < 0.001
                    continue
                end

                deltaXmn = aPos1[n, :] - aPos1[m, :]
                D = D + (deltaXmn * transpose(deltaXmn) * weights[m, n])

                deltaxmn = aPos2[n, :] - aPos2[m, :]
                A = A + (deltaxmn * transpose(deltaXmn) * weights[m, n])
            end
            F[m] = A * inv(D)
        end

        i = Matrix(1.0I, 3, 3)
        E = 0.5 .* (transpose.(F) .* F .- Ref(i))   #lagrangian Green
        eigenSystems = eigen.(E)

        getStretchedEigVec(eigenSys) = [
            sqrt(2 * eigenSys.values[1] + 1.0) * eigenSys.vectors[:, 1],
            sqrt(2 * eigenSys.values[2] + 1.0) * eigenSys.vectors[:, 2],
            sqrt(2 * eigenSys.values[3] + 1.0) * eigenSys.vectors[:, 3],
        ]

        stretchedPrincipalAxes[t] = [Vec3f.(getStretchedEigVec(eigSys)) for eigSys in eigenSystems]

        deviator = E .- (1 / 3 * tr.(E) .* Ref(i))
        eigenSystemsDeviator = eigen.(deviator)

        I1(ev::Vector{Float64}) = ev[1] + ev[2] + ev[3]
        I2(ev::Vector{Float64}) = sqrt(ev[1]^2 + ev[2]^2 + ev[3]^2)
        I3(ev::Vector{Float64}) =
            3 * sqrt(6) * (ev[1] * ev[2] * ev[3]) / ((ev[1]^2 + ev[2]^2 + ev[3]^2)^(3 / 2))

        invariant1 = [I1(eigenSystem.values) for eigenSystem in eigenSystems]
        invariant2 = [I2(eigenSystem.values) for eigenSystem in eigenSystemsDeviator]
        invariant3 = [I3(eigenSystem.values) for eigenSystem in eigenSystemsDeviator]

        transitionInvariants1[t] = invariant1
        transitionInvariants2[t] = invariant2
        transitionInvariants3[t] = invariant3
    end
    return transitionInvariants1, transitionInvariants2, transitionInvariants3, stretchedPrincipalAxes

end

"""
sort_transitions(rel, seq, dm)
sorts transitions relative to their distance to the specified transition using
distance matrix dm
"""
function sort_transitions(rel::Tuple{Int,Int}, seq::Vector{Tuple{Int,Int}}, dm::Matrix{Float32})
    # get row of rel
    idx = findfirst(item -> item == rel, seq)
    row = dm[idx, :]

    return map((x) -> x[2], sort(collect(zip(row, seq)), by=first))
end

