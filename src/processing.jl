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

function invLerp(a, b, v)
    return (v - a) / (b - a)
end

function computeTransitionInvariants(
    transitions::Vector{Transition},
    alignedPositions::Dict{Transition,Tuple{Matrix{Float32},Matrix{Float32}}},
    distances::PyDict{State,Matrix{Float32}}
)::Tuple{Dict{Transition,Vector{Float32}},Dict{Transition,Vector{Float32}},
    Dict{Transition,Vector{Float32}},Dict{Transition,Vector{Vector{Vec3f}}}}

    transitionInvariants1 = Dict{Transition,Vector{Float32}}()
    transitionInvariants2 = Dict{Transition,Vector{Float32}}()
    transitionInvariants3 = Dict{Transition,Vector{Float32}}()
    stretchedPrincipalAxes = Dict{Transition,Vector{Vector{Vec3f}}}()

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

                deltaXmn = view(aPos1, n, :) - view(aPos1, m, :)
                D = D + (deltaXmn * transpose(deltaXmn) * weights[m, n])

                deltaxmn = view(aPos2, n, :) - view(aPos2, m, :)
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
