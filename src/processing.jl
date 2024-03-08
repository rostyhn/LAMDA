

function computeTransitionInvariants(
    sequence::Vector{String},
    atomPositions::Dict{String,Matrix{Float64}},
    distanceMatrices::Dict{String,Matrix{Float64}},
)::Tuple{Dict{String, Vector{Point3f}},Dict{String,Vector{Float64}},Dict{String,Vector{Float64}},Dict{String,Vector{Float64}}}


    transitionInvariants1 = Dict{String,Vector}()
    transitionInvariants2 = Dict{String,Vector}()
    transitionInvariants3 = Dict{String,Vector}()

    transitionReferencePosition = Dict{String, Vector}()

    unique = 1
    @showprogress for sequenceStep = 1:(length(sequence)-1)

        currentState = sequence[sequenceStep]
        nextState = sequence[sequenceStep+1]

        transitionName = currentState * ">" * nextState

        if haskey(transitionInvariants1, currentState * ">" * nextState)
            continue
        end
        unique = unique + 1

        aPos1 = atomPositions[currentState]
        aPos2 = atomPositions[nextState]


        transitionReferencePosition[transitionName] = [ Makie.Point3f.( aPos1[i, 1], aPos1[i, 2], aPos1[i, 3] ) for i in 1:length(aPos1[:,1])]
        

        weights = 1 ./ ((distanceMatrices[currentState] + distanceMatrices[nextState]) ./ 2)

        F = Vector{Matrix{Float64}}(undef, length(aPos1[:, 1]))

        #  kdtree = KDTree(transpose(aPos1); leafsize = 5)
        #  nnIndices, dists = knn(kdtree, transpose(aPos1), 10)

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

        I = zeros(3, 3)
        I[1, 1] = 1.0
        I[2, 2] = 1.0
        I[3, 3] = 1.0

        E = 0.5 .* (transpose.(F) .* F .- Ref(I))   #lagrangian Green
        eigenSystems = eigen.(E)

        getStretchedEigVec(eigenSys) = [
            eigenSys.values[1] * eigenSys.vectors[:, 1],
            eigenSys.values[2] * eigenSys.vectors[:, 2],
            eigenSys.values[3] * eigenSys.vectors[:, 3],
        ]

        deviator = E .- (1 / 3 * tr.(E) .* Ref(I))
        eigenSystemsDeviator = eigen.(deviator)

        I1(ev::Vector{Float64}) = ev[1] + ev[2] + ev[3]
        # I2( ev::Vector{Float64} ) = ev[1]*ev[2] + ev[1]*ev[3] + ev[2]*ev[3]
        # I3( ev::Vector{Float64} ) = ev[1] * ev[2] * ev[3]
        I2(ev::Vector{Float64}) = sqrt(ev[1]^2 + ev[2]^2 + ev[3]^2)
        I3(ev::Vector{Float64}) =
            3 * sqrt(6) * (ev[1] * ev[2] * ev[3]) / ((ev[1]^2 + ev[2]^2 + ev[3]^2)^(3 / 2))

        invariant1 = [I1(eigenSystem.values) for eigenSystem in eigenSystems]
        invariant2 = [I2(eigenSystem.values) for eigenSystem in eigenSystemsDeviator]
        invariant3 = [I3(eigenSystem.values) for eigenSystem in eigenSystemsDeviator]

        transitionInvariants1[transitionName] = invariant1
        transitionInvariants2[transitionName] = invariant2
        transitionInvariants3[transitionName] = invariant3

    end

    println("Found * $(unique) * unique transitions")

    return transitionReferencePosition, transitionInvariants1, transitionInvariants2, transitionInvariants3

end
