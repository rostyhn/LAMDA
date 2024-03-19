function squaredNorm(a::Point3f )::Float32
   return a[1]*a[1] + a[2]*a[2] + a[3]*a[3]
end

function kernelFunction(point::Point3f, atomPosition::Point3f ,width::Float64 )::Float32
    scale = 1/((2pi)^(3/2) * width^3 )
    return scale * exp( -1* (squaredNorm(point - atomPosition))/(2*width^2) )
end 

# Moment feature map
function moment_map(diagram, max_level, H::Int64)


    # For H0, just compute lifetime moments
    if H==0
        numMoments = max_level
        mu = zeros(numMoments)
        #y = persistenceDiagram[:,2] - persistenceDiagram[:,1]
        y = persistence.(diagram[H+1])
        pop!(y)
        #@show last(y)
        for i = 1:max_level
            mu[i] = sum((y.^i))/sqrt(factorial(i))
        end
        return mu
    else
        numMoments = Int(max_level*(max_level+1)/2)
        mu = zeros(numMoments)

         mcount = 1
         x = birth.(diagram[H+1])
         y = persistence.(diagram[H+1])

        for i = 1:max_level
            for j = 1:i
                mu[mcount] = sum((x.^(i-j)).*(y.^j))*sqrt(binomial(i,j)/factorial(i))
                mcount += 1
            end
        end
        
        return mu
    end
end

function invLerp(a,b,v)
    return (v - a) / (b - a)
end


function moment_map( diagram, max_level )
    M0 = moment_map(diagram, max_level, 0)
    M1 = moment_map(diagram, max_level, 1)
    M2 = moment_map(diagram, max_level, 2)
    return vcat(M0, M1, M2)
end

# function moment_map_normalized( diagram, max_level )
#     subsample=200
#     N = 200

#     M0 = moment_map(diagram.*(sf^(1/3), max_level, 0)/sf
#     M1 = moment_map(diagram.*(sf^(1/3)), max_level, 1)/sf
#     # M2 = moment_map(diagram.*(sf^(1/3)), max_level, 2)/sf
#     return vcat(M0, M1)
# end

function computeMoment( values::Vector{Float64}, moment::Int64 )
    meanValue = mean(values)

    if moment == 1
        return meanValue
    end

    valAboutMean = values .- meanValue
    return sum( valAboutMean.^moment )/length(values) 
end

function computePersistenceDistances( persistenceDiagrams::Vector )::Matrix{Float64}
    out = zeros(length(persistenceDiagrams),length(persistenceDiagrams) )
    @showprogress Threads.@threads for k in 1:length(persistenceDiagrams)
        @inbounds out[k,k] = 0.0
        for j in 1:(k-1) 
            @inbounds out[j,k] = Wasserstein()(persistenceDiagrams[j], persistenceDiagrams[k])
        end
    end
    return Symmetric(out)
end

function computeDistances( invariants::Vector{Float64} )::Matrix{Float64}
    out = zeros(length(invariants),length(invariants) )
    Threads.@threads for k in 1:length(invariants)
        @inbounds out[k,k] = 0.0
        for j in 1:(k-1) 
            @inbounds out[j,k] = abs(invariants[j] - invariants[k])
        end
    end
    return Symmetric(out)
end


function computeTransitionInvariants(
    sequence::Vector{String},
    atomPositions::Dict{String,Matrix{Float64}},
    distanceMatrices::Dict{String,Matrix{Float64}},
)::Tuple{Dict{String, Vector{Point3f}},Dict{String,Vector{Float64}},Dict{String,Vector{Float64}},
Dict{String,Vector{Float64}}, Dict{String, Vector{Vector{Vec3f}}}}


    transitionInvariants1 = Dict{String,Vector}()
    transitionInvariants2 = Dict{String,Vector}()
    transitionInvariants3 = Dict{String,Vector}()

    transitionReferencePosition = Dict{String, Vector}()

    stretchedPrincipalAxes = Dict{String, Vector{Vector{Vec3f}}}()

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
            sqrt( 2 * eigenSys.values[1] + 1.0) * eigenSys.vectors[:, 1],
            sqrt( 2 * eigenSys.values[2] + 1.0) * eigenSys.vectors[:, 2],
            sqrt( 2 * eigenSys.values[3] + 1.0) * eigenSys.vectors[:, 3],
        ]

        stretchedPrincipalAxes[transitionName] = [ Vec3f.(getStretchedEigVec(eigSys)) for eigSys in eigenSystems]

        # @show sqrt( 2 * eigenSystems[1].values[1] + 1.0)
        # @show sqrt( 2 *eigenSystems[2] + 1.0)
        # @show sqrt( 2 *eigenSystems[1]+ 1.0)

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

    return transitionReferencePosition, transitionInvariants1, transitionInvariants2, transitionInvariants3, stretchedPrincipalAxes

end
