module TransVis

#Data handling
using GLMakie: apply_transform
using Makie: MakieCore, ray_at_cursor, position_on_plot, mouse_in_scene, shift_project, update_tooltip_alignment!, parent_scene, show_data
using Pickle
using JLD2
using CodecZlib

#Vis
using GLMakie
using Makie
using GeometryBasics

#Processing and Helpers
using NearestNeighbors
using LinearAlgebra
using TSne
using Base.Threads
using Statistics
using ProgressMeter
using Ripserer
using PersistenceDiagrams

include("io.jl")
include("processing.jl")
include("SelectionWindow.jl")
include("MolWindow.jl")
include("utils.jl")
include("math.jl")

export go

function buildBonds(key, volumeDataDict, bondDelta, atomPositions)
    points = Vector{Tuple{Point3f,Point3f}}()
    weights = Vector{Float64}()
    indices = Vector{Tuple{Int64,Int64}}()
    for i in 1:length(bondDelta[1, :])
        for j in 1:i
            v1 = volumeDataDict[i]
            v2 = volumeDataDict[j]
            bw = bondDelta[i, j]

            # avg = (abs((v1 + v2)) / 2) / volumeAbsMax
            # 0.05 is the threshold val for filtering
            # check against bond weight to make sure we're only looking at "real" bonds
            if abs(bw) > 0.0
                push!(points, (Point3f(atomPositions[key][i, :]), Point3f(atomPositions[key][j, :])))
                push!(weights, bw)
                push!(indices, (i, j))
            end
        end
    end
    return (points, weights, indices)
end

function go()

    GLMakie.closeall() #close all windows for rerun!

    plotWindow = Figure(size=(600, 400))

    axI1 = Axis(plotWindow[1:2, 1:4], xlabel="Atom Number", ylabel="K1")
    axI2 = Axis(plotWindow[3:4, 1:4], xlabel="Atom Number", ylabel="K2")
    axI3 = Axis(plotWindow[5:6, 1:4], xlabel="Atom Number", ylabel="mode(E)")
    axDR = Axis(plotWindow[1:6, 5:7], title="t-SNE")

    stateDataPath = "/home/frosty/Programs/Julia/TransVis/data/state_data copy/"
    sequencePath = "/home/frosty/Programs/Julia/TransVis/data/state_data copy/seq.txt"
    transitionPath = "/home/frosty/Programs/Julia/TransVis/data/nano_pt_labels.pickle"
    bondWeightPath = "/home/frosty/Programs/Julia/TransVis/data/bond_weights.pickle"
    connectivityPath = "/home/frosty/Programs/Julia/TransVis/data/connectivity.pickle"
    transitionDistanceMatrixPath = "/home/frosty/Programs/Julia/TransVis/data/distance_matrix.pickle"
    transitionSequencePath = "/home/frosty/Programs/Julia/TransVis/data/transition_sequence.pickle"

    combinedData = getDataSets(stateDataPath, sequencePath, transitionPath, bondWeightPath, connectivityPath, transitionDistanceMatrixPath, transitionSequencePath)

    transitionInvariants1 = combinedData["transitionInvariants1"]
    transitionInvariants2 = combinedData["transitionInvariants2"]
    transitionInvariants3 = combinedData["transitionInvariants3"]

    transitionDistanceMatrix = combinedData["transitionDistanceMatrix"]["matrix"]
    transitionSequence = combinedData["transitionSequence"]["sequence"]
    transitionRefPositions = combinedData["transitionRefPositions"]
    transitionLabels = combinedData["transitionLabels"]

    distanceMatrices = combinedData["distanceMatrices"]

    stretchedPrincipalAxes = combinedData["stretchedPrincipalAxes"]

    atomPositions = combinedData["atomPositions"]

    # get number of atoms
    num_atoms = size(Iterators.first(values(atomPositions)))[1]

    connectivity = combinedData["connectivity"]
    bondWeights = combinedData["bondWeights"]

    #get min max of all transition invariants 1
    minInvariant1 = 1.0e10
    maxInvariant2 = -1.0e10
    @time for (key, value) in transitionInvariants1
        for invariant1 in value
            if minInvariant1 > invariant1
                minInvariant1 = invariant1
            end
            if maxInvariant2 < invariant1
                maxInvariant2 = invariant1
            end
        end
    end
    println("invariant range:  $(minInvariant1) -  $(maxInvariant2)")

    invariant1MaxRange = max(abs(minInvariant1), abs(maxInvariant2))

    #get min max coordinates of atoms for bounding box
    minX = 1.0e10
    minY = 1.0e10
    minZ = 1.0e10
    maxX = -1.0e10
    maxY = -1.0e10
    maxZ = -1.0e10
    @time for (key, positions) in atomPositions
        for row in 1:length(positions[:, 1])
            if minX > positions[row, 1]
                minX = positions[row, 1]
            end
            if maxX < positions[row, 1]
                maxX = positions[row, 1]
            end
            if minY > positions[row, 2]
                minY = positions[row, 2]
            end
            if maxY < positions[row, 2]
                maxY = positions[row, 2]
            end
            if minZ > positions[row, 3]
                minZ = positions[row, 3]
            end
            if maxZ < positions[row, 3]
                maxZ = positions[row, 3]
            end
        end
    end

    volumeResolution = 0.2

    sampleRangeX = [minX-2*volumeResolution:volumeResolution:maxX+2*volumeResolution;]
    sampleRangeY = [minY-2*volumeResolution:volumeResolution:maxY+2*volumeResolution;]
    sampleRangeZ = [minZ-2*volumeResolution:volumeResolution:maxZ+2*volumeResolution;]

    @show length(sampleRangeX)
    @show length(sampleRangeY)
    @show length(sampleRangeZ)

    @show "computing kd trees"
    transitionKDTree = Dict{Tuple{Int,Int},KDTree}()
    @time for (key, value) in transitionRefPositions
        transitionKDTree[key] = KDTree(value)
    end

    kernelWidth = 1.0

    featureVectorMatrix = zeros(
        length(values(transitionInvariants1)),
        length(values(transitionInvariants1) |> first) * 1,
    )
    labels = zeros(length(values(transitionInvariants1)))


    mapNameToIdx = Dict{Tuple{Int,Int},Int64}()
    mapIdxToName = Vector{Tuple{Int,Int}}()

    row = 1
    for (transition, invariants1) in transitionInvariants1
        mapNameToIdx[transition] = row
        push!(mapIdxToName, transition)

        col = 1
        for invariant in invariants1
            featureVectorMatrix[row, col] = invariant |> abs
            col = col + 1
        end

        s1, s2 = transition
        flippedName = (s2, s1)

        if haskey(transitionLabels, transition)
            labels[row] = transitionLabels[transition]
        elseif haskey(transitionLabels, flippedName)
            labels[row] = transitionLabels[flippedName]
        else
            println("Transition not found: $(string(transition))")
        end

        row = row + 1
    end

    #build distance distanceMatrices
    invariantDistances = zeros(length(transitionInvariants1[(1, 3)]), length(transitionInvariants1[(1, 3)])) #nAtoms x nAtoms
    invariantDistances = Vector{Matrix}(undef, length(transitionInvariants1))
    @time for (key, value) in transitionInvariants1
        invariantDistances[mapNameToIdx[key]] = computeDistances(value)
    end

    maxMoment = 10
    invariantMomentFeatures = zeros(length(transitionInvariants1), maxMoment)
    @time for (key, value) in transitionInvariants1
        row = mapNameToIdx[key]
        for moment in 1:maxMoment
            invariantMomentFeatures[row, moment] = computeMoment(value, moment)
        end
    end

    # 6 is the slope - should only be even odds
    # 0.1 is the thickness of the white part
    cmap = resample_cmap(:bam, 100; alpha=([(-0.99):0.02:(0.99);] ./ 0.1) .^ 6)
    @show "cmap Range"
    @show length(cmap)


    bondDeltas = Dict{Tuple{Int,Int},Matrix{Float64}}()
    transforms = Dict{Tuple{Int,Int},Matrix{Float64}}()
    for t in transitionSequence
        dm1 = distanceMatrices[t[1]] .* connectivity[t[1]]'
        dm2 = distanceMatrices[t[2]] .* connectivity[t[2]]'

        # for now it's total delta
        bondDeltas[t] = (dm2 - dm1)

        p1 = atomPositions[t[1]]
        p2 = atomPositions[t[2]]
        transforms[t] = (abs.(p2 - p1))
    end


    volAbsMax = Observable(0.0)
    lsExtrema = Observable((floatmax(Float64), floatmin(Float64)))
    molWindows = Vector()

    function on_click(t, on_window_hover)
        atomPosTuple = (atomPositions[t[1]], atomPositions[t[2]])
        volData = zeros(length(sampleRangeX), length(sampleRangeY), length(sampleRangeZ))
        volDataDict = Dict{Int,Any}()

        glyphResolution = 0.1

        kdTree = transitionKDTree[t]
        for i in eachindex(sampleRangeX) # x
            for j in eachindex(sampleRangeY) # y
                for k in eachindex(sampleRangeZ) # z
                    point = Point3f(sampleRangeX[i], sampleRangeY[j], sampleRangeZ[k])
                    knn, dists = NearestNeighbors.knn(kdTree, point, 5)
                    kValue = sum(kernelFunction.(Ref(point), transitionRefPositions[t][knn], kernelWidth) .* transitionInvariants1[t][knn])
                    volData[i, j, k] = kValue
                    idx, d = NearestNeighbors.nn(kdTree, point)
                    volDataDict[idx] = kValue
                end
            end
        end
        thisVolAbsMax = max(abs(minimum(volData)), abs(maximum(volData)))
        volAbsMax[] = max(volAbsMax[], thisVolAbsMax)

        # 1.0 should be transitionGlyphSize
        sq = superquadric.(1.0, transitionRefPositions[t], stretchedPrincipalAxes[t], transitionInvariants2[t], -1.0, 3.0, glyphResolution)[:]
        ls = buildBonds(t[1], volDataDict, bondDeltas[t], atomPositions)

        thislsExtrema = extrema(ls[2])
        lsExtrema[] = (min(lsExtrema[][1], thislsExtrema[1]), max(lsExtrema[][1], thislsExtrema[2]))

        mw = build_mol_window((600, 400), t, atomPosTuple, volData, volAbsMax, volDataDict, sq, ls, kdTree, sampleRangeX, sampleRangeY, sampleRangeZ, cmap, on_window_hover, lsExtrema)

        push!(molWindows, mw)

        link_cameras_lscene(molWindows)
    end

    screen = GLMakie.Screen()

    available_matrices = Dict{String,Dict{Tuple{Int,Int},Matrix{Float64}}}()
    available_matrices["transforms"] = transforms
    available_matrices["bondDeltas"] = bondDeltas

    sorted = sort_transitions(transitionSequence[1], transitionSequence, transitionDistanceMatrix)
    reference_configuration = first(transitionSequence)
    display(screen, build_selection_window((600, 800), available_matrices, transitionSequence, on_click, (first(values(atomPositions)), transitionKDTree[reference_configuration], num_atoms, transitionSequence[1]), (minInvariant1, maxInvariant2, transitionInvariants1), transitionRefPositions, transitionKDTree))
end
end
