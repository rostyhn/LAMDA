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

function buildBonds(positions, volumeDataDict, bondDelta)
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
                push!(points, (Point3f(positions[i, :]), Point3f(positions[j, :])))
                push!(weights, bw)
                push!(indices, (i, j))
            end
        end
    end
    return (points, weights, indices)
end

function go()

    GLMakie.closeall() #close all windows for rerun!

    trajectories, active_trajectory_name = get_data_alt()
    active_trajectory = trajectories[active_trajectory_name]

    plotWindow = Figure(size=(600, 400))

    axI1 = Axis(plotWindow[1:2, 1:4], xlabel="Atom Number", ylabel="K1")
    axI2 = Axis(plotWindow[3:4, 1:4], xlabel="Atom Number", ylabel="K2")
    axI3 = Axis(plotWindow[5:6, 1:4], xlabel="Atom Number", ylabel="mode(E)")
    axDR = Axis(plotWindow[1:6, 5:7], title="t-SNE")

    transitionInvariants1 = active_trajectory["t1"]
    transitionInvariants2 = active_trajectory["t2"]
    transitionInvariants3 = active_trajectory["t3"]
    stretchedPrincipalAxes = active_trajectory["stretchedPrincipalAxes"]
    stateKDTree = active_trajectory["kdTree"]
    dms = active_trajectory["dms"]

    #@show keys() # dms["graph"]

    # transitionDistanceMatrix = combinedData["transitionDistanceMatrix"]["matrix"]
    transitionSequence = active_trajectory["transitions"]

    # sometimes need to grab first transition for setting sizes
    firstTransition = Iterators.first(transitionSequence)
    firstState = firstTransition[1]

    connectivity = active_trajectory["connectivity"]
    distanceMatrices = active_trajectory["distanceMatrices"]
    alignedPositionsMatrices = active_trajectory["alignedPositionsMatrices"] # positions as matrices
    alignedPositions = active_trajectory["alignedPositions"] # positions as vec point3fs

    # get number of atoms
    num_atoms = size(Iterators.first(values(alignedPositionsMatrices))[1])[1]

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

    # get min max coordinates of atoms for bounding box
    # we don't really need these positions anymore
    minX = 1.0e10
    minY = 1.0e10
    minZ = 1.0e10
    maxX = -1.0e10
    maxY = -1.0e10
    maxZ = -1.0e10
    @time for (key, positions) in alignedPositionsMatrices
        p1, p2 = positions

        for row in 1:length(p1[:, 1])
            if minX > min(p1[row, 1], p2[row, 1])
                minX = min(p1[row, 1], p2[row, 1])
            end
            if maxX < max(p1[row, 1], p2[row, 1])
                maxX = max(p1[row, 1], p2[row, 1])
            end
            if minY > min(p1[row, 2], p2[row, 2])
                minY = min(p1[row, 2], p2[row, 2])
            end
            if maxY < max(p1[row, 2], p2[row, 2])
                maxY = max(p1[row, 2], p2[row, 2])
            end
            if minZ > min(p1[row, 3], p2[row, 3])
                minZ = min(p1[row, 3], p2[row, 3])
            end
            if maxZ < max(p1[row, 3], p2[row, 3])
                maxZ = max(p1[row, 3], p2[row, 3])
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

    kernelWidth = 1.0

    featureVectorMatrix = zeros(
        length(values(transitionInvariants1)),
        length(values(transitionInvariants1) |> first) * 1,
    )
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
        row = row + 1
    end

    # build distance distanceMatrices
    # move this to cache as well
    invariantDistances = zeros(length(transitionInvariants1[firstTransition]), length(transitionInvariants1[firstTransition])) #nAtoms x nAtoms
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
        s1, s2 = t
        dm1 = distanceMatrices[s1] .* connectivity[s1]'
        dm2 = distanceMatrices[s2] .* connectivity[s2]'

        # for now it's total delta
        bondDeltas[t] = (dm2 - dm1)

        p1, p2 = get_from_t_dict(alignedPositionsMatrices, t)
        transforms[t] = (abs.(p2 - p1))
    end

    volAbsMax = Observable(0.05)
    lsExtrema = Observable((floatmax(Float64), floatmin(Float64)))

    molScreen = GLMakie.Screen()
    molGrid = Figure()


    views = Vector()
    all_scenes = Vector()
    for i in 1:3
        l = LScene(
            molGrid[i, 1],
            show_axis=false,
            scenekw=(backgroundcolor=:black, clear=true),
        )
        r = LScene(
            molGrid[i, 2],
            show_axis=false,
            scenekw=(backgroundcolor=:black, clear=true),
        )
        push!(views, (l, r))
        push!(all_scenes, l)
        push!(all_scenes, r)

        Camera3D(l.scene, center=false, eyeposition=Vec3f(30, 30, 30))
        Camera3D(r.scene, center=false, eyeposition=Vec3f(30, 30, 30))
    end
    colsize!(molGrid.layout, 1, Relative(1 / 2))
    colsize!(molGrid.layout, 2, Relative(1 / 2))

    link_cameras_lscenes(all_scenes)

    filterRange = lift(x -> LinRange(-x, x, 100), volAbsMax)

    volFilter = IntervalSlider(molGrid[5, 1:2], range=filterRange, startvalues=(0, 0))
    Colorbar(molGrid[6, 1:2], colormap=cmap, limits=lift(x -> (-x, x), volAbsMax), vertical=false)

    Label(molGrid[4, 1:2], lift(x -> string(x), volFilter.interval))

    cleanup_callbacks = Dict()

    display(molScreen, molGrid)
    viewIdx = 1

    function on_click(t, on_window_hover)
        pos1, pos2 = alignedPositions[t]

        volData = zeros(length(sampleRangeX), length(sampleRangeY), length(sampleRangeZ))
        volDataDict = Dict{Int,Any}()

        glyphResolution = 0.1

        kdTree1, kdTree2 = stateKDTree[t]

        for i in eachindex(sampleRangeX) # x
            for j in eachindex(sampleRangeY) # y
                for k in eachindex(sampleRangeZ) # z
                    point = Point3f(sampleRangeX[i], sampleRangeY[j], sampleRangeZ[k])
                    knn, dists = NearestNeighbors.knn(kdTree1, point, 5)
                    kValue = sum(kernelFunction.(Ref(point), pos1[knn], kernelWidth) .* transitionInvariants1[t][knn])
                    volData[i, j, k] = kValue
                    idx, d = NearestNeighbors.nn(kdTree1, point)
                    volDataDict[idx] = kValue
                end
            end
        end
        thisVolAbsMax = max(abs(minimum(volData)), abs(maximum(volData)))

        volAbsMax[] = max(volAbsMax[], thisVolAbsMax)

        # 1.0 should be transitionGlyphSize
        sq = superquadric.(1.0, pos1, stretchedPrincipalAxes[t], transitionInvariants2[t], -1.0, 3.0, glyphResolution)[:]
        ls = buildBonds(alignedPositionsMatrices[t][1], volDataDict, bondDeltas[t])

        thislsExtrema = extrema(ls[2])
        lsExtrema[] = (min(lsExtrema[][1], thislsExtrema[1]), max(lsExtrema[][1], thislsExtrema[2]))

        l, r = views[viewIdx]

        cleanup_func = get(cleanup_callbacks, viewIdx, function f() end)
        cleanup_func()

        cleanup = build_mol_window(l, r, t, alignedPositions[t], volData, volAbsMax, volDataDict, sq, ls, kdTree1, sampleRangeX, sampleRangeY, sampleRangeZ, cmap, on_window_hover, lsExtrema, volFilter.interval)

        cleanup_callbacks[viewIdx] = cleanup

        if viewIdx < length(views)
            viewIdx += 1
        else
            viewIdx = 1
        end
    end

    # do a convex hull of values for each type of transitionInvariant ?
    # layered radar plots?
    # need to be entirely rotationally invariant
    @show typeof(transitionInvariants1), typeof(transitionInvariants2), typeof(transitionInvariants3)
    screen = GLMakie.Screen()

    available_matrices = Dict{String,Dict{Tuple{Int,Int},Matrix{Float64}}}()
    available_matrices["transforms"] = transforms
    available_matrices["bondDeltas"] = bondDeltas

    # atomPositions, stateKDTree, numAtoms, firstTransition 
    display(screen, build_selection_window((600, 800), available_matrices, transitionSequence, on_click, num_atoms, Observable(firstTransition), (minInvariant1, maxInvariant2, transitionInvariants1), alignedPositions, stateKDTree, dms))
end
end
