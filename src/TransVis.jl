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

function buildBonds(positions, bondDelta)
    points = Vector{Tuple{Point3f,Point3f}}()
    weights = Vector{Float64}()
    indices = Vector{Tuple{Int64,Int64}}()

    for i in 1:length(bondDelta[1, :])
        for j in 1:i
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

    # should move molScreen into a new file
    molScreen = GLMakie.Screen()
    molGrid = Figure()

    views = Vector()
    all_scenes = Vector()
    for i in 1:3
        lab = Label(molGrid[i, 1], "", rotation=pi / 2)
        l = LScene(
            molGrid[i, 2],
            show_axis=false,
            scenekw=(backgroundcolor=:black, clear=true),
        )
        r = LScene(
            molGrid[i, 3],
            show_axis=false,
            scenekw=(backgroundcolor=:black, clear=true),
        )
        push!(views, (lab, l, r))
        push!(all_scenes, l)
        push!(all_scenes, r)

        Camera3D(l.scene, center=false, eyeposition=Vec3f(30, 30, 30))
        Camera3D(r.scene, center=false, eyeposition=Vec3f(30, 30, 30))

        rowsize!(molGrid.layout, i, Relative(0.25))
    end
    colsize!(molGrid.layout, 1, Relative(0.05))
    colsize!(molGrid.layout, 2, Relative(0.475))
    colsize!(molGrid.layout, 3, Relative(0.475))

    link_cameras_lscenes(all_scenes)

    filterRange = lift(x -> LinRange(-x, x, 100), volAbsMax)

    Label(molGrid[4, 1], "Volume Controls", rotation=pi / 2)
    sg = SliderGrid(molGrid[4, 2:3],
        (label="Volume Resolution", range=0.1:0.1:1, startvalue=0.2),
        (label="Kernel Width", range=0.1:0.1:2.0, startvalue=1.0),
        (label="Num Neighbors", range=1:1:num_atoms, startvalue=5))

    volFilter = IntervalSlider(molGrid[5, 1:2], range=filterRange, startvalues=(0, 0))
    Label(molGrid[5, 3], lift(x -> "Volume filter: " * string(round.(x, digits=6)), volFilter.interval))

    Colorbar(molGrid[6, 1:3], colormap=cmap, limits=lift(x -> (-x, x), volAbsMax), vertical=false)

    rowsize!(molGrid.layout, 4, Relative(0.25 / 3))
    rowsize!(molGrid.layout, 5, Relative(0.25 / 3))
    rowsize!(molGrid.layout, 6, Relative(0.25 / 3))

    cleanup_callbacks = Dict()

    display(molScreen, molGrid)
    viewIdx = 1

    sampleRanges = lift(sg.sliders[1].value) do vr
        return [minX-2*vr:vr:maxX+2*vr;],
        [minY-2*vr:vr:maxY+2*vr;],
        [minZ-2*vr:vr:maxZ+2*vr;]
    end

    kernelWidth = lift(sg.sliders[2].value) do kw
        return kw
    end

    num_neighbors = lift(sg.sliders[3].value) do nn
        return nn
    end

    function on_click(t, on_window_hover)
        pos1, pos2 = alignedPositions[t]
        kdTree1, kdTree2 = stateKDTree[t]

        volumeData = @lift begin
            volData = zeros(length($sampleRanges[1]), length($sampleRanges[2]), length($sampleRanges[3]))
            volDataDict = Dict{Int,Any}()

            for i in eachindex($sampleRanges[1]) # x
                for j in eachindex($sampleRanges[2]) # y
                    for k in eachindex($sampleRanges[3]) # z
                        point = Point3f($sampleRanges[1][i], $sampleRanges[1][j], $sampleRanges[1][k])
                        knn, dists = NearestNeighbors.knn(kdTree1, point, $num_neighbors)
                        kValue = sum(kernelFunction.(Ref(point), pos1[knn], $kernelWidth) .* transitionInvariants1[t][knn])
                        volData[i, j, k] = kValue
                        idx, d = NearestNeighbors.nn(kdTree1, point)
                        volDataDict[idx] = kValue
                    end
                end
            end
            thisVolAbsMax = max(abs(minimum(volData)), abs(maximum(volData)))
            $volAbsMax = max($volAbsMax, thisVolAbsMax)

            return volData, volDataDict
        end
        glyphResolution = 0.1

        # 1.0 should be transitionGlyphSize
        sq = superquadric.(1.0, pos1, stretchedPrincipalAxes[t], transitionInvariants2[t], -1.0, 3.0, glyphResolution)[:]
        ls = buildBonds(alignedPositionsMatrices[t][1], bondDeltas[t])

        thislsExtrema = extrema(ls[2])
        lsExtrema[] = (min(lsExtrema[][1], thislsExtrema[1]), max(lsExtrema[][1], thislsExtrema[2]))

        lab, l, r = views[viewIdx]

        cleanup_func = get(cleanup_callbacks, viewIdx, function f() end)
        cleanup_func()

        cleanup = build_mol_window(l, r, t, alignedPositions[t], volumeData, volAbsMax, sq, ls, kdTree1, sampleRanges, cmap, on_window_hover, lsExtrema, volFilter.interval)

        lab.text = string(t)

        cleanup_callbacks[viewIdx] = cleanup

        if viewIdx < length(views)
            viewIdx += 1
        else
            viewIdx = 1
        end
    end

    @show typeof(transitionInvariants1), typeof(transitionInvariants2), typeof(transitionInvariants3)
    screen = GLMakie.Screen()

    available_matrices = Dict{String,Dict{Tuple{Int,Int},Matrix{Float64}}}()
    available_matrices["transforms"] = transforms
    available_matrices["bondDeltas"] = bondDeltas

    # atomPositions, stateKDTree, numAtoms, firstTransition 
    display(screen, build_selection_window((600, 800), available_matrices, transitionSequence, on_click, num_atoms, Observable(firstTransition), (minInvariant1, maxInvariant2, transitionInvariants1), alignedPositions, stateKDTree, dms))
end
end
