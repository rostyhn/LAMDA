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

function go(trajectory_name::String)

    GLMakie.closeall() #close all windows for rerun!
    active_trajectory = get_data_alt(trajectory_name)

    transitionInvariants1 = active_trajectory["t1"]
    transitionInvariants2 = active_trajectory["t2"]
    transitionInvariants3 = active_trajectory["t3"]
    stretchedPrincipalAxes = active_trajectory["stretchedPrincipalAxes"]
    stateKDTree = active_trajectory["kdTree"]
    dms = active_trajectory["dms"]

    transitionSequence = active_trajectory["transitions"]

    connectivity = active_trajectory["connectivity"]
    distanceMatrices = active_trajectory["distanceMatrices"]
    alignedPositionsMatrices = active_trajectory["alignedPositionsMatrices"] # positions as matrices
    alignedPositions = active_trajectory["alignedPositions"] # positions as vec point3fs

    # get number of atoms
    num_atoms = size(Iterators.first(values(alignedPositionsMatrices))[1])[1]

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

    # 6 is the slope - should only be even odds
    # 0.1 is the thickness of the white part
    cmap = Observable(resample_cmap(:bam, 100; alpha=([(-0.99):0.02:(0.99);] ./ 0.1) .^ 6))

    bondDeltas = Dict{Tuple{Int,Int},Matrix{Float64}}()
    transforms = Dict{Tuple{Int,Int},Matrix{Float64}}()
    for t in transitionSequence
        s1, s2 = t
        dm1 = distanceMatrices[s1] .* connectivity[s1]'
        dm2 = distanceMatrices[s2] .* connectivity[s2]'

        # for now it's total delta
        bondDeltas[t] = dm2 - dm1

        p1, p2 = get_from_t_dict(alignedPositionsMatrices, t)
        transforms[t] = (abs.(p2 - p1))
    end


    available_matrices = Dict()
    available_matrices["transforms"] = normalize_matrices(transforms)
    available_matrices["bondDeltas"] = normalize_matrices(bondDeltas)

    volRange = Observable((floatmin(Float32), floatmax(Float32)))
    lsExtrema = Observable((-0.01, 0.01))

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

        Camera3D(l.scene, center=true, eyeposition=Vec3f(30, 30, 30))
        Camera3D(r.scene, center=true, eyeposition=Vec3f(30, 30, 30))

        rowsize!(molGrid.layout, i, Relative(0.25))
    end
    colsize!(molGrid.layout, 1, Relative(0.05))
    colsize!(molGrid.layout, 2, Relative(0.475))
    colsize!(molGrid.layout, 3, Relative(0.475))

    Label(molGrid[4, 1], "Volume Controls", rotation=pi / 2)
    sg = SliderGrid(molGrid[4, 2:3],
        (label="Volume Resolution", range=0.1:0.1:1, startvalue=0.2),
        (label="Kernel Width", range=0.1:0.1:2.0, startvalue=1.0),
        (label="Num Neighbors", range=1:1:num_atoms, startvalue=5))

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

    selected_invariant = Observable("t1")
    @time volumeData = @lift begin
        key = string($(sg.sliders[1].value), "_", $kernelWidth, "_", $num_neighbors, "_", $selected_invariant, "_", trajectory_name)
        cached_data = read_volume_cache(key)

        if cached_data == Nothing
            volData = Dict{Tuple{Int16,Int16},Array{Float32,3}}()
            volMax = Threads.Atomic{Float32}(floatmin(Float32))
            volMin = Threads.Atomic{Float32}(floatmax(Float32))
            invar = active_trajectory[$selected_invariant]

            # https://docs.julialang.org/en/v1/manual/multi-threading/\
            points = Vector{Tuple{Tuple{Int,Int,Int},Point3f}}()
            for i in eachindex($sampleRanges[1]) # x
                for j in eachindex($sampleRanges[2]) # y
                    for k in eachindex($sampleRanges[3]) # z
                        point = Point3f($sampleRanges[1][i], $sampleRanges[2][j], $sampleRanges[3][k])
                        push!(points, ((i, j, k), point))
                    end
                end
            end

            for t in transitionSequence
                vd = Array{Float32,3}(zeros(length($sampleRanges[1]), length($sampleRanges[2]), length($sampleRanges[3])))
                pos1, pos2 = get_from_t_dict(alignedPositions, t)
                kdTree1, kdTree2 = get_from_t_dict(stateKDTree, t)
                for ((i, j, k), point) in points
                    knn, dists = NearestNeighbors.knn(kdTree1, point, $num_neighbors)
                    kValue = sum(kernelFunction.(Ref(point), pos1[knn], $kernelWidth) .* invar[t][knn])
                    vd[i, j, k] = kValue
                    Threads.atomic_max!(volMax, kValue)
                    Threads.atomic_min!(volMin, kValue)
                end
                volData[t] = vd
            end
            volRange[] = (volMin[], volMax[])
            notify(volRange)

            save_volume_cache(key, volData, (volMin[], volMax[]))
        else
            volData = Dict{Tuple{Int16,Int16},Array{Float32,3}}(cached_data[1])
            volRange[] = Tuple{Float32,Float32}(cached_data[2])
            notify(volRange)
        end
        if $selected_invariant == "t2"
            cmap[] = resample_cmap(Reverse(:matter), 10; alpha=[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        else
            #TODO: fix cmap for t1 and t3
            cmap[] = resample_cmap(:bam, 100; alpha=([(-0.99):0.02:(0.99);] ./ 0.1) .^ 6)
        end
        notify(cmap)
        return volData
    end

    filterRange = lift(x -> LinRange(x[1], x[2], 100), volRange)
    volFilter = IntervalSlider(molGrid[5, 1:2], range=filterRange, startvalues=(0, 0))
    Label(molGrid[5, 3], lift(x -> "Volume filter: " * string(round.(x, digits=6)), volFilter.interval))

    Label(molGrid[6, 1], "Volume")
    Colorbar(molGrid[6, 2:3], colormap=cmap, limits=volRange, vertical=false)
    Label(molGrid[7, 1], "Bond Delta")
    Colorbar(molGrid[7, 2:3], colormap=:bwr, limits=lift(x -> x, lsExtrema), vertical=false)
    rowsize!(molGrid.layout, 4, Relative(0.25 / 3))
    rowsize!(molGrid.layout, 5, Relative(0.25 / 3))
    rowsize!(molGrid.layout, 6, Relative(0.25 / 6))
    rowsize!(molGrid.layout, 7, Relative(0.25 / 6))
    cleanup_callbacks = Dict()

    display(molScreen, molGrid)
    viewIdx = 1

    function on_click(t, on_window_hover)
        pos1, pos2 = get_from_t_dict(alignedPositions, t)
        kdTree1, kdTree2 = get_from_t_dict(stateKDTree, t)
        glyphResolution = 0.1

        # 1.0 should be transitionGlyphSize
        sq = superquadric.(1.0, pos1, stretchedPrincipalAxes[t], transitionInvariants2[t], -1.0, 3.0, glyphResolution)[:]
        ls = buildBonds(alignedPositionsMatrices[t][1], bondDeltas[t])

        thislsExtrema = extrema(ls[2])
        lsExtrema[] = (min(lsExtrema[][1], thislsExtrema[1]), max(lsExtrema[][1], thislsExtrema[2]))
        notify(lsExtrema)

        lab, l, r = views[viewIdx]

        cleanup_func = get(cleanup_callbacks, viewIdx, function f() end)
        cleanup_func()

        matrices = Dict()
        for (k, v) in available_matrices
            matrices[k] = (v[1][t], v[2], v[3])
        end

        cleanup = build_mol_window(l, r, t, alignedPositions[t], lift(x -> x[t], volumeData), volRange, sq, ls, kdTree1, sampleRanges, cmap, on_window_hover, lsExtrema, volFilter.interval, matrices)
        lab.text = string(t)

        cleanup_callbacks[viewIdx] = cleanup

        if viewIdx < length(views)
            viewIdx += 1
        else
            viewIdx = 1
        end
    end

    screen = GLMakie.Screen()
    # atomPositions, stateKDTree, numAtoms, firstTransition 
    window = build_selection_window((600, 800), available_matrices, transitionSequence, on_click, num_atoms, alignedPositions, stateKDTree, dms, volumeData, sampleRanges, volRange, cmap, selected_invariant)

    display(screen, window)
end
end
