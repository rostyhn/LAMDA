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
using Mmap

include("io.jl")
include("processing.jl")
include("SelectionWindow.jl")
include("MolWindow.jl")
include("utils.jl")
include("math.jl")

export go

function go(trajectory_name::String)

    GLMakie.closeall() #close all windows for rerun!
    active_trajectory = get_data_alt(trajectory_name)

    transitionInvariants1 = active_trajectory["t1"]
    transitionInvariants2 = active_trajectory["t2"]
    transitionInvariants3 = active_trajectory["t3"]

    stretchedPrincipalAxes = active_trajectory["stretchedPrincipalAxes"]
    stateKDTree = active_trajectory["kdTree"]
    dms = active_trajectory["dms"]
    scalars = active_trajectory["scalars"]

    transitionSequence = active_trajectory["transitions"]

    t_to_idx = Dict()
    for (i, t) in enumerate(transitionSequence)
        t_to_idx[t] = i
    end

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


    # should be cached
    bondDeltas = Dict{Tuple{Int16,Int16},Matrix{Float32}}()
    bdMin = floatmax(Float32)
    bdMax = floatmin(Float32)
    for t in transitionSequence
        s1, s2 = t
        dm1 = distanceMatrices[s1]
        dm2 = distanceMatrices[s2]

        # for now it's total delta
        bd = dm2 - dm1
        vals = vec(bd)

        bdMin = min(bdMin, minimum(vals))
        bdMax = max(bdMax, maximum(vals))

        bondDeltas[t] = bd
    end

    lsExtrema = (bdMin, bdMax)
    ls_cmap = resample_cmap(:bwr, 100; alpha=([(-0.99):0.02:(0.99);] ./ 0.1) .^ 6)

    available_matrices = Dict()
    available_matrices["bondDeltas"] = bondDeltas

    molGrid = Figure()
    Label(molGrid[1, 1], "Volume Controls", rotation=pi / 2)
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

    # 0.1 is the thickness of the white part
    volume_cmap = Observable(resample_cmap(:bam, 100; alpha=([(-0.99):0.02:(0.99);] ./ 0.1) .^ 6))
    volRange = Observable((floatmin(Float32), floatmax(Float32)))

    selected_invariant = Observable("t1")
    @time volumeData = @lift begin
        key = string($(sg.sliders[1].value), "_", $kernelWidth, "_", $num_neighbors, "_", $selected_invariant, "_", trajectory_name)
        w = length($sampleRanges[1])
        h = length($sampleRanges[2])
        d = length($sampleRanges[3])

        fp, is_cached = get_mmap_file(key)
        if !is_cached
            println("Calculating volume data for $(key); will be saved as $(hash(key))...")
            # https://docs.julialang.org/en/v1/manual/multi-threading/
            points = Vector{Tuple{Tuple{Int,Int,Int},Point3f}}()
            for i in eachindex($sampleRanges[1]) # x
                for j in eachindex($sampleRanges[2]) # y
                    for k in eachindex($sampleRanges[3]) # z
                        point = Point3f($sampleRanges[1][i], $sampleRanges[2][j], $sampleRanges[3][k])
                        push!(points, ((i, j, k), point))
                    end
                end
            end

            # setting shared = false does not save the results
            volData = Mmap.mmap(fp, Matrix{Float32}, (length(transitionSequence), w * h * d))
            chunks = Iterators.partition(enumerate(transitionSequence), div(length(transitionSequence), max(Threads.nthreads() - 1, 1)))

            #write() should be faster, question is how to do it sequentially
            # might want to set BLAS.set_num_threads(1)
            d_ch = Channel{Tuple{Array{Tuple{Int,Array{Float32}}},Float32,Float32,Float32}}()
            map(chunks) do chunk
                Threads.@spawn begin
                    sample_range = (length($sampleRanges[1]), length($sampleRanges[2]), length($sampleRanges[3]))
                    # split into subchunks to save memory
                    subchunks = Iterators.partition(chunk, 25)
                    for sc in subchunks
                        # might want to copy over alignedPositions, stateKDTree etc for the selected values
                        vd, sc_volmin, sc_volmax, sc_absvolmin = calculateVolumes(sc, sample_range, alignedPositions, stateKDTree, points, $num_neighbors, $kernelWidth, active_trajectory[$selected_invariant])
                        put!(d_ch, (vd, sc_volmin, sc_volmax, sc_absvolmin))
                    end
                end
            end

            absVolMin = floatmax(Float32)
            volMin = floatmax(Float32)
            volMax = floatmin(Float32)

            processed = 0
            while processed < length(transitionSequence)
                # possibly sort by idx before writing with write instead of mmap?
                vd, c_volmin, c_volmax, c_absvolmin = take!(d_ch)
                for (idx, d) in vd
                    volData[idx, :] = d
                end
                processed += length(vd)
                volMin = min(c_volmin, volMin)
                volMax = max(c_volmax, volMax)
                absVolMin = min(absVolMin, c_absvolmin)
            end
            close(d_ch)

            volRange[] = (volMin, volMax)
            notify(volRange)
            save_volume_cache(key, (volMin, volMax), (w, h, d), absVolMin)
        else
            volData = Mmap.mmap(fp, Matrix{Float32}, (length(transitionSequence), w * h * d), shared=false, grow=false)
            volRange[] = read_volume_cache(key)
            notify(volRange)
        end

        if $selected_invariant == "t2"
            volume_cmap[] = resample_cmap(:matter, 100; alpha=([0:0.01:0.99;] ./ 0.1) .^ 2)
        else
            # should be fine, seems off-center because abs(volMin) != abs(volMax)
            # could additionally calculate volAbsMin to remove noisy values
            volume_cmap[] = resample_cmap(:bam, 100; alpha=([(-0.99):0.02:(0.99);] ./ 0.1) .^ 6)
        end
        notify(volume_cmap)
        return volData
    end

    filterRange = lift(x -> LinRange(x[1], x[2], 100), volRange)
    volFilter = IntervalSlider(molGrid[2, 1:2], range=filterRange, startvalues=(0, 0))
    Label(molGrid[2, :], lift(x -> "Volume filter: " * string(round.(x, digits=6)), volFilter.interval))

    function on_click(t, on_window_hover)
        pos1, pos2 = alignedPositions[t]
        kdTree1, kdTree2 = stateKDTree[t]

        # 1.0 should be transitionGlyphSize
        sq = superquadric.(1.0, pos1, stretchedPrincipalAxes[t], transitionInvariants2[t], 3.0, 0.1)[:]
        ls = buildBonds(alignedPositionsMatrices[t][1], bondDeltas[t], connectivity[t[1]])

        build_mol_window(t, alignedPositions[t], lift((y, z) -> reshape(y[t_to_idx[t], :], (length(z[1]), length(z[2]), length(z[3]))), volumeData, sampleRanges), volRange, sq, ls, kdTree1, sampleRanges, volume_cmap, on_window_hover, lsExtrema, volFilter.interval, ls_cmap, scalars)
    end

    screen = GLMakie.Screen()
    # atomPositions, stateKDTree, numAtoms, firstTransition 
    window = build_selection_window((600, 800), available_matrices, transitionSequence, t_to_idx, on_click, num_atoms, alignedPositions, stateKDTree, dms, volumeData, sampleRanges, volRange, volume_cmap, selected_invariant)

    display(screen, window)
end
end
