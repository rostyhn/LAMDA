module TransVis

#Data handling
using GLMakie: apply_transform
using Makie: MakieCore, ray_at_cursor, position_on_plot, mouse_in_scene, shift_project, update_tooltip_alignment!, parent_scene, show_data
using Pickle
using JLD2
using CodecZlib
using Clustering: hclust, cutree
#Vis
using GLMakie
using Makie
using GeometryBasics

#Processing and Helpers
using NearestNeighbors
using LinearAlgebra
using Base.Threads
using Statistics
using ProgressMeter
using Ripserer
using Mmap


include("io.jl")
# unfortunately is used as part of the main module
include("multiprocess.jl")
include("processing.jl")
include("SelectionWindow.jl")
include("MolWindow.jl")
include("utils.jl")
include("math.jl")
include("dendrogram.jl")
include("SettingsWindow.jl")

export go

function go(trajectory_name::String; chunk_size=100, init_h_cutoff=0.3)
    GLMakie.closeall() #close all windows for rerun!
    active_trajectory = get_data_alt(trajectory_name)
    transitionInvariants1 = active_trajectory["t1"]
    transitionInvariants2 = active_trajectory["t2"]
    transitionInvariants3 = active_trajectory["t3"]

    stretchedPrincipalAxes = active_trajectory["stretchedPrincipalAxes"]
    dms = active_trajectory["dms"]
    scalars = active_trajectory["scalars"]
    scalar_range = active_trajectory["scalar_range"]
    connectivity = active_trajectory["connectivity"]
    distanceMatrices = active_trajectory["distanceMatrices"]
    alignedPositionsMatrices = active_trajectory["alignedPositionsMatrices"] # positions as matrices
    alignedPositions = active_trajectory["alignedPositions"] # positions as points
    kdTrees = active_trajectory["kdTrees"]
    alignments = active_trajectory["alignments"]

    transitionSequence = active_trajectory["transitions"]

    t_to_idx = Dict()
    for (i, t) in enumerate(transitionSequence)
        t_to_idx[t] = i
    end

    h_cutoff = Observable(init_h_cutoff)
    h_range = Observable((floatmin(Float32), floatmax(Float32)))
    selected_dm = Observable(first(keys(dms)))
    clustering = @lift begin
        println("Clustering $($selected_dm)...")
        m = dms[$selected_dm]
        res = hclust(m, linkage=:ward, branchorder=:barjoseph)

        h_range[] = extrema(res.heights)
        notify(h_range)
        return res
    end

    cluster_assignments = @lift begin
        return cutree($clustering, h=$h_cutoff)
    end

    cluster_groups = @lift begin
        assignments = $cluster_assignments
        groups = Dict{Int,Vector{Int}}()
        for (i, c) in enumerate(assignments)
            if c in keys(groups)
                g = groups[c]
            else
                g = Vector{Int}()
            end
            push!(g, i)
            groups[c] = g
        end
        return groups
    end

    # perfom intra-cluster alignment
    selected_alignment = Observable(first(keys(alignments)))

    alignment_rotations = @lift begin
        # figure out what transitions are grouped together
        features = alignments[$selected_alignment]

        rot = Dict{Tuple{Int16,Int16},Matrix{Float32}}()
        for (clusterIdx, g) in $cluster_groups

            # find reference t
            m = dms[$selected_dm]
            dist_sum = map(x -> sum(m[x, :][g]), g)
            ref_t_idx = argmin(dist_sum)

            ts = map(x -> transitionSequence[x], g)

            # for now, use first t as reference 
            ref_t = popat!(ts, ref_t_idx)
            rot[ref_t] = Matrix(I, 3, 3)

            ref_s1_pos = alignedPositionsMatrices[ref_t][1]
            ref_s2_pos = alignedPositionsMatrices[ref_t][2]
            ref_s1_com = reduce(vcat, map(x -> com(ref_s1_pos, x), eachcol(features[ref_t])))
            ref_s2_com = reduce(vcat, map(x -> com(ref_s2_pos, x), eachcol(features[ref_t])))
            # align each t to ref_t 
            for t in ts
                t_s1_pos = alignedPositionsMatrices[t][1]
                t_s1_com = reduce(vcat, map(x -> com(t_s1_pos, x), eachcol(features[t])))

                R1, res1 = pure_align(ref_s1_com, t_s1_com)
                R2, res2 = pure_align(ref_s2_com, t_s1_com)

                # convert to homogeneous matrix 
                R = (res1 < res2) ? R1 : R2

                rot[t] = R
            end
        end

        return rot
    end

    # get number of atoms
    num_atoms = size(Iterators.first(values(alignedPositionsMatrices))[1])[1]

    # get min max coordinates of atoms for bounding box
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

            alignedPos = map(x -> alignedPositions[x][1], transitionSequence)
            kd = map(x -> kdTrees[x][1], transitionSequence)
            invariants = map(x -> active_trajectory[$selected_invariant][x], transitionSequence)

            points = Vector{Tuple{Tuple{Int,Int,Int},Point3f}}()
            for i in eachindex($sampleRanges[1]) # x
                for j in eachindex($sampleRanges[2]) # y
                    for k in eachindex($sampleRanges[3]) # z
                        point = Point3f($sampleRanges[1][i], $sampleRanges[2][j], $sampleRanges[3][k])
                        push!(points, ((i, j, k), point))
                    end
                end
            end

            try
                absVolMin = floatmax(Float32)
                volMin = floatmax(Float32)
                volMax = floatmin(Float32)

                processed = 0
                prog = Progress(length(transitionSequence))
                update!(prog, processed)

                chunks = collect(Iterators.partition(eachindex(transitionSequence), chunk_size))

                # 500 seconds at the fastest
                io = open(fp, "a")
                for chunk in chunks
                    sub_chunks = collect(Iterators.partition(chunk, div(length(chunk), nthreads(:default))))
                    tasks = map(sub_chunks) do ts
                        Threads.@spawn :default begin
                            ap_chunk = @view alignedPos[ts]
                            kd_chunk = @view kd[ts]
                            iv_chunk = @view invariants[ts]
                            return calc_vols($sampleRanges, $num_neighbors, $kernelWidth, points, ts, kd_chunk, ap_chunk, iv_chunk)
                        end
                    end

                    errormonitor.(tasks)
                    data = fetch.(tasks)
                    vd = reduce(vcat, first.(data))

                    for d in vd
                        write(io, d)
                    end
                    processed += length(vd)

                    volMin = min(volMin, minimum(getindex.(data, 2)))
                    volMax = max(volMax, maximum(getindex.(data, 3)))
                    absVolMin = min(absVolMin, minimum(last.(data)))

                    update!(prog, processed)
                end
                close(io)
                save_volume_cache(key, (volMin, volMax), (w, h, d), absVolMin)
            catch
                rm(fp)
                return error("Volume calculation failed.")
            end
        end

        volData = Mmap.mmap(fp, Array{Float32,2}, (w * h * d, length(transitionSequence)), shared=false, grow=false)
        volRange[] = read_volume_cache(key)
        notify(volRange)

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
        pos1 = alignedPositions[t][1]
        kdTree1 = kdTrees[t][1]

        # 1.0 should be transitionGlyphSize
        sq = superquadric.(1.0, pos1, stretchedPrincipalAxes[t], transitionInvariants2[t], 3.0, 0.1)[:]
        ls = buildBonds(alignedPositionsMatrices[t][1], bondDeltas[t], connectivity[t[1]])

        build_mol_window(t, alignedPositions[t], lift((y, z) -> reshape(y[:, t_to_idx[t]], (length(z[1]), length(z[2]), length(z[3]))), volumeData, sampleRanges), volRange, sq, ls, kdTree1, sampleRanges, volume_cmap, on_window_hover, lsExtrema, volFilter.interval, ls_cmap, scalars)
    end

    screen = GLMakie.Screen()

    settings_window = build_settings_menu(selected_invariant, selected_alignment, collect(keys(alignments)))

    # atomPositions, stateKDTree, numAtoms, firstTransition 
    window = build_selection_window((600, 800), available_matrices, transitionSequence, t_to_idx, on_click, num_atoms, alignedPositionsMatrices, kdTrees, dms, volumeData, sampleRanges, volRange, volume_cmap, clustering, selected_dm, scalars, h_cutoff, cluster_groups, alignment_rotations, h_range, scalar_range, settings_window, cluster_assignments)

    display(screen, window)
end
end
