module TransVis

#Data handling
using GLMakie: apply_transform
using Makie: MakieCore, ray_at_cursor, position_on_plot, mouse_in_scene, shift_project, update_tooltip_alignment!, parent_scene, show_data
using Pickle
using JLD2
using CodecZlib
using Clustering: hclust, cutree
using FileIO
using ColorTypes
using FixedPointNumbers

#Vis
using GLMakie
using Makie
using GeometryBasics
# using GLFW

#Processing and Helpers
using NearestNeighbors
using LinearAlgebra
using Base.Threads
using Statistics
using ProgressMeter
using Ripserer
using Mmap

include("io.jl")
include("data_types.jl")
# unfortunately is used as part of the main module
include("multiprocess.jl")
include("processing.jl")
include("SelectionWindow.jl")
include("MolWindow.jl")
include("utils.jl")
include("math.jl")
include("dendrogram.jl")
include("SettingsWindow.jl")
include("ClusterWindow.jl")
include("ReductionWindow.jl")
include("ui.jl")

export go
const SINGLE_TRANSITION_RENDER_OPTIONS = ["Atom", "Volume", "Superquadric"]
const CLUSTER_COLORS = :tab20

function go(trajectory_name::String; kwargs...)
    GLMakie.closeall() #close all windows for rerun!
    active_trajectory = get_data_alt(trajectory_name)
    set_theme!(theme_latexfonts(); fontsize=18.0)

    screen_ref = Ref{Maybe{Screen}}(nothing)
    window = build_reduction_window(active_trajectory, main_window, screen_ref; kwargs...)
    # TODO: always set to first monitor so its consistent
    # Passing GLFW.Monitor doesn't work for some reason
    screen = GLMakie.Screen()
    screen_ref[] = screen
    display(screen, window)
end

function main_window(active_trajectory, screen_ref; chunk_size=100, init_h_cutoff=0.3, align_with=nothing)
    # GLMakie.closeall() # close reduction window 

    stretchedPrincipalAxes = active_trajectory["stretchedPrincipalAxes"]
    scalars = active_trajectory["scalars"]
    scalar_range = active_trajectory["scalar_range"]

    connectivity = active_trajectory["connectivity"]
    distanceMatrices = active_trajectory["distanceMatrices"]

    alignedPositionsMatrices = active_trajectory["alignedPositionsMatrices"] # positions as matrices
    alignedPositions = active_trajectory["alignedPositions"] # positions as points
    kdTrees = active_trajectory["kdTrees"]

    alignments = active_trajectory["alignments"]

    dm = active_trajectory["selected_dm"]

    transitionSequence = active_trajectory["reduced_transitions"]
    trajectory_name = active_trajectory["name"]

    per_t_scalar_ranges = active_trajectory["per_t_scalar_ranges"]
    per_t_scalars = active_trajectory["per_t_scalars"]

    # absolute index for volume data
    t_to_idx = Dict{Tuple{Int,Int},Int}(reverse.(collect(enumerate(active_trajectory["transitions"]))))

    per_t_scalars["t_to_idx"] = t_to_idx
    per_t_scalar_ranges["t_to_idx"] = (1, length(active_trajectory["transitions"]))

    h_cutoff = Observable(init_h_cutoff)
    h_range = Observable((floatmin(Float32), floatmax(Float32)))

    cluster_data = @lift begin
        clustering = hclust($dm, linkage=:ward, branchorder=:barjoseph)

        rm = zeros(size($dm))
        # gets the correct idx 
        idx_to_mtx = zeros(Int, size($dm)[1])
        t_to_mtx = Dict()
        mtx_to_t = Dict()
        for (i, r) in enumerate(clustering.order)
            rm[i, :] .= $dm[r, :][clustering.order]
            idx_to_mtx[r] = i
            t_to_mtx[transitionSequence[r]] = i
            mtx_to_t[i] = transitionSequence[r]
        end

        # get minimum and maximum of entire matrix for cmap
        fl = vec($dm)
        h_range[] = extrema(clustering.heights)
        notify(h_range)
        return ClusterData(clustering=clustering, matrix=rm, idx_to_mtx=idx_to_mtx, m_extrema=(extrema(fl)), t_to_mtx=t_to_mtx, mtx_to_t=mtx_to_t)
    end

    # vector of ints in transitionSequence order corresponding to the cluster each index is assigned
    cluster_info = @lift begin
        assignments = cutree($(cluster_data).clustering, h=$h_cutoff)
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

        #=
        pickled_groups = Dict{Int,Vector{Tuple{Int,Int}}}()
        for (clusterIdx, g) in groups
            ts = map(x -> transitionSequence[x], g)
            pickled_groups[clusterIdx] = ts
        end

        Pickle.store("clustering_$($h_cutoff).pickle", pickled_groups)
        =#

        reps = Dict{Int,Tuple{Int,Int}}()
        for (clusterIdx, g) in groups
            # find reference t
            m = $dm
            dist_sum = map(x -> sum(m[x, :][g]), g)
            ref_t_idx = g[argmin(dist_sum)]

            reps[clusterIdx] = transitionSequence[ref_t_idx]
        end

        lines, clusters = treepositions($(cluster_data).clustering, $h_cutoff)
        return ClusterInfo(groups=groups,
            representatives=reps,
            assignments=assignments,
            lines=lines,
            clusters=clusters,
            cutoff=$h_cutoff)
    end

    init_alignment = (!isnothing(align_with) && align_with in keys(alignments)) ? align_with : first(keys(alignments))
    # perfom intra-cluster alignment
    selected_alignment = Observable(init_alignment)

    alignment_rotations = @lift begin
        println("Calculating alignment with $($selected_alignment)")
        # figure out what transitions are grouped together
        features = alignments[$selected_alignment]

        rot = Dict{Tuple{Int16,Int16},Tuple{Array{Float32},Matrix{Float32},Bool,Tuple{Int,Int}}}()
        for (clusterIdx, g) in $(cluster_info).groups
            # find reference t
            m = $dm
            dist_sum = map(x -> sum(m[x, :][g]), g)
            ref_t_idx = argmin(dist_sum)

            ts = map(x -> transitionSequence[x], g)

            # for now, use first t as reference 
            ref_t = popat!(ts, ref_t_idx)

            ref_s1_pos = alignedPositionsMatrices[ref_t][1]
            ref_s1_com = reduce(vcat, map(x -> com(ref_s1_pos, x), eachcol(features[ref_t][1])))
            ref_s1_shift = mean(ref_s1_com, dims=1)

            ref_s1_com = reduce(vcat, map(x -> com(ref_s1_pos .- ref_s1_shift, x), eachcol(features[ref_t][1])))

            rot[ref_t] = (ref_s1_shift, Matrix(1.0I, 3, 3), false, ref_t)

            for t in ts
                t_s1_pos = alignedPositionsMatrices[t][1]
                t_s1_com = reduce(vcat, map(x -> com(t_s1_pos, x), eachcol(features[t][1])))
                t_s1_shift = mean(t_s1_com, dims=1)
                t_s1_com = reduce(vcat, map(x -> com(t_s1_pos .- t_s1_shift, x), eachcol(features[t][1])))

                t_s2_pos = alignedPositionsMatrices[t][2]
                t_s2_com = reduce(vcat, map(x -> com(t_s2_pos, x), eachcol(features[t][2])))
                t_s2_shift = mean(t_s2_com, dims=1)
                t_s2_com = reduce(vcat, map(x -> com(t_s2_pos .- t_s2_shift, x), eachcol(features[t][2])))

                R1, res1 = pure_align(ref_s1_com, t_s1_com)
                R2, res2 = pure_align(ref_s1_com, t_s2_com)

                R = (res1 < res2) ? R1 : R2
                shift = (res1 < res2) ? t_s1_shift : t_s2_shift
                rot[t] = (shift, R, res1 > res2, ref_t)
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
    absAvgBonds = Dict{Tuple{Int16,Int16},Array{Float32}}()
    bonds = Dict()
    bdMin = floatmax(Float32)
    bdMax = floatmin(Float32)
    println("Calculating bonds...")
    @showprogress for t in transitionSequence
        s1, s2 = t
        dm1 = distanceMatrices[s1]
        dm2 = distanceMatrices[s2]

        # for now it's total delta
        bd = dm2 - dm1
        vals = vec(bd)

        bdMin = min(bdMin, minimum(vals))
        bdMax = max(bdMax, maximum(vals))

        avgs = Vector{Float32}(undef, length(bd[:, 1]))
        for (i, r) in enumerate(eachrow(bd * connectivity[t[1]]))
            cartesians = length(findall(!iszero, r))
            avgs[i] = sum(abs.(r)) / cartesians
        end
        absAvgBonds[t] = avgs
        bondDeltas[t] = bd
        bonds[t] = calc_bonds(connectivity[t[1]])
    end

    scalars["absAvgBonds"] = absAvgBonds

    lsExtrema = (bdMin, bdMax)
    ls_cmap = resample_cmap(:bwr, 100; alpha=([(-0.99):0.02:(0.99);] ./ 0.1) .^ 6)

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

        # calculate volume data for all transitions just once
        abs_t_seq = active_trajectory["transitions"]

        fp, is_cached = get_mmap_file(key)
        if !is_cached
            println("Calculating volume data for $(key); will be saved as $(hash(key))...")

            alignedPos = map(x -> alignedPositions[x][1], abs_t_seq)
            kd = map(x -> kdTrees[x][1], abs_t_seq)
            invariants = map(x -> active_trajectory[$selected_invariant][x], abs_t_seq)

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
                prog = Progress(length(abs_t_seq))
                update!(prog, processed)

                chunks = collect(Iterators.partition(eachindex(abs_t_seq), chunk_size))

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

        volData = Mmap.mmap(fp, Array{Float32,2}, (w * h * d, length(abs_t_seq)), shared=false, grow=false)
        volRange[] = read_volume_cache(key)
        notify(volRange)

        if $selected_invariant == "t2"
            volume_cmap[] = resample_cmap(:matter, 100; alpha=([0:0.01:0.99;] ./ 0.1) .^ 2)
        else
            # should be fine, seems off-center because abs(volMin) != abs(volMax)
            lowmap = reverse(resample_cmap(:RdPu_3, 50; alpha=([(0.0):0.02:(0.99);] ./ 0.1) .^ 6))
            himap = resample_cmap(:greens, 50; alpha=([(0.0):0.02:(0.99);] ./ 0.1) .^ 6)
            volume_cmap[] = vcat(lowmap, himap)
        end
        notify(volume_cmap)

        return volData
    end

    filterRange = lift(x -> LinRange(x[1], x[2], 100), volRange)
    volFilter = IntervalSlider(molGrid[2, 1:2], range=filterRange, startvalues=(0, 0))
    Label(molGrid[2, :], lift(x -> "Volume filter: " * string(round.(x, digits=6)), volFilter.interval))

    invariantRange = @lift begin
        vals = values(active_trajectory[$selected_invariant])
        absInvMin = minimum(minimum.(vals))
        absInvMax = maximum(maximum.(vals))
        return (absInvMin, absInvMax)
    end

    function create_position_alignment_observer(transition)
        return @lift begin
            return apply_alignment($alignment_rotations[$transition], $alignedPositionsMatrices[$transition])
        end
    end

    # did this to avoid drilling down and passing parameters constantly
    atom_cmap = resample_cmap(:reds, 100, alpha=range(; start=0.01, stop=1.0, length=100))
    function render_atom_view(scene, transition, selected_scalar, time)
        t_ap = create_position_alignment_observer(transition)
        return simple_atom_view!(scene, t_ap, lift((x, y) -> scalars[x][y], selected_scalar, transition), scalar_range, atom_cmap, time)
    end

    function render_volume_view(scene, transition)
        vd = lift((x, y, z) ->
                reshape(x[:, t_to_idx[y]], (length(z[1]), length(z[2]), length(z[3]))), volumeData, transition, sampleRanges)
        return volume_view!(scene, vd, sampleRanges, volume_cmap, volRange, lift((x, y) -> x[y], alignment_rotations, transition))
    end

    function render_volume_view_no_obs(scene, transition)
        t_idx = lift(x -> t_to_idx[x], transition)
        vd = lift((x, y, z) ->
                reshape(x[:, y], (length(z[1]), length(z[2]), length(z[3]))), volumeData, t_idx, sampleRanges)

        return volume_view!(scene, vd, sampleRanges, volume_cmap, volRange, lift(x -> alignment_rotations[][x], transition))
    end

    # assume the list of ts doesn't change
    function render_static_movement_view(scene, ts, time)
        bondVals = reduce(vcat, map(x -> scalars["absAvgBonds"][x], ts))
        posValsTup = map(t -> apply_alignment(alignment_rotations[][t], alignedPositionsMatrices[t]), ts)

        inits = reduce(vcat, first.(posValsTup))
        fins = reduce(vcat, last.(posValsTup))

        return simple_atom_view!(scene, Observable((inits, fins)), Observable(bondVals), (0.5, 2.0), atom_cmap, time)
    end

    function render_movement_view(scene, clusters, time)
        # first attempt, this is really dependent on the quality of the alignment
        t_ap = @lift begin
            g = reduce(vcat, map(x -> $(cluster_info).groups[x], collect($clusters)))
            ts = map(x -> transitionSequence[x], g)
            bondVals = reduce(vcat, map(x -> scalars["absAvgBonds"][x], ts))
            posValsTup = map(t -> apply_alignment($alignment_rotations[t], alignedPositionsMatrices[t]), ts)

            inits = reduce(vcat, first.(posValsTup))
            fins = reduce(vcat, last.(posValsTup))

            return (inits, fins), bondVals
        end

        simple_atom_view!(scene, lift(x -> x[1], t_ap), lift(x -> x[2], t_ap), (0.5, 2.0), atom_cmap, time)

        return [], []
    end

    function render_superquadrics_view(scene, transition, inspector)
        t_ap = create_position_alignment_observer(transition)

        invariant = lift((x, y) -> active_trajectory[x][y], selected_invariant, transition)
        points = lift(x -> Point3f.(eachrow(x[1])), t_ap)
        spa = lift(x -> stretchedPrincipalAxes[x], transition)

        colors = lift((xx, y) -> map(x -> y[x], eachindex(xx)), points, invariant)
        sq = Observable(superquadric.(1.0, points[], spa[], 3.0, 0.1)[:]
        )
        calc_sq = on(spa, update=true, weak=true) do s
            sq[] = superquadric.(1.0, points[], s, 3.0, 0.1)[:]
        end

        il, is = superquadrics_view!(scene, points, sq, colors, volume_cmap, invariantRange, inspector)

        push!(il, calc_sq)
        return il, is
    end

    function time_slider(init_time, figure)
        time = Observable(init_time)

        t_slider = Slider(figure, range=0.0:0.05:1.0, startvalue=init_time)
        on(t_slider.value) do x
            time[] = x
        end

        sg = hgrid!(Label(figure, "t", font=:italic), t_slider, Label(figure, lift(x -> string(x), time)))

        return time, sg
    end

    # could be one func
    function render_menu(figure; default="Volume")
        scene_selector = Observable(default)
        render_menu = Menu(figure,
            options=SINGLE_TRANSITION_RENDER_OPTIONS,
            default=scene_selector[], tellwidth=false)

        on(render_menu.selection) do s
            scene_selector[] = s
            notify(scene_selector)
        end

        return scene_selector, render_menu
    end

    function scalar_menu(figure)
        opts = sort(collect(keys(scalars)))
        scalar_selection = Observable(first(opts))

        m = Menu(figure, options=opts, default=scalar_selection[])
        on(m.selection) do ms
            scalar_selection[] = ms
        end

        return scalar_selection, m
    end

    function atom_widgets(init_time, figure, grid)
        gg = GridLayout(grid[end+1, :])

        opts = sort(collect(keys(scalars)))
        time, t_slider = time_slider(init_time, figure)

        gg[1, 1:2] = t_slider

        scalar_selection, m = scalar_menu(figure)
        gg[2, 1] = m
        Colorbar(gg[2, 2], colorrange=scalar_range, vertical=false, colormap=atom_cmap, tellwidth=false)

        return gg, time, scalar_selection
    end

    # just pass this dictionary around and pass in the arguments it needs
    render_views = Dict()
    render_views["Atom"] = render_atom_view
    render_views["Volume"] = render_volume_view
    render_views["Volume_no_obs"] = render_volume_view_no_obs
    render_views["Superquadric"] = render_superquadrics_view
    render_views["Movement"] = render_movement_view
    render_views["SMovement"] = render_static_movement_view

    widgets = Dict()
    widgets["Atom"] = atom_widgets
    widgets["Movement"] = time_slider
    widgets["Render"] = render_menu
    widgets["Scalar"] = scalar_menu

    function on_click(t, on_window_hover)
        t_idx = t_to_idx[t]
        build_mol_window(t, t_idx, render_views, widgets)
    end

    settings_window = build_settings_menu(selected_invariant, selected_alignment, collect(keys(alignments)))

    rel_t_to_idx = Dict(reverse.(collect(enumerate(transitionSequence))))

    # atomPositions, stateKDTree, numAtoms, firstTransition 
    window = build_selection_window((600, 800), transitionSequence, rel_t_to_idx, on_click, num_atoms, dm, volRange, volume_cmap, cluster_data, cluster_info, scalars, h_cutoff, h_range, settings_window, render_views, widgets, invariantRange, active_trajectory["selected_dm_name"], active_trajectory["per_t_scalars"], active_trajectory["per_t_scalar_ranges"])


    #= 
    # creating screen after the window is built prevents subtle bugs
    # such as interactions being trigged before the window is rendered 
    =#
    screen = GLMakie.Screen()
    display(screen, window)

    # create inspector after render to avoid bugs
    DataInspector(window)

    # close reduction window
    close(screen_ref[])
end
end
