module TransVis

using PyCall
using Base.Threads
using ImageIO
using Pickle
using JLD2
using CodecZlib
using FileIO
using ColorTypes

using ProgressMeter
using LinearAlgebra
using Statistics
using StatsBase
using SparseArrays
using Distances
using GeometryBasics
using Mmap
using FixedPointNumbers

using MathTeXEngine
using NetworkLayout

using GLFW
using Makie: MakieCore, ray_at_cursor, position_on_plot, mouse_in_scene, shift_project, update_tooltip_alignment!, parent_scene, show_data, clear_temporary_plots!, Orthographic, apply_transform_and_model, Makie
using CairoMakie # for saving plots w/ SVG
using GLMakie: Screen, apply_transform, ScreenConfig
using GLMakie
using Observables

using Clustering: Clustering, hclust, cutree, kmedoids
using NearestNeighbors
using UMAP

include("constants.jl")
include("data_types.jl")
include("io.jl")
include("multiprocess.jl")
include("processing.jl")
include("utils.jl")
include("math.jl")
include("ui.jl")

include("vis/Dendrogram.jl")
include("vis/EmbeddingView.jl")
include("vis/Scratchpad.jl")

include("windows/ReductionWindow.jl")
include("windows/SelectionWindow.jl")
include("windows/ClusterWindow.jl")
include("windows/NoteWindow.jl")
include("windows/SettingsWindow.jl")

function __init__()
    py"""
    import pickle

    def load_pickle(fpath):
        with open(fpath, "rb") as f:
            data = pickle.load(f)
        return data
    """o

    @pyinclude(joinpath(dirname(@__FILE__), "ase_processing.py"))
    GLMakie.activate!()
end

export go
function go(trajectory_name::String; kwargs...)
    clear_vars()
    GLMakie.closeall() #close all windows for rerun!

    active_trajectory = get_data_alt(trajectory_name)
    set_theme!(UI_THEME)

    screen_ref = Ref{Maybe{Screen}}(nothing)
    @time window = build_reduction_window(active_trajectory, main_window, screen_ref; kwargs...)
    screen = GLMakie.Screen(title="LAMDA - Reduction Window")
    screen_ref[] = screen

    display(screen, window)
end

function clear_vars()
    for x in Base.@locals
        x = nothing
    end
    GC.gc(true)
end

function main_window(active_trajectory::Trajectory,
    screen_ref,
    dm,
    transitionSequence,
    selected_dm_name,
    ;
    chunk_size=100,
    init_h_cutoff::Float64=0.3,
    align_with=nothing)

    (;
        stretchedPrincipalAxes,
        scalars,
        scalar_ranges,
        alignedPositionsMatrices,
        kdTrees,
        alignments,
        name,
        t_to_idx,
        transitions
    ) = active_trajectory

    # absolute index for volume data
    rel_t_to_idx = Dict(reverse.(collect(enumerate(transitionSequence))))

    h_cutoff::Observable{Float64} = Observable(float(init_h_cutoff))
    h_range = Observable((floatmin(Float32), floatmax(Float32)))

    cluster_data = @lift begin
        clustering = hclust($dm, linkage=:ward, branchorder=:barjoseph)
        rm = view($dm, clustering.order, clustering.order)
        # gets the correct idx 
        t_to_mtx = Dict{Transition,Int}()
        mtx_to_t = Dict{Int,Transition}()
        for (i, r) in enumerate(clustering.order)
            t_to_mtx[transitionSequence[r]] = i
            mtx_to_t[i] = transitionSequence[r]
        end

        # get minimum and maximum of entire matrix for cmap
        fl = vec($dm)
        h_range[] = extrema(clustering.heights)
        notify(h_range)

        c2idx, c_to_parent, parent_to_c = get_hierarchy(clustering)
        # render dendrogram once so that we can use any piece of it in the cluster window
        lines, clusters, c2lx = treepositions(clustering, 0.0)

        return ClusterData(clustering=clustering,
            matrix=rm,
            c2idx=c2idx,
            clusters=clusters,
            lines=lines,
            c2lx=c2lx,
            c_to_parent=c_to_parent,
            parent_to_c=parent_to_c,
            m_extrema=(extrema(fl)),
            t_to_mtx=t_to_mtx,
            mtx_to_t=mtx_to_t)
    end

    # vector of ints in transitionSequence order corresponding to the cluster each index is assigned
    cluster_info = @lift begin
        assignments = cutree($(cluster_data).clustering, h=$h_cutoff)
        groups = Dict{Int,Vector{Transition}}()
        igroups = Dict{Int,Vector{Int}}()
        # current assigned cluster to idx
        ccidx2cidx = Dict{Int,Int}()
        # cluster to assignment idx 
        a2c = Dict{Int,Set{Int}}()
        for (i, c) in enumerate(assignments)
            if c in keys(groups)
                g = groups[c]
                ig = igroups[c]
            else
                g = Vector{Transition}()
                ig = Vector{Int}()
            end
            push!(g, transitionSequence[i])
            push!(ig, i)
            groups[c] = g
            igroups[c] = ig
        end

        for (idx, g) in igroups
            c = Set(g)
            a2c[idx] = c
            ccidx2cidx[idx] = ($cluster_data).c2idx[c]
        end

        #=
        pickled_groups = Dict{Int,Vector{Transition}}()
        for (clusterIdx, g) in groups
            ts = map(x -> transitionSequence[x], g)
            pickled_groups[clusterIdx] = ts
        end

        Pickle.store("clustering_$($h_cutoff).pickle", pickled_groups)
        =#

        reps = Dict{Int,Transition}()
        for (clusterIdx, g) in groups
            # find reference t
            m = $dm
            gi = map(x -> $(cluster_data).t_to_mtx[x], g)
            dist_sum = map(x -> sum(view(m, x, gi)), gi)
            reps[clusterIdx] = g[argmin(dist_sum)]
        end

        # instead of rendering the dendrogram twice like this and keeping two copies in memory, could modify dendrogram render to show lines under the cutoff differently
        lines, clusters, c2lx = treepositions($(cluster_data).clustering, $h_cutoff)
        return ClusterInfo(groups=groups,
            representatives=reps,
            assignments=assignments,
            lines=lines,
            rel_t_to_idx=rel_t_to_idx,
            clusters=clusters,
            c2lx=c2lx,
            a2c=a2c,
            cc2cidx=ccidx2cidx,
            h_range=$h_range,
            cutoff=$h_cutoff)
    end

    init_alignment = (!isnothing(align_with) && align_with in keys(alignments)) ? align_with : first(keys(alignments))
    selected_alignment = Observable(init_alignment)

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

        minX1, maxX1 = extrema(view(p1, :, 1))
        minY1, maxY1 = extrema(view(p1, :, 2))
        minZ1, maxZ1 = extrema(view(p1, :, 3))

        minX2, maxX2 = extrema(view(p2, :, 1))
        minY2, maxY2 = extrema(view(p2, :, 2))
        minZ2, maxZ2 = extrema(view(p2, :, 3))

        minX = min(minX, min(minX1, minX2))
        minY = min(minY, min(minY1, minY2))
        minZ = min(minZ, min(minZ1, minZ2))

        maxX = max(maxX, max(maxX1, maxX2))
        maxY = max(maxY, max(maxY1, maxY2))
        maxZ = max(maxZ, max(maxZ1, maxZ2))
    end

    # should be cached
    #=bondDeltas = Dict{Transition,Matrix{Float32}}()
    absAvgBonds = Dict{Transition,Array{Float32}}()
    bonds = Dict()
    bdMin = floatmax(Float32)
    bdMax = floatmin(Float32)
    avgMin = floatmax(Float32)
    avgMax = floatmin(Float32)

    println("Calculating bonds...")
    @showprogress for t in transitionSequence
        s1, s2 = t
        println("access matrix")
        @time dm1 = distanceMatrices[s1]
        @time dm2 = distanceMatrices[s2]

        # for now it's total delta
        println("sub")
        @time bd = dm2 - dm1
        @time vals = vec(bd)

        println("min max")
        @time bdMin = min(bdMin, minimum(vals))
        @time bdMax = max(bdMax, maximum(vals))

        println("avg")
        avgs = Vector{Float32}(undef, length(bd[:, 1]))
        @time for (i, r) in enumerate(eachrow(bd * connectivity[t[1]]))
            cartesians = length(findall(!iszero, r))
            avgs[i] = sum(abs.(r)) / cartesians
        end
        avgMin = min(avgMin, minimum(avgs))
        avgMax = max(avgMax, maximum(avgs))

        absAvgBonds[t] = avgs
        bonds[t] = calc_bonds(connectivity[t[1]])
    end=#

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
    volumeData::Observable{Matrix{Float32}} = @lift begin
        key = string($(sg.sliders[1].value), "_", $kernelWidth, "_", $num_neighbors, "_", $selected_invariant, "_", name)
        w = length($sampleRanges[1])
        h = length($sampleRanges[2])
        d = length($sampleRanges[3])

        # calculate volume data for all transitions just once
        fp, is_cached = get_mmap_file(key)
        if !is_cached
            println("Calculating volume data for $(key); will be saved as $(hash(key))...")

            alignedPos = map(x -> alignedPositionsMatrices[x][1], transitions)
            kd = map(x -> kdTrees[x][1], transitions)

            iv = select_invariant(active_trajectory, $selected_invariant)
            invariants = map(x -> iv[x], transitions)

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
                prog = Progress(length(transitions))
                update!(prog, processed)

                chunks = collect(Iterators.partition(eachindex(transitions), chunk_size))

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

        volData = Mmap.mmap(fp, Array{Float32,2}, (w * h * d, length(transitions)), shared=false, grow=false)
        volRange[] = read_volume_cache(key)
        #make it symmetric 
        #maximumRange = max(abs(volRange[][1]), abs(volRange[][2]))
        #volRange[] = (-maximumRange, maximumRange)
        notify(volRange)

        if $selected_invariant == "t2"
            volume_cmap[] = resample_cmap(:matter, 100; alpha=([0:0.01:0.99;] ./ 0.05) .^ 2)
        else
            # should be fine, seems off-center because abs(volMin) != abs(volMax)
            lowmap = reverse(resample_cmap(:RdPu_3, 50; alpha=([(0.0):0.02:(0.99);] ./ 0.05) .^ 6))
            himap = resample_cmap(:greens, 50; alpha=([(0.0):0.02:(0.99);] ./ 0.05) .^ 6)
            volume_cmap[] = vcat(lowmap, himap)
        end
        notify(volume_cmap)

        return volData
    end

    filterRange = lift(x -> LinRange(x[1], x[2], 100), volRange)
    volFilter = IntervalSlider(molGrid[2, 1:2], range=filterRange, startvalues=(0, 0))
    Label(molGrid[2, :], lift(x -> "Volume filter: " * string(round.(x, digits=6)), volFilter.interval))

    invariantRange = @lift begin
        iv = select_invariant(active_trajectory, $selected_invariant)
        vals = values(iv)
        absInvMin = minimum(minimum.(vals))
        absInvMax = maximum(maximum.(vals))
        return (absInvMin, absInvMax)
    end

    # https://docs.julialang.org/en/v1.12-dev/manual/performance-tips/#man-performance-captured
    # convenience function to avoid passing around all the data
    function calc_alignment(ts)
        if !isempty(ts)
            ts_idx = map(x -> rel_t_to_idx[x], ts)
            features = alignments[selected_alignment[]]

            dist_sum = map(x -> sum(view(dm[], x, ts_idx)), ts_idx)
            ref_t_idx = argmin(dist_sum)

            ref_t = ts[ref_t_idx]
            return calculate_alignment(ref_t, ts, alignedPositionsMatrices, features)
        else
            return Dict()
        end
    end


    function create_position_alignment_observer(transition, alignment)
        return @lift begin
            return apply_alignment($alignment[$transition], $alignedPositionsMatrices[$transition])
        end
    end

    # did this to avoid drilling down and passing parameters constantly
    atom_cmap = resample_cmap(:linear_wcmr_100_45_c42_n256, 100, alpha=range(; start=0.01, stop=1.0, length=100))
    function render_atom_view(scene, transition, selected_scalar, time, alignment)
        t_ap = create_position_alignment_observer(transition, alignment)
        res = let scalars = scalars, scalar_ranges = scalar_ranges, atom_cmap = atom_cmap
            simple_atom_view!(scene, t_ap,
                lift((x, y) -> scalars[x][y], selected_scalar, transition),
                lift(x -> scalar_ranges[x], selected_scalar),
                atom_cmap,
                time)
        end
        return res
    end

    function render_volume_view(scene::Makie.Scene, transition::Observable{Transition})
        # directly indexing the mmap creates a copy, need to use a view
        vvd = let volumeData = volumeData
            view(volumeData[], :, t_to_idx[transition[]])
        end
        vd = reshape(vvd,
            (
                length(sampleRanges[][1]),
                length(sampleRanges[][2]),
                length(sampleRanges[][3])
            )
        )
        return volume_view!(scene, vd, sampleRanges, volume_cmap, volRange)
    end

    function render_movement_view_ts(scene, ts, time, alignment, correlationThreshold)
        d = @lift begin
            posValsTup = map(t -> apply_alignment($alignment[t], alignedPositionsMatrices[t]), $ts)

            distances, t_to_mtx = get_local_matrix(cluster_data[], $ts)
            R = kmedoids(distances, 1)
            representativeIdx = first(R.medoids)

            positions = [Point3f.(eachrow(p)) for p in first.(posValsTup)]
            refPositions = positions[representativeIdx] # chose the median in the future
            velocities = last.(posValsTup) .- first.(posValsTup)

            vd = fill(Point3f(0.0, 0.0, 0.0), length(refPositions))
            correlationMeasure = zeros(Float32, length(refPositions))

            num_neighbors = 50
            clusterKd = KDTree.(positions)
            for pId in eachindex(refPositions)
                vectorList = fill(Point3f(0.0, 0.0, 0.0), length(clusterKd))
                for t in eachindex(clusterKd)

                    knn, dists = NearestNeighbors.knn(clusterKd[t], refPositions[pId], num_neighbors)
                    uValue = sum(((2pi)^(3 / 2) * kernelWidth[]^3) * kernelFunction.(Ref(refPositions[pId]), positions[t][knn, :], kernelWidth[]) .* velocities[t][knn, 1])
                    vValue = sum(((2pi)^(3 / 2) * kernelWidth[]^3) * kernelFunction.(Ref(refPositions[pId]), positions[t][knn, :], kernelWidth[]) .* velocities[t][knn, 2])
                    wValue = sum(((2pi)^(3 / 2) * kernelWidth[]^3) * kernelFunction.(Ref(refPositions[pId]), positions[t][knn, :], kernelWidth[]) .* velocities[t][knn, 3])
                    vd[pId] += Point3f(uValue, vValue, wValue) * 1.0 / (length(clusterKd))
                    vectorList[t] = Point3f(uValue, vValue, wValue)
                end
                meanV = mean(vectorList)
                for v in vectorList
                    correlationMeasure[pId] += (dot(meanV, v)) / (dot(meanV, meanV) + dot(v, v))
                end
                correlationMeasure[pId] *= 1.0 / length(vectorList)
                correlationMeasure[pId] += 0.5
            end

            inits = first.(posValsTup)[representativeIdx]
            fins = last.(posValsTup)[representativeIdx]
            return (inits, fins), vd, correlationMeasure
        end

        return simple_arrow_view!(scene,
            lift(x -> x[1], d),
            time,
            CLUSTER_CONSENSUS_COLORMAP,
            lift(x -> x[2], d),
            lift(x -> x[3], d),
            correlationThreshold)
    end

    function render_superquadrics_view(scene, transition, inspector, alignment)
        t_ap = create_position_alignment_observer(transition, alignment)

        invariant = lift((x, y) -> active_trajectory[x][y], selected_invariant, transition)
        points = lift(x -> Point3f.(eachrow(x[1])), t_ap)
        spa = lift(x -> stretchedPrincipalAxes[x], transition)

        colors = lift((xx, y) -> map(x -> y[x], eachindex(xx)), points, invariant)
        sq = Observable(superquadric.(1.0, points[], spa[], 3.0, 0.1)[:]
        )
        calc_sq = on(spa, update=true, weak=true) do s
            sq[] = superquadric.(1.0, points[], s, 3.0, 0.1)[:]
        end

        il, is, plots = superquadrics_view!(scene, points, sq, colors, volume_cmap, invariantRange, inspector)

        push!(il, calc_sq)
        return il, is, plots
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

    function correlation_slider(figure, default)
        correlationThreshold = Observable(default)
        c_slider = Slider(figure, range=0.0:0.01:1.0, startvalue=default)
        on(c_slider.value) do x
            correlationThreshold[] = x
        end
        sg = hgrid!(Label(figure, "Correlation", font=:italic),
            c_slider,
            Label(figure, lift(x -> string(x), correlationThreshold), tellwidth=false))

        return correlationThreshold, sg
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
        Colorbar(gg[2, 2],
            colorrange=lift(x -> scalar_ranges[x], scalar_selection),
            vertical=false,
            colormap=atom_cmap,
            tellwidth=false)

        return gg, time, scalar_selection
    end

    function embed_colorbar(figure, render_selection, scalar_selection)
        scalar_range = lift(x -> scalar_ranges[x], scalar_selection)

        currentRange = Observable((0.0, 1.0))
        currentCMap = Observable(to_colormap(:reds))

        onany(render_selection, scalar_range, volRange, volume_cmap, update=true) do rs, sr, vr, vc
            if rs == "Volume" || rs == "Superquadric"
                currentRange[] = vr
                currentCMap[] = vc
            else
                currentRange[] = sr
                currentCMap[] = atom_cmap
            end
        end

        cbar = Colorbar(figure,
            colorrange=currentRange,
            vertical=false,
            colormap=currentCMap)

        return cbar
    end

    # just pass this dictionary around and pass in the arguments it needs
    render_views = Dict()
    render_views["Atom"] = render_atom_view
    render_views["Volume"] = render_volume_view
    render_views["Superquadric"] = render_superquadrics_view
    render_views["SMovement"] = render_movement_view_ts

    widgets = Dict()
    widgets["Atom"] = atom_widgets
    widgets["Movement"] = time_slider
    widgets["Render"] = render_menu
    widgets["Scalar"] = scalar_menu
    widgets["Colorbar"] = embed_colorbar
    widgets["CorrThreshold"] = correlation_slider

    calculators = Dict()
    calculators["Alignment"] = calc_alignment
    calculators["GetTransitions"] = get_transitions

    settings_window = build_settings_menu(selected_invariant, selected_alignment, collect(keys(alignments)))

    ds::MaybeObservable{DataInspector} = Observable(nothing)
    # atomPositions, stateKDTree, numAtoms, firstTransition 
    window = build_selection_window(transitionSequence,
        rel_t_to_idx,
        num_atoms,
        dm,
        volRange,
        volume_cmap,
        cluster_data,
        cluster_info,
        scalars,
        h_cutoff,
        h_range,
        settings_window,
        render_views,
        widgets,
        invariantRange,
        selected_dm_name,
        #active_trajectory["per_t_scalars"],
        #active_trajectory["per_t_scalar_ranges"],
        calculators,
        name,
        ds
    )

    #= 
    # creating screen after the window is built prevents subtle bugs
    # such as interactions being trigged before the window is rendered 
    =#
    # create inspector after render to avoid bugs
    ds[] = DataInspector(window)

    screen = GLMakie.Screen(title="LAMDA - Selection Window")
    display(screen, window)

    close(screen_ref[])

end
end # close module
