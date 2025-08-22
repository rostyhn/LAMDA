function build_reduction_window(active_trajectory::Trajectory;
    init_h_cutoff::AbstractFloat=0.3,
    distance_matrix::Maybe{String}=nothing,
    kwargs...)

    set_theme!(UI_THEME)
    window = Figure(size=(1920, 1080))

    top_bar(window, "Reduction", 2)
    # only need transitions and distance matrix
    dms::Dict{String,Matrix{Float32}} = active_trajectory.dms
    transitionSequence::Vector{Transition} = active_trajectory.transitions

    t_to_idx::Dict{Transition,UInt16} = active_trajectory.t_to_idx

    h_cutoff::Observable{Float32} = Observable(Float32(init_h_cutoff))
    h_range::Observable{Tuple{Float32,Float32}} = Observable((floatmin(Float32), floatmax(Float32)))

    selected_dm::Observable{String} = Observable((isnothing(distance_matrix)) ? first(keys(dms)) : distance_matrix)
    clustering::Observable{Clustering.Hclust{Float32}} = @lift begin
        println("Clustering $($selected_dm)...")
        res = hclust(dms[$selected_dm], linkage=:ward, branchorder=:barjoseph)

        h_range[] = extrema(res.heights)
        notify(h_range)
        return res
    end

    cluster_groups::Observable{Dict{UInt16,Vector{UInt16}}} = @lift begin
        # vector of ints in transitionSequence order corresponding to the cluster each index is assigned
        assignments::Vector{UInt16} = cutree($clustering, h=$h_cutoff)
        groups = Dict{UInt16,Vector{UInt16}}()
        for (i, c) in enumerate(assignments)
            g = get(groups, c, [])
            push!(g, i)
            groups[c] = g
        end

        return groups
    end

    # reorders distance matrix according to clustering
    reordered_matrix = @lift begin
        m = dms[$selected_dm]
        # gets the correct idx 
        rm = view(m, $clustering.order, $clustering.order)
        idx_to_mtx = Vector{UInt16}(undef, size(m)[1])
        for (i, r) in enumerate($clustering.order)
            idx_to_mtx[r] = i
        end
        # get minimum and maximum of entire matrix for cmap
        fl = vec(m) #vec doesn't allocate
        return rm, idx_to_mtx, (minimum(fl), maximum(fl))
    end

    control_grid = GridLayout()
    window[2, 1:2] = control_grid

    dm_menu = Menu(window, options=collect(keys(dms)), default=selected_dm[])
    dm_menu_listener = on(dm_menu.selection, weak=true) do val
        selected_dm[] = val
    end

    cutoff_label = Label(window, "Cutoff")
    cutoff_tb = Textbox(window, validator=Float64, placeholder=string(h_cutoff[]))

    cutoff_listener = on(cutoff_tb.stored_string, weak=true) do s
        h_cutoff[] = parse(Float64, s)
        notify(h_cutoff)
    end

    go_btn = Button(window, label="Explore")

    control_grid[1, 1] = hgrid!(
        Label(window, "Selected matrix"),
        dm_menu,
        cutoff_label,
        cutoff_tb,
        go_btn)

    hist_ax = Axis(control_grid[2, 1],
        title="Average intra-cluster distance",
        backgroundcolor=:transparent)

    hm_ax = Axis(window[3, 1],
        title=lift(x -> "Original: $(size(x[1]))", reordered_matrix),
        backgroundcolor=:transparent)
    hidedecorations!(hm_ax)
    deregister_interaction!(hm_ax, :rectanglezoom)

    heatmap!(hm_ax,
        lift(x -> x[1], reordered_matrix),
        colorrange=lift(x -> x[3], reordered_matrix),
        colormap=DISTANCE_MATRIX_COLORMAP)

    # draw clusters on screen and also calculate some stats on each group
    rendered_clusters = []
    avgs = @lift begin
        foreach(x -> delete!(parent_scene(x), x), rendered_clusters)
        cmap = CLUSTER_COLORMAP
        avgs = Array{Float32}(undef, length(keys($cluster_groups)))

        for (i, (c, ts_idx)) in enumerate($cluster_groups)
            idx_to_mtx = $reordered_matrix[2]
            m_idx = view(idx_to_mtx, ts_idx)

            lo = minimum(m_idx)
            hi = maximum(m_idx)

            p = draw_bbox_pixel_space!(hm_ax.scene, lo, hi; color=cmap[mod1(c, length(cmap))])
            vals = view($reordered_matrix[1], m_idx, m_idx)
            utri = triu!(trues(size(vals)))
            avgs[i] = mean(vec(vals[utri]))

            push!(rendered_clusters, p)
        end
        return avgs
    end
    hist!(hist_ax, avgs,
        strokewidth=1,
        strokecolor=:black,
        color=:grey
    )

    avg_listener = on(avgs) do v
        reset_limits!(hist_ax)
    end

    # apply reduction
    reduced = @lift begin
        n = length(collect(keys($cluster_groups)))
        red_t_list = Vector{Transition}(undef, n)
        red_t_to_idx = Dict{Transition,Int16}()
        idxes = Vector{Int16}(undef, n)
        for (i, (clusterIdx, g)) in enumerate($cluster_groups)
            # find reference t
            m = dms[$selected_dm]
            dist_sum = map(x -> sum(view(m, x, g)), g)
            ref_t_idx = argmin(dist_sum)

            ts = view(transitionSequence, g)

            ref_t = ts[ref_t_idx]
            t_idx = t_to_idx[ref_t] # get absolute transitionSequence index
            mtx_idx = $reordered_matrix[2][t_idx]

            red_t_list[i] = ref_t
            red_t_to_idx[ref_t] = i
            idxes[i] = mtx_idx
        end

        redmat = view($reordered_matrix[1], idxes, idxes)

        # should just do this here and pass it down to main instead of doing it twice
        cluster = hclust(redmat, linkage=:ward, branchorder=:barjoseph)
        return redmat, red_t_list, cluster
    end

    red_hm_ax = Axis(window[3, 2],
        title=lift(x -> "Reduced: $(size(x[1]))", reduced),
        backgroundcolor=:transparent)

    hidedecorations!(red_hm_ax)
    deregister_interaction!(red_hm_ax, :rectanglezoom)

    heatmap!(red_hm_ax, lift(x -> view(x[1], x[3].order, x[3].order), reduced),
        colorrange=lift(x -> x[3], reordered_matrix),
        colormap=DISTANCE_MATRIX_COLORMAP)

    go_listener = on(go_btn.clicks, weak=true) do n
        screen = window.scene.current_screens[1]
        @time main_window(active_trajectory,
            screen,
            reduced[][1],
            reduced[][2],
            reduced[][3],
            selected_dm[],
            h_cutoff[],
            kwargs...)

        empty!(window)
        Makie.free(window.scene)
        window = nothing
    end

    final_cleanup = function ()
        @debug "Reduction window cleanup"
        off(go_listener)
        off(avg_listener)
        off(dm_menu_listener)
        off(cutoff_listener)
        if !isnothing(window)
            empty!(window)
            Makie.free(window.scene)
            window = nothing
        end

        go_listener = nothing
        avg_listener = nothing
        dm_menu_listener = nothing
        cutoff_listener = nothing
        empty!(rendered_clusters)

        Observables.clear(selected_dm)
        Observables.clear(cluster_groups)
        Observables.clear(reduced)
        Observables.clear(clustering)
        Observables.clear(avgs)
        Observables.clear(reordered_matrix)
        Observables.clear(h_cutoff)

    end

    Colorbar(window[4, 1:2],
        limits=lift(x -> x[3], reordered_matrix),
        label="Distances",
        vertical=false,
        colormap=DISTANCE_MATRIX_COLORMAP)

    #linkaxes!(hm_ax, red_hm_ax)
    @debug "Finished Reduction Window"
    return window, final_cleanup
end

function main_window(active_trajectory::Trajectory,
    screen::GLMakie.Screen,
    dm,
    transitionSequence::Vector{Transition},
    clustering::Clustering.Hclust{Float32},
    selected_dm_name::String,
    init_h_cutoff::AbstractFloat;
    chunk_size::Integer=100,
    align_with::Maybe{String}=nothing)

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

    function select_invariant(selection::String)
        if selection == "t1"
            iv = Ref(active_trajectory.t1)
        elseif selection == "t2"
            iv = Ref(active_trajectory.t2)
        elseif selection == "t3"
            iv = Ref(active_trajectory.t3)
        else
            error("Invalid invariant selected")
        end

        return iv
    end
    # absolute index for volume data

    rel_t_to_idx::Dict{Transition,UInt16} = Dict(reverse.(collect(enumerate(transitionSequence))))
    h_cutoff::Observable{Float32} = Observable(Float32(init_h_cutoff))
    h_range::Tuple{Float32,Float32} = extrema(clustering.heights)

    rm = view(dm, clustering.order, clustering.order)
    # gets the correct idx 
    t_to_mtx = Dict{Transition,UInt16}()
    mtx_to_t = Dict{UInt16,Transition}()
    for (i, r) in enumerate(clustering.order)
        t_to_mtx[transitionSequence[r]] = i
        mtx_to_t[i] = transitionSequence[r]
    end

    # get minimum and maximum of entire matrix for cmap

    c2idx, c_to_parent, parent_to_c = get_hierarchy(clustering)
    # render dendrogram once so that we can use any piece of it in the cluster window
    lines, clusters, c2lx = treepositions(clustering, 0.0)

    fl = vec(dm)
    cluster_data = ClusterData(clustering=clustering,
        matrix=rm,
        c2idx=c2idx,
        clusters=clusters,
        lines=lines,
        c2lx=c2lx,
        c_to_parent=c_to_parent,
        parent_to_c=parent_to_c,
        m_extrema=extrema(fl),
        t_to_mtx=t_to_mtx,
        mtx_to_t=mtx_to_t)

    # vector of ints in transitionSequence order corresponding to the cluster each index is assigned
    cluster_info = @lift begin
        assignments = cutree(cluster_data.clustering, h=$h_cutoff)
        groups = Dict{UInt16,Vector{Transition}}()
        igroups = Dict{UInt16,Vector{Int}}()
        # current assigned cluster to idx
        ccidx2cidx = Dict{UInt16,UInt16}()
        # cluster to assignment idx 
        a2c = Dict{UInt16,Set{UInt16}}()
        for (i, c) in enumerate(assignments)
            g = get(groups, c, [])
            ig = get(igroups, c, [])
            push!(g, transitionSequence[i])
            push!(ig, i)

            groups[c] = g
            igroups[c] = ig
        end

        for (idx, g) in igroups
            c = Set(g)
            a2c[idx] = c
            ccidx2cidx[idx] = cluster_data.c2idx[c]
        end

        reps = Dict{Int,Transition}()
        for (clusterIdx, g) in groups
            gi = map(x -> cluster_data.t_to_mtx[x], g)
            dist_sum = map(x -> sum(view(dm, x, gi)), gi)
            reps[clusterIdx] = g[argmin(dist_sum)]
        end

        # instead of rendering the dendrogram twice like this and keeping two copies in memory, could modify dendrogram render to show lines under the cutoff differently
        sLines, sClusters, sC2lx = treepositions(cluster_data.clustering, $h_cutoff)
        return ClusterInfo(groups=groups,
            representatives=reps,
            assignments=assignments,
            lines=sLines,
            rel_t_to_idx=rel_t_to_idx,
            clusters=sClusters,
            c2lx=sC2lx,
            a2c=a2c,
            cc2cidx=ccidx2cidx,
            h_range=h_range,
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

            iv = select_invariant($selected_invariant)
            invariants = map(x -> iv[][x], transitions)

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
        iv = select_invariant($selected_invariant)
        vals = values(iv[])
        absInvMin = minimum(minimum.(vals))
        absInvMax = maximum(maximum.(vals))
        return (absInvMin, absInvMax)
    end

    # https://docs.julialang.org/en/v1.12-dev/manual/performance-tips/#man-performance-captured
    # convenience function to avoid passing around all the data
    function calc_alignment(ts::Vector{Transition})::Tuple{Transition,Dict{Transition,Tuple{Matrix{Float32},Bool}}}
        res = let rel_t_to_idx = rel_t_to_idx,
            alignments = alignments,
            dm = dm,
            alignedPositionsMatrices = alignedPositionsMatrices,
            selected_alignment = selected_alignment

            if !isempty(ts)
                ts_idx = map(x -> rel_t_to_idx[x], ts)
                features = alignments[selected_alignment[]]

                dist_sum = map(x -> sum(view(dm, x, ts_idx)), ts_idx)
                ref_t_idx = argmin(dist_sum)

                ref_t = ts[ref_t_idx]
                return ref_t, calculate_alignment(ref_t, ts, alignedPositionsMatrices, features)
            else
                return (1, 1), Dict{Transition,Tuple{Matrix{Float32},Bool}}()
            end
        end
        return res
    end


    # did this to avoid drilling down and passing parameters constantly
    atom_cmap = resample_cmap(:linear_wcmr_100_45_c42_n256, 100, alpha=range(; start=0.01, stop=1.0, length=100))
    function render_atom_view(scene::Makie.Scene, transition::Transition, selected_scalar, time, flip=false)
        res = let scalars = scalars, scalar_ranges = scalar_ranges, atom_cmap = atom_cmap, alignedPositionsMatrices = alignedPositionsMatrices
            ap = flip ? reverse(alignedPositionsMatrices[transition]) : alignedPositionsMatrices[transition]
            simple_atom_view!(scene, ap,
                lift(x -> scalars[x][transition], selected_scalar),
                lift(x -> scalar_ranges[x], selected_scalar),
                atom_cmap,
                time)
        end
        return res
    end

    function render_volume_view(scene::Makie.Scene, transition::Transition)
        # directly indexing the mmap creates a copy, need to use a view
        vvd = let volumeData = volumeData, t_to_idx = t_to_idx
            view(volumeData[], :, t_to_idx[transition])
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

    function render_movement_view_ts(scene::Makie.Scene, ts::Vector{Transition}, time::Observable{Float32},
        alignment::Dict{Transition,Tuple{Matrix{Float32},Bool}},
        correlationThreshold::Observable{Float32})

        res = let alignedPositionsMatrices = alignedPositionsMatrices, cluster_data = cluster_data, kernelWidth = kernelWidth

            posValsTup = map(t -> apply_alignment(alignment[t], alignedPositionsMatrices[t]), ts)
            distances, t_to_mtx = get_local_matrix(cluster_data, ts)
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
                    pos = view(positions[t], knn, :)
                    uValue = sum(((2pi)^(3 / 2) * kernelWidth[]^3) * kernelFunction.(Ref(refPositions[pId]), pos, kernelWidth[]) .* view(velocities[t], knn, 1))
                    vValue = sum(((2pi)^(3 / 2) * kernelWidth[]^3) * kernelFunction.(Ref(refPositions[pId]), pos, kernelWidth[]) .* view(velocities[t], knn, 2))
                    wValue = sum(((2pi)^(3 / 2) * kernelWidth[]^3) * kernelFunction.(Ref(refPositions[pId]), pos, kernelWidth[]) .* view(velocities[t], knn, 3))
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

            plt = simple_arrow_view!(scene,
                (inits, fins),
                time,
                CLUSTER_CONSENSUS_COLORMAP,
                vd,
                correlationMeasure,
                correlationThreshold)
            return plt
        end

        return res
    end

    function render_superquadrics_view(scene::Makie.Scene, transition::Transition, flip::Bool=false)
        il, is, plots = let alignedPositionsMatrices = alignedPositionsMatrices, stretchedPrincipalAxes = stretchedPrincipalAxes
            t_ap = alignedPositionsMatrices[transition]
            idx = flip ? 2 : 1
            points = Point3f.(eachrow(t_ap[idx]))
            # do the invariant values need to be flipped as well?
            spa = stretchedPrincipalAxes[transition]
            colors = lift(x -> view(select_invariant(x)[][transition], eachindex(points)), selected_invariant)

            # both fns allocate a bunch of space
            sq = collect(superquadric.(1.0, points, spa, 3.0, 0.1))
            il, is, plots = superquadrics_view!(scene, points, sq, colors, volume_cmap, invariantRange)
            push!(is, colors)
            return il, is, plots
        end
    end

    function time_slider(init_time::Float32, figure::Makie.Figure)
        time = Observable(init_time)

        t_slider = Slider(figure, range=0.0:0.05:1.0, startvalue=init_time)
        on(t_slider.value) do x
            time[] = x
        end

        sg = hgrid!(Label(figure, "t", font=:italic), t_slider, Label(figure, lift(x -> string(x), time)))

        return time, sg
    end

    function correlation_slider(figure::Makie.Figure, default::Float32)
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
    function render_menu(figure::Makie.Figure; default::String="Superquadric")
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

    scalar_opts = sort(collect(keys(scalars)))
    function scalar_menu(figure::Makie.Figure)
        scalar_selection = Observable(first(scalar_opts))
        m = Menu(figure, options=scalar_opts, default=scalar_selection[])
        on(m.selection) do ms
            scalar_selection[] = ms
        end

        return scalar_selection, m
    end

    function atom_widgets(init_time::Float32, figure::Makie.Figure, grid)
        gg = GridLayout(grid[end+1, :])

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

    function embed_colorbar(figure::Makie.Figure, render_selection::Observable{String}, scalar_selection::Observable{String})
        scalar_range = lift(x -> scalar_ranges[x], scalar_selection)

        currentRange = Observable((0.0, 1.0))
        currentCMap = Observable(atom_cmap)

        listener = onany(render_selection, scalar_range, volRange, volume_cmap, update=true, weak=true) do rs, sr, vr, vc
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

        return cbar, listener
    end

    # just pass this dictionary around and pass in the arguments it needs
    render_views::Dict{String,Function} = Dict{String,Function}("Atom" => render_atom_view, "Volume" => render_volume_view,
        "Superquadric" => render_superquadrics_view, "SMovement" => render_movement_view_ts)

    widgets::Dict{String,Function} = Dict{String,Function}("Atom" => atom_widgets,
        "Movement" => time_slider,
        "Render" => render_menu,
        "Scalar" => scalar_menu,
        "Colorbar" => embed_colorbar,
        "CorrThreshold" => correlation_slider)

    calculators::Dict{String,Function} = Dict{String,Function}("Alignment" => calc_alignment, "GetTransitions" => get_transitions)
    settings_window = build_settings_menu(selected_invariant, selected_alignment, collect(keys(alignments)))

    window, cleanup = build_selection_window(
        transitionSequence,
        rel_t_to_idx,
        cluster_data,
        cluster_info,
        h_cutoff,
        settings_window,
        render_views,
        widgets,
        selected_dm_name,
        calculators,
        name,
    )

    # create inspector after render to avoid bugs
    ds = DataInspector(window)
    display(screen, window)
    on(events(window).window_open) do e
        if !e
            @debug "Killing main"
            cleanup()
            dm = nothing
            transitionSequence = nothing
            clustering = nothing

            Observables.clear(cluster_info)
            Observables.clear(volumeData)
            #Observables.clear(invariantRange)

            for x in values(calculators)
                x = nothing
            end
            empty!(calculators)

            for x in values(widgets)
                x = nothing
            end
            empty!(widgets)

            for x in values(render_views)
                x = nothing
            end
            empty!(render_views)

            empty!(window)
            empty!(settings_window)
            Makie.free(settings_window.scene)
            Makie.free(window.scene)

            select_invariant = nothing
            settings_window = nothing
            window = nothing
            ds = nothing
        end
    end
end

