function build_reduction_window(active_trajectory::Trajectory,
    dataPath::String,
    cachePath::String;
    init_h_cutoff::AbstractFloat=0.3,
    distance_matrix::Maybe{String}=nothing,
    kwargs...)

    set_theme!(UI_THEME)
    window = Figure(size=(1920, 1080))

    top_bar(window, "Reduction", 2)
    # only need transitions and distance matrix
    dms::Dict{String,Matrix{Float32}} = active_trajectory.dms
    transitionSequence::Vector{Transition} = active_trajectory.transitions
    t_to_idx::Dict{Transition,Int} = active_trajectory.t_to_idx

    h_cutoff::Observable{Float32} = Observable(Float32(init_h_cutoff), ignore_equal_values=true)

    selected_dm::Observable{String} = Observable((isnothing(distance_matrix)) ? first(keys(dms)) : distance_matrix, ignore_equal_values=true)
    clustering::Observable{Clustering.Hclust{Float32}} = @lift begin
        @info "Clustering $($selected_dm)..."
        res = hclust(dms[$selected_dm], linkage=:ward, branchorder=:barjoseph)
        return res
    end

    cluster_groups::Observable{Dict{Int,Vector{Int}}} = @lift begin
        # vector of ints in transitionSequence order corresponding to the cluster each index is assigned
        assignments::Vector{Index} = cutree($clustering, h=$h_cutoff)
        groups = Dict{Index,Vector{Index}}()
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
        idx_to_mtx = Vector{Int}(undef, size(m)[1])
        for (i, r) in enumerate($clustering.order)
            idx_to_mtx[r] = i
        end

        # gets the correct idx 
        rm = view(m, $clustering.order, $clustering.order)
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
        red_t_to_idx = Dict{Transition,Index}()
        idxes = Vector{Index}(undef, n)

        m = dms[$selected_dm]
        for (i, (clusterIdx, g)) in enumerate($cluster_groups)
            # find reference t
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
            dataPath,
            cachePath;
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
    dm::AbstractArray{Float32},
    transitionSequence::Vector{Transition},
    clustering::Clustering.Hclust{Float32},
    selected_dm_name::String,
    init_h_cutoff::AbstractFloat,
    dataPath::String,
    cachePath::String;
    align_with::Maybe{String}=nothing)

    (;
        stretchedPrincipalAxes,
        scalars,
        scalar_ranges,
        alignedPositionsMatrices,
        alignments,
        name,
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
    rel_t_to_idx::Dict{Transition,Int} = Dict(reverse.(collect(enumerate(transitionSequence))))
    h_cutoff::Observable{Float32} = Observable(Float32(init_h_cutoff))

    rm = view(dm, clustering.order, clustering.order)
    cluster_data = ClusterData(clustering, transitionSequence, rm)
    cluster_info = @lift begin
        return ClusterInfo(cluster_data, transitionSequence, $h_cutoff)
    end

    init_alignment = (!isnothing(align_with) && align_with in keys(alignments)) ? align_with : first(keys(alignments))
    selected_alignment = Observable(init_alignment)

    # should be fine, seems off-center because abs(volMin) != abs(volMax)
    lowmap = reverse(resample_cmap(:RdPu_3, 50;
        alpha=([(0.0):0.02:(0.99);] ./ 0.05) .^ 6))
    himap = resample_cmap(:greens, 50;
        alpha=([(0.0):0.02:(0.99);] ./ 0.05) .^ 6)
    t3map = vcat(lowmap, himap)

    volume_cmaps = Dict("t1" => resample_cmap(:bam, 100;
            alpha=([(-0.99):0.02:(0.99);] ./ 0.1) .^ 6),
        "t2" => resample_cmap(:matter, 100;
            alpha=([0:0.01:0.99;] ./ 0.05) .^ 2),
        "t3" => t3map)

    function get_invariant_range(x)
        iv = select_invariant(x)
        vals = values(iv[])
        absInvMin = minimum(minimum.(vals))
        absInvMax = maximum(maximum.(vals))
        return (absInvMin, absInvMax)
    end

    invariantRanges = Dict("t1" => get_invariant_range("t1"),
        "t2" => get_invariant_range("t2"),
        "t3" => get_invariant_range("t3"))

    # https://docs.julialang.org/en/v1.12-dev/manual/performance-tips/#man-performance-captured
    # convenience function to avoid passing around all the data
    function calc_alignment(ts::AbstractArray{Transition})::Tuple{Transition,Dict{Transition,Tuple{Matrix{Float32},Bool}}}
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
    #:linear_wcmr_100_45_c42_n256 
    atom_cmap = resample_cmap(:linear_bmy_10_95_c71_n256, 100,
        alpha=range(; start=0.1, stop=1.0, length=100))

    function render_atom_view(scene::Makie.Scene,
        transition::Transition,
        selected_scalar::Observable{String},
        time::Observable{<:AbstractFloat},
        flip::Bool=false)

        res = let scalars = scalars, scalar_ranges = scalar_ranges, atom_cmap = atom_cmap, alignedPositionsMatrices = alignedPositionsMatrices
            ap = flip ? reverse(alignedPositionsMatrices[transition]) : alignedPositionsMatrices[transition]
            simple_atom_view!(scene,
                ap,
                lift(x -> scalars[x][transition], selected_scalar),
                lift(x -> scalar_ranges[x], selected_scalar),
                atom_cmap,
                time
            )
        end
        return res
    end

    function render_superquadrics_view(scene::Makie.Scene, transition::Transition, si::Observable{String}, flip::Bool=false)
        il, is, plots = let alignedPositionsMatrices = alignedPositionsMatrices,
            stretchedPrincipalAxes = stretchedPrincipalAxes

            t_ap = alignedPositionsMatrices[transition]
            idx = flip ? 2 : 1
            points = Point3f.(eachrow(t_ap[idx]))
            # do the invariant values need to be flipped as well?
            spa = stretchedPrincipalAxes[transition]
            colors = @lift view(select_invariant($si)[][transition], eachindex(points))

            # both fns allocate a bunch of space
            il, is, plots = superquadrics_view!(scene,
                points,
                colors,
                spa,
                lift(x -> volume_cmaps[x], si),
                lift(x -> invariantRanges[x], si))

            #push!(is, colors)
            return il, is, plots
        end
    end

    function render_movement_view_ts(scene::Makie.Scene,
        ts::AbstractArray{Transition},
        time::Observable{Float32},
        alignment::Dict{Transition,Tuple{Matrix{Float32},Bool}},
        correlationThreshold::Observable{<:AbstractFloat})

        res = let alignedPositionsMatrices = alignedPositionsMatrices,
            cluster_data = cluster_data,
            kernelWidth = 1.0

            posValsTup = map(t -> apply_alignment(alignment[t], alignedPositionsMatrices[t]), ts)
            distances, t_to_mtx, mtx_to_t = get_local_matrix(cluster_data, ts)
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
                    k = ((2pi)^(3 / 2) * kernelWidth[]^3)
                    kf = kernelFunction.(Ref(refPositions[pId]), pos, kernelWidth[])

                    uValue = sum(k * kf .* view(velocities[t], knn, 1))
                    vValue = sum(k * kf .* view(velocities[t], knn, 2))
                    wValue = sum(k * kf .* view(velocities[t], knn, 3))

                    vd[pId] += Point3f(uValue, vValue, wValue) * (1.0 / length(clusterKd))
                    vectorList[t] = Point3f(uValue, vValue, wValue)
                end
                meanV = mean(vectorList)
                dotmV = dot(meanV, meanV)
                for v in vectorList
                    correlationMeasure[pId] += (dot(meanV, v)) / (dotmV + dot(v, v))
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

    function time_slider(figure::Makie.Figure,
        time::Observable{Float32}=Observable(Float32(0.0)))

        t_slider = Slider(figure, range=0.0:0.05:1.0, startvalue=time[])
        onany(t_slider, t_slider.value) do _, x
            time[] = x
        end

        sg = hgrid!(Label(figure, "t", font=:italic),
            t_slider,
            Label(figure, lift(x -> string(x), time)))

        return time, sg
    end

    function invariants_menu(figure::Makie.Figure,
        si::Observable{String}=Observable("t1"))

        invar_menu = Menu(figure,
            options=["t1", "t2", "t3"],
            default=si[],
            tellwidth=false)

        onany(invar_menu, invar_menu.selection) do _, val
            si[] = val
        end

        return si, invar_menu
    end

    function correlation_slider(figure::Makie.Figure, default::AbstractFloat)
        correlationThreshold = Observable(default)
        c_slider = Slider(figure, range=0.0:0.01:1.0, startvalue=default)
        onany(c_slider, c_slider.value) do _, x
            correlationThreshold[] = x
        end
        sg = hgrid!(Label(figure, "Correlation", font=:italic),
            c_slider,
            Label(figure, lift(x -> string(x), correlationThreshold), tellwidth=false))

        return correlationThreshold, sg
    end

    # could be one func
    function render_menu(figure::Makie.Figure;
        scene_selector::Observable{String}=Observable("Atom"))

        render_menu = Menu(figure,
            options=SINGLE_TRANSITION_RENDER_OPTIONS,
            default=scene_selector[], tellwidth=false)

        onany(render_menu, render_menu.selection) do _, s
            scene_selector[] = s
        end

        return scene_selector, render_menu
    end

    scalar_opts = sort(collect(keys(scalars)))
    function scalar_menu(figure::Makie.Figure,
        scalar_selection::Observable{String}=Observable(first(scalar_opts))
    )
        m = Menu(figure, options=scalar_opts, default=scalar_selection[])
        onany(m, m.selection) do _, ms
            scalar_selection[] = ms
        end

        return scalar_selection, m
    end

    function atom_widgets(figure::Makie.Figure,
        grid,
        init_time::Observable{Float32}=Observable(Float32(0.0)),
        scalar_selection::Observable{String}=Observable(first(scalar_opts))
    )

        gg = GridLayout(grid[end+1, :])

        time, t_slider = time_slider(init_time, figure)

        gg[1, 1:2] = t_slider

        _, m = scalar_menu(figure, scalar_selection)
        gg[2, 1] = m

        Colorbar(gg[2, 2],
            colorrange=lift(x -> scalar_ranges[x], scalar_selection),
            vertical=false,
            colormap=atom_cmap,
            tellwidth=false)

        return gg, time, scalar_selection
    end

    function embed_colorbar(figure::Makie.Figure,
        render_selection::Observable{String},
        scalar_selection::Observable{String},
        invariant_selection::Observable{String}
    )
        scalar_range = lift(x -> scalar_ranges[x], scalar_selection)
        invariant_range = lift(x -> invariantRanges[x], invariant_selection)
        volume_cmap = lift(x -> volume_cmaps[x], invariant_selection)

        currentRange = Observable((0.0, 1.0))
        currentCMap = Observable(atom_cmap)

        cbar = Colorbar(figure,
            colorrange=currentRange,
            vertical=false,
            colormap=currentCMap)

        listener = onany(cbar, render_selection, scalar_range, invariant_range, volume_cmap, update=true, weak=true) do _, rs, sr, vr, vc
            if rs == "Superquadric"
                currentRange[] = vr
                currentCMap[] = vc
            else
                currentRange[] = sr
                currentCMap[] = atom_cmap
            end
        end

        return cbar, listener
    end

    # just pass this dictionary around and pass in the arguments it needs
    render_views::Dict{String,Function} = Dict{String,Function}(
        "Atom" => render_atom_view,
        "Superquadric" => render_superquadrics_view,
        "SMovement" => render_movement_view_ts)

    widgets::Dict{String,Function} = Dict{String,Function}(
        "Atom" => atom_widgets,
        "Movement" => time_slider,
        "Render" => render_menu,
        "Invariant" => invariants_menu,
        "Scalar" => scalar_menu,
        "Colorbar" => embed_colorbar,
        "CorrThreshold" => correlation_slider)

    calculators::Dict{String,Function} = Dict{String,Function}("Alignment" => calc_alignment)
    # get number of atoms
    num_atoms = size(Iterators.first(values(alignedPositionsMatrices))[1])[1]
    settings_window = build_settings_menu(selected_alignment, collect(keys(alignments)), num_atoms)

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
        dataPath
    )

    # create inspector after render to avoid bugs
    ds = DataInspector(window)
    GLMakie.set_title!(screen, "LAMDA - Selection Window")
    display(screen, window)
    on(events(window).window_open) do e
        if !e
            @debug "Killing main"
            cleanup()
            dm = nothing
            transitionSequence = nothing
            clustering = nothing

            Observables.clear(cluster_info)

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

