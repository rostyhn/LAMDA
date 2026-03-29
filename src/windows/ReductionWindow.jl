function build_reduction_window(active_trajectory::Trajectory,
    dataPath::String;
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

    # TODO: should be able to select linkage method
    clustering::Observable{Clustering.Hclust{Float32}} = lift(window.scene, selected_dm) do sel
        @info "Clustering $sel"
        res = hclust(dms[sel], linkage=:average, branchorder=:barjoseph)
        return res
    end

    cluster_groups::Observable{Dict{Int,Vector{Int}}} = lift(window.scene, clustering, h_cutoff) do clus, cutoff
        # vector of ints in transitionSequence order corresponding to the cluster each index is assigned
        assignments::Vector{Index} = cutree(clus, h=cutoff)
        groups = Dict{Index,Vector{Index}}()
        for (i, c) in enumerate(assignments)
            g = get(groups, c, [])
            push!(g, i)
            groups[c] = g
        end

        return groups
    end

    # reorders distance matrix according to clustering
    reordered_matrix = lift(window.scene, selected_dm, clustering) do sel, c
        m = dms[sel]
        idx_to_mtx = Vector{Int}(undef, size(m)[1])
        for (i, r) in enumerate(c.order)
            idx_to_mtx[r] = i
        end

        # gets the correct idx 
        rm = view(m, c.order, c.order)
        # get minimum and maximum of entire matrix for cmap
        fl = vec(m) #vec doesn't allocate
        return rm, idx_to_mtx, (minimum(fl), maximum(fl))
    end

    control_grid = GridLayout()
    window[2, 1:2] = control_grid

    dm_menu = Menu(window, options=collect(keys(dms)), default=selected_dm[])
    Makie.onany(window.scene, dm_menu.selection) do val
        selected_dm[] = val
    end

    cutoff_label = Label(window, "Cutoff")
    cutoff_tb = Textbox(window, validator=Float64, placeholder=string(h_cutoff[]))

    Makie.onany(window.scene, cutoff_tb.stored_string) do s
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
    rendered_clusters = Ref([])
    avgs = lift(window.scene, cluster_groups, reordered_matrix) do cg, rm
        foreach(x -> delete!(parent_scene(x), x), rendered_clusters[])
        cmap = CLUSTER_COLORMAP
        avgs = Array{Float32}(undef, length(keys(cg)))
        pos = Vector{Tuple{Int,Int}}(undef, length(keys(cg)))

        for (i, (c, ts_idx)) in enumerate(cg)
            idx_to_mtx = rm[2]
            m_idx = view(idx_to_mtx, ts_idx)

            lo = minimum(m_idx)
            hi = maximum(m_idx)
            vals = view(rm[1], m_idx, m_idx)
            utri = triu!(trues(size(vals)))
            avgs[i] = mean(vec(vals[utri]))
            pos[i] = (lo, hi)
        end

        avgrng = extrema(avgs)
        rendered_clusters[] = map(x -> draw_bbox_pixel_space!(hm_ax.scene, x[1][1], x[1][2];
                colormap=:binary,
                colorrange=avgrng,
                color=x[2]), collect(zip(pos, avgs)))

        return avgs
    end
    hist!(hist_ax, avgs,
        strokewidth=1,
        strokecolor=:black,
        color=:grey
    )

    Makie.onany(window.scene, avgs) do v
        reset_limits!(hist_ax)
    end

    # apply reduction
    reduced = lift(window.scene, cluster_groups, selected_dm, reordered_matrix) do cg, sel, reo
        n = length(collect(keys(cg)))
        red_t_list = Vector{Transition}(undef, n)
        red_t_to_idx = Dict{Transition,Index}()
        idxes = Vector{Index}(undef, n)

        m = dms[sel]
        for (i, (clusterIdx, g)) in enumerate(cg)
            # find reference t
            dist_sum = map(x -> sum(view(m, x, g)), g)
            ref_t_idx = argmin(dist_sum)

            ts = view(transitionSequence, g)

            ref_t = ts[ref_t_idx]
            t_idx = t_to_idx[ref_t] # get absolute transitionSequence index
            mtx_idx = reo[2][t_idx]

            red_t_list[i] = ref_t
            red_t_to_idx[ref_t] = i
            idxes[i] = mtx_idx
        end

        redmat = view(reo[1], idxes, idxes)

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

    Makie.on(window.scene, reduced) do r
        autolimits!(red_hm_ax)
    end

    Makie.onany(window.scene, go_btn.clicks) do n
        screen = window.scene.current_screens[1]
        @time setup_selection_window(active_trajectory,
            screen,
            reduced[][1],
            reduced[][2],
            reduced[][3],
            selected_dm[],
            dataPath;
            kwargs...)
        empty!(window)
        Makie.free(window.scene)
        window = nothing
    end

    final_cleanup = function ()
        @debug "Reduction window cleanup"
        if !isnothing(window)
            empty!(window)
            Makie.free(window.scene)
            window = nothing
        end
        empty!(rendered_clusters[])

        Observables.clear(selected_dm)
        Observables.clear(cluster_groups)
        Observables.clear(reduced)
        Observables.clear(clustering)
        #Observables.clear(avgs)
        #Observables.clear(reordered_matrix)
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


