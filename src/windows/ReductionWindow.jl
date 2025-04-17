function build_reduction_window(active_trajectory::Trajectory, on_click, screen_ref; init_h_cutoff=0.3, distance_matrix=nothing, kwargs...)
    set_theme!(UI_THEME)
    window = Figure(size=(1920, 1080))

    menu_bar = top_bar(window, "Reduction", 2)

    # only need transitions and distance matrix
    dms::Dict{String,Matrix{Float32}} = active_trajectory.dms
    transitionSequence::Vector{Transition} = active_trajectory.transitions

    t_to_idx::Dict{Transition,Int} = active_trajectory.t_to_idx

    h_cutoff = Observable(init_h_cutoff)
    h_range = Observable((floatmin(Float32), floatmax(Float32)))

    init_dist_mat = (isnothing(distance_matrix)) ? first(keys(dms)) : distance_matrix
    selected_dm = Observable(init_dist_mat)
    clustering = @lift begin
        println("Clustering $($selected_dm)...")
        m = dms[$selected_dm]
        res = hclust(m, linkage=:ward, branchorder=:barjoseph)

        h_range[] = extrema(res.heights)
        notify(h_range)
        return res
    end

    cluster_groups = @lift begin
        # vector of ints in transitionSequence order corresponding to the cluster each index is assigned
        assignments = cutree($clustering, h=$h_cutoff)
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

    # reorders distance matrix according to clustering
    reordered_matrix = @lift begin
        m = dms[$selected_dm]
        # gets the correct idx 
        rm = view(m, $clustering.order, $clustering.order)
        idx_to_mtx = zeros(Int, size(m)[1])
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
    on(dm_menu.selection) do val
        selected_dm[] = val
    end

    cutoff_label = Label(window, "Cutoff")
    cutoff_tb = Textbox(window, validator=Float64, placeholder=string(h_cutoff[]))

    on(cutoff_tb.stored_string) do s
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

    hm = heatmap!(hm_ax,
        lift(x -> x[1], reordered_matrix),
        colorrange=lift(x -> x[3], reordered_matrix),
        colormap=DISTANCE_MATRIX_COLORMAP)

    # draw clusters on screen and also calculate some stats on each group
    rendered_clusters = []
    avgs = @lift begin
        foreach(x -> delete!(parent_scene(x), x), rendered_clusters)
        cmap = CLUSTER_COLORMAP
        avgs = []
        for (c, ts_idx) in $cluster_groups
            idx_to_mtx = $reordered_matrix[2]
            m_idx = map(x -> idx_to_mtx[x], ts_idx)

            lo = minimum(m_idx)
            hi = maximum(m_idx)

            p = draw_bbox_pixel_space!(hm_ax.scene, lo, hi; color=cmap[mod1(c, length(cmap))])
            vals = view($reordered_matrix[1], m_idx, m_idx)
            utri = triu!(trues(size(vals)))

            push!(avgs, mean(vec(vals[utri])))
            push!(rendered_clusters, p)
        end
        return avgs
    end
    hist!(hist_ax, avgs,
        strokewidth=1,
        strokecolor=:black,
        color=:grey
    )

    on(avgs) do v
        reset_limits!(hist_ax)
    end

    # apply reduction
    reduced = @lift begin
        n = length(collect(keys($cluster_groups)))
        red_t_list = Transition[]
        red_t_to_idx = Dict{Transition,Int}()
        idxes = zeros(Int, n)
        for (i, (clusterIdx, g)) in enumerate($cluster_groups)
            # find reference t
            m = dms[$selected_dm]
            dist_sum = map(x -> sum(view(m, x, g)), g)
            ref_t_idx = argmin(dist_sum)

            ts = view(transitionSequence, g)

            ref_t = ts[ref_t_idx]
            t_idx = t_to_idx[ref_t] # get absolute transitionSequence index
            mtx_idx = $reordered_matrix[2][t_idx]

            push!(red_t_list, ref_t)
            red_t_to_idx[ref_t] = i
            idxes[i] = mtx_idx
        end

        redmat = view($reordered_matrix[1], idxes, idxes)

        # should just do this here and pass it down to main instead of doing it twice
        clustering = hclust(redmat, linkage=:ward, branchorder=:barjoseph)
        rm = view(redmat, clustering.order, clustering.order)

        return redmat, red_t_list, rm
    end

    red_hm_ax = Axis(window[3, 2],
        title=lift(x -> "Reduced: $(size(x[1]))", reduced),
        backgroundcolor=:transparent)

    hidedecorations!(red_hm_ax)
    deregister_interaction!(red_hm_ax, :rectanglezoom)

    red_hm = heatmap!(red_hm_ax, lift(x -> x[3], reduced),
        colorrange=lift(x -> x[3], reordered_matrix),
        colormap=DISTANCE_MATRIX_COLORMAP)

    on(go_btn.clicks) do n
        empty!(window)
        GC.gc(true)
        on_click(active_trajectory,
            screen_ref,
            Observable(reduced[][1]),
            reduced[][2],
            selected_dm[];
            init_h_cutoff=init_h_cutoff,
            kwargs...)
    end

    Colorbar(window[4, 1:2],
        limits=lift(x -> x[3], reordered_matrix),
        label="Distances",
        vertical=false,
        colormap=DISTANCE_MATRIX_COLORMAP)

    #linkaxes!(hm_ax, red_hm_ax)

    return window
end
