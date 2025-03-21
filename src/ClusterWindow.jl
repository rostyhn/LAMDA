using Makie

function find_cluster_representatives(distances::Matrix{Float64}, ts, maximumNumber::Int=5)
    # @show size(distances)
    # @show typeof(ts)
    if length(distances[1,:]) > maximumNumber
        R = kmedoids(distances, maximumNumber)
    end
    if length(ts) > maximumNumber
        return ts[R.medoids]
    else
        return ts
    end
end

function build_cluster_window(clusters,
    ts,
    ref_t,
    idx_to_mtx_idx,
    vals,
    scalars,
    mat_range,
    render_views,
    widgets,
    bins,
    on_transition_select,
    hovered_transition,
    hovered_cluster,
    inspector_ref::MaybeObservable{DataInspector};
    on_cluster_select=(x) -> (),
    on_window_hover=(x) -> (),
    fig_size=(400, 400)
)

    window = Figure(size=fig_size)

    # the transitions being hovered on in the dist matrix
    mat_hovered = Observable((0, 0))

    scene_selector = Observable("Volume")
    scalar_selector = Observable(first(sort(collect(keys(scalars)))))

    title = "Cluster $(str_limit(clusters; len=25))"
    menu_bar = top_bar(window, title, 3)

    render_menu = Menu(window,
        options=SINGLE_TRANSITION_RENDER_OPTIONS,
        default=scene_selector[], tellwidth=false)

    on(render_menu.selection) do s
        scene_selector[] = s
        notify(scene_selector)
    end

    scalar_menu = Menu(window,
        options=sort(collect(keys(scalars))),
        default=scalar_selector[],
    )

    on(scalar_menu.selection) do s
        scalar_selector[] = s
        notify(scalar_selector)
    end

    time, t_slider = widgets["Movement"](0.0, window)
    btn_centroid = Button(window, label="Show centroid")

    window[2, 1:2] = hgrid!(
        btn_centroid,
        Label(window, "Render mode"),
        render_menu,
        scalar_menu,
        t_slider)

    sortperm!(idx_to_mtx_idx, ts)

    # transition to matrix index dict
    t_to_mtx = Dict()
    for (t, i) in zip(ts, idx_to_mtx_idx)
        t_to_mtx[t] = i
    end

    umap_graph_view!(window[3, 1:2],
        ts,
        vals,
        scene_selector,
        scalar_selector,
        time,
        render_views,
        hovered_transition,
        highlight_borders=lift(x -> !isnothing(x) && length(collect(intersect(clusters, x))) > 0, hovered_cluster),
        on_click=on_transition_select)

    mat_grid = GridLayout()
    window[2:3, 3] = mat_grid

    utri = triu!(trues(size(vals)))
    hist_vals = vec(vals[utri])

    cluster_cmap = to_colormap(CLUSTER_COLORS)
    cluster_color = to_color(:grey)
    cl = collect(clusters)
    if length(cl) == 1
        cluster_color = cluster_cmap[mod1(first(cl), length(cluster_cmap))]
    end

    centroid_grid = GridLayout()
    mat_grid[1, 1] = centroid_grid

    Label(centroid_grid[1, 1], "Cluster average", font=:bold, tellwidth=false)
    centroid_scene = LScene(
        centroid_grid[2, 1],
        show_axis=false,
        scenekw=(backgroundcolor=:black, clear=true),
    )

    render_views["SMovement"](centroid_scene, find_cluster_representatives(vals,ts), time) # precalculate distances based on ts
    btn_centroid_to_scratchpad = Button(centroid_grid[3, 1], label="To scratchpad", tellwidth=false)

    on(btn_centroid_to_scratchpad.clicks) do n
        on_cluster_select(clusters)
    end

    hist_ax = Axis(mat_grid[2, 1], title="Intra-cluster distances",
        backgroundcolor=:transparent, tellwidth=false, tellheight=false)

    deregister_interaction!(hist_ax, :rectanglezoom)
    hideydecorations!(hist_ax)

    hist!(hist_ax,
        hist_vals,
        normalization=:density,
        strokewidth=1,
        strokecolor=:black,
        color=cluster_color,
        bins=bins
    )

    hm_ax, hm = heatmap(mat_grid[3, 1], vals, colorrange=mat_range)
    hidedecorations!(hm_ax)
    deregister_interaction!(hm_ax, :rectanglezoom)

    on(events(hm_ax).mouseposition) do mp
        plot, _ = pick(hm_ax)
        if is_mouseinside(hm_ax.scene)
            if plot == hm
                xy = mouseposition(hm_ax)
                i, j = Int.(round.(xy))
                mat_hovered[] = (i, j)
            end
        else
            mat_hovered[] = (0, 0)
        end
        notify(mat_hovered)
        return Consume(false)
    end

    on(events(window).entered_window) do entered
        if entered
            on_window_hover(clusters)
        else
            on_window_hover(nothing)
        end
    end

    return window
end
