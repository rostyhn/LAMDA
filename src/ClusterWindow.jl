using Makie

function build_cluster_window(
    clusters::Observable{Set{Int}},
    cluster_data::Observable{SingleClusterData},
    scalars,
    mat_range,
    render_views,
    widgets,
    bins,
    on_transition_select,
    hovered_transition,
    hovered_cluster,
    inspector_ref::MaybeObservable{DataInspector},
    calculators;
    on_cluster_select=(x) -> (),
    on_window_hover=(x) -> (),
    on_up=(x) -> (),
    on_left=(x) -> (),
    on_right=(x) -> (),
    on_downleft=(x) -> (),
    on_downright=(x) -> (),
    fig_size=(400, 400)
)

    window = Figure(size=fig_size)

    # the transitions being hovered on in the dist matrix
    mat_hovered = Observable((0, 0))

    scene_selector = Observable("Volume")
    scalar_selector = Observable(first(sort(collect(keys(scalars)))))

    title = lift(x -> "Cluster " * str_limit(x; len=25), clusters)
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

    alignment = @lift begin
        return calculators["Alignment"]($(cluster_data).ts)
    end

    window[2, 1:2] = hgrid!(
        btn_centroid,
        Label(window, "Render mode"),
        render_menu,
        scalar_menu,
        t_slider)

    hb = lift((x, y) -> !isnothing(x) && length(collect(intersect(y, x))) > 0,
        hovered_cluster,
        clusters)

    umap_graph_view!(window[3, 1:2],
        lift(x -> (x.ts, x.mat, x.colors), cluster_data),
        scene_selector,
        scalar_selector,
        time,
        render_views,
        hovered_transition,
        alignment;
        highlight_borders=hb,
        on_click=on_transition_select)

    mat_grid = GridLayout()
    window[2:3, 3] = mat_grid

    #=hist_vals = @lift begin
        utri = triu!(trues(size(vals)))
        return vec(vals[utri])
    end

    cluster_cmap = to_colormap(CLUSTER_COLORS)
    cluster_color = to_color(:grey)
    cl = collect(clusters)
    if length(cl) == 1
        cluster_color = cluster_cmap[mod1(first(cl), length(cluster_cmap))]
    end=#

    centroid_grid = GridLayout()
    mat_grid[1, 1] = centroid_grid

    Label(centroid_grid[1, 1], "Cluster average", font=:bold, tellwidth=false)
    centroid_scene = LScene(
        centroid_grid[2, 1],
        show_axis=false,
        scenekw=(backgroundcolor=:black, clear=true),
    )

    render_views["SMovement"](centroid_scene, lift(x -> x.ts, cluster_data), time, alignment)
    btn_centroid_to_scratchpad = Button(centroid_grid[3, 1], label="To scratchpad", tellwidth=false)

    on(btn_centroid_to_scratchpad.clicks) do n
        on_cluster_select(clusters[])
    end

    #=hist_ax = Axis(mat_grid[2, 1], title="Intra-cluster distances",
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
    )=#

    vals = lift(x -> x.mat, cluster_data)
    hm_ax, hm = heatmap(mat_grid[2, 1], vals, colorrange=mat_range)
    on(vals) do v
        reset_limits!(hm_ax)
    end
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

    on(events(window).keyboardbutton) do event
        if ispressed(window, Exclusively(LEFT_KEY))
            on_left(clusters)
        elseif ispressed(window, Exclusively(RIGHT_KEY))
            on_right(clusters)
        elseif ispressed(window, Exclusively(UP_KEY))
            on_up(clusters)
        elseif ispressed(window, Exclusively(LEFT_DOWN))
            on_downleft(clusters)
        elseif ispressed(window, Exclusively(RIGHT_DOWN))
            on_downright(clusters)
        end
    end

    on(events(window).entered_window) do entered
        if entered
            on_window_hover(clusters[])
        else
            on_window_hover(nothing)
        end
    end
    return window
end
