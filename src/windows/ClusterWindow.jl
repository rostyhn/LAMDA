function build_cluster_window(
    clusters::Observable{Set{Int}},
    cluster_data::Observable{SingleClusterData},
    all_cluster_data::Observable{ClusterData},
    cluster_info::ClusterInfo,
    scalars,
    mat_range,
    render_views,
    widgets,
    bins,
    on_transition_select,
    hovered_transition,
    hovered_cluster,
    inspector_ref::MaybeObservable{DataInspector},
    calculators,
    cluster_annotations;
    on_cluster_select=(x, y) -> (),
    on_window_hover=(x) -> (),
    on_up=(x) -> (),
    on_left=(x) -> (),
    on_right=(x) -> (),
    on_downleft=(x) -> (),
    on_downright=(x) -> (),
    fig_size=(1920, 1080)
)
    set_theme!(UI_THEME)
    window = Figure(size=fig_size)

    # the transitions being hovered on in the dist matrix
    mat_hovered = Observable((0, 0))

    title = lift((x, y) -> str_limit(get_val(y, "titles", x), len=25), clusters, cluster_annotations)
    notes = lift((x, y) -> get_val(y, "notes", x), clusters, cluster_annotations)
    menu_bar = top_bar(window, title, 3)

    btn_notes = Button(window, label="Notes")
    menu_bar[1, 1] = btn_notes

    function update_cluster(name, val)
        set_val(cluster_annotations[], clusters[], name, val)
        notify(cluster_annotations)
    end

    screen = nothing
    nw = nothing
    on(btn_notes.clicks) do n
        if isnothing(screen)
            nw = NoteWindow(title, notes, update_cluster)
            screen = GLMakie.Screen(title="LAMDA - $(title[]) Notes")
            display(screen, nw)
        else
            close(screen)
            screen = nothing
            Makie.free(nw.scene)
            nw = nothing
        end
    end

    on(clusters) do c
        if !isnothing(nw)
            close(screen)
            screen = nothing
            Makie.free(nw.scene)
            nw = nothing
        end
    end


    ts = Observable(cluster_data[].ts)
    alignment = @lift begin
        ts.val = $(cluster_data).ts
        return $(cluster_data).alignment
    end

    hb = lift((x, y) -> !isnothing(x) && length(collect(intersect(y, x))) > 0,
        hovered_cluster,
        clusters)

    colors = lift(x -> x.colors, cluster_data)
    function update_colors(cutoff)
        cut_clusters = Ref([])
        clusters_above_cutoff(clusters[],
            all_cluster_data[],
            cluster_data[],
            cutoff,
            cut_clusters)

        rel_ts = map(x -> cluster_info.rel_t_to_idx[x], cluster_data[].ts)
        rel_ts_to_c = []
        for idx in rel_ts
            for c in cut_clusters[]
                if idx in c
                    push!(rel_ts_to_c, c)
                    break
                end
            end
        end

        colors[] = map(x -> cluster_color(all_cluster_data[], x), rel_ts_to_c)
        notify(colors)
    end

    scene_selector, render_menu = widgets["Render"](window)
    scalar_selector, scalar_menu = widgets["Scalar"](window)
    cbar = widgets["Colorbar"](window, scene_selector, scalar_selector)
    time, t_slider = widgets["Movement"](0.0, window)

    btn_centroid = Button(window, label="Show centroid")
    on(btn_centroid.clicks) do n
        hovered_transition[] = cluster_data[].ref_t
        notify(hovered_transition)
    end

    lm, embedding = layout_menu(window, cluster_data)

    embedding_view!(window[2, 1:2],
        lift((x, y) -> (x.ts, x.alignment, y), cluster_data, embedding),
        scene_selector,
        scalar_selector,
        time,
        render_views,
        hovered_transition,
        hovered_cluster,
        colors,
        cluster_info;
        highlight_borders=hb,
        on_click=on_transition_select)

    rg = hgrid!(render_menu, scalar_menu)

    window[3, 1:2] = hgrid!(
        vgrid!(rg, cbar),
        vgrid!(t_slider, hgrid!(lm, btn_centroid))
    )

    mat_grid = GridLayout()
    window[2:3, 3] = mat_grid

    centroid_grid = GridLayout()
    mat_grid[1, 1] = centroid_grid

    Label(centroid_grid[1, 1], "Group Displacement", font=:bold, tellwidth=false)
    centroid_scene = LScene(
        centroid_grid[2, 1],
        show_axis=false,
        scenekw=(backgroundcolor=EMBEDDED_SCENE_BACKGROUND, clear=true),
    )

    vals = lift(x -> x.mat, cluster_data)

    correlation, corr_slider = widgets["CorrThreshold"](window, 0.7)
    btn_centroid_to_scratchpad = Button(window, label="To scratchpad", tellwidth=false)
    centroid_grid[3, 1] = hgrid!(btn_centroid_to_scratchpad, corr_slider)
    render_views["SMovement"](centroid_scene,
        ts,
        time,
        alignment,
        correlation)

    on(btn_centroid_to_scratchpad.clicks) do n
        # save correlation to scratchpad, quick fix for now
        # will have a better solution later
        on_cluster_select(clusters[], correlation[])
    end

    dendrogram_ax = Axis(mat_grid[2, 1], tellwidth=false, tellheight=false)
    deregister_interaction!(dendrogram_ax, :rectanglezoom)
    hidedecorations!(dendrogram_ax)

    dendrogram!(dendrogram_ax,
        cluster_data,
        all_cluster_data,
        hovered_cluster,
        cluster_annotations,
        on_cutoff_line_drag=update_colors;
        cutoff_reset=true,
        colormap=CLUSTER_COLORS)

    hm_ax, hm = heatmap(mat_grid[3, 1],
        vals,
        colorrange=mat_range,
        colormap=DISTANCE_MATRIX_COLORMAP)

    on(vals) do v
        reset_limits!(hm_ax)
        center!(hm_ax.scene)
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

    lastBbox = nothing
    on(hovered_transition) do ht
        if !isnothing(lastBbox)
            delete!(parent_scene(lastBbox), lastBbox)
        end
        if ht in cluster_data[].ts
            mtx = cluster_data[].t_to_mtx[ht]
            lastBbox = draw_bbox_pixel_space!(hm_ax.scene, mtx, mtx)
        end
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

    #=on(events(window).entered_window) do entered
        if entered
            on_window_hover(clusters[])
        else
            on_window_hover(nothing)
        end
    end=#
    return window
end

function layout_menu(window, cluster_data)
    opts = ["Grid", "UMAP"]
    m = Menu(window, options=opts, default=first(opts))
    pts = @lift begin
        ms = $(m.selection)
        ts = $(cluster_data).ts
        mat = $(cluster_data).mat

        if ms == "UMAP"
            if length(ts) > 2
                em = transpose(umap(transpose(mat), 2;
                    metric=:precomputed,
                    min_dist=1,
                    n_neighbors=min(length(ts) - 1, 15)))
                points = map(x -> Point2f(x), eachrow(em))
            else
                points = Point2f[]
                for (i, t) in enumerate(ts)
                    push!(points, Point2f((i - 1), 0.0))
                end
            end
        else
            s = Int(round(sqrt(length(ts))))
            points = Point2f[]
            r = 0
            for (i, t) in enumerate(ts)
                x = mod1(i, s) * 1
                if x == 1
                    r += 1
                end
                y = r
                push!(points, Point2f(x, y))
            end
        end
        return points
    end

    return m, pts
end
