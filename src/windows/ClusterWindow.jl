function build_cluster_window(
    clusters::Observable{ClusterSet},
    cluster_data::Observable{SingleClusterData},
    all_cluster_data::Base.RefValue{ClusterData},
    render_views::Dict{String,Function},
    widgets::Dict{String,Function},
    on_transition_select::Function,
    hovered_transition::MaybeObservable{Transition},
    hovered_cluster::MaybeObservable{ClusterSet},
    cluster_annotations::Observable{ClusterAnnotation};
    on_cluster_select::Function=(x, y) -> (),
    create_window::Function=(x) -> (),
    on_up::Function=(x) -> (),
    on_left::Function=(x) -> (),
    on_right::Function=(x) -> (),
    on_downleft::Function=(x) -> (),
    on_downright::Function=(x) -> (),
    switch_cluster::Function=(x, y) -> (),
    fig_size::Tuple{Integer,Integer}=(1920, 1080)
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
        set_val(cluster_annotations[], cluster_data[].cluster, name, val)
        notify(cluster_annotations)
    end

    screen = nothing
    nw = nothing
    notes_click_listener = on(btn_notes.clicks, weak=true) do n
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

    function cleanup_notes()
        if !isnothing(nw)
            close(screen)
            screen = nothing
            Makie.free(nw.scene)
            nw = nothing
        end
    end

    colors = lift(x -> x.colors, cluster_data)
    cutoff = Observable{Float32}(0.0)
    function update_colors(cutoff_val)
        cut_clusters = clusters_above_cutoff(clusters[],
            all_cluster_data[],
            cutoff_val)

        rel_ts = Set(cluster_data[].rel_ts)
        rel_ts_to_c = []
        for idx in rel_ts
            for c in cut_clusters
                if idx in c
                    push!(rel_ts_to_c, c)
                    break
                end
            end
        end
        cutoff[] = cutoff_val
        colors[] = map(x -> all_cluster_data[].colors[x], rel_ts_to_c)
    end

    scene_selector, render_menu = widgets["Render"](window)
    scalar_selector, scalar_menu = widgets["Scalar"](window)
    invariant_selection = Observable("t1")

    cbar, cbar_listeners = widgets["Colorbar"](window, scene_selector, scalar_selector, invariant_selection)

    time, t_slider = widgets["Movement"](window)

    btn_centroid = Button(window, label="Show centroid")
    centroid_click_listener = on(btn_centroid.clicks, weak=true) do n
        hovered_transition[] = cluster_data[].ref_t
    end

    embedding = layout(cluster_data, all_cluster_data[], cutoff)
    embedding_cleanup = embedding_view!(window[2, 1:2],
        cluster_data,
        embedding,
        scene_selector,
        scalar_selector,
        invariant_selection,
        time,
        render_views,
        hovered_transition,
        hovered_cluster,
        colors,
        on_click=on_transition_select)

    EMBEDDING_HELP = "PGUP - Increase size of visualizations\nPGDOWN - Decrease size of visualizations\nCTRL + LMB - Reset axis limits, helpful if points seem to disappear\nDouble LMB on transition - Add to scratchpad\nMWHL - Zoom\nRMB + drag on empty space to pan camera\n"
    help_icon(window, window[2, 1:2], EMBEDDING_HELP)

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

    correlation, corr_slider = widgets["CorrThreshold"](window)
    btn_centroid_to_scratchpad = Button(window,
        label="To scratchpad", tellwidth=false)
    centroid_grid[3, 1] = hgrid!(btn_centroid_to_scratchpad, corr_slider)
    c_plts = Ref([])
    centroid_plt_data = @lift begin
        (; ts, alignment) = $cluster_data
        foreach(x -> delete!(centroid_scene.scene, x), c_plts[])
        h, s, v, d = render_views["SMovement"](centroid_scene.scene, ts, time, alignment, correlation)
        c_plts[] = [h, s, v]
        return d
    end

    to_scratchpad_listener = on(btn_centroid_to_scratchpad.clicks, weak=true) do n
        # save correlation to scratchpad, quick fix for now
        on_cluster_select(clusters[], correlation[])
    end

    dendrogram_ax = Axis(mat_grid[2, 1], tellwidth=false, tellheight=false)
    deregister_interaction!(dendrogram_ax, :rectanglezoom)
    hidedecorations!(dendrogram_ax)

    dendrogram!(dendrogram_ax,
        all_cluster_data[],
        hovered_cluster,
        cluster_annotations,
        on_cutoff_line_drag=update_colors;
        cutoff_reset=true,
        root=clusters,
        on_click=(x -> switch_cluster(x, clusters)),
        on_rmb=create_window
    )

    DENDROGRAM_HELP = "LMB on stem - show cluster in this window \nMiddle click on stem - show cluster in new window \nLeft click and drag on cutoff - change cluster colors in embedding view \nARROW UP - Show parent cluster"

    help_icon(window, mat_grid[2, 1], DENDROGRAM_HELP)

    hm_ax, hm = heatmap(mat_grid[3, 1],
        lift(x -> x.mat, cluster_data),
        colorrange=all_cluster_data[].m_extrema,
        colormap=DISTANCE_MATRIX_COLORMAP)

    g = GridLayout()
    g[1, 1] = scalar_menu
    g[1, 2] = t_slider

    onany(window, scene_selector) do _, x
        clear_layout(g)
        if x == "Atom"
            _, m = widgets["Scalar"](window, scalar_selector)
            _, ts = widgets["Movement"](window, time)
            g[1, 1] = m
            g[1, 2] = ts
        else
            _, m = widgets["Invariant"](window, invariant_selection)
            g[1, 1:2] = m
        end
    end

    rg = hgrid!(render_menu, g)
    window[3, 1:2] = vgrid!(rg, hgrid!(cbar, btn_centroid))

    v_listener = on(cluster_data) do cd
        cutoff[] = 0.0
        cleanup_notes()
        reset_limits!(hm_ax)
        center!(hm_ax.scene)
    end
    hidedecorations!(hm_ax)
    deregister_interaction!(hm_ax, :rectanglezoom)

    mp_listener = on(events(hm_ax).mouseposition, weak=true) do mp
        plot, _ = pick(hm_ax)
        if is_mouseinside(hm_ax.scene)
            if plot == hm
                xy = mouseposition(hm_ax)
                i, j = Int.(round.(xy))
                mat_hovered[] = (i, j)
                if i == j
                    hovered_transition[] = cluster_data[].mtx_to_t[i]
                else
                    hovered_transition[] = nothing
                end
            end
        else
            mat_hovered[] = (0, 0)
        end
        return Consume(false)
    end

    lastBbox = nothing
    hv_listener = on(hovered_transition, weak=true) do ht
        if !isnothing(lastBbox)
            delete!(parent_scene(lastBbox), lastBbox)
        end
        if ht in cluster_data[].ts
            mtx = cluster_data[].t_to_mtx[ht]
            lastBbox = draw_bbox_pixel_space!(hm_ax.scene, mtx, mtx)
        end
    end

    keyboard_listener = on(events(window).keyboardbutton) do event
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

    cluster_cleanup = function ()
        @debug "Clear inside cluster window"
        embedding_cleanup()
        cleanup_notes()
        embedding_cleanup = nothing

        update_colors = nothing
        clear_listener_list(cbar_listeners)

        off(keyboard_listener)
        off(to_scratchpad_listener)
        off(hv_listener)
        off(mp_listener)
        off(v_listener)
        off(notes_click_listener)
        off(centroid_click_listener)

        centroid_click_listener = nothing
        keyboard_listener = nothing
        to_scratchpad_listener = nothing
        hv_listener = nothing
        mp_listener = nothing
        v_listener = nothing
        notes_listener = nothing
        notes_click_listener = nothing

        Observables.clear(embedding)
        Observables.clear(correlation)
        Observables.clear(colors)
        Observables.clear(title)
        Observables.clear(notes)
        Observables.clear(centroid_plt_data[])
        Observables.clear(centroid_plt_data)

        ts = nothing
        alignment = nothing
        correlation = nothing
        vals = nothing
        colors = nothing
        movement_obs = nothing
        title = nothing
        notes = nothing

        empty!(centroid_scene)
        Makie.free(centroid_scene)
        empty!(window)
        Makie.free(window.scene)
    end

    return window, cluster_cleanup
end


function layout(
    cluster_data::Observable{SingleClusterData},
    all_cluster_data::ClusterData,
    cutoff::Observable{Float32})::Observable{Vector{Point2f}}

    return @lift begin
        cd = all_cluster_data
        scd = $cluster_data
        ts = scd.ts

        leaf_order = dfs_leaves(cd, scd.cluster)
        t_order = reduce(vcat, get_transitions.(Ref(cd), leaf_order))

        # could potentially use sqrt_int to build a perfect rect
        # but may be a.) lopsided and b.) the library says it may be buggy
        s = Int(ceil(sqrt(length(ts))))
        H = gilbertindices((s, s))

        points = Dict()
        for (i, t) in enumerate(t_order)
            points[t] = Point2f(Tuple(H[i])...)
        end

        return collect(map(x -> points[x], ts))
    end
end
