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
        set_val(cluster_annotations[], clusters[], name, val)
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

    notes_listener = on(clusters, weak=true) do c
        if !isnothing(nw)
            close(screen)
            screen = nothing
            Makie.free(nw.scene)
            nw = nothing
        end
    end


    colors = lift(x -> x.colors, cluster_data)

    function update_colors(cutoff)
        cut_clusters = clusters_above_cutoff(clusters[],
            all_cluster_data[],
            cutoff)

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

        colors[] = map(x -> all_cluster_data[].colors[x], rel_ts_to_c)
    end

    scene_selector, render_menu = widgets["Render"](window)
    scalar_selector, scalar_menu = widgets["Scalar"](window)
    cbar, cbar_listeners = widgets["Colorbar"](window, scene_selector, scalar_selector)
    time, t_slider = widgets["Movement"](Float32(0.0), window)

    btn_centroid = Button(window, label="Show centroid")
    centroid_click_listener = on(btn_centroid.clicks, weak=true) do n
        hovered_transition[] = cluster_data[].ref_t
    end

    lm, embedding = layout_menu(window, cluster_data, all_cluster_data[])

    embedding_cleanup = embedding_view!(window[2, 1:2],
        cluster_data,
        embedding,
        scene_selector,
        scalar_selector,
        time,
        render_views,
        hovered_transition,
        hovered_cluster,
        colors,
        on_click=on_transition_select)

    EMBEDDING_HELP = "PGUP - Increase size of visualizations PGDOWN - Decrease size of visualizations ARROW UP - Show parent cluster"

    rg = hgrid!(render_menu, scalar_menu)
    lg = hgrid!(t_slider, inline_image(window, HELP_ICON, EMBEDDING_HELP))
    colsize!(lg, 1, Auto(true, 4))
    colsize!(lg, 2, Auto(false))

    window[3, 1:2] = hgrid!(
        vgrid!(rg, cbar),
        vgrid!(lg, hgrid!(lm, btn_centroid))
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

    correlation, corr_slider = widgets["CorrThreshold"](window, Float32(0.7))
    btn_centroid_to_scratchpad = Button(window, label="To scratchpad", tellwidth=false)
    centroid_grid[3, 1] = hgrid!(btn_centroid_to_scratchpad, corr_slider)

    centroid_plt_data = @lift begin
        (; ts, alignment) = $cluster_data
        empty!(centroid_scene.scene.plots)
        return render_views["SMovement"](centroid_scene.scene, ts, time, alignment, correlation)
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
    )

    hm_ax, hm = heatmap(mat_grid[3, 1],
        lift(x -> x.mat, cluster_data),
        colorrange=all_cluster_data[].m_extrema,
        colormap=DISTANCE_MATRIX_COLORMAP)

    v_listener = on(cluster_data) do _
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
        embedding_cleanup = nothing

        update_colors = nothing
        clear_listener_list(cbar_listeners)

        off(keyboard_listener)
        off(to_scratchpad_listener)
        off(hv_listener)
        off(mp_listener)
        off(v_listener)
        off(notes_listener)
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
        Observables.clear(centroid_plt_data[][4])
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

function grid_layout(items::AbstractVector{<:Any})::Vector{Point2f}
    s = Int(round(sqrt(length(items))))
    points = Vector{Point2f}(undef, length(items))
    r = 0
    for i in eachindex(items)
        x = mod1(i, s) * 1
        if x == 1
            r += 1
        end
        y = r
        points[i] = Point2f(Float32(x), Float32(y))
    end
    return points
end

function layout_menu(window::Makie.Figure,
    cluster_data::Observable{SingleClusterData},
    all_cluster_data::ClusterData)::Tuple{Makie.Menu,Observable{Vector{Point2f}}}


    opts = ["SortedGrid", "Grid", "MDS", "UMAP"]
    m = Menu(window, options=opts, default=first(opts))
    pts = @lift begin
        ms = $(m.selection)
        cd = all_cluster_data
        scd = $(cluster_data)

        ts = scd.ts
        mat = scd.mat

        if ms == "MDS"
            mds = fit(MDS, mat; distances=true, maxoutdim=2)
            points = Point2f.(eachcol(predict(mds)))
        elseif ms == "UMAP"
            if length(ts) > 2
                em = umap(mat, 2;
                    metric=:precomputed,
                    min_dist=1,
                    n_neighbors=min(length(ts) - 1, 15))
                points = Point2f.(eachcol(em))
            else
                @warn "Not enough points for UMAP layout. Using grid layout instead."
                points = grid_layout(ts)
            end
        elseif ms == "Grid"
            points = grid_layout(ts)
        elseif ms == "SortedGrid"
            children = get_children(cd, scd.cluster)
            if !isnothing(children)
                lc, rc = children
                # need to ensure points are in the same order as ts
                lt = get_transitions(cd, lc)
                lp = Dict(collect((zip(lt, grid_layout(lt)))))

                rt = get_transitions(cd, rc)
                # shift right points by the amount that the left takes up
                rp = Dict(collect((zip(rt, grid_layout(rt) .+ Point2f(Int(round(sqrt(length(lt)))) + 1, 0.0)))))

                points::Vector{Point2f} = collect(map(x -> get(lp, x, get(rp, x, nothing)), ts))

            else
                points = grid_layout(ts)
            end
        end
        return points
    end

    return m, pts
end
