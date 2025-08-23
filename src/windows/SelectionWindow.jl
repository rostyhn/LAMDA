const MIN_NODE_SIZE = 10.0
const MAX_NODE_SIZE = 100.0

function build_selection_window(
    t_list::Vector{Transition},
    rel_t_to_idx::Dict{Transition,UInt16},
    cluster_data::ClusterData,
    cluster_info::Observable{ClusterInfo},
    h_cutoff::Observable{Float32},
    settings_window::Makie.Figure,
    render_views::Dict{String,Function},
    widgets::Dict{String,Function},
    matColLabel::String,
    calculators::Dict{String,Function},
    trajectory_name::String;
    fig_size::Tuple{Integer,Integer}=(1920, 1080)
)
    set_theme!(UI_THEME)

    window::Makie.Figure = Figure(size=fig_size)
    menu_bar = top_bar(window, "Selection Window", 2)

    init_transitions = Set{Transition}()
    selected_transitions = Observable{Set{Transition}}(init_transitions)
    hovered_transition = MaybeObservable{Transition}()

    cluster_annotations::Observable{ClusterAnnotation} = Observable(ClusterAnnotations())

    init_clusters = Set{Set{UInt16}}()
    selected_clusters = Observable{Set{Set{UInt16}}}(init_clusters)

    # will complain about being passed "nothing" as a value if something isn't inside the set
    hovered_cluster = MaybeObservable{Set{UInt16}}(Set{UInt16}())

    # used to place transitions into scratchpad
    function on_transition_select(t)
        push!(selected_transitions[], t)
        notify(selected_transitions)
    end

    c2corr = Ref(Dict{Set{UInt16},Float32}())
    function on_cluster_select(c, corr)
        push!(selected_clusters[], c)
        c2corr[][c] = corr
        notify(selected_clusters)
    end

    # can be more clever
    function cw_on_up(cc)
        parent = get_parent(cluster_data, cc[])
        if parent != cc[]
            cc[] = parent
            notify(cc)
        end
    end

    function switch_cluster(x, cc)
        cc[] = x
        notify(cc)
    end

    cluster_window_listeners = []
    function on_show_cluster_click(clusters)
        init_memory = Sys.free_memory() / 2^20
        cc = Observable(clusters)
        scd = @lift begin
            # adding a print statement makes it work...
            print("")
            ts = get_transitions(t_list, $cc)
            ref_t, alignment = calculators["Alignment"](ts)

            mat, t_to_mtx = get_local_matrix(cluster_data, ts)
            return buildSingleClusterData(
                cluster=$cc,
                ref_t=ref_t,
                ts=ts,
                mat=mat,
                alignment=alignment,
                cluster_data=cluster_data,
                cluster_info=cluster_info[],
                rel_t_to_idx=rel_t_to_idx,
                t_to_mtx=t_to_mtx)
        end

        w, cluster_cleanup = build_cluster_window(
            cc,
            scd,
            Ref(cluster_data),
            cluster_info[],
            cluster_data.m_extrema,
            render_views,
            widgets,
            on_transition_select,
            hovered_transition,
            hovered_cluster,
            cluster_annotations,
            on_cluster_select=on_cluster_select,
            on_up=cw_on_up,
            switch_cluster=switch_cluster,
        )
        s = GLMakie.Screen(title="Cluster $(str_limit(clusters))")
        display(s, w)

        # create inspector after render to avoid bugs
        ds = DataInspector(w)
        close_listener = on(events(w).window_open, weak=true) do e
            if !e
                @debug "Clear outside cluster window"
                if !isnothing(cluster_cleanup)
                    cluster_cleanup()
                    cluster_cleanup = nothing
                    Observables.clear(cc)
                    Observables.clear(scd)
                    scd = nothing
                    cc = nothing
                end
                GC.gc(true)
                ds = nothing
                empty!(w)
                Makie.free(w.scene)
                final_memory = Sys.free_memory() / 2^20
                @debug init_memory, final_memory
            end
        end
        push!(cluster_window_listeners, close_listener)
    end

    dGrid = GridLayout()
    window[2:3, 1] = dGrid

    graph_ax = Axis(dGrid[1, 1], backgroundcolor=:transparent, title=matColLabel)
    deregister_interaction!(graph_ax, :rectanglezoom)
    hidexdecorations!(graph_ax)
    hm_ax = Axis(dGrid[2, 1], backgroundcolor=:transparent)

    deregister_interaction!(hm_ax, :rectanglezoom)
    hidedecorations!(hm_ax)

    function update_cutoff(x::Float64)
        # reset hovered_cluster to avoid crashing
        hovered_cluster.val = nothing
        notify(hovered_cluster)

        h_cutoff[] = x
        notify(h_cutoff)
    end

    dendrogram!(graph_ax,
        cluster_data,
        hovered_cluster,
        cluster_annotations;
        cutoff=h_cutoff,
        on_click=on_show_cluster_click,
        on_cutoff_line_drag=update_cutoff,
        colormap=CLUSTER_COLORS)

    heatmap!(hm_ax, cluster_data.matrix,
        colorrange=cluster_data.m_extrema,
        colormap=DISTANCE_MATRIX_COLORMAP)

    hm_m_events = addmouseevents!(hm_ax.scene)
    hm_m_listener = on(hm_m_events.obs) do e
        if e.type === MouseEventTypes.leftdown
            plot, _ = pick(hm_ax)
            if !isnothing(plot)
                ord = cluster_data.mtx_to_t
                xy = mouseposition(hm_ax)
                i, j = Int.(round.(xy))
                t1 = ord[i]
                t2 = ord[j]
                push!(selected_transitions[], t1)

                if t1 != t2
                    push!(selected_transitions[], t2)
                end

                notify(selected_transitions)
            end
        end
    end

    rendered_clusters = []
    @lift begin
        foreach(x -> delete!(parent_scene(x), x), rendered_clusters)
        for (c, ts) in $(cluster_info).groups
            t_to_mtx = cluster_data.t_to_mtx
            m_idx = map(x -> t_to_mtx[x], ts)

            lo = minimum(m_idx)
            hi = maximum(m_idx)

            p = draw_bbox_pixel_space!(hm_ax.scene,
                lo,
                hi;
                color=cluster_color(cluster_info[], first(ts)))

            push!(rendered_clusters, p)
        end
    end

    function calc_cluster_bounding_box(hc, cd, t_to_mtx, last_bBox::Maybe{Wireframe{Tuple{GeometryBasics.HyperRectangle{2,Float64}}}})::Maybe{Wireframe{Tuple{GeometryBasics.HyperRectangle{2,Float64}}}}
        if !isnothing(last_bBox)
            delete!(parent_scene(last_bBox), last_bBox)
        end

        if !isnothing(hc)
            ts = get_transitions(t_list, hc)

            m_idx = map(x -> t_to_mtx[x], ts)

            lo = minimum(m_idx)
            hi = maximum(m_idx)

            color = cluster_color(cd, hc)

            return draw_bbox_pixel_space!(hm_ax.scene, lo, hi; color=color, width=3)
        end
        return draw_bbox_pixel_space!(hm_ax.scene, 0, 0; width=3)
    end

    # https://github.com/MakieOrg/Makie.jl/blob/master/src/interaction/inspector.jl
    hm_last_bBox::Wireframe{Tuple{GeometryBasics.HyperRectangle{2,Float64}}} = draw_bbox_pixel_space!(hm_ax.scene, 0, 0; width=3)
    on(hovered_cluster) do hc
        res = calc_cluster_bounding_box(hc, cluster_data, cluster_data.t_to_mtx, hm_last_bBox)
        hm_last_bBox = res
    end

    settings_btn = Button(window, label="Settings", halign=:right)
    screen = nothing
    on(settings_btn.clicks) do n
        # n has how many times the button's been clicked
        if isnothing(screen)
            screen = GLMakie.Screen(title="LAMDA - Settings")
            display(screen, settings_window)
        else
            close(screen)
            screen = nothing
        end
    end

    export_btn = Button(window, label="Export", halign=:right)
    export_menu = Menu(window, options=["Scratchpad", "All"], default="Scratchpad", tellwidth=false, halign=:right)
    menu_bar[1, 3] = export_btn
    menu_bar[1, 4] = export_menu
    menu_bar[1, 5] = settings_btn

    dGrid[3, 1] = Colorbar(window,
        vertical=false,
        colorrange=cluster_data.m_extrema,
        colormap=DISTANCE_MATRIX_COLORMAP)

    linkxaxes!(hm_ax, graph_ax)

    tGrid = GridLayout()
    scg = GridLayout()

    window[2:3, 2] = vgrid!(tGrid, scg)

    render_selection, scratchpad_render_menu = widgets["Render"](window)
    scalar_selection, scalar_menu = widgets["Scalar"](window)
    time, scratchpad_t_slider = widgets["Movement"](Float32(0.0), window)
    sc_cbar, cbar_listeners = widgets["Colorbar"](window, render_selection, scalar_selection)

    ax, scratchpad, scratchpad_cleanup = scratchpad!(
        window,
        tGrid[1, 1],
        selected_transitions,
        cluster_info,
        cluster_data,
        render_views,
        render_selection,
        scalar_selection,
        time,
        selected_clusters,
        t_list,
        rel_t_to_idx,
        calculators,
        cluster_annotations,
        c2corr;
        hovered_cluster=hovered_cluster,
        hovered=hovered_transition,
        on_click=on_show_cluster_click)

    on(export_btn.clicks) do n
        # export all clusters on screen
        ep = relative_path("export")
        if !isdir(ep)
            mkdir(ep)
        end

        if export_menu.selection[] == "All"
            export_all(trajectory_name,
                cluster_info[],
                cluster_data,
                t_list,
                cluster_annotations[],
                ep;
                overwrite=true)
        else
            dpath = get_ase_dict_path(trajectory_name)
            export_scratchpad(scratchpad, t_list, ep, dpath)
        end

        # thought we could do pdfs?
        rp = joinpath(ep, "report.png")
        save(rp, ax.scene)
    end

    scg[1, 1] = scratchpad_render_menu
    scg[1, 2] = hgrid!(scalar_menu, scratchpad_t_slider)
    scg[2, 1:2] = sc_cbar
    @debug "Finished selection window"

    cleanup = function ()
        @debug "Killing selection"
        scratchpad_cleanup()
        on_transition_select = nothing
        on_cluster_select = nothing
        cw_on_up = nothing
        switch_cluster = nothing
        on_show_cluster_click = nothing
        update_cutoff = nothing
        calc_cluster_bounding_box = nothing

        empty!(ax)
        Makie.free(ax.scene)
        clear_listener_list(cbar_listeners)
        clear_listener_list(cluster_window_listeners)
    end
    return window, cleanup
end
