using Makie: clear_temporary_plots!, Orthographic, SparseArrays, apply_transform_and_model
using GLMakie: Screen
using StatsBase
using UMAP
using ImageIO
using NetworkLayout
using Observables

const MIN_NODE_SIZE = 10.0
const MAX_NODE_SIZE = 100.0

function build_selection_window(
    t_list,
    rel_t_to_idx::Dict{Tuple{Int16,Int16},Int},
    on_click,
    num_atoms,
    dm,
    volRange,
    vol_cmap,
    cluster_data::Observable{ClusterData},
    cluster_info::Observable{ClusterInfo},
    scalars,
    h_cutoff,
    h_range,
    settings_window,
    render_views,
    widgets,
    invariantRange,
    matColLabel,
    per_t_scalars,
    per_t_scalar_ranges,
    calculators;
    fig_size=(600, 800)
)

    window = Figure(size=fig_size)
    menu_bar = top_bar(window, "Overview", 2)

    init_transitions = Set{Tuple{Int,Int}}()
    selected_transitions = Observable{Set{Tuple{Int,Int}}}(init_transitions)
    hovered_transition = MaybeObservable{Tuple{Int,Int}}()

    init_clusters = Set{Set{Int}}()
    selected_clusters = Observable{Set{Set{Int}}}(init_clusters)

    # will complain about being passed "nothing" as a value if something isn't inside the set
    hovered_cluster = MaybeObservable{Set{Int}}(Set{Int}(1))

    # used to place transitions into scratchpad
    function on_transition_select(t)
        push!(selected_transitions[], t)
        notify(selected_transitions)
    end

    function on_cluster_select(c)
        push!(selected_clusters[], c)
        notify(selected_clusters)
    end

    bins = @lift begin
        return $h_range[1]:1:($h_range[2]+1)
    end

    function on_cluster_window_hover(c)
        if !isnothing(c)
            hovered_cluster[] = c
            notify(hovered_cluster)
        else
            if !isnothing(hovered_cluster[])
                hovered_cluster.val = nothing
                hovered_cluster[] = hovered_cluster[]
                notify(hovered_cluster)
            end
        end
    end

    # can be more clever
    function cw_on_up(cc)
        parent = get_parent(cluster_info[], cc[])
        if parent != cc[]
            cc[] = parent
            notify(cc)
        end
    end

    function cw_on_left(cc)
        n = get_neighbor(cluster_info[], cc[], 1)
        if n != cc[]
            cc[] = n
            notify(cc)
        end
    end

    function cw_on_right(cc)
        n = get_neighbor(cluster_info[], cc[], 2)
        if n != cc[]
            cc[] = n
            notify(cc)
        end
    end

    function cw_on_downleft(cc)
        children = get_children(cluster_info[], cc[])
        if !isnothing(children)
            lc, rc = children
            if lc != cc[]
                cc[] = lc
                notify(cc)
            end
        end
    end

    function cw_on_downright(cc)
        children = get_children(cluster_info[], cc[])
        if !isnothing(children)
            lc, rc = children
            if rc != cc[]
                cc[] = rc
                notify(cc)
            end
        end
    end


    num_open_windows = 0
    open_cluster_windows = Dict{Int,Screen}()
    function on_show_cluster_click(clusters)
        cc = Observable(clusters)
        scd = @lift begin
            ref_t = find_group_centroid($cc, cluster_data[], cluster_info[], t_list)

            ts_idx = reduce(vcat, map(x -> cluster_info[].groups[x], collect($cc)))
            ts = t_list[ts_idx]


            ts_idx_to_mtx_idx = map(x -> cluster_data[].idx_to_mtx[x], ts_idx)
            mtx_idx = sort(ts_idx_to_mtx_idx)
            mat = cluster_data[].matrix[mtx_idx, mtx_idx]

            return buildSingleClusterData(
                cluster=cc[],
                ref_t=ref_t,
                ts=ts,
                mat=mat,
                cluster_info=cluster_info[],
                rel_t_to_idx=rel_t_to_idx,
                idx_to_mtx_idx=collect(eachindex(ts_idx_to_mtx_idx)))
        end

        ds::MaybeObservable{DataInspector} = Observable(nothing)
        w = build_cluster_window(
            cc,
            scd,
            scalars,
            cluster_data[].m_extrema,
            render_views,
            widgets,
            bins,
            on_transition_select,
            hovered_transition,
            hovered_cluster,
            ds,
            calculators,
            on_window_hover=on_cluster_window_hover,
            on_cluster_select=on_cluster_select,
            on_up=cw_on_up,
            on_left=cw_on_left,
            on_right=cw_on_right,
            on_downleft=cw_on_downleft,
            on_downright=cw_on_downright,
        )
        s = GLMakie.Screen(title="Cluster $(str_limit(clusters))")
        display(s, w)

        num_open_windows += 1
        w_idx = num_open_windows
        # create inspector after render to avoid bugs
        ds[] = DataInspector(w)
        open_cluster_windows[w_idx] = s

        on(events(w).window_open) do e
            if !e
                delete!(open_cluster_windows, w_idx)
                empty!(w)
                close(s)
            end
        end
    end

    # close cluster views if clustering changes
    on(cluster_info) do c
        foreach(s -> close(s), values(open_cluster_windows))
        empty!(open_cluster_windows)
    end

    cutoff_tb = Textbox(window, validator=Float64, placeholder=string(h_cutoff[]))
    on(cutoff_tb.stored_string) do s
        # reset hovered_cluster to avoid crashing
        hovered_cluster.val = nothing
        notify(hovered_cluster)

        h_cutoff[] = parse(Float64, s)
        notify(h_cutoff)
    end

    dGrid = GridLayout()
    window[2:3, 1] = dGrid
    #colsize!(window.layout, 2, Relative(0.66))

    Label(dGrid[1, 1:2], matColLabel, font=:bold, fontsize=20)
    graph_ax = Axis(dGrid[2, 1], backgroundcolor=:transparent)
    deregister_interaction!(graph_ax, :rectanglezoom)
    hidexdecorations!(graph_ax)

    dGrid[2, 2] = vgrid!(
        cutoff_tb,
        Label(window, "Cutoff", tellwidth=false))

    hm_ax = Axis(dGrid[3, 1], backgroundcolor=:transparent)

    rowsize!(dGrid, 2, Relative(0.25))
    deregister_interaction!(hm_ax, :rectanglezoom)
    hidedecorations!(hm_ax)

    function on_dendrogram_click(clusters)
        on_show_cluster_click(clusters)
    end

    dendrogram!(graph_ax,
        cluster_info,
        hovered_cluster;
        on_click=on_dendrogram_click,
        colormap=CLUSTER_COLORS)

    hm = heatmap!(hm_ax, lift(x -> x.matrix, cluster_data))
    hm_m_events = addmouseevents!(hm_ax.scene)

    on(hm_m_events.obs) do e
        if e.type === MouseEventTypes.leftdown
            plot, _ = pick(hm_ax)
            if !isnothing(plot)
                ord = cluster_data[].mtx_to_t
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

    cluster_cmap = to_colormap(CLUSTER_COLORS)
    rendered_clusters = []
    @lift begin
        foreach(x -> delete!(parent_scene(x), x), rendered_clusters)
        for (c, ts_idx) in $(cluster_info).groups
            idx_to_mtx = $(cluster_data).idx_to_mtx
            m_idx = map(x -> idx_to_mtx[x], ts_idx)

            lo = minimum(m_idx)
            hi = maximum(m_idx)

            p = draw_bbox_pixel_space!(hm_ax.scene, lo, hi; color=cluster_cmap[mod1(c, length(cluster_cmap))])

            push!(rendered_clusters, p)
        end
    end

    function calc_cluster_bounding_box(hc, cg, idx_to_mtx, last_bBox::Maybe{Wireframe{Tuple{GeometryBasics.HyperRectangle{2,Float64}}}})::Maybe{Wireframe{Tuple{GeometryBasics.HyperRectangle{2,Float64}}}}
        if !isnothing(last_bBox)
            delete!(parent_scene(last_bBox), last_bBox)
        end

        if !isnothing(hc) && intersect(hc, Set(collect(keys(cg)))) == hc && length(hc) > 0
            ts_idx = reduce(vcat, map(x -> cg[x], collect(hc)))

            m_idx = map(x -> idx_to_mtx[x], ts_idx)

            lo = minimum(m_idx)
            hi = maximum(m_idx)

            if length(hc) != 1
                color = to_color(:grey)
            else
                color = cluster_cmap[mod1(first(collect(hc)), length(cluster_cmap))]
            end

            return draw_bbox_pixel_space!(hm_ax.scene, lo, hi; color=color, width=3)
        end
        return draw_bbox_pixel_space!(hm_ax.scene, 0, 0; width=3)
    end

    # https://github.com/MakieOrg/Makie.jl/blob/master/src/interaction/inspector.jl
    hm_last_bBox::Maybe{Wireframe{Tuple{GeometryBasics.HyperRectangle{2,Float64}}}} = nothing
    @lift begin
        res = calc_cluster_bounding_box($hovered_cluster, cluster_info[].groups, cluster_data[].idx_to_mtx, hm_last_bBox)
        hm_last_bBox = res
    end

    settings_btn = Button(window, label="Settings", halign=:right)
    screen = nothing
    on(settings_btn.clicks) do n
        # n has how many times the button's been clicked
        if isnothing(screen)
            screen = GLMakie.Screen(title="LAMDA Settings")
            display(screen, settings_window)
        else
            close(screen)
            screen = nothing
        end
    end
    menu_bar[1, 3] = settings_btn
    dGrid[3, 2] = Colorbar(window, limits=lift(x -> x.m_extrema, cluster_data))
    rowsize!(dGrid, 3, Relative(0.65))
    linkxaxes!(hm_ax, graph_ax)

    tGrid = GridLayout()
    scg = GridLayout()

    window[2:3, 2] = vgrid!(tGrid, scg)

    render_selection, scratchpad_render_menu = widgets["Render"](window)
    scalar_selection, scalar_menu = widgets["Scalar"](window)
    time, scratchpad_t_slider = widgets["Movement"](0.0, window)

    function on_scratchpad_hover(t)
        if t isa Tuple{Int,Int}
            t_idx = rel_t_to_idx[t]
            hovered_cluster[] = Set(cluster_info[].assignments[t_idx])
        else
            hovered_cluster[] = t
        end
        notify(hovered_cluster)
    end

    scratchpad!(
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
        on_hover=on_scratchpad_hover,
        hovered_cluster=hovered_cluster,
        hovered=hovered_transition,
        on_click=on_show_cluster_click)

    scg[1, 1] = scratchpad_render_menu
    scg[1, 2] = scalar_menu
    scg[2, 1:2] = scratchpad_t_slider

    return window
end
