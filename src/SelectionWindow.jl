using Makie: clear_temporary_plots!, Orthographic, SparseArrays
using GLMakie: Screen
using StatsBase
using UMAP

const cluster_colors = :tab20

function build_selection_window(fig_size,
    t_list,
    t_to_idx,
    on_click,
    num_atoms,
    dm,
    volRange,
    vol_cmap,
    clustering,
    scalars,
    h_cutoff,
    cluster_groups,
    h_range,
    settings_window,
    cluster_assignments,
    render_views,
    widgets,
    invariantRange,
    cluster_representatives,
)

    window = Figure(size=fig_size)

    # reorders distance matrix according to clustering
    reordered_matrix = @lift begin
        m = $dm
        rm = zeros(size(m))

        # gets the correct idx 
        idx_to_mtx = zeros(Int, size(m)[1])
        for (i, r) in enumerate($clustering.order)
            rm[i, :] .= m[r, :][$clustering.order]
            idx_to_mtx[r] = i
        end

        # get minimum and maximum of entire matrix for cmap
        fl = vec(m)
        return rm, idx_to_mtx, (minimum(fl), maximum(fl))
    end
    # can't get it to align left
    # title =Label(window[1, 1], "TransVis", justification=:left, fontsize=30, tellwidth=false)

    open_cluster_windows = Dict{Set{Int},Screen}()
    function on_show_cluster_click(clusters)
        if !(clusters in keys(open_cluster_windows))
            ts_idx = reduce(vcat, map(x -> cluster_groups[][x], collect(clusters)))
            ts = t_list[ts_idx]

            ts_idx_to_mtx_idx = map(x -> reordered_matrix[][2][x], ts_idx)
            mtx_idx = sort(ts_idx_to_mtx_idx)
            mat = reordered_matrix[][1]
            vals = mat[mtx_idx, mtx_idx]

            # want to update volume data in case user messes with volume params
            # but we keep atom positions consistent with the alignment that existed at the time of creation
            w = build_cluster_window(clusters,
                ts,
                collect(eachindex(ts_idx_to_mtx_idx)),
                vals,
                scalars,
                t_to_idx,
                reordered_matrix[][3],
                render_views)
            s = GLMakie.Screen(title="Cluster $(str_limit(clusters))")
            display(s, w)

            open_cluster_windows[clusters] = s
        end
    end

    # close cluster views if clustering changes
    on(cluster_groups) do c
        foreach(s -> close(s), values(open_cluster_windows))
    end

    # will complain about being passed "nothing" as a value if something isn't inside the set
    l_hovered_cluster = Observable(Set{Int}(1))
    r_hovered_cluster = Observable(Set{Int}(1))

    cGrid = GridLayout()
    window[1, 1] = cGrid

    setup_cluster_view!(window,
        cGrid,
        (1, 1),
        l_hovered_cluster,
        cluster_groups,
        reordered_matrix,
        on_show_cluster_click
    )

    setup_cluster_view!(window,
        cGrid,
        (1, 2),
        r_hovered_cluster,
        cluster_groups,
        reordered_matrix,
        on_show_cluster_click
    )

    tGrid = GridLayout()
    window[2, 1] = tGrid

    # contains actual matrix index, the transition idx, the tuple itself and the cluster assignment
    hovered_transitions = Observable((1, 1))
    hovered_info = @lift begin
        l, r = $hovered_transitions
        function build_info(idx)
            t_idx = $clustering.order[idx]
            cluster = $cluster_assignments[t_idx]
            rep = $cluster_representatives[cluster]
            return (cluster, t_to_idx[rep], rep, t_idx)
        end

        li = build_info(l)
        ri = build_info(r)

        l_hovered_cluster[] = Set{Int}(li[1])
        notify(l_hovered_cluster)

        r_hovered_cluster[] = Set{Int}(ri[1])
        notify(r_hovered_cluster)

        return li, ri
    end

    setup_transition_view!(window, tGrid, (1, 1), lift(x -> x[1], hovered_info), vol_cmap, volRange, on_click, render_views, widgets, invariantRange)

    setup_transition_view!(window, tGrid, (1, 2), lift(x -> x[2], hovered_info), vol_cmap, volRange, on_click, render_views, widgets, invariantRange)

    cutoff_tb = Textbox(window, validator=Float64, placeholder=string(h_cutoff[]), tellwidth=false)
    on(cutoff_tb.stored_string) do s
        # reset hovered_cluster to avoid crashing
        l_hovered_cluster[] = Set{Int}(1)
        notify(l_hovered_cluster)

        r_hovered_cluster[] = Set{Int}(1)
        notify(r_hovered_cluster)

        h_cutoff[] = parse(Float64, s)
        notify(h_cutoff)
    end

    dGrid = GridLayout()
    window[1:2, 2] = dGrid

    dGrid[1, 1] = hgrid!(
        Label(window, "Cluster cutoff value", tellwidth=false),
        cutoff_tb)

    graph_ax = Axis(dGrid[2, 1], backgroundcolor=:transparent)
    deregister_interaction!(graph_ax, :rectanglezoom)
    hidexdecorations!(graph_ax)

    hm_ax, hm = heatmap(dGrid[3, 1], lift(x -> x[1], reordered_matrix))

    hidedecorations!(hm_ax)
    deregister_interaction!(hm_ax, :rectanglezoom)

    on(events(hm_ax).mouseposition) do mp
        plot, _ = pick(hm_ax)
        if is_mouseinside(hm_ax.scene)
            if plot == hm
                xy = mouseposition(hm_ax)
                i, j = Int.(round.(xy))
                hovered_transitions[] = (i, j)
                notify(hovered_info)
            end
        end
        return Consume(false)
    end

    rendered_clusters = []
    @lift begin
        foreach(x -> delete!(parent_scene(x), x), rendered_clusters)
        cmap = to_colormap(cluster_colors)
        for (c, ts_idx) in $cluster_groups
            idx_to_mtx = $reordered_matrix[2]
            m_idx = map(x -> idx_to_mtx[x], ts_idx)

            lo = minimum(m_idx)
            hi = maximum(m_idx)

            p = draw_bbox_pixel_space!(hm_ax.scene, lo, hi; color=cmap[mod1(c, length(cmap))])

            push!(rendered_clusters, p)
        end
    end

    function calc_cluster_bounding_box(hc, cg, rm, last_bBox)
        if !isnothing(last_bBox)
            delete!(parent_scene(last_bBox), last_bBox)
        end

        if length(hc) > 0
            ts_idx = reduce(vcat, map(x -> cg[x], collect(hc)))
            idx_to_mtx = rm

            m_idx = map(x -> idx_to_mtx[x], ts_idx)

            lo = minimum(m_idx)
            hi = maximum(m_idx)

            return draw_bbox_pixel_space!(hm_ax.scene, lo, hi)
        end
    end

    # https://github.com/MakieOrg/Makie.jl/blob/master/src/interaction/inspector.jl
    l_last_bBox = nothing
    r_last_bBox = nothing
    @lift begin
        l_last_bBox = calc_cluster_bounding_box($l_hovered_cluster, $cluster_groups, $reordered_matrix[2], l_last_bBox)
        r_last_bBox = calc_cluster_bounding_box($r_hovered_cluster, $cluster_groups, $reordered_matrix[2], r_last_bBox)
    end

    function on_dendrogram_hover(c)
        #=if !isempty(c)
            hovered_cluster[] = c
            notify(hovered_cluster)
        end=#
    end

    dendrogram!(graph_ax, clustering, h_cutoff, h_range; hover_callbackfn=on_dendrogram_hover, colormap=cluster_colors)
    linkxaxes!(graph_ax, hm_ax)

    settings_btn = Button(window, label="Settings")
    screen = nothing
    on(settings_btn.clicks) do n
        # n has how many times the button's been clicked
        if isnothing(screen)
            screen = GLMakie.Screen(title="TransVis Settings")
            display(screen, settings_window)
        else
            close(screen)
            screen = nothing
        end
    end

    dGrid[4, 1] = hgrid!(
        Colorbar(window, limits=lift(x -> x[3], reordered_matrix), vertical=false, size=16),
        settings_btn
    )

    return window
end

function setup_transition_view!(
    fig,
    parentGrid,
    loc,
    hovered,
    vol_cmap,
    volumeRange,
    on_click,
    render_views,
    widgets,
    invariantRange
)

    cluster_idx = lift(x -> x[1], hovered)
    t_idx = lift(x -> x[2], hovered)
    t = lift(x -> x[3], hovered)

    rootScene = LScene(
        fig,
        show_axis=false,
        scenekw=(backgroundcolor=:black, clear=true),
    )

    inspector = DataInspector(rootScene)

    sel = Observable("Volume")

    m = Menu(fig,
        options=collect(keys(render_views)),
        default=sel[])

    on(m.selection) do cw
        sel[] = cw
        notify(sel)
    end

    btn = Button(fig, label="Show")
    on(btn.clicks) do n
        on_click(t[], () -> ())
    end

    l = Label(fig, lift((x, y) -> string("Cluster $(y) - $(x)"), t, cluster_idx), tellwidth=false)

    i, j = loc
    g = vgrid!(rootScene, hgrid!(l, m, btn))
    parentGrid[i, j] = g

    function choose_scene(selection)
        if selection == "Volume"
            gg = GridLayout(g[end+1, :])

            Colorbar(gg[1, :],
                colorrange=volumeRange,
                vertical=false,
                colormap=vol_cmap,
                tellwidth=false)

            render_views[selection](rootScene, t_idx, t)
            return [], [gg]
        else
            if selection == "Superquadric"
                gg = GridLayout(g[end+1, :])
                Colorbar(gg[1, :],
                    colorrange=invariantRange,
                    vertical=false,
                    colormap=vol_cmap,
                    tellwidth=false)
                il, is = render_views[selection](rootScene, inspector, t)
                return il, [gg]
            elseif selection == "Atom"
                gg, time, scalar_vals = widgets["Atom"](0.0, g)
                render_views[selection](rootScene, t, scalar_vals, time)
                return [], [gg]
            else
                gg = GridLayout(g[end+1, :])
                time, slider = widgets["Movement"](0.0, gg)
                render_views[selection](rootScene, cluster_idx, time)
                return [], [gg]
            end
        end
    end

    scene_switcher(rootScene, g, sel, choose_scene)

    return rootScene
end


function setup_cluster_view!(fig,
    parentGrid,
    loc,
    clusters,
    cluster_groups,
    reordered_matrix,
    on_cluster_button_click
)
    i, j = loc
    cluster_grid = GridLayout()
    parentGrid[i, j] = cluster_grid

    cmap = to_colormap(cluster_colors)
    cluster_grid[1, 1] = Label(fig,
        lift(x -> "Cluster $(str_limit(x))", clusters),
        tellwidth=false)

    show_cluster_btn = Button(cluster_grid[1, 2], label="Show")

    on(show_cluster_btn.clicks) do n
        on_cluster_button_click(clusters[])
    end

    hist_values = @lift begin
        d = []
        for c_idx in collect($clusters)
            ts_idx = cluster_groups[][c_idx]
            mtx_idx = map(x -> reordered_matrix[][2][x], ts_idx)
            mat = reordered_matrix[][1]
            vals = mat[mtx_idx, mtx_idx]
            utri = triu!(trues(size(vals)))
            push!(d, vec(vals[utri]))
        end

        return reduce(vcat, d)
    end

    hist_ax = Axis(cluster_grid[2, 1:2], title="Intra-cluster distances",
        backgroundcolor=:transparent, tellwidth=false, tellheight=false)

    # hide y labels because otherwise the width of each column gets adjusted
    hideydecorations!(hist_ax)

    # TODO: copy over code for custom implementation 
    # https://github.com/MakieOrg/Makie.jl/blob/master/src/stats/hist.jl 
    # unfortunately bar_labels doesn't work
    color = @lift begin
        cl = collect($clusters)
        if length(cl) != 1
            return to_color(:grey)
        else
            return cmap[mod1(first(cl), length(cmap))]
        end
    end

    hist!(hist_ax,
        hist_values,
        normalization=:density,
        strokewidth=1,
        strokecolor=:black,
        color=color
    )

    on(hist_values) do hv
        reset_limits!(hist_ax)
    end

    return hist_ax
end
