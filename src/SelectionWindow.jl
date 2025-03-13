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
    render_views,
    widgets,
    invariantRange,
    cluster_representatives,
    matColLabel,
    per_t_scalars,
    per_t_scalar_ranges
)

    window = Figure(size=fig_size)
    menu_bar = top_bar(window, "Overview", 2)

    # reorders distance matrix according to clustering
    reordered_matrix = @lift begin
        m = $dm
        rm = zeros(size(m))

        # gets the correct idx 
        idx_to_mtx = zeros(Int, size(m)[1])
        t_to_mtx = Dict()
        for (i, r) in enumerate($clustering.order)
            rm[i, :] .= m[r, :][$clustering.order]
            idx_to_mtx[r] = i
            t_to_mtx[t_list[r]] = i
        end

        # get minimum and maximum of entire matrix for cmap
        fl = vec(m)
        return rm, idx_to_mtx, (minimum(fl), maximum(fl)), t_to_mtx
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
            w = build_cluster_window(
                clusters,
                ts,
                collect(eachindex(ts_idx_to_mtx_idx)),
                vals,
                scalars,
                t_to_idx,
                reordered_matrix[][3],
                render_views,
                widgets
            )
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

    tGrid = GridLayout()
    window[2:3, 1] = tGrid

    bins = @lift begin
        return $h_range[1]:1:($h_range[2]+1)
    end

    setup_cluster_view!(window,
        tGrid,
        (1, 1),
        l_hovered_cluster,
        cluster_groups,
        reordered_matrix,
        on_show_cluster_click,
        bins
    )

    setup_cluster_view!(window,
        tGrid,
        (1, 2),
        r_hovered_cluster,
        cluster_groups,
        reordered_matrix,
        on_show_cluster_click,
        bins
    )

    @show length(clustering[].order)

    function build_info(clusters, cluster_reps, rm, clustering)
        # find centroid between all clusters 
        dm = rm[1]
        t_to_mtx = rm[4]
        cluster_list = collect(clusters)

        reps = map(x -> cluster_reps[x], cluster_list)
        mtx_idx = map(x -> t_to_mtx[x], reps)

        dist_sum = map(x -> sum(dm[x, :][mtx_idx]), mtx_idx)
        ref_t_idx = mtx_idx[argmin(dist_sum)]

        f_rep = t_list[clustering.order[ref_t_idx]]
        return (clusters, t_to_idx[f_rep], f_rep)
    end

    l_info = lift((x, y, z, w) -> build_info(x, y, z, w), l_hovered_cluster, cluster_representatives, reordered_matrix, clustering)
    r_info = lift((x, y, z, w) -> build_info(x, y, z, w), r_hovered_cluster, cluster_representatives, reordered_matrix, clustering)
    setup_transition_view!(window, tGrid, (2, 1), l_info, vol_cmap, volRange, on_click, render_views, widgets, invariantRange)
    setup_transition_view!(window, tGrid, (2, 2), r_info, vol_cmap, volRange, on_click, render_views, widgets, invariantRange)

    cutoff_tb = Textbox(window, validator=Float64, placeholder=string(h_cutoff[]))
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
    window[2:3, 2] = dGrid
    colsize!(window.layout, 2, Relative(0.66))

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

    function on_dendrogram_click(clusters, keyboard)
        if Keyboard.a in keyboard
            l_hovered_cluster[] = clusters
            notify(l_hovered_cluster)
        elseif Keyboard.d in keyboard
            r_hovered_cluster[] = clusters
            notify(r_hovered_cluster)
        end
    end

    dendrogram!(graph_ax, clustering, h_cutoff, h_range; on_click=on_dendrogram_click, colormap=cluster_colors)
    heatmap!(hm_ax, lift(x -> x[1], reordered_matrix))

    cluster_cmap = to_colormap(cluster_colors)
    rendered_clusters = []
    @lift begin
        foreach(x -> delete!(parent_scene(x), x), rendered_clusters)
        for (c, ts_idx) in $cluster_groups
            idx_to_mtx = $reordered_matrix[2]
            m_idx = map(x -> idx_to_mtx[x], ts_idx)

            lo = minimum(m_idx)
            hi = maximum(m_idx)

            p = draw_bbox_pixel_space!(hm_ax.scene, lo, hi; color=cluster_cmap[mod1(c, length(cluster_cmap))])

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

            if length(hc) != 1
                color = to_color(:grey)
            else
                color = cluster_cmap[mod1(first(collect(hc)), length(cluster_cmap))]
            end

            return draw_bbox_pixel_space!(hm_ax.scene, lo, hi; color=color, width=3)
        end
    end

    # https://github.com/MakieOrg/Makie.jl/blob/master/src/interaction/inspector.jl
    l_last_bBox = nothing
    r_last_bBox = nothing
    @lift begin
        l_last_bBox = calc_cluster_bounding_box($l_hovered_cluster, $cluster_groups, $reordered_matrix[2], l_last_bBox)
        r_last_bBox = calc_cluster_bounding_box($r_hovered_cluster, $cluster_groups, $reordered_matrix[2], r_last_bBox)
    end

    settings_btn = Button(window, label="Settings", halign=:right)
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
    menu_bar[1, 3] = settings_btn
    dGrid[3, 2] = Colorbar(window, limits=lift(x -> x[3], reordered_matrix))
    rowsize!(dGrid, 3, Relative(0.65))

    band_sel = Observable(first(sort(collect(keys(per_t_scalars)))))

    x_vals = lift(x -> eachindex(x.order), clustering)
    colors = lift((x, z) -> map(y -> per_t_scalars[z][t_list[y]], x.order), clustering, band_sel)
    colorrange = lift(x -> per_t_scalar_ranges[x], band_sel)

    band_ax = Axis(dGrid[4, 1], backgroundcolor=:transparent, title="Per-transition scalar values")
    band_plot = vlines!(band_ax,
        x_vals,
        color=colors,
        colorrange=colorrange,
        linewidth=3,
        inspector_label=(plot, idx, pos) -> "$(plot.color[][idx])")

    hidedecorations!(band_ax)
    deregister_interaction!(band_ax, :rectanglezoom)

    band_menu = Menu(window, options=sort(collect(keys(per_t_scalars))), default=band_sel[])
    band_cbar = Colorbar(window, band_plot; vertical=false)
    dGrid[5, 1:2] = hgrid!(band_menu, band_cbar)

    on(band_menu.selection) do s
        band_sel[] = s
        notify(band_sel)
    end

    linkxaxes!(hm_ax, graph_ax, band_ax)

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

    # prevents camera from moving around
    Camera3D(parent_scene(rootScene); left_key=false, right_key=false)
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

    l = Label(fig, lift(x -> string("$(x)"), t), tellwidth=false)

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
                gg, time, scalar_vals = widgets["Atom"](0.0, fig, g)
                render_views[selection](rootScene, t, scalar_vals, time)
                return [], [gg]
            else
                gg = GridLayout(g[end+1, :])
                time, slider = widgets["Movement"](0.0, fig)
                gg[1, 1:2] = slider
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
    on_cluster_button_click,
    bins,
)
    i, j = loc
    cluster_grid = GridLayout()
    parentGrid[i, j] = cluster_grid

    cmap = to_colormap(cluster_colors)
    cluster_grid[1, 1] = Label(fig,
        lift(x -> "Cluster $(str_limit(x;len=25))", clusters),
        halign=:left,
        font=:bold,
        tellwidth=false)

    show_cluster_btn = Button(cluster_grid[1, 2], label="Show")

    on(show_cluster_btn.clicks) do n
        on_cluster_button_click(clusters[])
    end

    hist_values = @lift begin
        ts_idx = reduce(vcat, (map(x -> cluster_groups[][x], collect($clusters))))
        mtx_idx = sort(map(x -> reordered_matrix[][2][x], ts_idx))
        mat = reordered_matrix[][1]
        vals = mat[mtx_idx, mtx_idx]
        utri = triu!(trues(size(vals)))
        return vec(vals[utri])
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
        color=color,
        bins=bins
    )

    on(hist_values) do hv
        reset_limits!(hist_ax)
    end

    return hist_ax
end
