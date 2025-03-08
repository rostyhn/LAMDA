using Makie: clear_temporary_plots!, Orthographic, SparseArrays
using GLMakie: Screen
using StatsBase
using UMAP

const cluster_colors = :tab20

function build_selection_window(fig_size,
    data,
    t_list,
    t_to_idx,
    on_click,
    num_atoms,
    alignedPositionsMatrices,
    transitionKDTree,
    dms,
    volData,
    sampleRanges,
    volRange,
    vol_cmap,
    clustering,
    selected_dm,
    scalars,
    h_cutoff,
    cluster_groups,
    alignment_rotations,
    h_range,
    scalar_range,
    settings_window,
    cluster_assignments
)

    window = Figure(size=fig_size)

    # reorders distance matrix according to clustering
    reordered_matrix = @lift begin
        m = dms[$selected_dm]
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
                alignedPositionsMatrices,
                alignment_rotations,
                volData,
                sampleRanges,
                scalars,
                scalar_range,
                t_to_idx,
                vol_cmap,
                volRange
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
    hovered_cluster = Observable(Set{Int}(1))

    cGrid = GridLayout()
    window[1, 1] = cGrid

    setup_cluster_view(window,
        cGrid,
        (1, 1:2),
        hovered_cluster,
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
            t_idx = clustering[].order[idx]
            return (idx, t_idx, t_list[t_idx])
        end

        li = build_info(l)
        ri = build_info(r)

        cl = $cluster_assignments[li[2]]
        cr = $cluster_assignments[ri[2]]
        if cl == cr
            hovered_cluster[] = Set{Int}(cl)
            notify(hovered_cluster)
        end

        return build_info(l), build_info(r)
    end

    ltv = setup_transition_view(window, tGrid, (1, 1), alignedPositionsMatrices, lift(x -> x[1], hovered_info), scalars, sampleRanges, volData, vol_cmap, volRange, alignment_rotations, on_click, scalar_range)

    rtv = setup_transition_view(window, tGrid, (1, 2), alignedPositionsMatrices, lift(x -> x[2], hovered_info), scalars, sampleRanges, volData, vol_cmap, volRange, alignment_rotations, on_click, scalar_range)

    link_cameras_lscenes([ltv, rtv])

    graph_ax = Axis(window[1, 2], backgroundcolor=:transparent)
    hidexdecorations!(graph_ax)

    hm_ax, hm = heatmap(window[2, 2], lift(x -> x[1], reordered_matrix))

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

    # https://github.com/MakieOrg/Makie.jl/blob/master/src/interaction/inspector.jl
    last_bBox = nothing
    @lift begin
        if !isnothing(last_bBox)
            delete!(parent_scene(last_bBox), last_bBox)
        end
        if length($hovered_cluster) > 0
            ts_idx = reduce(vcat, map(x -> $cluster_groups[x], collect($hovered_cluster)))
            idx_to_mtx = $reordered_matrix[2]

            m_idx = map(x -> idx_to_mtx[x], ts_idx)

            lo = minimum(m_idx)
            hi = maximum(m_idx)

            last_bBox = draw_bbox_pixel_space!(hm_ax.scene, lo, hi)
        end
    end

    function on_dendrogram_hover(c)
        if !isempty(c)
            hovered_cluster[] = c
            notify(hovered_cluster)
        end
    end

    dendrogram!(graph_ax, clustering, h_cutoff, h_range; hover_callbackfn=on_dendrogram_hover, colormap=cluster_colors)

    cutoff_slider = Slider(window, range=lift(x -> x[1]:0.01:x[2], h_range), startvalue=h_cutoff[], update_while_dragging=false)
    on(cutoff_slider.value) do x
        # reset hovered_cluster to avoid crashing
        hovered_cluster[] = Set{Int}(1)
        notify(hovered_cluster)

        h_cutoff[] = x
        notify(h_cutoff)
    end

    linkxaxes!(graph_ax, hm_ax)

    window[3, 1] = hgrid!(Label(window, "Cluster cutoff value"),
        cutoff_slider,
        Label(window, lift(x -> string(round(x; sigdigits=3)), h_cutoff)))

    dm_menu = Menu(window, options=collect(keys(dms)), default=selected_dm[])
    on(dm_menu.selection) do val
        selected_dm[] = val
    end

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

    window[3, 2] = hgrid!(Label(window, "Distance matrix"),
        dm_menu,
        Colorbar(window, limits=lift(x -> x[3], reordered_matrix), vertical=false, size=16),
        settings_btn
    )

    return window
end


# should be in its own function
function setup_transition_view(
    fig,
    parentGrid,
    loc,
    ap,
    hovered,
    scalars,
    sampleRanges,
    volData,
    vol_cmap,
    volumeRange,
    alignment_rotations,
    on_click,
    scalar_range,
)
    t_idx = lift(x -> x[2], hovered)
    t = lift(x -> x[3], hovered)

    rootScene = LScene(
        fig,
        show_axis=false,
        scenekw=(backgroundcolor=:black, clear=true),
    )

    DataInspector(rootScene)

    m = Menu(fig,
        options=["Volume",
            "Initial State",
            "Final State"],
        default="Volume")

    sel = Observable("Volume")
    on(m.selection) do cw
        sel[] = cw
        notify(sel)
    end

    btn = Button(fig, label="Show")
    on(btn.clicks) do n
        on_click(t_idx[], () -> ())
    end

    l = Label(fig, lift(x -> string(x), t), tellwidth=false)

    i, j = loc
    g = vgrid!(rootScene, hgrid!(l, m, btn))
    parentGrid[i, j] = g

    opts = sort(collect(keys(scalars)))

    atom_cmap = resample_cmap(:reds, 147, alpha=range(; start=0.01, stop=1.0, length=147))
    function atom_widgets(init_time, transition)
        gg = GridLayout(g[end+1, :])

        opts = sort(collect(keys(scalars)))

        #TODO: add labels to t_slider
        time = Observable(init_time)
        t_slider = Slider(gg[1, 1:2], range=0.0:0.05:1.0, startvalue=init_time)
        on(t_slider.value) do x
            time[] = x
        end

        # sadly, GLmakie is not thread-safe, so can't make a play button

        scalar_vals = Observable(scalars[first(opts)][transition[]])

        m = Menu(gg[2, 1], options=opts)
        on(m.selection) do ms
            scalar_vals[] = scalars[ms][transition[]]
        end

        Colorbar(gg[2, 2], colorrange=scalar_range, vertical=false, colormap=atom_cmap, tellwidth=false)

        return gg, time, scalar_vals
    end

    function choose_scene(selection)
        if selection == "Volume"
            gg = GridLayout(g[end+1, :])

            Colorbar(gg[1, :],
                colorrange=volumeRange,
                vertical=false,
                colormap=vol_cmap,
                tellwidth=false)

            vd = lift((x, y, z) ->
                    reshape(x[:, y], (length(z[1]), length(z[2]), length(z[3]))), volData, t_idx, sampleRanges)

            volume_view!(rootScene, vd, sampleRanges, vol_cmap, volumeRange; rotation=lift((x, y) -> x[y], alignment_rotations, t))
            return [], [gg]
        else
            t_ap = @lift begin
                init = ap[$hovered[3]][1] * $alignment_rotations[$hovered[3]]
                final = ap[$hovered[3]][2] * $alignment_rotations[$hovered[3]]
                return (init, final)
            end

            if selection == "Initial State"
                gg, time, scalar_vals = atom_widgets(0.0, t)
            else
                gg, time, scalar_vals = atom_widgets(1.0, t)
            end

            simple_atom_view!(rootScene, t_ap, scalar_vals, scalar_range, atom_cmap, time)
            return [], [gg]
        end
    end

    scene_switcher(rootScene, g, sel, choose_scene)

    return rootScene
end


function setup_cluster_view(fig,
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

    hist_ax = Axis(cluster_grid[2, 1:2],
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
