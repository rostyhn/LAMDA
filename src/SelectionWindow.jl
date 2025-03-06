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

    open_cluster_windows = Dict{Int,Screen}()
    function on_show_cluster_click(c_idx)
        if !(c_idx in keys(open_cluster_windows))
            ts_idx = cluster_groups[][c_idx]
            ts = t_list[ts_idx]

            mtx_idx = map(x -> reordered_matrix[][2][x], ts_idx)
            mat = reordered_matrix[][1]
            vals = mat[mtx_idx, mtx_idx]

            # want to update volume data in case user messes with volume params
            # but we keep atom positions consistent with the alignment that existed at the time of creation
            rel_t_idx = collect(eachindex(ts))

            w = build_cluster_window(c_idx, ts, rel_t_idx, vals,
                alignedPositionsMatrices,
                alignment_rotations,
                volData,
                sampleRanges,
                scalars,
                scalar_range, t_to_idx, vol_cmap, volRange
            )
            s = GLMakie.Screen(title="Cluster $(c_idx)")
            display(s, w)

            open_cluster_windows[c_idx] = s
        end
    end

    # close cluster views if clustering changes
    on(cluster_groups) do c
        foreach(s -> close(s), values(open_cluster_windows))
    end

    # grid for the transition views
    tGrid = GridLayout()
    window[1:2, 1] = tGrid

    hl = Observable(1)

    # contains actual matrix index, the transition idx, the tuple itself and the cluster assignment
    hl_info = @lift begin
        t_idx = $clustering.order[$hl]
        return ($hl, t_idx, t_list[t_idx], $cluster_assignments[t_idx])
    end

    ltv = setup_transition_view(window, tGrid, (1, 1), alignedPositionsMatrices, hl_info, scalars, sampleRanges, volData, vol_cmap, volRange, alignment_rotations, on_click, scalar_range)
    l_hist_ax, l_hist_r = setup_cluster_view(window,
        tGrid, (2, 1),
        lift((x, y) -> y[x[2]], hl_info, cluster_assignments),
        cluster_groups,
        reordered_matrix,
        on_show_cluster_click
    )

    hr = Observable(2)
    hr_info = @lift begin
        t_idx = $clustering.order[$hr]
        return ($hr, t_idx, t_list[t_idx], $cluster_assignments[t_idx])
    end

    rtv = setup_transition_view(window, tGrid, (1, 2), alignedPositionsMatrices, hr_info, scalars, sampleRanges, volData, vol_cmap, volRange, alignment_rotations, on_click, scalar_range)

    r_hist_ax, r_hist_r = setup_cluster_view(window,
        tGrid,
        (2, 2),
        lift((x, y) -> y[x[2]], hr_info, cluster_assignments),
        cluster_groups,
        reordered_matrix,
        on_show_cluster_click
    )

    @lift begin
        mr = (-0.01, max($l_hist_r, $r_hist_r) + 0.01)

        xlims!(l_hist_ax, mr)
        reset_limits!(l_hist_ax; xauto=false)

        xlims!(r_hist_ax, mr)
        reset_limits!(r_hist_ax; xauto=false)
    end

    link_cameras_lscenes([ltv, rtv])

    # will complain about being passed "nothing" as a value if something isn't inside the set
    hovered_cluster = Observable(Set{Int}(1))

    graph_ax = Axis(window[1, 2], backgroundcolor=:transparent)
    hidexdecorations!(graph_ax)

    hm_ax, hm = heatmap(window[2, 2], lift(x -> x[1], reordered_matrix))

    hidedecorations!(hm_ax)
    deregister_interaction!(hm_ax, :rectanglezoom)

    on(events(hm_ax).mouseposition) do mp
        plot, _ = pick(hm_ax)
        if plot == hm
            xy = mouseposition(hm_ax)
            i, j = Int.(round.(xy))
            hl[] = i
            hr[] = j
            notify(hl)
            notify(hr)
        end
        return Consume(false)
    end

    rendered_clusters = []
    @lift begin
        foreach(x -> delete!(parent_scene(x), x), rendered_clusters)
        cmap = to_colormap(cluster_colors)
        for (c, ts_idx) in $cluster_groups
            idx_to_mtx = $reordered_matrix[2]
            p = show_cluster_on_hmap(ts_idx, idx_to_mtx, hm_ax.scene; color=cmap[c%length(cmap)+1])
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
            last_bBox = show_cluster_on_hmap(ts_idx, idx_to_mtx, hm_ax)
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

function show_cluster_on_hmap(ts_idx, idx_to_mtx, scene; color=:red)
    m_idx = map(x -> idx_to_mtx[x], ts_idx)

    lo = minimum(m_idx)
    hi = maximum(m_idx)

    # need to draw n bounding boxes over the heatmap
    bbox = Rect2(lo - 0.5, lo - 0.5, (hi - lo) + 1, (hi - lo) + 1)

    p = wireframe!(
        scene, bbox, color=color,
        visible=true, inspectable=false,
        depth_shift=-1.0f-3
    )
    return p
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
        on_click(hovered[][3], () -> ())
    end

    l = Label(fig, lift(x -> string(x[3]), hovered), tellwidth=false)

    i, j = loc
    g = vgrid!(rootScene, hgrid!(l, m, btn))
    parentGrid[i, j] = g

    opts = sort(collect(keys(scalars)))

    t_idx = lift(x -> x[2], hovered)
    t = lift(x -> x[3], hovered)

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
    c_idx,
    cluster_groups,
    reordered_matrix,
    on_cluster_button_click
)
    i, j = loc
    cluster_grid = GridLayout()
    parentGrid[i, j] = cluster_grid

    cmap = to_colormap(cluster_colors)
    cluster_grid[1, 1] = Label(fig,
        lift(x -> "Cluster $(x)", c_idx),
        tellwidth=false)

    show_cluster_btn = Button(cluster_grid[1, 2], label="Show")

    on(show_cluster_btn.clicks) do n
        on_cluster_button_click(c_idx[])
    end

    hist_r = Observable(0.0)

    hist_values = @lift begin
        # only change on hovered because hovered may be invalid
        ts_idx = cluster_groups[][$c_idx]
        mtx_idx = map(x -> reordered_matrix[][2][x], ts_idx)
        mat = reordered_matrix[][1]
        vals = mat[mtx_idx, mtx_idx]
        utri = triu!(trues(size(vals)))
        d = vec(vals[utri])
        hist_r[] = maximum(d)
        notify(hist_r)
        return d
    end

    hist_ax = Axis(cluster_grid[2, 1:2],
        backgroundcolor=:transparent, tellwidth=false, tellheight=false)

    # hide y labels because otherwise the width of each column gets adjusted
    hideydecorations!(hist_ax)

    # TODO: copy over code for custom implementation 
    # https://github.com/MakieOrg/Makie.jl/blob/master/src/stats/hist.jl 
    # unfortunately bar_labels doesn't work
    hist!(hist_ax,
        hist_values,
        normalization=:density,
        strokewidth=1,
        strokecolor=:black,
        color=lift(x -> cmap[x%length(cmap)+1], c_idx))

    return hist_ax, hist_r
end
