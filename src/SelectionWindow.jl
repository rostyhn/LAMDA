using Makie: clear_temporary_plots!, Orthographic, SparseArrays
using StatsBase
using UMAP


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

    # grid for the transition views
    tGrid = GridLayout()
    window[1:2, 1] = tGrid

    hl = Observable(1)

    # contains actual matrix index, the transition idx, the tuple itself and the cluster assignment
    hl_info = @lift begin
        t_idx = $clustering.order[$hl]
        return ($hl, t_idx, t_list[t_idx], $cluster_assignments[t_idx])
    end

    ltv, l_hist_ax, l_hist_r = setup_transition_view(window, tGrid, (1, 1), alignedPositionsMatrices, hl_info, scalars, sampleRanges, volData, vol_cmap, volRange, t_to_idx, alignment_rotations, on_click, scalar_range, cluster_groups, reordered_matrix)

    hr = Observable(2)
    hr_info = @lift begin
        t_idx = $clustering.order[$hr]
        return ($hr, t_idx, t_list[t_idx], $cluster_assignments[t_idx])
    end

    rtv, r_hist_ax, r_hist_r = setup_transition_view(window, tGrid, (1, 2), alignedPositionsMatrices, hr_info, scalars, sampleRanges, volData, vol_cmap, volRange, t_to_idx, alignment_rotations, on_click, scalar_range, cluster_groups, reordered_matrix)

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

    cluster_cmap = :tab20
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
        cmap = to_colormap(cluster_cmap)
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

    dendrogram!(graph_ax, clustering, h_cutoff; hover_callbackfn=on_dendrogram_hover, colormap=cluster_cmap)

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

function simple_atom_view!(scene, g, ap, t, order, scalars, sel, alignment_rotations, scalar_range)
    opts = sort(collect(keys(scalars)))

    gg = GridLayout(g[end+1, :])
    m = Menu(gg[1, 1], options=opts, default=length(sel[]) == 0 ? first(opts) : sel[])

    cmap = resample_cmap(:reds, 147, alpha=range(; start=0.01, stop=1.0, length=147))

    # instead of setting the model, we just multiply by the raw rotation matrix 
    rot_pos = lift((x, y) -> ap[x[3]][order] * y[x[3]], t, alignment_rotations)
    scatter!(scene,
        lift(x -> x[:, 1], rot_pos),
        lift(x -> x[:, 2], rot_pos),
        lift(x -> x[:, 3], rot_pos);
        color=lift((x, y) -> scalars[y][x[3]], t, m.selection),
        colorrange=scalar_range,
        colormap=resample_cmap(:reds, 147, alpha=range(; start=0.01, stop=1.0, length=147)),
        inspector_label=(self, i, p) -> "Atom $(i); weight: $(self.color[][i])",
        markersize=30)

    cbar = Colorbar(gg[1, 2], colorrange=scalar_range, vertical=false, colormap=cmap, tellwidth=false)

    return [], [(gg, [m, cbar])]
end

function setup_transition_view(fig, parentGrid, loc, ap, hovered, scalars, sampleRanges, volData, vol_cmap, volumeRange, t_to_idx, alignment_rotations, on_click, scalar_range, cluster_groups, reordered_matrix)
    rootScene = LScene(
        fig,
        show_axis=false,
        scenekw=(backgroundcolor=:black, clear=true),
    )

    m = Menu(fig,
        options=["Volume",
            "Initial State",
            "Final State"],
        default="Volume")

    btn = Button(fig, label="Show")
    on(btn.clicks) do n
        on_click(hovered[][3], () -> ())
    end

    l = Label(fig, lift(x -> string(x[3]), hovered), tellwidth=false)
    i, j = loc
    g = vgrid!(rootScene, hgrid!(l, m, btn))
    parentGrid[i, j] = g

    DataInspector(rootScene)

    sel = Observable("Volume")
    bp_sel = Observable("")
    ap_sel = Observable("")

    scene_listeners = Vector{Any}()
    ui_elements = Vector{Any}()

    on(m.selection) do cw
        sel[] = cw
        notify(sel)
    end

    @lift begin
        # cleanup
        empty!(rootScene)

        for listener in scene_listeners
            off(listener)
            listener = Nothing
        end
        empty!(scene_listeners)

        # clear UI elements
        for c in ui_elements
            gg, elements = c
            for e in elements
                empty!(e.blockscene)
                delete!(e)
            end
            if !isnothing(gg)
                Makie.trim!(gg)
                # only way to delete a gridlayout
                GridLayoutBase.remove_from_gridlayout!(gg.layoutobservables.gridcontent[])
            end
        end
        Makie.trim!(g)

        if $sel == "Volume"
            gg = GridLayout(g[end+1, :])

            vd = lift((x, y, z) ->
                    reshape(x[:, y[2]], (length(z[1]), length(z[2]), length(z[3]))), volData, hovered, sampleRanges)

            v = volume!(rootScene,
                lift(x -> extrema(x[1]), sampleRanges),
                lift(x -> extrema(x[2]), sampleRanges),
                lift(x -> extrema(x[3]), sampleRanges),
                vd;
                colormap=vol_cmap,
                algorithm=:absorption,
                fxaa=false,
                transparency=true,
                shading=NoShading,
                colorrange=volumeRange)

            # FIXME sometimes the volume will get rotated so hard it disappears
            on(hovered) do h
                R = alignment_rotations[][h[3]]
                rr = hcat(R, [0, 0, 0])
                fr = transpose(vcat(rr, transpose([0; 0; 0; 1])))
                v.model[] = fr
                notify(v.model)
            end

            v.inspectable[] = false

            cbar = Colorbar(gg[1, :],
                colorrange=volumeRange, vertical=false, colormap=vol_cmap, tellwidth=false)

            il = []
            is = [(gg, [cbar])]
        elseif $sel == "Initial State"
            il, is = simple_atom_view!(rootScene, g, ap, hovered, 1, scalars, bp_sel, alignment_rotations, scalar_range)
        else
            il, is = simple_atom_view!(rootScene, g, ap, hovered, 2, scalars, ap_sel, alignment_rotations, scalar_range)
        end

        for l in il
            push!(scene_listeners, l)
        end

        for s in is
            push!(ui_elements, s)
        end
    end

    # setup cluster view for selected transition

    cluster_grid = GridLayout()
    parentGrid[i+1, j] = cluster_grid

    cmap = to_colormap(:tab20)
    cluster_grid[1, 1] = Label(fig,
        lift(x -> "Cluster $(x[4])", hovered),
        tellwidth=false)
    #color=lift(x -> cmap[x[4]%length(cmap)+1], hovered))

    hist_r = Observable(0.0)

    hist_values = @lift begin
        # only change on hovered because hovered may be invalid
        ts_idx = cluster_groups[][$hovered[4]]
        mtx_idx = map(x -> reordered_matrix[][2][x], ts_idx)
        mat = reordered_matrix[][1]
        vals = mat[mtx_idx, mtx_idx]
        utri = triu!(trues(size(vals)))
        d = vec(vals[utri])
        hist_r[] = maximum(d)
        notify(hist_r)
        return d
    end

    hist_ax = Axis(cluster_grid[2, 1],
        backgroundcolor=:transparent, tellwidth=false, tellheight=false)

    # hide y labels because otherwise the width of each column gets adjusted
    hideydecorations!(hist_ax)

    # TODO: copy over code for custom implementation 
    # https://github.com/MakieOrg/Makie.jl/blob/master/src/stats/hist.jl 
    # unfortunately bar_labels doesn't work
    h = hist!(hist_ax,
        hist_values,
        normalization=:density,
        strokewidth=1,
        strokecolor=:black,
        color=lift(x -> cmap[x[4]%length(cmap)+1], hovered))

    return rootScene, hist_ax, hist_r
end
