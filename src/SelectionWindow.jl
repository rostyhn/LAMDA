using Makie: clear_temporary_plots!, Orthographic, SparseArrays
using StatsBase
using UMAP

function build_selection_window(fig_size,
    data,
    t_list,
    t_to_idx,
    on_click,
    num_atoms,
    alignedPositions,
    transitionKDTree, dms, volData, sampleRanges, volRange, vol_cmap, selected_invariant, clustering, selected_dm, scalars, h_cutoff, cluster_groups)

    window = Figure(size=fig_size)
    user_groups = Observable(Dict())

    # reorders distance matrix according to clustering
    reordered_matrix = @lift begin
        m = dms[$selected_dm]
        rm = zeros(size(m))

        # gets the correct idx 
        mtx_to_t = Dict()
        t_to_mtx = Dict()
        for (i, r) in enumerate($clustering.order)
            rm[i, :] .= m[r, :][$clustering.order]
            mtx_to_t[i] = t_list[r]
            t_to_mtx[t_list[r]] = i
        end

        # get minimum and maximum of entire matrix for cmap
        fl = vec(m)
        return rm, mtx_to_t, (minimum(fl), maximum(fl)), t_to_mtx
    end
    # can't get it to align left
    # title =Label(window[1, 1], "TransVis", justification=:left, fontsize=30, tellwidth=false)
    grid = GridLayout()
    window[1, 1] = grid
    hl = Observable(first(t_list))
    hr = Observable(last(t_list))

    ltv = setup_transition_view(window, grid, (1, 1), alignedPositions, hl, scalars, sampleRanges, volData, vol_cmap, volRange, t_to_idx)
    rtv = setup_transition_view(window, grid, (1, 2), alignedPositions, hr, scalars, sampleRanges, volData, vol_cmap, volRange, t_to_idx)

    link_cameras_lscenes([ltv, rtv])

    hovered_cluster = Observable(Set{Int64}(1))

    cluster_cmap = :tab20
    graph_ax = Axis(window[2:3, 1], backgroundcolor=:transparent)
    hm_ax, hm = heatmap(window[1:2, 2], lift(x -> x[1], reordered_matrix))

    hidedecorations!(hm_ax)
    deregister_interaction!(hm_ax, :rectanglezoom)

    on(events(hm_ax).mouseposition) do mp
        plot, _ = pick(hm_ax)
        if plot == hm
            xy = mouseposition(hm_ax)
            i, j = Int.(round.(xy))
            hl[] = reordered_matrix[][2][i]
            hr[] = reordered_matrix[][2][j]
            notify(hl)
            notify(hr)
        end
        return Consume(false)
    end

    @lift begin
        cmap = to_colormap(cluster_cmap)
        for (c, ts_idx) in $cluster_groups
            ts = t_list[ts_idx]
            t_to_mtx = $reordered_matrix[4]
            show_cluster_on_hmap(ts, t_to_mtx, hm_ax.scene; color=cmap[c%length(cmap)+1])
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
            ts = t_list[ts_idx]

            t_to_mtx = $reordered_matrix[4]

            last_bBox = show_cluster_on_hmap(ts, t_to_mtx, hm_ax)
        end
    end

    function on_dendrogram_hover(c)
        if !isempty(c)
            hovered_cluster[] = c
            notify(hovered_cluster)
        end
    end

    @lift begin
        dendrogram!(graph_ax, $clustering, $h_cutoff; hover_callbackfn=on_dendrogram_hover, colormap=cluster_cmap)
    end

    dm_menu = Menu(window, options=collect(keys(dms)))
    on(dm_menu.selection) do val
        selected_dm[] = val
    end

    settings_btn = Button(window, label="Settings")
    settings_window = build_settings_menu(selected_invariant)
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

function show_cluster_on_hmap(ts, t_to_mtx, scene; color=:red)
    m_idx = map(x -> t_to_mtx[x], ts)

    lo = minimum(m_idx)
    hi = maximum(m_idx)

    # need to draw n bounding boxes over the heatmap
    bbox = Rect2(lo - 0.5, lo - 0.5, (hi - lo) + 0.5, (hi - lo) + 0.5)

    p = wireframe!(
        scene, bbox, color=color,
        visible=true, inspectable=false,
        depth_shift=-1.0f-3
    )
    return p
end

function simple_atom_view!(scene, g, ap, t, order, scalars, sel)
    opts = sort(collect(keys(scalars)))

    gg = GridLayout(g[end+1, :])
    m = Menu(gg[1, 1], options=opts, default=length(sel[]) == 0 ? first(opts) : sel[])

    colorInfo = @lift begin
        opt = $(m.selection)
        vals = scalars[opt][$t][order]

        extremaVals = extrema(vals)
        # three cases:
        # sequential ascending, descending and diverging
        # divergent if we are approximately around 0 when subtracting the absolute values of the min and max
        minVal, maxVal = extremaVals
        if isapprox(abs(maxVal) - abs(minVal), 0; atol=1)
            cmap = resample_cmap(:bam, 100; alpha=([(-0.99):0.02:(0.99);] ./ 0.1) .^ 6)
        else
            # ascending sequential if min is closer to 0
            cmap = resample_cmap(:reds, 147, alpha=range(; start=0.01, stop=1.0, length=147)) #seq ascending
            if abs(minVal) > abs(maxVal)
                reverse!(cmap)
            end
        end

        sel[] = opt
        notify(sel)

        return vals, extremaVals, cmap
    end

    scatter!(scene, lift((x, y) -> x[y][order], ap, t),
        color=lift(x -> x[1], colorInfo),
        colorrange=lift(x -> x[2], colorInfo),
        colormap=lift(x -> x[3], colorInfo),
        inspector_label=(self, i, p) -> "Atom $(i); weight: $(self.color[][i])",
        markersize=30)

    cbar = Colorbar(gg[1, 2], colorrange=lift(x -> x[2], colorInfo), vertical=false, colormap=lift(x -> x[3], colorInfo), tellwidth=false)

    return [], [(gg, [m, cbar])]
end

function setup_transition_view(fig, parentGrid, loc, ap, hovered, scalars, sampleRanges, volData, vol_cmap, volumeRange, t_to_idx)
    rootScene = LScene(
        fig,
        show_axis=false,
        scenekw=(backgroundcolor=:black, clear=true),
    )

    m = Menu(fig, options=["Volume", "Initial State", "Final State"],
        default="Volume")

    l = Label(fig, lift(x -> string(x), hovered), tellwidth=false)
    i, j = loc
    g = vgrid!(rootScene, hgrid!(l, m))
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
                    reshape(x[:, t_to_idx[y]], (length(z[1]), length(z[2]), length(z[3]))), volData, hovered, sampleRanges)

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
                colorrange=volumeRange,
                visible=true)
            v.inspectable[] = false

            cbar = Colorbar(gg[1, :],
                colorrange=volumeRange, vertical=false, colormap=vol_cmap, tellwidth=false)

            il = []
            is = [(gg, [cbar])]
        elseif $sel == "Initial State"
            il, is = simple_atom_view!(rootScene, g, ap, hovered, 1, scalars, bp_sel)
        else
            il, is = simple_atom_view!(rootScene, g, ap, hovered, 2, scalars, ap_sel)
        end

        for l in il
            push!(scene_listeners, l)
        end

        for s in is
            push!(ui_elements, s)
        end
    end

    return rootScene
end
