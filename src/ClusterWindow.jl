using Makie

function build_cluster_window(c_idx, ts, rel_t_idx, vals, alignedPositionMatrices,
    alignment_rotations, volData, sampleRanges, scalars, scalarRange, t_to_idx, vol_cmap, volumeRange; fig_size=(400, 400))
    window = Figure(size=fig_size)

    # the transitions being hovered on in the dist matrix
    hl = Observable(1)
    hr = Observable(2)

    scene_selector = Observable("Initial State")
    scalar_selector = Observable(first(keys(scalars)))

    title = Label(window[1, 1], "Cluster $(c_idx)", tellwidth=false)

    scalar_menu = Menu(window[1, 2],
        options=sort(collect(keys(scalars))),
        default=scalar_selector[], tellwidth=false)

    on(scalar_menu.selection) do s
        scalar_selector[] = s
        notify(scalar_selector)
    end

    render_menu = Menu(window[1, 3],
        options=["Initial State", "Volume"],
        default=scene_selector[], tellwidth=false)

    on(render_menu.selection) do s
        scene_selector[] = s
        notify(scene_selector)
    end

    tGrid = GridLayout()
    window[2, 1:2] = tGrid

    idx = 1
    for i in 1:4
        for j in 1:4
            rootScene = LScene(
                tGrid[i, j],
                show_axis=false,
                scenekw=(backgroundcolor=:black, clear=true),
            )

            if idx <= length(ts)
                linked_transition_view(rootScene, window, tGrid, (i, j), Observable(ts[idx]), scene_selector, scalar_selector, alignedPositionMatrices, alignment_rotations, ts, scalars, scalarRange, t_to_idx, volData, sampleRanges, vol_cmap, volumeRange)
            end
            idx += 1
        end
    end

    hm_ax, hm = heatmap(window[2, 3], vals, colorrange=(0.0, 1.0))
    DataInspector(hm)

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

    return window
end

function linked_transition_view(rootScene, fig, parentGrid, loc, t, scene_selection, scalar_selection, ap, alignment_rotations, ts, scalars, scalar_range, t_to_idx, volData, sampleRanges, vol_cmap, volumeRange)
    DataInspector(rootScene)

    l = Label(fig, lift(x -> string(x), t), tellwidth=false)
    i, j = loc

    g = vgrid!(rootScene, l)
    parentGrid[i, j] = g

    t_ap = @lift begin
        init = ap[$t][1] * $alignment_rotations[$t]
        final = ap[$t][2] * $alignment_rotations[$t]

        return (init, final)
    end

    t_idx = lift(x -> t_to_idx[x], t)

    atom_cmap = resample_cmap(:reds, 147, alpha=range(; start=0.01, stop=1.0, length=147))

    function select_fn(selection)
        if selection == "Volume"
            volume_view!(rootScene, t_idx, t, volData, sampleRanges, vol_cmap, volumeRange, alignment_rotations; update=true)
            return [], []
        else
            simple_atom_view!(rootScene, t_ap, lift((x, y) -> scalars[x][y], scalar_selection, t), scalar_range, atom_cmap, Observable(0.0))
            return [], []
        end
    end

    scene_switcher(rootScene, g, scene_selection, select_fn)
end
