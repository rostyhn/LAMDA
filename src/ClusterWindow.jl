using Makie

function build_cluster_window(c_idx, ts, rel_t_idx, vals, aap, volData, sampleRanges, scalars, scalarRange; fig_size=(400, 400))
    window = Figure(size=fig_size)

    # the transitions being hovered on in the dist matrix
    hl = Observable(1)
    hr = Observable(2)

    scene_selector = Observable("Initial State")
    scalar_selector = Observable(first(keys(scalars)))

    title = Label(window[1, 1], "Cluster $(c_idx)", tellwidth=false)

    scalar_menu = Menu(window[1, 2],
        options=sort(collect(keys(scalars))),
        default=scalar_selector[])

    on(scalar_menu.selection) do s
        scalar_selector[] = s
        notify(scalar_selector)
    end

    tGrid = GridLayout()
    window[2, 1] = tGrid

    idx = 1
    for i in 1:3
        for j in 1:3
            rootScene = LScene(
                tGrid[i, j],
                show_axis=false,
                scenekw=(backgroundcolor=:black, clear=true),
            )

            if idx < length(ts)
                linked_transition_view(rootScene, window, tGrid, (i, j), Observable(idx), scene_selector, scalar_selector, aap, ts, scalars, scalarRange)
            end
            idx += 1
        end
    end

    hm_ax, hm = heatmap(window[2, 2], vals, colorrange=(0.0, 1.0))
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

function linked_transition_view(rootScene, fig, parentGrid, loc, t, scene_selection, scalar_selection, ap, ts, scalars, scalar_range)
    DataInspector(rootScene)

    tt = lift(x -> ts[x], t)
    l = Label(fig, lift(x -> string(x), tt), tellwidth=false)
    i, j = loc

    g = vgrid!(rootScene, l)
    parentGrid[i, j] = g

    t_ap = lift(x -> ap[x], t)

    function select_fn(selection)
        return simple_atom_view!(rootScene, g, t_ap, tt, scalars, scalar_selection, scalar_range; show_menu=false)
    end

    scene_switcher(rootScene, g, scene_selection, select_fn)
end
