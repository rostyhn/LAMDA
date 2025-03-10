using Makie

const GRID_SIZE = 16
const GRID_X = Int(sqrt(GRID_SIZE))
const GRID_Y = Int(sqrt(GRID_SIZE))
const SCENE_SELECTED = to_color(:grey)
const BLACK = to_color(:black)

function build_cluster_window(clusters, ts, idx_to_mtx_idx, vals, alignedPositionMatrices,
    alignment_rotations, volData, sampleRanges, scalars, scalarRange, t_to_idx, vol_cmap, volumeRange, mat_range; fig_size=(400, 400))
    window = Figure(size=fig_size)

    # the transitions being hovered on in the dist matrix
    mat_hovered = Observable((0, 0))

    scene_selector = Observable("Initial State")
    scalar_selector = Observable(first(keys(scalars)))

    title = Label(window, "Cluster $(str_limit(clusters))", fontsize=30)

    render_menu = Menu(window,
        options=["Initial State", "Volume"],
        default=scene_selector[], tellwidth=false)

    on(render_menu.selection) do s
        scene_selector[] = s
        notify(scene_selector)
    end

    scalar_menu = Menu(window,
        options=sort(collect(keys(scalars))),
        default=scalar_selector[],
    )

    on(scalar_menu.selection) do s
        scalar_selector[] = s
        notify(scalar_selector)
    end

    time = Observable(0.0)
    t_slider = Slider(window, range=0.0:0.05:1.0, startvalue=0.0)
    on(t_slider.value) do x
        time[] = x
    end

    window[1, 1:2] = hgrid!(title,
        Label(window, "Render mode"),
        render_menu,
        scalar_menu,
        t_slider)


    l_btn = Button(window, label="◀", tellwidth=false)
    r_btn = Button(window, label="▶", tellwidth=false)

    # transition coupled with matrix coords
    ts_pairs = sort(collect(zip(ts, idx_to_mtx_idx)), by=x -> x[2])

    curr_page = Observable(1)
    ts_chunks = collect(Iterators.partition(ts_pairs, GRID_SIZE))
    num_pages = length(ts_chunks)

    on(l_btn.clicks) do n
        if curr_page[] > 1
            curr_page[] -= 1
        end
    end

    on(r_btn.clicks) do n
        if curr_page[] < length(ts_chunks)
            curr_page[] += 1
        end
    end

    pg_label = Label(window, lift(x -> "Page $(x) of $(num_pages)", curr_page), tellwidth=false)

    tGrid = GridLayout()
    window[2, 1:2] = vgrid!(tGrid, hgrid!(l_btn, pg_label, r_btn))

    hm_ax, hm = heatmap(window[2, 3], vals, colorrange=mat_range)
    DataInspector(hm)

    # draw boxes around pages
    page_boxes = []
    for ts_p in ts_chunks
        idxs = last.(ts_p)
        lo = minimum(idxs)
        hi = maximum(idxs)
        p = draw_bbox_pixel_space!(hm_ax, lo, hi; color=:grey)
        push!(page_boxes, p)
    end

    last_cp = 0
    on(curr_page, update=true) do cp
        page_boxes[cp].color[] = :red
        if last_cp > 0
            page_boxes[last_cp].color[] = :grey
        end
        last_cp = cp
    end

    hidedecorations!(hm_ax)
    deregister_interaction!(hm_ax, :rectanglezoom)

    on(events(hm_ax).mouseposition) do mp
        plot, _ = pick(hm_ax)
        if is_mouseinside(hm_ax.scene)
            if plot == hm
                xy = mouseposition(hm_ax)
                i, j = Int.(round.(xy))
                mat_hovered[] = (i, j)
            end
        else
            mat_hovered[] = (0, 0)
        end
        notify(mat_hovered)
        return Consume(false)
    end
    fp = first(ts_pairs)

    scenes = []
    scene_info = []
    for i in 1:GRID_X
        for j in 1:GRID_Y
            rootScene = LScene(
                tGrid[i, j],
                show_axis=false,
                scenekw=(backgroundcolor=:black, clear=true),
            )
            DataInspector(rootScene)

            t = Observable(fp[1])
            mtx_idx = Observable(fp[2])

            is_visible = Observable(false)
            idx = (i - 1) * 4 + j

            # could be more efficient if this gets updated per page instead of per transition
            on(curr_page, update=true) do cp
                curr_chunk = ts_chunks[cp]

                if idx <= length(curr_chunk)
                    t[] = curr_chunk[idx][1]
                    mtx_idx[] = curr_chunk[idx][2]
                    is_visible[] = true
                else
                    is_visible[] = false
                end
            end
            linked_transition_view(rootScene, window, tGrid, (i, j), t, scene_selector, scalar_selector, alignedPositionMatrices, alignment_rotations, scalars, scalarRange, t_to_idx, volData, sampleRanges, vol_cmap, volumeRange, time, is_visible, mtx_idx)

            push!(scenes, rootScene.scene)
            push!(scene_info, mtx_idx)
        end
    end

    # draws rectangle on matrix whenever a transition is hovered over
    # not elegant, but it works and is relatively efficient
    bBox = nothing
    last_bBox = 0
    on(events(window).mouseposition) do mp
        if is_mouseinside(window)
            found = false
            for (i, s) in enumerate(scenes)
                if mp in viewport(s)[]
                    if last_bBox != i
                        if !isnothing(bBox)
                            delete!(parent_scene(bBox), bBox)
                        end
                        bBox = draw_bbox_pixel_space!(hm_ax, scene_info[i][], scene_info[i][])
                        last_bBox = i
                    end
                    found = true
                    break
                end
            end
            if !found
                if !isnothing(bBox)
                    last_bBox = 0
                    delete!(parent_scene(bBox), bBox)
                    bBox = nothing
                end
            end
        else
            if !isnothing(bBox)
                last_bBox = 0
                delete!(parent_scene(bBox), bBox)
                bBox = nothing
            end
        end
    end

    function color_scene!(idx, color)
        if idx != 0
            s = scenes[mod1(idx, length(scenes))]
            s.backgroundcolor[] = color
        end
    end


    last_lh = 0
    last_rh = 0
    on(mat_hovered, update=true) do h
        lh, rh = h

        curr_chunk = last.(ts_chunks[curr_page[]])

        if lh != last_lh
            if lh in curr_chunk
                color_scene!(lh, SCENE_SELECTED)
            end
            color_scene!(last_lh, BLACK)
            last_lh = lh
        end

        if rh != last_rh
            if rh in curr_chunk
                color_scene!(rh, SCENE_SELECTED)
            end
            color_scene!(last_rh, BLACK)
            last_rh = rh
        end
    end

    return window
end


function linked_transition_view(rootScene, fig, parentGrid, loc, t, scene_selection, scalar_selection, ap, alignment_rotations, scalars, scalar_range, t_to_idx, volData, sampleRanges, vol_cmap, volumeRange, time, is_visible, mtx_idx)
    l = Label(fig, lift(x -> "$(x)", t), tellwidth=false, visible=lift(x -> x, is_visible))
    i, j = loc

    g = vgrid!(rootScene, l)
    parentGrid[i, j] = g

    t_idx = lift(x -> t_to_idx[x], t)

    atom_cmap = resample_cmap(:reds, 147, alpha=range(; start=0.01, stop=1.0, length=147))

    function select_fn(selection)
        if selection == "Volume"
            vd = lift((x, y, z) ->
                    reshape(x[:, y], (length(z[1]), length(z[2]), length(z[3]))), volData, t_idx, sampleRanges)
            v_lo, v_hi = volume_view!(rootScene, vd, sampleRanges, vol_cmap, volumeRange; rotation=lift((x, y) -> x[y], alignment_rotations, t), update=true)
            @lift begin
                v_lo.visible[] = $is_visible
                v_hi.visible[] = $is_visible
                notify(v_lo.visible)
                notify(v_hi.visible)
            end
            return [], []
        else
            t_ap = @lift begin
                init = ap[$t][1] * $alignment_rotations[$t]
                final = ap[$t][2] * $alignment_rotations[$t]

                return (init, final)
            end

            s = simple_atom_view!(rootScene, t_ap, lift((x, y) -> scalars[x][y], scalar_selection, t), scalar_range, atom_cmap, time)
            @lift begin
                s.visible[] = $is_visible
                notify(s.visible)
            end
            return [], []
        end
    end

    scene_switcher(rootScene, g, scene_selection, select_fn)
end
