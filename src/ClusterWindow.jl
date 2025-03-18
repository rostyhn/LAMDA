using Makie

const GRID_SIZE = 16
const GRID_X = Int(sqrt(GRID_SIZE))
const GRID_Y = Int(sqrt(GRID_SIZE))
const SCENE_SELECTED = to_color(:grey)
const BLACK = to_color(:black)

function build_cluster_window(clusters,
    ts,
    ref_t,
    idx_to_mtx_idx,
    vals,
    scalars,
    mat_range,
    render_views,
    widgets,
    bins,
    on_transition_select,
    hovered_transition,
    inspector_ref::MaybeObservable{DataInspector};
    on_window_hover=(x) -> (),
    fig_size=(400, 400)
)

    window = Figure(size=fig_size)

    # the transitions being hovered on in the dist matrix
    mat_hovered = Observable((0, 0))

    scene_selector = Observable("Atom")
    scalar_selector = Observable(first(sort(collect(keys(scalars)))))

    title = "Cluster $(str_limit(clusters; len=25))"
    menu_bar = top_bar(window, title, 3)

    render_menu = Menu(window,
        options=SINGLE_TRANSITION_RENDER_OPTIONS,
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

    time, t_slider = widgets["Movement"](0.0, window)
    btn_centroid = Button(window, label="Show centroid")

    window[2, 1:2] = hgrid!(
        btn_centroid,
        Label(window, "Render mode"),
        render_menu,
        scalar_menu,
        t_slider)

    l_btn = Button(window, label="◀", tellwidth=false)
    r_btn = Button(window, label="▶", tellwidth=false)

    # sort transitions by idx in matrix
    sortperm!(idx_to_mtx_idx, ts)

    # transition to matrix index dict
    t_to_mtx = Dict(reverse.(collect(enumerate(ts))))

    curr_page = Observable(1)
    ts_chunks = collect(Iterators.partition(ts, GRID_SIZE))
    num_pages = length(ts_chunks)

    centroid_page = 1
    for chunk in ts_chunks
        if ref_t in chunk
            break
        end
        centroid_page += 1
    end

    on(btn_centroid.clicks) do n
        curr_page[] = centroid_page
        notify(curr_page)
    end

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
    window[3, 1:2] = vgrid!(tGrid, hgrid!(l_btn, pg_label, r_btn))

    mat_grid = GridLayout()
    window[2:3, 3] = mat_grid

    utri = triu!(trues(size(vals)))
    hist_vals = vec(vals[utri])

    cluster_cmap = to_colormap(CLUSTER_COLORS)
    cluster_color = to_color(:grey)
    cl = collect(clusters)
    if length(cl) == 1
        cluster_color = cluster_cmap[mod1(first(cl), length(cluster_cmap))]
    end

    centroid_grid = GridLayout()
    mat_grid[1, 1] = centroid_grid

    Label(centroid_grid[1, 1], "Cluster average", font=:bold, tellwidth=false)
    centroid_scene = LScene(
        centroid_grid[2, 1],
        show_axis=false,
        scenekw=(backgroundcolor=:black, clear=true),
    )

    render_views["SMovement"](centroid_scene, ts, time)

    hist_ax = Axis(mat_grid[2, 1], title="Intra-cluster distances",
        backgroundcolor=:transparent, tellwidth=false, tellheight=false)

    deregister_interaction!(hist_ax, :rectanglezoom)
    hideydecorations!(hist_ax)

    hist!(hist_ax,
        hist_vals,
        normalization=:density,
        strokewidth=1,
        strokecolor=:black,
        color=cluster_color,
        bins=bins
    )

    hm_ax, hm = heatmap(mat_grid[3, 1], vals, colorrange=mat_range)
    hidedecorations!(hm_ax)
    deregister_interaction!(hm_ax, :rectanglezoom)

    #rowsize!(mat_grid, 2, Relative(0.75))

    # draw boxes around pages
    page_boxes = []
    for ts_page in ts_chunks
        idxs = map(x -> t_to_mtx[x], ts_page)
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

    scenes = []
    scene_info = []
    for i in 1:GRID_X
        for j in 1:GRID_Y
            rootScene = LScene(
                tGrid[i, j],
                show_axis=false,
                scenekw=(backgroundcolor=:black, clear=true),
            )

            idx = (i - 1) * 4 + j
            init_t = nothing
            if idx <= length(ts)
                init_t = ts[idx]
            end

            t = MaybeObservable{Tuple{Int,Int}}(init_t)
            is_visible = Observable(false)
            # could be more efficient if this gets updated per page instead of per transition
            on(curr_page, update=true) do cp
                curr_chunk = ts_chunks[cp]

                if idx <= length(curr_chunk)
                    t[] = curr_chunk[idx]
                    is_visible[] = true
                else
                    t.val = nothing
                    #notify(t)
                    is_visible[] = false
                end
            end

            linked_transition_view(rootScene,
                window,
                tGrid,
                (i, j),
                t,
                scene_selector,
                scalar_selector,
                scalars,
                time,
                is_visible,
                render_views,
                inspector_ref,
                lift(x -> ref_t == x, t)
            )

            push!(scenes, rootScene.scene)
            push!(scene_info, t)
        end
    end

    # draws rectangle on matrix whenever a transition is hovered over
    # not elegant, but it works and is relatively efficient
    # saves us from having 16 observables
    bBox = nothing
    m_events = addmouseevents!(window.scene)
    on(m_events.obs) do e
        if e.type === MouseEventTypes.over
            found = false
            currently_rendered = collect(ts_chunks[curr_page[]])
            for (i, s) in enumerate(scenes[1:length(currently_rendered)])
                if is_mouseinside(s)
                    if hovered_transition[] != currently_rendered[i]
                        hovered_transition[] = currently_rendered[i]
                        notify(hovered_transition)
                    end
                    found = true
                    break
                end
            end
            if !found && !isnothing(hovered_transition[])
                hovered_transition.val = nothing
                notify(hovered_transition)
            end
        elseif e.type == MouseEventTypes.leftdoubleclick
            if !isnothing(hovered_transition[])
                on_transition_select(hovered_transition[])
            end
        end
    end

    on(events(window).entered_window) do entered
        if entered
            on_window_hover(clusters)
        else
            on_window_hover(nothing)
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

    scene_idx = 0
    on(hovered_transition) do t
        currently_rendered_transitions = ts_chunks[curr_page[]]
        if !isnothing(bBox)
            delete!(parent_scene(bBox), bBox)
            color_scene!(scene_idx, BLACK)
        end
        if !isnothing(t) && t in currently_rendered_transitions
            scene_idx = findfirst(==(t), currently_rendered_transitions)
            mtx_idx = t_to_mtx[t]
            bBox = draw_bbox_pixel_space!(hm_ax, mtx_idx, mtx_idx)
            color_scene!(scene_idx, SCENE_SELECTED)
        end
    end

    on(mat_hovered) do h
        lh, rh = h

        # get current page's indexes
        curr_chunk = ts_chunks[curr_page[]]
        current_idxs = map(x -> t_to_mtx[x], curr_chunk)

        if lh != last_lh
            if lh in current_idxs
                color_scene!(lh, SCENE_SELECTED)
            end
            color_scene!(last_lh, BLACK)
            last_lh = lh
        end

        if rh != last_rh
            if rh in current_idxs
                color_scene!(rh, SCENE_SELECTED)
            end
            color_scene!(last_rh, BLACK)
            last_rh = rh
        end
    end


    return window
end


function linked_transition_view(rootScene,
    fig,
    parentGrid,
    loc,
    t,
    scene_selection,
    scalar_selection,
    scalars,
    time,
    is_visible,
    render_views,
    inspector,
    is_centroid
)

    is_empty = lift(x -> isnothing(x), t)

    l = Label(fig, lift(x -> "$(x)", t), tellwidth=false, visible=lift(x -> x, is_visible), font=lift(x -> (x) ? :bold : :regular, is_centroid))
    i, j = loc

    g = vgrid!(rootScene, l)
    parentGrid[i, j] = g
    parentGrid[i, j] = Box(fig, strokecolor=:green, color=:transparent, visible=(lift(x -> x, is_centroid)))

    function select_fn(selection)
        #band-aid solution for now, will break if user goes to last page and then switches selection
        if is_empty[]
            return [], []
        end

        @lift begin
            parent_scene(rootScene).visible[] = $is_visible
            notify(parent_scene(rootScene).visible)
        end

        if selection == "Volume"
            render_views[selection](rootScene, t)
            return [], []
        elseif selection == "Atom"
            render_views[selection](rootScene, t, scalar_selection, time)
            return [], []
        elseif selection == "Superquadric"
            return render_views[selection](rootScene, t, inspector[])
        else
            return render_views[selection](rootScene, t, time)
        end
    end


    scene_switcher(rootScene, g, scene_selection, select_fn)
end
