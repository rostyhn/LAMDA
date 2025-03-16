using Makie: clear_temporary_plots!, Orthographic, SparseArrays, apply_transform_and_model
using GLMakie: Screen
using StatsBase
using UMAP
using ImageIO
using NetworkLayout

const MIN_NODE_SIZE = 10.0
const MAX_NODE_SIZE = 100.0

function build_selection_window(fig_size,
    t_list,
    t_to_idx::Dict{Tuple{Int,Int},Int},
    on_click,
    num_atoms,
    dm,
    volRange,
    vol_cmap,
    cluster_data::Observable{ClusterData},
    cluster_info::Observable{ClusterInfo},
    scalars,
    h_cutoff,
    h_range,
    settings_window,
    render_views,
    widgets,
    invariantRange,
    matColLabel,
    per_t_scalars,
    per_t_scalar_ranges
)

    window = Figure(size=fig_size)
    menu_bar = top_bar(window, "Overview", 2)

    init_transitions = Set{Tuple{Int,Int}}()
    selected_transitions = Observable{Set{Tuple{Int,Int}}}(init_transitions)
    hovered_transition = MaybeObservable{Tuple{Int,Int}}()

    # used to place transitions into scratchpad
    function on_transition_select(t)
        push!(selected_transitions[], t)
        notify(selected_transitions)
    end

    bins = @lift begin
        return $h_range[1]:1:($h_range[2]+1)
    end

    open_cluster_windows = Dict{Set{Int},Screen}()
    function on_show_cluster_click(clusters)
        if !(clusters in keys(open_cluster_windows))
            ts_idx = reduce(vcat, map(x -> cluster_info[].groups[x], collect(clusters)))
            ts = t_list[ts_idx]

            ts_idx_to_mtx_idx = map(x -> cluster_data[].idx_to_mtx[x], ts_idx)
            mtx_idx = sort(ts_idx_to_mtx_idx)
            mat = cluster_data[].matrix
            vals = mat[mtx_idx, mtx_idx]

            # want to update volume data in case user messes with volume params
            # but we keep atom positions consistent with the alignment that existed at the time of creation
            ds::MaybeObservable{DataInspector} = Observable(nothing)
            w = build_cluster_window(
                clusters,
                ts,
                collect(eachindex(ts_idx_to_mtx_idx)),
                vals,
                scalars,
                cluster_data[].m_extrema,
                render_views,
                widgets,
                bins,
                on_transition_select,
                hovered_transition,
                ds
            )
            s = GLMakie.Screen(title="Cluster $(str_limit(clusters))")
            display(s, w)

            # create inspector after render to avoid bugs
            ds[] = DataInspector(w)


            open_cluster_windows[clusters] = s
            # might be causing a memory leak
            on(events(w).window_open) do is_open
                if !is_open
                    delete!(open_cluster_windows, clusters)
                end
            end
        end
    end

    # close cluster views if clustering changes
    on(cluster_data) do c
        foreach(s -> close(s), values(open_cluster_windows))
        empty!(open_cluster_windows)
    end

    # will complain about being passed "nothing" as a value if something isn't inside the set
    hovered_cluster = Observable(Set{Int}(1))

    cutoff_tb = Textbox(window, validator=Float64, placeholder=string(h_cutoff[]))
    on(cutoff_tb.stored_string) do s
        # reset hovered_cluster to avoid crashing
        hovered_cluster[] = Set{Int}(1)
        notify(hovered_cluster)

        h_cutoff[] = parse(Float64, s)
        notify(h_cutoff)
    end

    dGrid = GridLayout()
    window[2:3, 1] = dGrid
    #colsize!(window.layout, 2, Relative(0.66))

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

    function on_dendrogram_click(clusters)
        on_show_cluster_click(clusters)
    end

    dendrogram!(graph_ax,
        lift(x -> x.clustering, cluster_data),
        h_cutoff,
        h_range,
        hovered_cluster;
        on_click=on_dendrogram_click,
        colormap=CLUSTER_COLORS)
    hm = heatmap!(hm_ax, lift(x -> x.matrix, cluster_data))
    hm_m_events = addmouseevents!(hm_ax.scene)

    on(hm_m_events.obs) do e
        if e.type === MouseEventTypes.leftdown
            plot, _ = pick(hm_ax)
            if !isnothing(plot)
                ord = cluster_data[].clustering.order
                xy = mouseposition(hm_ax)
                i, j = Int.(round.(xy))
                t_idx1 = ord[i]
                t_idx2 = ord[j]
                t1 = t_list[t_idx1]
                push!(selected_transitions[], t1)

                if t_idx1 != t_idx2
                    t2 = t_list[t_idx2]
                    push!(selected_transitions[], t2)
                end

                notify(selected_transitions)
            end
        end
    end

    cluster_cmap = to_colormap(CLUSTER_COLORS)
    rendered_clusters = []
    @lift begin
        foreach(x -> delete!(parent_scene(x), x), rendered_clusters)
        for (c, ts_idx) in $(cluster_info).groups
            idx_to_mtx = $(cluster_data).idx_to_mtx
            m_idx = map(x -> idx_to_mtx[x], ts_idx)

            lo = minimum(m_idx)
            hi = maximum(m_idx)

            p = draw_bbox_pixel_space!(hm_ax.scene, lo, hi; color=cluster_cmap[mod1(c, length(cluster_cmap))])

            push!(rendered_clusters, p)
        end
    end

    function calc_cluster_bounding_box(hc, cg, idx_to_mtx, last_bBox)
        if !isnothing(last_bBox)
            delete!(parent_scene(last_bBox), last_bBox)
        end

        if intersect(hc, Set(collect(keys(cg)))) == hc && length(hc) > 0
            ts_idx = reduce(vcat, map(x -> cg[x], collect(hc)))

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
    hm_last_bBox = nothing
    @lift begin
        hm_last_bBox = calc_cluster_bounding_box($hovered_cluster, cluster_info[].groups, cluster_data[].idx_to_mtx, hm_last_bBox)
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
    dGrid[3, 2] = Colorbar(window, limits=lift(x -> x.m_extrema, cluster_data))
    rowsize!(dGrid, 3, Relative(0.65))

    band_sel = Observable(first(sort(collect(keys(per_t_scalars)))))

    x_vals = lift(x -> eachindex(x.clustering.order), cluster_data)
    colors = lift((x, z) -> map(y -> per_t_scalars[z][t_list[y]], x.clustering.order), cluster_data, band_sel)
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

    tGrid = GridLayout()
    scg = GridLayout()

    window[2:3, 2] = vgrid!(tGrid, scg)

    # Box(tGrid[1, 1], color=:black)
    scratchpad_ax = Axis(tGrid[1, 1], backgroundcolor=:black, title="Scratchpad")
    deregister_interaction!(scratchpad_ax, :rectanglezoom)
    hidedecorations!(scratchpad_ax)

    render_selection, scratchpad_render_menu = widgets["Render"](window)
    scalar_selection, scalar_menu = widgets["Scalar"](window)
    scg[1, 1] = scratchpad_render_menu
    scg[1, 2] = scalar_menu

    scratchpad!(window, scratchpad_ax, selected_transitions, cluster_info, cluster_data, render_views, render_selection, scalar_selection, hovered=hovered_transition)

    return window
end

function scratchpad!(window,
    ax,
    selected_transitions::Observable{Set{Tuple{Int,Int}}},
    cluster_info::Observable{ClusterInfo},
    cluster_data::Observable{ClusterData},
    render_views,
    render_selection,
    scalar_selection;
    hovered::MaybeObservable{Tuple{Int,Int}}=MaybeObservable{Tuple{Int,Int}}(nothing),
    on_click=(x) -> ()
)

    campixel!(ax.scene)

    selected_render = Observable("Volume")
    imgs = Observable{Vector{ColorMatrix}}(ColorMatrix[])
    colors = Observable{Vector{RGBAf}}(RGBAf[])
    points = Observable{Vector{Point2f}}(Point2f[])
    # run once on creation to bind axis
    cluster_cmap = to_colormap(CLUSTER_COLORS)

    # create hidden screen to render to
    fig = Figure()
    campixel!(fig.scene)
    render_ax = LScene(fig.scene,
        bbox=BBox(0, 100, 0, 100),
        show_axis=false,
        scenekw=(clear=true, size=(100, 100), backgroundcolor=:black))

    cam3d!(render_ax.scene)

    num_transitions = 0
    rt_to_idx = Dict{Tuple{Int,Int},Int}() # gets plotted index of transition
    idx_to_rt = Tuple{Int,Int}[] # gets transition from plotted idx
    on(selected_transitions) do st
        if length(st) != 0
            last_rendered = collect(values(rt_to_idx))

            new_points = Point2f[]
            new_colors = RGBAf[]
            new_imgs = ColorMatrix[]

            new_transitions = filter(x -> !(x in keys(rt_to_idx)), collect(st))
            t_to_mtx = cluster_data[].t_to_mtx
            assignments = cluster_info[].assignments

            for t in new_transitions
                num_transitions += 1
                rt_to_idx[t] = num_transitions
                push!(idx_to_rt, t)

                mtx_idx = t_to_mtx[t]

                buf = IOBuffer()
                config = Makie.merge_screen_config(ScreenConfig, Dict{Symbol,Any}(:visible => false))
                s = Screen(render_ax.scene, config, buf, MIME"image/png"())
                views = []
                if render_selection[] == "Volume"
                    # still a memory leak somewhere
                    vlo, vhi = render_views["Volume_no_obs"](render_ax, Observable(t))
                    views = [vlo, vhi]
                else
                    atom_s = render_views["Atom"](render_ax, t, scalar_selection, Observable(0.0))
                    views = [atom_s]
                end
                center!(render_ax.scene)
                show(buf, MIME"image/png"(), render_ax.scene, update=false)
                img = FileIO.load(Stream{FileIO.format"PNG"}(buf))

                push!(new_imgs, img)
                foreach(x -> delete!(render_ax, x), views)
                empty!(views)

                close(buf)
                close(s)

                push!(new_points, Point2f(0.0, 0.0))
                node_color = cycle_colormap(assignments[mtx_idx], cluster_cmap)
                push!(new_colors, set_color_alpha(node_color, 0.6))
            end

            all_points = vcat(points.val, new_points)


            # do this so they don't overlap
            points.val = spring(zeros(length(st), length(st)); C=1.0, pin=Dict(last_rendered .=> true), initialpos=all_points)
            colors.val = vcat(colors.val, new_colors)
            imgs.val = vcat(imgs.val, new_imgs)

            imgs[] = imgs[]
        end
    end

    onany(render_selection, scalar_selection) do rs, ss
        new_imgs = ColorMatrix[]

        for t in keys(rt_to_idx)
            buf = IOBuffer()
            config = Makie.merge_screen_config(ScreenConfig, Dict{Symbol,Any}(:visible => false))
            s = Screen(render_ax.scene, config, buf, MIME"image/png"())
            views = []
            if rs == "Volume"
                # still a memory leak somewhere
                vlo, vhi = render_views["Volume_no_obs"](render_ax, Observable(t))
                views = [vlo, vhi]
            else
                # need to pass down observable hence scalar_selection instead of ss
                atom_s = render_views["Atom"](render_ax, t, scalar_selection, Observable(0.0))
                views = [atom_s]
            end
            center!(render_ax.scene)

            show(buf, MIME"image/png"(), render_ax.scene, update=false)
            img = FileIO.load(Stream{FileIO.format"PNG"}(buf))

            push!(new_imgs, img)
            foreach(x -> delete!(render_ax, x), views)
            empty!(views)

            close(buf)
            close(s)
        end
        imgs[] = new_imgs
    end

    # forces the plot to be created only when there are points to render 
    nodes = nothing
    mp_listener = nothing
    on(imgs) do new_imgs
        # plot needs to be rebuilt each time because otherwise the underlying texture buffer is out of date
        # https://github.com/MakieOrg/Makie.jl/blob/master/GLMakie/src/glshaders/particles.jl, line 188 
        if !isnothing(mp_listener)
            off(mp_listener)
            mp_listener = nothing
            GC.gc()
        end

        if !isnothing(nodes)
            delete!(ax, nodes)
        end

        function on_hit(plt, idx, pos)
            hovered[] = idx_to_rt[idx]
            notify(hovered)
            return string(idx_to_rt[idx])
        end

        nodes = scatter!(ax, points; marker=new_imgs,
            strokecolor=colors,
            inspector_label=on_hit,
            strokewidth=5,
            markersize=lift(x -> size.(x), imgs))
    end

    selected_point = Ref(0)
    m_events = addmouseevents!(ax.scene)
    on(m_events.obs) do e
        if e.type === MouseEventTypes.leftdragstart
            plt, idx = pick(ax.scene)
            if plt == nodes
                selected_point[] = idx
            else
                selected_point[] = 0
            end
        elseif e.type == MouseEventTypes.leftdrag
            # will not prevent dragging off the scene 
            if selected_point[] != 0
                points[][selected_point[]] = mouseposition(ax)
                notify(points)
            end
        elseif e.type == MouseEventTypes.leftdragstop
            selected_point[] = 0
        elseif e.type == MouseEventTypes.over
            plt, idx = pick(ax.scene)
            if plt isa Makie.Mesh && !isnothing(hovered[])
                hovered[] = nothing
                notify(hovered)
            end
        elseif e.type == MouseEventTypes.leftdoubleclick
            # add textbox at point
            x, y = mouseposition_px(window.scene)

            txt = Textbox(window.scene, bbox=BBox(x, x + 50, y, y + 50), placeholder="...", textcolor=:white, focused=true)
            px, py = mouseposition(ax)
            on(txt.stored_string) do s
                text!(ax.scene, px, py; text=s, color=:white)
            end

            on(txt.focused) do is_focused
                if !is_focused
                    delete!(txt)
                end
            end

        end
    end

    highlighted = []
    on(hovered) do hov
        for (h, ogCol) in highlighted
            colors.val[h] = ogCol
        end
        empty!(highlighted)

        if !isnothing(hov) && hov in keys(rt_to_idx)
            idx = rt_to_idx[hov]
            ogColor = colors.val[idx]
            colors.val[idx] = set_color_alpha(ogColor, 1.0)
            push!(highlighted, (idx, ogColor))
        end

        colors[] = colors[]
        notify(colors)
    end
end

function umap_graph_view!(window, umap_ax,
    reordered_matrix,
    cluster_info,
    cluster_cmap,
    t_to_idx,
    render_views,
    hovered=Observable(Set{Int}(1));
    on_click=(x) -> (),
)
    campixel!(umap_ax.scene)

    selected_render = Observable("Volume")

    umap_cluster_idx = Observable(sort(collect(keys(cluster_representatives[]))))
    umap_colors = Observable(map(x -> cluster_cmap[mod1(x, length(cluster_cmap))], umap_cluster_idx[]))

    function on_hover(plt, idx, pos)
        hovered[] = Set{Int}(umap_cluster_idx[][idx])
        notify(hovered)
        return string(umap_cluster_idx[][idx])
    end

    highlighted = []
    on(hovered) do hov
        for (h, ogCol) in highlighted
            umap_colors.val[h] = ogCol
        end
        empty!(highlighted)

        for c in collect(hov)
            ogColor = umap_colors.val[c]
            umap_colors.val[c] = set_color_alpha(ogColor, 1.0)
            push!(highlighted, (c, ogColor))
        end

        umap_colors[] = umap_colors[]
        notify(umap_colors)
    end

    imgs = Observable(map(x -> Matrix{ColorTypes.RGB{FixedPointNumbers.N0f8}}(undef, 100, 100), sort(collect(keys(cluster_representatives[])))))

    render_ax = LScene(umap_ax.scene,
        show_axis=false,
        bbox=BBox(0, 100, 0, 100),
        scenekw=(backgroundcolor=:black, clear=true, size=(100, 100))
    )

    @time embedding = @lift begin
        println("Computing umap embedding...")

        dm = $reordered_matrix[1]
        t_to_mtx = $reordered_matrix[4]

        new_cluster_idx = sort(collect(keys($cluster_representatives)))
        umap_cluster_idx.val = new_cluster_idx

        reps = map(x -> $cluster_representatives[x], new_cluster_idx)
        mtx_idx = map(x -> t_to_mtx[x], reps)
        rep_mat = reduce(hcat, map(x -> dm[x, :][mtx_idx], mtx_idx))

        em = transpose(umap(transpose(rep_mat), 2; metric=:precomputed, n_neighbors=min(15, length(reps) - 1)))
        new_colors = map(x -> set_color_alpha(cluster_cmap[mod1(x, length(cluster_cmap))], 0.6), new_cluster_idx)
        umap_colors.val = new_colors

        render_ax.scene.visible[] = true
        println("Rendering representatives...")
        new_imgs = []
        cs = sort(collect(keys($cluster_representatives)))
        @time for c_idx in cs
            t = $cluster_representatives[c_idx]
            idx = t_to_idx[t]
            buf = IOBuffer()
            cam3d!(render_ax.scene)

            if selected_render[] == "Volume"
                # still a memory leak somewhere
                render_views["Volume_no_obs"](render_ax, idx, t)
                center!(render_ax.scene)
            end

            show(buf, MIME"image/png"(), render_ax.scene, update=false)
            push!(new_imgs, FileIO.load(Stream{FileIO.format"PNG"}(buf)))
            empty!(render_ax.scene)
            close(buf)
        end
        GC.gc()
        render_ax.scene.visible[] = false

        imgs.val = new_imgs
        return map(x -> Point2f(x), eachrow(em))
    end

    MIN_SIZE = 10.0
    MAX_SIZE = 100.0

    marker_size = Observable(MIN_SIZE)

    og_xlim = Observable(umap_ax.xaxis.attributes.limits[])
    og_ylim = Observable(umap_ax.yaxis.attributes.limits[])

    umap_nodes = scatter!(umap_ax, embedding; inspector_label=on_hover, marker=imgs, markersize=marker_size, strokecolor=umap_colors, strokewidth=5)

    on(embedding, update=true) do e
        umap_cluster_idx[] = umap_cluster_idx[]
        umap_colors[] = umap_colors[]
        imgs[] = imgs[]
        notify(umap_cluster_idx)
        notify(umap_colors)
        notify(imgs)

        reset_limits!(umap_ax)
        og_xlim[] = umap_ax.xaxis.attributes.limits[]
        og_ylim[] = umap_ax.yaxis.attributes.limits[]

        marker_size[] = MIN_SIZE
        notify(marker_size)
    end

    on(events(umap_ax.scene).mousebutton) do event
        if is_mouseinside(umap_ax.scene)
            if event.button == Mouse.left && event.action == Mouse.press
                on_click(Set(hovered[]))
            end
        end
    end

    function calc_size(xlim)
        og_x_extent = (og_xlim[][2] - og_xlim[][1])
        x_extent = (xlim[2] - xlim[1])
        x_size = MIN_SIZE * (1.0 / (x_extent / og_x_extent))

        return round(min(max(MIN_SIZE, x_size), MAX_SIZE))
    end

    onany(umap_ax.xaxis.attributes.limits, umap_ax.scene.camera.resolution) do xlim, res
        size_px = calc_size(xlim)
        marker_size[] = size_px
        notify(marker_size)
    end

    return umap_nodes
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
                gg, time, scalar_vals = widgets["Atom"](0.0, fig)
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
