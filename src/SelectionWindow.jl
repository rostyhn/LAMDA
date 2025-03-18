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
    rel_t_to_idx::Dict{Tuple{Int16,Int16},Int},
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

    init_clusters = Set{Set{Int}}()
    selected_clusters = Observable{Set{Set{Int}}}(init_clusters)

    # will complain about being passed "nothing" as a value if something isn't inside the set
    hovered_cluster = MaybeObservable{Set{Int}}(Set{Int}(1))

    # used to place transitions into scratchpad
    function on_transition_select(t)
        push!(selected_transitions[], t)
        notify(selected_transitions)
    end

    function on_cluster_select(c)
        push!(selected_clusters[], c)
        notify(selected_clusters)
    end

    bins = @lift begin
        return $h_range[1]:1:($h_range[2]+1)
    end

    function on_cluster_window_hover(c)
        if !isnothing(c)
            hovered_cluster[] = c
            notify(hovered_cluster)
        else
            if !isnothing(hovered_cluster[])
                hovered_cluster.val = nothing
                hovered_cluster[] = hovered_cluster[]
                notify(hovered_cluster)
            end
        end
    end

    open_cluster_windows = Dict{Set{Int},Screen}()
    function on_show_cluster_click(clusters)
        if !(clusters in keys(open_cluster_windows))

            ref_t = find_group_centroid(clusters, cluster_data[], cluster_info[], t_list)

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
                ref_t,
                collect(eachindex(ts_idx_to_mtx_idx)),
                vals,
                scalars,
                cluster_data[].m_extrema,
                render_views,
                widgets,
                bins,
                on_transition_select,
                hovered_transition,
                ds,
                on_window_hover=on_cluster_window_hover,
                on_cluster_select=on_cluster_select
            )
            s = GLMakie.Screen(title="Cluster $(str_limit(clusters))")
            display(s, w)

            # create inspector after render to avoid bugs
            ds[] = DataInspector(w)
            open_cluster_windows[clusters] = s
        end
    end

    # close cluster views if clustering changes
    on(cluster_info) do c
        foreach(s -> close(s), values(open_cluster_windows))
        empty!(open_cluster_windows)
    end

    cutoff_tb = Textbox(window, validator=Float64, placeholder=string(h_cutoff[]))
    on(cutoff_tb.stored_string) do s
        # reset hovered_cluster to avoid crashing
        hovered_cluster.val = nothing
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
        cluster_info,
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
                ord = cluster_data[].mtx_to_t
                xy = mouseposition(hm_ax)
                i, j = Int.(round.(xy))
                t1 = ord[i]
                t2 = ord[j]
                push!(selected_transitions[], t1)

                if t1 != t2
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

    function calc_cluster_bounding_box(hc, cg, idx_to_mtx, last_bBox::Maybe{Wireframe{Tuple{GeometryBasics.HyperRectangle{2,Float64}}}})::Maybe{Wireframe{Tuple{GeometryBasics.HyperRectangle{2,Float64}}}}
        if !isnothing(last_bBox)
            delete!(parent_scene(last_bBox), last_bBox)
        end

        if !isnothing(hc) && intersect(hc, Set(collect(keys(cg)))) == hc && length(hc) > 0
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
        return draw_bbox_pixel_space!(hm_ax.scene, 0, 0; width=3)
    end

    # https://github.com/MakieOrg/Makie.jl/blob/master/src/interaction/inspector.jl
    hm_last_bBox::Maybe{Wireframe{Tuple{GeometryBasics.HyperRectangle{2,Float64}}}} = nothing
    @lift begin
        res = calc_cluster_bounding_box($hovered_cluster, cluster_info[].groups, cluster_data[].idx_to_mtx, hm_last_bBox)
        hm_last_bBox = res
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
    #scratchpad_time, scratchpad_t_slider = widgets["Movement"](0.0, window)
    #scg[2, 1:2] = scratchpad_t_slider

    function on_scratchpad_hover(t)
        if t isa Tuple{Int,Int}
            t_idx = rel_t_to_idx[t]
            hovered_cluster[] = Set(cluster_info[].assignments[t_idx])
        else
            hovered_cluster[] = t
        end
        notify(hovered_cluster)
    end

    scratchpad!(window,
        scratchpad_ax,
        selected_transitions,
        cluster_info,
        cluster_data,
        render_views,
        render_selection,
        scalar_selection,
        selected_clusters,
        t_list,
        rel_t_to_idx,
        on_hover=on_scratchpad_hover,
        hovered_cluster=hovered_cluster,
        hovered=hovered_transition)

    return window
end

function scratchpad!(window,
    ax,
    selected_transitions::Observable{Set{Tuple{Int,Int}}},
    cluster_info::Observable{ClusterInfo},
    cluster_data::Observable{ClusterData},
    render_views,
    render_selection,
    scalar_selection,
    selected_clusters,
    t_list,
    rel_t_to_idx;
    hovered::MaybeObservable{Tuple{Int,Int}}=MaybeObservable{Tuple{Int,Int}}(nothing),
    hovered_cluster::MaybeObservable{Set{Int}},
    on_hover,
    on_click=(x) -> ()
)

    campixel!(ax.scene)

    imgs = Observable{Vector{ColorMatrix}}(ColorMatrix[])
    colors = Observable{Vector{RGBAf}}(RGBAf[])
    points = Observable{Vector{Point2f}}(Point2f[])
    # run once on creation to bind axis
    cluster_cmap = to_colormap(CLUSTER_COLORS)

    IMG_SIZE = 200
    # create hidden screen to render to
    fig = Figure()
    campixel!(fig.scene)
    render_ax = LScene(fig.scene,
        bbox=BBox(0, IMG_SIZE, 0, IMG_SIZE),
        show_axis=false,
        scenekw=(clear=true, size=(IMG_SIZE, IMG_SIZE), backgroundcolor=:black))

    cam3d!(render_ax.scene)

    num_objs = 0
    c_to_idx = Dict{Set{Int},Int}() # gets plotted index of centroid
    rt_to_idx = Dict{Tuple{Int,Int},Int}() # gets plotted index of transition
    idx_to_obj = Union{Set{Int},Tuple{Int,Int}}[] # gets transition from plotted idx

    function delete_obj!(obj)
        num_objs -= 1
        plt_idx = 0
        rel_dict = rt_to_idx
        if obj isa Set{Int}
            rel_dict = c_to_idx
        end
        plt_idx = rel_dict[obj]
        delete!(rel_dict, obj)

        points.val = deleteat!(points.val, plt_idx)
        colors.val = deleteat!(colors.val, plt_idx)
        imgs.val = deleteat!(imgs.val, plt_idx)
        idx_to_obj = deleteat!(idx_to_obj, plt_idx)
        empty!(c_to_idx)
        empty!(rt_to_idx)

        for (new_idx, obj) in enumerate(idx_to_obj)
            if obj isa Set{Int}
                c_to_idx[obj] = new_idx
            else
                rt_to_idx[obj] = new_idx
            end
        end
    end

    on(selected_transitions) do st
        if length(st) != 0
            last_rendered = vcat(collect(values(rt_to_idx)), collect(values(c_to_idx)))

            new_points = Point2f[]
            new_imgs = ColorMatrix[]
            new_colors = RGBAf[]

            new_transitions = filter(x -> !(x in keys(rt_to_idx)), collect(st))
            #t_to_mtx = cluster_data[].t_to_mtx
            assignments = cluster_info[].assignments

            for t in new_transitions
                num_objs += 1
                rt_to_idx[t] = num_objs
                push!(idx_to_obj, t)

                mtx_idx = rel_t_to_idx[t]

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
            points.val = spring(zeros(length(all_points), length(all_points)); C=0.1, pin=Dict(last_rendered .=> true), initialpos=all_points)

            colors.val = vcat(colors.val, new_colors)
            imgs.val = vcat(imgs.val, new_imgs)

            imgs[] = imgs[]
        end
    end

    on(selected_clusters) do sc
        if length(sc) != 0
            last_rendered = vcat(collect(values(rt_to_idx)), collect(values(c_to_idx)))

            new_points = Point2f[]
            new_imgs = ColorMatrix[]
            new_colors = RGBAf[]

            new_clusters = filter(x -> !(x in keys(c_to_idx)), collect(sc))

            for c in new_clusters
                num_objs += 1
                c_to_idx[c] = num_objs
                push!(idx_to_obj, c)

                buf = IOBuffer()
                config = Makie.merge_screen_config(ScreenConfig, Dict{Symbol,Any}(:visible => false))
                s = Screen(render_ax.scene, config, buf, MIME"image/png"())

                # not sure why cluster_info never gets updated here
                g = reduce(vcat, map(x -> cluster_info[].groups[x], collect(c)))
                ts = map(x -> t_list[x], g)
                av = render_views["SMovement"](render_ax, ts, Observable(0.0))
                center!(render_ax.scene)
                show(buf, MIME"image/png"(), render_ax.scene, update=false)
                img = FileIO.load(Stream{FileIO.format"PNG"}(buf))

                push!(new_imgs, img)
                delete!(render_ax, av)

                close(buf)
                close(s)

                push!(new_points, Point2f(0.0, 0.0))
                node_color = to_color(:grey)
                if length(c) == 1
                    node_color = cycle_colormap(first(collect(c)), cluster_cmap)
                end
                push!(new_colors, set_color_alpha(node_color, 0.6))

            end

            all_points = vcat(points.val, new_points)
            points.val = spring(zeros(length(all_points), length(all_points)); C=0.1, pin=Dict(last_rendered .=> true), initialpos=all_points)
            colors.val = vcat(colors.val, new_colors)
            imgs.val = vcat(imgs.val, new_imgs)

            imgs[] = imgs[]

        end
    end

    on(cluster_info) do ci
        for c in keys(c_to_idx)
            delete_obj!(c)
        end

        assignments = ci.assignments
        for t in keys(rt_to_idx)
            plt_idx = rt_to_idx[t]
            t_idx = rel_t_to_idx[t]
            node_color = cycle_colormap(assignments[t_idx], cluster_cmap)
            colors.val[plt_idx] = set_color_alpha(node_color, 0.6)
        end
        empty!(selected_clusters[])
        selected_clusters[] = selected_clusters[]
        # only trigger imgs if not empty, otherwise it'll try to render 
        if !isempty(imgs[])
            imgs[] = imgs[]
        end
    end


    onany(render_selection, scalar_selection) do rs, ss
        for t in keys(rt_to_idx)
            plt_idx = rt_to_idx[t]
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

            foreach(x -> delete!(render_ax, x), views)
            empty!(views)

            close(buf)
            close(s)
            imgs.val[plt_idx] = img
        end
        imgs[] = imgs[]
    end

    # attempt to control marker size, works fine but limits constantly get reset which is annoying
    # using markerspace = :data causes aspect ratio warping
    MIN_SIZE = 50
    MAX_SIZE = 200

    og_xlim::MaybeObservable{Any} = Observable(nothing)
    marker_size = Observable(MIN_SIZE)
    function calc_size(xlim)
        og_x_extent = (og_xlim[][2] - og_xlim[][1])
        x_extent = (xlim[2] - xlim[1])
        x_size = MIN_SIZE * (1.0 / (x_extent / og_x_extent))

        return round(min(max(MIN_SIZE, x_size), MAX_SIZE))
    end

    on(events(ax.scene).scroll, priority=1) do (dx, dy)
        xlim = ax.xaxis.attributes.limits[]
        if !isnothing(og_xlim[])
            size_px = calc_size(xlim)
            marker_size[] = size_px
            notify(marker_size)
        end
        return Consume(false)
    end

    function on_hit(plt, idx, pos)
        obj = idx_to_obj[idx]
        s = string(obj)
        if obj isa Tuple{Int,Int}
            hovered[] = obj
            notify(hovered)
        else
            s = str_limit(obj)
        end
        on_hover(obj)
        return s
    end

    # forces the plot to be created only when there are points to render 
    nodes = nothing
    on(imgs) do new_imgs
        hovered_cluster[] = nothing
        notify(hovered_cluster)
        # plot needs to be rebuilt each time because otherwise the underlying texture buffer is out of date
        # https://github.com/MakieOrg/Makie.jl/blob/master/GLMakie/src/glshaders/particles.jl, line 188 

        if !isnothing(nodes)
            delete!(ax, nodes)
        end

        new_nodes = scatter!(ax, points;
            marker=new_imgs,
            strokecolor=colors,
            inspector_label=on_hit,
            strokewidth=5,
            markersize=marker_size)

        if isnothing(nodes)
            og_xlim[] = ax.xaxis.attributes.limits[]
        end
        nodes = new_nodes
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
            if plt isa Makie.Mesh
                if !isnothing(hovered[])
                    hovered[] = nothing
                    notify(hovered)
                end
                if !isnothing(hovered_cluster[])
                    hovered_cluster[] = nothing
                    notify(hovered_cluster)
                end
                # can use this to select the text and do stuff
                #elseif plt isa Makie.Text
                #    @show plt
            end
        elseif e.type == MouseEventTypes.leftdoubleclick
            # add textbox at point
            x, y = mouseposition_px(window.scene)

            txt = Textbox(window.scene, bbox=BBox(x, x + 50, y, y + 50), placeholder="...", textcolor=:white, focused=true)
            px, py = mouseposition(ax)
            on(txt.stored_string) do s
                text!(ax.scene, px, py; text=s, color=:white, markerspace=:data)
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

    # can cause segfaults, probably cause so much is happening 
    #=on(hovered_cluster) do hc
        for (h, ogCol) in highlighted
            colors.val[h] = ogCol
        end
        empty!(highlighted)

        if !isnothing(hc) && hc in keys(c_to_idx)
            idx = c_to_idx[hc]
            ogColor = colors.val[idx]
            colors.val[idx] = set_color_alpha(ogColor, 1.0)
            push!(highlighted, (idx, ogColor))
        end
        colors[] = colors[]
        notify(colors)
    end=#
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
