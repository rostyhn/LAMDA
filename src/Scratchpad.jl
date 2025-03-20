using Observables

function scratchpad!(
    window,
    loc,
    selected_transitions::Observable{Set{Tuple{Int,Int}}},
    cluster_info::Observable{ClusterInfo},
    cluster_data::Observable{ClusterData},
    render_views,
    render_selection,
    scalar_selection,
    atom_time,
    selected_clusters,
    t_list,
    rel_t_to_idx;
    hovered::MaybeObservable{Tuple{Int,Int}}=MaybeObservable{Tuple{Int,Int}}(nothing),
    hovered_cluster::MaybeObservable{Set{Int}},
    on_hover,
    on_click=(x) -> (),
    markersize=150
)
    ax = Axis(loc, backgroundcolor=:transparent, title="Scratchpad")
    deregister_interaction!(ax, :rectanglezoom)
    hidedecorations!(ax)
    campixel!(ax.scene)

    points = Observable{Vector{Point2f}}(Point2f[Point2f(0.0)])
    # run once on creation to bind axis
    reset_limits!(ax)
    center!(ax.scene)

    ins = DataInspector(ax)

    cluster_cmap = to_colormap(CLUSTER_COLORS)

    idx_to_obj = Observable{Vector{Union{Set{Int},Tuple{Int,Int}}}}(Union{Set{Int},Tuple{Int,Int}}[]) # gets transition from plotted idx
    obj_to_idx = Ref(Dict{Union{Set{Int},Tuple{Int,Int}},Int}())
    rendered_idxes = Ref(Set{Int}())
    num_objs = Ref(1)
    views = Ref([])

    function delete_obj!(obj, rendered_idxes, views, obj_to_idx, num_objs)
        plt_idx = obj_to_idx[][obj]
        v_idx = plt_idx - 1

        scene, vp, size, listener, mouse_listener, scene_color = views[][v_idx]
        if !isnothing(listener)
            off(listener)
            listener = nothing
        end
        off(mouse_listener)
        mouse_listener = nothing

        empty!(scene)
        scene.visible[] = false

        Observables.clear(vp)
        Observables.clear(size)
        Observables.clear(scene_color)
        delete!(obj_to_idx[], obj)
    end

    onany(selected_transitions, selected_clusters) do st, sc
        if length(st) != 0 || length(sc) != 0
            last_rendered = vcat([1], collect(rendered_idxes[]))
            new_points = Point2f[]
            new_objs = filter(x -> !(x in keys(obj_to_idx[])), vcat(collect(st), collect(sc)))
            for obj in new_objs
                num_objs[] += 1
                push!(new_points, Point2f(0.0, 0.0))
                push!(idx_to_obj.val, obj)
                obj_to_idx[][obj] = num_objs[]
            end

            all_points = vcat(points.val, new_points)

            # do this so they don't overlap
            points.val = spring(zeros(length(all_points), length(all_points)); C=0.1, pin=Dict(last_rendered .=> true), initialpos=all_points)
            notify(points)
            notify(idx_to_obj)
        end
    end


    on(cluster_info) do ci
        clusters = filter(x -> x isa Set{Int}, collect(keys(obj_to_idx[])))
        for c in clusters
            delete_obj!(c, rendered_idxes, views, obj_to_idx, num_objs)
        end

        assignments = ci.assignments
        #=for t in keys(rt_to_idx)
            plt_idx = rt_to_idx[t]
            t_idx = rel_t_to_idx[t]
            node_color = cycle_colormap(assignments[t_idx], cluster_cmap)
            colors.val[plt_idx] = set_color_alpha(node_color, 0.6)
        end=#
        empty!(selected_clusters[])
        selected_clusters[] = selected_clusters[]
    end

    function on_hit(plt, idx, pos)
        obj = idx_to_obj[][idx-1]
        s = string(obj)
        if obj isa Set{Int}
            s = str_limit(obj)
        end
        return s
    end

    nodes = scatter!(ax, points, marker=:rect, visible=false, inspector_label=on_hit)

    marker_4d = Point4f(markersize, markersize, 0, 0)

    on(idx_to_obj) do idxes
        for (idx, obj) in enumerate(idxes)
            plt_idx = idx + 1
            if !(plt_idx in rendered_idxes[])
                pos = position_on_plot(nodes, plt_idx, apply_transform=false)
                # x, y is in global pixel coords
                x, y = shift_project(ax.scene, apply_transform_and_model(nodes, pos))
                # calculate shifted size of marker
                ms = Int.(round.(ax.scene.camera.projectionview[] * marker_4d))[1]
                size = Observable((ms, ms))

                vp = Observable(Rect2i(x - (ms / 2), y - (ms / 2),
                    ms,
                    ms))

                scene_color = Observable(:black)

                ax3d = Scene(ax.scene, show_axis=false,
                    viewport=vp,
                    backgroundcolor=scene_color,
                    clear=true,
                    size=size)

                on(scene_color) do sc
                    ax3d.backgroundcolor[] = to_color(sc)
                end

                cam3d!(ax3d)
                translate!(ax3d, 0, 0, 100)

                inspector = DataInspector(ax3d)
                listener = nothing
                mouse_listener = nothing

                m_events = addmouseevents!(ax3d)

                mouse_listener = on(m_events.obs) do event
                    i = obj_to_idx[][obj]
                    if event.type === MouseEventTypes.over
                        show_data(ins, nodes, i)
                        if obj isa Tuple{Int,Int}
                            hovered[] = obj
                            t_idx = rel_t_to_idx[obj]
                            cluster = Set(cluster_info[].assignments[t_idx])
                            notify(hovered)
                        else
                            cluster = obj
                        end
                        hovered_cluster[] = cluster
                        notify(hovered_cluster)
                    elseif event.type === MouseEventTypes.middledrag
                        points[][i] = mouseposition(ax)
                        notify(points)
                    elseif event.type === MouseEventTypes.rightdown
                        hovered_cluster[] = nothing
                        notify(hovered_cluster)

                        hovered[] = nothing
                        notify(hovered)
                        if obj isa Tuple{Int,Int}
                            delete!(selected_transitions[], obj)
                            notify(selected_transitions)
                        else
                            delete!(selected_clusters[], obj)
                            notify(selected_clusters)
                        end

                        delete_obj!(obj, rendered_idxes, views, obj_to_idx, num_objs)
                        notify(idx_to_obj)
                        notify(points)

                    elseif event.type === MouseEventTypes.leftdoubleclick
                        cluster = obj
                        if obj isa Tuple{Int,Int}
                            t_idx = rel_t_to_idx[obj]
                            cluster = Set(cluster_info[].assignments[t_idx])
                        end
                        on_click(cluster)
                    end
                end

                if obj isa Tuple{Int,Int}
                    plt_rendered = []
                    listener = on(render_selection, update=true) do rs
                        foreach(x -> delete!(ax3d, x), plt_rendered)
                        empty!(plt_rendered)
                        if rs == "Volume"
                            v_lo, v_hi = render_views[rs](ax3d, Observable(obj))
                            plt_rendered = [v_lo, v_hi]
                        elseif rs == "Atom"
                            s = render_views["Atom"](ax3d, Observable(obj),
                                scalar_selection,
                                atom_time,
                                lift(x -> collect(x), selected_transitions))
                            plt_rendered = [s]
                        else
                            il, is, plots = render_views["Superquadric"](ax3d,
                                Observable(obj),
                                inspector,
                                lift(x -> collect(x), selected_transitions)
                            )
                            plt_rendered = plots
                        end
                        center!(ax3d)
                    end
                else
                    render_views["CMovement"](ax3d, obj, atom_time)
                    center!(ax3d)
                end
                push!(views[], (ax3d, vp, size, listener, mouse_listener, scene_color))
                push!(rendered_idxes[], plt_idx)
            end
        end
    end

    onany(ax.xaxis.attributes.limits, ax.yaxis.attributes.limits, points) do xlim, ylim, pp
        ms = Int.(round.(ax.scene.camera.projectionview[] * marker_4d))[1]
        for (i, (ax3d, vp, size, listener, mouse_listener)) in enumerate(views[])
            plt_idx = i + 1
            pos = position_on_plot(nodes, plt_idx, apply_transform=false)
            x, y = shift_project(ax.scene, apply_transform_and_model(nodes, pos))

            vp.val = Rect2i(x - ms[], y - ms[], ms[], ms[])
            vp.val = GeometryBasics.intersect(vp.val, ax.scene.viewport[])
            vw = widths(vp.val)

            if any(w -> w <= 0, vw)
                vp.val = Rect2i(0, 0, 0, 0)
            end

            notify(vp)
            notify(size)
        end
    end

    d_start = Point2f(0.0)
    d_end = Point2f(0.0)
    bbox = Observable(BBox(0, 0, 0, 0))
    m_events = addmouseevents!(ax.scene)
    boxes = []
    on(m_events.obs) do e
        if e.type == MouseEventTypes.over
            plt, idx = pick(ax.scene)
            if isnothing(plt)
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
        elseif e.type === MouseEventTypes.leftdragstart
            d_start = mouseposition(ax.scene)
            w = wireframe!(ax.scene, bbox, color=:red)
            w.inspectable[] = false
        elseif e.type === MouseEventTypes.leftdrag
            d_end = mouseposition(ax.scene)
            l = (d_start[1] < d_end[1]) ? d_start[1] : d_end[1]
            r = (l == d_start[1]) ? d_end[1] : d_start[1]

            b = (d_start[2] < d_end[2]) ? d_start[2] : d_end[2]
            t = (b == d_start[2]) ? d_end[2] : d_start[2]
            bbox[] = BBox(l, r, b, t)
        elseif e.type === MouseEventTypes.leftdragstop
            push!(boxes, bbox[])
            bbox = Observable(BBox(0, 0, 0, 0))
        elseif e.type == MouseEventTypes.leftdoubleclick
            plt, idx = pick(ax.scene)
            if isnothing(plt)
                # add textbox at point if empty space was clicked
                x, y = mouseposition_px(window.scene)

                txt = Textbox(window.scene, bbox=BBox(x, x + 50, y, y + 50),
                    placeholder="...",
                    textcolor=:black,
                    focused=true)

                px, py = mouseposition(ax)
                on(txt.stored_string) do s
                    text!(ax.scene, px, py; text=s, color=:black)
                end

                on(txt.focused) do is_focused
                    if !is_focused
                        delete!(txt)
                    end
                end
            end
        end
    end

    highlighted = []
    on(hovered) do hov
        for (v_idx) in highlighted
            views[][v_idx][6][] = :black
        end
        empty!(highlighted)

        if !isnothing(hov) && hov in keys(obj_to_idx[])
            v_idx = obj_to_idx[][hov] - 1
            views[][v_idx][6][] = :grey
            push!(highlighted, v_idx)
        end
    end

    highlighted_clusters = []
    on(hovered_cluster) do hov
        for (v_idx) in highlighted_clusters
            views[][v_idx][6][] = :black
        end
        empty!(highlighted_clusters)

        if !isnothing(hov) && hov in keys(obj_to_idx[])
            v_idx = obj_to_idx[][hov] - 1
            views[][v_idx][6][] = :grey
            push!(highlighted_clusters, v_idx)
        end
    end
end
