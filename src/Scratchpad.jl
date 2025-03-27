using Observables
const TITLE_HOTKEY = Keyboard.t

function scratchpad!(
    window,
    loc,
    selected_transitions::Observable{Set{Transition}},
    cluster_info::Observable{ClusterInfo},
    cluster_data::Observable{ClusterData},
    render_views,
    render_selection,
    scalar_selection,
    atom_time,
    selected_clusters,
    t_list,
    rel_t_to_idx,
    calculators,
    cluster_annotations,
    window_inspector,
    c2corr;
    hovered::MaybeObservable{Transition}=MaybeObservable{Transition}(nothing),
    hovered_cluster::MaybeObservable{Set{Int}},
    on_click=(x) -> (),
    markersize=200)

    ax = Axis(loc, backgroundcolor=:transparent, title="Scratchpad")
    deregister_interaction!(ax, :rectanglezoom)
    hidedecorations!(ax)
    campixel!(ax.scene)

    points = Observable{Vector{Point2f}}(Point2f[Point2f(0.0)])
    # run once on creation to bind axis
    reset_limits!(ax)
    center!(ax.scene)

    ins = DataInspector(ax)

    # could be a dictionary, helps with tracking and don't have to worry about setting idx
    idx_to_obj = Observable{Vector{Union{Set{Int},Transition}}}(Union{Set{Int},Transition}[]) # gets transition from plotted idx
    obj_to_idx = Ref(Dict{Union{Set{Int},Transition},Int}())
    rendered_idxes = Ref(Set{Int}())
    num_objs = Ref(1)
    views = Ref([])

    d_start = Point2f(0.0)
    d_end = Point2f(0.0)
    c_bbox = Observable(BBox(0, 0, 0, 0))
    boxes = Ref(Dict{Int,Any}())
    sw = wireframe!(ax.scene, c_bbox, color=:black, visible=false)
    sw.inspectable[] = false

    register_interaction!(ax, :create_group) do e::MouseEvent, axis
        if e.type === MouseEventTypes.leftdragstart
            d_start = mouseposition(ax.scene)
            sw.visible[] = true
        elseif e.type === MouseEventTypes.leftdrag
            d_end = mouseposition(ax.scene)
            l = (d_start[1] < d_end[1]) ? d_start[1] : d_end[1]
            r = (l == d_start[1]) ? d_end[1] : d_start[1]

            b = (d_start[2] < d_end[2]) ? d_start[2] : d_end[2]
            t = (b == d_start[2]) ? d_end[2] : d_start[2]
            c_bbox[] = BBox(l, r, b, t)
            notify(c_bbox)
        elseif e.type === MouseEventTypes.leftdragstop
            # finish placing box
            w = poly!(ax.scene, c_bbox[], color=:transparent, strokewidth=2, strokecolor=:black)
            w.inspectable[] = false

            s_idx = length(ax.scene.plots)
            boxes[][s_idx] = c_bbox[]
            sw.visible[] = false
            c_bbox[] = BBox(0, 0, 0, 0)
        end
    end

    txt_to_notes = Ref(Dict{Int,Any}())
    register_interaction!(ax, :create_text) do e::MouseEvent, axis
        if e.type == MouseEventTypes.leftdoubleclick
            plt, idx = pick(ax.scene)
            if isnothing(plt)
                # add textbox at point if empty space was clicked
                x, y = mouseposition_px(window.scene)
                is_title = ispressed(ax.scene, TITLE_HOTKEY)

                txt = Textbox(window.scene,
                    bbox=BBox(x, x + 50, y, y + 50),
                    placeholder=" ",
                    textcolor=:black,
                    focused=true)

                pos = mouseposition(ax)
                on(txt.stored_string) do s
                    text!(ax.scene, pos; text=s, color=:black, font=(is_title) ? :bold : :regular)
                    s_idx = length(ax.scene.plots)
                    txt_to_notes[][s_idx] = (pos, s, is_title)
                end

                on(txt.focused) do is_focused
                    if !is_focused
                        delete!(txt)
                    end
                end
            end
        end
    end

    ax_m_events = addmouseevents!(ax.scene)
    on(ax_m_events.obs) do e
        if e.type == MouseEventTypes.over
            plt, idx = pick(ax.scene)
            if plt isa Makie.Text
                plt.color[] = to_color(:grey)
            end
        elseif e.type == MouseEventTypes.rightclick
            plt, idx = pick(ax.scene)
            if plt isa Makie.Text
                delete!(ax.scene, plt)
                delete!(txt_to_notes[], idx)
            elseif plt isa Makie.Lines
                # pick returns Lines instead of the polygon itself, super annoying
                for (i, abplot) in enumerate(ax.scene.plots)
                    if plt in abplot.plots
                        delete!(boxes[], i)
                    end
                end
                delete!(ax.scene, plt)
            end
        end
        return Consume(false)
    end

    function delete_obj!(obj, rendered_idxes, views, obj_to_idx, num_objs)
        plt_idx = obj_to_idx[][obj]
        v_idx = plt_idx - 1
        scene = views[][v_idx]

        Makie.free(scene)
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

    nodes = scatter!(ax, points, marker=:rect, visible=false)
    nodes.inspectable[] = false

    marker_4d = Point4f(markersize, markersize, 0, 0)

    # guarantees updated version of clusterinfo
    function get_t_cluster(obj)
        # gets current cluster assigned to idx
        idx = rel_t_to_idx[obj]
        x = lift(y -> y.a2c[y.assignments[idx]], cluster_info)
        xval = x[]
        Observables.clear(x)
        return xval
    end

    function obj_to_str(obj)
        s = string(obj)
        if obj isa Set{Int}
            s = str_limit(get_val(cluster_annotations[], "titles", obj))
        end
        return s
    end

    function show_inspector(obj)
        tt = window_inspector[].plot
        s = obj_to_str(obj)
        mp = mouseposition(ax.scene)
        smp = shift_project(ax.scene, apply_transform_and_model(nodes, mp))
        update_tooltip_alignment!(window_inspector[], smp)

        tt.text[] = s
        tt.visible[] = true
    end

    function hide_inspector()
        tt = window_inspector[].plot
        tt.visible[] = false
    end

    viewports = Ref([])
    frame_colors = Ref([])
    on(idx_to_obj) do idxes
        ts = collect(selected_transitions[])
        s_alignment = Observable(calculators["Alignment"](ts))
        for (idx, obj) in enumerate(idxes)
            plt_idx = idx + 1
            if !(plt_idx in rendered_idxes[])
                pos = position_on_plot(nodes, plt_idx, apply_transform=false)
                # x, y is in global pixel coords
                x, y = shift_project(ax.scene, apply_transform_and_model(nodes, pos))
                # calculate shifted size of marker
                ms = Int.(round.(ax.scene.camera.projectionview[] * marker_4d))[1]
                vp = Rect2i(x - (ms / 2), y - (ms / 2), ms, ms)

                # viewports need to be in data space
                push!(viewports[], pos)

                ax3d = Scene(ax.scene, show_axis=false,
                    viewport=vp,
                    backgroundcolor=EMBEDDED_SCENE_BACKGROUND,
                    clear=true,
                    size=(ms, ms))

                cam3d!(ax3d)
                translate!(ax3d, 0, 0, 100)

                frame_color = @lift begin
                    if obj isa Transition
                        # get the currently assigned color of this transition
                        return set_color_alpha(cluster_color($(cluster_info), obj), 0.6)
                    else
                        # gets the assigned cluster color
                        return set_color_alpha(cluster_color($(cluster_data), obj), 0.6)
                    end
                end

                frame = wireframe!(
                    ax3d,
                    Rect2f(-1, -1, 2, 2),
                    transformation=(:xy, 0),
                    color=frame_color,
                    overdraw=true,
                    linewidth=10,
                    space=:clip,
                    depth_shift=1.0e-3,
                    inspectable=false
                )

                inspector = DataInspector(ax3d)
                m_events = addmouseevents!(ax3d)

                on(m_events.obs) do event
                    i = obj_to_idx[][obj]
                    if event.type === MouseEventTypes.over
                        deactivate_interaction!(ax, :create_group)
                        deactivate_interaction!(ax, :create_text)
                        show_inspector(obj)

                        if obj isa Transition
                            hovered[] = obj
                            cluster = get_t_cluster(obj)
                            notify(hovered)
                        else
                            cluster = obj
                        end
                        hovered_cluster[] = cluster
                        notify(hovered_cluster)
                    elseif event.type === MouseEventTypes.out
                        hovered[] = nothing
                        notify(hovered)
                        hovered_cluster[] = nothing
                        notify(hovered_cluster)

                        hide_inspector()
                        activate_interaction!(ax, :create_group)
                        activate_interaction!(ax, :create_text)
                    elseif event.type === MouseEventTypes.middledrag
                        points[][i] = mouseposition(ax)
                        notify(points)
                    elseif event.type === MouseEventTypes.rightdown
                        hovered_cluster[] = nothing
                        notify(hovered_cluster)

                        hovered[] = nothing
                        notify(hovered)
                        if obj isa Transition
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
                        if obj isa Transition
                            cluster = get_t_cluster(obj)
                        end
                        on_click(cluster)
                    end
                end

                # move to outside loop, will be far more efficient
                if obj isa Transition
                    plt_rendered = []
                    on(render_selection, update=true) do rs
                        foreach(x -> delete!(ax3d, x), plt_rendered)
                        empty!(plt_rendered)
                        if rs == "Volume"
                            v_lo, v_hi = render_views[rs](ax3d, Observable(obj))
                            plt_rendered = [v_lo, v_hi]
                        elseif rs == "Atom"
                            s = render_views["Atom"](ax3d, Observable(obj),
                                scalar_selection,
                                atom_time,
                                s_alignment)
                            plt_rendered = [s]
                        else
                            il, is, plots = render_views["Superquadric"](ax3d,
                                Observable(obj),
                                inspector,
                                s_alignment
                            )
                            plt_rendered = plots
                        end
                        center!(ax3d)
                    end
                else
                    # get transitions from general cluster object instead of the current one
                    ts = get_transitions(t_list, obj)
                    alignment = calculators["Alignment"](ts)
                    render_views["SMovement"](ax3d, Observable(ts), atom_time, Observable(alignment), Observable(c2corr[][obj]))
                    center!(ax3d)
                end
                push!(views[], ax3d)
                push!(frame_colors[], frame_color)
                push!(rendered_idxes[], plt_idx)
            end
        end
    end

    onany(ax.xaxis.attributes.limits, ax.yaxis.attributes.limits, points, ax.scene.viewport) do xlim, ylim, pp, svp
        ms = Int.(round.(ax.scene.camera.projectionview[] * marker_4d))[1]
        for (i, scene) in enumerate(views[])
            plt_idx = i + 1
            pos = position_on_plot(nodes, plt_idx, apply_transform=false)
            x, y = shift_project(ax.scene, apply_transform_and_model(nodes, pos))

            vp = Rect2i(x - (ms / 2), y - (ms / 2), ms, ms)
            vp = GeometryBasics.intersect(vp, svp)
            vw = widths(vp)

            if any(w -> w <= 0, vw)
                vp = Rect2i(0, 0, 0, 0)
            end

            viewports[][i] = pos
            scene.viewport[] = vp
        end
    end

    highlighted = Ref([])
    onany(hovered, hovered_cluster) do hov, hc
        if isnothing(hov) && isnothing(hc)
            for (v_idx, ogColor) in highlighted[]
                frame_colors[][v_idx][] = set_color_alpha(ogColor, 0.6)
            end
            empty!(highlighted[])
        end

        if !isnothing(hov) && hov in keys(obj_to_idx[])
            v_idx = obj_to_idx[][hov] - 1
            ogColor = frame_colors[][v_idx][]
            frame_colors[][v_idx][] = set_color_alpha(ogColor, 1.0)
            push!(highlighted[], (v_idx, ogColor))
        end

        if !isnothing(hc)
            ts = filter(x -> x isa Transition, collect(keys(obj_to_idx[])))
            for t in ts
                c = get_cluster_of_transition(cluster_info[], t)
                if length(intersect(c, hc)) > 0
                    v_idx = obj_to_idx[][t] - 1
                    ogColor = frame_colors[][v_idx][]
                    frame_colors[][v_idx][] = set_color_alpha(ogColor, 1.0)
                    push!(highlighted[], (v_idx, ogColor))
                end
            end

            if hc in keys(obj_to_idx[])
                v_idx = obj_to_idx[][hc] - 1
                ogColor = frame_colors[][v_idx][]
                frame_colors[][v_idx][] = set_color_alpha(ogColor, 1.0)
                push!(highlighted[], (v_idx, ogColor))
            end
        end
    end

    return ax, Scratchpad(boxes=boxes, views=viewports, notes=txt_to_notes, objs=obj_to_idx)
end

@kwdef mutable struct Scratchpad
    boxes
    views
    notes
    objs
end

# returns a dictionary of box indices to their children and the top level boxes in the hierarchy
# any "loose" children are returned separately
function group_scratchpad(s::Scratchpad)
    # start with the smallest boxes and work outwards for points
    boxes = collect(values(s.boxes[]))

    sort!(boxes, by=x -> area(x))
    seen_boxes = Set()
    seen_text = Set()
    seen = Set()
    hierarchy = Dict()
    titles = Dict()

    for (bIdx, box) in enumerate(boxes)
        title = string(bIdx)
        children = Union{Set{Int},Transition,String,Int}[]
        for (obj, idx) in s.objs[]
            v_idx = idx - 1
            vp = s.views[][v_idx]
            # seen lets us place things at bottom level
            if vp in box && !(idx in seen)
                push!(children, obj)
                push!(seen, idx)
            end
        end

        for (i, (pos, text, is_title)) in enumerate(values(s.notes[]))
            if pos in box && !(i in seen_text)
                if !is_title
                    push!(children, text)
                else
                    title = text
                end
                push!(seen_text, i)
            end
        end

        for (c_bIdx, c_box) in enumerate(boxes)
            if bIdx != c_bIdx
                if c_box in box
                    push!(children, c_bIdx)
                    push!(seen_boxes, c_bIdx)
                end
            end
        end
        titles[bIdx] = title
        hierarchy[bIdx] = children
    end

    top_level = filter(x -> !(x in seen_boxes), keys(hierarchy))

    loose = Union{Set{Int},Transition,String}[]
    for (obj, idx) in s.objs[]
        if !(idx in seen)
            push!(loose, obj)
        end
    end

    for (i, n) in enumerate(values(s.notes[]))
        if !(i in seen_text)
            push!(loose, n[2])
        end
    end

    return top_level, hierarchy, loose, titles
end

function export_scratchpad(s, t_list, ep, dpath)
    # export loose data in top folder
    top_level, hierarchy, loose, titles = group_scratchpad(s)
    if !isempty(loose)
        export_scratchpad_children(dpath, ep, loose, t_list)
    end

    for b in top_level
        export_scratchpad_children_recurse(b, hierarchy, t_list, ep, dpath, titles)
    end
end

function export_scratchpad_children_recurse(bIdx, hierarchy, t_list, parent_dir, dpath, titles)
    cf = joinpath(parent_dir, titles[bIdx])
    if !isdir(cf)
        mkdir(cf)
    end
    children = hierarchy[bIdx]
    export_scratchpad_children(dpath, cf, children, t_list)
    bChildren = filter(x -> x isa Int, children)
    foreach(b -> export_scratchpad_children_recurse(b, hierarchy, t_list, cf, dpath, titles), bChildren)
end

function export_scratchpad_children(dpath, p, children, t_list)
    # write notes in folder
    notes = filter(x -> x isa String, children)
    if !isempty(notes)
        foreach(x -> x * "\n", notes)
        note = reduce(*, notes)
        nf = joinpath(p, "notes.txt")
        write(nf, note)
    end

    # concat all children
    clusters = filter(x -> x isa Set{Int}, children)
    transitions = filter(x -> x isa Transition, children)

    ts = vcat(transitions, reduce(vcat, map(x -> get_transitions(t_list, x), clusters), init=[]))

    if !isempty(ts)
        export_t = export_transitions()
        export_t(dpath, p, ts)
    end
end
