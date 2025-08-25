const TITLE_HOTKEY = Keyboard.t

function scratchpad!(
    window::Makie.Figure,
    loc,
    selected_transitions::Observable{Set{Transition}},
    cluster_info::Observable{ClusterInfo},
    cluster_data::ClusterData,
    render_views::Dict{String,Function},
    render_selection::Observable{String},
    scalar_selection::Observable{String},
    atom_time::Observable{Float32},
    selected_clusters::Observable{Set{ClusterSet}},
    t_list::Vector{Transition},
    rel_t_to_idx::Dict{Transition,Int},
    calculators::Dict{String,Function},
    cluster_annotations::Observable{ClusterAnnotation},
    c2corr::Base.RefValue{Dict{ClusterSet,Float32}};
    hovered::MaybeObservable{Transition}=MaybeObservable{Transition}(nothing),
    hovered_cluster::MaybeObservable{ClusterSet},
    on_click::Function=(x) -> (),
    markersize=200)

    plot_theme = Theme(MeshScatter=(
            inspectable=false,
            markercolor=to_color(:blue)
        ),
        fontsize=18.0,
        inspectable=true,
        markercolor=to_color(:blue)
    )

    ct = Makie.merge(theme_latexfonts(), plot_theme)

    set_theme!(ct)
    ax = Axis(loc, backgroundcolor=:transparent, title="Scratchpad")
    deregister_interaction!(ax, :rectanglezoom)
    hidedecorations!(ax)
    campixel!(ax.scene)

    points = Observable{Vector{Point2f}}(Point2f[Point2f(0.0)])
    # run once on creation to bind axis
    reset_limits!(ax)
    center!(ax.scene)

    obj_to_idx = Observable(Dict{Union{ClusterSet,Transition},Index}())
    rendered_idxes = Ref(Set{Index}())
    num_objs = Ref(1)
    views = Ref([])

    d_start = Point2f(0.0)
    d_end = Point2f(0.0)
    c_bbox = Observable(BBox(0, 0, 0, 0))
    boxes = Ref(Dict{Index,Rect2}())
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
        elseif e.type === MouseEventTypes.leftdragstop
            # finish placing box
            w = poly!(ax.scene, c_bbox[], color=:transparent, strokewidth=2, strokecolor=:black, inspectable=false)

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
    ax_m_listener = on(ax_m_events.obs, weak=true) do e
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

    viewports = Ref([])
    frame_colors = Ref([])
    scene_listeners = Ref([])
    function delete_obj!(obj, views, obj_to_idx)
        plt_idx = obj_to_idx[][obj]
        v_idx = plt_idx - 1
        scene = views[][v_idx]
        # Makie.free seems to destroy theme object...
        # need to also delete listeners here, will do later
        off(scene_listeners[][v_idx])
        Makie.free(scene)
        delete!(obj_to_idx[], obj)

        notify(obj_to_idx)
    end

    select_listeners = onany(selected_transitions, selected_clusters) do st, sc
        if length(st) != 0 || length(sc) != 0
            last_rendered = vcat([1], collect(rendered_idxes[]))
            new_points = Point2f[]
            new_objs = filter(x -> !(x in keys(obj_to_idx[])), vcat(collect(st), collect(sc)))
            for obj in new_objs
                num_objs[] += 1
                push!(new_points, Point2f(0.0, 0.0))
                obj_to_idx[][obj] = num_objs[]
            end

            all_points = vcat(points.val, new_points)

            # do this so they don't overlap
            points.val = spring(zeros(length(all_points), length(all_points)); C=0.1, pin=Dict(last_rendered .=> true), initialpos=all_points)
            notify(points)
            notify(obj_to_idx)
        end
    end

    nodes = scatter!(ax, points, marker=:rect, visible=false)
    nodes.inspectable[] = false

    marker_4d = Point4f(markersize, markersize, 0, 0)

    function get_obj_cluster(obj)
        if obj isa Transition
            return get_cluster_of_transition(cluster_info[], rel_t_to_idx, obj)
        end
        return obj
    end

    function obj_to_str(obj)
        s = string(obj)
        if obj isa ClusterSet
            s = str_limit(get_val(cluster_annotations[], "titles", obj))
        end
        return s
    end

    function show_inspector(obj)
        window_inspector = DataInspector(window)
        tt = window_inspector.plot
        s = obj_to_str(obj)
        mp = mouseposition(ax.scene)
        smp = shift_project(ax.scene, apply_transform_and_model(nodes, mp))
        update_tooltip_alignment!(window_inspector, smp)

        tt.text[] = s
        tt.visible[] = true
    end

    function hide_inspector()
        window_inspector = DataInspector(window)
        tt = window_inspector.plot
        tt.visible[] = false
    end



    idx_listener = on(obj_to_idx, weak=true) do idxes
        ts = collect(selected_transitions[])
        _, alignment = calculators["Alignment"](ts)
        ms = Int.(round.(ax.scene.camera.projectionview[] * marker_4d))[1]

        for (obj, idx) in idxes
            plt_idx = idx
            if !(plt_idx in rendered_idxes[])
                pos = position_on_plot(nodes, plt_idx, apply_transform=false)
                # x, y is in global pixel coords
                x, y = shift_project(ax.scene, apply_transform_and_model(nodes, pos))
                # calculate shifted size of marker

                vp = Rect2i(x - (ms / 2), y - (ms / 2), ms, ms)

                # viewports need to be in data space
                push!(viewports[], pos)

                set_theme!(ct)
                ax3d = Scene(ax.scene,
                    show_axis=false,
                    viewport=vp,
                    backgroundcolor=EMBEDDED_SCENE_BACKGROUND,
                    clear=true,
                    camera=cam3d!,
                    size=(ms, ms))

                # translate!(ax3d, 0, 0, 100) ? 

                frame_color = @lift begin
                    if obj isa Transition
                        ci = $(cluster_info)
                        # get the currently assigned color of this transition
                        return set_color_alpha(cluster_color(ci, cluster_data, rel_t_to_idx, obj), 0.6)
                    else
                        # gets the assigned cluster color
                        return set_color_alpha(cluster_data.colors[obj], 0.6)
                    end
                end

                wireframe!(
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

                m_events = addmouseevents!(ax3d)

                mouse_listener = on(m_events.obs, weak=true) do event
                    i = obj_to_idx[][obj]
                    if event.type === MouseEventTypes.over
                        deactivate_interaction!(ax, :create_group)
                        deactivate_interaction!(ax, :create_text)

                        cluster = get_obj_cluster(obj)
                        if obj isa Transition
                            hovered[] = obj
                        end
                        hovered_cluster[] = cluster
                    elseif event.type === MouseEventTypes.out
                        hovered[] = nothing
                        hovered_cluster[] = nothing

                        activate_interaction!(ax, :create_group)
                        activate_interaction!(ax, :create_text)
                    elseif event.type === MouseEventTypes.middledrag
                        points[][i] = mouseposition(ax)
                        notify(points)
                    elseif event.type === MouseEventTypes.rightdown
                        hovered_cluster[] = nothing
                        hovered[] = nothing

                        if obj isa Transition
                            delete!(selected_transitions[], obj)
                            notify(selected_transitions)
                        else
                            delete!(selected_clusters[], obj)
                            notify(selected_clusters)
                        end

                        delete_obj!(obj, views, obj_to_idx)
                        notify(points)
                    elseif event.type === MouseEventTypes.leftdoubleclick
                        cluster = get_obj_cluster(obj)
                        on_click(cluster)
                    end
                end

                push!(scene_listeners[], mouse_listener)

                # move to outside loop, will be far more efficient
                if obj isa Transition
                    plt_rendered = []
                    rs_listener = on(render_selection, weak=true, update=true) do rs
                        foreach(x -> delete!(ax3d, x), plt_rendered)
                        empty!(plt_rendered)
                        if rs == "Volume"
                            v_lo, v_hi = render_views[rs](ax3d, obj)
                            plt_rendered = [v_lo, v_hi]
                        elseif rs == "Atom"
                            s = render_views["Atom"](ax3d,
                                obj,
                                scalar_selection,
                                atom_time)
                            plt_rendered = [s]
                        else
                            il, is, plots = render_views["Superquadric"](ax3d,
                                obj,)
                            plt_rendered = plots
                        end
                        center!(ax3d)
                    end
                    push!(scene_listeners[], rs_listener)
                else
                    # get transitions from general cluster object instead of the current one
                    ts = get_transitions(t_list, obj)
                    ref_t, alignment = calculators["Alignment"](ts)
                    render_views["SMovement"](ax3d, ts, atom_time, alignment, Observable(c2corr[][obj]))
                    center!(ax3d)
                end
                if obj isa Transition
                    apply_alignment_to_scene(ax3d, alignment[obj])
                end

                push!(views[], ax3d)
                push!(frame_colors[], frame_color)
                push!(rendered_idxes[], plt_idx)
            else
                if obj isa Transition
                    if idx - 1 < length(views[]) && obj in keys(alignment)
                        apply_alignment_to_scene(views[][idx-1], alignment[obj])
                    end
                end
            end
        end
    end

    ax_listeners = onany(ax.xaxis.attributes.limits, ax.yaxis.attributes.limits, points, ax.scene.viewport, weak=true) do xlim, ylim, pp, svp
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
    hv_listeners = onany(hovered, hovered_cluster, weak=true) do hov, hc
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
                c = get_cluster_of_transition(cluster_info[], rel_t_to_idx, t)
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

    cleanup = function ()
        @debug "cleanup scratchpad"
        clear_listener_list(select_listeners)
        clear_listener_list(hv_listeners)
        clear_listener_list(ax_listeners)
        clear_listener_list(scene_listeners[])

        if !isnothing(idx_listener)
            off(idx_listener)
            idx_listener = nothing

            Observables.clear(obj_to_idx)
            Observables.clear(points)

            delete_obj! = nothing
            get_t_cluster = nothing
            obj_to_str = nothing
            show_inspector = nothing
            hide_inspector = nothing
        end
    end

    return ax, Scratchpad(boxes=boxes, views=viewports, notes=txt_to_notes, objs=obj_to_idx), cleanup
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
        children = Union{ClusterSet,Transition,String,Int}[]
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

    loose = Union{ClusterSet,Transition,String}[]
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
    bChildren = filter(x -> x isa Integer, children)
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
