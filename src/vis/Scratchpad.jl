const TITLE_HOTKEY = Keyboard.t

#TODO: bind listeners to scratchpad lifetime with Makie.onany calls
function scratchpad!(
    window::Makie.Figure,
    loc,
    selected_transitions::Observable{Set{Transition}},
    cluster_info::Observable{ClusterInfo},
    cluster_data::ClusterData,
    render_views::Dict{String,Function},
    render_selection::Observable{String},
    scalar_selection::Observable{String},
    bond_selection::Observable{String},
    invariant_selection::Observable{String},
    atom_time::Observable{Float32},
    selected_clusters::Observable{Set{ClusterSet}},
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
    obj_to_idx = Observable(Dict{Union{ClusterSet,Transition},Index}())
    rendered_idxes = Ref(Set{Index}())
    num_objs = Ref(1)
    views = Ref([])
    currently_hovered = Observable("")

    d_start = Point2f(0.0)
    d_end = Point2f(0.0)
    c_bbox = Observable(BBox(0, 0, 0, 0))
    boxes = Ref(Dict{Index,Rect2}())
    sw = wireframe!(ax.scene,
        c_bbox,
        color=:black,
        inspectable=false,
        visible=false)

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
            poly!(ax.scene,
                c_bbox[],
                color=:transparent,
                strokewidth=2,
                strokecolor=:black,
                inspectable=false)

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
                Makie.onany(txt.blockscene, txt.stored_string) do s
                    text!(ax.scene, pos; text=s, color=:black, font=(is_title) ? :bold : :regular)
                    s_idx = length(ax.scene.plots)
                    txt_to_notes[][s_idx] = (pos, s, is_title)
                end

                Makie.onany(txt.blockscene, txt.focused) do is_focused
                    if !is_focused
                        delete!(txt)
                    end
                end
            end
        end
    end

    ax_m_events = addmouseevents!(ax.scene)
    onmouseover(ax_m_events) do e
        plt, idx = pick(ax.scene)
        if plt isa Makie.Text
            plt.color[] = to_color(:grey)
        end
        return Consume(false)
    end

    onmouserightdown(ax_m_events) do e
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
        return Consume(false)
    end

    viewports = Ref([])
    frame_colors = Ref([])
    frame_widths = Ref([])

    function delete_obj!(obj, views, obj_to_idx)
        plt_idx = obj_to_idx[][obj]
        v_idx = plt_idx - 1
        scene = views[][v_idx]
        # Makie.free seems to destroy theme object...
        Makie.free(scene)
        delete!(obj_to_idx[], obj)
        notify(obj_to_idx)
    end

    Makie.onany(ax.scene, selected_transitions, selected_clusters) do st, sc
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
            # only update ax when points are added to avoid bugs
            autolimits!(ax)
            center!(ax.scene)
        end
    end

    nodes = scatter!(ax, points, marker=:rect, color=:transparent, inspectable=false)
    marker_4d = Point4f(markersize, markersize, 0, 0)

    function get_obj_cluster(obj)
        if obj isa Transition
            return get_cluster_of_transition(cluster_info[], rel_t_to_idx, obj)
        end
        return obj
    end

    function obj_to_str(obj)
        if obj isa ClusterSet
            s = str_limit(get_val(cluster_annotations[], "titles", obj))
        else
            s = "($(join(string.(obj, base=10), ",")))"
        end
        return s
    end

    Makie.onany(ax.scene, obj_to_idx) do idxes
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
                push!(viewports[], pos)

                set_theme!(ct)
                ax3d = Scene(ax.scene,
                    show_axis=false,
                    viewport=vp,
                    backgroundcolor=EMBEDDED_SCENE_BACKGROUND,
                    clear=true,
                    camera=cam3d!,
                    size=(ms, ms))

                frame_color = lift(ax3d, cluster_info) do ci
                    if obj isa Transition
                        return cluster_color(ci, cluster_data, rel_t_to_idx, obj)
                    else
                        return cluster_data.colors[obj]
                    end
                end

                linewidth = Observable(3)

                wireframe!(
                    ax3d,
                    Rect2f(-1, -1, 2, 2),
                    transformation=(:xy, 0),
                    color=frame_color,
                    overdraw=true,
                    linewidth=linewidth,
                    space=:clip,
                    depth_shift=1.0e-3,
                    inspectable=false
                )

                m_events = addmouseevents!(ax3d)
                Makie.onany(ax3d, m_events.obs) do event
                    i = obj_to_idx[][obj]
                    if event.type === MouseEventTypes.over
                        deactivate_interaction!(ax, :create_group)
                        deactivate_interaction!(ax, :create_text)

                        if obj isa Transition
                            hovered[] = obj
                        else
                            hovered_cluster[] = obj
                        end
                        currently_hovered[] = obj_to_str(obj)

                    elseif event.type === MouseEventTypes.out
                        hovered[] = nothing
                        hovered_cluster[] = nothing
                        currently_hovered[] = ""

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

                # move to outside loop, will be far more efficient
                if obj isa Transition
                    Makie.onany(ax3d, render_selection, update=true) do rs
                        foreach(x -> delete!(ax3d, x),
                            filter(y -> !(y isa Wireframe), ax3d.plots))
                        if rs == "Atom"
                            render_views["Atom"](ax3d,
                                obj,
                                scalar_selection,
                                atom_time)
                        elseif rs == "Bonds"
                            render_views["Bonds"](ax3d,
                                obj, bond_selection)
                        else
                            render_views["Superquadric"](ax3d,
                                obj, invariant_selection)
                        end
                        center!(ax3d)
                        apply_alignment_to_scene(ax3d, alignment[obj])
                    end
                else
                    # get transitions from general cluster object instead of the current one
                    ts = get_transitions(cluster_data, obj)
                    ref_t, alignment = calculators["Alignment"](ts)
                    render_views["SMovement"](ax3d, ts, atom_time, alignment, Observable(c2corr[][obj]))
                    center!(ax3d)
                end

                push!(frame_widths[], linewidth)
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

    function shift_point(d, ms::Int, svp)
        i, scene = d
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


    Makie.onany(ax.scene,
        ax.xaxis.attributes.limits,
        ax.yaxis.attributes.limits,
        points,
        ax.scene.viewport) do xlim, ylim, pp, svp
        ms = Int.(round.(ax.scene.camera.projectionview[] * marker_4d))[1]
        shift_point.(enumerate(views[]), Ref(ms), Ref(svp))
    end

    highlighted = Ref([])
    Makie.onany(ax.scene, hovered) do hov
        objidx = to_value(obj_to_idx)
        if !isnothing(hov) && haskey(objidx, hov)
            cluster = get_obj_cluster(hov)
            hovered_cluster[] = cluster
        end
    end

    Makie.onany(ax.scene, hovered_cluster) do hc
        if isnothing(hc)
            for v_idx in highlighted[]
                frame_widths[][v_idx][] = 3
            end
            empty!(highlighted[])
        end

        if !isnothing(hc)
            objidx = to_value(obj_to_idx)
            ts = filter(x -> x isa Transition, collect(keys(objidx)))
            for t in ts
                c = get_cluster_of_transition(cluster_info[], rel_t_to_idx, t)
                if length(intersect(c, hc)) > 0
                    v_idx = obj_to_idx[][t] - 1
                    frame_widths[][v_idx][] = 10
                    push!(highlighted[], v_idx)
                end
            end

            c_plt_idx = get(objidx, hc, nothing)
            if !isnothing(c_plt_idx)
                v_idx = c_plt_idx - 1
                frame_widths[][v_idx][] = 10
                push!(highlighted[], v_idx)
            end
        end
    end

    contents = function ()
        return group_scratchpad(boxes[], viewports[], txt_to_notes[], obj_to_idx[])
    end

    cleanup = function ()
        @debug "cleanup scratchpad"
        if !isnothing(obj_to_idx)
            Observables.clear(obj_to_idx)
            Observables.clear(points)

            delete_obj! = nothing
            get_t_cluster = nothing
            obj_to_str = nothing
        end
    end

    return ax, contents, cleanup, currently_hovered
end

@kwdef struct Scratchpad
    top_level
    hierarchy
    loose
    titles
end

# returns a dictionary of box indices to their children and the top level boxes in the hierarchy
# any "loose" children are returned separately
function group_scratchpad(cboxes, views, notes, objs)::Scratchpad
    # start with the smallest boxes and work outwards for points
    boxes = collect(values(cboxes))

    sort!(boxes, by=x -> area(x))
    seen_boxes = Set()
    seen_text = Set()
    seen = Set()
    hierarchy = Dict()
    titles = Dict()

    for (bIdx, box) in enumerate(boxes)
        title = string(bIdx)
        children = Union{ClusterSet,Transition,String,Int}[]
        for (obj, idx) in objs
            v_idx = idx - 1
            vp = views[v_idx]
            # seen lets us place things at bottom level
            if vp in box && !(idx in seen)
                push!(children, obj)
                push!(seen, idx)
            end
        end

        for (i, (pos, text, is_title)) in enumerate(values(notes))
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
    for (obj, idx) in objs
        if !(idx in seen)
            push!(loose, obj)
        end
    end

    for (i, n) in enumerate(values(notes))
        if !(i in seen_text)
            push!(loose, n[2])
        end
    end

    return Scratchpad(top_level=top_level, hierarchy=hierarchy, loose=loose, titles=titles)
end

function export_scratchpad(s::Scratchpad, cd::ClusterData, ep::String, dpath::String)
    # export loose data in top folder
    (; top_level, hierarchy, loose, titles) = s
    if !isempty(loose)
        export_scratchpad_children(dpath, ep, loose, cd)
    end

    for b in top_level
        _export_scratchpad_children(b, hierarchy, cd, ep, dpath, titles)
    end
end

function _export_scratchpad_children(bIdx, hierarchy, cd::ClusterData, parent_dir::String, dpath::String, titles)
    cf = joinpath(parent_dir, titles[bIdx])
    if !isdir(cf)
        mkdir(cf)
    end
    children = hierarchy[bIdx]
    export_scratchpad_children(dpath, cf, children, cd)
    bChildren = filter(x -> x isa Integer, children)
    foreach(b -> _export_scratchpad_children(b, hierarchy, cd, cf, dpath, titles), bChildren)
end

function export_scratchpad_children(dpath::String, p, children, cd::ClusterData)
    # write notes in folder
    notes = filter(x -> x isa String, children)
    if !isempty(notes)
        foreach(x -> x * "\n", notes)
        note = reduce(*, notes)
        nf = joinpath(p, "notes.txt")
        write(nf, note)
    end

    # concat all children
    clusters = filter(x -> x isa ClusterSet, children)
    transitions = filter(x -> x isa Transition, children)
    ts = vcat(transitions, reduce(vcat, map(x -> get_transitions(cd, x), clusters), init=[]))

    if !isempty(ts)
        export_t = export_transitions()
        export_t(joinpath(dpath, "t_ase_dict.pickle"), p, ts)
    end
end
