function get_extents(p, size)
    return p[1] - size / 2, p[1] + size / 2, p[2] - size / 2, p[2] + size / 2
end

function clear_listeners!(d, t)
    bound = get(d, t, nothing)
    if !isnothing(bound)
        for x in bound[1]
            off(x)
            x = nothing
        end
        empty!(bound[1])

        for y in bound[2]
            Observables.clear(y)
            y = nothing
        end
        empty!(bound[2])

        delete!(d, t)
    end
end

function embedding_view!(
    loc,
    cluster_data::Observable{SingleClusterData},
    embedding::Observable{Vector{Point2f}},
    selected_render::Observable{String},
    selected_scalar::Observable{String},
    atom_time::Observable{Float32},
    render_views::Dict{String,Function},
    hovered::MaybeObservable{Transition},
    hovered_cluster::MaybeObservable{ClusterSet},
    colors::Observable{Vector{RGBAf}};
    on_click::Function=(x) -> (),
    markersize::Observable{Int}=Observable(100),
)
    ax = Axis(loc, backgroundcolor=:transparent,
        autolimitaspect=1)
    deregister_interaction!(ax, :rectanglezoom)
    hidedecorations!(ax)
    campixel!(ax.scene)

    # invisible scatter plot to set up camera
    markersize_4d = lift(x -> Point4f(x, x, 0, 0), markersize)

    # debug vars
    show_alignment = Observable(true)
    resolve_overlap = Observable(true)

    umap_nodes = scatter!(ax,
        embedding,
        marker=:rect,
        color=:transparent,
        inspector_label=(ins, idx, pos) -> join(string.(cluster_data[].ts[idx], base=10), ","))

    on(embedding, update=true) do e
        autolimits!(ax)
        center!(ax.scene)
    end

    kb_events = on(events(ax.scene).keyboardbutton, weak=true) do event
        if ispressed(ax.scene, Exclusively(Keyboard.page_up))
            markersize[] = markersize[] + 25
        elseif ispressed(ax.scene, Exclusively(Keyboard.page_down))
            markersize[] = markersize[] - 25
        elseif ispressed(ax.scene, Exclusively(Keyboard.f))
            show_alignment[] = !show_alignment[]
            if show_alignment[]
                println("aligned")
            else
                println("identity")
            end
        elseif ispressed(ax.scene, Exclusively(Keyboard.o))
            resolve_overlap[] = !resolve_overlap[]
            if resolve_overlap[]
                println("fixing overlap")
            else
                println("not fixing overlap")
            end
        end
    end

    ins = DataInspector()
    frame_colors = Ref([])
    views::Base.RefValue{Vector{Base.RefValue{Makie.Scene}}} = Ref(Base.RefValue{Makie.Scene}[])
    all_listeners = Dict{Transition,Any}()
    scene_listeners = Ref([])
    highlighted = Ref([])

    t_to_pltidx = @lift begin
        @debug "Rendering embedding"
        disable_interactions(ax)

        # instead of clearing everything, why don't we keep them and only delete non-existing ones?
        for (al, ml) in scene_listeners[]
            off(al[])
            off(ml[])
        end

        for ax3d in views[]
            s = ax3d[]
            empty!(s)
            Makie.free(s)
            s = nothing
            ax3d = nothing
        end

        empty!(views[])
        empty!(frame_colors[]) # update frame colors
        empty!(highlighted[])
        GC.gc(true)

        reset_limits!(ax)
        center!(ax.scene)

        cd = $(cluster_data)
        ts = cd.ts
        alignment = cd.alignment

        t_to_pltidx = Dict(reverse.(enumerate(ts)))

        ms = Int.(round.(ax.scene.camera.projectionview[] * markersize_4d[]))[1]

        @time for (i, t) in enumerate(ts)
            clear_listeners!(all_listeners, t)
            pos = position_on_plot(umap_nodes, i, apply_transform=false)
            # x, y is in global pixel coords
            x, y = shift_project(ax.scene, apply_transform_and_model(umap_nodes, pos))
            # calculate shifted size of marker
            vp = Rect2i(x - (ms / 2), y - (ms / 2), ms, ms)

            ax3d = Scene(ax.scene,
                show_axis=false,
                viewport=vp,
                backgroundcolor=EMBEDDED_SCENE_BACKGROUND,
                clear=true,
                camera=cam3d!,
                size=(ms, ms))

            alignment_listener = on(show_alignment, update=true, weak=true) do showAlignment
                if showAlignment
                    R, flip = alignment[t]
                    rr = hcat(R, [0, 0, 0])
                    fr = transpose(vcat(rr, transpose([0; 0; 0; 1])))
                    ax3d.transformation.model[] = Float64.(fr)
                else
                    ax3d.transformation.model[] = Matrix(1.0I, 4, 4)
                end
            end

            # sets to color of original leaves
            frame_color = Observable(set_color_alpha(colors[][i], 0.6))
            # sets to color of assignment 
            # set_color_alpha(cluster_color(cluster_info, t), 0.6))

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
                if event.type === MouseEventTypes.over
                    #show_data(ins, umap_nodes, i)
                    hovered[] = t

                elseif event.type === MouseEventTypes.out
                    hovered[] = nothing
                    hovered_cluster[] = nothing
                elseif event.type === MouseEventTypes.leftdoubleclick
                    on_click(t)
                end
            end

            # initial render
            sr = selected_render[]
            flip = alignment[t][2]
            if sr == "Volume"
                render_views[sr](ax3d, t)
            elseif sr == "Atom"
                render_views["Atom"](ax3d,
                    t,
                    selected_scalar,
                    atom_time,
                    flip
                )
            else
                listeners, obs, _ = render_views["Superquadric"](ax3d,
                    t,
                    flip)
                all_listeners[t] = (listeners, obs)
            end
            center!(ax3d)
            yield()
            push!(views[], Ref(ax3d))
            push!(frame_colors[], frame_color)
            push!(scene_listeners[], (Ref(alignment_listener), Ref(mouse_listener)))
        end
        center!(ax.scene)
        hovered[] = nothing
        enable_interactions(ax)
        return t_to_pltidx
    end

    sr_listener = on(selected_render, weak=true) do sr
        disable_interactions(ax)
        if length(views[]) == length(embedding[])
            ts = cluster_data[].ts
            alignment = cluster_data[].alignment
            @time for (i, ptr) in enumerate(views[])
                ax3d = ptr[]
                t = ts[i]
                flip = alignment[t][2]
                clear_listeners!(all_listeners, t)
                foreach(x -> delete!(ax3d, x), filter(y -> !(y isa Wireframe), ax3d.plots))
                if sr == "Volume"
                    render_views[sr](ax3d, t)
                elseif sr == "Atom"
                    render_views["Atom"](ax3d,
                        t,
                        selected_scalar,
                        atom_time, flip)
                else
                    listeners, obs, _ = render_views["Superquadric"](ax3d,
                        t, flip)
                    all_listeners[t] = (listeners, obs)
                end
                center!(ax3d)
                # block for a millisecond so makie can catch up
                # otherwise it seems like the renderer gets overwhelmed & it just goes oom
                yield()
            end
            GC.gc(true)
        end
        enable_interactions(ax)
    end

    #https://github.com/MakieOrg/Makie.jl/blob/381cf4a1ade5bf1a36b254ce6daccb5cbc71939e/GLMakie/assets/shader/dots.vert#L55

    #broadcast to try and speed it up a bit
    function shift_point(d, ms::Int)
        i, ptr = d
        scene = ptr[]
        pos = position_on_plot(umap_nodes, i, apply_transform=false)
        x, y = shift_project(ax.scene, apply_transform_and_model(umap_nodes, pos))

        vp = Rect2i(x - (ms / 2), y - (ms / 2), ms, ms)
        vp = GeometryBasics.intersect(vp, ax.scene.viewport[])
        vw = widths(vp)

        if any(w -> w <= 0, vw)
            vp = Rect2i(0, 0, 0, 0)
        end

        scene.viewport[] = vp
    end

    ax_listener = onany(ax.xaxis.attributes.limits, ax.yaxis.attributes.limits, markersize_4d, embedding, t_to_pltidx, weak=true) do xlim, ylim, mkr, p, pindx
        if length(views[]) == length(p)
            ms = Int.(round.(ax.scene.camera.projectionview[] * mkr))[1]
            shift_point.(enumerate(views[]), Ref(ms))
        end
    end

    c_listener = on(colors, weak=true) do c_list
        if length(c_list) == length(frame_colors[])
            empty!(highlighted[])
            for (i, c) in enumerate(c_list)
                frame_colors[][i][] = set_color_alpha(c, 0.6)
            end
        end
    end


    # converts hovered into its cluster
    # since you can't hover both a cluster and a transition simultaneously, this frees up the logic underneath
    hover_converter = on(hovered, weak=true) do hov
        pindx = to_value(t_to_pltidx)
        if !isnothing(hov) && haskey(pindx, hov)
            c = get_cluster_of_transition(cluster_data[], pindx[hov])
            hovered_cluster[] = c
        end
    end

    hover_listener = on(hovered_cluster, weak=true) do hc
        for (v_idx, ogColor) in highlighted[]
            frame_colors[][v_idx][] = set_color_alpha(ogColor, 0.6)
        end
        empty!(highlighted[])

        if !isnothing(hc)
            for i in values(to_value(t_to_pltidx))
                c = get_cluster_of_transition(cluster_data[], i)
                if !isnothing(c) && length(intersect(c, hc)) > 0
                    if i <= length(frame_colors[])
                        ogColor = frame_colors[][i][]
                        frame_colors[][i][] = set_color_alpha(ogColor, 1.0)
                        push!(highlighted[], (i, ogColor))
                    end
                end
            end
        end
    end


    # if I wanted to do this I could just write C
    cleanup = function ()
        @debug "kill embedding"

        if !isnothing(sr_listener)
            off(sr_listener)
            sr_listener = nothing
            off(c_listener)
            c_listener = nothing
            off(kb_events)
            kb_events = nothing
            off(hover_listener)
            hover_listener = nothing
            off(hover_converter)
            hover_converter = nothing
        end

        for (al, ml) in scene_listeners[]
            off(al[])
            off(ml[])
        end

        for ptr in views[]
            ax3d = ptr[]
            empty!(ax3d)
            Makie.free(ax3d)
            ax3d = nothing
            ptr = nothing
        end
        clear_listener_list(ax_listener)

        empty!(views[])
        Observables.clear.(frame_colors[])
        empty!(frame_colors[]) # update frame colors
        for t in keys(all_listeners)
            clear_listeners!(all_listeners, t)
        end

        empty!(ax.scene)
        Makie.free(ax.scene)
    end

    return cleanup
end
