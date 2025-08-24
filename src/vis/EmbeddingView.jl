function get_extents(p, size)
    return p[1] - size / 2, p[1] + size / 2, p[2] - size / 2, p[2] + size / 2
end

function clear_listeners!(d, t)
    bound = get(d, t, nothing)
    if !isnothing(bound)
        for ptr in bound[1]
            x = ptr[]
            off(x)
            x = nothing
            ptr = nothing
        end
        for ptr in bound[2]
            y = ptr[]
            Observables.clear(y)
            y = nothing
            ptr = nothing
        end
        empty!(bound[1])
        empty!(bound[2])
        delete!(d, t)
    end
end

function embedding_view!(
    loc,
    cluster_data::Observable{SingleClusterData},
    embedding::Observable{AbstractArray{AbstractArray{Float32}}},
    selected_render::Observable{String},
    selected_scalar::Observable{String},
    atom_time::Observable{Float32},
    render_views::Dict{String,Function},
    hovered::MaybeObservable{Transition},
    hovered_cluster::MaybeObservable{ClusterSet},
    colors::Observable{Vector{RGBAf}},
    cluster_info::ClusterInfo;
    on_click::Function=(x) -> (),
    markersize::Observable{Int}=Observable(100),
)
    ax = Axis(loc, backgroundcolor=:transparent)
    deregister_interaction!(ax, :rectanglezoom)
    hidedecorations!(ax)
    campixel!(ax.scene)

    # invisible scatter plot to set up camera
    markersize_4d = lift(x -> Point4f(x, x, 0, 0), markersize)

    # debug vars
    show_alignment = Observable(true)
    resolve_overlap = Observable(true)

    jittered_points = @lift begin
        points = $embedding
        final = []

        if $resolve_overlap
            proj_marker = $(ax.scene.camera.pixel_space) * $markersize_4d
            ms = max(proj_marker[1], proj_marker[2])
            kd = RangeTree(Matrix{Float64}(undef, 2, 0), ms / 2)

            for pt in points
                np = deepcopy(pt)
                found = false
                iter = 0
                while !found
                    overlaps = final[AdaptiveKDTrees.RangeSearch.find_in_range(kd, np, ms / sqrt(pi))]
                    if length(overlaps) == 0 || iter == 100
                        push!(final, np)
                        AdaptiveKDTrees.RangeSearch.add_point!(kd, np)
                        found = true
                    else
                        po = first(overlaps)
                        dminx, dmaxx, dminy, dmaxy = get_extents(np, ms)
                        sminx, smaxx, sminy, smaxy = get_extents(po, ms)

                        overlap_x = min(dmaxx, smaxx) - max(dminx, sminx)
                        overlap_y = min(dmaxy, smaxy) - max(dminy, sminy)

                        dir = np - po

                        if overlap_x < overlap_y
                            np += Point2f(sign(dir[1]) * overlap_x, 0.0)
                        else
                            np += Point2f(0.0, sign(dir[2]) * overlap_y)
                        end
                    end
                    iter += 1
                end
            end
            return Point2f.(final)
        end
        return Point2f.(points)
    end

    umap_nodes = scatter!(ax,
        jittered_points,
        marker=:rect,
        color=:transparent,#:blue,
        inspector_label=(ins, idx, pos) -> string(cluster_data[].ts[idx]))

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
    views = Ref([])
    all_listeners = Dict{Transition,Any}()
    scene_listeners = Ref([])

    times = 0
    t_to_pltidx = @lift begin
        @show "re-rendering"
        times += 1
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

                    c = get_cluster_of_transition(cluster_info, t)
                    hovered_cluster[] = c
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
        hovered[] = nothing
        enable_interactions(ax)
        return t_to_pltidx
    end

    sr_listener = on(selected_render, weak=true) do sr
        disable_interactions(ax)
        if length(views[]) == length(embedding[])
            ts = cluster_data[].ts
            alignment = cluster_data[].alignment
            @time for (i, ptr) in enumerate(views)
                ax3d = ptr[]
                t = ts[i]
                flip = alignment[t][2]
                clear_listeners!(all_listeners, t)
                foreach(x -> delete!(ax3d, x), filter(y -> !(y isa Wireframe), ax3d.plots))
                @debug @show ax3d
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
    ax_listener = onany(ax.xaxis.attributes.limits, ax.yaxis.attributes.limits, markersize_4d, weak=true) do xlim, ylim, mkr
        if length(views[]) == length(embedding[])
            ms = Int.(round.(ax.scene.camera.projectionview[] * mkr))[1]
            for (i, ptr) in enumerate(views[])
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
        end
    end

    highlighted = Ref([])
    c_listener = on(colors, weak=true) do c_list
        if length(c_list) == length(frame_colors[])
            empty!(highlighted[])
            for (i, c) in enumerate(c_list)
                frame_colors[][i][] = c
            end
        end
    end

    hover_listener = onany(hovered, hovered_cluster, weak=true) do hov, hc
        if isnothing(hov) && isnothing(hc)
            for (v_idx, ogColor) in highlighted[]
                frame_colors[][v_idx][] = set_color_alpha(ogColor, 0.6)
            end
            empty!(highlighted[])
            return
        end

        if !isnothing(hov) && haskey(to_value(t_to_pltidx), hov)
            v_idx = to_value(t_to_pltidx)[hov]
            ogColor = frame_colors[][v_idx][]
            frame_colors[][v_idx][] = set_color_alpha(ogColor, 1.0)
            push!(highlighted[], (v_idx, ogColor))
        end

        if !isnothing(hc)
            ts = keys(to_value(t_to_pltidx))
            for t in ts
                c = get_cluster_of_transition(cluster_info, t)
                if length(intersect(c, hc)) > 0
                    v_idx = to_value(t_to_pltidx)[t]
                    ogColor = frame_colors[][v_idx][]
                    frame_colors[][v_idx][] = set_color_alpha(ogColor, 1.0)
                    push!(highlighted[], (v_idx, ogColor))
                end
            end
        end
    end


    # if I wanted to do this I could just write C
    cleanup = function ()
        @debug "kill embedding"
        off(sr_listener)
        sr_listener = nothing
        for l in hover_listener
            off(l)
            l = nothing
        end
        empty!(hover_listener)

        Observables.clear(jittered_points)

        off(c_listener)
        c_listener = nothing

        off(kb_events)
        kb_events = nothing

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
