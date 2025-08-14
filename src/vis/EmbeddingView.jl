function embedding_view!(
    loc,
    data,
    selected_render,
    selected_scalar,
    atom_time,
    render_views,
    hovered,
    hovered_cluster,
    colors,
    cluster_info;
    point_margin=5,
    highlight_borders=Observable(false),
    on_click=(x) -> (),
    markersize=Observable(100),
)
    ax = Axis(loc, backgroundcolor=:transparent)
    #=on(highlight_borders, update=true) do hb
        border_color = to_color(:black)
        if hb
            border_color = to_color(:red)
        end
        ax.topspinecolor[] = border_color
        ax.bottomspinecolor[] = border_color
        ax.leftspinecolor[] = border_color
        ax.rightspinecolor[] = border_color
    end=#
    deregister_interaction!(ax, :rectanglezoom)
    hidedecorations!(ax)
    campixel!(ax.scene)
    # invisible scatter plot to set up camera
    markersize_4d = lift(x -> Point4f(x, x, 0, 0), markersize)
    jittered_points = @lift begin
        points = $data[3]
        final = []

        kd = RangeTree(Matrix{Float64}(undef, 2, 0), 0)
        ms = 0.5

        for p in points
            # calculate shifted size of marker
            np = deepcopy(p)
            found = false

            while !found
                overlaps = final[AdaptiveKDTrees.RangeSearch.find_in_range(kd, np, ms)]
                if length(overlaps) == 0
                    push!(final, np)
                    AdaptiveKDTrees.RangeSearch.add_point!(kd, np)
                    found = true
                else
                    po = first(overlaps)
                    dir = po - np
                    np -= dir
                end
            end
        end
        return final
    end

    umap_nodes = scatter!(ax,
        jittered_points,
        marker=:rect,
        color=:transparent,
        inspector_label=(ins, idx, pos) -> string(data[][1][idx]))

    on(events(ax.scene).keyboardbutton) do event
        if ispressed(ax.scene, Exclusively(Keyboard.page_up))
            markersize[] = markersize[] + 25
            notify(markersize)
        elseif ispressed(ax.scene, Exclusively(Keyboard.page_down))
            markersize[] = markersize[] - 25
            notify(markersize)
        end
    end

    ins = DataInspector(umap_nodes)
    t_to_pltidx = Observable(Dict(reverse.(enumerate(data[][1]))))

    frame_colors = Ref([])
    views = Observable([])

    @lift begin
        @show "re-rendering"
        disable_interactions(ax)
        hovered[] = nothing
        notify(hovered)

        # instead of clearing everything, why don't we keep them and only delete non-existing ones?
        for ax3d in views[]
            Makie.free(ax3d)
        end
        empty!(views.val)
        empty!(frame_colors[]) # update frame colors
        GC.gc(true)

        reset_limits!(ax)
        center!(ax.scene)

        alignment = $data[2]
        for (i, t) in enumerate($data[1])
            pos = position_on_plot(umap_nodes, i, apply_transform=false)
            # x, y is in global pixel coords
            x, y = shift_project(ax.scene, apply_transform_and_model(umap_nodes, pos))
            # calculate shifted size of marker
            ms = Int.(round.(ax.scene.camera.projectionview[] * markersize_4d[]))[1]
            vp = Rect2i(x - (ms / 2), y - (ms / 2), ms, ms)

            ax3d = Scene(ax.scene,
                show_axis=false,
                viewport=vp,
                backgroundcolor=EMBEDDED_SCENE_BACKGROUND,
                clear=true,
                size=(ms, ms))
            cam3d!(ax3d)

            # sets to color of original leaves
            frame_color = Observable(colors[][i])
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
            on(m_events.obs) do event
                if event.type === MouseEventTypes.over
                    #show_data(ins, umap_nodes, i)
                    hovered[] = t
                    notify(hovered)

                    c = get_cluster_of_transition(cluster_info, t)
                    hovered_cluster[] = c
                    notify(hovered_cluster)
                elseif event.type === MouseEventTypes.out
                    hovered[] = nothing
                    notify(hovered)

                    hovered_cluster[] = nothing
                    notify(hovered_cluster)
                elseif event.type === MouseEventTypes.leftdoubleclick
                    on_click(t)
                end
            end

            # initial render
            sr = selected_render[]
            if sr == "Volume"
                render_views[sr](ax3d, t, alignment[t])
            elseif sr == "Atom"
                render_views["Atom"](ax3d,
                    t,
                    selected_scalar,
                    atom_time,
                    alignment[t]
                )
            else
                inspector = DataInspector(ax3d)
                render_views["Superquadric"](ax3d,
                    t,
                    inspector,
                    alignment[t])
            end
            center!(ax3d)
            yield()

            push!(views.val, ax3d)
            push!(frame_colors[], frame_color)
        end
        t_to_pltidx[] = Dict(reverse.(enumerate(data[][1])))
        notify(views)
        enable_interactions(ax)

    end

    on(selected_render) do sr
        disable_interactions(ax)
        if length(views[]) == length(data[][3])
            alignment = data[][2]
            for (i, ax3d) in enumerate(views[])
                t = data[][1][i]
                foreach(x -> delete!(ax3d, x), filter(y -> !(y isa Wireframe), ax3d.plots))
                if sr == "Volume"
                    render_views[sr](ax3d, t, alignment[t])
                elseif sr == "Atom"
                    render_views["Atom"](ax3d,
                        t,
                        selected_scalar,
                        atom_time,
                        alignment[t])
                else
                    inspector = DataInspector(ax3d)
                    render_views["Superquadric"](ax3d,
                        t,
                        inspector,
                        alignment[t])
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
    onany(ax.xaxis.attributes.limits, ax.yaxis.attributes.limits, markersize_4d) do xlim, ylim, mkr
        if length(views[]) == length(data[][3])
            ms = Int.(round.(ax.scene.camera.projectionview[] * mkr))[1]
            for (i, scene) in enumerate(views[])
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
    c_listener = on(colors) do c_list
        if length(c_list) == length(frame_colors[])
            empty!(highlighted[])
            for (i, c) in enumerate(c_list)
                frame_colors[][i][] = c
            end
        end
    end

    hover_listener = onany(hovered, hovered_cluster) do hov, hc
        if isnothing(hov) && isnothing(hc)
            for (v_idx, ogColor) in highlighted[]
                frame_colors[][v_idx][] = set_color_alpha(ogColor, 0.6)
            end
            empty!(highlighted[])
        end

        if !isnothing(hov) && hov in data[][1]
            v_idx = t_to_pltidx[][hov]
            ogColor = frame_colors[][v_idx][]
            frame_colors[][v_idx][] = set_color_alpha(ogColor, 1.0)
            push!(highlighted[], (v_idx, ogColor))
        end

        if !isnothing(hc)
            ts = collect(keys(t_to_pltidx[]))
            for t in ts
                c = get_cluster_of_transition(cluster_info, t)
                if length(intersect(c, hc)) > 0
                    v_idx = t_to_pltidx[][t]
                    ogColor = frame_colors[][v_idx][]
                    frame_colors[][v_idx][] = set_color_alpha(ogColor, 1.0)
                    push!(highlighted[], (v_idx, ogColor))
                end
            end
        end
    end

    # if I wanted to do this I could just write C
    on(events(ax.scene).window_open) do e
        if !e
            for l in hover_listener
                off(l)
                l = nothing
            end
            empty!(hover_listener)
            #off(c_listener)
            c_listener = nothing

            for ax3d in views[]
                empty!(ax3d)
                Makie.free(ax3d)
            end
            empty!(views[])
            empty!(frame_colors[]) # update frame colors
            GC.gc(true)
        end
    end

    return umap_nodes
end
