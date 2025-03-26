using Observables

function umap_graph_view!(
    loc,
    data,
    selected_render,
    selected_scalar,
    atom_time,
    render_views,
    hovered,
    alignment,
    hovered_cluster,
    cluster_info;
    highlight_borders=Observable(false),
    on_click=(x) -> (),
    markersize=100,
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

    embedding = @lift begin
        disable_interactions(ax)
        if length($data[1]) > 2
            em = transpose(umap(transpose($data[2]), 2;
                metric=:precomputed,
                min_dist=2,
                n_neighbors=min(length($data[1]) - 1, 15)))
            return map(x -> Point2f(x), eachrow(em))
        else
            points = Point2f[]
            for (i, t) in enumerate($data[1])
                push!(points, Point2f((i - 1) * 5, 0.0))
            end
            return points
        end
    end

    # invisible scatter plot to set up camera
    umap_nodes = scatter!(ax,
        embedding,
        marker=:rect,
        color=:transparent,
        inspector_label=(ins, idx, pos) -> string(data[][1][idx]))

    ins = DataInspector(umap_nodes)
    center!(ax.scene)

    t_to_pltidx = Observable(Dict(reverse.(enumerate(data[][1]))))
    markersize_4d = Point4f(markersize, markersize, 0, 0)

    frame_colors = Ref([])
    views = Observable([])
    @lift begin
        hovered[] = nothing
        notify(hovered)
        for (ax3d, rendered) in views[]
            Makie.free(ax3d)
        end
        empty!(views[])
        empty!(frame_colors[]) # update frame colors
        GC.gc()

        for (i, t) in enumerate(data.val[1])
            pos = $embedding[i]
            # x, y is in global pixel coords
            x, y = shift_project(ax.scene, apply_transform_and_model(umap_nodes, pos))
            # calculate shifted size of marker
            ms = Int.(round.(ax.scene.camera.projectionview[] * markersize_4d))[1]
            vp = Rect2i(x - (ms / 2), y - (ms / 2), ms, ms)

            ax3d = Scene(ax.scene,
                show_axis=false,
                viewport=vp,
                backgroundcolor=EMBEDDED_SCENE_BACKGROUND,
                clear=true,
                size=(ms, ms))
            cam3d!(ax3d)

            # sets to color of original leaves
            frame_color = Observable(data[][3][i])
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

            rendered = Ref([])
            m_events = addmouseevents!(ax3d)
            on(m_events.obs) do event
                if event.type === MouseEventTypes.over
                    show_data(ins, umap_nodes, i)
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
            push!(views[], (ax3d, rendered))
            push!(frame_colors[], frame_color)
        end
        t_to_pltidx[] = Dict(reverse.(enumerate(data[][1])))
    end

    onany(selected_render, views, embedding; update=true) do sr, v, e
        if length(v) == length(e)
            for (i, (ax3d, rendered)) in enumerate(v)
                t = data[][1][i]
                foreach(x -> delete!(ax3d, x), rendered[])
                empty!(rendered[])
                if sr == "Volume"
                    v_lo, v_hi = render_views[sr](ax3d, Observable(t))
                    rendered[] = [v_lo, v_hi]
                elseif sr == "Atom"
                    s = render_views["Atom"](ax3d,
                        Observable(t),
                        selected_scalar,
                        atom_time,
                        alignment)
                    rendered[] = [s]
                else
                    inspector = DataInspector(ax3d)
                    il, is, plots = render_views["Superquadric"](ax3d,
                        Observable(t),
                        inspector,
                        alignment)
                    rendered[] = plots
                end
                center!(ax3d)
                # block for a millisecond so makie can catch up
                # otherwise it seems like the renderer gets overwhelmed & it just goes oom
                sleep(0.001)
            end
        end
        reset_limits!(ax)
        center!(ax.scene)
        enable_interactions(ax)
    end

    #https://github.com/MakieOrg/Makie.jl/blob/381cf4a1ade5bf1a36b254ce6daccb5cbc71939e/GLMakie/assets/shader/dots.vert#L55
    onany(ax.xaxis.attributes.limits, ax.yaxis.attributes.limits) do xlim, ylim
        if length(views[]) == length(embedding[])
            ms = Int.(round.(ax.scene.camera.projectionview[] * markersize_4d))[1]
            for (i, (scene, rendered)) in enumerate(views[])
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
    onany(hovered, hovered_cluster) do hov, hc
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

    return umap_nodes
end
