function umap_graph_view!(
    loc,
    ts,
    matrix,
    selected_render,
    selected_scalar,
    atom_time,
    render_views,
    hovered=MaybeObservable{Tuple{Int,Int}};
    highlight_borders=Observable(false),
    on_click=(x) -> (),
    markersize=100,
)
    ax = Axis(loc, backgroundcolor=:transparent)
    on(highlight_borders, update=true) do hb
        border_color = to_color(:black)
        if hb
            border_color = to_color(:red)
        end
        ax.topspinecolor[] = border_color
        ax.bottomspinecolor[] = border_color
        ax.leftspinecolor[] = border_color
        ax.rightspinecolor[] = border_color
    end
    deregister_interaction!(ax, :rectanglezoom)
    hidedecorations!(ax)

    campixel!(ax.scene)

    if length(ts) > 1
        em = transpose(umap(transpose(matrix), 2; metric=:precomputed, n_neighbors=min(length(ts) - 1, 15)))
        embedding = map(x -> Point2f(x), eachrow(em))
    else
        embedding = [Point2f(0.0, 0.0)]
    end

    # invisible scatter plot to set up camera
    umap_nodes = scatter!(ax, embedding, marker=:rect, color=:transparent, inspector_label=(ins, idx, pos) -> string(ts[idx]))

    ins = DataInspector(umap_nodes)
    center!(ax.scene)

    t_to_pltidx = Dict(reverse.(enumerate(ts)))

    views = []
    for (i, t) in enumerate(ts)
        pos = position_on_plot(umap_nodes, i, apply_transform=false)
        # x, y is in global pixel coords
        x, y = shift_project(ax.scene, apply_transform_and_model(umap_nodes, pos))
        # calculate shifted size of marker
        ms = Int.(round.(ax.scene.camera.projectionview[] * Point4f(markersize, markersize, 0, 0)))[1]
        size = Observable((ms, ms))

        vp = Observable(Rect2i(x - (ms / 2), y - (ms / 2),
            ms,
            ms))

        ax3d = Scene(ax.scene,
            show_axis=false,
            viewport=vp,
            backgroundcolor=:black,
            clear=true,
            size=size)

        cam3d!(ax3d)

        inspector = DataInspector(ax3d)

        rendered = []
        @lift begin
            foreach(x -> delete!(ax3d, x), rendered)
            empty!(rendered)
            if $selected_render == "Volume"
                v_lo, v_hi = render_views[$selected_render](ax3d, Observable(t))
                rendered = vcat(rendered, [v_lo, v_hi])
            elseif $selected_render == "Atom"
                s = render_views["Atom"](ax3d,
                    Observable(t),
                    selected_scalar,
                    atom_time,
                    Observable(ts))

                rendered = [s]
            else
                il, is, plots = render_views["Superquadric"](ax3d,
                    Observable(t),
                    inspector, Observable(ts))
                rendered = plots
            end
            center!(ax3d)
        end

        m_events = addmouseevents!(ax3d)
        on(m_events.obs) do event
            if event.type === MouseEventTypes.over
                show_data(ins, umap_nodes, i)
                hovered[] = t
                notify(hovered)
            elseif event.type === MouseEventTypes.out
                hovered[] = nothing
                notify(hovered)
            elseif event.type === MouseEventTypes.leftdoubleclick
                on_click(t)
            end
        end
        push!(views, (vp, size, ax3d))
    end

    #https://github.com/MakieOrg/Makie.jl/blob/381cf4a1ade5bf1a36b254ce6daccb5cbc71939e/GLMakie/assets/shader/dots.vert#L55
    onany(ax.xaxis.attributes.limits, ax.yaxis.attributes.limits) do xlim, ylim
        ms = Int.(round.(ax.scene.camera.projectionview[] * Point4f(markersize, markersize, 0, 0)))[1]
        for (i, (vp, size, scene)) in enumerate(views)
            pos = position_on_plot(umap_nodes, i, apply_transform=false)
            x, y = shift_project(ax.scene, apply_transform_and_model(umap_nodes, pos))

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

    highlighted = []
    on(hovered) do hov
        for (v_idx) in highlighted
            views[v_idx][3].backgroundcolor[] = to_color(:black)
        end
        empty!(highlighted)

        if !isnothing(hov) && hov in ts
            v_idx = t_to_pltidx[hov]
            views[v_idx][3].backgroundcolor[] = to_color(:grey)
            push!(highlighted, v_idx)
        end
    end

    return umap_nodes
end
