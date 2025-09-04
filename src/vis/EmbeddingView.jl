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
    invariant_selection::Observable{String},
    atom_time::Observable{Float32},
    render_views::Dict{String,Function},
    hovered::MaybeObservable{Transition},
    hovered_cluster::MaybeObservable{ClusterSet},
    colors::Observable{Vector{RGBAf}};
    on_click::Function=(x) -> (),
    markersize::Observable{Int}=Observable(100),
)
    # the makie.onany calls here ensure that the listeners are bound to the scene and get gc'd correctly
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

    nodes = scatter!(ax,
        embedding,
        marker=:rect,
        color=:transparent,
        inspector_label=(ins, idx, pos) -> join(string.(cluster_data[].ts[idx], base=10), ","))

    Makie.onany(ax.scene, embedding, update=true) do e
        autolimits!(ax)
        center!(ax.scene)
    end

    on(events(ax.scene).keyboardbutton) do event
        if ispressed(ax.scene, Exclusively(Keyboard.page_up))
            markersize[] = markersize[] + 25
        elseif ispressed(ax.scene, Exclusively(Keyboard.page_down))
            markersize[] = markersize[] - 25
        elseif ispressed(ax.scene, Exclusively(Keyboard.a))
            show_alignment[] = !show_alignment[]
            if show_alignment[]
                println("aligned")
            else
                println("identity")
            end
            return Consume(true)
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
    highlighted = Ref([])
    frame_widths = Ref([])

    #=
    r_window = Figure(size=(2 * 1920, 2 * 1920))
    render_ax = LScene(r_window.scene,
        show_axis=false,
        scenekw=(backgroundcolor=EMBEDDED_SCENE_BACKGROUND, camera=cam3d!, clear=true),
    )
    r_window[1, 1] = render_ax=#

    function create_scene(d, ms, alignment, render_fn)
        i, t = d
        pos = position_on_plot(nodes, i, apply_transform=false)
        # x, y is in global pixel coords
        x, y = shift_project(ax.scene, apply_transform_and_model(nodes, pos))
        # calculate shifted size of marker
        vp = Rect2i(x - (ms / 2), y - (ms / 2), ms, ms)

        ax3d = Scene(ax.scene,
            show_axis=false,
            viewport=vp,
            backgroundcolor=EMBEDDED_SCENE_BACKGROUND,
            clear=true,
            camera=cam3d!,
            size=(ms, ms))

        # for debug purposes only!
        Makie.onany(ax3d, show_alignment, update=true) do showAlignment
            if showAlignment
                apply_alignment_to_scene(ax3d, alignment[t])
            else
                ax3d.transformation.model[] = Matrix(1.0I, 4, 4)
            end
        end

        frame_color = Observable(colors[][i])
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

        flip = alignment[t][2]
        m_events = addmouseevents!(ax3d)
        onmouseover(m_events) do e
            #@show join(string.(t, base=10), ",")
            hovered[] = t
        end
        onmouseout(m_events) do e
            hovered[] = nothing
            hovered_cluster[] = nothing
        end
        # screenshot function
        #=onmousemiddledown(m_events) do e
            render_ax.scene.visible[] = true
            foreach(x -> delete!(render_ax.scene, x), render_ax.scene.plots)
            render_fn(render_ax.scene, t, flip)
            apply_alignment_to_scene(render_ax.scene, alignment[t])
            center!(render_ax.scene)

            update_cam!(render_ax.scene, cameracontrols(ax3d))
            x = "$(join(string.(t, base=10), "_")).png"
            GLMakie.save(x, r_window, update=false)
            render_ax.scene.visible[] = false
        end=#
        onmouseleftdoubleclick(m_events) do e
            on_click(t)
        end

        # initial render
        render_fn(ax3d, t, flip)
        center!(ax3d)
        push!(views[], Ref(ax3d))
        push!(frame_colors[], frame_color)
        push!(frame_widths[], linewidth)
    end

    t_to_pltidx = lift(ax.scene, cluster_data) do cd
        @debug "Rendering embedding"
        disable_interactions(ax)

        # instead of clearing everything, why don't we keep them and only delete non-existing ones?
        for ax3d in views[]
            s = ax3d[]
            empty!(s)
            Makie.free(s)
            s = nothing
            ax3d = nothing
        end

        empty!(views[])
        empty!(frame_widths[])
        empty!(frame_colors[]) # update frame colors
        empty!(highlighted[])
        GC.gc(true)

        ts = cd.ts
        alignment = cd.alignment

        t_to_pltidx = Dict(reverse.(enumerate(ts)))

        sr = selected_render[]
        render_fn = (sr == "Atom") ? (x, y, z) ->
            render_views["Atom"](
                x,
                y,
                selected_scalar,
                atom_time,
                z
            ) : (x, y, z) ->
            render_views["Superquadric"](x,
                y,
                invariant_selection,
                z)

        ms = Int.(round.(ax.scene.camera.projectionview[] * markersize_4d[]))[1]
        create_scene.(enumerate(ts), Ref(ms), Ref(alignment), Ref(render_fn))

        center!(ax.scene)
        hovered[] = nothing
        enable_interactions(ax)
        return t_to_pltidx
    end

    function update_scene(d, ts, alignment, sr, render_fn)
        i, ptr = d
        ax3d = ptr[]
        t = ts[i]
        flip = alignment[t][2]
        foreach(x -> delete!(ax3d, x),
            filter(y -> !(y isa Wireframe), ax3d.plots))

        render_fn(ax3d, t, flip)
        center!(ax3d)
        # block for a millisecond so makie can catch up
        # otherwise it seems like the renderer gets overwhelmed & it just goes oom
    end

    Makie.onany(ax.scene, selected_render) do sr
        disable_interactions(ax)
        if length(views[]) == length(embedding[])
            ts = cluster_data[].ts
            alignment = cluster_data[].alignment
            sr = selected_render[]
            render_fn = (sr == "Atom") ? (x, y, z) ->
                render_views["Atom"](
                    x,
                    y,
                    selected_scalar,
                    atom_time,
                    z
                ) : (x, y, z) ->
                render_views["Superquadric"](x,
                    y,
                    invariant_selection,
                    z)
            update_scene.(enumerate(views[]), Ref(ts), Ref(alignment), Ref(sr), Ref(render_fn))
            GC.gc(true)
        end
        enable_interactions(ax)
    end

    function shift_point(d, ms::Int)
        i, ptr = d
        scene = ptr[]
        pos = position_on_plot(nodes, i, apply_transform=false)
        x, y = shift_project(ax.scene, apply_transform_and_model(nodes, pos))

        vp = Rect2i(x - (ms / 2), y - (ms / 2), ms, ms)
        vp = GeometryBasics.intersect(vp, ax.scene.viewport[])
        vw = widths(vp)

        if any(w -> w <= 0, vw)
            vp = Rect2i(0, 0, 0, 0)
        end

        scene.viewport[] = vp
    end

    #https://github.com/MakieOrg/Makie.jl/blob/381cf4a1ade5bf1a36b254ce6daccb5cbc71939e/GLMakie/assets/shader/dots.vert#L55
    Makie.onany(ax.scene,
        ax.xaxis.attributes.limits,
        ax.yaxis.attributes.limits,
        markersize_4d,
        embedding,
        t_to_pltidx,
        ax.scene.viewport
    ) do xlim, ylim, mkr, p, pindx, svp
        if length(views[]) == length(p)
            ms = Int.(round.(ax.scene.camera.projectionview[] * mkr))[1]
            shift_point.(enumerate(views[]), Ref(ms))
        end
    end

    Makie.onany(ax.scene, colors) do c_list
        if length(c_list) == length(frame_colors[])
            empty!(highlighted[])
            for (i, c) in enumerate(c_list)
                frame_colors[][i][] = c
            end
        end
    end


    # converts hovered into its cluster
    # since you can't hover both a cluster and a transition simultaneously, this frees up the logic underneath
    Makie.onany(ax.scene, hovered) do hov
        pindx = to_value(t_to_pltidx)
        if !isnothing(hov) && haskey(pindx, hov)
            c = get_cluster_of_transition(cluster_data[], pindx[hov])
            hovered_cluster[] = c
        end
    end

    Makie.onany(ax.scene, hovered_cluster) do hc
        for v_idx in highlighted[]
            frame_widths[][v_idx][] = 3
        end
        empty!(highlighted[])

        if !isnothing(hc)
            for i in values(to_value(t_to_pltidx))
                c = get_cluster_of_transition(cluster_data[], i)
                if !isnothing(c) && length(intersect(c, hc)) > 0
                    if i <= length(frame_widths[])
                        frame_widths[][i][] = 10
                        push!(highlighted[], i)
                    end
                end
            end
        end
    end


    # if I wanted to do this I could just write C
    cleanup = function ()
        @debug "kill embedding"
        for ptr in views[]
            ax3d = ptr[]
            empty!(ax3d)
            Makie.free(ax3d)
            ax3d = nothing
            ptr = nothing
        end

        empty!(views[])
        Observables.clear.(frame_colors[])
        empty!(frame_colors[]) # update frame colors
        empty!(frame_widths[])

        empty!(ax.scene)
        Makie.free(ax.scene)
    end

    return cleanup
end
