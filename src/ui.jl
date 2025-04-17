function clear_layout(layout::GridLayout)
    # Begin by removing the blocks from the recursive GridLayout structure
    items_to_remove = []
    for block in Makie.contents(layout)
        if typeof(block) == GridLayout
            clear_layout(block)
        else
            push!(items_to_remove, block)
        end
    end

    for i in items_to_remove
        empty!(i.blockscene)
        delete!(i)
    end

    Makie.trim!(layout)
    GridLayoutBase.remove_from_gridlayout!(layout.layoutobservables.gridcontent[])
end

# switches what is being rendered inside a scene cleanly.
# pass a select_fn with selector as a parameter and then basically do whatever you want
# can modify the grid the scene belongs to and it will get cleared up here
function scene_switcher(scene, grid, selector, select_fn)
    scene_listeners = Vector{Any}()
    ui_elements = Vector{Any}()
    @lift begin
        # cleanup
        empty!(scene)

        for listener in scene_listeners
            off(listener)
            listener = nothing
        end
        empty!(scene_listeners)

        # clear UI elements
        for g in ui_elements
            clear_layout(g)
        end
        Makie.trim!(grid)

        il, is = select_fn($selector)

        for l in il
            push!(scene_listeners, l)
        end

        for s in is
            push!(ui_elements, s)
        end
    end
end

function simple_atom_view!(scene, ap::Observable{Tuple{Matrix{Float32},Matrix{Float32}}}, scalars::Observable{Vector{Float32}}, scalar_range, cmap, time::Observable{Float64})
    int_pos = lift((x, y) -> x[1] + ((x[2] - x[1]) .* y), ap, time)

    # makes it so the atom view can handle points changing
    colors = Observable(scalars[])
    points = Observable(Point3f.(eachrow(int_pos[])))
    onany(int_pos, scalars) do ip, s
        points.val = Point3f.(eachrow(ip))
        colors[] = s
        points[] = points[]
    end

    s = meshscatter!(scene,
        points;
        color=colors,
        colorrange=scalar_range,
        lowclip=:transparent,
        colormap=cmap,
        ssao=true,
        transparency=true,
        markersize=0.7)

    s.inspectable[] = false
    update_cam!(parent_scene(s))

    return s
end

function simple_arrow_view!(scene,
    ap::Observable{Tuple{Matrix{Float32},Matrix{Float32}}},
    time::Observable{Float64},
    cmap,
    vel::Observable{Vector{GeometryBasics.Point{3,Float32}}},
    correlation::Observable{Vector{Float32}},
    corrThreshold::Observable{Float64})

    # int_pos = lift((x, y) -> x[1] + ((x[2] - x[1]) .* y), ap, time) # median is moving for debugging

    # use this function to set any variables that need to be equal length in a makie plot, need velocities, points and colors
    # i know its annoying to use a tuple, but its the only way to prevent crashes
    d = @lift begin
        points = Point3f.(eachrow($ap[1])) .+ ($vel .* $time)
        velocities = $vel .* ($correlation .>= Ref($corrThreshold))
        return points, velocities, $correlation
    end

    h = arrows!(scene,
        lift(x -> x[1], d),
        lift(x -> x[2], d);
        color=lift(x -> x[3], d),
        arrowsize=1.2,
        colorrange=lift(x -> (x, 1.0), corrThreshold),
        colormap=cmap,
        lowclip=:transparent,
        transparency=true,
        inspectable=false,
    )

    v = meshscatter!(scene,
        lift(x -> x[1], d);
        color=lift(x -> x[3], d),
        colorrange=lift(x -> (0.0, x), corrThreshold),
        marker=:Sphere,
        colormap=:gist_yarg,
        lowclip=:transparent,
        highclip=:transparent,
        transparency=true,
        inspectable=false,
        markersize=0.2)

    s = meshscatter!(scene,
        lift(x -> x[1], d);
        color=lift(x -> x[3], d),
        marker=:Sphere,
        transparency=true,
        inspectable=false,
        lowclip=:transparent,
        colormap=cmap,
        colorrange=lift(x -> (x, 1.0), corrThreshold),
        markersize=0.7)

    update_cam!(parent_scene(s))
    center!(parent_scene(s))

    return h, s, v
end

function volume_view!(scene, vd, sampleRanges, vol_cmap, volumeRange)
    t = Observable(Transformation())

    v_lo = volume!(scene,
        lift(x -> extrema(x[1]), sampleRanges),
        lift(x -> extrema(x[2]), sampleRanges),
        lift(x -> extrema(x[3]), sampleRanges),
        vd;
        colormap=lift(x -> x[1:49], vol_cmap),
        highclip=:transparent,
        lowclip=:transparent,
        algorithm=:absorption,
        fxaa=false,
        transparency=true,
        transformation=t,
        shading=NoShading,
        inspectable=false,
        colorrange=lift(x -> (x[1], 0.0), volumeRange))

    v_hi = volume!(scene,
        lift(x -> extrema(x[1]), sampleRanges),
        lift(x -> extrema(x[2]), sampleRanges),
        lift(x -> extrema(x[3]), sampleRanges),
        vd;
        colormap=lift(x -> x[50:100], vol_cmap),
        highclip=:transparent,
        lowclip=:transparent,
        algorithm=:absorption,
        fxaa=false,
        transparency=true,
        shading=NoShading,
        transformation=t,
        inspectable=false,
        colorrange=lift(x -> (0.0, x[2]), volumeRange))


    # FIXME sometimes the volume will get rotated so hard it disappears
    # could be a floating point precision issue?
    # if called before screen is rendered it crashes
    #=on(rotation, update=update) do rot
        shift, R, flip, ref_t = rot
        rr = hcat(R, [0, 0, 0])
        fr = transpose(vcat(rr, transpose([0; 0; 0; 1])))

        # https://github.com/MakieOrg/Makie.jl/blob/master/GLMakie/src/drawing_primitives.jl
        t[].origin[] = Float64.(shift)
        t[].model[] = Float64.(fr)

        notify(t)
        update_cam!(parent_scene(v_lo))
    end=#

    # update_cam!(parent_scene(v_lo))

    return v_lo, v_hi
end

function superquadrics_view!(scene, points, sq, colors, vol_cmap, invariantRange, inspector)
    # try to only render visible points, helps with point picking when hovering 
    v_lo = lift((x, y) -> getindex.(filter(x -> x[1] < -0.01, collect(zip(x, eachindex(y)))), 2), colors, points)
    v_hi = lift((x, y) -> getindex.(filter(x -> x[1] > 0.01, collect(zip(x, eachindex(y)))), 2), colors, points)

    lo_sq = Observable(sq[][v_lo[]])
    lo_col = Observable(colors[][v_lo[]])

    hi_sq = Observable(sq[][v_hi[]])
    hi_col = Observable(colors[][v_hi[]])

    on(v_lo) do idx
        lo_sq.val = sq[][idx]
        lo_col[] = colors[][idx]
        notify(lo_sq)
    end

    on(v_hi) do idx
        hi_sq.val = sq[][idx]
        hi_col[] = colors[][idx]
        notify(hi_sq)
    end

    m_lo = mesh!(
        scene,
        lo_sq,
        color=lo_col,
        #highclip=:transparent,
        #transparency=true,
        # shading=NoShading,
        colorrange=lift(x -> (x[1], 0.0), invariantRange),
        colormap=lift(x -> x[1:49], vol_cmap),
        # fxaa=false
    )
    m_lo.inspectable[] = false

    m_hi = mesh!(
        scene,
        hi_sq,
        color=hi_col,
        #lowclip=:transparent,
        # transparency=true,
        # shading=NoShading,
        colorrange=lift(x -> (0.0, x[2]), invariantRange),
        colormap=lift(x -> x[50:100], vol_cmap),
        # fxaa=false
    )
    v = meshscatter!(scene,
        points;
        color=:gray,
        marker=:Sphere,
        transparency=true,
        inspectable=false,
        markersize=0.2)

    m_hi.inspectable[] = false

    cam_listener = on(lo_sq) do ls
        update_cam!(parent_scene(m_lo))
    end

    # no other way around this other than this super ugly way, 
    # makie renders this as one plot, which the inspector grabs a bounding box around
    #=sqHoverListener = on(events(scene).mouseposition) do mp
        if is_mouseinside(scene)
            # might be able to use onpick()
            plot, idx = pick(scene)
            if plot != Nothing
                pos = position_on_plot(plot, idx)
                if !isnan(pos)
                    inspector.plot.text[] = string(plot.color[][idx])
                    inspector.plot.visible[] = true
                    inspector.plot.position = mp
                end
            end
            return Consume(true)
        end
        return Consume(false)
    end=#

    update_cam!(parent_scene(m_lo))
    return [cam_listener], [], [m_lo, m_hi, v]
end

function draw_bbox_pixel_space!(scene, lo, hi; color=:red, width=1)
    bbox = Rect2(lo - 0.5, lo - 0.5, (hi - lo) + 1, (hi - lo) + 1)

    p = wireframe!(
        scene,
        bbox,
        color=color,
        visible=true,
        inspectable=false,
        depth_shift=-1.0f-3,
        linewidth=width
    )
    return p
end

function top_bar(window, title, num_cols)
    g = GridLayout()
    # https://juliagraphics.github.io/Colors.jl/stable/namedcolors/
    Box(window[1, 1:num_cols], color=:grey95, strokevisible=false)

    window[1, 1:num_cols] = g
    g[1, 1] = Label(window, "LAMDA", fontsize=30, font=:bold, halign=:left)
    g[1, 2] = Label(window, title, fontsize=30, font=:italic, tellwidth=false, halign=:left)

    gg = GridLayout()
    g[1, 3] = gg

    # useful to see exactly how much room you need 
    # Box(g[1, 3], color=:green)

    return gg
end

function set_text(txtbox, s)
    txtbox.displayed_string[] = s
    txtbox.stored_string[] = s
end

function minimize_screen(s::GLMakie.Screen; m=GLFW.GetPrimaryMonitor())
    vd = GLFW.GetVideoMode(m)
    h = div(vd.height, 2)
    w = div(vd.width, 2)
    # should place the window in the top left corner of the screen
    GLFW.SetWindowMonitor(s.glscreen, GLFW.Monitor(C_NULL), 0.0, vd.height, w, h, GLFW.DONT_CARE)
end
