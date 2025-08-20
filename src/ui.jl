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

function simple_atom_view!(scene, ap, scalars::Observable{Vector{Float32}}, scalar_range, cmap, time::Observable{Float64})
    points = lift(x -> Point3f.(eachrow((ap[1] + ((ap[2] - ap[1]) .* x)))), time)
    s = meshscatter!(scene,
        points;
        color=scalars,
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

    h = arrows3d!(scene,
        lift(x -> x[1], d),
        lift(x -> x[2], d);
        color=lift(x -> x[3], d),
        markerscale=1.2,
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
        inspectable=false,
        colorrange=lift(x -> (0.0, x[2]), volumeRange))

    # if called before screen is rendered it crashes

    # https://github.com/MakieOrg/Makie.jl/blob/master/GLMakie/src/drawing_primitives.jl
    update_cam!(parent_scene(v_lo))

    # update_cam!(parent_scene(v_lo))

    return v_lo, v_hi
end

function superquadrics_view!(scene, points, sq, colors, vol_cmap, invariantRange)
    # try to only render visible points, helps with point picking when hovering 

    ip = lift(x -> collect(zip(x, eachindex(points))), colors)
    v_lo = lift(xx -> getindex.(filter(x -> x[1] < -0.01, xx), 2), ip)
    v_hi = lift(xx -> getindex.(filter(x -> x[1] > 0.01, xx), 2), ip)

    lo_sq = Observable(view(sq, v_lo[]))
    # in what universe is this sane 
    lo_col = Observable(reduce(vcat, map(x -> fill(x[2], length(x[1].vertex_attributes[:position])),
            collect(zip(view(sq, v_lo[]), view(colors[], v_lo[])))), init=Float32[]))

    hi_sq = Observable(view(sq, v_hi[]))
    hi_col = Observable(reduce(vcat, map(x -> fill(x[2], length(x[1].vertex_attributes[:position])),
            collect(zip(view(sq, v_hi[]), view(colors[], v_hi[])))), init=Float32[]))

    on(v_lo) do idx
        meshes = view(sq, idx)
        sel_col = view(colors[], idx)
        lo_sq.val = meshes

        lo_col[] = reduce(vcat, map(x -> fill(x[2], length(x[1].vertex_attributes[:position])), collect(zip(meshes, sel_col))), init=[])
        notify(lo_sq)
    end

    on(v_hi, update=true) do idx
        meshes = view(sq, idx)
        sel_col = view(colors[], idx)
        hi_sq.val = meshes

        hi_col[] = reduce(vcat, map(x -> fill(x[2], length(x[1].vertex_attributes[:position])), collect(zip(meshes, sel_col))), init=[])

        notify(hi_sq)
    end

    m_lo = mesh!(
        scene,
        lo_sq,
        color=lo_col,
        colorrange=lift(x -> (x[1], 0.0), invariantRange),
        colormap=lift(x -> x[1:49], vol_cmap),
        inspectable=false
    )

    m_hi = mesh!(
        scene,
        hi_sq,
        color=hi_col,
        colorrange=lift(x -> (0.0, x[2]), invariantRange),
        colormap=lift(x -> x[50:100], vol_cmap),
        inspectable=false
    )
    v = meshscatter!(scene,
        points;
        color=:gray,
        marker=:Sphere,
        transparency=true,
        inspectable=false,
        markersize=0.2)

    cam_listener = on(lo_sq) do ls
        update_cam!(parent_scene(m_lo))
    end

    update_cam!(parent_scene(m_lo))
    return [cam_listener], [lo_sq, hi_sq, lo_col, hi_col], [m_lo, m_hi, v]
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

function minimize_screen(s::GLMakie.Screen; monitor=GLFW.GetPrimaryMonitor())
    vd = GLFW.GetVideoMode(monitor)
    h = div(vd.height, 2)
    w = div(vd.width, 2)
    # should place the window in the top left corner of the screen
    GLFW.SetWindowMonitor(s.glscreen, GLFW.Monitor(C_NULL), 0.0, 0.0, w, h, GLFW.DONT_CARE)
end

function move_window(s::GLMakie.Screen; monitor=GLFW.GetPrimaryMonitor())
    vd = GLFW.GetVideoMode(monitor)
    mp = GLFW.GetMonitorPos(monitor)
    h = div(vd.height, 2)
    w = div(vd.width, 2)
    GLFW.HideWindow(s.glscreen)
    #GLFW.SetWindowMonitor(s.glscreen, GLFW.Monitor(C_NULL), mp.x, mp.y, w, h, GLFW.DONT_CARE)
    GLFW.SetWindowMonitor(s.glscreen, monitor, mp.x, mp.y, vd.width, vd.height, vd.refreshrate)
    yield()

    GLFW.ShowWindow(s.glscreen)
    @show s.glscreen, GLFW.GetWindowMonitor(s.glscreen)
    #minimize_screen(s, monitor=monitor)
end

function inline_image(fig, img, tooltip::String)
    sc = Scene(fig.scene)
    campixel!(sc)
    ax = Axis(sc, aspect=AxisAspect(1))
    hidedecorations!(ax)
    hidespines!(ax)
    disable_interactions(ax)

    image!(ax, rotr90(img), inspector_label=(x, y, z) -> tooltip)
    return ax
end
