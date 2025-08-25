function simple_atom_view!(scene::Makie.Scene,
    ap::Tuple{Matrix{Float32},Matrix{Float32}},
    scalars::Observable{Vector{Float32}},
    scalar_range::Observable{Tuple{Float32,Float32}},
    cmap,
    time::Observable{Float32})

    points = @lift Point3f.(eachrow((ap[1] + ((ap[2] - ap[1]) .* $time))))
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

function apply_alignment_to_scene(scene, alignment)
    R, flip = alignment
    rr = hcat(R, [0, 0, 0])
    fr = transpose(vcat(rr, transpose([0; 0; 0; 1])))
    scene.transformation.model[] = Float64.(fr)
end

function simple_arrow_view!(scene::Makie.Scene,
    ap::Tuple{Matrix{Float32},Matrix{Float32}},
    time::Observable{Float32},
    cmap,
    vel::Vector{GeometryBasics.Point{3,Float32}},
    correlation::Vector{Float32},
    corrThreshold::Observable{Float32})

    # use this function to set any variables that need to be equal length in a makie plot, need velocities, points and colors
    # i know its annoying to use a tuple, but its the only way to prevent crashes
    d = @lift begin
        points = Point3f.(eachrow(ap[1])) .+ (vel .* Ref($time))
        velocities = vel .* (correlation .>= Ref($corrThreshold))
        return points, velocities, correlation
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

    return h, s, v, d
end

function volume_view!(scene::Makie.Scene,
    vd::AbstractArray{Float32},
    sampleRangeExtrema::Observable{Tuple{Tuple{Float64,Float64},Tuple{Float64,Float64},Tuple{Float64,Float64}}},
    vol_cmap::Observable{Vector{ColorTypes.RGBA{Float32}}},
    volumeRange::Observable{Tuple{Float32,Float32}})

    v_lo = volume!(scene,
        lift(x -> x[1], sampleRangeExtrema),
        lift(x -> x[2], sampleRangeExtrema),
        lift(x -> x[3], sampleRangeExtrema),
        vd;
        colormap=lift(x -> view(x, 1:49), vol_cmap),
        highclip=:transparent,
        lowclip=:transparent,
        algorithm=:absorption,
        fxaa=false,
        transparency=true,
        shading=NoShading,
        inspectable=false,
        colorrange=lift(x -> (x[1], 0.0), volumeRange))

    v_hi = volume!(scene,
        lift(x -> x[1], sampleRangeExtrema),
        lift(x -> x[2], sampleRangeExtrema),
        lift(x -> x[3], sampleRangeExtrema),
        vd;
        colormap=lift(x -> view(x, 50:100), vol_cmap),
        highclip=:transparent,
        lowclip=:transparent,
        algorithm=:absorption,
        fxaa=false,
        transparency=true,
        shading=NoShading,
        inspectable=false,
        colorrange=lift(x -> (0.0, x[2]), volumeRange))

    # https://github.com/MakieOrg/Makie.jl/blob/master/GLMakie/src/drawing_primitives.jl
    update_cam!(parent_scene(v_lo))

    return v_lo, v_hi
end

function superquadrics_view!(scene::Makie.Scene,
    points::Vector{Point3f},
    sq::Vector{<:GeometryBasics.AbstractMesh},
    colors::Observable{<:AbstractArray{Float32}},
    vol_cmap::Observable{Vector{RGBAf}},
    invariantRange::Observable{Tuple{Float32,Float32}})
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

    vlol = on(v_lo, weak=true) do idx
        meshes = view(sq, idx)
        sel_col = view(colors[], idx)
        lo_sq.val = meshes

        lo_col[] = reduce(vcat, map(x -> fill(x[2], length(x[1].vertex_attributes[:position])), collect(zip(meshes, sel_col))), init=[])
        notify(lo_sq)
    end

    vhil = on(v_hi, weak=true) do idx
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
    return [cam_listener, vlol, vhil], [lo_sq, hi_sq, lo_col, hi_col], [m_lo, m_hi, v]
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
