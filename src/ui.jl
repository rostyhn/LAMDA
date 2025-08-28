function grid_layout(items::AbstractVector{<:Any})::Vector{Point2f}
    s = Int(round(sqrt(length(items))))
    points = Vector{Point2f}(undef, length(items))
    r = 0
    for i in eachindex(items)
        x = mod1(i, s) * 1
        if x == 1
            r += 1
        end
        y = r
        points[i] = Point2f(Float32(x), Float32(y))
    end
    return points
end

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

function apply_alignment_to_scene(scene::Makie.Scene, alignment)
    R, flip = alignment
    # not sure if this should be transposed or not
    rr = hcat(R, [0, 0, 0])
    fr = transpose(vcat(rr, transpose([0; 0; 0; 1])))
    scene.transformation.model[] = Float64.(fr)
end

function simple_arrow_view!(scene::Makie.Scene,
    ap::Tuple{Matrix{Float32},Matrix{Float32}},
    time::Observable{<:AbstractFloat},
    cmap,
    vel::Vector{GeometryBasics.Point{3,Float32}},
    correlation::Vector{Float32},
    corrThreshold::Observable{<:AbstractFloat})

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

function fill_sq(mData::Tuple{Vector{Point3f},Vector{TriangleFace{UInt16}}},
    c::AbstractArray{Float32})::GeometryBasics.Mesh

    p, f = mData
    return GeometryBasics.mesh(p, f, color=per_face(fill(c, length(f)), f))
end

function superquadrics_view!(scene::Makie.Scene,
    points::Vector{Point3f},
    colors::Observable{<:AbstractArray{Float32}},
    spa::Vector{Vector{Vec3f}},
    vol_cmap::Observable{Vector{RGBAf}},
    invariantRange::Observable{Tuple{Float32,Float32}})
    # try to only render visible points, helps with point picking when hovering 
    sq = @lift begin
        ip = collect(zip($colors, eachindex(points)))

        v_lo = getindex.(filter(x -> x[1] < -0.01, ip), 2)
        v_hi = getindex.(filter(x -> x[1] > 0.01, ip), 2)

        lo_sq = GeometryBasics.merge(fill_sq.(superquadric.(1.0, view(points, v_lo), view(spa, v_lo), 3.0, 0.2), view($colors, v_lo)))
        hi_sq = GeometryBasics.merge(fill_sq.(superquadric.(1.0, view(points, v_hi), view(spa, v_hi), 3.0, 0.2), view($colors, v_hi)))

        return lo_sq, hi_sq
    end

    m_lo = mesh!(
        scene,
        lift(x -> x[1], sq),
        colorrange=lift(x -> (x[1], 0.0), invariantRange),
        colormap=lift(x -> x[1:49], vol_cmap),
        inspectable=false
    )

    m_hi = mesh!(
        scene,
        lift(x -> x[2], sq),
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

    cam_listener = on(sq) do ls
        update_cam!(parent_scene(m_lo))
    end

    update_cam!(parent_scene(m_lo))
    return [cam_listener], [lo_sq, hi_sq], [m_lo, m_hi, v]
end

function draw_bbox_pixel_space!(scene, lo, hi; color=:red, width::Int=1)
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

# as long as this is added last, it should be shown correctly on the screen
# TODO: observable version
function multiline_tooltip(fig, text; margin=2.5, fontsize=16)
    texts = reverse(collect(split(text, "\n")))

    # 1px == 3/4pt
    fs_px = fontsize * (4 / 3)

    # gets actual pixel width of each string
    bboxes = Makie.text_bb.(texts, Makie.defaultfont(), fontsize)
    widths = getindex.(map(x -> x.widths, bboxes), 1)

    size = (maximum(widths) + margin, length(texts) * fs_px + margin)
    tt = Scene(fig.scene, size=size,
        show_axis=false,
        viewport=Rect2i(100, 100, size...),
        backgroundcolor=colorant"#ffffca",
        camera=campixel!,
        clear=true,
    )

    p = poly!(tt, Rect2i(0, 0, size...), color=colorant"#ffffca",
        inspectable=false, strokecolor=:black, strokewidth=1)

    txt = text!(tt, fill(margin, length(texts)), eachindex(texts) .* fs_px;
        inspectable=false,
        overdraw=true,
        color=:black,
        align=(:left, :top),
        text=texts,
        fontsize=fontsize)

    translate!(p, 0, 0, 100)
    translate!(txt, 0, 0, 102)

    return tt, size
end

function inline_image(fig, img, text::String; kwargs...)
    ax = Axis(fig; aspect=AxisAspect(1), kwargs...)
    hidedecorations!(ax)
    hidespines!(ax)
    disable_interactions(ax)

    attach_image(fig, ax, img, text)
    return ax
end

function attach_image(fig, ax, img, text)
    image!(ax, rotr90(img), inspectable=false)
    tt, tSize = multiline_tooltip(fig, text)
    tt.visible[] = false
    m_events = addmouseevents!(ax.scene)
    on(m_events.obs) do e
        if e.type == MouseEventTypes.over
            px, py = float.(mouseposition(fig.scene))
            # proposed viewport
            vp = Rect2i(px - (tSize[1] / 2), py + 5, tSize)

            # intersected viewport
            ivp = GeometryBasics.intersect(vp, fig.scene.viewport[])

            wvp = widths(vp)
            wivp = widths(ivp)

            rx = wvp[1] - wivp[1]
            ry = wvp[2] - wivp[2]

            if rx > 0 || ry > 0
                o = origin(vp)
                tt.viewport[] = Rect2i(o[1] - rx, o[2] - ry, tSize)
            else
                tt.viewport[] = vp
            end

            tt.visible[] = true

        else
            tt.visible[] = false
        end
    end
    return ax
end

function inset_image(fig, loc, img, text; kwargs...)
    ax = Axis(loc; aspect=AxisAspect(1), kwargs...)
    hidedecorations!(ax)
    hidespines!(ax)
    disable_interactions(ax)

    attach_image(fig, ax, img, text)
    return ax
end

function help_icon(fig, loc, txt)
    # places help icon in top right corner
    inset_image(fig, loc, HELP_ICON, txt; width=Makie.Fixed(15), height=Makie.Fixed(15), halign=1.0, valign=1.0, tellwidth=false, tellheight=false)
end
