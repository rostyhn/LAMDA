function apply_alignment_to_scene(scene::Makie.Scene, alignment)
    R, flip = alignment
    rr = hcat(R, [0, 0, 0])
    fr = transpose(vcat(rr, transpose([0; 0; 0; 1])))
    scene.transformation.model[] = Float64.(fr)
end

#TODO: these can be rewritten as recipes
#https://docs.makie.org/stable/explanations/recipes.html 
function simple_atom_view!(scene::Makie.Scene,
    ap::Tuple{Matrix{Float32},Matrix{Float32}},
    scalars::Observable{Vector{Float32}},
    scalar_range::Observable{Tuple{Float32,Float32}},
    cmap,
    time::Observable{Float32})

    points = lift(scene, time) do t
        return Point3f.(eachrow((ap[1] + ((ap[2] - ap[1]) .* t))))
    end

    meshscatter!(scene,
        points;
        color=scalars,
        colorrange=scalar_range,
        lowclip=:transparent,
        colormap=cmap,
        ssao=true,
        transparency=true,
        markersize=0.7,
        inspectable=false)

    update_cam!(scene)
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
    d = lift(scene, time, corrThreshold) do t, ct
        points = Point3f.(eachrow(ap[1])) .+ (vel .* Ref(t))
        velocities = vel .* (correlation .>= Ref(ct))
        return points, velocities, correlation
    end

    # the api was changed and made it a lot more difficult to get the
    # arrow sizes right
    h = arrows3d!(scene,
        lift(x -> x[1], d),
        lift(x -> x[2], d);
        color=lift(x -> x[3], d),
        normalize=true,
        tipradius=12.0,
        markerscale=6,
        tiplength=12.0,
        shaftlength=12.0,
        lengthscale=1.2,
        minshaftlength=12.0,
        tailradius=0,
        colorrange=lift(x -> (x, 1.0), corrThreshold),
        colormap=cmap,
        lowclip=:transparent,
        transparency=true,
        inspectable=false,
    )

    s = meshscatter!(scene,
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

    v = meshscatter!(scene,
        lift(x -> x[1], d);
        color=lift(x -> x[3], d),
        marker=:Sphere,
        transparency=true,
        inspectable=false,
        lowclip=:transparent,
        colormap=cmap,
        colorrange=lift(x -> (x, 1.0), corrThreshold),
        markersize=0.7)

    update_cam!(scene)
    center!(scene)

    return h, s, v
end

function fill_sq(mData::Tuple{Vector{Point3f},Vector{TriangleFace{UInt16}}},
    c::Float32)::GeometryBasics.Mesh
    p, f = mData
    return GeometryBasics.mesh(p, f, color=per_face(fill(c, length(f)), f))
end

function superquadrics_view!(scene::Makie.Scene,
    points::Vector{Point3f},
    colors::Observable{<:AbstractArray{Float32}},
    spa::Vector{Vector{Vec3f}},
    vol_cmap::Observable{Vector{RGBAf}},
    invariantRange::Observable{Tuple{Float32,Float32}};
    resolution=0.5,
)
    # try to only render visible points, helps with point picking when hovering 
    sq = lift(scene, colors) do c
        ip = collect(zip(c, eachindex(points)))

        v_lo = getindex.(filter(x -> x[1] < -0.01, ip), 2)
        v_hi = getindex.(filter(x -> x[1] > 0.01, ip), 2)
        v_z = getindex.(filter(x -> x[1] <= 0.01 && x[1] >= -0.01, ip), 2)

        if length(v_lo) > 0
            lo_sq = GeometryBasics.merge(fill_sq.(superquadric.(1.0, view(points, v_lo), view(spa, v_lo), 3.0, resolution), view(c, v_lo)))
        else
            lo_sq = fill_sq((fill(Point3f(0.0, 0.0, 0.0), 3),
                    [TriangleFace((UInt16(1), UInt16(2), UInt16(3)))]), Float32(0.0))
        end

        if length(v_hi) > 0
            hi_sq = GeometryBasics.merge(fill_sq.(superquadric.(1.0, view(points, v_hi), view(spa, v_hi), 3.0, resolution), view(c, v_hi)))
        else
            hi_sq = fill_sq((fill(Point3f(0.0, 0.0, 0.0), 3),
                    [TriangleFace((UInt16(1), UInt16(2), UInt16(3)))]), Float32(0.0))
        end
        return lo_sq, hi_sq, v_z
    end

    mesh!(
        scene,
        lift(x -> x[1], sq),
        colorrange=lift(x -> (x[1], 0.0), invariantRange),
        colormap=lift(x -> x[1:49], vol_cmap),
        inspectable=false,
        transparency=true
    )

    mesh!(
        scene,
        lift(x -> x[2], sq),
        colorrange=lift(x -> (0.0, x[2]), invariantRange),
        colormap=lift((x, y) -> y[1] < 0.0 ? x[50:100] : x, vol_cmap, invariantRange),
        inspectable=false,
        transparency=true
    )

    meshscatter!(scene,
        lift(x -> points[x[3]], sq);
        color=:gray,
        marker=:Sphere,
        transparency=true,
        inspectable=false,
        markersize=0.2)

    update_cam!(scene)

end

function draw_bbox_pixel_space!(scene, lo, hi; kwargs...)
    bbox = Rect2(lo - 0.5, lo - 0.5, (hi - lo) + 1, (hi - lo) + 1)
    p = wireframe!(
        scene,
        bbox;
        visible=true,
        inspectable=false,
        depth_shift=-1.0f-3,
        kwargs...
    )
    return p
end

function clear_layout(layout::GridLayout)
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

function setup_menu(figure::Makie.Figure, options::Vector{String}, s::Observable{String}; kwargs...)
    render_menu = Menu(figure;
        options=options,
        default=s[],
        kwargs...)

    Makie.onany(render_menu.blockscene, render_menu.selection) do sel
        s[] = sel
    end

    return s, render_menu
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

function inline_image(fig, img, text; kwargs...)
    ax = Axis(fig; aspect=AxisAspect(1), kwargs...)
    hidedecorations!(ax)
    hidespines!(ax)
    disable_interactions(ax)

    attach_image(fig, ax, img, text)
    return ax
end

function tooltip_ax(loc, txt; kwargs...)
    ax = Axis(loc; backgroundcolor=:transparent, kwargs...)
    hidedecorations!(ax)
    hidespines!(ax)
    disable_interactions(ax)

    text!(ax, (0, 0); text=txt)
    #ax.scene.visible[] = false

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

function bondview!(scene, pts, seg)

    meshscatter!(scene, pts;
        markersize=0.3,
        alpha=0.4,
        ssao=true,
        transparency=true,
        inspectable=false,
        color=:gray)

    linesegments!(scene,
        seg;
        color=:red,
        inspectable=false,
        linewidth=2.0)

    update_cam!(scene)
end


function _segments(points, bonds)
    topology = findall(!iszero, bonds)

    segpts = Vector{Point3f}(undef, 2length(topology))
    k = 1
    for (i, j) in [Tuple(I) for I in topology]
        segpts[k] = points[i]
        segpts[k+1] = points[j]
        k += 2
    end

    return segpts
end

