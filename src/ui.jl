# switches what is being rendered inside a scene cleanly.
# pass a select_fn with selector as a parameter and then basically do whatever you want
# can modify the grid the scene belongs to and it will get cleared up here
using Makie

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

function simple_atom_view!(scene, ap, scalars, scalar_range, cmap, time)
    int_pos = lift((x, y) -> x[1] + ((x[2] - x[1]) .* y), ap, time)

    s = scatter!(scene,
        lift(x -> x[:, 1], int_pos),
        lift(x -> x[:, 2], int_pos),
        lift(x -> x[:, 3], int_pos);
        color=lift(x -> x, scalars),
        colorrange=scalar_range,
        colormap=cmap,
        depthsorting=true,
        inspector_label=(self, i, p) -> "Atom $(i); weight: $(self.color[][i])",
        markersize=30)

    update_cam!(parent_scene(s))

    return s
end

function volume_view!(scene, vd, sampleRanges, vol_cmap, volumeRange, rotation; update=false)

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

    v_hi.inspectable[] = false
    v_lo.inspectable[] = false

    update_cam!(parent_scene(v_lo))

    return v_lo, v_hi
end


function superquadrics_view!(scene, points, stretchedPrincipalAxes, volumeData, vol_cmap, invariantRange)
    aa1 = lift((xx, y) -> map(x -> y[x], eachindex(xx)), points, volumeData)
    sq = Observable(superquadric.(1.0, points[], stretchedPrincipalAxes[], 3.0, 0.1)[:])

    calc_sq = on(stretchedPrincipalAxes; update=true) do spa
        sq[] = superquadric.(1.0, points[], spa, 3.0, 0.1)[:]
        notify(sq)
    end

    m_lo = mesh!(
        scene,
        sq,
        color=aa1,
        lowclip=:transparent,
        highclip=:transparent,
        transparency=true,
        colorrange=lift(x -> (x[1], 0.0), invariantRange),
        colormap=lift(x -> x[1:49], vol_cmap),
        fxaa=false,
    )
    m_lo.inspectable[] = false

    m_hi = mesh!(
        scene,
        sq,
        color=aa1,
        lowclip=:transparent,
        highclip=:transparent,
        transparency=true,
        colorrange=lift(x -> (0.0, x[2]), invariantRange),
        colormap=lift(x -> x[50:100], vol_cmap),
        fxaa=false,
    )
    m_hi.inspectable[] = false

    #=
    sqHoverListener = on(events(scene).mouseposition) do mp
        if is_mouseinside(scene)
            plot, idx = pick(scene)
            if plot == ls
                inspector.plot.text[] = string("Weight ", ls.color[][idx])
                inspector.plot.visible[] = true
                inspector.plot.position = mp
                return Consume(true)
            elseif plot != Nothing
                pos = position_on_plot(plot, idx)
                idx, d = NearestNeighbors.nn(transitionKDTree, pos)
                if !isnan(pos)
                    inspector.plot.text[] = string("Atom ", idx)
                    inspector.plot.visible[] = true
                    inspector.plot.position = mp
                    return Consume(true)
                end
            else
                return Consume(true)
            end
        end
        return Consume(false)
    end=#

    update_cam!(parent_scene(m_lo))
    return [calc_sq], []
end



function draw_bbox_pixel_space!(scene, lo, hi; color=:red)
    bbox = Rect2(lo - 0.5, lo - 0.5, (hi - lo) + 1, (hi - lo) + 1)

    p = wireframe!(
        scene, bbox, color=color,
        visible=true, inspectable=false,
        depth_shift=-1.0f-3
    )
    return p
end
