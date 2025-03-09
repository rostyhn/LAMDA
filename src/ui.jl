# switches what is being rendered inside a scene cleanly.
# pass a select_fn with selector as a parameter and then basically do whatever you want
# can modify the grid the scene belongs to and it will get cleared up here

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
        inspector_label=(self, i, p) -> "Atom $(i); weight: $(self.color[][i])",
        markersize=30)

    return s
end

function volume_view!(scene, vd, sampleRanges, vol_cmap, volumeRange; rotation=Observable(Matrix{Float32}(1.0I, 3, 3)), update=false)
    v = volume!(scene,
        lift(x -> extrema(x[1]), sampleRanges),
        lift(x -> extrema(x[2]), sampleRanges),
        lift(x -> extrema(x[3]), sampleRanges),
        vd;
        colormap=vol_cmap,
        algorithm=:absorption,
        fxaa=false,
        transparency=true,
        shading=NoShading,
        colorrange=volumeRange)

    # FIXME sometimes the volume will get rotated so hard it disappears
    # if called before screen is rendered it crashes
    #=on(rotation, update=update) do R
        rr = hcat(R, [0, 0, 0])
        fr = transpose(vcat(rr, transpose([0; 0; 0; 1])))
        v.model[] = fr
        notify(v.model)
    end=#

    v.inspectable[] = false

    return v
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
