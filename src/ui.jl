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
        # TODO: make recursive, will allow grids inside grids
        for c in ui_elements
            gg, elements = c
            for e in elements
                empty!(e.blockscene)
                delete!(e)
            end
            if !isnothing(gg)
                Makie.trim!(gg)
                # only way to delete a gridlayout
                GridLayoutBase.remove_from_gridlayout!(gg.layoutobservables.gridcontent[])
            end
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

function simple_atom_view!(scene, g, ap, t, scalars, sel, scalar_range; init_time=0.0, show_menu=true)
    gg = GridLayout(g[end+1, :])
    ui_elements = []

    #TODO: add labels to t_slider
    time = Observable(init_time)
    t_slider = Slider(gg[1, 1:2], range=0.0:0.05:1.0, startvalue=init_time)
    on(t_slider.value) do x
        time[] = x
    end
    push!(ui_elements, t_slider)

    if show_menu
        opts = sort(collect(keys(scalars)))
        m = Menu(gg[2, 1], options=opts, default=sel[])
        on(m.selection) do ms
            sel[] = ms
            notify(sel)
        end
        push!(ui_elements, m)
    end

    cmap = resample_cmap(:reds, 147, alpha=range(; start=0.01, stop=1.0, length=147))

    int_pos = lift((x, y) -> x[1] + ((x[2] - x[1]) .* y), ap, time)

    scatter!(scene,
        lift(x -> x[:, 1], int_pos),
        lift(x -> x[:, 2], int_pos),
        lift(x -> x[:, 3], int_pos);
        color=lift((x, y) -> scalars[y][x], t, sel),
        colorrange=scalar_range,
        colormap=resample_cmap(:reds, 147, alpha=range(; start=0.01, stop=1.0, length=147)),
        inspector_label=(self, i, p) -> "Atom $(i); weight: $(self.color[][i])",
        markersize=30)

    if show_menu
        cbar = Colorbar(gg[2, 2], colorrange=scalar_range, vertical=false, colormap=cmap, tellwidth=false)
        push!(ui_elements, cbar)
    end

    return [], [(gg, ui_elements)]
end

