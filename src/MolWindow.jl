using Makie: clear_temporary_plots!, Orthographic, GridLayout, clear!, GridLayoutBase

function build_mol_window(transition, atomPositions, volumeData, volumeRange, superquadrics, lineSets, transitionKDTree, sampleRanges, vol_cmap, on_window_hover, lsExtrema, filterVal, ls_cmap, scalars, fig_size=(400, 400))

    # https://github.com/MakieOrg/Makie.jl/blob/master/src/interaction/ray_casting.jl, delete_from_parent!, delete_from_parent!, GridLayoutBase
    ap1, ap2 = atomPositions

    # atom positions should be a tuple of both states involved
    aa1 = lift(y -> map(x -> y[x[1]], enumerate(eachrow(ap1))), volumeData)

    # pass down selected from main range filter
    selected = @lift begin
        selected = Vector{Int}()
        for (i, v) in enumerate($aa1)
            # inverse filter, blue area will be removed!
            if v < $filterVal[1] || v > $filterVal[2]
                push!(selected, i)
            end
        end
        return selected
    end

    # need to filter out linesets
    selectedLineSets = lift(selected) do kept
        selectedLineSets = Vector{Int}()
        for (i, e) in enumerate(lineSets[3])
            if e[1] in kept && e[2] in kept
                push!(selectedLineSets, i)
            end
        end
        return selectedLineSets
    end

    # these functions must return:
    # a list of plots, listeners, and ui elements (gridLayout, [elements]) they created
    bp = function (scene, inspector, g)
        return setup_atom_view!(scene, g, ap1, transition, 1, selected, scalars)
    end

    afp = function (scene, inspector, g)
        return setup_atom_view!(scene, g, ap2, transition, 2, selected, scalars)
    end

    sq = function (scene, inspector, g)
        ls = linesegments!(scene,
            lift(x -> lineSets[1][x], selectedLineSets),
            color=lift(x -> lineSets[2][x], selectedLineSets),
            colorrange=lsExtrema,
            colormap=ls_cmap)
        ls.inspectable[] = false

        m = mesh!(
            scene,
            lift(x -> superquadrics[x], selected),
            color=lift((x, y) -> y[x], selected, aa1),
            colorrange=volumeRange,
            colormap=vol_cmap,
            fxaa=false,
        )
        m.inspectable[] = false

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
        end
        return [sqHoverListener], []
    end

    vol = function (scene, inspector, g)
        v = volume!(scene,
            lift(x -> x[1], sampleRanges),
            lift(x -> x[2], sampleRanges),
            lift(x -> x[3], sampleRanges),
            volumeData;
            colormap=vol_cmap,
            algorithm=:absorption,
            fxaa=false,
            transparency=true,
            shading=NoShading,
            colorrange=volumeRange,
            visible=true)
        v.inspectable[] = false

        return [], []
    end

    render_funcs = Dict()
    render_funcs["State $(transition[1])"] = bp
    render_funcs["State $(transition[2])"] = afp
    render_funcs["Superquadrics"] = sq
    render_funcs["Volume"] = vol

    molWindow = Figure(size=fig_size)
    molWindow[1, 1:2] = hgrid!(Label(molWindow, string(transition), tellwidth=false))
    setup_state_view!(molWindow, (2, 1), "Volume", render_funcs)
    setup_state_view!(molWindow, (2, 2), "Volume", render_funcs)

    screen = GLMakie.Screen(title="TransVis - $transition")
    display(screen, molWindow)

    #on(events(molWindow).entered_window) do is_hovered
    # on_window_hover(transition, is_hovered)
    #end
end

function setup_atom_view!(scene, g, ap, t, order, selected, scalars)
    opts = ["selected"; sort(collect(keys(scalars)))]

    gg = GridLayout(g[3, :])
    m = Menu(gg[1, 1], options=opts, default="selected")

    colorInfo = @lift begin # might be leaking memory
        opt = $(m.selection)
        vals = [i in $selected ? 0.0 : 1.0 for i in 1:147]
        extremaVals = (0.0, 1.0)
        labelfn = (self, i, p) -> "Atom $(i)"
        cmap = to_colormap(:redsblues)

        if opt != "selected"
            vals = scalars[opt][t][order]
            extremaVals = extrema(vals)
            labelfn = (self, i, p) -> "Atom $(i); weight: $(self.color[][i])"

            # three cases:
            # sequential ascending, descending and diverging
            # divergent if we are approximately around 0 when subtracting the absolute values of the min and max
            minVal, maxVal = extremaVals
            if isapprox(abs(maxVal) - abs(minVal), 0; atol=1)
                println("using divergent colorscheme")
                cmap = resample_cmap(:bam, 100; alpha=([(-0.99):0.02:(0.99);] ./ 0.1) .^ 6)
            else
                # ascending sequential if min is closer to 0
                cmap = resample_cmap(:reds, 147, alpha=range(; start=0.01, stop=1.0, length=147)) #seq ascending
                if abs(minVal) > abs(maxVal)
                    println("using descending sequential colorscheme")
                    # otherwise reverse it
                    reverse!(cmap)
                end
            end
        end
        empty!(scene)

        # complicated because scatter! sets the shader for the scene the first time, changes it when type of color changes
        scatter!(scene, ap, color=vals, colorrange=extremaVals, colormap=cmap,
            inspector_label=labelfn, markersize=30)
        return extremaVals, cmap
    end
    cbar = Colorbar(gg[1, 2], colorrange=lift(x -> x[1], colorInfo), vertical=false, colormap=lift(x -> x[2], colorInfo), tellwidth=false)

    return [], [(gg, [m, cbar])]
end

function setup_state_view!(fig, loc, startState, render_funcs)
    rootScene = LScene(
        fig,
        show_axis=false,
        scenekw=(backgroundcolor=:black, clear=true),
    )

    m = Menu(fig, options=keys(render_funcs),
        default=startState)

    i, j = loc
    g = vgrid!(m, rootScene)
    fig[i, j] = g
    # g is a reference to the underlying gridlayout for the state, can mutate it to create UI elements

    inspector = DataInspector(rootScene)

    initial_render_func = render_funcs[startState]
    il, is = initial_render_func(rootScene, inspector, g)

    # need to do this because otherwise julia assumes the type of the output vector
    # then tries to convert plots to different types
    scene_listeners = Vector{Any}()
    ui_elements = Vector{Any}()

    for l in il
        push!(scene_listeners, l)
    end

    for s in is
        push!(ui_elements, s)
    end

    on(m.selection) do cw
        cam = camera(rootScene)
        eyepos = cam.eyeposition[]
        lookat = cam.lookat[]

        empty!(rootScene)

        for listener in scene_listeners
            off(listener)
            listener = Nothing
        end
        empty!(scene_listeners)

        # clear UI elements
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
        Makie.trim!(g)

        rf = render_funcs[cw]
        listeners, scenes = rf(rootScene, inspector, g)

        for l in listeners
            push!(scene_listeners, l)
        end

        for s in scenes
            push!(ui_elements, s)
        end
        update_cam!(rootScene.scene, eyepos, lookat)
    end
end
