using Makie: clear_temporary_plots!, Orthographic, GridLayout, clear!

function build_mol_window(beforeView, afterView, transition, atomPositions, volumeData, volumeAbsMax, superquadrics, lineSets, transitionKDTree, sampleRanges, cmap, on_window_hover, lsExtrema, filterVal, matrices)

    ap1, ap2 = atomPositions

    # atom positions should be a tuple of both states involved
    aa1 = lift(y -> map(x -> get(y[2], x[1], 0.0), enumerate(eachrow(ap1))), volumeData)

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
    # a list of plots, listeners, and scenes they created
    bp = function (scene, inspector)
        return [scatter!(scene, ap1, color=lift(x -> [i in x ? :red : :blue for i in 1:147], selected),
            inspector_label=(self, i, p) -> string("Atom ", i))], [], []
    end

    afp = function (scene, inspector)
        return [scatter!(scene, ap2,
            color=lift(x -> [i in x ? :red : :blue for i in 1:147], selected),
            inspector_label=(self, i, p) -> string("Atom ", i))], [], []
    end

    sq = function (scene, inspector)
        ls = linesegments!(scene,
            lift(x -> lineSets[1][x], selectedLineSets),
            color=lift(x -> lineSets[2][x], selectedLineSets),
            colorrange=lift(x -> x, lsExtrema),
            lowclip=:black,
            colormap=:bwr)
        ls.inspectable[] = false

        m = mesh!(
            scene,
            lift(x -> superquadrics[x], selected),
            color=lift((x, y) -> y[x], selected, aa1),
            # prevents it from recoloring each time the slider moves
            colorrange=lift(x -> (-x, x), volumeAbsMax),
            colormap=:bam,
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
        return [ls, m], [sqHoverListener], []
    end

    vol = function (scene, inspector)
        cam3d!(scene)
        v = volume!(scene,
            lift(x -> x[1], sampleRanges),
            lift(x -> x[2], sampleRanges),
            lift(x -> x[3], sampleRanges),
            lift(x -> x[1], volumeData);
            colormap=cmap,
            algorithm=:absorption,
            fxaa=false,
            transparency=true,
            shading=NoShading,
            colorrange=lift(x -> (-x, x), volumeAbsMax),
            visible=true)
        v.inspectable[] = false
        return [v], [], []
    end

    render_funcs = Dict()
    render_funcs["State 1"] = bp
    render_funcs["State 2"] = afp
    render_funcs["Superquadrics"] = sq
    render_funcs["Volume"] = vol


    for (k, v) in matrices
        mat_func = function (scene, inspector)
            # scene.scene because technically the scene being passed in is an lScene

            # bit of a hack, place a 2D scene on top of the 3D scene
            # get its actual pixel coords, then drop an axis ontop of that position
            # this way, the 3D camera doesn't get messed up
            scene_bbox = lift(pixelarea(scene.scene)) do r
                x, y = origin(r)
                w, h = widths(r)
                return BBox(x, x + w, y, y + h)
            end

            overlay = Scene(scene.scene)
            campixel!(overlay)

            # this would work if we properly destroyed the scene
            DataInspector(overlay)

            ax = Axis(overlay, bbox=scene_bbox)
            h = heatmap!(ax, v[1], colorrange=(v[2], v[3]), colormap=:viridis)

            return [], [], [overlay]
        end
        render_funcs[k] = mat_func
    end

    cl = setup_state_view!(beforeView, "State 1", render_funcs)
    cr = setup_state_view!(afterView, "State 2", render_funcs)

    cleanup = function ()
        cl()
        cr()
    end

    return cleanup
end

function setup_state_view!(rootScene, startState, render_funcs)
    inspector = DataInspector(rootScene)

    initial_render_func = render_funcs[startState]
    ip, il, is = initial_render_func(rootScene, inspector)

    # need to do this because otherwise julia assumes the type of the output vector
    # then tries to convert plots to different types
    rendered_plots = Vector{Any}()
    scene_listeners = Vector{Any}()
    overlays = Vector{Any}()

    for p in ip
        push!(rendered_plots, p)
    end

    for l in il
        push!(scene_listeners, l)
    end

    for s in is
        push!(overlays, s)
    end

    tt = Scene(rootScene.scene)
    campixel!(tt)

    menu_bbox = Observable(BBox(0, 0, 0, 0))
    m = Menu(tt, options=keys(render_funcs),
        default=startState, is_open=true, bbox=menu_bbox)

    contextMenuListener = on(events(rootScene).mousebutton, priority=1) do event
        if event.button == Mouse.right && event.action == Mouse.press && is_mouseinside(rootScene)
            x, y = events(rootScene.parent).mouseposition[]
            menu_bbox[] = BBox(x, x + 150, y - 100, y)
            notify(menu_bbox)
            m.is_open = true

            # block other events
            return Consume(true)
        end
        return Consume(false)
    end

    cwListener = on(m.selection) do cw
        cam = camera(rootScene)
        eyepos = cam.eyeposition[]
        lookat = cam.lookat[]

        for p in rendered_plots
            delete!(rootScene, p)
        end
        empty!(rendered_plots)

        for listener in scene_listeners
            off(listener)
            listener = Nothing
        end
        empty!(scene_listeners)

        # works, but is probably causing a memory leak -
        # the scene is still in memory
        for s in overlays
            empty!(s)
        end
        empty!(overlays)

        rf = render_funcs[cw]
        plots, listeners, scenes = rf(rootScene, inspector)

        for p in plots
            push!(rendered_plots, p)
        end

        for l in listeners
            push!(scene_listeners, l)
        end

        for s in scenes
            push!(overlays, s)
        end
        update_cam!(rootScene.scene, eyepos, lookat)
    end

    #on(events(molWindow).entered_window) do is_hovered
    # on_window_hover(transition, is_hovered)
    #end
    cleanup = function ()
        off(contextMenuListener)
        contextMenuListener = Nothing

        off(cwListener)
        cwListener = Nothing

        for listener in scene_listeners
            off(listener)
            listener = Nothing
        end
        # inspectors should get cleared off here
        empty!(rootScene)
    end


    return cleanup
end
