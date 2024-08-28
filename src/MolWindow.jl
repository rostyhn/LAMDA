using Makie: clear_temporary_plots!, Orthographic, GridLayout, clear!

function build_mol_window(beforeView, afterView, transition, atomPositions, volumeDat, volumeAbsMax, superquadrics, lineSets, transitionKDTree, sampleRanges, cmap, on_window_hover, lsExtrema, filterVal)

    ap1, ap2 = atomPositions

    # atom positions should be a tuple of both states involved
    aa1 = lift(y -> map(x -> get(y[2], x[1], 0.0), enumerate(eachrow(ap1))), volumeDat)

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

    bp = function (scene)
        s = scatter!(scene, ap1, color=lift(x -> [i in x ? :red : :blue for i in 1:147], selected))
        s.inspectable[] = false
        return s
    end

    afp = function (scene)
        s = scatter!(scene, ap2, color=lift(x -> [i in x ? :red : :blue for i in 1:147], selected))
        s.inspectable[] = false
        return s
    end

    # could pass down functions instead, the state view doesn't need all of this data at all
    # Dict of strings to functions

    cl = setup_state_view!(beforeView, bp, ap1, ap2, aa1, superquadrics, lineSets, sampleRanges, lsExtrema, cmap, volumeDat, volumeAbsMax, selected, selectedLineSets, transitionKDTree, "State 1")

    cr = setup_state_view!(afterView, afp, ap1, ap2, aa1, superquadrics, lineSets, sampleRanges, lsExtrema, cmap, volumeDat, volumeAbsMax, selected, selectedLineSets, transitionKDTree, "State 2")

    cleanup = function ()
        cl()
        cr()
    end

    return cleanup
end

# could refactor to have less parameters
function setup_state_view!(rootScene, initial_render_func, ap1, ap2, aa1,
    superquadrics, lineSets, sampleRanges,
    lsExtrema, cmap, volumeData, volumeAbsMax, selected, selectedLineSets, transitionKDTree,
    startState
)

    ip = initial_render_func(rootScene)
    rendered_plots = Vector()
    push!(rendered_plots, ip)

    scene_listeners = Vector()

    tt = Scene(rootScene.scene)
    campixel!(tt)

    menu_bbox = Observable(BBox(0, 0, 0, 0))
    m = Menu(tt, options=["State 1", "State 2", "Superquadric", "Volume Render"], default=startState, is_open=true, bbox=menu_bbox)

    # this needs to be deleted as well
    contextMenuListener = on(events(rootScene).mousebutton, priority=1) do event
        if event.button == Mouse.right && event.action == Mouse.press && is_mouseinside(rootScene)
            x, y = events(rootScene.parent).mouseposition[]
            menu_bbox[] = BBox(x, x + 100, y - 100, y)
            notify(menu_bbox)
            m.is_open = true

            # block other events
            return Consume(true)
        end
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

        if cw == "State 1"
            bp = scatter!(rootScene, ap1, color=lift(x -> [i in x ? :red : :blue for i in 1:147], selected))
            bp.inspectable[] = false

            push!(rendered_plots, bp)
        elseif cw == "State 2"
            bp = scatter!(rootScene, ap2, color=lift(x -> [i in x ? :red : :blue for i in 1:147], selected))
            bp.inspectable[] = false

            push!(rendered_plots, bp)
        elseif cw == "Superquadric"
            ls = linesegments!(rootScene,
                lift(x -> lineSets[1][x], selectedLineSets),
                color=lift(x -> lineSets[2][x], selectedLineSets),
                colorrange=lift(x -> x, lsExtrema),
                inspector_label=(self, idx, pos) -> string("Weight ", self.color[][idx]),
                lowclip=:black,
                colormap=:bam)
            push!(rendered_plots, ls)

            bp = mesh!(
                rootScene,
                lift(x -> superquadrics[x], selected),
                color=lift((x, y) -> y[x], selected, aa1),
                # prevents it from recoloring each time the slider moves
                colorrange=lift(x -> (-x, x), volumeAbsMax),
                colormap=:bam,
                fxaa=false,
            )
            push!(rendered_plots, bp)
            bp.inspectable[] = false

            inspector = DataInspector(rootScene)

            sqHoverListener = on(events(rootScene).mouseposition) do mp
                plot, idx = pick(rootScene)
                if plot != Nothing && is_mouseinside(rootScene)
                    pos = position_on_plot(plot, idx)
                    idx, d = NearestNeighbors.nn(transitionKDTree, pos)
                    if !isnan(pos)
                        inspector.plot.text[] = string("Atom ", idx)
                        inspector.plot.visible[] = true
                        inspector.plot.position = mp
                        return Consume(true)
                    end
                end
                return Consume(false)
            end

            # push!(inspectors, inspector)
            push!(scene_listeners, sqHoverListener)
        else
            bp = volume!(rootScene,
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

            bp.inspectable[] = false
            push!(rendered_plots, bp)
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
