using Makie: clear_temporary_plots!, Orthographic, GridLayout

function build_mol_window(beforeView, afterView, transition, atomPositions, volumeData, volumeAbsMax, volumeDataDict, superquadrics, lineSets, transitionKDTree, sampleRangeX, sampleRangeY, sampleRangeZ, cmap, on_window_hover, lsExtrema, filterVal)

    # idea is that right clicking will show a context menu that allows you to change the content of the scene, bbox, clear!
    ap1, ap2 = atomPositions

    # atom positions should be a tuple of both states involved
    aa1 = map(x -> get(volumeDataDict, x[1], 0.0), enumerate(eachrow(ap1)))
    mm = extrema(aa1)

    # pass down selected from main range filter
    selected = lift(filterVal) do interval
        selected = Vector{Int}()
        for (i, v) in enumerate(aa1)
            # inverse filter, blue area will be removed!
            if v < interval[1] || v > interval[2]
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

    bp = scatter!(beforeView, ap1, color=lift(x -> [i in x ? :red : :blue for i in 1:147], selected))
    afp = scatter!(afterView, ap2, color=lift(x -> [i in x ? :red : :blue for i in 1:147], selected))

    # could pass down functions instead, the state view doesn't need all of this data at all
    setup_state_view!(beforeView, bp, ap1, ap2, aa1, superquadrics, lineSets, sampleRangeX, sampleRangeY, sampleRangeZ, lsExtrema, cmap, volumeData, volumeAbsMax, selected, selectedLineSets)
    setup_state_view!(afterView, afp, ap1, ap2, aa1, superquadrics, lineSets, sampleRangeX, sampleRangeY, sampleRangeZ, lsExtrema, cmap, volumeData, volumeAbsMax, selected, selectedLineSets)

end

# could refactor to have less parameters
function setup_state_view!(rootScene, initial_plot, ap1, ap2, aa1,
    superquadrics, lineSets, sampleRangeX, sampleRangeY, sampleRangeZ,
    lsExtrema, cmap, volumeData, volumeAbsMax, selected, selectedLineSets
)
    rendered_plots = Vector()
    push!(rendered_plots, initial_plot)

    tt = Scene(rootScene.scene)
    campixel!(tt)

    menu_bbox = Observable(BBox(0, 0, 0, 0))
    current_view = Observable("State 1")
    m = Menu(tt, options=["State 1", "State 2", "Superquadric", "Volume Render"], default="State 1", is_open=true, bbox=menu_bbox)

    on(events(rootScene).mousebutton, priority=1) do event
        if event.button == Mouse.right && event.action == Mouse.press && is_mouseinside(rootScene)
            x, y = events(rootScene.parent).mouseposition[]
            menu_bbox[] = BBox(x, x + 100, y - 100, y)
            notify(menu_bbox)
            m.is_open = true
        end
    end

    on(m.selection) do selection
        current_view[] = selection
    end

    on(current_view) do cw
        cam = camera(rootScene)
        eyepos = cam.eyeposition[]
        lookat = cam.lookat[]

        for p in rendered_plots
            delete!(rootScene, p)
        end
        empty!(rendered_plots)

        if cw == "State 1"
            bp = scatter!(rootScene, ap1, color=lift(x -> [i in x ? :red : :blue for i in 1:147], selected))
            push!(rendered_plots, bp)

        elseif cw == "State 2"
            bp = scatter!(rootScene, ap2, color=lift(x -> [i in x ? :red : :blue for i in 1:147], selected))
            push!(rendered_plots, bp)

        elseif cw == "Superquadric"
            bp = mesh!(
                rootScene,
                lift(x -> superquadrics[x], selected),
                color=lift(x -> aa1[x], selected),
                # prevents it from recoloring each time the slider moves
                colorrange=lift(x -> (-x, x), volumeAbsMax),
                colormap=:bam,
                fxaa=false,
            )
            push!(rendered_plots, bp)

            bp.inspectable[] = false

            ls = linesegments!(rootScene,
                lift(x -> lineSets[1][x], selectedLineSets),
                color=lift(x -> lineSets[2][x], selectedLineSets),
                colorrange=lift(x -> x, lsExtrema),
                inspector_label=(self, idx, pos) -> string("Weight ", self.color[][idx]),
                lowclip=:black,
                colormap=:bam)
            push!(rendered_plots, ls)

            #=inspector = DataInspector(atomView)

            on(events(atomView).mouseposition) do mp
                plot, idx = pick(glyps)
                if plot == glyps.plots[1]
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
            end=#
        else
            bp = volume!(rootScene, sampleRangeX, sampleRangeY, sampleRangeZ,
                volumeData;
                colormap=cmap,
                algorithm=:absorption,
                fxaa=false,
                transparency=true,
                shading=NoShading,
                colorrange=lift(x -> (-x, x), volumeAbsMax),
                visible=true)
            push!(rendered_plots, bp)
        end
        update_cam!(rootScene.scene, eyepos, lookat)
    end

    #=
    Colorbar(molWindow[6, 4:6], vol, vertical=false)
    =#
    #on(events(molWindow).entered_window) do is_hovered
    # on_window_hover(transition, is_hovered)
    #end

end
