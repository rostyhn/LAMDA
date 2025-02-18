using Makie: clear_temporary_plots!, Orthographic, GridLayout, clear!

function build_mol_window(transition, atomPositions, volumeData, volumeRange, superquadrics, lineSets, transitionKDTree, sampleRanges, cmap, on_window_hover, lsExtrema, filterVal, fig_size=(400, 400))

    # https://github.com/MakieOrg/Makie.jl/blob/master/src/interaction/ray_casting.jl
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
            colorrange=lsExtrema,
            lowclip=:black,
            colormap=:bwr)
        ls.inspectable[] = false

        m = mesh!(
            scene,
            lift(x -> superquadrics[x], selected),
            color=lift((x, y) -> y[x], selected, aa1),
            # prevents it from recoloring each time the slider moves
            colorrange=volumeRange,
            colormap=:bam,
            fxaa=false,
        )
        m.inspectable[] = false

        # weak = true removes the connection when the return value of on is gc'd
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
        v = volume!(scene,
            lift(x -> x[1], sampleRanges),
            lift(x -> x[2], sampleRanges),
            lift(x -> x[3], sampleRanges),
            volumeData;
            colormap=cmap,
            algorithm=:absorption,
            fxaa=false,
            transparency=true,
            shading=NoShading,
            colorrange=volumeRange,
            visible=true)
        v.inspectable[] = false
        return [v], [], []
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

function setup_state_view!(fig, loc, startState, render_funcs)
    rootScene = LScene(
        fig,
        show_axis=false,
        scenekw=(backgroundcolor=:black, clear=true),
    )

    m = Menu(fig, options=keys(render_funcs),
        default=startState)

    i, j = loc
    fig[i, j] = vgrid!(m, rootScene)

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

    on(m.selection) do cw
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
            filter!(x -> x != s, rootScene.scene.children)
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
end
