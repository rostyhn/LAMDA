using Makie: clear_temporary_plots!, Orthographic

function build_mol_window(fig_size, transition, atomPositions, volumeData, volumeAbsMax, volumeDataDict, superquadrics, lineSets, transitionKDTree, sampleRangeX, sampleRangeY, sampleRangeZ, cmap)

    molWindow = Figure(size=fig_size)

    atomView = LScene(
        molWindow[1:5, 1:3],
        show_axis=false,
        scenekw=(backgroundcolor=:black, clear=true),
    )

    volumeView = LScene(
        molWindow[1:5, 4:6],
        show_axis=false,
        scenekw=(backgroundcolor=:white, clear=true),
    )

    # atom positions should be a tuple of both states involved
    ap1 = atomPositions[1]
    aa1 = map(x -> get(volumeDataDict, x[1], 0.0), enumerate(eachrow(ap1)))

    mm = extrema(aa1)
    filterRange = LinRange(mm[1], mm[2], 100)

    volFilter = IntervalSlider(molWindow[7, 1:3], range=filterRange, startvalues=(0, 0))
    Label(molWindow[6, 1], lift(x -> string(x), volFilter.interval))
    selected = lift(volFilter.interval) do interval
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

    glyps = mesh!(
        atomView,
        lift(x -> superquadrics[x], selected),
        color=lift(x -> aa1[x], selected),
        # prevents it from recoloring each time the slider moves
        colorrange=extrema(aa1),
        colormap=:bam,
        fxaa=false,
    )
    glyps.inspectable[] = false

    linesegments!(atomView,
        lift(x -> lineSets[1][x], selectedLineSets),
        color=lift(x -> lineSets[2][x], selectedLineSets),
        colorrange=extrema(lineSets[2]),
        inspector_label=(self, idx, pos) -> string("Weight ", self.color[][idx]),
        lowclip=:black,
        colormap=:bam)
    # https://github.com/MakieOrg/Makie.jl/blob/master/src/interaction/ray_casting.jl

    inspector = DataInspector(atomView)

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
    end

    vol = volume!(volumeView, sampleRangeX, sampleRangeY, sampleRangeZ,
        volumeData;
        colormap=cmap,
        algorithm=:absorption,
        #isorange = 0.000001,
        #isovalue = 0.0,
        #colorscale = abs,
        #absorption= lift(x->x, transitionGlyphSize),
        fxaa=false,
        transparency=true,
        shading=NoShading,
        colorrange=(-volumeAbsMax, volumeAbsMax),
        visible=true)

    link_cameras_lscene(molWindow)
    Colorbar(molWindow[6, 4:6], vol, vertical=false)

    screen = GLMakie.Screen()
    display(screen, molWindow)
end
