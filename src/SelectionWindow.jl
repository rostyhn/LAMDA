using Makie: clear_temporary_plots!, Orthographic, SparseArrays
using StatsBase
using Graphs
using GraphMakie
using NetworkLayout
using UMAP
using Clustering

function dist_plot!(scene, x_positions, y_positions, currently_selected, t_list, on_click, reference_configuration, x_label, y_label)
    positions = @lift begin
        pos = Vector{Point2f}()
        for t in t_list
            # invalid values will be 0
            push!(pos, Point2f(get($x_positions, t, 0.0), get($y_positions, t, 0.0)))
        end
        return pos
    end

    colors = Observable(fill(:blue, length(positions[])))

    sc = scatter!(scene, positions, color=colors, markersize=5)

    scene.xlabel = x_label[]
    scene.ylabel = y_label[]

    on(x_label) do val
        scene.xlabel = val
    end

    on(y_label) do val
        scene.ylabel = val
    end

    sc.inspectable[] = false
    inspector = DataInspector(scene)

    on(events(scene).mouseposition) do mp
        colors[] = fill(:blue, length(colors[]))
        plot, idx = pick(scene)
        if plot == sc
            t = t_list[idx]
            pos = positions[][idx]
            x = round(pos[1], digits=3)
            y = round(pos[2], digits=3)

            currently_selected[] = t
            inspector.plot.text[] = "$t\nX:$x Y:$y"
            inspector.plot.visible[] = true
            inspector.plot.position = mp
            colors[][idx] = :red
        end
        notify(colors)
        return Consume(true)
    end

    on(events(scene).mousebutton) do event
        if event.button == Mouse.left && event.action == Mouse.press
            plot, idx = pick(scene)
            pos = position_on_plot(plot, idx)
            if !isnan(pos) && (plot == sc)
                t = t_list[idx]
                if events(scene).keyboardbutton[] == Makie.KeyEvent(Makie.Keyboard.left_control, Makie.Keyboard.press)
                    reference_configuration[] = t
                else
                    on_click(t, x -> ())
                end
            end
        end
        if event.button == Mouse.right && event.action == Mouse.press
            reset_limits!(scene)
        end

        return Consume(false)
    end

    on(x_positions) do d
        vals = collect(values(d))
        xlims!(minimum(vals) - 0.1, maximum(vals) + 0.1)
    end

    on(y_positions) do d
        vals = collect(values(d))
        ylims!(minimum(vals) - 0.01, maximum(vals) + 0.1)
    end

    return sc
end

function atom_selection_view!(scene, ref_config, selected_atoms, num_atoms, atomPositions, highlighted_atoms)
    a_data = @lift begin
        return atomPositions[$ref_config][1]
    end

    color = @lift begin
        colored = color_selected($selected_atoms, num_atoms)
        colored[$highlighted_atoms] .= :pink
        return colored
    end

    segment_selector = scatter!(scene, lift(x -> x, a_data),
        color=color,
        inspector_label=(self, i, p) -> string("Atom ", i)
    )

    # for now, let's just select individual atoms
    on(events(scene).mousebutton, priority=1) do event
        if event.button == Mouse.left && event.action == Mouse.press
            plot, idx = pick(scene)
            if plot == segment_selector
                if idx in selected_atoms[]
                    delete!(selected_atoms[], idx)
                else
                    push!(selected_atoms[], idx)
                end

                notify(selected_atoms)

                return Consume(true)
            end
        end
        return Consume(false)
    end

    return segment_selector
end

function get_matrix_data(label, dms, reference_configuration, selected_atoms, iv1, alignedPositions, transitionKDTree)
    if label == "LNCD"
        numberOfBins = 100
        minInvariant1, maxInvariant1, transitionInvariants1 = iv1

        @show stepSize = (maxInvariant1 - minInvariant1) / numberOfBins
        binEdges = [minInvariant1:stepSize:maxInvariant1;]

        transitionDistribution = Dict{Tuple{Int,Int},Vector{SparseVector{Float64}}}() # in transition (String), List of control points { sparse neighbourhood distribution }  
        @time for (t, values) in transitionInvariants1

            p1, p2 = get_from_t_dict(alignedPositions, t)
            k1, k2 = get_from_t_dict(transitionKDTree, t)

            transitionDistribution[t] = computeInvariantDistributionInNeighborhood(values, p1, binEdges, 10, k1)
        end
        @show "done with distributions"

        distancesToReference = computeLNCD.(Ref(transitionDistribution),
            Ref(reference_configuration),
            keys(transitionInvariants1),
            Ref(selected_atoms))
        zipped = collect(zip(collect(keys(transitionInvariants1)), distancesToReference))

        ref_distances = Dict()
        for (t, d) in zipped
            ref_distances[t] = d
        end

        return ref_distances
    end

    m = dms[label]["matrix"]
    t_to_idx = dms[label]["t_to_idx"]
    row_idx = t_to_idx[reference_configuration]
    row = m[row_idx, :]

    graph_dist = Dict()
    for (t, idx) in t_to_idx
        graph_dist[t] = row[idx]
    end

    return graph_dist
end

function calc_graph_connectivity(num_vertices, dm, threshold)
    # need to do it this way so that all nodes are rendered at first and indices remain consistent
    ci = Tuple.(findall(x -> x < threshold && x != 0, dm))
    g = SimpleGraph(num_vertices)
    for idx in ci
        s1, s2 = idx
        add_edge!(g, s1, s2)
    end

    return g
end

function build_selection_window(fig_size,
    data,
    t_list,
    on_click,
    num_atoms,
    reference_configuration,
    iv1,
    alignedPositions,
    transitionKDTree, dms, volData, sampleRanges, volumeAbsMax, cmap)

    window = Figure(size=fig_size)

    user_groups = Observable(Dict())
    selected_dm = Observable(first(keys(dms)))
    dm_menu = Menu(window[1, :], options=collect(keys(dms)))
    on(dm_menu.selection) do val
        selected_dm[] = val
    end

    t_to_idx = Dict()
    for (i, t) in enumerate(t_list)
        t_to_idx[t] = i
    end

    reordered_matrix = @lift begin
        m = dms[$selected_dm]
        res = hclust(m, linkage=:ward, branchorder=:barjoseph)
        rm = zeros(size(m))

        # gets the correct 
        mtx_to_t = Dict()
        for (i, r) in enumerate(res.order)
            rm[i, :] = map(x -> m[r, :][x], res.order)
            mtx_to_t[i] = t_list[r]
        end

        return rm, mtx_to_t
    end

    grid = GridLayout()
    window[2, 1] = grid
    hl = Observable(first(t_list))
    hr = Observable(last(t_list))
    svl = LScene(grid[1, 1], show_axis=false, scenekw=(backgroundcolor=:black, clear=true))
    svr = LScene(grid[1, 2], show_axis=false, scenekw=(backgroundcolor=:black, clear=true))

    Label(grid[2, 1], lift(x -> string(x), hl), tellwidth=false)
    Label(grid[2, 2], lift(x -> string(x), hr), tellwidth=false)

    vl = volume!(svl,
        lift(x -> x[1], sampleRanges),
        lift(x -> x[2], sampleRanges),
        lift(x -> x[3], sampleRanges),
        lift((x, y) -> y[x], hl, volData);
        colormap=cmap,
        algorithm=:absorption,
        fxaa=false,
        transparency=true,
        shading=NoShading,
        colorrange=lift(x -> (-x, x), volumeAbsMax),
        overdraw=true,
        visible=true)
    vl.inspectable[] = false

    vr = volume!(svr,
        lift(x -> x[1], sampleRanges),
        lift(x -> x[2], sampleRanges),
        lift(x -> x[3], sampleRanges),
        lift((x, y) -> y[x], hr, volData);
        colormap=cmap,
        algorithm=:absorption,
        fxaa=false,
        transparency=true,
        shading=NoShading,
        colorrange=lift(x -> (-x, x), volumeAbsMax),
        overdraw=true,
        visible=true)
    vr.inspectable[] = false

    graph_ax = Axis(window[3, 1], backgroundcolor=:transparent)
    hm_ax, hm = heatmap(window[2:3, 2], lift(x -> x[1], reordered_matrix))
    deregister_interaction!(hm_ax, :rectanglezoom)
    #deregister_interaction!(hm_ax, :dragpan)
    #deregister_interaction!(hm_ax, :scrollzoom)

    deregister_interaction!(graph_ax, :rectanglezoom)

    embedding = @lift begin
        em = transpose(umap(dms[$selected_dm], 2; metric=:precomputed))
        embedding = map(x -> Point2f(x), eachrow(em))
    end

    hovered = Observable(first(t_list))
    tt_bbox = Observable(BBox(0, 0, 0, 0))

    colors = Observable(fill(:blue, length(embedding[])))

    sc = scatter!(graph_ax, embedding; color=colors)
    text!(graph_ax, embedding; text=map(x -> string(x), t_list))
    hidedecorations!(graph_ax)

    campixel!(graph_ax.scene)
    ax3d = LScene(graph_ax.scene, show_axis=false, bbox=tt_bbox, scenekw=(backgroundcolor=:black, clear=true, size=(250, 250), zorder=100), height=250, width=250)
    ax3d.scene.visible[] = false

    v = volume!(ax3d,
        lift(x -> x[1], sampleRanges),
        lift(x -> x[2], sampleRanges),
        lift(x -> x[3], sampleRanges),
        lift((x, y) -> y[x], hovered, volData);
        colormap=cmap,
        algorithm=:absorption,
        fxaa=false,
        transparency=true,
        shading=NoShading,
        colorrange=lift(x -> (-x, x), volumeAbsMax),
        overdraw=true,
        visible=true)
    v.inspectable[] = false
    translate!(ax3d.scene, 0, 0, 10)

    center!(ax3d.scene)

    on(events(graph_ax).mouseposition) do mp
        plot, idx = pick(graph_ax)
        if plot == sc
            t = t_list[idx]
            x, y = events(graph_ax.parent).mouseposition[]
            tt_bbox[] = BBox(x + 15, x + 265, y - 265, y - 15)
            hovered[] = t_list[idx]
            ax3d.scene.visible[] = true
            notify(hovered)
            notify(tt_bbox)
            center!(ax3d.scene)
        else
            ax3d.scene.visible[] = false
        end

        return Consume(false)
    end

    on(events(graph_ax).mousebutton) do event
        if event.button == Mouse.left && event.action == Mouse.press
            plot, idx = pick(graph_ax)
            pos = position_on_plot(plot, idx)
            if !isnan(pos) && (plot == sc)
                on_click(t_list[idx], x -> ())
            end
        end
        return Consume(true)
    end

    on(events(hm_ax).mouseposition) do mp
        colors[] = fill(:blue, length(colors[]))
        plot, _ = pick(hm_ax)
        if plot == hm
            xy = mouseposition(hm_ax)
            i, j = Int.(round.(xy))
            # convert to old idx - could also just use reordered matrix as input to umap
            oldi = t_to_idx[reordered_matrix[][2][i]]
            oldj = t_to_idx[reordered_matrix[][2][j]]
            colors[][oldi] = :red
            colors[][oldj] = :red
            hl[] = t_list[oldi]
            hr[] = t_list[oldj]
            notify(hl)
            notify(hr)
            notify(colors)
        end
        #t = t_list[idx]
        #hovered[] = t_list[idx]
        #ax3d.scene.visible[] = true
        #notify(hovered)
        #notify(tt_bbox)
        #center!(ax3d.scene)
        return Consume(false)
    end

    return window
end

function color_selected(selected_atoms, num_atoms)
    colors = [:blue for _ in range(1, num_atoms)]
    for idx in selected_atoms
        colors[idx] = :red
    end
    return colors
end
