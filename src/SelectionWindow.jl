using Makie: clear_temporary_plots!, Orthographic, SparseArrays
using StatsBase
using Graphs
using GraphMakie
using NetworkLayout

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

function calc_graph_connectivity(dm, threshold)
    ci = Graphs.SimpleEdge.(Tuple.(findall(x-> x < threshold && x != 0, dm)))
    g = SimpleGraphFromIterator(ci)
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
    transitionKDTree, dms)

    window = Figure(size=fig_size)

    #options = push!(collect(keys(dms)), "LNCD")
    selected_dm = Observable(first(keys(dms)))
    dm_menu = Menu(window[1, 1], options=collect(keys(dms)))
    on(dm_menu.selection) do val
        selected_dm[] = val
    end

    sg = SliderGrid(window[1,2],
                    (label = "Distance threshold", range = 0.01:0.01:1, startvalue=0.05))
    threshold = sg.sliders[1].value
    
    selection_scene = Axis(window[2, :])
    deregister_interaction!(selection_scene, :rectanglezoom)
    
    dist_graph = @lift begin
        m = dms[$selected_dm]["matrix"]
        return calc_graph_connectivity(m, $threshold)
    end

    node_labels = @lift begin
        # assume that vertex 1 in the graph corresponds to transition 1 in t_list 
        t_to_idx = dms[$selected_dm]["t_to_idx"]
        n = vertices($dist_graph)
        return map(x -> string(t_list[x]), n)
    end

    edge_weights = @lift begin
        m = dms[$selected_dm]["matrix"]
        # inverse map val to 1 -> 0, making closer distances more apparent
        return map(x -> (:gray, 1 - (m[src(x),dst(x)] / $threshold)), collect(edges($dist_graph)))
    end

    p = graphplot!(selection_scene, dist_graph, nlabels=node_labels, edge_color=edge_weights, edge_width=1)
    hidedecorations!(selection_scene)

    function onNodeClick(idx, e, ax)
        @show idx, t_list[idx]
        on_click(t_list[idx], x -> ())
    end
    
    register_interaction!(selection_scene, :nodeclick, NodeClickHandler(onNodeClick))

    return window
end

function color_selected(selected_atoms, num_atoms)
    colors = [:blue for _ in range(1, num_atoms)]
    for idx in selected_atoms
        colors[idx] = :red
    end
    return colors
end
