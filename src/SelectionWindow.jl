using Makie: clear_temporary_plots!, Orthographic, SparseArrays
using StatsBase

function dist_plot!(scene, x_positions, y_positions, currently_selected, t_list, on_click, reference_configuration, x_label, y_label)
    positions = @lift begin
        pos = Vector{Point2f}()
        for t in t_list
            push!(pos, Point2f(get($x_positions, t, 0.0), get($y_positions, t, 0.0)))
        end
        return pos
    end

    colors = Observable(fill(:blue, length(positions[])))

    sc = scatter!(scene, positions, color=colors)
    scene.xlabel = x_label
    scene.ylabel = y_label

    sc.inspectable[] = false
    inspector = DataInspector(scene)

    on(events(scene).mouseposition) do mp
        colors[] = fill(:blue, length(colors[]))
        plot, idx = pick(scene)
        if plot == sc
            currently_selected[] = t_list[idx]
            inspector.plot.text[] = string(t_list[idx])
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

function atom_selection_view!(scene, ref_config, selected_atoms, num_atoms, atomPositions)
    a_data = @lift begin
        return atomPositions[$ref_config][1]
    end

    segment_selector = scatter!(scene, lift(x -> x, a_data),
        color=lift(x -> color_selected(x, num_atoms), selected_atoms),
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


function build_selection_window(fig_size,
    data::Dict{String,Dict{Tuple{Int,Int},Matrix{Float64}}},
    seq,
    on_click,
    num_atoms,
    reference_configuration,
    iv1,
    alignedPositions,
    transitionKDTree, dms)

    window = Figure(size=fig_size)

    selected_data = Observable(first(keys(data)))
    matrix_selection = Menu(window[1, 1], options=collect(keys(data)))

    on(matrix_selection.selection) do val
        selected_data[] = val
    end

    selected_atoms = Observable(Set(1))

    atom_scene = Axis3(window[2, 1], title=lift(x -> string(x), reference_configuration), aspect=:equal)

    matrix_scene = Axis(window[2, 2])
    hidedecorations!(matrix_scene)

    selection_scene = Axis(window[3, :])

    ref_distances = @lift begin
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

        distancesToReference = computeLNCD.(Ref(transitionDistribution), Ref($reference_configuration), keys(transitionInvariants1), Ref($selected_atoms))
        zipped = collect(zip(collect(keys(transitionInvariants1)), distancesToReference))

        ref_distances = Dict()
        for (t, d) in zipped
            ref_distances[t] = d
        end

        return ref_distances
    end

    # for now graph, but should be user-selectable
    m = dms["graph"]["matrix"]
    t_to_idx = dms["graph"]["t_to_idx"]

    graph_dist = @lift begin
        row_idx = dms["graph"]["t_to_idx"][$reference_configuration]
        row = m[row_idx, :]

        graph_dist = Dict()
        for (t, idx) in t_to_idx
            graph_dist[t] = row[idx]
        end
        return graph_dist
    end

    minInvariant1, maxInvariant1, transitionInvariants1 = iv1
    t_list = collect(keys(t_to_idx))

    # data, min, max
    processed_data = lift(x -> load_data(data[x]), selected_data)
    currently_selected = Observable{Any}(Nothing)

    dist_plot!(selection_scene, graph_dist, ref_distances, currently_selected, t_list, on_click, reference_configuration, "Graph Distance", "LNCD Score")

    mat_2d = @lift begin
        mat = zeros(1, 1)
        if $currently_selected != Nothing
            mat = $processed_data[1][$currently_selected]
        end
        return mat
    end

    matrix_plot = heatmap!(matrix_scene, mat_2d, colorrange=lift(x -> (x[2], x[3]), processed_data), colormap=:viridis)
    matrix_plot.inspectable[] = false

    Colorbar(window[1, 2], colormap=:viridis, limits=lift(x -> (x[2], x[3]), processed_data), vertical=false)

    on(mat_2d) do r
        reset_limits!(matrix_scene)
    end

    atom_selection_view!(atom_scene, reference_configuration, selected_atoms, num_atoms, alignedPositions)
    return window
end

function color_selected(selected_atoms, num_atoms)
    colors = [:blue for _ in range(1, num_atoms)]
    for idx in selected_atoms
        colors[idx] = :red
    end
    return colors
end

function load_data(data)
    # order matrices by distance relative to i
    d = collect(values(data))

    # simple min-max norm
    max_val = maximum(map((x) -> maximum(x), d))
    min_val = minimum(map((x) -> minimum(x), d))

    norm = Dict{Tuple{Int,Int},Matrix}()

    for (transition, val) in data
        norm[transition] = (val .- min_val) / (max_val - min_val)
    end

    return norm, min_val, max_val
end


