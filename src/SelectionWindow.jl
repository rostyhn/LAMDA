using Makie: clear_temporary_plots!, Orthographic, SparseArrays

function build_selection_window(fig_size,
    data::Dict{String,Dict{Tuple{Int,Int},Matrix{Float64}}},
    seq,
    on_click,
    reference_configuration,
    iv1,
    atomPositions,
    stateKDTree)

    window = Figure(size=fig_size)

    selected_data = Observable(first(keys(data)))
    num_transitions = length(seq)

    matrix_selection = Menu(window[1, 1], options=collect(keys(data)))

    on(matrix_selection.selection) do val
        selected_data[] = val
    end

    selected_atoms = Observable(Set(1))

    atom_view = LScene(window[2, 1], show_axis=false,
        scenekw=scenekw = (backgroundcolor=:white, clear=true))

    scene_2d = Axis(window[2, 2])
    hidedecorations!(scene_2d)

    selection_scene = Axis(window[3, :])

    # space between matrices
    transition_spacing = 18.0
    # space btwn matrix elements
    matrix_spacing = 5.0

    order = @lift begin
        numberOfBins = 100
        minInvariant1, maxInvariant1, transitionInvariants1 = iv1

        @show stepSize = (maxInvariant1 - minInvariant1) / numberOfBins
        binEdges = [minInvariant1:stepSize:maxInvariant1;]

        transitionDistribution = Dict{Tuple{Int,Int},Vector{SparseVector{Float64}}}() # in transition (String), List of control points { sparse neighbourhood distribution }  
        @time for (t, values) in transitionInvariants1
            s1, _ = t
            transitionDistribution[t] = computeInvariantDistributionInNeighborhood(values, atomPositions[s1], binEdges, 10, stateKDTree[s1])
        end
        @show "done with distributions"

        distancesToReference = computeLNCD.(Ref(transitionDistribution), Ref(reference_configuration[4]), keys(transitionInvariants1), Ref($selected_atoms))

        zipped = collect(zip(collect(keys(transitionInvariants1)), distancesToReference))
        sort!(zipped, by=x -> x[end])
        return map(x -> x[1], zipped), distancesToReference
    end

    on(order) do r
        vals = r[2]
        xlims!(selection_scene, minimum(vals), maximum(vals))
        reset_limits!(selection_scene)
    end

    hist!(selection_scene, lift(x -> x[2], order), bins=100)

    on(events(selection_scene).mousebutton) do event
        if event.button == Mouse.left
            if event.action == Mouse.press
                plot, idx = pick(selection_scene)
                pos = position_on_plot(plot, idx)
                # need to retrieve which elements got placed in this bin
                if !isnan(pos)

                    #d_idx = Int(round(pos[1] / (transition_spacing)))

                    # call on click here with the idx, main will handle the rest
                    # pass the highlight function down to on_click
                    #on_click(order[][d_idx], highlight)
                    return Consume(true)
                end
            end
        end
        return Consume(false)
    end

    # processed_data = lift((x, y) -> load_data(data[x], y), selected_data, order[1])

    #= matrix view code
    # coords, colors for points
    point_data = lift(x -> generate_points(x[1], x[2], matrix_spacing, transition_spacing), processed_data)

    currently_selected = Observable(1)

    sliderTransition = SliderGrid(
        window[4, :],
        (
            label="Transition",
            range=[1:num_transitions;],
            startvalue=1,
        ),
    )

    pointValueFilter = SliderGrid(
        window[5, :],
        (
            label="Point Filter",
            range=0.0:0.01:1.0,
            startvalue=0.0,
        ),
    )

    filterValue = lift(x -> x, pointValueFilter.sliders[1].value)

    # need to filter out based on atom numbers
    filteredCoords, filteredColors = splitobs(@lift begin
        # filteredIdx = vcat(map(x -> [x + 147 * i for i in range(0, length($processed_data[1]))], collect($selected_atoms))...)

        # TODO: apply the other filter now
        filteredIdx = map((x) -> x[1], findall(x -> x > $filterValue, $point_data[2]))
        filteredCoords = map(x -> $point_data[1][x], filteredIdx)
        filteredColors = map(x -> $point_data[2][x], filteredIdx)
        return (filteredCoords, filteredColors)
    end)

    mat_2d = @lift begin
        mat = $processed_data[1][$currently_selected]

        mat_mask = ones(size(mat))
        mask = findall(x -> x < $filterValue, mat)
        mat_mask[mask] .= NaN

        # depends on the matrix! 
        #sa = collect($selected_atoms)
        # TODO throw in a test for if matrix_shape[1] == 2, then add columns
        #mat_mask = fill(NaN, size(mat))
        # mat_mask[sa, :] .= 1

        res = mat .* mat_mask
        return res
    end

    matrix_plot = heatmap!(scene_2d, mat_2d, colorrange=(0.0, 1.0), colormap=:viridis)
    matrix_plot.inspectable[] = false

    # invisible bounding box we use to get the position from pick
    # when no points are selected
    scatter_bbox = mesh!(scene, lift(x -> Rect3f(Point3f(0.0),
                Point3f(num_transitions * transition_spacing, x[2][1] * matrix_spacing, x[2][2] * matrix_spacing)), processed_data),
        visible=true, alpha=0.01, color=:white, transparency=true)

    scatter_bbox.inspectable[] = false

    points = scatter!(scene, lift(x -> x, filteredCoords), color=lift(x -> x, filteredColors), colorrange=(0.0, 1.0))
    points.inspectable[] = false

    inspector = DataInspector(scene)
    a = inspector.attributes

    on(events(scene).mouseposition) do mp
        plot, idx = pick(scene)
        pos = position_on_plot(plot, idx)
        if !isnan(pos) && (plot == points || plot == scatter_bbox)
            # index of data point
            d_idx = Int(round(pos[1] / (transition_spacing)))
            if d_idx > 0 && d_idx < num_transitions
                b_box = bBox(d_idx, matrix_spacing, transition_spacing, processed_data[][2])
                if inspector.selection != plot
                    clear_temporary_plots!(inspector, plot)
                    p = wireframe!(scene, b_box, inspectable=false, color=:red)
                    push!(inspector.temp_plots, p)
                elseif !isempty(inspector.temp_plots)
                    p = inspector.temp_plots[1]
                    p[1][] = b_box
                end
                inspector.plot.text[] = string(order[1][][d_idx])
                inspector.plot.position = mp
                inspector.plot.visible[] = true
                currently_selected[] = d_idx
            end
            return Consume(true)
        end
        return Consume(false)
    end

    # create a highlight function for matrices of this size
    hovered = Vector()
    highlight = function hi(t, is_hovered)
        # get index of t
        if is_hovered
            idx = findfirst(item -> item == t, order[1][])
            b_box = bBox(idx, matrix_spacing, transition_spacing, processed_data[][2])
            p = wireframe!(scene, b_box, inspectable=false, color=:blue)
            push!(hovered, p)
        else
            for p in hovered
                delete!(parent_scene(p), p)
            end
            empty!(hovered)
        end
    end

    on(events(scene).mousebutton) do event
        if event.button == Mouse.left
            if event.action == Mouse.press
                plot, idx = pick(scene)
                pos = position_on_plot(plot, idx)
                if !isnan(pos) && (plot == points || plot == scatter_bbox)
                    d_idx = Int(round(pos[1] / (transition_spacing)))

                    # call on click here with the idx, main will handle the rest
                    # pass the highlight function down to on_click
                    on_click(order[][d_idx], highlight)
                    return Consume(true)
                end
            end
        end
        return Consume(false)
    end
    =#

    # start off unselected by default
    segment_selector = scatter!(atom_view, reference_configuration[1],
        color=lift(x -> color_selected(x, reference_configuration[3]), selected_atoms)
    )
    segment_selector.inspectable[] = false

    inspector = DataInspector(atom_view)

    # for now, let's just select individual atoms
    on(events(atom_view).mousebutton, priority=1) do event
        if event.button == Mouse.left
            if event.action == Mouse.press
                plot, idx = pick(atom_view)
                if plot == segment_selector
                    pos = position_on_plot(plot, idx)
                    idx, d = NearestNeighbors.nn(reference_configuration[2], pos)
                    if idx in selected_atoms[]
                        delete!(selected_atoms[], idx)
                    else
                        push!(selected_atoms[], idx)
                    end

                    notify(selected_atoms)

                    return Consume(true)
                end
            end
        end
        return Consume(false)
    end

    on(events(atom_view).mouseposition, priority=-1) do mp
        plot, idx = pick(atom_view)
        if plot == segment_selector
            pos = position_on_plot(plot, idx)
            if !isnan(pos)
                idx, d = NearestNeighbors.nn(reference_configuration[2], pos)
                inspector.plot.text[] = string("Atom ", idx)
                inspector.plot.visible[] = true
                inspector.plot.position = mp
                return Consume(true)
            end
            return Consume(false)
        end
    end

    # listen to currently selected transition
    # this actually does work but is super finicky!
    # lift(x -> set_close_to!(sliderTransition.sliders[1], x), currently_selected)

    # function to calculate center of point at index
    #=
    center = lift(sliderTransition.sliders[1].value) do val
        return get_center(val, matrix_spacing, transition_spacing, processed_data[][2])
    end

    eyepos = Observable(Vec3f(-processed_data[][2][1] * matrix_spacing, -processed_data[][2][1] * matrix_spacing, processed_data[][2][2] * matrix_spacing))
    eyepos = lift(sliderTransition.sliders[1].value) do val
        return Vec3f(center[][1] - (processed_data[][2][1] * matrix_spacing), eyepos[][2], eyepos[][3])
    end

    cc = Makie.Camera3D(scene.scene, center=false, lookat=lift(x -> x, center), eyeposition=lift(x -> x, eyepos))
    center!(scene.scene)
    =#
    return window
end

function get_center(idx, matrix_spacing, transition_spacing, matrix_shape)
    x = idx * transition_spacing
    y = (matrix_shape[1] / 2) * matrix_spacing
    z = (matrix_shape[2] / 2) * matrix_spacing

    return Vec3f(x, y, z)
end

function bBox(idx, matrix_spacing, transition_spacing, matrix_shape)
    # give the box some width
    minX = (idx * transition_spacing) - (transition_spacing / 2)
    maxX = (idx * transition_spacing) + (transition_spacing / 2)

    minY = 1
    maxY = matrix_shape[1]

    minZ = 1
    maxZ = matrix_shape[2]

    return Rect3f(Point3f(minX, minY, minZ),
        Point3f(transition_spacing, (maxY - minY) * matrix_spacing, (maxZ - minZ) * matrix_spacing))
end

function color_selected(selected_atoms, num_atoms)
    colors = [:blue for _ in range(1, num_atoms)]
    for idx in selected_atoms
        colors[idx] = :red
    end
    return colors
end

function load_data(data, order)
    # order matrices by distance relative to i
    d = map(x -> data[x], order)

    # we assume that all matrices are the same size
    matrix_shape = size(d[1])

    # simple min-max norm
    max_val = maximum(map((x) -> maximum(x), d))
    min_val = minimum(map((x) -> minimum(x), d))
    norm = map((x) -> (x .- min_val) / (max_val - min_val), d)

    return norm, matrix_shape
end

function generate_points(data, matrix_shape, matrix_spacing, transition_spacing)
    p = Vector{Point3f}()
    alpha = Vector{Float64}()

    num_matrices = length(data)

    for z in 1:num_matrices
        m = data[z]
        for y in 1:matrix_shape[2]
            for x in 1:matrix_shape[1]
                push!(p, Point3f(float(z) * transition_spacing, float(x) * matrix_spacing, float(y) * matrix_spacing))
                push!(alpha, m[x, y])
            end
        end
    end
    return p, alpha
end
