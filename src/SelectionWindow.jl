using Makie: clear_temporary_plots!, Orthographic

function pair(v)
    pairs = Vector{Pair{Any,Any}}()
    for i in 1:length(v)-1
        e1 = v[i]
        e2 = v[i+1]
        push!(pairs, Pair(e1, e2))
    end
    return pairs
end

function build_selection_window(fig_size,
    data::Dict{Tuple{Int,Int},Matrix{Float64}}, order, on_click)
    window = Figure(size=fig_size)
    scene = LScene(window[1, 1], show_axis=true,
        scenekw=scenekw = (backgroundcolor=:white, clear=true))

    # space between matrices
    transition_spacing = 18.0
    # space btwn matrix elements
    matrix_spacing = 5.0


    # order matrices by distance relative to i
    # we assume that all matrices are the same size
    d = map(x -> data[x], order)
    matrix_shape = size(d[1])
    max_val = maximum(map((x) -> maximum(x), d))
    min_val = minimum(map((x) -> minimum(x), d))

    # broadcasting somehow messes up the values?
    norm = map((x) -> (x .- min_val) / (max_val - min_val), d)
    pairs = pair(norm)
    #diff = Vector{Any}()
    #for p in pairs
    #    push!(diff, p.second - p.first)
    #end

    sliderTransition = SliderGrid(
        window[2, 1],
        (
            label="Transition",
            range=[1:length(norm);],
            startvalue=1,
            format=x -> string(order[x]),
        ),
    )

    pointValueFilter = SliderGrid(
        window[3, 1],
        (
            label="Point Filter",
            range=0.0:0.01:1.0,
            startvalue=0.0,
        ),
    )

    filterValue = lift(x -> x, pointValueFilter.sliders[1].value)

    coords, colors = generate_points(matrix_shape, length(norm), matrix_spacing, transition_spacing, norm)

    filteredIdx = lift(y -> map((x) -> x[1], findall(x -> x > y, colors)), filterValue)
    filteredCoords = lift(y -> map(x -> coords[x], y), filteredIdx)
    filteredColors = lift(y -> map(x -> colors[x], y), filteredIdx)

    # invisible bounding box we use to get the position from pick
    # when no points are selected
    scatter_bbox = mesh!(scene, Rect3f(Point3f(0.0),
            Point3f(length(norm) * transition_spacing, matrix_shape[1] * matrix_spacing, matrix_shape[2] * matrix_spacing)),
        visible=true, alpha=0.01, color=:white, transparency=true)

    scatter_bbox.inspectable[] = false

    points = scatter!(scene, lift(x -> x, filteredCoords), color=lift(x -> x, filteredColors), colorrange=(0.0, 1.0))
    points.inspectable[] = false

    inspector = DataInspector(scene)
    a = inspector.attributes

    # currently_selected = Observable(1)

    on(events(scene).mouseposition) do mp
        plot, idx = pick(scene)
        pos = position_on_plot(plot, idx)
        if !isnan(pos)
            # index of data point
            d_idx = Int(round(pos[1] / (transition_spacing)))
            if d_idx > 0 && d_idx < length(d)
                bBox = highlight(d_idx, matrix_spacing, transition_spacing, matrix_shape)
                if inspector.selection != plot
                    clear_temporary_plots!(inspector, plot)
                    p = wireframe!(scene, bBox, inspectable=false, color=:red)
                    push!(inspector.temp_plots, p)
                elseif !isempty(inspector.temp_plots)
                    p = inspector.temp_plots[1]
                    p[1][] = bBox
                end
                inspector.plot.text[] = string(order[d_idx])
                inspector.plot.position = mp
                inspector.plot.visible[] = true
                # currently_selected[] = d_idx
            end
            return Consume(true)
        end
        return Consume(false)
    end

    on(events(scene).mousebutton) do event
        if event.button == Mouse.left
            if event.action == Mouse.press
                plot, idx = pick(scene)
                pos = position_on_plot(plot, idx)
                if !isnan(pos)
                    if plot == points || plot == scatter_bbox
                        d_idx = Int(round(pos[1] / (transition_spacing)))
                        # call on click here with the idx, main will handle the rest
                        on_click(order[d_idx])
                    end
                end

            end
        end
    end

    # listen to currently selected transition
    # this actually does work but is super finicky!
    # lift(x -> set_close_to!(sliderTransition.sliders[1], x), currently_selected)

    # function to calculate center of point at index
    center = lift(sliderTransition.sliders[1].value) do val
        return get_center(val, matrix_spacing, transition_spacing, matrix_shape)
    end

    eyepos = Observable(Vec3f(-matrix_shape[1] * matrix_spacing, -matrix_shape[1] * matrix_spacing, matrix_shape[2] * matrix_spacing))
    eyepos = lift(sliderTransition.sliders[1].value) do val
        return Vec3f(center[][1] - (matrix_shape[1] * matrix_spacing), eyepos[][2], eyepos[][3])
    end


    cc = Makie.Camera3D(scene.scene, center=false, lookat=lift(x -> x, center), eyeposition=lift(x -> x, eyepos))
    center!(scene.scene)

    return window
end

function get_center(idx, matrix_spacing, transition_spacing, matrix_shape)
    x = idx * transition_spacing
    y = (matrix_shape[1] / 2) * matrix_spacing
    z = (matrix_shape[2] / 2) * matrix_spacing

    return Vec3f(x, y, z)
end

function highlight(idx, matrix_spacing, transition_spacing, matrix_shape)
    # give the box some width
    minX = (idx * transition_spacing) - (transition_spacing / 2)
    maxX = (idx * transition_spacing) + (transition_spacing / 2)

    minY = 1
    maxY = matrix_shape[1]

    minZ = 1
    maxZ = matrix_shape[2]

    return bBox(minX, minY, minZ, maxX, maxY, maxZ, matrix_spacing, transition_spacing)
end

function bBox(minX, minY, minZ, maxX, maxY, maxZ, matrix_spacing, transition_spacing)
    return Rect3f(Point3f(minX, minY, minZ),
        Point3f(transition_spacing, (maxY - minY) * matrix_spacing, (maxZ - minZ) * matrix_spacing))
end

function generate_points(matrix_shape, num_matrices, matrix_spacing, transition_spacing, data)
    p = Vector{Point3f}()
    alpha = Vector{Float64}()

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
