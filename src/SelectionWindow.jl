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
    data::Dict{Tuple{Int,Int},Matrix{Float64}}, order)
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
    #min_val = minimum(map((x) -> minimum(x), d))

    # broadcasting somehow messes up the values?
    norm = map((x) -> x / max_val, d)

    pairs = pair(norm)
    #diff = Vector{Any}()
    #for p in pairs
    #    push!(diff, p.second - p.first)
    #end


    coords, alpha = generate_points(matrix_shape, length(d), matrix_spacing, transition_spacing, norm)
    points = scatter!(scene, coords, color=alpha)
    points.inspectable[] = false

    inspector = DataInspector(scene)
    a = inspector.attributes

    on(events(scene).mouseposition) do mp
        plot, idx = pick(scene)
        pos = position_on_plot(plot, idx)
        if !isnan(pos) && plot == points
            # index of data point
            d_idx = Int(pos[1] / (transition_spacing))
            if d_idx > 0 && d_idx < length(d)
                bBox = highlight(d_idx, matrix_spacing, transition_spacing, matrix_shape)
                if inspector.selection != plot
                    clear_temporary_plots!(inspector, plot)
                    p = wireframe!(scene, bBox, inspectable=false)
                    push!(inspector.temp_plots, p)
                elseif !isempty(inspector.temp_plots)
                    p = inspector.temp_plots[1]
                    p[1][] = bBox
                end
            end
            return Consume(true)
        end
        return Consume(false)
    end

    # function to calculate center of point at index

    sliderTransition = SliderGrid(
        window[2, 1],
        (
            label="Transition",
            range=[1:length(norm);],
            startvalue=1,
            format=x -> string(order[x]),
        ),
    )

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
        @show m
        for y in 1:matrix_shape[2]
            for x in 1:matrix_shape[1]
                if m[x, y] > 0.3
                    push!(p, Point3f(float(z) * transition_spacing, float(x) * matrix_spacing, float(y) * matrix_spacing))
                    push!(alpha, m[x, y])
                end
            end
        end
    end
    return p, alpha
end
