function build_selection_window(fig_size, data::Dict{Tuple{Int,Int},Matrix{Float64}}, order)
    window = Figure(size=fig_size)

    scene = LScene(window[1, 1], show_axis=false,
        scenekw=scenekw = (backgroundcolor=:white, clear=true))

    # order matrices by distance relative to i

    # we assume that all matrices are the same size
    d = map(x -> data[x], order) 
    max_val = maximum(map((x) -> maximum(x), d))
    norm = map((x) -> x / max_val, d)
    coords, alpha = generate_points(size(d[1]), length(d), 5.0, norm)
    scatter!(scene, coords, color=alpha)
    return window
end


function generate_points(matrix_shape, num_matrices, spacing, data)
    p = Vector{Point3f}()
    alpha = Vector{Float64}()
    
    for z in 1:num_matrices
        m = data[z]
        for y in 1:matrix_shape[2]
            for x in 1:matrix_shape[1]
                if m[x,y] > 0.3
                    push!(p, Point3f(float(x), float(z) * spacing, float(y)) * spacing)
                    push!(alpha, m[x,y])
                end
            end
        end
    end
    return p, alpha
end
