using Makie
using StatsBase

function get_st_clusters(merge, i, clusterIdx)
    if i < 0
        return Set(clusterIdx[-i])
    end

    lt = merge[i, 1]
    rt = merge[i, 2]

    c_lt = get_st_clusters(merge, lt, clusterIdx)
    c_rt = get_st_clusters(merge, rt, clusterIdx)

    return union(c_lt, c_rt)
end

function treepositions(hc, cutoff; orientation=:vertical)::Tuple{Vector{Any},Vector{Set{Int}}}
    clusterIdx = cutree(hc; h=cutoff)
    order = StatsBase.indexmap(hc.order)
    nodepos = Dict(-i => (float(order[i]), 0.0) for i in hc.order)

    lines = []
    clusters = []
    for i in 1:size(hc.merges, 1)
        # negative id is a leaf, positive is a subtree
        lt = hc.merges[i, 1] # left subtree
        rt = hc.merges[i, 2] # right subtree

        x1, y1 = nodepos[lt]
        x2, y2 = nodepos[rt]
        xpos = (x1 + x2) / 2
        ypos = hc.heights[i]
        nodepos[i] = (xpos, ypos)

        if ypos > cutoff
            push!(lines, (Point2(x1, max(cutoff, y1)), Point2(x1, ypos)))
            push!(clusters, get_st_clusters(hc.merges, lt, clusterIdx))

            # stem
            push!(lines, (Point2(x1, ypos), Point2(x2, ypos)))
            push!(clusters, get_st_clusters(hc.merges, i, clusterIdx))

            push!(lines, (Point2(x2, max(cutoff, y2)), Point2(x2, ypos)))
            push!(clusters, get_st_clusters(hc.merges, rt, clusterIdx))
        end
    end

    if orientation == :horizontal
        return lines, clusters
    else
        return lines, clusters
    end
end

function dendrogram!(ax, h, cutoff, h_range; hover_callbackfn=(x -> ()), colormap=:tab20, rootcolor=:black, hovered_index=Observable(0), kwargs...)
    cmap = to_colormap(colormap)

    #FIXME still fires twice thanks to multiple observables
    @time dendrogram = @lift begin
        println("Calculating dendrogram...")
        @time lines, clusters = treepositions($h, $cutoff; kwargs...)
        colors = []
        for c in clusters
            if length(c) == 1
                clusterIdx = first(collect(c))
                color = cmap[mod1(clusterIdx, length(cmap))]
            else
                color = rootcolor
            end
            push!(colors, color)
        end

        # to get label idx just divide by 2
        labelfn = (plt, idx, pos) -> str_limit(clusters[div(idx, 2)])

        last_bBox = nothing
        function on_hover(inspector, plot, idx)
            status = show_data(inspector, plot, idx)
            if status && length(clusters[div(idx, 2)]) > 0
                hover_callbackfn(clusters[div(idx, 2)])
            end

            return status
        end

        cutoff_line = ([0, length(h[].order)], [$cutoff, $cutoff])

        cl_to_idx = Dict{Set{Int},Int}()
        for (i, c) in enumerate(clusters)
            cl_to_idx[c] = i
        end

        return lines, colors, labelfn, on_hover, cutoff_line, cl_to_idx
    end

    colors = lift(x -> x[2], dendrogram)

    linesegments!(ax,
        lift(x -> x[1], dendrogram);
        color=colors,
        inspector_label=lift(x -> x[3], dendrogram),
        inspector_hover=lift(x -> x[4], dendrogram))

    last_bBox = nothing

    #=
    @lift begin

        # FIXME doesn't keep up with changes in dendrogram
        c_dict = $dendrogram[6]
        idx = c_dict[$hovered_index]

        if !isnothing(last_bBox)
            colors[][last_bBox] = $dendrogram[2][last_bBox]
        end

        colors[][idx] = to_color(:red)
        notify(colors)
        last_bBox = idx
    end
    =#

    # add cutoff line
    l = lines!(ax, lift(x -> x[5][1], dendrogram), lift(x -> x[5][2], dendrogram);
        linestyle=:dash,
        color=:grey,
        visible=lift((x, y) -> x > minimum(y.heights), cutoff, h))
    l.inspectable[] = false

    # add listeners to reset limits whenever something changes
    @lift begin
        lo, hi = $h_range
        ylims!(ax, (lo - 0.1, hi + 0.1))
        reset_limits!(ax, yauto=false)
    end
end
