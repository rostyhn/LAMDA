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

function treepositions(hc, cutoff; orientation=:vertical)
    clusterIdx = cutree(hc; h=cutoff)
    order = StatsBase.indexmap(hc.order)
    nodepos = Dict(-i => (float(order[i]), 0.0) for i in hc.order)

    xs = []
    ys = []
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
            push!(xs, [x1, x1])
            push!(ys, [max(cutoff, y1), ypos])
            push!(clusters, get_st_clusters(hc.merges, lt, clusterIdx))

            # stem
            push!(xs, [x1, x2])
            push!(ys, [ypos, ypos])
            push!(clusters, get_st_clusters(hc.merges, i, clusterIdx))

            push!(xs, [x2, x2])
            push!(ys, [max(cutoff, y2), ypos])
            push!(clusters, get_st_clusters(hc.merges, rt, clusterIdx))
        end
    end
    if orientation == :horizontal
        return ys, xs, clusters
    else
        return xs, ys, clusters
    end
end

function dendrogram!(ax, h, cutoff; hover_callbackfn=(x -> ()), colormap=:tab20, rootcolor=:black, kwargs...)
    cmap = to_colormap(colormap)

    println("Calculating dendrogram...")
    @time tp = treepositions(h, cutoff; kwargs...)

    function on_hover(inspector, plot, idx, clusters)
        status = show_data(inspector, plot, idx)
        if status
            hover_callbackfn(clusters)
        end
        return status
    end

    for (x, y, clusters) in zip(tp...)
        if length(clusters) == 1
            clusterIdx = first(collect(clusters))
            color = cmap[(clusterIdx%length(cmap))+1]
        else
            color = rootcolor
        end

        lines!(ax, x, y;
            color,
            inspector_label=(plot, index, position) -> "$(string(clusters)[1:min(end, 40)])$(length(string(clusters)) > 40 ? "..." : "")",
            inspector_hover=(ins, plot, idx) -> on_hover(ins, plot, idx, clusters))
    end

    # add cutoff line
    if cutoff > minimum(h.heights)
        l = lines!(ax, [0, length(h.order)], [cutoff, cutoff]; linestyle=:dash, color=:grey)
        l.inspectable[] = false
    end
end
