using Makie
using StatsBase

function get_st_cluster(merge, i, clusterIdx)
    if i < 0
        return clusterIdx[-i]
    end

    lt = merge[i, 1]
    rt = merge[i, 2]

    c_lt = get_st_cluster(merge, lt, clusterIdx)
    c_rt = get_st_cluster(merge, rt, clusterIdx)

    if c_lt == c_rt
        return c_lt
    end

    return -1
end

function treepositions(hc, cutoff; orientation=:vertical)
    clusterIdx = cutree(hc; h=cutoff)
    order = StatsBase.indexmap(hc.order)
    nodepos = Dict(-i => (float(order[i]), 0.0) for i in hc.order)

    xs = []
    ys = []
    clusterIDs = []
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
            push!(clusterIDs, get_st_cluster(hc.merges, lt, clusterIdx))

            # stem
            push!(xs, [x1, x2])
            push!(ys, [ypos, ypos])
            push!(clusterIDs, get_st_cluster(hc.merges, i, clusterIdx))

            push!(xs, [x2, x2])
            push!(ys, [max(cutoff, y2), ypos])
            push!(clusterIDs, get_st_cluster(hc.merges, rt, clusterIdx))
        end
    end
    if orientation == :horizontal
        return ys, xs, clusterIDs
    else
        return xs, ys, clusterIDs
    end
end


function dendrogram!(ax, h, cutoff; colormap=:tab20, rootcolor=:black, kwargs...)
    cmap = to_colormap(colormap)

    println("Calculating dendrogram...")
    @time tp = treepositions(h, cutoff; kwargs...)

    for (x, y, clusterIdx) in zip(tp...)
        if clusterIdx == -1
            color = rootcolor
        else
            color = cmap[(clusterIdx%length(cmap))+1]
        end
        lines!(ax, x, y; color)
    end

    # add cutoff line
    if cutoff > minimum(h.heights)
        lines!(ax, [0, length(h.order)], [cutoff, cutoff]; linestyle=:dash, color=:grey)
    end
end
