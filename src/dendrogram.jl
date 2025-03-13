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

function dendrogram!(ax, h, cutoff, h_range; hover_callbackfn=(x -> ()), colormap=:tab20, rootcolor=:black, on_click=((x, y) -> ()), kwargs...)
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
        function get_cluster(i)
            return clusters[div(i, 2)]
        end

        cutoff_line = ([0, length(h[].order)], [$cutoff, $cutoff])

        cl_to_idx = Dict{Set{Int},Int}()
        for (i, c) in enumerate(clusters)
            cl_to_idx[c] = i
        end

        return lines, colors, cutoff_line, cl_to_idx, get_cluster, clusters
    end

    l_highlighted = []
    r_highlighted = []

    on(dendrogram) do d
        empty!(l_highlighted)
        empty!(r_highlighted)
    end

    d_colors = lift(x -> x[2], dendrogram)
    c_dict = lift(x -> x[4], dendrogram)

    hovered = Observable(Set{Int}(1))

    function on_hover(plt, idx, pos)
        cl = dendrogram[][5](2)
        if div(idx, 2) < length(dendrogram[][6])
            cl = dendrogram[][5](idx)
        end
        hover_callbackfn(cl)
        hovered[] = cl
        notify(hovered)
        return str_limit(cl)
    end

    ls = linesegments!(ax,
        lift(x -> x[1], dendrogram);
        color=d_colors,
        inspector_label=on_hover,
    )

    on(events(parent_scene(ls)).mousebutton) do e
        if is_mouseinside(parent_scene(ls))
            if e.button == Mouse.left && e.action == Mouse.press
                ks = events(ls).keyboardstate
                on_click(hovered[], ks)

                highlighted = (Keyboard.a in ks) ? l_highlighted : r_highlighted
                for (h, ogCol) in highlighted
                    d_colors.val[h] = ogCol
                end
                empty!(highlighted)

                for c in collect(hovered[])
                    idx = c_dict[][Set(c)]
                    ogColor = d_colors.val[idx]
                    d_colors.val[idx] = (Keyboard.a in ks) ? to_color(:green) : to_color(:red)
                    push!(highlighted, (idx, ogColor))
                end
                d_colors[] = d_colors[]
                notify(d_colors)
            end
        end
        return Consume(true)
    end

    # add cutoff line
    l = lines!(ax, lift(x -> x[3][1], dendrogram), lift(x -> x[3][2], dendrogram);
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
