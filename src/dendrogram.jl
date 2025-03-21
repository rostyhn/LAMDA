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

function treepositions(hc, cutoff)::Tuple{
    Vector{Any},
    Vector{Set{Int}},
    Dict{Set{Int},Set{Int}},
    Dict{Set{Int},Tuple{Set{Int},Set{Int}}},
    Dict{Set{Int},Vector{Int}}}

    clusterIdx = cutree(hc; h=cutoff)
    order = StatsBase.indexmap(hc.order)
    nodepos = Dict(-i => (float(order[i]), 0.0) for i in hc.order)

    lines = []
    clusters = []
    c_to_parent = Dict{Set{Int},Set{Int}}()
    parent_to_c = Dict{Set{Int},Tuple{Set{Int},Set{Int}}}()
    c_to_idx = Dict{Set{Int},Vector{Int}}()
    lx = 2
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
            lg = get_st_clusters(hc.merges, lt, clusterIdx)
            push!(lines, (Point2(x1, max(cutoff, y1)), Point2(x1, ypos)))
            push!(clusters, lg)

            pg = get_st_clusters(hc.merges, i, clusterIdx)
            # stem
            push!(lines, (Point2(x1, ypos), Point2(x2, ypos)))
            push!(clusters, pg)

            rg = get_st_clusters(hc.merges, rt, clusterIdx)
            push!(lines, (Point2(x2, max(cutoff, y2)), Point2(x2, ypos)))
            push!(clusters, rg)

            lg_ar = get(c_to_idx, lg, [])
            rg_ar = get(c_to_idx, rg, [])
            pg_ar = get(c_to_idx, pg, [])

            c_to_idx[lg] = push!(lg_ar, lx - 1)
            c_to_idx[rg] = push!(rg_ar, lx + 1)
            c_to_idx[pg] = push!(pg_ar, lx)

            c_to_parent[lg] = pg
            c_to_parent[rg] = pg
            parent_to_c[pg] = (lg, rg)
            lx += 3
        end
    end

    return lines, clusters, c_to_parent, parent_to_c, c_to_idx
end

# gets line positions for a specified branch in the dendrogram 
function branch(ci::ClusterInfo, root::Set{Int})
    children = Ref([])
    dfs(ci, root, children)

    new_lines = []
    corrected_children = []

    for c in children[]
        idx = ci.c_to_idx[c]
        for i in 1:length(idx)
            push!(corrected_children, c)
        end
        append!(new_lines, map(x -> ci.lines[x], idx))
    end

    return corrected_children, new_lines
end

function dendrogram!(ax,
    cluster_info,
    hovered=Observable(Set{Int}(1));
    hover_callbackfn=(x -> ()),
    colormap=:tab20,
    rootcolor=:black,
    on_click=((x, y) -> ()),
    kwargs...)

    ax.xgridvisible = false
    ax.ygridvisible = false

    cmap = to_colormap(colormap)

    @time dendrogram = @lift begin
        println("Calculating dendrogram...")
        clusters = $(cluster_info).clusters
        lines = $(cluster_info).lines
        cutoff = $(cluster_info).cutoff

        colors = []
        for c in clusters
            if length(c) == 1
                clusterIdx = first(collect(c))
                color = cmap[mod1(clusterIdx, length(cmap))]
            else
                color = to_color(rootcolor)
            end
            push!(colors, set_color_alpha(color, 0.2))
        end

        # to get label idx just divide by 2
        function get_cluster(i)
            return clusters[div(i, 2)]
        end

        cutoff_line = ([0, length($(cluster_info).assignments)], [cutoff, cutoff])

        cl_to_idx = Dict{Set{Int},Int}()
        for (i, c) in enumerate(clusters)
            cl_to_idx[c] = i
        end

        return lines, colors, cutoff_line, cl_to_idx, get_cluster, clusters
    end

    highlighted = []
    on(dendrogram) do d
        empty!(highlighted)
    end

    d_colors = lift(x -> x[2], dendrogram)
    c_dict = lift(x -> x[4], dendrogram)

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

    on(hovered) do hov
        for (h, ogCol) in highlighted
            d_colors.val[h] = ogCol
        end
        empty!(highlighted)

        if !isnothing(hov)
            for c in collect(hov)
                if Set(c) in keys(c_dict[])
                    idx = c_dict[][Set(c)]
                    ogColor = d_colors.val[idx]
                    d_colors.val[idx] = set_color_alpha(ogColor, 1.0)
                    push!(highlighted, (idx, ogColor))
                end
            end
        end
        d_colors[] = d_colors[]
        notify(d_colors)
    end

    ls = linesegments!(ax,
        lift(x -> x[1], dendrogram);
        color=d_colors,
        inspector_label=on_hover,
    )

    m_events = addmouseevents!(ax.scene)

    on(m_events.obs) do e
        if e.type === MouseEventTypes.leftdown
            if !isnothing(hovered[])
                on_click(hovered[])
            end
        elseif e.type === MouseEventTypes.over
            plot, idx = pick(ax)
            if isnothing(plot) && !isnothing(hovered[])
                hovered.val = nothing
                notify(hovered)
            end
        elseif e.type == MouseEventTypes.out
            if !isnothing(hovered[])
                hovered.val = nothing
                notify(hovered)
            end
        end
    end

    # add cutoff line
    #=l = lines!(ax, lift(x -> x[3][1], dendrogram), lift(x -> x[3][2], dendrogram);
        linestyle=:dash,
        color=:grey)

    l.inspectable[] = false

    # add listeners to reset limits whenever something changes
    @lift begin
        lo, hi = $(cluster_info).h_range
        ylims!(ax, (lo - 0.1, hi + 0.1))
        reset_limits!(ax, yauto=false)
    end=#
end
