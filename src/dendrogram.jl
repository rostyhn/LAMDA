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

# basically assigns each cluster a unique id
function get_hierarchy(hc)
    c2idx = Dict{Set{Int},Int}()
    clusterIdx = collect(eachindex(hc.order))
    c_to_parent = Dict{Set{Int},Set{Int}}()
    parent_to_c = Dict{Set{Int},Tuple{Set{Int},Set{Int}}}()

    for i in 1:size(hc.merges, 1)
        pg = get_st_clusters(hc.merges, i, clusterIdx)
        c2idx[pg] = i

        lt = hc.merges[i, 1]
        rt = hc.merges[i, 2]

        lg = get_st_clusters(hc.merges, lt, clusterIdx)
        rg = get_st_clusters(hc.merges, rt, clusterIdx)

        if lt < 0
            c2idx[lg] = i
        end
        if rt < 0
            c2idx[rg] = i
        end

        c_to_parent[lg] = pg
        c_to_parent[rg] = pg
        parent_to_c[pg] = (lg, rg)

    end

    return c2idx, c_to_parent, parent_to_c
end

# renders the clustering at the specified cutoff value
function treepositions(hc, cutoff)::Tuple{
    Vector{Any},
    Vector{Set{Int}},
    Dict{Set{Int},Vector{Int}}}

    # guarantees consistent labelling with main cluster info 
    clusterIdx = collect(eachindex(hc.order))
    order = StatsBase.indexmap(hc.order)
    nodepos = Dict(-i => (float(order[i]), 0.0) for i in hc.order)

    lines = []
    clusters = []
    c2lx = Dict{Set{Int},Vector{Int}}()
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

            lg_ar = get(c2lx, lg, [])
            rg_ar = get(c2lx, rg, [])
            pg_ar = get(c2lx, pg, [])

            c2lx[lg] = push!(lg_ar, lx - 1)
            c2lx[rg] = push!(rg_ar, lx + 1)
            c2lx[pg] = push!(pg_ar, lx)

            lx += 3
        end
    end

    return lines, clusters, c2lx
end

# gets line positions for a specified branch in the dendrogram 
function branch(cd::ClusterData, root::Set{Int})
    children = Ref([])
    dfs(cd, root, children)

    new_lines = []
    corrected_children = []

    for c in children[]
        idx = cd.c2lx[c]
        for i in 1:length(idx)
            push!(corrected_children, c)
        end
        append!(new_lines, map(x -> cd.lines[x], idx))
    end

    return corrected_children, new_lines
end

function dendrogram!(ax,
    cluster_info,
    cluster_data,
    hovered::MaybeObservable{Set{Int}},
    cluster_annotations;
    hover_callbackfn=(x -> ()),
    colormap=:tab20,
    rootcolor=:black,
    on_click=(x -> ()),
    kwargs...)

    ax.xgridvisible = false
    ax.ygridvisible = false

    dendrogram = @lift begin
        clusters = $(cluster_info).clusters
        lines = $(cluster_info).lines

        colors = []
        for c in clusters
            color = cluster_color(cluster_data[], c)
            push!(colors, set_color_alpha(color, 0.3))
        end

        # to get label idx just divide by 2
        function get_cluster(i)
            return clusters[div(i, 2)]
        end

        all_x = reduce(vcat, map(x -> [x[1][1], x[2][1]], lines))
        min_x, max_x = extrema(all_x)
        all_y = reduce(vcat, map(x -> [x[1][2], x[2][2]], lines))
        min_y, max_y = extrema(all_y)
        cutoff_line = ([min_x, max_x], [min_y, min_y])

        cl_to_idx = Dict{Set{Int},Int}()
        for (i, c) in enumerate(clusters)
            cl_to_idx[c] = i
        end

        return lines, colors, cutoff_line, cl_to_idx, get_cluster, clusters, (min_x, max_x), (min_y, max_y)
    end

    highlighted = []
    on(dendrogram) do d
        empty!(highlighted)
    end

    d_colors = lift(x -> x[2], dendrogram)
    c_dict = lift(x -> x[4], dendrogram)

    function on_hover(plt, idx, pos)
        cl = dendrogram[][5](2)
        if div(idx, 2) <= length(dendrogram[][6])
            cl = dendrogram[][5](idx)
        end
        hover_callbackfn(cl)

        if hovered[] != cl
            hovered[] = cl
            notify(hovered)
        end

        s = get_val(cluster_annotations[], "titles", cl)
        return str_limit(s)
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
            mp = mouseposition_px(ax.scene)
            plot, idx = pick(ax, mp, min(Int(round(dendrogram[][7][2] - dendrogram[][7][2] / 4)), 10))
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
    l = lines!(ax, lift(x -> x[3][1], dendrogram), lift(x -> x[3][2], dendrogram);
        linestyle=:dash,
        color=:grey)

    l.inspectable[] = false

    # add listeners to reset limits whenever something changes
    @lift begin
        xlo, xhi = $dendrogram[7]
        ylo, yhi = $dendrogram[8]
        xlims!(ax, (xlo - 1), (xhi + 1))
        ylims!(ax, (0.0, yhi + 0.1))
    end
end
