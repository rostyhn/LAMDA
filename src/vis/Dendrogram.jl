# renders the clustering at the specified cutoff value
function treepositions(hc::Clustering.Hclust, cutoff::AbstractFloat)::Tuple{
    Vector{Tuple{Point2f,Point2f}},
    Vector{ClusterSet}}

    # guarantees consistent labeling with main cluster info 
    clusterIdx = collect(eachindex(hc.order))
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
        end
    end

    return lines, clusters
end

function treepositions(hc::Clustering.Hclust, root::ClusterSet)::Tuple{
    Vector{Tuple{Point2f,Point2f}},
    Vector{ClusterSet}
}

    # guarantees consistent labeling with main cluster info 
    clusterIdx = collect(eachindex(hc.order))
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

        pg = get_st_clusters(hc.merges, i, clusterIdx)
        intersection = intersect(root, pg)
        if length(intersection) != 0
            lg = get_st_clusters(hc.merges, lt, clusterIdx)
            push!(lines, (Point2(x1, y1), Point2(x1, ypos)))
            push!(clusters, lg)

            # stem
            push!(lines, (Point2(x1, ypos), Point2(x2, ypos)))
            push!(clusters, pg)

            rg = get_st_clusters(hc.merges, rt, clusterIdx)
            push!(lines, (Point2(x2, y2), Point2(x2, ypos)))
            push!(clusters, rg)

            if intersection == root
                break
            end

        end
    end

    return lines, clusters
end

function dendrogram!(ax::Makie.Axis,
    cluster_data::ClusterData,
    hovered::MaybeObservable{ClusterSet},
    cluster_annotations::Observable{ClusterAnnotation};
    hover_callbackfn::Function=(x -> ()),
    on_click::Function=(x -> ()),
    on_rmb::Function=(x -> ()),
    on_cutoff_line_drag::Function=(x -> ()),
    cutoff_reset::Bool=false,
    root::Union{Observable{ClusterSet},Observable{Nothing}}=Observable(nothing),
    cutoff::Union{Observable{<:AbstractFloat},Observable{Nothing}}=Observable(nothing),
    kwargs...)


    ax.xgridvisible = false
    ax.ygridvisible = false
    dendrogram = @lift begin
        # to get label idx just divide by 2
        if isnothing($root) && isnothing($cutoff)
            return error("Must specify either root or cutoff to render dendrogram.")
        end

        if !isnothing($cutoff)
            lines, clusters = treepositions(cluster_data.clustering, $cutoff)
        else
            lines, clusters = treepositions(cluster_data.clustering, $root)
        end

        # for some reason, width indices are set by point instead of by color?
        c2lx = Dict{ClusterSet,Vector{Int}}()
        x = 1
        for y in clusters
            ar = get!(c2lx, y, [])
            push!(ar, x)
            push!(ar, x + 1)
            x += 2
        end

        function get_cluster(i)
            return clusters[div(i, 2)]
        end

        colors = Vector{RGBAf}(undef, length(clusters))
        for (i, c) in enumerate(clusters)
            colors[i] = cluster_data.colors[c]
        end

        all_x = reduce(vcat, map(x -> [x[1][1], x[2][1]], lines), init=[])
        min_x, max_x = extrema(all_x)
        all_y = reduce(vcat, map(x -> [x[1][2], x[2][2]], lines), init=[])
        min_y, max_y = extrema(all_y)
        cut_line = ([min_x, max_x], [min_y, min_y])

        return lines, colors, cut_line, c2lx, get_cluster, clusters, (min_x, max_x), (min_y, max_y)
    end

    highlighted = []
    on(dendrogram) do d
        empty!(highlighted)
    end

    d_colors = lift(x -> x[2], dendrogram)
    d_width = lift(x -> fill(1, length(x[1]) * 2), dendrogram)
    c_dict = lift(x -> x[4], dendrogram)

    function on_hover(plt, idx, pos)
        cl = dendrogram[][5](2)
        if div(idx, 2) <= length(dendrogram[][6])
            cl = dendrogram[][5](idx)
        end
        hover_callbackfn(cl)

        if isnothing(hovered[]) || hovered[] != cl
            hovered[] = cl
        end

        s = get_val(cluster_annotations[], "titles", cl)
        return str_limit(s)
    end

    on(hovered) do hov
        d_width.val[highlighted] .= 1
        empty!(highlighted)

        if !isnothing(hov)
            children = descend_tree(cluster_data, hov)
            highlighted = reduce(vcat,
                filter!(!isnothing,
                    map(x -> get(c_dict[], x, nothing), children)
                ),
                init=[]
            )
            d_width.val[highlighted] .= 5
        end
        d_width[] = d_width[]
        notify(d_width)
    end

    linesegments!(ax,
        lift(x -> x[1], dendrogram);
        color=d_colors,
        linewidth=d_width,
        inspector_label=on_hover,
    )

    cutoff_line::Observable{Tuple{Vector{Float64},Vector{Float64}}} = Observable(dendrogram[][3])
    on(dendrogram) do d
        # reset cutoff line
        if cutoff_reset
            cutoff_line[] = (d[3][1], [0, 0])
        else
            cutoff_line[] = (d[3][1], cutoff_line[][2])
        end
    end

    cutoff_hovered = Observable(false)
    dragging = Ref(false)

    # add cutoff line
    l = lines!(ax,
        lift(x -> x[1], cutoff_line),
        lift(x -> x[2], cutoff_line);
        linewidth=5,
        color=lift(x -> (x) ? to_color(colorant"#D3D3D3") : to_color(:grey), cutoff_hovered))
    l.inspectable[] = false

    m_events = addmouseevents!(ax.scene)

    on(m_events.obs) do e
        if e.type === MouseEventTypes.leftdown
            if !dragging[] && !isnothing(hovered[]) && !cutoff_hovered[]
                on_click(hovered[])
            end
        elseif e.type == MouseEventTypes.rightdown
            if !dragging[] && !isnothing(hovered[]) && !cutoff_hovered[]
                on_rmb(hovered[])
            end
        elseif e.type === MouseEventTypes.over
            plot, idx = pick(ax, 10)
            if isnothing(plot) && !isnothing(hovered[])
                hovered.val = nothing
                notify(hovered)
            end

            if plot == l
                cutoff_hovered[] = true
            else
                cutoff_hovered[] = false
            end
            #= elseif e.type === MouseEventTypes.leftdoubleclick
                cc = deepcopy(d_colors[])
                d_colors[] = map(x -> set_color_alpha(x, 1.0), d_colors[])
                notify(d_colors)
                save("$(time()).pdf", ax.scene, backend=CairoMakie, update=false)
                d_colors[] = cc
                notify(d_colors) =#
        elseif e.type == MouseEventTypes.out
            if !isnothing(hovered[])
                hovered.val = nothing
                notify(hovered)
            end
        elseif e.type == MouseEventTypes.leftdragstart
            if cutoff_hovered[]
                dragging[] = true
            end
        elseif e.type == MouseEventTypes.leftdrag
            if dragging[]
                px, py = float.(mouseposition(ax.scene))
                y = max(0.0, py)
                cutoff_line[] = (cutoff_line[][1], [y, y])
            end
        elseif e.type == MouseEventTypes.leftdragstop
            if dragging[]
                px, py = float.(mouseposition(ax.scene))

                dragging[] = false
                on_cutoff_line_drag(max(0.0, py))
            end
        end
    end

    # add listeners to reset limits whenever something changes
    @lift begin
        xlo, xhi = $dendrogram[7]
        ylo, yhi = $dendrogram[8]
        xlims!(ax, (xlo - 1), (xhi + 1))
        ylims!(ax, (-5, yhi + 0.1))
    end
end
