using Makie: clear_temporary_plots!, Orthographic, SparseArrays
using StatsBase
using Graphs
using GraphMakie
using NetworkLayout
using UMAP
using Clustering

function get_matrix_data(label, dms, reference_configuration, selected_atoms, iv1, alignedPositions, transitionKDTree)
    if label == "LNCD"
        numberOfBins = 100
        minInvariant1, maxInvariant1, transitionInvariants1 = iv1

        @show stepSize = (maxInvariant1 - minInvariant1) / numberOfBins
        binEdges = [minInvariant1:stepSize:maxInvariant1;]

        transitionDistribution = Dict{Tuple{Int,Int},Vector{SparseVector{Float64}}}() # in transition (String), List of control points { sparse neighbourhood distribution }  
        @time for (t, values) in transitionInvariants1

            p1, p2 = get_from_t_dict(alignedPositions, t)
            k1, k2 = get_from_t_dict(transitionKDTree, t)

            transitionDistribution[t] = computeInvariantDistributionInNeighborhood(values, p1, binEdges, 10, k1)
        end
        @show "done with distributions"

        distancesToReference = computeLNCD.(Ref(transitionDistribution),
            Ref(reference_configuration),
            keys(transitionInvariants1),
            Ref(selected_atoms))
        zipped = collect(zip(collect(keys(transitionInvariants1)), distancesToReference))

        ref_distances = Dict()
        for (t, d) in zipped
            ref_distances[t] = d
        end

        return ref_distances
    end

    m = dms[label]["matrix"]
    t_to_idx = dms[label]["t_to_idx"]
    row_idx = t_to_idx[reference_configuration]
    row = m[row_idx, :]

    graph_dist = Dict()
    for (t, idx) in t_to_idx
        graph_dist[t] = row[idx]
    end

    return graph_dist
end


function build_selection_window(fig_size,
    data,
    t_list,
    t_to_idx,
    on_click,
    num_atoms,
    alignedPositions,
    transitionKDTree, dms, volData, sampleRanges, volRange, cmap, selected_invariant)

    window = Figure(size=fig_size)

    user_groups = Observable(Dict())

    selected_dm = Observable(first(keys(dms)))

    reordered_matrix = @lift begin
        println("Clustering $($selected_dm)")

        m = dms[$selected_dm]
        res = hclust(m, linkage=:ward, branchorder=:barjoseph)
        rm = zeros(size(m))

        # gets the correct idx 
        mtx_to_t = Dict()
        for (i, r) in enumerate(res.order)
            rm[i, :] .= m[r, :][res.order]
            mtx_to_t[i] = t_list[r]
        end

        # get minimum and maximum of entire matrix for cmap
        fl = vec(m)
        return rm, mtx_to_t, (minimum(fl), maximum(fl))
    end

    grid = GridLayout()
    window[1, 1] = grid
    hl = Observable(first(t_list))
    hr = Observable(last(t_list))

    svl = LScene(grid[1, 1], show_axis=false, scenekw=(backgroundcolor=:black, clear=true))
    svr = LScene(grid[1, 2], show_axis=false, scenekw=(backgroundcolor=:black, clear=true))

    Label(grid[2, 1], lift(x -> string(x), hl), tellwidth=false)
    Label(grid[2, 2], lift(x -> string(x), hr), tellwidth=false)

    invar_menu = Menu(window, options=["t1", "t2", "t3"], tellwidth=false)
    on(invar_menu.selection) do val
        selected_invariant[] = val
    end
    grid[3, :] = hgrid!(Label(window, "Selected invariant"),
        invar_menu,
        Colorbar(window, colormap=cmap, limits=volRange, vertical=false, size=16))

    @time vl = volume!(svl,
        lift(x -> x[1], sampleRanges),
        lift(x -> x[2], sampleRanges),
        lift(x -> x[3], sampleRanges),
        lift((x, y, z) -> reshape(y[t_to_idx[x], :], (length(z[1]), length(z[2]), length(z[3]))), hl, volData, sampleRanges);
        colormap=cmap,
        algorithm=:absorption,
        fxaa=false,
        transparency=true,
        shading=NoShading,
        colorrange=lift(x -> x, volRange),
        overdraw=true,
        visible=true)
    vl.inspectable[] = false

    @time vr = volume!(svr,
        lift(x -> x[1], sampleRanges),
        lift(x -> x[2], sampleRanges),
        lift(x -> x[3], sampleRanges),
        lift((x, y, z) -> reshape(y[t_to_idx[x], :], (length(z[1]), length(z[2]), length(z[3]))), hr, volData, sampleRanges);
        colormap=cmap,
        algorithm=:absorption,
        fxaa=false,
        transparency=true,
        shading=NoShading,
        colorrange=volRange,
        overdraw=true,
        visible=true)
    vr.inspectable[] = false

    graph_ax = Axis(window[2:3, 1], backgroundcolor=:transparent)
    campixel!(graph_ax.scene)

    hm_ax, hm = heatmap(window[1:2, 2], lift(x -> x[1], reordered_matrix), inspector_label=(i, p, idx) -> "")

    hidedecorations!(hm_ax)
    DataInspector(hm)
    deregister_interaction!(hm_ax, :rectanglezoom)
    deregister_interaction!(graph_ax, :rectanglezoom)

    dm_menu = Menu(window, options=collect(keys(dms)))
    on(dm_menu.selection) do val
        selected_dm[] = val
    end

    window[3, 2] = hgrid!(Label(window, "Distance matrix"),
        dm_menu,
        Colorbar(window, limits=lift(x -> x[3], reordered_matrix), vertical=false, size=16))

    embedding = @lift begin
        println("Calculating umap embedding for $($selected_dm)")
        # https://github.com/dillondaudert/UMAP.jl/blob/master/src/umap_.jl
        # not a major bottleneck but should be cached eventually
        @time em = transpose(umap(dms[$selected_dm], 2; metric=:precomputed, spread=250))
        return map(x -> Point2f(x), eachrow(em))
    end

    hovered = Observable(first(t_list))
    tt_bbox = Observable(BBox(0, 0, 0, 0))

    colors = Observable(fill(:blue, length(embedding[])))

    @time sc = scatter!(graph_ax, embedding; color=colors)
    sc.inspectable[] = false
    @time text!(graph_ax, embedding; text=map(x -> string(x), t_list))
    hidedecorations!(graph_ax)

    on(embedding) do _
        autolimits!(graph_ax)
    end

    ax3d = LScene(graph_ax.scene, show_axis=false, bbox=tt_bbox, scenekw=(backgroundcolor=:black, clear=true, size=(250, 250), zorder=100), height=250, width=250)
    ax3d.scene.visible[] = false

    v = volume!(ax3d,
        lift(x -> x[1], sampleRanges),
        lift(x -> x[2], sampleRanges),
        lift(x -> x[3], sampleRanges),
        lift((x, y, z) -> reshape(y[t_to_idx[x], :], (length(z[1]), length(z[2]), length(z[3]))), hovered, volData, sampleRanges);
        colormap=cmap,
        algorithm=:absorption,
        fxaa=false,
        transparency=true,
        shading=NoShading,
        colorrange=volRange,
        overdraw=true,
        visible=true)
    v.inspectable[] = false
    translate!(ax3d.scene, 0, 0, 10)

    center!(ax3d.scene)

    on(events(graph_ax).mouseposition) do mp
        plot, idx = pick(graph_ax)
        if plot == sc
            x, y = events(graph_ax.parent).mouseposition[]
            tt_bbox[] = BBox(x + 15, x + 265, y - 265, y - 15)
            hovered[] = t_list[idx]
            ax3d.scene.visible[] = true
            notify(hovered)
            notify(tt_bbox)
            center!(ax3d.scene)
        else
            ax3d.scene.visible[] = false
        end

        return Consume(false)
    end

    on(events(graph_ax).mousebutton) do event
        if event.button == Mouse.left && event.action == Mouse.press
            plot, idx = pick(graph_ax)
            pos = position_on_plot(plot, idx)
            if !isnan(pos) && (plot == sc)
                on_click(t_list[idx], x -> ())
            end
        end
        return Consume(true)
    end

    on(events(hm_ax).mouseposition) do mp
        colors[] = fill(:blue, length(colors[]))
        plot, _ = pick(hm_ax)
        if plot == hm
            xy = mouseposition(hm_ax)
            i, j = Int.(round.(xy))
            oldi = t_to_idx[reordered_matrix[][2][i]]
            oldj = t_to_idx[reordered_matrix[][2][j]]
            colors[][oldi] = :red
            colors[][oldj] = :red
            hl[] = t_list[oldi]
            hr[] = t_list[oldj]
            notify(hl)
            notify(hr)
            notify(colors)
        end
        #t = t_list[idx]
        #hovered[] = t_list[idx]
        #ax3d.scene.visible[] = true
        #notify(hovered)
        #notify(tt_bbox)
        #center!(ax3d.scene)
        return Consume(false)
    end

    return window
end

function color_selected(selected_atoms, num_atoms)
    colors = [:blue for _ in range(1, num_atoms)]
    for idx in selected_atoms
        colors[idx] = :red
    end
    return colors
end
