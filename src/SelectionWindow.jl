using Makie: clear_temporary_plots!, Orthographic, SparseArrays
using StatsBase
using UMAP

function build_selection_window(fig_size,
    data,
    t_list,
    t_to_idx,
    on_click,
    num_atoms,
    alignedPositions,
    transitionKDTree, dms, volData, sampleRanges, volRange, vol_cmap, selected_invariant, clustering, selected_dm, scalars, h_cutoff, cluster_groups)

    window = Figure(size=fig_size)
    user_groups = Observable(Dict())

    # reorders distance matrix according to clustering
    reordered_matrix = @lift begin
        m = dms[$selected_dm]
        rm = zeros(size(m))

        # gets the correct idx 
        mtx_to_t = Dict()
        t_to_mtx = Dict()
        for (i, r) in enumerate($clustering.order)
            rm[i, :] .= m[r, :][$clustering.order]
            mtx_to_t[i] = t_list[r]
            t_to_mtx[t_list[r]] = i
        end

        # get minimum and maximum of entire matrix for cmap
        fl = vec(m)
        return rm, mtx_to_t, (minimum(fl), maximum(fl)), t_to_mtx
    end

    grid = GridLayout()
    window[1, 1] = grid
    hl = Observable(first(t_list))
    hr = Observable(last(t_list))

    ltv = setup_transition_view(window, grid, (1, 1), alignedPositions, hl, scalars, sampleRanges, volData, vol_cmap, volRange, t_to_idx)
    rtv = setup_transition_view(window, grid, (1, 2), alignedPositions, hr, scalars, sampleRanges, volData, vol_cmap, volRange, t_to_idx)

    link_cameras_lscenes([ltv, rtv])

    invar_menu = Menu(window, options=["t1", "t2", "t3"], tellwidth=false)
    on(invar_menu.selection) do val
        selected_invariant[] = val
    end

    #grid[3, :] = hgrid!(Label(window, "Selected invariant"),
    #    invar_menu,
    #    Colorbar(window, colormap=vol_cmap, limits=volRange, vertical=false, size=16))

    hovered_cluster = Observable(Set{Int64}(1))
    graph_ax = Axis(window[2:3, 1], backgroundcolor=:transparent)
    hm_ax, hm = heatmap(window[1:2, 2], lift(x -> x[1], reordered_matrix))
    hm_inspector = DataInspector(hm)

    hidedecorations!(hm_ax)
    deregister_interaction!(hm_ax, :rectanglezoom)

    #=
    on(events(hm_ax).mouseposition) do mp
        plot, _ = pick(hm_ax)
        if plot == hm
            xy = mouseposition(hm_ax)
            i, j = Int.(round.(xy))
            oldi = t_to_idx[reordered_matrix[][2][i]]
            oldj = t_to_idx[reordered_matrix[][2][j]]
            hl[] = t_list[oldi]
            hr[] = t_list[oldj]
            notify(hl)
            notify(hr)
        end
        return Consume(false)
    end
    =#

    # https://github.com/MakieOrg/Makie.jl/blob/master/src/interaction/inspector.jl
    last_bBox = nothing
    @lift begin
        if !isnothing(last_bBox)
            delete!(parent_scene(last_bBox), last_bBox)
        end
        if length($hovered_cluster) > 0
            ts_idx = reduce(vcat, map(x -> $cluster_groups[x], collect($hovered_cluster)))
            ts = t_list[ts_idx]

            t_to_mtx = $reordered_matrix[4]

            m_idx = map(x -> t_to_mtx[x], ts)

            lo = minimum(m_idx)
            hi = maximum(m_idx)

            # need to draw n bounding boxes over the heatmap
            bbox = Rect2(lo, lo, hi - lo, hi - lo)

            p = wireframe!(
                hm_ax.scene, bbox, color=:red,
                visible=true, inspectable=false,
                depth_shift=-1.0f-3
            )
            last_bBox = p
        end
    end

    function on_dendrogram_hover(c)
        if !isempty(c)
            hovered_cluster[] = c
            notify(hovered_cluster)
        end
    end

    @lift begin
        dendrogram!(graph_ax, $clustering, $h_cutoff; hover_callbackfn=on_dendrogram_hover)
    end


    dm_menu = Menu(window, options=collect(keys(dms)))
    on(dm_menu.selection) do val
        selected_dm[] = val
    end

    window[3, 2] = hgrid!(Label(window, "Distance matrix"),
        dm_menu,
        Colorbar(window, limits=lift(x -> x[3], reordered_matrix), vertical=false, size=16))

    return window
end

function simple_atom_view(scene, g, ap, t, order, scalars, sel)
    opts = sort(collect(keys(scalars)))

    gg = GridLayout(g[end+1, :])
    m = Menu(gg[1, 1], options=opts, default=length(sel[]) == 0 ? first(opts) : sel[])

    colorInfo = @lift begin
        opt = $(m.selection)
        vals = scalars[opt][t][order]

        extremaVals = extrema(vals)
        labelfn = (self, i, p) -> "Atom $(i); weight: $(self.color[][i])"

        # three cases:
        # sequential ascending, descending and diverging
        # divergent if we are approximately around 0 when subtracting the absolute values of the min and max
        minVal, maxVal = extremaVals
        if isapprox(abs(maxVal) - abs(minVal), 0; atol=1)
            println("using divergent colorscheme")
            cmap = resample_cmap(:bam, 100; alpha=([(-0.99):0.02:(0.99);] ./ 0.1) .^ 6)
        else
            # ascending sequential if min is closer to 0
            cmap = resample_cmap(:reds, 147, alpha=range(; start=0.01, stop=1.0, length=147)) #seq ascending
            if abs(minVal) > abs(maxVal)
                println("using descending sequential colorscheme")
                # otherwise reverse it
                reverse!(cmap)
            end
        end

        sel[] = opt
        notify(sel)

        return vals, extremaVals, cmap, labelfn
    end

    scatter!(scene, ap, color=lift(x -> x[1], colorInfo),
        colorrange=lift(x -> x[2], colorInfo),
        colormap=lift(x -> x[3], colorInfo),
        inspector_label=lift(x -> x[4], colorInfo),
        markersize=30)

    cbar = Colorbar(gg[1, 2], colorrange=lift(x -> x[2], colorInfo), vertical=false, colormap=lift(x -> x[3], colorInfo), tellwidth=false)

    return [], [(gg, [m, cbar])]
end



function setup_transition_view(fig, parentGrid, loc, ap, hovered, scalars, sampleRanges, volData, vol_cmap, volumeRange, t_to_idx)
    rootScene = LScene(
        fig,
        show_axis=false,
        scenekw=(backgroundcolor=:white, clear=true),
    )

    m = Menu(fig, options=["Volume", "Initial State", "Final State"],
        default="Volume")

    l = Label(fig, lift(x -> string(x), hovered), tellwidth=false)
    i, j = loc
    g = vgrid!(m, rootScene, l)
    parentGrid[i, j] = g

    inspector = DataInspector(rootScene)

    sel = Observable("Volume")
    bp_sel = Observable("")
    ap_sel = Observable("")

    scene_listeners = Vector{Any}()
    ui_elements = Vector{Any}()

    function cleanup_t_view()
        cam = camera(rootScene)
        empty!(rootScene)

        for listener in scene_listeners
            off(listener)
            listener = Nothing
        end
        empty!(scene_listeners)

        # clear UI elements
        for c in ui_elements
            gg, elements = c
            for e in elements
                empty!(e.blockscene)
                delete!(e)
            end
            if !isnothing(gg)
                Makie.trim!(gg)
                # only way to delete a gridlayout
                GridLayoutBase.remove_from_gridlayout!(gg.layoutobservables.gridcontent[])
            end
        end
        Makie.trim!(g)
        #GC.gc() # stops memleak
    end

    menu_listener = nothing

    @lift begin
        bp = function (scene, inspector, g)
            return simple_atom_view(scene, g, $ap[$hovered][1], $hovered, 1, scalars, bp_sel)
        end

        afp = function (scene, inspector, g)
            return simple_atom_view(scene, g, $ap[$hovered][2], $hovered, 2, scalars, ap_sel)
        end

        vol = function (scene, inspector, g)
            vd = reshape($volData[:, t_to_idx[$hovered]], (length($sampleRanges[1]), length($sampleRanges[2]), length($sampleRanges[3])))

            v = volume!(scene,
                lift(x -> extrema(x[1]), sampleRanges),
                lift(x -> extrema(x[2]), sampleRanges),
                lift(x -> extrema(x[3]), sampleRanges),
                vd;
                colormap=vol_cmap,
                algorithm=:absorption,
                fxaa=false,
                transparency=true,
                shading=NoShading,
                colorrange=volumeRange,
                visible=true)
            v.inspectable[] = false

            return [], []
        end

        rf = Dict()
        rf["Initial State"] = bp
        rf["Final State"] = afp
        rf["Volume"] = vol

        cleanup_t_view()
        if !isnothing(menu_listener)
            off(menu_listener)
            menu_listener = nothing
        end

        initial_render_func = rf[$sel]
        il, is = initial_render_func(rootScene, inspector, g)

        for l in il
            push!(scene_listeners, l)
        end

        for s in is
            push!(ui_elements, s)
        end

        menu_listener = on(m.selection) do cw
            cleanup_t_view()

            listeners, scenes = rf[cw](rootScene, inspector, g)

            for l in listeners
                push!(scene_listeners, l)
            end

            for s in scenes
                push!(ui_elements, s)
            end
            update_cam!(rootScene.scene)
            sel[] = cw
            notify(sel)
        end
    end

    return rootScene
end
