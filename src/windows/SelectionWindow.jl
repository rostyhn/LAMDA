const MIN_NODE_SIZE = 10.0
const MAX_NODE_SIZE = 100.0

function setup_selection_window(active_trajectory::Trajectory,
    screen::GLMakie.Screen,
    dm::AbstractArray{Float32},
    transitionSequence::Vector{Transition},
    clustering::Clustering.Hclust{Float32},
    selected_dm_name::String,
    dataPath::String,
    align_with::Maybe{String}=nothing;
    sq_resolution=0.5)

    @debug "Allocating data and functions"
    init_memory = Sys.free_memory() / 2^20
    (;
        stretchedPrincipalAxes,
        scalars,
        scalar_ranges,
        alignedPositionsMatrices,
        alignments,
        name,
        bonds,
    ) = active_trajectory

    function select_invariant(selection::String)
        if selection == "K1"
            iv = Ref(active_trajectory.t1)
        elseif selection == "K2"
            iv = Ref(active_trajectory.t2)
        elseif selection == "K3"
            iv = Ref(active_trajectory.t3)
        else
            error("Invalid invariant selected")
        end

        return iv
    end

    rel_t_to_idx::Dict{Transition,Int} = Dict(reverse.(collect(enumerate(transitionSequence))))
    h_cutoff::Observable{Float32} = Observable(Float32(0.0))

    rm = view(dm, clustering.order, clustering.order)
    cluster_data = ClusterData(clustering, transitionSequence, rm)
    cluster_info = @lift begin
        return ClusterInfo(cluster_data, transitionSequence, $h_cutoff)
    end

    init_alignment = (!isnothing(align_with) && align_with in keys(alignments)) ? align_with : first(keys(alignments))
    selected_alignment = Observable(init_alignment)

    # should be fine, seems off-center because abs(volMin) != abs(volMax)
    lowmap = reverse(resample_cmap(:RdPu_3, 50;
        alpha=([(0.0):0.02:(0.99);] ./ 0.75) .^ 2))
    himap = resample_cmap(:greens, 50;
        alpha=([(0.0):0.02:(0.99);] ./ 0.75) .^ 2)
    t3map = vcat(lowmap, himap)

    lowmap = reverse(resample_cmap(:RdPu_3, 50;
        alpha=([(0.0):0.02:(0.99);] ./ 0.3) .^ 2))
    himap = resample_cmap(:greens, 50;
        alpha=([(0.0):0.02:(0.99);] ./ 0.3) .^ 2)
    t1map = vcat(lowmap, himap)

    volume_cmaps = Dict("K1" => t1map,
        "K2" => resample_cmap(:matter, 100;
            alpha=([0:0.01:0.99;] ./ 0.05) .^ 2),
        "K3" => t3map)

    function get_invariant_range(x)
        iv = select_invariant(x)
        vals = values(iv[])
        absInvMin = abs(minimum(minimum.(vals)))
        absInvMax = abs(maximum(maximum.(vals)))

        # not an ideal solution but it works
        r = max(absInvMin, absInvMax)
        return (-r, r)
    end

    invariantRanges = Dict("K1" => get_invariant_range("K1"),
        "K2" => get_invariant_range("K2"),
        "K3" => get_invariant_range("K3"))

    # https://docs.julialang.org/en/v1.12-dev/manual/performance-tips/#man-performance-captured
    # convenience function to avoid passing around all the data
    function calc_alignment(ts::AbstractArray{Transition})::Tuple{Transition,Dict{Transition,Tuple{Matrix{Float32},Bool}}}
        res = let rel_t_to_idx = rel_t_to_idx,
            alignments = alignments,
            dm = dm,
            alignedPositionsMatrices = alignedPositionsMatrices,
            selected_alignment = selected_alignment

            if !isempty(ts)
                ts_idx = map(x -> rel_t_to_idx[x], ts)
                features = alignments[selected_alignment[]]

                dist_sum = map(x -> sum(view(dm, x, ts_idx)), ts_idx)
                ref_t_idx = argmin(dist_sum)

                ref_t = ts[ref_t_idx]
                return ref_t, calculate_alignment(ref_t, ts, alignedPositionsMatrices, features)
            else
                return (1, 1), Dict{Transition,Tuple{Matrix{Float32},Bool}}()
            end
        end
        return res
    end

    # did this to avoid drilling down and passing parameters constantly
    #:linear_wcmr_100_45_c42_n256 
    atom_cmap = resample_cmap(:linear_bmy_10_95_c71_n256, 100,
        alpha=range(; start=0.05, stop=1.0, length=100))
    binary_atomcmap = resample_cmap(:redsblues, 3,
        alpha=[1.0, 0.1, 1.0])

    function get_atom_cmap(s)
        if occursin("signed", to_value(s))
            return binary_atomcmap
        else
            return atom_cmap
        end
    end

    function render_atom_view(scene::Makie.Scene,
        transition::Transition,
        selected_scalar::Observable{String},
        time::Observable{<:AbstractFloat},
        flip::Bool=false)

        let scalars = scalars, scalar_ranges = scalar_ranges, atom_cmap = atom_cmap, alignedPositionsMatrices = alignedPositionsMatrices
            #ap = flip ? reverse(alignedPositionsMatrices[transition]) : alignedPositionsMatrices[transition]
            ap = alignedPositionsMatrices[transition]
            simple_atom_view!(scene,
                ap,
                lift(x -> scalars[x][transition], selected_scalar),
                lift(x -> scalar_ranges[x], selected_scalar),
                lift(x -> get_atom_cmap(x), selected_scalar),
                time
            )
        end
    end

    function render_superquadrics_view(scene::Makie.Scene, transition::Transition, si::Observable{String}, flip::Bool=false)
        let alignedPositionsMatrices = alignedPositionsMatrices,
            stretchedPrincipalAxes = stretchedPrincipalAxes

            t_ap = alignedPositionsMatrices[transition]
            idx = 1#flip ? 2 : 1
            points = Point3f.(eachrow(t_ap[idx]))
            # do the invariant values need to be flipped as well?
            spa = stretchedPrincipalAxes[transition]
            colors = @lift view(select_invariant($si)[][transition], eachindex(points))

            # both fns allocate a bunch of space
            superquadrics_view!(scene,
                points,
                colors,
                spa,
                lift(x -> volume_cmaps[x], si),
                lift(x -> invariantRanges[x], si);
                resolution=sq_resolution
            )
        end
    end

    function render_movement_view_ts(scene::Makie.Scene,
        ts::AbstractArray{Transition},
        time::Observable{Float32},
        alignment::Dict{Transition,Tuple{Matrix{Float32},Bool}},
        correlationThreshold::Observable{<:AbstractFloat})

        res = let alignedPositionsMatrices = alignedPositionsMatrices,
            cluster_data = cluster_data,
            kernelWidth = 1.0

            posValsTup = map(t -> apply_alignment(alignment[t], alignedPositionsMatrices[t]), ts)
            distances, t_to_mtx, mtx_to_t = get_local_matrix(cluster_data, ts)
            R = kmedoids(distances, 1)
            representativeIdx = first(R.medoids)

            positions = [Point3f.(eachrow(p)) for p in first.(posValsTup)]
            refPositions = positions[representativeIdx] # chose the median in the future
            velocities = last.(posValsTup) .- first.(posValsTup)

            vd = fill(Point3f(0.0, 0.0, 0.0), length(refPositions))
            correlationMeasure = zeros(Float32, length(refPositions))

            num_neighbors = 50
            clusterKd = KDTree.(positions)
            for pId in eachindex(refPositions)
                vectorList = fill(Point3f(0.0, 0.0, 0.0), length(clusterKd))
                for t in eachindex(clusterKd)
                    knn, dists = NearestNeighbors.knn(clusterKd[t], refPositions[pId], num_neighbors)
                    pos = view(positions[t], knn, :)
                    k = ((2pi)^(3 / 2) * kernelWidth[]^3)
                    kf = kernelFunction.(Ref(refPositions[pId]), pos, kernelWidth[])

                    uValue = sum(k * kf .* view(velocities[t], knn, 1))
                    vValue = sum(k * kf .* view(velocities[t], knn, 2))
                    wValue = sum(k * kf .* view(velocities[t], knn, 3))

                    vd[pId] += Point3f(uValue, vValue, wValue) * (1.0 / length(clusterKd))
                    vectorList[t] = Point3f(uValue, vValue, wValue)
                end
                meanV = mean(vectorList)
                dotmV = dot(meanV, meanV)
                for v in vectorList
                    correlationMeasure[pId] += (dot(meanV, v)) / (dotmV + dot(v, v))
                end
                correlationMeasure[pId] *= 1.0 / length(vectorList)
                correlationMeasure[pId] += 0.5
            end

            inits = first.(posValsTup)[representativeIdx]
            fins = last.(posValsTup)[representativeIdx]

            plt = simple_arrow_view!(scene,
                (inits, fins),
                time,
                CLUSTER_CONSENSUS_COLORMAP,
                vd,
                correlationMeasure,
                correlationThreshold)
            return plt
        end

        return res
    end

    function time_slider(figure::Makie.Figure,
        time::Observable{Float32}=Observable(Float32(0.0)))

        t_slider = Slider(figure, range=0.0:0.05:1.0, startvalue=time[])
        Makie.onany(t_slider.blockscene, t_slider.value) do x
            time[] = x
        end

        sg = hgrid!(Label(figure, "t", font=:italic),
            t_slider,
            Label(figure, lift(x -> string(x), time)))

        return time, sg
    end


    function render_menu(figure::Makie.Figure, scene_selector::Observable{String}=Observable("Atom"))
        return setup_menu(figure, SINGLE_TRANSITION_RENDER_OPTIONS, scene_selector; tellwidth=false)
    end

    scalar_opts = sort(collect(keys(scalars)))
    function scalar_menu(figure::Makie.Figure, scalar_selection::Observable{String}=Observable(first(scalar_opts)))
        return setup_menu(figure, scalar_opts, scalar_selection)
    end

    SQ_HELP = "K1: x -> -Inf indicates extension; x -> Inf dilation \nK2 - magnitude of distortion \nK3: x-> -1 indicates linear anisotropy (rods); x -> 1 planar anisotropy (disks)"
    function invariants_menu(figure::Makie.Figure,
        si::Observable{String}=Observable("K1"))
        _, invar_menu = setup_menu(figure, ["K1", "K2", "K3"], si)
        g = hgrid!(invar_menu, inline_image(figure, HELP_ICON, SQ_HELP; tellheight=false))
        colsize!(g, 2, Relative(0.125))
        return si, g
    end


    function correlation_slider(figure::Makie.Figure, correlationThreshold::Observable{Float32}=Observable(Float32(0.7)))
        c_slider = Slider(figure, range=0.0:0.01:1.0, startvalue=correlationThreshold[])
        Makie.onany(c_slider.blockscene, c_slider.value) do x
            correlationThreshold[] = x
        end

        sg = hgrid!(Label(figure, "Correlation", font=:italic),
            c_slider,
            Label(figure, lift(x -> string(x), correlationThreshold), tellwidth=false))

        return correlationThreshold, sg
    end

    function embed_colorbar(figure::Makie.Figure,
        render_selection::Observable{String},
        scalar_selection::Observable{String},
        invariant_selection::Observable{String}
    )
        scalar_range = lift(x -> scalar_ranges[x], scalar_selection)
        invariant_range = lift(x -> invariantRanges[x], invariant_selection)
        volume_cmap = lift(x -> volume_cmaps[x], invariant_selection)

        currentRange = Observable((0.0, 1.0))
        currentCMap = Observable(atom_cmap)

        cbar = Colorbar(figure,
            colorrange=currentRange,
            vertical=false,
            colormap=currentCMap)

        listener = Makie.onany(cbar.blockscene,
            render_selection,
            scalar_range,
            invariant_range,
            volume_cmap,
            update=true,
        ) do rs, sr, vr, vc

            if rs == "Superquadric"
                currentRange[] = vr
                currentCMap[] = vc
            else
                currentRange[] = sr
                currentCMap[] = atom_cmap
            end
        end

        return cbar, listener
    end

    # just pass this dictionary around and pass in the arguments it needs
    render_views::Dict{String,Function} = Dict{String,Function}(
        "Atom" => render_atom_view,
        "Superquadric" => render_superquadrics_view,
        "SMovement" => render_movement_view_ts)

    widgets::Dict{String,Function} = Dict{String,Function}(
        "Movement" => time_slider,
        "Render" => render_menu,
        "Invariant" => invariants_menu,
        "Scalar" => scalar_menu,
        "Colorbar" => embed_colorbar,
        "CorrThreshold" => correlation_slider)

    if !isnothing(bonds)
        bond_opts = sort(collect(keys(bonds)))

        function bonds_menu(figure::Makie.Figure, bond_selection::Observable{String}=Observable(first(bond_opts)))
            return setup_menu(figure, bond_opts, bond_selection)
        end

        function initial_final_toggle(fig::Makie.Figure, toggle::Observable{Bool}=Observable(false))
            t = Toggle(fig)
            on(t.active) do state
                toggle[] = state
            end
            return toggle, t
        end

        num_atoms = 147
        null_mat = Matrix{Float32}(zeros(num_atoms, num_atoms))
        function render_bonds(scene::Makie.Scene,
            transition::Transition,
            selected_bond::Observable{String},
            showFinal::Observable{Bool})

            let alignedPositionsMatrices = alignedPositionsMatrices, bonds = bonds
                ap = alignedPositionsMatrices[transition]
                s1, s2 = transition
                selBonds = Observable((null_mat, null_mat))
                Makie.onany(scene, selected_bond) do sb
                    bond_dict = bonds[sb]
                    selBonds[] = (get(bond_dict, s1, null_mat), get(bond_dict, s2, null_mat))
                end
                bondview!(scene, ap, selBonds, showFinal)
            end
        end

        widgets["BondToggle"] = initial_final_toggle
        widgets["Bonds"] = bonds_menu
        render_views["Bonds"] = render_bonds
    end

    calculators::Dict{String,Function} = Dict{String,Function}("Alignment" => calc_alignment)
    settings_window = build_settings_menu(selected_alignment, collect(keys(alignments)))

    final_memory = Sys.free_memory() / 2^20

    @debug init_memory, final_memory
    window, cleanup = selection_window_ui(
        rel_t_to_idx,
        cluster_data,
        cluster_info,
        h_cutoff,
        settings_window,
        render_views,
        widgets,
        selected_dm_name,
        calculators,
        name,
        dataPath
    )

    # create inspector after render to avoid bugs
    ds = DataInspector(window)
    GLMakie.set_title!(screen, "LAMDA - Selection Window")
    display(screen, window)
    on(events(window).window_open) do e
        if !e
            @debug "Killing main"
            cleanup()
            dm = nothing
            transitionSequence = nothing
            clustering = nothing

            Observables.clear(cluster_info)

            for x in values(calculators)
                x = nothing
            end
            empty!(calculators)

            for x in values(widgets)
                x = nothing
            end
            empty!(widgets)

            for x in values(render_views)
                x = nothing
            end
            empty!(render_views)

            if !isnothing(window)
                empty!(window)
                empty!(settings_window)
                Makie.free(settings_window.scene)
                Makie.free(window.scene)

                select_invariant = nothing
                settings_window = nothing
                window = nothing
                ds = nothing
            end
        end
    end
end

function selection_window_ui(
    rel_t_to_idx::Dict{Transition,Index},
    cluster_data::ClusterData,
    cluster_info::Observable{ClusterInfo},
    h_cutoff::Observable{Float32},
    settings_window::Makie.Figure,
    render_views::Dict{String,Function},
    widgets::Dict{String,Function},
    matColLabel::String,
    calculators::Dict{String,Function},
    trajectory_name::String,
    dataPath::String;
    fig_size::Tuple{Integer,Integer}=(1920, 1080)
)
    @debug "Building selection window"
    sinit_memory = Sys.free_memory() / 2^20

    set_theme!(UI_THEME)

    window::Makie.Figure = Figure(size=fig_size)
    menu_bar = top_bar(window, "Selection Window", 2)

    init_transitions = Set{Transition}()
    selected_transitions = Observable{Set{Transition}}(init_transitions)
    hovered_transition = MaybeObservable{Transition}(nothing, ignore_equal_values=true)

    cluster_annotations::Observable{ClusterAnnotation} = Observable(ClusterAnnotations())

    init_clusters = Set{ClusterSet}()
    selected_clusters = Observable{Set{ClusterSet}}(init_clusters)

    # will complain about being passed "nothing" as a value if something isn't inside the set
    hovered_cluster = MaybeObservable{ClusterSet}(Set{UInt16}(), ignore_equal_values=true)

    # used to place transitions into scratchpad
    function on_transition_select(t::Transition)
        push!(selected_transitions[], t)
        notify(selected_transitions)
    end

    c2corr = Ref(Dict{ClusterSet,Float32}())
    function on_cluster_select(c::ClusterSet, corr::Float32)
        push!(selected_clusters[], c)
        c2corr[][c] = corr
        notify(selected_clusters)
    end

    # can be more clever
    function cw_on_up(cc::Observable{ClusterSet})
        parent = get_parent(cluster_data, cc[])
        if parent != cc[]
            cc[] = parent
        end
    end

    function switch_cluster(x::ClusterSet, cc::Observable{ClusterSet})
        cc[] = x
    end

    function on_show_cluster_click(clusters::ClusterSet)
        init_memory = Sys.free_memory() / 2^20

        cc = Observable(clusters, ignore_equal_values=true)
        w, cluster_cleanup = build_cluster_window(
            cc,
            rel_t_to_idx,
            Ref(cluster_data),
            render_views,
            widgets,
            calculators,
            on_transition_select,
            hovered_transition,
            hovered_cluster,
            cluster_annotations,
            on_cluster_select=on_cluster_select,
            create_window=(x) -> on_show_cluster_click(x),
            on_up=cw_on_up,
            switch_cluster=switch_cluster,
        )
        s = GLMakie.Screen(title="LAMDA - Cluster Window")
        display(s, w)

        Makie.onany(w.scene, cc, cluster_annotations, update=true) do c, ca
            title = "LAMDA - Cluster $(str_limit(get_val(ca, "titles", c), len=25))"
            GLMakie.set_title!(s, title)
        end


        # create inspector after render to avoid bugs
        ds = DataInspector(w)
        Makie.onany(window.scene, events(w).window_open) do e
            if !e
                @debug "Clear outside cluster window"
                if !isnothing(cluster_cleanup)
                    cluster_cleanup()
                    cluster_cleanup = nothing
                    Observables.clear(cc)
                    cc = nothing
                end
                GC.gc(true)
                ds = nothing
                empty!(w)
                Makie.free(w.scene)
                final_memory = Sys.free_memory() / 2^20
                @debug init_memory, final_memory
            end
        end
    end

    dGrid = GridLayout()
    window[2:3, 1] = dGrid

    graph_ax = Axis(dGrid[1, 1], backgroundcolor=:transparent, title=matColLabel)
    deregister_interaction!(graph_ax, :rectanglezoom)
    hidexdecorations!(graph_ax)

    hm_ax = Axis(dGrid[2, 1], backgroundcolor=:transparent)

    deregister_interaction!(hm_ax, :rectanglezoom)
    hidedecorations!(hm_ax)

    function update_cutoff(x::Float64)
        # reset hovered_cluster to avoid crashing
        hovered_cluster.val = nothing
        notify(hovered_cluster)

        h_cutoff[] = x
    end

    dendrogram!(graph_ax,
        cluster_data,
        hovered_cluster,
        cluster_annotations;
        cutoff=h_cutoff,
        on_click=on_show_cluster_click,
        on_cutoff_line_drag=update_cutoff
    )

    heatmap!(hm_ax, cluster_data.matrix,
        colorrange=cluster_data.m_extrema,
        colormap=DISTANCE_MATRIX_COLORMAP)

    hm_m_events = addmouseevents!(hm_ax.scene)
    onmouseleftdown(hm_m_events) do e
        plot, _ = pick(hm_ax)
        if !isnothing(plot)
            ord = cluster_data.mtx_to_t
            xy = mouseposition(hm_ax)
            i, j = Int.(round.(xy))
            t1 = ord[i]
            t2 = ord[j]
            push!(selected_transitions[], t1)
            if t1 != t2
                push!(selected_transitions[], t2)
            end

            notify(selected_transitions)
        end
    end

    function calc_cluster_bounding_box(hc::Maybe{ClusterSet}, cd::ClusterData)::Maybe{Wireframe{Tuple{GeometryBasics.HyperRectangle{2,Float64}}}}
        if !isnothing(hc)
            ts = get_transitions(cd, hc)
            m_idx = map(x -> cd.t_to_mtx[x], ts)

            lo = minimum(m_idx)
            hi = maximum(m_idx)

            return draw_bbox_pixel_space!(hm_ax.scene, lo, hi; color=cd.colors[hc], linewidth=1)
        end
        return nothing
    end

    rendered_clusters = []
    Makie.onany(window.scene, h_cutoff, update=true) do _
        foreach(x -> delete!(parent_scene(x), x), rendered_clusters)
        for c in values(cluster_info[].a2c)
            p = calc_cluster_bounding_box(Set(c), cluster_data)
            if !isnothing(p)
                push!(rendered_clusters, p)
            end
        end
    end

    # https://github.com/MakieOrg/Makie.jl/blob/master/src/interaction/inspector.jl
    last_bBox::Maybe{Wireframe{Tuple{GeometryBasics.HyperRectangle{2,Float64}}}} = draw_bbox_pixel_space!(hm_ax.scene, 0, 0; linewidth=1)
    Makie.onany(window.scene, hovered_cluster) do hc
        if !isnothing(last_bBox)
            delete!(parent_scene(last_bBox), last_bBox)
        end
        last_bBox = calc_cluster_bounding_box(hc, cluster_data)
    end

    settings_btn = Button(window, label="Settings", halign=:right)
    screen = nothing
    on(settings_btn.clicks) do n
        # n has how many times the button's been clicked
        if isnothing(screen)
            screen = GLMakie.Screen(title="LAMDA - Settings")
            display(screen, settings_window)
        else
            close(screen)
            screen = nothing
        end
    end

    export_btn = Button(window, label="Export", halign=:right)
    export_menu = Menu(window, options=["Scratchpad", "Cutoff", "All"], default="Scratchpad", tellwidth=false, halign=:right)


    menu_bar[1, 3] = export_btn
    menu_bar[1, 4] = export_menu
    leaves_only_cb = Toggle(menu_bar[1, 5], active=false)
    menu_bar[1, 6] = Label(window, "Leaves only", halign=:left)
    menu_bar[1, 7] = settings_btn

    dGrid[3, 1] = Colorbar(window,
        vertical=false,
        colorrange=cluster_data.m_extrema,
        colormap=DISTANCE_MATRIX_COLORMAP)

    linkxaxes!(hm_ax, graph_ax)

    tGrid = GridLayout()
    scg = GridLayout()

    window[2:3, 2] = vgrid!(tGrid, scg)

    render_selection, scratchpad_render_menu = widgets["Render"](window)
    scalar_selection, scalar_menu = widgets["Scalar"](window)
    invariant_selection = Observable("K1")

    time, scratchpad_t_slider = widgets["Movement"](window)
    sc_cbar, cbar_listeners = widgets["Colorbar"](window, render_selection, scalar_selection, invariant_selection)

    ax,
    scratchpad_contents,
    scratchpad_cleanup,
    scratchpad_hovered = scratchpad!(
        window,
        tGrid[1, 1],
        selected_transitions,
        cluster_info,
        cluster_data,
        render_views,
        render_selection,
        scalar_selection,
        invariant_selection,
        time,
        selected_clusters,
        rel_t_to_idx,
        calculators,
        cluster_annotations,
        c2corr;
        hovered_cluster=hovered_cluster,
        hovered=hovered_transition,
        on_click=on_show_cluster_click)

    SCRATCHPAD_HELP = "MMB and drag - move object\nRMB on object - delete\nLMB and drag - create visual group\nRMB and drag on empty space - pan camera\nDouble LMB - create text annotation; hold T to make the annotation a title\nDouble LMB on object - open cluster window"
    help_icon(window, tGrid[1, 1], SCRATCHPAD_HELP)
    tooltip_ax(tGrid[1, 1], scratchpad_hovered;
        valign=1.00,
        halign=0.01)
    on(export_btn.clicks) do n
        # export all clusters on screen
        ep = relative_path("export")
        if !isdir(ep)
            mkdir(ep)
        else
            @warn "Removing export folder to create new one"
            rm(ep, force=true, recursive=true)
            mkdir(ep)
        end

        render_fn = (x, y) ->
            render_views["Atom"](
                x,
                y,
                scalar_selection,
                Observable(Float32(0.0)),
                false
            )

        if export_menu.selection[] == "All"
            @info "Beginning full export"
            @time export_all(trajectory_name,
                cluster_info[],
                cluster_data,
                cluster_annotations[],
                ep,
                dataPath,
                render_fn;
                overwrite=true,
                leaves_only=leaves_only_cb.active[],
                use_cutoff=false)
        elseif export_menu.selection[] == "Cutoff"
            @time export_all(trajectory_name,
                cluster_info[],
                cluster_data,
                cluster_annotations[],
                ep,
                dataPath,
                render_fn;
                overwrite=true,
                leaves_only=leaves_only_cb.active[],
                use_cutoff=true)
        else
            @time export_scratchpad(scratchpad_contents(), cluster_data, ep, dataPath)
        end

        # thought we could do pdfs?
        rp = joinpath(ep, "report.png")
        save(rp, ax.scene)
        @info "Export finished."
    end

    scg[1, 1] = scratchpad_render_menu
    g = GridLayout()
    g[1, 1] = scalar_menu
    g[1, 2] = scratchpad_t_slider
    scg[1, 2] = g

    Makie.onany(window.scene, render_selection) do x
        clear_layout(g)
        if x == "Atom"
            _, m = widgets["Scalar"](window, scalar_selection)
            _, ts = widgets["Movement"](window, time)
            g[1, 1] = m
            g[1, 2] = ts
        else
            _, m = widgets["Invariant"](window, invariant_selection)
            g[1, 1:2] = m
        end
    end

    scg[2, 1:2] = sc_cbar
    sfinal_memory = Sys.free_memory() / 2^20
    @debug "Finished selection window"
    @debug sinit_memory, sfinal_memory

    cleanup = function ()
        @debug "Killing selection"
        scratchpad_cleanup()
        on_transition_select = nothing
        on_cluster_select = nothing
        cw_on_up = nothing
        switch_cluster = nothing
        on_show_cluster_click = nothing
        update_cutoff = nothing
        calc_cluster_bounding_box = nothing

        if !isnothing(ax)
            empty!(ax)
            Makie.free(ax.scene)
        end

        clear_listener_list(cbar_listeners)
    end
    return window, cleanup
end
