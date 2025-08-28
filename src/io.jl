function alignAtomPositions(xp::Matrix{AbstractFloat}, x::Matrix{AbstractFloat})::Matrix{AbstractFloat}
    #s2 changes s1 stays
    s = mean(x, dims=1)
    sp = mean(xp, dims=1)

    xs = x .- s
    xps = xp .- sp
    xx = transpose(xs) * xs
    xpx = transpose(xps) * xs

    xxi = inv(xx)
    R = xpx * xxi

    return transpose(R * transpose(xs)) .+ sp
end

function check_directory_format(dir)
    contents = readdir(dir)
    return "ase_dict.pickle" in contents &&
           "transitions.pickle" in contents
end

function readDistanceMatrixFolder(folder::String)
    dms = Dict{String,Matrix{Float32}}()
    for dmf in readdir(folder, join=true)
        dm_name = basename(dmf)
        fname, ext = splitext(dm_name)
        if isfile(dmf) && ext == ".pickle"
            try
                dms[fname] = Matrix{Float32}(Pickle.npyload(dmf))
            catch e
                @warn "Distance matrix $(basename) failed to load: $(e)."
            end
        end
    end
    return dms
end

function get_data_alt(dataPath::String, cachePath::String)::Trajectory
    t = basename(dataPath)
    @debug dataPath
    @debug cachePath

    if isdir(dataPath)
        if check_directory_format(dataPath)
            cache_file = joinpath(cachePath, "$(t).jdl2")
            @debug cache_file

            transitions_pickle = joinpath(dataPath, "transitions.pickle")
            transitions::Vector{Transition} = Vector{Transition}(Pickle.npyload(transitions_pickle))

            ase_pickle = joinpath(dataPath, "ase_dict.pickle")

            distances_pickle = joinpath(dataPath, "distances.pickle")
            alignedPositions_pickle = joinpath(dataPath, "aligned_positions.pickle")
            data_pickles = [alignedPositions_pickle, distances_pickle]

            # if any do not exist, compute them in python before continuing
            if any(x -> !isfile(x), data_pickles)
                @info "Processing ASE data..."
                py"process_dataset"(transitions, ase_pickle, t)
            end

            load_pickle = py"load_pickle"o

            rawAlignedPositionsMatrices = pycall(load_pickle,
                PyDict{Transition,Tuple{Matrix{Float32},Matrix{Float32}}},
                alignedPositions_pickle)

            # convert to point3fs & generate kd trees
            alignedPositionsMatrices = Dict{Transition,Tuple{Matrix{Float32},Matrix{Float32}}}()
            for (t::Transition, m::Tuple{Matrix{Float32},Matrix{Float32}}) in rawAlignedPositionsMatrices
                # center atom positions first
                cm1 = mean(m[1], dims=1)
                cm2 = mean(m[2], dims=1)
                p1 = (m[1] .- cm1)
                p2 = (m[2] .- cm2)

                alignedPositionsMatrices[t] = (p1, p2)
            end
            rawAlignedPositionsMatrices = nothing

            if isdir(cachePath) && cache_file in readdir(cachePath, join=true)
                @info "Loading $(t) from cache."
                invariants = JLD2.jldopen(cache_file) do file
                    Dict{String,Any}(file["invariants"])
                end
            else
                @info "Calculating data for $(t)."
                if !isdir(cachePath)
                    mkpath(cachePath)
                end

                distanceMatrices::PyDict{State,Matrix{Float32}} = pycall(load_pickle, PyDict{State,Matrix{Float32}}, distances_pickle)
                @info "Calculating transition invariants."
                (t1, t2, t3, stretchedPrincipalAxes) =
                    computeTransitionInvariants(transitions, alignedPositionsMatrices, distanceMatrices)

                invariants = Dict{String,Any}("t1" => t1,
                    "t2" => t2,
                    "t3" => t3,
                    "stretchedPrincipalAxes" => stretchedPrincipalAxes)

                JLD2.jldsave("$(cache_file)"; invariants,)
            end

            # can probably clean this up to use one generic function
            dmf = joinpath(dataPath, "dms")
            if !isdir(dmf)
                return error("Distance matrix folder not found in $(dataPath).")
            end

            dms = readDistanceMatrixFolder(dmf)
            if isempty(dms)
                return error("No distance matrices found.")
            end

            scalars = Dict{String,Dict{Transition,Array{Float32}}}()
            scalar_ranges = Dict{String,Tuple{Float32,Float32}}()
            # load in scalars if present
            scalarf = joinpath(dataPath, "scalars")
            if isdir(scalarf)
                println("Loading per-atom scalars...")
                for sf in readdir(scalarf, join=true)
                    fname, ext = splitext(sf)
                    if isfile(sf) && ext == ".pickle"
                        try
                            d = Dict{Transition,Array{Float32}}(Pickle.npyload(sf))
                            totExtrema::Vector{Tuple{Float32,Float32}} = extrema.(values(d))
                            totMin = minimum(first.(totExtrema))
                            totMax = maximum(last.(totExtrema))
                            scalar_ranges[basename(fname)] = (totMin, totMax)
                            scalars[basename(fname)] = d
                        catch e
                            @warn "Scalar $(sf) failed to load: $(e)"
                        end
                    end
                end
            else
                @warn "No scalars folder found in $(dataPath), ignoring."
            end

            # load in alignment features
            alignmentf = joinpath(dataPath, "alignment")
            alignments = Dict{String,Dict{State,Matrix{Float32}}}()

            if isdir(alignmentf)
                for af in readdir(alignmentf, join=true)
                    fname, ext = splitext(af)
                    if isfile(af) && ext == ".pickle"
                        alignment_name = basename(fname)
                        try
                            alignments[alignment_name] = Dict{State,Matrix{Float32}}(Pickle.npyload(af))
                        catch e
                            @warn "Alignment $(af) failed to load: $(e)"
                        end
                    end
                end
            else
                return error("Alignment folder not found in $(dataPath).")
            end

            trajectory_data = Trajectory(name=t,
                transitions=transitions,
                alignedPositionsMatrices=alignedPositionsMatrices,
                t1=invariants["t1"],
                t2=invariants["t2"],
                t3=invariants["t3"],
                stretchedPrincipalAxes=invariants["stretchedPrincipalAxes"],
                dms=dms,
                scalars=scalars,
                scalar_ranges=scalar_ranges,
                alignments=alignments,
                t_to_idx=Dict(reverse.(collect(enumerate(transitions)))))
            GC.gc(true)
        else
            return error("Supplied folder $(dataPath) does not contain transitions or state information.")
        end
    else
        return error("Data folder $(dataPath) does not exist.")
    end
    return trajectory_data
end

function get_mmap_file(key, cachePath::String)
    h = hash(key)
    cache_file = joinpath(cachePath, "$(h).bin")
    return cache_file, isdir(cachePath) && cache_file in readdir(cachePath, join=true)
end

function export_cluster(trajectory_name::String,
    dataPath::String,
    exportPath::String,
    cluster::ClusterSet,
    ca::ClusterAnnotation,
    cd::ClusterData,
    ci::ClusterInfo)

    cluster_name = get_val(ca, "titles", cluster)
    cluster_notes = get(ca["notes"], cluster, nothing)

    dname = filesafestr(cluster_name)
    cp = joinpath(exportPath, dname)
    if !isdir(cp)
        mkdir(cp)
    end

    if !isnothing(cluster_notes)
        nf = joinpath(cp, "notes.txt")
        write(nf, cluster_notes)
    end

    children = get_children(cd, cluster)
    if !isnothing(children) && all(map(x -> cd.heights[x] > ci.cutoff, collect(children)))
        lc, rc = children
        export_cluster(trajectory_name, dataPath, cp, lc, ca, cd, ci)
        export_cluster(trajectory_name, dataPath, cp, rc, ca, cd, ci)
    else
        ts = get_transitions(cd, cluster)
        dpath = joinpath(dataPath, "t_ase_dict.pickle")
        export_t = export_transitions()
        export_t(dpath, cp, ts)
    end
end

function export_transitions()
    py"""
    import pickle
    from ase.io import extxyz

    def export_transitions(dpath, cp, ts): 
        with open(dpath, "rb") as f:
            d = pickle.load(f)
        for t in ts:
            s1, s2 = t
            s1a, s2a = d[t]
            extxyz.write_extxyz(open(f"{cp}/%i-%i.xyz"%(s1,s2),'w'), [s1a,s2a], columns=['symbols', 'positions', 'tags'])
    """
    return py"export_transitions"
end

function export_all(trajectory_name::String,
    ci::ClusterInfo,
    cd::ClusterData,
    ca::ClusterAnnotation,
    exportPath::String,
    dataPath::String;
    overwrite=false)

    if !isdir(exportPath)
        mkdir(exportPath)
    else
        if overwrite
            rm(exportPath, force=true, recursive=true)
            mkdir(exportPath)
        end
    end

    export_cluster(trajectory_name, dataPath, exportPath, cd.root, ca, cd, ci)
end
