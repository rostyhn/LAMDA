function alignAtomPositions(xp::Matrix, x::Matrix)::Matrix
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

function get_data_folders(path)
    dirs = filter!(x -> check_directory_format(x),
        filter!(x -> isdir(x), readdir(path, join=true)))

    if length(dirs) == 0
        return error("No valid folders in data folder.")
    end

    return dirs
end

function readDistanceMatrixFolder(folder)
    dms = Dict()
    for dmf in readdir(folder, join=true)
        # each distance matrix should be in a folder with the matrix
        dm_name = basename(dmf)
        if isdir(dmf)
            dm_path = joinpath(dmf, "dm.pickle")
            if isfile(dm_path)
                dms[dm_name] = Matrix{Float32}(Pickle.npyload(dm_path))
            else
                println("$dm_name not loaded.")
            end
        end
    end
    return dms
end

function read_volume_cache(key)
    h = hash(key)
    rootPath = dirname(dirname(@__FILE__))
    cachePath = joinpath(rootPath, "cache")
    cache_file = joinpath(cachePath, "$(h).jdl2")

    result = Nothing
    if isdir(cachePath) && cache_file in readdir(cachePath, join=true)
        println("Loading $(key) from $(basename(cache_file))")
        result = JLD2.jldopen(cache_file) do file
            file["volume_range"]
        end
    end
    return result
end

function save_volume_cache(key, volume_range, dimensions, absVolMin)
    h = hash(key)
    rootPath = dirname(dirname(@__FILE__))
    cachePath = joinpath(rootPath, "cache")
    cache_file = joinpath(cachePath, "$(h).jdl2")

    println("Saving $(key) as $(basename(cache_file))")
    JLD2.jldsave("$(cache_file)"; volume_range, dimensions, absVolMin)
end

function get_data_alt(trajectory_name)
    rootPath = dirname(dirname(@__FILE__))
    dataPath = joinpath(rootPath, "data")
    cachePath = joinpath(rootPath, "cache")

    if isdir(dataPath)
        trajectories = Dict(map(x -> (basename(x), x), get_data_folders(dataPath)))

        if trajectory_name in keys(trajectories)
            t = trajectories[trajectory_name]

            cache_file = joinpath(cachePath, "$(trajectory_name).jdl2")
            transitions_pickle = joinpath(t, "transitions.pickle")
            transitions = Vector{Transition}(Pickle.npyload(transitions_pickle))

            ase_pickle = joinpath(t, "ase_dict.pickle")

            distances_pickle = joinpath(t, "distances.pickle")
            connectivity_pickle = joinpath(t, "connectivity.pickle")
            alignedPositions_pickle = joinpath(t, "aligned_positions.pickle")
            data_pickles = [distances_pickle, connectivity_pickle, alignedPositions_pickle]

            # if any do not exist, compute them in python before continuing
            if any(x -> !isfile(x), data_pickles)
                @show "Processing ASE data..."
                @pyinclude(joinpath(dirname(@__FILE__), "ase_processing.py"))
                py"process_dataset"(transitions, ase_pickle, t)
            end

            py"""
            import pickle
            def load_pickle(fpath):
                with open(fpath, "rb") as f:
                    data = pickle.load(f)
                return data
            """
            load_pickle = py"load_pickle"

            # seems like Pickle.jl fails here
            distanceMatrices = Dict{State,Matrix{Float32}}(load_pickle(distances_pickle))
            connectivity = Dict{State,Matrix{Float32}}(load_pickle(connectivity_pickle)) # i,j == 1 iff atoms i,j are connected 
            rawAlignedPositionsMatrices = Dict{Transition,Tuple{Matrix{Float32},Matrix{Float32}}}(load_pickle(alignedPositions_pickle))

            # convert to point3fs & generate kd trees
            println("Computing KDTrees.")
            alignedPositions = Dict{Transition,Tuple{Vector{Point3f},Vector{Point3f}}}()
            alignedPositionsMatrices = Dict{Transition,Tuple{Matrix{Float32},Matrix{Float32}}}()
            # https://github.com/KristofferC/NearestNeighbors.jl
            # can store kdTrees as indices only, relinking positions when needed
            # no need to cache this data, it computes really quickly
            kdTrees = Dict{Transition,Tuple{KDTree,KDTree}}()
            for (t, m) in rawAlignedPositionsMatrices
                # center atom positions first
                cm1 = mean(m[1], dims=1)
                cm2 = mean(m[2], dims=1)
                p1 = (m[1] .- cm1)
                p2 = (m[2] .- cm2)

                alignedPositionsMatrices[t] = (p1, p2)

                pp1, pp2 = (map(x -> Point3f(x), eachrow(p1)), map(x -> Point3f(x), eachrow(p2)))
                alignedPositions[t] = (pp1, pp2)
                kdTrees[t] = (KDTree(pp1), KDTree(pp2))
            end

            if isdir(cachePath) && cache_file in readdir(cachePath, join=true)
                println("Loading $(trajectory_name) from cache.")
                @time trajectory_data = JLD2.jldopen(cache_file) do file
                    Dict{Any,Any}(file["trajectory_data"])
                end
            else
                println("Calculating data for $(trajectory_name).")
                if !isdir(cachePath)
                    mkdir(cachePath)
                end

                println("Calculating transition invariants.")
                (t1, t2, t3, stretchedPrincipalAxes) =
                    computeTransitionInvariants(transitions, alignedPositionsMatrices, distanceMatrices)

                trajectory_data = Dict{Any,Any}("t1" => t1,
                    "t2" => t2,
                    "t3" => t3,
                    "stretchedPrincipalAxes" => stretchedPrincipalAxes)

                @time JLD2.jldsave("$(cache_file)"; trajectory_data,)
            end

            # no need to cache data that is already available
            trajectory_data["distanceMatrices"] = distanceMatrices
            trajectory_data["connectivity"] = connectivity
            trajectory_data["transitions"] = transitions
            trajectory_data["alignedPositionsMatrices"] = alignedPositionsMatrices
            trajectory_data["alignedPositions"] = alignedPositions
            trajectory_data["kdTrees"] = kdTrees
            trajectory_data["name"] = trajectory_name

            # can probably clean this up to use one generic function
            dmf = joinpath(t, "dms")
            if !isdir(dmf)
                return error("Distance matrix folder not found.")
            end

            dms = readDistanceMatrixFolder(dmf)
            if isempty(dms)
                return error("No distance matrices found.")
            end

            scalars = Dict()
            scalar_ranges = Dict()
            # load in scalars if present
            scalarf = joinpath(t, "scalars")
            if isdir(scalarf)
                println("Loading per-atom scalars...")
                for sf in readdir(scalarf, join=true)
                    fname, ext = splitext(sf)
                    if isfile(sf) && ext == ".pickle"
                        d = Dict{Transition,Array{Float32}}(Pickle.npyload(sf))
                        totExtrema = extrema.(values(d))
                        totMin = minimum(first.(totExtrema))
                        totMax = maximum(last.(totExtrema))
                        scalar_ranges[basename(fname)] = (totMin, totMax)
                        scalars[basename(fname)] = d
                    end
                end
            else
                println("No scalars folder found, ignoring.")
            end

            per_t_scalars = Dict()
            per_t_scalar_ranges = Dict()
            # load in scalars if present
            tscalarf = joinpath(t, "per_t_scalars")
            if isdir(tscalarf)
                println("Loading per-transition scalars...")
                for sf in readdir(tscalarf, join=true)
                    fname, ext = splitext(sf)
                    if isfile(sf) && ext == ".pickle"
                        d = Dict{Transition,Float32}(Pickle.npyload(sf))
                        per_t_scalars[basename(fname)] = d
                        per_t_scalar_ranges[basename(fname)] = extrema(collect(values(d)))
                    end
                end
            else
                println("No scalars folder found, ignoring.")
            end

            # load in alignment features
            alignmentf = joinpath(t, "alignment")
            alignments = Dict()

            if isdir(alignmentf)
                for af in readdir(alignmentf, join=true)
                    fname, ext = splitext(af)
                    if isfile(af) && ext == ".pickle"
                        alignment_name = basename(fname)
                        alignments[alignment_name] = Dict{Transition,Tuple{Matrix{Float32},Matrix{Float32}}}(Pickle.npyload(af))
                    end
                end
            else
                return error("Alignment folder not found.")
            end

            # TODO: check for correctness
            trajectory_data["alignments"] = alignments
            trajectory_data["scalars"] = scalars
            trajectory_data["scalar_ranges"] = scalar_ranges
            trajectory_data["per_t_scalars"] = per_t_scalars
            trajectory_data["per_t_scalar_ranges"] = per_t_scalar_ranges
            trajectory_data["dms"] = dms
        else
            return error("Trajectory \"$(trajectory_name)\" not found in data folder.")
        end
    else
        return error("Data folder does not exist.")
    end

    return trajectory_data
end

function get_mmap_file(key)
    h = hash(key)
    rootPath = dirname(dirname(@__FILE__))
    cachePath = joinpath(rootPath, "cache")
    cache_file = joinpath(cachePath, "$(h).bin")
    return cache_file, isdir(cachePath) && cache_file in readdir(cachePath, join=true)
end

function export_cluster(trajectory_name, path, cluster, ca, cd, ci, t_list)
    cluster_name = get_val(ca, "titles", cluster)
    cluster_notes = get(ca["notes"], cluster, nothing)

    dname = filesafestr(cluster_name)
    cp = joinpath(path, dname)
    if !isdir(cp)
        mkdir(cp)
    end

    if !isnothing(cluster_notes)
        nf = joinpath(cp, "notes.txt")
        write(nf, cluster_notes)
    end

    # this will break export
    children = get_children(cd, cluster)
    if !isnothing(children) && all(map(x -> x in ci.clusters, collect(children)))
        lc, rc = children
        export_cluster(trajectory_name, cp, lc, ca, cd, ci, t_list)
        export_cluster(trajectory_name, cp, rc, ca, cd, ci, t_list)
    else
        # get children of cluster
        ts = get_transitions(t_list, cluster)
        dpath = get_ase_dict_path(trajectory_name)
        export_t = export_transitions()
        export_t(dpath, cp, ts)
    end
end

function get_ase_dict_path(trajectory_name)
    dir = dirname(dirname(@__FILE__))
    ddir = joinpath(dir, "data")
    # need to get name of trajectory
    tdpath = joinpath(ddir, trajectory_name)
    return joinpath(tdpath, "t_ase_dict.pickle")
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

function export_all(trajectory_name, ci::ClusterInfo, cd::ClusterData, t_list, ca, exportPath; overwrite=false)
    if !isdir(exportPath)
        mkdir(exportPath)
    else
        if overwrite
            rm(exportPath, force=true, recursive=true)
            mkdir(exportPath)
        end
    end

    root = get_root(cd)
    export_cluster(trajectory_name, exportPath, root, ca, cd, ci, t_list)

end
