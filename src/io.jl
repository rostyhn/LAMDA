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

function readDistanceMatrixFolder(folder)::Dict{String,Matrix{Float32}}
    dms = Dict{String,Matrix{Float32}}()
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
        trajectories = Dict{String,String}(map(x -> (basename(x), x), get_data_folders(dataPath)))

        if trajectory_name in keys(trajectories)
            t = trajectories[trajectory_name]

            cache_file = joinpath(cachePath, "$(trajectory_name).jdl2")
            transitions_pickle = joinpath(t, "transitions.pickle")
            transitions = Vector{Transition}(Pickle.npyload(transitions_pickle))

            ase_pickle = joinpath(t, "ase_dict.pickle")

            distances_pickle = joinpath(t, "distances.pickle")
            alignedPositions_pickle = joinpath(t, "aligned_positions.pickle")
            data_pickles = [alignedPositions_pickle, distances_pickle]

            # if any do not exist, compute them in python before continuing
            if any(x -> !isfile(x), data_pickles)
                @show "Processing ASE data..."
                py"process_dataset"(transitions, ase_pickle, t)
            end

            load_pickle = py"load_pickle"o

            rawAlignedPositionsMatrices = pycall(load_pickle, PyDict{Transition,Tuple{Matrix{Float32},Matrix{Float32}}}, alignedPositions_pickle)

            # convert to point3fs & generate kd trees
            println("Computing KDTrees.")
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

                kdTrees[t] = (KDTree(p1; reorder=false), KDTree(p2; reorder=false))
            end
            rawAlignedPositionsMatrices = nothing

            if isdir(cachePath) && cache_file in readdir(cachePath, join=true)
                println("Loading $(trajectory_name) from cache.")
                @time invariants = JLD2.jldopen(cache_file) do file
                    Dict{String,Any}(file["trajectory_data"])
                end
            else
                println("Calculating data for $(trajectory_name).")
                if !isdir(cachePath)
                    mkdir(cachePath)
                end

                distanceMatrices = pycall(load_pickle, PyDict{State,Matrix{Float32}}, distances_pickle)
                println("Calculating transition invariants.")
                (t1, t2, t3, stretchedPrincipalAxes) =
                    computeTransitionInvariants(transitions, alignedPositionsMatrices, distanceMatrices)

                invariants = Dict{String,Any}("t1" => t1,
                    "t2" => t2,
                    "t3" => t3,
                    "stretchedPrincipalAxes" => stretchedPrincipalAxes)

                distanceMatrices = nothing
                @time JLD2.jldsave("$(cache_file)"; invariants,)
            end

            # can probably clean this up to use one generic function
            dmf = joinpath(t, "dms")
            if !isdir(dmf)
                return error("Distance matrix folder not found.")
            end

            dms = readDistanceMatrixFolder(dmf)
            if isempty(dms)
                return error("No distance matrices found.")
            end

            scalars = Dict{String,Dict{Transition,Array{Float32}}}()
            scalar_ranges = Dict{String,Tuple{Float32,Float32}}()
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

            # load in alignment features
            alignmentf = joinpath(t, "alignment")
            alignments = Dict{String,Dict{State,Matrix{Float32}}}()

            if isdir(alignmentf)
                for af in readdir(alignmentf, join=true)
                    fname, ext = splitext(af)
                    if isfile(af) && ext == ".pickle"
                        alignment_name = basename(fname)
                        alignments[alignment_name] = Dict{State,Matrix{Float32}}(Pickle.npyload(af))
                    end
                end
            else
                return error("Alignment folder not found.")
            end

            trajectory_data = Trajectory(name=trajectory_name, transitions=transitions,
                alignedPositionsMatrices=alignedPositionsMatrices,
                kdTrees=kdTrees,
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
