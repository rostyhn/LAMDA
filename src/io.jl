export get_data_alt

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
    return "distances.pickle" in contents &&
           "transitions.pickle" in contents &&
           "connectivity.pickle" in contents &&
           "aligned_positions.pickle" in contents
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
        result = JLD2.jldopen(cache_file; compress=true) do file
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
    JLD2.jldsave("$(cache_file)", true; volume_range, dimensions, absVolMin)
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
            if isdir(cachePath) && cache_file in readdir(cachePath, join=true)
                println("Loading $(trajectory_name) from cache.")
                @time trajectory_data = JLD2.jldopen(cache_file; compress=true) do file
                    file["trajectory_data"]
                end
            else
                println("Calculating data for $(trajectory_name).")
                if !isdir(cachePath)
                    mkdir(cachePath)
                end

                distances_pickle = joinpath(t, "distances.pickle")
                connectivity_pickle = joinpath(t, "connectivity.pickle")
                transitions_pickle = joinpath(t, "transitions.pickle")
                alignedPositions_pickle = joinpath(t, "aligned_positions.pickle")

                distanceMatrices = Dict{Int16,Matrix{Float32}}(Pickle.npyload(distances_pickle))
                connectivity = Dict{Int16,Matrix{Float32}}(Pickle.npyload(connectivity_pickle)) # i,j == 1 iff atoms i,j are connected 
                transitions = Vector{Tuple{Int16,Int16}}(Pickle.npyload(transitions_pickle))
                alignedPositionsMatrices = Dict{Tuple{Int16,Int16},Tuple{Matrix{Float32},Matrix{Float32}}}(Pickle.npyload(alignedPositions_pickle))

                (t1, t2, t3, stretchedPrincipalAxes) =
                    computeTransitionInvariants(transitions, alignedPositionsMatrices, distanceMatrices)

                # converts into array of Point3fs
                alignedAtomPositions = Dict{Tuple{Int16,Int16},Tuple{Vector{Point3f},Vector{Point3f}}}()
                for (t, aligned) in alignedPositionsMatrices
                    p1, p2 = aligned
                    alignedAtomPositions[t] = (map(x -> Point3f(x), eachrow(p1)), map(x -> Point3f(x), eachrow(p2)))
                end

                # https://github.com/KristofferC/NearestNeighbors.jl
                # can store kdTrees as indices only, relinking positions when needed
                println("Computing KDTrees.")
                stateKDTree = Dict{Tuple{Int16,Int16},Tuple{KDTree,KDTree}}()
                @time for (t, aligned) in alignedAtomPositions
                    p1, p2 = aligned
                    stateKDTree[t] = (KDTree(p1), KDTree(p2))
                end

                trajectory_data = Dict("distanceMatrices" => distanceMatrices,
                    "alignedPositions" => alignedAtomPositions,
                    "alignedPositionsMatrices" => alignedPositionsMatrices,
                    "connectivity" => connectivity,
                    "transitions" => transitions,
                    "t1" => t1,
                    "t2" => t2,
                    "t3" => t3,
                    "kdTree" => stateKDTree,
                    "stretchedPrincipalAxes" => stretchedPrincipalAxes)

                @time JLD2.jldsave("$(cache_file)", true; trajectory_data,)
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

            scalars = Dict()
            # load in scalars if present
            scalarf = joinpath(t, "scalars")
            if isdir(scalarf)
                println("Loading scalars...")
                for sf in readdir(scalarf, join=true)
                    scalar_name = basename(sf)
                    if isfile(sf)
                        scalars[scalar_name] = Dict{Tuple{Int16,Int16},Tuple{Array{Float32},Array{Float32}}}(Pickle.npyload(sf))
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
                    alignment_name = basename(af)
                    if isfile(af)
                        alignments[alignment_name] = Dict{Tuple{Int16,Int16},Matrix{Float32}}(Pickle.npyload(af))
                    end
                end
            else
                return error("Alignment folder not found.")
            end

            # TODO: check for correctness
            trajectory_data["alignments"] = alignments
            trajectory_data["scalars"] = scalars
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
