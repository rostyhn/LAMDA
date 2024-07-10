function getDataSets(
    stateDataPath::String,
    sequencePath::String,
    transitionLabelPath::String,
    bondWeightPath::String,
    connectivityPath::String,
    transitionDistanceMatrixPath::String,
    transitionSequencePath::String,
)::Dict{String,Dict}

    combinedData = Dict{String,Dict}() # combined set
    transitionInvariants1 = Dict{Tuple{Int,Int},Vector{Float64}}()
    transitionInvariants2 = Dict{Tuple{Int,Int},Vector{Float64}}()
    transitionInvariants3 = Dict{Tuple{Int,Int},Vector{Float64}}()
    atomPositions = Dict{Int,Matrix{Float64}}()
    distanceMatrices = Dict{Int,Matrix{Float64}}()
    transitionRefPositions = Dict{Tuple{Int,Int},Vector{Point3f}}()
    transitionLabels = Dict{Tuple{Int,Int},Int}()
    #eigenvalues = Dict{String, Vector{Vec3f}}
    stretchedPrincipalAxes = Dict{Tuple{Int,Int},Vector{Vector{Vec3f}}}()

    # check folders

    # need to load in transition sequence because we can't guarantee order
    # no idea if the underlying implementation of sets would yield the
    # same results in julia vs. python
    transitionDistanceMatrix = Pickle.npyload(transitionDistanceMatrixPath)
    transitionSequence = Vector{Tuple{Int,Int}}(Pickle.npyload(transitionSequencePath))
    transitionLabels = Dict{Tuple{Int,Int},Int}(Pickle.npyload(transitionLabelPath))

    sequence = readlines(sequencePath)
    sequenceHash = Base.hash(sequence)

    rootPath = dirname(dirname(@__FILE__))
    println("Root directory is: $(rootPath)")

    if isfile("$(rootPath)/cache/transitionInvariants1_$(sequenceHash).jld2")
        println("Found precomputed data, loading data...")

        @time atomPositions = JLD2.jldopen(
            "$(rootPath)/cache/atomPositions_$(sequenceHash).jld2";
            compress=true,
        ) do file
            file["atomPositions"]
        end

        @time transitionRefPositions = JLD2.jldopen(
            "$(rootPath)/cache/transitionRefPositions_$(sequenceHash).jld2";
            compress=true,
        ) do file
            file["transitionRefPositions"]
        end


        @time distanceMatrices = JLD2.jldopen(
            "$(rootPath)/cache/distanceMatrices_$(sequenceHash).jld2";
            compress=true,
        ) do file
            file["distanceMatrices"]
        end

        @time transitionInvariants1 = JLD2.jldopen(
            "$(rootPath)/cache/transitionInvariants1_$(sequenceHash).jld2";
            compress=true,
        ) do file
            file["transitionInvariants1"]
        end

        @time transitionInvariants2 = JLD2.jldopen(
            "$(rootPath)/cache/transitionInvariants2_$(sequenceHash).jld2";
            compress=true,
        ) do file
            file["transitionInvariants2"]
        end

        @time transitionInvariants3 = JLD2.jldopen(
            "$(rootPath)/cache/transitionInvariants3_$(sequenceHash).jld2";
            compress=true,
        ) do file
            file["transitionInvariants3"]
        end

        @time stretchedPrincipalAxes = JLD2.jldopen(
            "$(rootPath)/cache/stretchedPrincipalAxes_$(sequenceHash).jld2";
            compress=true,
        ) do file
            file["stretchedPrincipalAxes"]
        end

        @time bondWeights = JLD2.jldopen(
            "$(rootPath)/cache/bondWeights_$(sequenceHash).jld2";
            compress=true,
        ) do file
            file["bondWeights"]
        end

        @time connectivity = JLD2.jldopen(
            "$(rootPath)/cache/connectivity_$(sequenceHash).jld2";
            compress=true,
        ) do file
            file["connectivity"]
        end

        println("loading successful")
    else
        println("No precomputed data found! Computing now...")
        println("Reading dataset....")

        if !isdir("$(rootPath)/cache")
            mkdir("$(rootPath)/cache")
        end

        distanceMatrices = loadDistanceMatricesFromData(stateDataPath)
        atomPositions = loadAtomPositionsFromData(stateDataPath)
        bondWeights = loadBondWeightsFromData(bondWeightPath)
        atomPositions = alignAtomPositions(atomPositions |> keys |> first, atomPositions)
        # they're the same type so it should just work - we could probably abstract this out later
        connectivity = loadBondWeightsFromData(connectivityPath)

        (transitionRefPositions, transitionInvariants1, transitionInvariants2, transitionInvariants3, stretchedPrincipalAxes) =
            computeTransitionInvariants(transitionSequence, atomPositions, distanceMatrices)

        println("Storing  data.... $(rootPath)/cache/$(sequenceHash).jld2")

        @time JLD2.jldsave(
            "$(rootPath)/cache/atomPositions_$(sequenceHash).jld2",
            true;
            atomPositions,
        )
        @time JLD2.jldsave(
            "$(rootPath)/cache/transitionRefPositions_$(sequenceHash).jld2",
            true;
            transitionRefPositions,
        )
        @time JLD2.jldsave(
            "$(rootPath)/cache/distanceMatrices_$(sequenceHash).jld2",
            true;
            distanceMatrices,
        )
        @time JLD2.jldsave(
            "$(rootPath)/cache/transitionInvariants1_$(sequenceHash).jld2",
            true;
            transitionInvariants1,
        )
        @time JLD2.jldsave(
            "$(rootPath)/cache/transitionInvariants2_$(sequenceHash).jld2",
            true;
            transitionInvariants2,
        )
        @time JLD2.jldsave(
            "$(rootPath)/cache/transitionInvariants3_$(sequenceHash).jld2",
            true;
            transitionInvariants3,
        )
        @time JLD2.jldsave(
            "$(rootPath)/cache/stretchedPrincipalAxes_$(sequenceHash).jld2",
            true;
            stretchedPrincipalAxes,
        )

        @time JLD2.jldsave(
            "$(rootPath)/cache/bondWeights_$(sequenceHash).jld2",
            true;
            bondWeights,
        )

        @time JLD2.jldsave(
            "$(rootPath)/cache/connectivity_$(sequenceHash).jld2",
            true;
            connectivity,
        )
        println("Storing successfull")
    end

    combinedData = Dict{String,Dict}() # combined set    

    combinedData["transitionInvariants1"] = transitionInvariants1
    combinedData["transitionInvariants2"] = transitionInvariants2
    combinedData["transitionInvariants3"] = transitionInvariants3
    combinedData["atomPositions"] = atomPositions
    combinedData["distanceMatrices"] = distanceMatrices
    combinedData["transitionRefPositions"] = transitionRefPositions

    combinedData["bondWeights"] = bondWeights
    combinedData["connectivity"] = connectivity
    combinedData["transitionLabels"] = transitionLabels
    combinedData["stretchedPrincipalAxes"] = stretchedPrincipalAxes

    # julia doesn't like mixed types in dictionaries...
    combinedData["transitionDistanceMatrix"] = Dict("matrix" => transitionDistanceMatrix)
    combinedData["transitionSequence"] = Dict("sequence" => transitionSequence)

    return combinedData
end


function loadAtomPositionsFromData(stateDataPath::String)::Dict{Int,Matrix{Float64}}
    stateFiles = readdir(stateDataPath)
    filter!(e -> e ≠ ".DS_Store", stateFiles) # MacOS weirdness...
    filter!(e -> e ≠ "seq.txt", stateFiles) # filter sequence
    filter!(e -> !occursin("distance_matrix", e), stateFiles) # filter distances

    @show length(stateFiles)

    positionData = Dict{Int,Matrix}()

    addStateToDict!.(Ref(positionData), Ref(stateDataPath), stateFiles)

    return positionData
end


function loadDistanceMatricesFromData(stateDataPath::String)::Dict{Int,Matrix{Float64}}
    stateFiles = readdir(stateDataPath)
    filter!(e -> e ≠ ".DS_Store", stateFiles) # MacOS weirdness...
    filter!(e -> e ≠ "seq.txt", stateFiles) # filter sequence
    filter!(e -> !occursin("positions", e), stateFiles) # filter positions

    @show length(stateFiles)

    distanceMatrices = Dict{Int,Matrix}()

    addStateToDict!.(Ref(distanceMatrices), Ref(stateDataPath), stateFiles)

    return distanceMatrices
end

function loadBondWeightsFromData(path::String)::Dict{Int,Matrix{Float64}}
    bw = Pickle.npyload(open(path))
    # for now we assume that it's a dense representation
    f((k, v)) = k => Matrix{Float64}(v)

    return Dict(Iterators.map(f, pairs(bw)))
end

function addPositionsToDict!(dict::Dict{Int,Matrix}, pathToFile::String, fileName::String)
    stateId = SubString(fileName, 1:((findfirst("_", fileName)|>first)-1))
    dict[stateId] = Pickle.npyload(open(pathToFile * fileName))
end

function addStateToDict!(dict::Dict{Int,Matrix}, pathToFile::String, fileName::String)
    stateId = parse(Int, SubString(fileName, 1:((findfirst("_", fileName)|>first)-1)))
    dict[stateId] = Pickle.npyload(open(pathToFile * fileName))
end

function getSequence(sequencePath::String)
    return readlines(sequencePath)
end

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
           "positions.pickle" in contents
end

function get_data_folders(path)
    dirs = filter!(x -> check_directory_format(x),
        filter!(x -> isdir(x), readdir(path, join=true)))

    if length(dirs) == 0
        return error("No valid folders in data folder.")
    end

    return dirs
end

function get_data_alt()
    rootPath = dirname(dirname(@__FILE__))
    dataPath = joinpath(rootPath, "data")
    cachePath = joinpath(rootPath, "cache")

    current_active = Nothing
    all_data = Dict()
    if isdir(dataPath)
        trajectories = get_data_folders(dataPath)
        for t in trajectories
            trajectory_name = basename(t)
            cache_file = joinpath(cachePath, "$(trajectory_name).jdl2")
            if isdir(cachePath) && cache_file in readdir(cachePath, join=true)
                println("Loading $(trajectory_name) from cache.")
                @time trajectory_data = JLD2.jldopen(cache_file; compress=true) do file
                    file["trajectory_data"]
                end
            else
                println("Calculating data for $(trajectory_name).")

                distances_pickle = joinpath(t, "distances.pickle")
                positions_pickle = joinpath(t, "positions.pickle")
                connectivity_pickle = joinpath(t, "connectivity.pickle")
                transitions_pickle = joinpath(t, "transitions.pickle")


                distanceMatrices = Dict{Int,Matrix}(Pickle.npyload(distances_pickle))
                positions = Dict{Int,Matrix}(Pickle.npyload(positions_pickle))
                connectivity = Dict{Int,Matrix}(Pickle.npyload(connectivity_pickle))
                transitions = Set{Tuple{Int,Int}}(Pickle.npyload(transitions_pickle))

                # stores aligned version of s2 for each transition
                alignedS2Positions = Dict{Tuple{Int,Int},Matrix}()
                for t in transitions
                    s1, s2 = t
                    p1 = positions[s1]
                    p2 = positions[s2]
                    alignedS2Positions[t] = alignAtomPositions(p1, p2)
                end

                (t1, t2, t3, stretchedPrincipalAxes) =
                    computeTransitionInvariants(transitions, positions, alignedS2Positions, distanceMatrices)

                # converts into array of Point3fs
                atomPositions = Dict{Int,Vector{Point3f}}()
                for (stateID, val) in positions
                    atomPositions[stateID] = map(x -> Point3f(x), eachrow(val))
                end

                println("Computing KDTrees.")
                stateKDTree = Dict{Int,KDTree}()
                @time for (stateID, val) in atomPositions
                    stateKDTree[stateID] = KDTree(val)
                end


                trajectory_data = Dict("distanceMatrices" => distanceMatrices,
                    "positions" => atomPositions,
                    "positionMatrices" => positions,
                    "connectivity" => connectivity,
                    "transitions" => transitions,
                    "alignedS2Positions" => alignedS2Positions,
                    "t1" => t1,
                    "t2" => t2,
                    "t3" => t3,
                    "kdTree" => stateKDTree,
                    "stretchedPrincipalAxes" => stretchedPrincipalAxes)
                @time JLD2.jldsave("$(cache_file)", true; trajectory_data,)
            end
            all_data[trajectory_name] = trajectory_data
            current_active = trajectory_name
        end
    else
        return error("Data folder does not exist.")
    end

    # current_active is the name of the last trajectory it read
    return (all_data, current_active)
end
