function getDataSets(
    stateDataPath::String,
    sqeuencePath::String,
    transitionLabelPath::String,
    bondWeightPath::String,
)::Dict{String,Dict}

    combinedData = Dict{String,Dict}() # combined set
    transitionInvariants1 = Dict{String,Vector{Float64}}()
    transitionInvariants2 = Dict{String,Vector{Float64}}()
    transitionInvariants3 = Dict{String,Vector{Float64}}()
    atomPositions = Dict{String,Matrix{Float64}}()
    distanceMatrices = Dict{String,Matrix{Float64}}()
    transitionRefPositions = Dict{String, Vector{Point3f}}()
    transitionLabels = Dict{String,Int64}()
    #eigenvalues = Dict{String, Vector{Vec3f}}
    stretchedPrincipalAxes = Dict{String, Vector{Vector{Vec3f}}}()



    transitionLabelData = Pickle.npyload(transitionLabelPath)

    for (key, value) in transitionLabelData
        source = key |> first |> string
        target = key |> last |> string

        name = source * ">" * target
        transitionLabels[name] = value
    end

    sequence = readlines(sqeuencePath)
    sequenceHash = Base.hash(sequence)


    rootPath = dirname(dirname(@__FILE__))
    println("Root directory is: $(rootPath)")

    if isfile("$(rootPath)/cache/transitionInvariants1_$(sequenceHash).jld2")
        println("Found precomputed data, loading data...")

        @time atomPositions = JLD2.jldopen(
            "$(rootPath)/cache/atomPositions_$(sequenceHash).jld2";
            compress = true,
        ) do file
            file["atomPositions"]
        end

        @time transitionRefPositions = JLD2.jldopen(
            "$(rootPath)/cache/transitionRefPositions_$(sequenceHash).jld2";
            compress = true,
        ) do file
            file["transitionRefPositions"]
        end


        @time distanceMatrices = JLD2.jldopen(
            "$(rootPath)/cache/distanceMatrices_$(sequenceHash).jld2";
            compress = true,
        ) do file
            file["distanceMatrices"]
        end

        @time transitionInvariants1 = JLD2.jldopen(
            "$(rootPath)/cache/transitionInvariants1_$(sequenceHash).jld2";
            compress = true,
        ) do file
            file["transitionInvariants1"]
        end

        @time transitionInvariants2 = JLD2.jldopen(
            "$(rootPath)/cache/transitionInvariants2_$(sequenceHash).jld2";
            compress = true,
        ) do file
            file["transitionInvariants2"]
        end

        @time transitionInvariants3 = JLD2.jldopen(
            "$(rootPath)/cache/transitionInvariants3_$(sequenceHash).jld2";
            compress = true,
        ) do file
            file["transitionInvariants3"]
        end

        @time stretchedPrincipalAxes = JLD2.jldopen(
            "$(rootPath)/cache/stretchedPrincipalAxes_$(sequenceHash).jld2";
            compress = true,
        ) do file
            file["stretchedPrincipalAxes"]
        end

        @time bondWeights = JLD2.jldopen(
            "$(rootPath)/cache/bondWeights_$(sequenceHash).jld2";
            compress = true,
        ) do file
            file["bondWeights"]
        end


        println("loading successfull")
    else
        println("No precomputed data found! Computing now...")
        println("Reading dataset....")
        distanceMatrices = loadDistanceMatricesFromData(stateDataPath)
        atomPositions = loadAtomPositionsFromData(stateDataPath)
        bondWeights = loadBondWeightsFromData(bondWeightPath)
        atomPositions = alignAtomPositions(atomPositions |> keys |> first, atomPositions)

        #@show transitionRefPositions

        (transitionRefPositions, transitionInvariants1, transitionInvariants2, transitionInvariants3, stretchedPrincipalAxes) =
            computeTransitionInvariants(sequence, atomPositions, distanceMatrices)

 
        #@show typeof(stretchedPrincipalAxes)
       # @show keys(stretchedPrincipalAxes)

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

        println("Storing successfull")
    end

    combinedData = Dict{String,Dict}() # combined set    

    combinedData["transitionInvariants1"] = transitionInvariants1
    combinedData["transitionInvariants2"] = transitionInvariants2
    combinedData["transitionInvariants3"] = transitionInvariants3
    combinedData["atomPositions"] = atomPositions
    combinedData["distanceMatrices"] = distanceMatrices
    @show "start"
    combinedData["transitionRefPositions"] = transitionRefPositions
    @show "end"

    combinedData["bondWeights"] = bondWeights
    combinedData["transitionLabels"] = transitionLabels
    combinedData["stretchedPrincipalAxes"] = stretchedPrincipalAxes

    return combinedData
end


function loadAtomPositionsFromData(stateDataPath::String)::Dict{String,Matrix{Float64}}
    stateFiles = readdir(stateDataPath)
    filter!(e -> e ≠ ".DS_Store", stateFiles) # MacOS weirdness...
    filter!(e -> e ≠ "seq.txt", stateFiles) # filter sequence
    filter!(e -> !occursin("distance_matrix", e), stateFiles) # filter distances

    @show length(stateFiles)

    positionData = Dict{String,Matrix}()

    addStateToDict.(Ref(positionData), Ref(stateDataPath), stateFiles)

    # this might not be neccessary, as i could trat the states as strings
    #extractIdFromString( input::String ) = parse(Int64, SubString( input, 1:((findfirst("_", input ) |> first) - 1) ))

    return positionData
end


function loadDistanceMatricesFromData(stateDataPath::String)::Dict{String,Matrix{Float64}}
    stateFiles = readdir(stateDataPath)
    filter!(e -> e ≠ ".DS_Store", stateFiles) # MacOS weirdness...
    filter!(e -> e ≠ "seq.txt", stateFiles) # filter sequence
    filter!(e -> !occursin("positions", e), stateFiles) # filter positions

    @show length(stateFiles)

    distanceMatrices = Dict{String,Matrix}()

    addStateToDict.(Ref(distanceMatrices), Ref(stateDataPath), stateFiles)

    # this might not be neccessary, as i could trat the states as strings
    #extractIdFromString( input::String ) = parse(Int64, SubString( input, 1:((findfirst("_", input ) |> first) - 1) ))

    return distanceMatrices
end

function loadBondWeightsFromData(path::String)::Dict{String, Matrix{Float64}}
    bw = Pickle.npyload(open(path))
    # for now we assume that it's a dense representation
    # TODO: add ability to read sparse format matrices
    f((k,v)) = string(k) => Matrix{Float64}(v)
    
    return Dict(Iterators.map(f, pairs(bw)))
end

function addPositionsToDict(dict::Dict{String,Matrix}, pathToFile::String, fileName::String)
    stateId = SubString(fileName, 1:((findfirst("_", fileName)|>first)-1))
    dict[stateId] = Pickle.npyload(open(pathToFile * fileName))
    return nothing
end


function addStateToDict(dict::Dict{String,Matrix}, pathToFile::String, fileName::String)
    stateId = SubString(fileName, 1:((findfirst("_", fileName)|>first)-1))
    dict[stateId] = Pickle.npyload(open(pathToFile * fileName))
    return nothing
end

function getSequence(sqeuencePath::String)
    return readlines(sqeuencePath)
end

function alignAtomPositions( referenceState::String, atomPositions::Dict{String,Matrix{Float64}})::Dict{String,Matrix{Float64}}

    alignedAtomPositions = Dict{String, Matrix{Float64}}()

    referencePositions = atomPositions[referenceState]
    alignedAtomPositions[referenceState] = referencePositions

    for (stateName, positions) in atomPositions

        if stateName == referenceState
            continue
        end

        #s2 changes  s1 stays
        x = positions
        xp = referencePositions

        s = mean(x, dims=1)
        sp = mean(xp, dims=1)

        xs = x .- s
        xps = xp .- sp
        
        # @show size(xs)
        # @show size(xps)

        xx = transpose(xs) * xs   
        xpx = transpose(xps) * xs 

        # @show size(xx)
        # @show size(xpx)

        xxi = inv(xx)
        R = xpx * xxi

        # @show size(transpose(R * transpose(xs)) )
        # @show size(transpose(sp))

        alignedAtomPositions[stateName] = transpose(R * transpose(xs))  .+ sp
    end

    return alignedAtomPositions
end
