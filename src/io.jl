function getDataSets( stateDataPath::String, sqeuencePath::String, transitionLabelPath::String )::Dict{String, Dict}
    
    combinedData = Dict{String, Dict}() # combined set
    transitionInvariants1 = Dict{String,Vector}() 
    transitionInvariants2 = Dict{String,Vector}() 
    transitionInvariants3 = Dict{String,Vector}() 
    atomPositions = Dict{String, Matrix}()
    distanceMatrices = Dict{String, Matrix}()
    transitionRefPositions = Dict{String, Vector}()
    transitionLabels = Dict{String, Int64}()


    transitionLabelData = Pickle.npyload(transitionLabelPath)

    for (key, value) in transitionLabelData
        source = key |> first |> string
        target = key |> last |> string

        name = source *">"*target
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

        println("loading successfull")
    else
        println("No precomputed data found! Computing now...")
        println("Reading dataset....")
        distanceMatrices = loadDistanceMatricesFromData(stateDataPath)
        atomPositions = loadAtomPositionsFromData(stateDataPath)
    
        (transitionInvariants1, transitionInvariants2, transitionInvariants3) = computeTransitionInvariants( sequence, atomPositions, distanceMatrices ) 

        println("Storing  data.... $(rootPath)/cache/$(sequenceHash).jld2")

        @time JLD2.jldsave("$(rootPath)/cache/atomPositions_$(sequenceHash).jld2", true; atomPositions)
        @time JLD2.jldsave("$(rootPath)/cache/transitionRefPositions_$(sequenceHash).jld2", true; transitionRefPositions)
        @time JLD2.jldsave("$(rootPath)/cache/distanceMatrices_$(sequenceHash).jld2", true; distanceMatrices)
        @time JLD2.jldsave("$(rootPath)/cache/transitionInvariants1_$(sequenceHash).jld2", true; transitionInvariants1)
        @time JLD2.jldsave("$(rootPath)/cache/transitionInvariants2_$(sequenceHash).jld2", true; transitionInvariants2)
        @time JLD2.jldsave("$(rootPath)/cache/transitionInvariants3_$(sequenceHash).jld2", true; transitionInvariants3)
        
        println("Storing successfull")
    end

    combinedData = Dict{String, Dict}() # combined set    

    combinedData["transitionInvariants1"] = transitionInvariants1
    combinedData["transitionInvariants2"] = transitionInvariants2
    combinedData["transitionInvariants3"] = transitionInvariants3
    combinedData["atomPositions"] = atomPositions
    combinedData["distanceMatrices"] = distanceMatrices
    combinedData["transitionRefPositions"] = transitionRefPositions
    combinedData["transitionLabels"] = transitionLabels

    return combinedData
end


function loadAtomPositionsFromData( stateDataPath::String )::Dict{String, Matrix{Float64}}
    stateFiles = readdir(stateDataPath)
    filter!(e->e ≠ ".DS_Store",stateFiles) # MacOS weirdness...
    filter!(e->e ≠ "seq.txt",stateFiles) # filter sequence
    filter!(e->!occursin("distance_matrix", e), stateFiles) # filter distances

    @show length(stateFiles)

    positionData = Dict{String, Matrix}()

    addStateToDict.(Ref(positionData), Ref(stateDataPath), stateFiles)

    # this might not be neccessary, as i could trat the states as strings
    #extractIdFromString( input::String ) = parse(Int64, SubString( input, 1:((findfirst("_", input ) |> first) - 1) ))

    return positionData
end


function loadDistanceMatricesFromData( stateDataPath::String )::Dict{String, Matrix{Float64}}
    stateFiles = readdir(stateDataPath)
    filter!(e->e ≠ ".DS_Store",stateFiles) # MacOS weirdness...
    filter!(e->e ≠ "seq.txt",stateFiles) # filter sequence
    filter!(e->!occursin("positions", e), stateFiles) # filter positions

    @show length(stateFiles)

    distanceMatrices = Dict{String, Matrix}()

    addStateToDict.(Ref(distanceMatrices), Ref(stateDataPath), stateFiles)

    # this might not be neccessary, as i could trat the states as strings
    #extractIdFromString( input::String ) = parse(Int64, SubString( input, 1:((findfirst("_", input ) |> first) - 1) ))

    return distanceMatrices
end


function addPositionsToDict( dict::Dict{String, Matrix}, pathToFile::String, fileName::String)
    stateId =SubString( fileName, 1:((findfirst("_", fileName ) |> first) - 1) )
    dict[stateId] = Pickle.npyload(open(pathToFile*fileName))
    return nothing
end


function addStateToDict( dict::Dict{String, Matrix}, pathToFile::String, fileName::String)
    stateId =SubString( fileName, 1:((findfirst("_", fileName ) |> first) - 1) )
    dict[stateId] = Pickle.npyload(open(pathToFile*fileName))
    return nothing
end

function getSequence( sqeuencePath::String )
    return  readlines(sqeuencePath)
end

