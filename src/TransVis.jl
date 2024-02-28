module TransVis

using Pickle
using LinearAlgebra
using GLMakie
using GeometryBasics
using Makie
using Statistics
using UMAP
using NearestNeighbors
using Base.Threads
using Distances
using JLD2
using CodecZlib
using TSne


export go


function quat_from_rotmatrix( matrix::Matrix{Float64} )
   #matrix = transpose(matrix)
    trace = tr( matrix );

    if( trace > 0 )
        s = sqrt( trace + 1 ) * 2;
        m_w = 0.25 * s;
        m_x = ( matrix[3, 2] - matrix[2, 3] ) / s;
        m_y = ( matrix[1, 3] - matrix[3, 1] ) / s;
        m_z = ( matrix[2, 1] - matrix[1, 2] ) / s;
    elseif( ( matrix[1, 1] > matrix[2, 2] ) & ( matrix[1, 1] > matrix[3, 3] ) )
        s = sqrt( 1 + matrix[1, 1] - matrix[2, 2] - matrix[3, 3] ) * 2;
        m_w = ( matrix[3, 2] - matrix[2, 3] ) / s;
        m_x = 0.25 * s;
        m_y = ( matrix[1, 2] + matrix[2, 1] ) / s;
        m_z = ( matrix[1, 3] + matrix[3, 1] ) / s;
    elseif( matrix[2, 2] > matrix[3, 3] )
        s = sqrt( 1.0 + matrix[2, 2] - matrix[1, 1] - matrix[3, 3] ) * 2;
        m_w = ( matrix[1, 3] - matrix[3, 1] ) / s;
        m_x = ( matrix[1, 2] + matrix[2, 1] ) / s;
        m_y = 0.25 * s;
        m_z = ( matrix[2, 3] + matrix[3, 2] ) / s;
    else
        s = sqrt( 1.0 + matrix[3, 3] - matrix[1, 1] - matrix[2, 2] ) * 2;
        m_w = ( matrix[2, 1] - matrix[1, 2] ) / s;
        m_x = ( matrix[1, 3] + matrix[3, 1] ) / s;
        m_y = ( matrix[2, 3] + matrix[3, 2] ) / s;
        m_z = 0.25 * s;
    
    end
    return Quaternion( m_w, m_x, m_y, m_z)
end

# function quat_from_rotmatrix(dcm::AbstractMatrix{T}) where {T<:Real}
#     a2 = 1 + dcm[1,1] + dcm[2,2] + dcm[3,3]
#     a = sqrt(a2)/2
#     b,c,d = (dcm[3,2]-dcm[2,3])/4a, (dcm[1,3]-dcm[3,1])/4a, (dcm[2,1]-dcm[1,2])/4a
#     return Quaternion(a,b,c,d)
# end

function transform_mesh(msh, mat4x4::Matrix{Float64})
    pos_trans = Point3.(Ref(mat4x4) .* to_ndim.(Point4f0, msh.position, 0))
    GLMakie.Mesh(pos_trans, GLMakie.faces(msh))
end

function link_cameras_lscene(f; step=.01)
    scenes = filter(x -> x isa LScene, f.content)
    cameras = map(x -> cameracontrols(x.scene), scenes)
  
    for i ∈ 1:length(cameras)
      on(cameras[i].eyeposition) do eye
        for j ∈ 1:length(cameras)
          i == j && continue
          if sum(abs, eye - cameras[j].eyeposition[]) > step
            update_cam!(scenes[j].scene, cameras[i])
          end
        end
      end
    end
    f
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

# function computeWeight()

# end
function angle(a, b)
    return acosd(clamp(a⋅b/(norm(a)*norm(b)), -1, 1))
end

function fractionalAnisotropy( ev::Vector{Float64} )
    meanEV = (ev[1] + ev[2] + ev[3])/3.0
    a = sqrt( (ev[1] - meanEV)^2 + (ev[2] - meanEV)^2 + (ev[3] - meanEV)^2  ) / sqrt( ev[1]^2 + ev[2]^2 + ev[3]^2  )
    return sqrt(3.0/2.0) * a  
end

function go() 

    fig = Figure()
   
    lsceneLeft = LScene(fig[1:3, 1:3], show_axis=false, scenekw = (backgroundcolor = :whitesmoke, clear = true))
    lsceneRight = LScene(fig[1:3, 4:6], show_axis=false, scenekw = (backgroundcolor = :whitesmoke, clear = true))
    #lscenec = LScene(fig[1:2, 7:9], show_axis=false, scenekw = (backgroundcolor = :whitesmoke, clear = true))
    #lscene2 = LScene(fig[3:4, 1:3], show_axis=false, scenekw = (backgroundcolor = :whitesmoke, clear = true))
    
    axI1 = Axis(fig[5:6, 1:4], xlabel = "Atom Number", ylabel = "Invariant 1")
    axI2 = Axis(fig[7:8, 1:4], xlabel = "Atom Number", ylabel = "Invariant 2")
    axI3 = Axis(fig[9:10, 1:4], xlabel = "Atom Number", ylabel = "Invariant 3")

    axDR = Axis(fig[5:10, 5:6],  title = "t-SNE")
    #axDR = LScene(fig[5:10, 5:6], show_axis=false, scenekw = (backgroundcolor = :white, clear = true))

    # axI1 = PolarAxis(fig[4:5, 1:6], title = "Transition Invariants")
    # axI2 = PolarAxis(fig[6:7, 1:6], title = "Transition Invariants")
    # axI3 = PolarAxis(fig[8:9, 1:6], title = "Transition Invariants")


 
    print( "Using $(Threads.nthreads()) threads\n")

    stateDataPath = "/Users/Bote/Documents/ASU/state_data copy/"
    sequence = readlines("/Users/Bote/Documents/ASU/state_data copy/seq.txt")


    transitionLabelData = Pickle.npyload("/Users/Bote/Documents/ASU/nano_pt_labels.pickle")


    transitionInvariants1 = Dict{String,Vector}() 
    transitionInvariants2 = Dict{String,Vector}() 
    transitionInvariants3 = Dict{String,Vector}() 
    #transitionFA = Dict{String,Vector}() 
    atomPositions = Dict{String, Matrix}()
    distanceMatrices = Dict{String, Matrix}()

    transitionRefPositions = Dict{String, Vector}()


    @show typeof(transitionLabelData)
    #@show transitionLabels
#    @show transitionLabelData[(27,1)]

    transitionLabels = Dict{String, Int64}()

    for (key, value) in transitionLabelData
        source = key |> first |> string
        target = key |> last |> string

        name = source *">"*target
        transitionLabels[name] = value
    end

    #test = Pickle.npyload("/Users/Bote/Documents/ASU/state_data/1_positions.pickle")


 

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
    
        debugEarlyKill = 300
    
        unique = 1
        @time for sequenceStep in 1:(length(sequence)-1) #1:debugEarlyKill
    
            #i = 60
            currentState = sequence[sequenceStep]
            nextState = sequence[sequenceStep+1]
    
            transitionName = currentState*">"*nextState
       
            if haskey( transitionInvariants1, currentState*">"*nextState)
                continue
            end
            unique = unique +1
    
            currentDM = distanceMatrices[currentState]
            nextDM = distanceMatrices[nextState]
    
            maxDM = max.(currentDM, nextDM)
            differenceDM = nextDM .- currentDM
            signature = differenceDM ./ maxDM
    
            #signature = exp.(signature)
    
            for j in 1:length(signature[:,1])
                signature[j,j] = 0
            end
    
    
            aPos1 = atomPositions[currentState]
            aPos2 = atomPositions[nextState]
    
            #@show length(aPos1[:,1])
        
            transitionRefPositions[transitionName] = [ Makie.Point3f.( aPos1[i, 1], aPos1[i, 2], aPos1[i, 3] ) for i in 1:length(aPos1[:,1])]
    
            weights = 1 ./ ((distanceMatrices[currentState] + distanceMatrices[nextState] )./2)
    
            F = Vector{Matrix{Float64}}(undef, length(aPos1[:,1]))
    
            #  kdtree = KDTree(transpose(aPos1); leafsize = 5)
            #  nnIndices, dists = knn(kdtree, transpose(aPos1), 10)
    
            for m in 1:length(aPos1[:,1])
    
                D = zeros(3,3) 
                A = zeros(3,3) 
            
                for n in 1:length(aPos1[:,1]) #nnIndices[m] 
                    if m == n || weights[m,n] < 0.001
                        continue
                    end
    
                    deltaXmn =  aPos1[n,:] - aPos1[m,:] 
                    D = D + (deltaXmn * transpose(deltaXmn) * weights[m,n])
    
                    deltaxmn =  aPos2[n,:] - aPos2[m,:] 
                    A = A + (deltaxmn * transpose(deltaXmn) * weights[m,n])
                end
                F[m] = A*inv(D)
            end
    
            I = zeros(3,3)
            I[1,1] = 1.0
            I[2,2] = 1.0
            I[3,3] = 1.0
    
            #E = 0.5*( Ref(I) .- inv.(F .* transpose.(F) )) #eulerian Almansi
             E = 0.5 .* (transpose.(F) .* F  .- Ref(I))   #lagrangian Green
            #E = transpose.(F) .* F # Cauchy-Green
            eigenSystems = eigen.(E)
    
           # @show  eigenSystems[2].values[3] * eigenSystems[2].vectors[:,3] 
           # @show E[2] * eigenSystems[2].vectors[:,3] 
    
    
            getStretchedEigVec( eigenSys ) = [eigenSys.values[1] * eigenSys.vectors[:,1], eigenSys.values[2] * eigenSys.vectors[:,2], eigenSys.values[3] * eigenSys.vectors[:,3]]
            #stretchedBases = getStretchedEigVec.(eigenSystems) # vector is stretchedBases[index][n,:]
    
            deviator = E .- (1/3 * tr.(E) .* Ref(I))
            eigenSystemsDeviator = eigen.(deviator)
    
            I1( ev::Vector{Float64} ) = ev[1] + ev[2] + ev[3]
            # I2( ev::Vector{Float64} ) = ev[1]*ev[2] + ev[1]*ev[3] + ev[2]*ev[3]
            # I3( ev::Vector{Float64} ) = ev[1] * ev[2] * ev[3]
            I2( ev::Vector{Float64} ) = sqrt(ev[1]^2 + ev[2]^2 + ev[3]^2)
            I3( ev::Vector{Float64} ) =  3*sqrt(6)* (ev[1] * ev[2] * ev[3])/((ev[1]^2 + ev[2]^2 + ev[3]^2)^(3/2))
    
    
            invariant1 = [ I1( eigenSystem.values) for eigenSystem in eigenSystems ]
            invariant2 = [ I2( eigenSystem.values) for eigenSystem in eigenSystemsDeviator ]
            invariant3 = [ I3( eigenSystem.values) for eigenSystem in eigenSystemsDeviator ]
    
            transitionInvariants1[transitionName] = invariant1
            transitionInvariants2[transitionName] = invariant2
            transitionInvariants3[transitionName] = invariant3
    
        end
                
        @show unique

        println("Storing  data.... $(rootPath)/cache/$(sequenceHash).jld2")

        @time JLD2.jldsave("$(rootPath)/cache/atomPositions_$(sequenceHash).jld2", true; atomPositions)
        @time JLD2.jldsave("$(rootPath)/cache/transitionRefPositions_$(sequenceHash).jld2", true; transitionRefPositions)
        @time JLD2.jldsave("$(rootPath)/cache/distanceMatrices_$(sequenceHash).jld2", true; distanceMatrices)
        @time JLD2.jldsave("$(rootPath)/cache/transitionInvariants1_$(sequenceHash).jld2", true; transitionInvariants1)
        @time JLD2.jldsave("$(rootPath)/cache/transitionInvariants2_$(sequenceHash).jld2", true; transitionInvariants2)
        @time JLD2.jldsave("$(rootPath)/cache/transitionInvariants3_$(sequenceHash).jld2", true; transitionInvariants3)
        
        println("Storing successfull")
    end


    # for (key, values) in transitionInvariants1 #INVARIANTS HAVE CORR 1 or -1
    #     testMatrix = [reshape(transitionInvariants1[key], 1, :); reshape(transitionInvariants2[key], 1, :)]
    #     @show  all( (abs.(cor( testMatrix )) .- 1) .< 0.0001)
    #     testMatrix = [reshape(transitionInvariants1[key], 1, :); reshape(transitionInvariants3[key], 1, :)]
    #     @show  all( (abs.(cor( testMatrix )) .- 1) .< 0.0001)
    #     testMatrix = [reshape(transitionInvariants2[key], 1, :); reshape(transitionInvariants3[key], 1, :)]
    #     @show  all( (abs.(cor( testMatrix )) .- 1) .< 0.0001)
    #     @show "--------------------------------------------------"
    # end


    # for (transition, value ) in transitionInvariants1
    #     for i in 1:length(value)
    #         transitionInvariants1[transition][i] = transitionInvariants1[transition][i] |> abs  < 0.01 ? 0 : transitionInvariants1[transition][i]
    #         transitionInvariants2[transition][i] = transitionInvariants2[transition][i] |> abs  < 0.01 ? 0 : transitionInvariants2[transition][i]
    #         transitionInvariants3[transition][i] = transitionInvariants3[transition][i] |> abs  < 0.01 ? 0 : transitionInvariants3[transition][i]
    #     end
    # end

    atoms = [1:1:length(values(transitionInvariants1) |> first);]

 #   balltree = BallTree(data, Minkowski(3.5); reorder = false)


   @show length(atoms) 
   @show length(values(transitionInvariants1))

    # stem!.( axI1, Ref(atoms), values(transitionInvariants1), alpha=0.7 )
    # stem!.( axI2, Ref(atoms), values(transitionInvariants2), alpha=0.7 )
    # stem!.( axI3, Ref(atoms), values(transitionInvariants3), alpha=0.7 )

    lines!.( axI1, Ref(atoms), values(transitionInvariants1), alpha=0.01 )        
    lines!.( axI2, Ref(atoms), values(transitionInvariants2), alpha=0.01 )
    lines!.( axI3, Ref(atoms), values(transitionInvariants3), alpha=0.01 )


   featureVectorMatrix = zeros( length( values(transitionInvariants1)),length( values(transitionInvariants1)|> first ) *1 )
   labels = zeros( length( values(transitionInvariants1)) )


   mapNameToIdx = Dict{String, Int64}()
   mapIdxToName = Vector{String}()

   row = 1
   for (transition, invariants1) in transitionInvariants1
        mapNameToIdx[transition] = row
        push!(mapIdxToName, transition)

        col = 1
        for invariant in invariants1
            featureVectorMatrix[row, col] = invariant
            col = col +1
        end

        # col = length( values(transitionInvariants1)|> first ) + 1
        # for invariant in transitionInvariants2[transition]
        #     featureVectorMatrix[row, col] = invariant
        #     col = col +1
        # end

        # col = length( values(transitionInvariants1)|> first )*2 + 1
        # for invariant in transitionInvariants3[transition]
        #     featureVectorMatrix[row, col] = invariant
        #     col = col +1
        # end


        if haskey(transitionLabels, transition)
            labels[row] = transitionLabels[transition]
       else
            labels[row] = -1
       end

       row = row + 1

   end
   
   @show featureVectorMatrix[100,:]

#    transitionAdjacencyMatrix = zeros( length(mapIdxToName), length(mapIdxToName)  ) 


#     Threads.@threads for k in 1:length(mapIdxToName)
#         for j in 1:(k-1) 
#             if k == j
#                 continue
#             end
#             @inbounds transitionAdjacencyMatrix[j,k] =   Cityblock()(featureVectorMatrix[k, :], featureVectorMatrix[j, :])
#         end
#     end
#     transitionAdjacencyMatrix = Symmetric(transitionAdjacencyMatrix)




   #@show size(featureVectorMatrix) 
   #@time embedding = umap(transitionAdjacencyMatrix, 2; metric=:precomputed)
   #@time embedding = umap(transpose(featureVectorMatrix), 2, n_neighbors=15, metric=SqEuclidean())
   #scatter!(axDR, embedding, color=labels, colormap = (:viridis, 1.0))

   selectedTransition = Observable{String}("1>3")

   #tsne(X, ndim, reduce_dims, max_iter, perplexit; [keyword arguments])
   @time Y = tsne(featureVectorMatrix[:,:], 2, 20, 2000, 40.0; );
   tsnePlot = scatter!(axDR, Y, color=labels, colormap = (:viridis, 1.0))

   on(events(fig).mousebutton, priority = 2) do event
    if event.button == Mouse.left && event.action == Mouse.press
        # Delete marker
        plt, i = pick(axDR)  
        if plt == tsnePlot
            # deleteat!(positions[], i)
            selectedTransition[] = mapIdxToName[i]
            notify(selectedTransition)
            @show plt
            @show i
            @show selectedTransition[]
            @show mapNameToIdx[mapIdxToName[i]]

            @show Y[mapNameToIdx[mapIdxToName[i]],:]
            #scatter!(axDR , Y[mapNameToIdx[mapIdxToName[i]],:], color=:black)
            return Consume(true)
        end
        return Consume(false)

    end
    # return Consume(false)
end


   #@show labels
   #@show length(labels)




    # for i in 1:20
    #     @show invariant1[i]
    #     @show invariant2[i]
    #     @show invariant3[i]
    #     @show fracAniso[i]/10.
    #     @show "----------------"
    # end

   # maxEV =  [ maximum(eigenSystem.values) |> sqrt for eigenSystem in eigenSystems ]

    #@show fracAniso[1]

   # @show size(stretchedBases[1][1][1])
    # testIdx = 130

    # @show norm(stretchedBases[testIdx][1,:] )
    # @show norm(stretchedBases[testIdx][2,:] )
    # @show norm(stretchedBases[testIdx][3,:] )
    # @show norm(stretchedBases[testIdx][1,:] )*norm(stretchedBases[testIdx][2,:] )*norm(stretchedBases[testIdx][3,:] )


    # rots = Vector{Quaternion}( undef, length(aPos1[:,1]))
    # sizes = Vector{Point3f0}(undef, length(aPos1[:,1]))

    # for i in 1:length(rots)

    #     sizes[i] = Point3f0( eigenSystems[i].values[1],  eigenSystems[i].values[2], eigenSystems[i].values[3])

    #     rot = zeros(3,3)
    #     ev1 = eigenSystems[i].vectors[:,1]
    #     ev2 = eigenSystems[i].vectors[:,2]
    #     ev3 = eigenSystems[i].vectors[:,3] 

    #     if abs(norm(ev1) - 1) > 0.000001
    #         @show "SHIT"
    #     end
    #     if abs(norm(ev2) - 1) > 0.000001
    #         @show "SHIT"
    #     end
    #     if abs(norm(ev3) - 1) > 0.000001
    #         @show "SHIT"
    #     end

    #     # ensure we have a right hand system
    #     if( signbit( ev3 ⋅ cross( ev1, ev2 ) ) )
    #         ev3 = ev3 .* -1
    #     end

    #     rot[1, 1] = ev1[1] #ev1( 0 );
    #     rot[2, 1] = ev1[2] #ev1( 1 );
    #     rot[3, 1] = ev1[3] #ev1( 2 );
    #     rot[1, 2] = ev2[1] #ev2( 0 );
    #     rot[2, 2] = ev2[2] #ev2( 1 );
    #     rot[3, 2] = ev2[3] #ev2( 2 );
    #     rot[1, 3] = ev3[1] #ev3( 0 );
    #     rot[2, 3] = ev3[2] #ev3( 1 );
    #     rot[3, 3] = ev3[3] #ev3( 2 );
    #     #@show QuatRotation(rot) 

    #     rots[i] = normalize( quat_from_rotmatrix(rot) )
    # end

   # rots = normalize.(rand(Quaternion, length(positions)))

    #transformedMeshes = transform_mesh.(baseSphereMeshes, Ref(scaleTrans))

    #transformedMeshes = [  mesh for mesh in baseSphereMeshes]


   # transformationMatrix =  GLMakie.rotationmatrix_y(pi/4) * GLMakie.scalematrix(Vec3f0(1, 1, 2))
   
   # p = meshscatter!(lscenea, aPos1,markersize=invariant1 , color=invariant1, colormap=:PuRd)
    #p2 = meshscatter!(lsceneb, aPos1,markersize=invariant3, color=invariant3, colormap=:PuRd)
    # = meshscatter!(lscenec, aPos1,markersize=invariant3 .* 0.2, color=invariant3)
    #p3 = meshscatter!(lscenec, aPos1,markersize=fracAniso, color = fracAniso)

    # arrowDir = [ Vec3f(eigenSystems[i].vectors[:,1] .* eigenSystems[i].values[1] ) for i in 1:length(stretchedBases)]
    # arrowDir2 = [ Vec3f(eigenSystems[i].vectors[:,2] .* eigenSystems[i].values[2] ) for i in 1:length(stretchedBases)]
    # arrowDir3 = [ Vec3f(eigenSystems[i].vectors[:,3] .* eigenSystems[i].values[3] ) for i in 1:length(stretchedBases)]

    # arrowPos = [ Point3f(aPos1[i,1],aPos1[i,2],aPos1[i,3] ) for i in 1:length(aPos1[:,1])]

    #@show eigenSystems[1].values[:]

    # arrows!(lscenec, arrowPos, arrowDir; color=:red, alpha=0.5, markersize= eigenSystems[i].values[1] )
    # arrows!(lscenec, arrowPos, arrowDir2; color=:red, alpha=0.5, markersize= eigenSystems[i].values[2]  )
    # arrows!(lscenec, arrowPos, arrowDir3; color=:red, alpha=0.5, markersize= eigenSystems[i].values[3]  )


    # meshscatter!(lsceneFA, aPos1,color = fracAniso, markersize = sizes .* fracAniso, rotations = rots, colormap=:PuRd )



    transitionSequence = Vector{String}()

    for sequenceStep in 1:(length(sequence)-1)
        currentState = sequence[sequenceStep]
        nextState = sequence[sequenceStep+1]
        transitionName = currentState*">"*nextState
        push!(transitionSequence, transitionName)

    end
 #   @show transitionRange = [1:length(transitionSequence);]


   # @show transitionSequence

    sliderTransition = SliderGrid(
        fig[4, 1:6],
        (label= "Transition", range = [1:length(transitionSequence);],
        startvalue = 1,format = x -> string(transitionSequence[x]))
    )
    currentTransition = lift(sliderTransition.sliders[1].value) do val
        @show transitionSequence[val]
        return transitionSequence[val]
     end
    

    #  class = lift( sliderTransition.sliders[1].value ) do val
    #     return transitionLabels[transitionSequence[val]]
    #  end


    #@show lines[1]

    #@show transitionRefPositions[transitionSequence[1]]

     #lines!(lsceneDistances, lift(x->x, lines), color=bondDiff, colormap=:balance, colorrange=(-maxDiff,maxDiff))

     scatter!(lsceneLeft, lift(x->transitionRefPositions[x], currentTransition) ;markersize=lift(x->abs.(transitionInvariants1[x]) * 70, currentTransition),
      color=lift(x->transitionInvariants1[x], currentTransition), colormap=:bwr, colorrange=(-0.4,0.4))

     scatter!(lsceneRight, lift(x->transitionRefPositions[x], selectedTransition) ;markersize=lift(x->abs.(transitionInvariants1[x]) * 70, selectedTransition),
      color=lift(x->transitionInvariants1[x], selectedTransition), colormap=:bwr, colorrange=(-0.4,0.4))

     stem!( axI1, atoms, lift(x->transitionInvariants1[x], selectedTransition))
     stem!( axI1, atoms, lift(x->transitionInvariants1[x], currentTransition))

     stem!( axI2, atoms, lift(x->transitionInvariants2[x], selectedTransition))
     stem!( axI2, atoms, lift(x->transitionInvariants2[x], currentTransition))

     stem!( axI3, atoms, lift(x->transitionInvariants3[x], selectedTransition))
     stem!( axI3, atoms, lift(x->transitionInvariants3[x], currentTransition))

     #scatter!(axDR , lift(x->Y[mapNameToIdx[x],:], selectedTransition), color=:black, marker = Circle)
     

    #  stem!( axI1, Ref(atoms), values(transitionInvariants1), alpha=0.7 )
    #  stem!( axI1, Ref(atoms), values(transitionInvariants1), alpha=0.7 )


    link_cameras_lscene(fig)


    #DataInspector(fig)
    fig
end

end # module TransVis
