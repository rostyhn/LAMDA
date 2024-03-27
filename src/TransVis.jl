module TransVis

#Data handling
using Pickle
using JLD2
using CodecZlib

# #GUI
# using Gtk4
# using Gtk4Makie

#Vis
using GLMakie
using Makie
using GeometryBasics

#Processing and Helpers
using NearestNeighbors
#using Distances
using LinearAlgebra
using TSne
#using GeometryBasics
using Base.Threads
using Statistics
using ProgressMeter
using Ripserer
using PersistenceDiagrams
#using UMAP
#using MultivariateStats

include("io.jl")
include("processing.jl")


export go



function getIndexFromSQMesh(i::Int64, resolution::Float64)

    #number of points in sq mesh
    #[0:resolution:pi;]
    #push!(phiRange, pi) #ass pi to close the hole at the end introduced by resolution
    thetaRange = [0:resolution:2*pi;] #

    numberY = trunc(Int, pi / resolution) + 2
    numberX = trunc(Int, 2 * pi / resolution) + 1
    number = numberX * numberY
    return trunc(Int, i / number) + 1
end

function signPow(base, exponent)::Float64
    return sign(base) * abs(base)^exponent
end

function qz(phi::Float64, theta::Float64, alpha::Float64, beta::Float64, K2::Float64)
    x = signPow(cos(theta), alpha) * signPow(sin(phi), beta)
    y = signPow(sin(theta), alpha) * signPow(sin(phi), beta)
    z = signPow(cos(phi), beta)

    return Point3f(x, y, z)
end

function qx(phi::Float64, theta::Float64, alpha::Float64, beta::Float64, K2::Float64)
    x = signPow(cos(phi), beta)
    y = -signPow(sin(theta), alpha) * signPow(sin(phi), beta)
    z = signPow(cos(theta), alpha) * signPow(sin(phi), beta)
    return Point3f(x, y, z)

end

function superquadric(scale::Float64, position::Point3f, principalStretches::Vector{GeometryBasics.Vec{3,Float32}}, K1::Float64, K3::Float64, sharpness::Float64, resolution=0.2)
    points = Vector{Point3f}()

    #K2 is the volume perserving fractionalAnisotropy
    #K3 is the mode defining the type of anisotropy: -1 planar to 1 linear

    stretchRatio1 = norm(principalStretches[3])
    stretchRatio2 = norm(principalStretches[2])
    stretchRatio3 = norm(principalStretches[1])

    #debug
    # stretchRatio1 = 5.
    # stretchRatio2 = 1.
    # stretchRatio3 = 1.

    stretchDirection1 = principalStretches[3] / stretchRatio1
    stretchDirection2 = principalStretches[2] / stretchRatio2
    stretchDirection3 = principalStretches[1] / stretchRatio3


    cl = (stretchRatio1 - stretchRatio2) / (stretchRatio1 + stretchRatio2 + stretchRatio3)   #linear anisotopy
    cp = 2 * (stretchRatio2 - stretchRatio3) / (stretchRatio1 + stretchRatio2 + stretchRatio3) # planar anisotropy
    cs = 3 * stretchRatio3 / (stretchRatio1 + stretchRatio2 + stretchRatio3)

    # @show cl
    # @show cp
    # @show cs
    #@show stretchRatio1 * stretchRatio1 * stretchRatio1

    #@show cs

    phiRange = [0:resolution:pi;]  #vertical: south -> north
    push!(phiRange, pi) #ass pi to close the hole at the end introduced by resolution
    thetaRange = [0:resolution:2*pi;] #horizontal: west -> east


    if cl >= cp
        alpha = signPow((1 - cp), sharpness)
        beta = signPow((1 - cl), sharpness)

        for phi in phiRange
            for theta in thetaRange
                push!(points, qx(phi, theta, alpha, beta, K1))
            end
        end
    else
        alpha = (1 - cl)^sharpness
        beta = (1 - cp)^sharpness

        for phi in phiRange
            for theta in thetaRange
                push!(points, qz(phi, theta, alpha, beta, K1))
            end
        end
    end


    scaleMatrix = zeros(3, 3)
    scaleMatrix[1, 1] = stretchRatio1
    scaleMatrix[2, 2] = stretchRatio2
    scaleMatrix[3, 3] = stretchRatio3

    scaleMatrix = scaleMatrix * scale

    rotationMatrix = zeros(3, 3)
    rotationMatrix[:, 1] = stretchDirection1
    rotationMatrix[:, 2] = stretchDirection2
    rotationMatrix[:, 3] = stretchDirection3

    if det(rotationMatrix) < 0
        rotationMatrix[:, 1] = -1 * rotationMatrix[:, 1]
    end

    transform = rotationMatrix * scaleMatrix

    #meshscatter!( scene, position[1],position[2],position[3]; color= :black)
    #meshscatter!( scene, position[1] + 5*stretchDirection1[1], position[2]+5*stretchDirection1[2], position[3]+5*stretchDirection1[3]; color= :black)
    # @show scaleMatrix
    # @show points[1]
    points = Point3f.(Ref(transform) .* points) .+ Ref(position)
    # @show points[1]

    #scatter!(scene, points)

    nPhi = length(phiRange)
    nTheta = length(thetaRange)

    indices = Vector{Tuple{UInt32,UInt32,UInt32}}() # triangles over the points

    for y in 1:(nPhi-1)
        for x in 1:nTheta

            #@show "???"
            p11 = x + nTheta * (y - 1)
            p21 = x < nTheta ? (x + 1) + nTheta * (y - 1) : 1 + nTheta * (y - 1)
            p31 = x + nTheta * (y)

            p12 = x < nTheta ? (x + 1) + nTheta * (y - 1) : 1 + nTheta * (y - 1)
            p22 = x < nTheta ? (x + 1) + nTheta * y : 1 + nTheta * y # index 
            p32 = x + nTheta * (y)

            push!(indices, (p11, p31, p21))
            push!(indices, (p32, p22, p12))

        end
    end

    # if K3 >=0 #linear
    #     points = [ p[1]*exp(abs(K2)) for p in points]
    # else #planar
    #     points = [ p[2]*exp(abs(K2)) for p in points]
    #     points = [ p[3]*exp(abs(K2)) for p in points]
    # end

    triFaces = TriangleFace.(indices)

    # Create the Mesh
    mesh = GeometryBasics.Mesh(points, triFaces)

    return mesh
end

function link_cameras_lscene(f; step=0.01)
    scenes = filter(x -> x isa LScene, f.content)
    cameras = map(x -> cameracontrols(x.scene), scenes)

    for i in eachindex(cameras)
        on(cameras[i].eyeposition) do eye
            for j in eachindex(cameras)
                i == j && continue
                if sum(abs, eye - cameras[j].eyeposition[]) > step
                    update_cam!(scenes[j].scene, cameras[i])
                end
            end
        end
    end
    f
end



function angle(a, b)
    return acosd(clamp(a ⋅ b / (norm(a) * norm(b)), -1, 1))
end

function fractionalAnisotropy(ev::Vector{Float64})
    meanEV = (ev[1] + ev[2] + ev[3]) / 3.0
    a =
        sqrt((ev[1] - meanEV)^2 + (ev[2] - meanEV)^2 + (ev[3] - meanEV)^2) /
        sqrt(ev[1]^2 + ev[2]^2 + ev[3]^2)
    return sqrt(3.0 / 2.0) * a
end



function go()

    GLMakie.closeall() #close all windows for rerun!

    plotWindow = Figure(size=(600, 400))
    molWindow = Figure(size=(600, 400))


    lscenePre = LScene(
        molWindow[1:3, 1:3],
        show_axis=false,
        scenekw=(backgroundcolor=:white, clear=true),
    )
    lscenePost = LScene(
        molWindow[4:6, 1:3],
        show_axis=false,
        scenekw=(backgroundcolor=:white, clear=true),
    )

    lsceneRightVolume = LScene(
        molWindow[1:6, 4:6],
        show_axis=false,
        scenekw=(backgroundcolor=:white, clear=true),
    )
    # lsceneRightAtoms = LScene(
    #     molWindow[4:6, 4:6],
    #     show_axis = false,
    #     scenekw = (backgroundcolor = :white, clear = true),
    # )



    axI1 = Axis(plotWindow[1:2, 1:4], xlabel="Atom Number", ylabel="K1")
    axI2 = Axis(plotWindow[3:4, 1:4], xlabel="Atom Number", ylabel="K2")
    axI3 = Axis(plotWindow[5:6, 1:4], xlabel="Atom Number", ylabel="mode(E)")
    axDR = Axis(plotWindow[1:6, 5:7], title="t-SNE")

    transitionGlyphSizeSlider = Slider(molWindow[4:6, 7], range=0.1:0.01:4, horizontal=false, startvalue=1)



    stateDataPath = "/home/frosty/Programs/Julia/TransVis/data/state_data copy/"
    sequencePath = "/home/frosty/Programs/Julia/TransVis/data/state_data copy/seq.txt"
    transitionPath = "/home/frosty/Programs/Julia/TransVis/data/nano_pt_labels.pickle"
    bondWeightPath = "/home/frosty/Programs/Julia/TransVis/data/bond_weights.pickle"

    combinedData = getDataSets(stateDataPath, sequencePath, transitionPath, bondWeightPath)

    transitionInvariants1 = combinedData["transitionInvariants1"]
    transitionInvariants2 = combinedData["transitionInvariants2"]
    transitionInvariants3 = combinedData["transitionInvariants3"]

    transitionRefPositions = combinedData["transitionRefPositions"]
    transitionLabels = combinedData["transitionLabels"]

    distanceMatrices = combinedData["distanceMatrices"]

    stretchedPrincipalAxes = combinedData["stretchedPrincipalAxes"]

    atomPositions = combinedData["atomPositions"]

    bondWeights = combinedData["bondWeights"]

    #get min max of all transition invariants 1
    minInvariant1 = 1.0e10
    maxInvariant2 = -1.0e10
    @time for (key, value) in transitionInvariants1
        for invariant1 in value
            if minInvariant1 > invariant1
                minInvariant1 = invariant1
            end
            if maxInvariant2 < invariant1
                maxInvariant2 = invariant1
            end
        end
    end
    println("invariant range:  $(minInvariant1) -  $(maxInvariant2)")

    invariant1MaxRange = max(abs(minInvariant1), abs(maxInvariant2))

    #get min max coordinates of atoms for bounding box
    minX = 1.0e10
    minY = 1.0e10
    minZ = 1.0e10
    maxX = -1.0e10
    maxY = -1.0e10
    maxZ = -1.0e10
    @time for (key, positions) in atomPositions
        for row in 1:length(positions[:, 1])
            if minX > positions[row, 1]
                minX = positions[row, 1]
            end
            if maxX < positions[row, 1]
                maxX = positions[row, 1]
            end
            if minY > positions[row, 2]
                minY = positions[row, 2]
            end
            if maxY < positions[row, 2]
                maxY = positions[row, 2]
            end
            if minZ > positions[row, 3]
                minZ = positions[row, 3]
            end
            if maxZ < positions[row, 3]
                maxZ = positions[row, 3]
            end
        end
    end

    volumeResolution = 0.2

    sampleRangeX = [minX-2*volumeResolution:volumeResolution:maxX+2*volumeResolution;]
    sampleRangeY = [minY-2*volumeResolution:volumeResolution:maxY+2*volumeResolution;]
    sampleRangeZ = [minZ-2*volumeResolution:volumeResolution:maxZ+2*volumeResolution;]

    @show length(sampleRangeX)
    @show length(sampleRangeY)
    @show length(sampleRangeZ)


    @show "computing kd trees"
    transitionKDTree = Dict{String,KDTree}()
    @time for (key, value) in transitionRefPositions
        transitionKDTree[key] = KDTree(value)
    end

    kernelWidth = 1.0

    volumeData = Observable(zeros(length(sampleRangeX), length(sampleRangeY), length(sampleRangeZ)))

    volumeAbsMax = Observable(1.0)

    #@show "volume comp time"
    # currentTransition = lift(sliderTransition.sliders[1].value) do val
    #     return transitionSequence[val]
    # end


    # extremes = extrema( volumeData )
    # maxvariation = max(extremes|>first, extremes|>last)
    # volumeData = volumeData ./ maxvariation


    #volumeData = Float32.((volumeData .- minimum(volumeData)) ./ (maximum(volumeData) - minimum(volumeData)))

    #    @show volumeData[1,1,1]
    transitionGlyphSize = lift(transitionGlyphSizeSlider.value) do val
        return val
    end

    atoms = [1:1:length(values(transitionInvariants1) |> first);]

    #   balltree = BallTree(data, Minkowski(3.5); reorder = false)

    # lines!.(axI1, Ref(atoms), values(transitionInvariants1), alpha = 0.01)
    # lines!.(axI2, Ref(atoms), values(transitionInvariants2), alpha = 0.01)
    # lines!.(axI3, Ref(atoms), values(transitionInvariants3), alpha = 0.01)

    featureVectorMatrix = zeros(
        length(values(transitionInvariants1)),
        length(values(transitionInvariants1) |> first) * 1,
    )
    labels = zeros(length(values(transitionInvariants1)))


    mapNameToIdx = Dict{String,Int64}()
    mapIdxToName = Vector{String}()

    row = 1
    for (transition, invariants1) in transitionInvariants1
        mapNameToIdx[transition] = row
        push!(mapIdxToName, transition)

        col = 1
        for invariant in invariants1
            featureVectorMatrix[row, col] = invariant |> abs
            col = col + 1
        end

        # col = length(values(transitionInvariants1) |> first) + 1
        # for invariant in transitionInvariants2[transition]
        #     featureVectorMatrix[row, col] = invariant |> abs
        #     col = col + 1
        # end

        # col = length(values(transitionInvariants1) |> first) * 2 + 1
        # for invariant in transitionInvariants3[transition]
        #     featureVectorMatrix[row, col] = invariant |> abs
        #     col = col + 1
        # end


        #@show transition
        splitToStates = split(transition, ">")
        flippedName = splitToStates[2] * ">" * splitToStates[1]

        # @show flippedName

        if haskey(transitionLabels, transition)
            labels[row] = transitionLabels[transition]
        elseif haskey(transitionLabels, flippedName)
            labels[row] = transitionLabels[flippedName]
        else
            @show "Transition Label not found for " * transition
        end

        row = row + 1

    end

    #@show featureVectorMatrix[100,:]

    #build distance distanceMatrices
    invariantDistances = zeros(length(transitionInvariants1["1>3"]), length(transitionInvariants1["1>3"])) #nAtoms x nAtoms
    invariantDistances = Vector{Matrix}(undef, length(transitionInvariants1))
    @time for (key, value) in transitionInvariants1
        invariantDistances[mapNameToIdx[key]] = computeDistances(value)
    end

    maxMoment = 10
    invariantMomentFeatures = zeros(length(transitionInvariants1), maxMoment)
    @time for (key, value) in transitionInvariants1
        row = mapNameToIdx[key]
        for moment in 1:maxMoment
            invariantMomentFeatures[row, moment] = computeMoment(value, moment)
        end
    end


    @show size(transitionRefPositions["1>3"])
    @show size(transitionInvariants1["1>3"])

    pcaInput = zeros(4, length(transitionRefPositions["1>3"]))
    for i in 1:length(transitionRefPositions["1>3"])
        pcaInput[1, i] = transitionRefPositions["1>3"][i][1]
        pcaInput[2, i] = transitionRefPositions["1>3"][i][2]
        pcaInput[3, i] = transitionRefPositions["1>3"][i][3]
        pcaInput[4, i] = transitionInvariants1["1>3"][i]
    end

    # @show size(pcaInput)
    # @show pca = fit(PCA, pcaInput;maxoutdim=3)


    #@time invariantDistances = computeDistances(transitionInvariants1["1>3"])
    #    @show length(invariantDistances)
    @show "compting PD"
    #@time persistenceDiagrams =  ripserer.(invariantDistances[1:200], dim_max = 2)

    #momentMaps = moment_map.(persistenceDiagrams, 4)
    #momentMatrix =  reduce(vcat,transpose.(momentMaps))

    #pdDistanceMatrix = computePersistenceDistances(persistenceDiagrams)
    #@show "done"
    #@time  

    #plot!( axI3, persistenceDiagrams )

    rescale(A; dims=1) = (A .- mean(A, dims=dims)) ./ max.(std(A, dims=dims), eps())
    featureVectorMatrix = featureVectorMatrix |> rescale

    #invariantMomentFeatures = rescale(invariantMomentFeatures; )


    #tsne(X, ndim, reduce_dims, max_iter, perplexit; [keyword arguments])
    #@time Y = tsne(featureVectorMatrix[:, :], 2, 20, 400, 50.0;)
    #@time Y = tsne(invariantMomentFeatures, 2, 20, 1000, 30.0; )
    #@time Y = umap(pdDistanceMatrix, 2; metric=:precomputed, n_neighbors=20)
    #@time Y = umap(transpose(momentMatrix), 2; metric=Cityblock(), n_neighbors=50)


    #tsnePlot = scatter!(axDR, Y, color = labels, colormap = (:viridis, 0.5))
    #tsnePlot = scatter!(axDR, invariantMomentFeatures[:,3:4], color = labels, colormap = (:viridis, 0.5))

    #@time embedding = ManifoldLearning.fit(DiffMap, transpose(featureVectorMatrix[:, :]);maxoutdim=3,t=2, α=0.0, ɛ=1.0)
    #Y = predict(embedding) 
    #tsnePlot = scatter!(axDR, Y, color = labels, colormap = (:viridis, 1.0))

    on(events(plotWindow).mousebutton, priority=2) do event
        if event.button == Mouse.left && event.action == Mouse.press
            # Delete marker
            plt, i = pick(axDR)

            if plt == tsnePlot
                # deleteat!(positions[], i)
                currentTransition[] = mapIdxToName[i]
                notify(currentTransition)
                return Consume(true)
            end
            return Consume(false)
        end
        return Consume(false)
    end




    sequence = getSequence(sequencePath)

    transitionSequence = Vector{String}()

    selectedAtom = Observable{Int64}(0)

    for sequenceStep = 1:(length(sequence)-1)
        currentState = sequence[sequenceStep]
        nextState = sequence[sequenceStep+1]
        transitionName = currentState * ">" * nextState
        push!(transitionSequence, transitionName)

    end

    sliderTransition = SliderGrid(
        molWindow[7, 1:6],
        (
            label="Transition",
            range=[1:length(transitionSequence);],
            startvalue=1,
            format=x -> string(transitionSequence[x]),
        ),
    )




    currentTransition = lift(sliderTransition.sliders[1].value) do val
        return transitionSequence[val]
    end

    currentStatePair = lift(sliderTransition.sliders[1].value) do val
        transString = transitionSequence[val]
        return split(transString, ">")
    end

    lineSets = Dict{String,Tuple{Vector{Tuple{Point3f,Point3f}},Vector{Float64}}}()
    @time for (key, value) in distanceMatrices
        points = Vector{Tuple{Point3f,Point3f}}()
        weights = Vector{Float64}()
        for i in 1:length(value[1, :])
            for j in 1:i
                bw = bondWeights[key][i, j]
                if bw > 0.0
                    push!(points, (Point3f(atomPositions[key][i, :]), Point3f(atomPositions[key][j, :])))
                    push!(weights, bw)
                end
            end
        end
        max_weight = maximum(max, weights)
        colors = map(x -> invLerp(0.0, max_weight, x), weights)
        lineSets[key] = (points, colors)
    end

    linesegments!(lscenePre,
        lift(x -> lineSets[x[1]][1], currentStatePair),
        color=lift(x -> lineSets[x[1]][2], currentStatePair),
        inspector_label=(self, idx, pos) -> string("Weight ", self.color[][idx]),
        lowclip=:black,
        colorrange=(0.0, 1.0),
        colormap=:heat)

    linesegments!(lscenePost,
        lift(x -> lineSets[x[2]][1], currentStatePair),
        color=lift(x -> lineSets[x[2]][2], currentStatePair),
        inspector_label=(self, idx, pos) -> string("Weight ", self.color[][idx]),
        colorrange=(0.0, 1.0),
        lowclip=:black,
        colormap=:heat)

    on(currentTransition) do val
        Threads.@threads for i in eachindex(sampleRangeX) # x
            for j in eachindex(sampleRangeY) # y
                for k in eachindex(sampleRangeZ) # z
                    point = Point3f(sampleRangeX[i], sampleRangeY[j], sampleRangeZ[k])
                    knn, dists = NearestNeighbors.knn(transitionKDTree[val], point, 5)
                    kValue = sum(kernelFunction.(Ref(point), transitionRefPositions[val][knn], kernelWidth) .* transitionInvariants1[val][knn])
                    volumeData[][i, j, k] = kValue
                end
            end
        end
        volumeAbsMax[] = max(abs(minimum(volumeData[])), abs(maximum(volumeData[])))
        #@show volumeAbsMax[]
        notify(volumeAbsMax)
        notify(volumeData)
    end


    scatter!(
        lscenePre,
        lift(x -> atomPositions[x[1]], currentStatePair);
        markersize=25,
        color=:gray,
        colormap=:bam,
        colorrange=(-0.4, 0.4),
        inspector_label=(self, idx, pos) -> string("Atom ", idx)
    )
    scatter!(
        lscenePost,
        lift(x -> atomPositions[x[2]], currentStatePair);
        markersize=25,
        color=:gray,
        colormap=:bam,
        colorrange=(-0.4, 0.4),
        inspector_label=(self, idx, pos) -> string("Atom ", idx)
    )

    DataInspector(lscenePre)
    DataInspector(lscenePost)

    testCase = stretchedPrincipalAxes["1>3"]

    # glyps = meshscatter!(
    #     lsceneRight,
    #     lift(x -> transitionRefPositions[x], currentTransition);
    #     markersize = lift(x -> x, transitionGlyphSize),
    #     marker=superquadric( testCase[1],  0.99, 1.0 ,0.3),
    #     color = lift(x -> transitionInvariants1[x], currentTransition),
    #     colormap = :bwr,
    #     colorrange = (-0.4, 0.4),
    # )



    glyphResolution = 0.1

    glyps = mesh!(
        lsceneRightVolume,
        lift((x, y) -> superquadric.(y, transitionRefPositions[x], stretchedPrincipalAxes[x], transitionInvariants2[x], -1.0, 3.0, glyphResolution)[:], currentTransition, transitionGlyphSize),
        transparency=false,
        color=:white,
        #color = lift(x -> transitionInvariants1[x], currentTransition),
        #colormap = :bam,
        #colorrange = (-invariant1MaxRange, invariant1MaxRange),
        fxaa=false,
        alpha=1.0,
        visible=false,
    )
    #Colorbar(molWindow[6,4:6], glyps, vertical = false)

    cmap = resample_cmap(:bam, 100; alpha=([(-0.99):0.02:(0.99);] ./ 0.1) .^ 6)
    @show "cmap Range"
    @show length(cmap)
    # r = 15:30


    vol = volume!(lsceneRightVolume, sampleRangeX, sampleRangeY, sampleRangeZ,
        lift(x -> x, volumeData);
        colormap=cmap,
        algorithm=:absorption,
        #isorange = 0.000001,
        #isovalue = 0.0,
        #colorscale = abs,
        #absorption= lift(x->x, transitionGlyphSize),
        fxaa=false,
        transparency=true,
        shading=NoShading,
        colorrange=lift(x -> (-x, x), volumeAbsMax),
        visible=true,)

    Colorbar(molWindow[6, 4:6], vol, vertical=false)



    on(events(lsceneRightVolume).mousebutton, priority=2) do event
        if event.button == Mouse.left && event.action == Mouse.pressed
            # Delete marker
            plt, i = pick(glyps)
            # @show plt
            # @show glyps
            # #@show plt.plots
            # @show glyps.plots[1]

            if plt == glyps.plots[1]
                #@show i
                #@show 
                idx = getIndexFromSQMesh(i, glyphResolution)
                selectedAtom[] = idx

                return Consume(false)
            end
            return Consume(false)
        end
        return Consume(false)
    end



    #@show typeof(testCase)
    # @show keys(stretchedPrincipalAxes)

    #meshscatter!( lsceneRight, lift(x -> transitionRefPositions[x], currentTransition);markersize = lift(x -> x, transitionGlyphSize) , marker=superquadric( testCase[1],  0.99, 1.0 ,0.3) )


    stem!(axI1, atoms, lift(x -> transitionInvariants1[x], currentTransition))

    stem!(axI2, atoms, lift(x -> transitionInvariants2[x], currentTransition))

    #stem!(axI3, atoms, lift(x -> transitionInvariants3[x], currentTransition))
    stem!(axI3, invariantMomentFeatures[:, 3])



    link_cameras_lscene(molWindow)


    screen1 = GLMakie.Screen()
    #screen2 = GLMakie.Screen()

    display(screen1, molWindow)
    #display(screen2, plotWindow)


    #animation stuff
    # fps = 30
    # nframes = 120

    # keep function alive while program is running
    # while true || !nothing(screen2)
    #     sleep(1 / (fps * abs(speedFactor[])))
    # end


end


end # module TransVis
