module TransVis

#Data handling
using GLMakie: apply_transform
using Makie: MakieCore, ray_at_cursor, position_on_plot, mouse_in_scene, shift_project, update_tooltip_alignment!, parent_scene, show_data
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
include("SelectionWindow.jl")
include("MolWindow.jl")

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

function buildBonds(key, volumeDataDict, bondDelta, volumeAbsMax, atomPositions)
    points = Vector{Tuple{Point3f,Point3f}}()
    weights = Vector{Float64}()
    indices = Vector{Tuple{Int64,Int64}}()
    for i in 1:length(bondDelta[1, :])
        for j in 1:i
            v1 = volumeDataDict[i]
            v2 = volumeDataDict[j]
            bw = bondDelta[i, j]

            avg = (abs((v1 + v2)) / 2) / volumeAbsMax
            # 0.05 is the threshold val for filtering
            # check against bond weight to make sure we're only looking at "real" bonds
            if abs(bw) > 0.0
                push!(points, (Point3f(atomPositions[key][i, :]), Point3f(atomPositions[key][j, :])))
                push!(weights, bw)
                push!(indices, (i, j))
            end
        end
    end
    return (points, weights, indices)
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

    atomView = LScene(
        molWindow[1:5, 1:3],
        show_axis=false,
        scenekw=(backgroundcolor=:white, clear=true),
    )

    volumeView = LScene(
        molWindow[1:5, 4:6],
        show_axis=false,
        scenekw=(backgroundcolor=:white, clear=true),
    )

    axI1 = Axis(plotWindow[1:2, 1:4], xlabel="Atom Number", ylabel="K1")
    axI2 = Axis(plotWindow[3:4, 1:4], xlabel="Atom Number", ylabel="K2")
    axI3 = Axis(plotWindow[5:6, 1:4], xlabel="Atom Number", ylabel="mode(E)")
    axDR = Axis(plotWindow[1:6, 5:7], title="t-SNE")

    transitionGlyphSizeSlider = Slider(molWindow[4:6, 7], range=0.1:0.01:4, horizontal=false, startvalue=1)

    stateDataPath = "/home/frosty/Programs/Julia/TransVis/data/state_data copy/"
    sequencePath = "/home/frosty/Programs/Julia/TransVis/data/state_data copy/seq.txt"
    transitionPath = "/home/frosty/Programs/Julia/TransVis/data/nano_pt_labels.pickle"
    bondWeightPath = "/home/frosty/Programs/Julia/TransVis/data/bond_weights.pickle"
    connectivityPath = "/home/frosty/Programs/Julia/TransVis/data/connectivity.pickle"
    transitionDistanceMatrixPath = "/home/frosty/Programs/Julia/TransVis/data/distance_matrix.pickle"
    transitionSequencePath = "/home/frosty/Programs/Julia/TransVis/data/transition_sequence.pickle"

    combinedData = getDataSets(stateDataPath, sequencePath, transitionPath, bondWeightPath, connectivityPath, transitionDistanceMatrixPath, transitionSequencePath)

    transitionInvariants1 = combinedData["transitionInvariants1"]
    transitionInvariants2 = combinedData["transitionInvariants2"]
    transitionInvariants3 = combinedData["transitionInvariants3"]

    transitionDistanceMatrix = combinedData["transitionDistanceMatrix"]["matrix"]
    transitionSequence = combinedData["transitionSequence"]["sequence"]
    transitionRefPositions = combinedData["transitionRefPositions"]
    transitionLabels = combinedData["transitionLabels"]

    distanceMatrices = combinedData["distanceMatrices"]

    stretchedPrincipalAxes = combinedData["stretchedPrincipalAxes"]

    atomPositions = combinedData["atomPositions"]

    connectivity = combinedData["connectivity"]
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
    transitionKDTree = Dict{Tuple{Int,Int},KDTree}()
    @time for (key, value) in transitionRefPositions
        transitionKDTree[key] = KDTree(value)
    end

    kernelWidth = 1.0

    volumeData = Observable(zeros(length(sampleRangeX), length(sampleRangeY), length(sampleRangeZ)))
    volumeDataDict = Observable(Dict{Int,Any}())

    volumeAbsMax = Observable(1.0)
    volumeMin = Observable(1.0)

    transitionGlyphSize = lift(transitionGlyphSizeSlider.value) do val
        return val
    end

    atoms = [1:1:length(values(transitionInvariants1) |> first);]

    featureVectorMatrix = zeros(
        length(values(transitionInvariants1)),
        length(values(transitionInvariants1) |> first) * 1,
    )
    labels = zeros(length(values(transitionInvariants1)))


    mapNameToIdx = Dict{Tuple{Int,Int},Int64}()
    mapIdxToName = Vector{Tuple{Int,Int}}()

    row = 1
    for (transition, invariants1) in transitionInvariants1
        mapNameToIdx[transition] = row
        push!(mapIdxToName, transition)

        col = 1
        for invariant in invariants1
            featureVectorMatrix[row, col] = invariant |> abs
            col = col + 1
        end

        s1, s2 = transition
        flippedName = (s2, s1)

        if haskey(transitionLabels, transition)
            labels[row] = transitionLabels[transition]
        elseif haskey(transitionLabels, flippedName)
            labels[row] = transitionLabels[flippedName]
        else
            println("Transition not found: $(string(transition))")
        end

        row = row + 1
    end

    #build distance distanceMatrices
    invariantDistances = zeros(length(transitionInvariants1[(1, 3)]), length(transitionInvariants1[(1, 3)])) #nAtoms x nAtoms
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


    @show size(transitionRefPositions[(1, 3)])
    @show size(transitionInvariants1[(1, 3)])

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

    @lift begin
        for i in eachindex(sampleRangeX) # x
            for j in eachindex(sampleRangeY) # y
                for k in eachindex(sampleRangeZ) # z
                    point = Point3f(sampleRangeX[i], sampleRangeY[j], sampleRangeZ[k])
                    knn, dists = NearestNeighbors.knn(transitionKDTree[$currentTransition], point, 5)
                    kValue = sum(kernelFunction.(Ref(point), transitionRefPositions[$currentTransition][knn], kernelWidth) .* transitionInvariants1[$currentTransition][knn])
                    volumeData[][i, j, k] = kValue
                    idx, d = NearestNeighbors.nn(transitionKDTree[$currentTransition], point)
                    volumeDataDict[][idx] = kValue
                end
            end
        end
        volumeAbsMax[] = max(abs(minimum(volumeData[])), abs(maximum(volumeData[])))
        volumeMin[] = minimum(volumeData[])
        notify(volumeAbsMax)
        notify(volumeData)
    end

    # 6 is the slope - should only be even odds
    # 0.1 is the thickness of the white part
    cmap = resample_cmap(:bam, 100; alpha=([(-0.99):0.02:(0.99);] ./ 0.1) .^ 6)
    @show "cmap Range"
    @show length(cmap)

    ap1 = lift(x -> atomPositions[x[1]], currentTransition)

    aa1 = @lift begin
        return map(x -> get(volumeDataDict[], x[1], 0.0), enumerate(eachrow($ap1)))
    end

    filterRange = @lift begin
        mm = extrema($aa1)
        return LinRange(mm[1], mm[2], 100)
    end

    volFilter = IntervalSlider(molWindow[6, 1:3], range=filterRange[], startvalues=(0.0001, -0.0001))
    Label(molWindow[5, 1], lift(x -> string(x), volFilter.interval))
    filtered = lift(volFilter.interval) do interval
        filtered = Vector{Int64}()
        for (i, v) in enumerate(aa1[])
            # inverse filter, blue area will be removed!
            if v < interval[1] || v > interval[2]
                push!(filtered, i)
            end
        end
        return filtered
    end

    glyphResolution = 0.1

    superquadrics = lift((x, y) -> superquadric.(y, transitionRefPositions[x], stretchedPrincipalAxes[x], transitionInvariants2[x], -1.0, 3.0, glyphResolution)[:], currentTransition, transitionGlyphSize)
    glyps = mesh!(
        atomView,
        lift((x, y) -> x[y], superquadrics, filtered),
        color=lift(x -> aa1[][x], filtered),
        # prevents it from recoloring each time the slider moves
        colorrange=lift(x -> extrema(x), aa1),
        colormap=:bam,
        fxaa=false,
    )
    glyps.inspectable[] = false

    # https://github.com/MakieOrg/Makie.jl/blob/master/src/interaction/ray_casting.jl

    inspector = DataInspector(atomView)

    on(events(atomView).mouseposition) do mp
        plot, idx = pick(glyps)
        if plot == glyps.plots[1]
            pos = position_on_plot(plot, idx)
            idx, d = NearestNeighbors.nn(transitionKDTree[currentTransition[]], pos)
            if !isnan(pos)
                inspector.plot.text[] = string("Atom ", idx)
                inspector.plot.visible[] = true
                inspector.plot.position = mp
                return Consume(true)
            end
        end
        return Consume(false)
    end

    bondDeltas = Dict{Tuple{Int,Int},Matrix{Float64}}()
    transforms = Dict{Tuple{Int,Int},Matrix{Float64}}()
    for t in transitionSequence
        dm1 = distanceMatrices[t[1]] .* connectivity[t[1]]'
        dm2 = distanceMatrices[t[2]] .* connectivity[t[2]]'

        #totalDistanceMatrix = (dm2 + dm1) .+ 0.0000001

        # for now it's total delta
        bondDeltas[t] = (dm2 - dm1)

        p1 = atomPositions[t[1]]
        p2 = atomPositions[t[2]]
        transforms[t] = (abs.(p2 - p1))
    end

    lineSets = @lift begin
        bondDelta = bondDeltas[$currentTransition]
        bonds = buildBonds($currentTransition[1], $volumeDataDict, bondDelta, $volumeAbsMax, atomPositions)
        return bonds[1], bonds[2]
    end

    linesegments!(atomView,
        lift(x -> x[1], lineSets),
        color=lift(x -> x[2], lineSets),
        inspector_label=(self, idx, pos) -> string("Weight ", self.color[][idx]),
        lowclip=:black,
        colormap=:bam)
    # r = 15:30

    vol = volume!(volumeView, sampleRangeX, sampleRangeY, sampleRangeZ,
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
        visible=true)

    Colorbar(molWindow[6, 4:6], vol, vertical=false)

    stem!(axI1, atoms, lift(x -> transitionInvariants1[x], currentTransition))
    stem!(axI2, atoms, lift(x -> transitionInvariants2[x], currentTransition))
    stem!(axI3, invariantMomentFeatures[:, 3])

    link_cameras_lscene(molWindow)

    function on_click(t)
        # atom positions for transition
        # volumeData, voluemDataDict, volumeAbsMax
        # superquadrics, linesets,
        # kdTree
        # sampleRangex,y,z, cmap
        atomPosTuple = (atomPositions[t[1]], atomPositions[t[2]])
        volData = zeros(length(sampleRangeX), length(sampleRangeY), length(sampleRangeZ))
        volDataDict = Dict{Int,Any}()

        kdTree = transitionKDTree[t]
        for i in eachindex(sampleRangeX) # x
            for j in eachindex(sampleRangeY) # y
                for k in eachindex(sampleRangeZ) # z
                    point = Point3f(sampleRangeX[i], sampleRangeY[j], sampleRangeZ[k])
                    knn, dists = NearestNeighbors.knn(kdTree, point, 5)
                    kValue = sum(kernelFunction.(Ref(point), transitionRefPositions[t][knn], kernelWidth) .* transitionInvariants1[t][knn])
                    volData[i, j, k] = kValue
                    idx, d = NearestNeighbors.nn(kdTree, point)
                    volDataDict[idx] = kValue
                end
            end
        end
        volAbsMax = max(abs(minimum(volData)), abs(maximum(volData)))
        volMin = minimum(volData)

        # 1.0 should be transitionGlyphSize
        sq = superquadric.(1.0, transitionRefPositions[t], stretchedPrincipalAxes[t], transitionInvariants2[t], -1.0, 3.0, glyphResolution)[:]
        ls = buildBonds(t[1], volDataDict, bondDeltas[t], volAbsMax, atomPositions)

        build_mol_window((600, 400), t, atomPosTuple, volData, volAbsMax, volDataDict, sq, ls, kdTree, sampleRangeX, sampleRangeY, sampleRangeZ, cmap)
    end

    #screen1 = GLMakie.Screen()
    screen2 = GLMakie.Screen()

    sorted = sort_transitions(transitionSequence[1], transitionSequence, transitionDistanceMatrix)
    #display(screen1, molWindow)
    #display(screen2, plotWindow)
    display(screen2, build_selection_window((600, 800), transforms, sorted, on_click))
end
end
