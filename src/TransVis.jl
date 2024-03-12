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

include("io.jl")
include("processing.jl")


export go


function transformSQPoint(  principarStretches::Vector{GeometryBasics.Vec{3, Float32}}, point::Point3f )

    # @show point[1]
    # @show principarStretches[1]

    # @show norm(principarStretches[1])

    return Point3f( 3 * point[1], 
    0.2 * point[2], 
    0.2 * point[3])

    # return Point3f( norm(principarStretches[1]) * point[1], 
    #         norm(principarStretches[2]) * point[2], 
    #         norm(principarStretches[3]) * point[3])

end

function signPow(base, exponent)::Float64
    return sign(base)* abs(base)^exponent
end

function qz( phi::Float64, theta::Float64, alpha::Float64, beta::Float64, K2::Float64  )
    x = signPow( cos(theta), alpha ) *  signPow(sin(phi), beta) 
    y = signPow( sin(theta), alpha ) *  signPow( sin(phi), beta) 
    z = signPow( cos(phi), beta )

    return Point3f(x,y,z)
end 

function qx( phi::Float64, theta::Float64, alpha::Float64, beta::Float64, K2::Float64  )
    x = signPow( cos(phi), beta ) 
    y = -signPow( sin(theta), alpha ) *  signPow( sin(phi), beta )
    z = signPow( cos(theta), alpha) *  signPow( sin(phi), beta)
    return Point3f(x,y,z)

end 

function superquadric(scale::Float64, scene ,position::Point3f ,principalStretches::Vector{GeometryBasics.Vec{3, Float32}}, K2::Float64, K3::Float64, sharpness::Float64, resolution=0.2 )
    points = Vector{Point3f}()

    #K2 is the volume perserving fractionalAnisotropy
    #K3 is the mode defining the type of anisotropy: -1 planar to 1 linear

    @show scale


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
    cp = 2*(stretchRatio2 - stretchRatio3) /  (stretchRatio1 + stretchRatio2 + stretchRatio3) # planar anisotropy
    cs =  3*stretchRatio3 / (stretchRatio1 + stretchRatio2 + stretchRatio3)

    # @show cl
    # @show cp
    # @show cs
    @show stretchRatio1 * stretchRatio1 * stretchRatio1

    @show cs

    phiRange = [0:resolution:pi;]  #vertical: south -> north
    push!(phiRange, pi) #ass pi to close the hole at the end introduced by resolution
    thetaRange = [0:resolution:2*pi;] #horizontal: west -> east


    if cl >= cp
        alpha = signPow((1 - cp), sharpness)
        beta = signPow((1 - cl),sharpness)

        for phi in phiRange
            for theta in thetaRange
                push!(points, qx( phi, theta, alpha, beta, K2 ))
            end
        end
    else
        alpha = (1 - cl)^sharpness
        beta = (1 - cp)^sharpness

        for phi in phiRange
            for theta in thetaRange
                push!(points, qz( phi, theta, alpha, beta, K2 ) )
            end
        end
    end


    scaleMatrix = zeros(3,3)
    scaleMatrix[1,1] = stretchRatio1
    scaleMatrix[2,2] = stretchRatio2
    scaleMatrix[3,3] = stretchRatio3

    scaleMatrix = scaleMatrix*scale

    rotationMatrix = zeros(3,3)
    rotationMatrix[:,1] = stretchDirection1 
    rotationMatrix[:,2] = stretchDirection2
    rotationMatrix[:,3] = stretchDirection3 

    if  det(rotationMatrix) < 0
        rotationMatrix[:,1] = -1 * rotationMatrix[:,1]
    end

    transform = rotationMatrix * scaleMatrix

    #meshscatter!( scene, position[1],position[2],position[3]; color= :black)
    #meshscatter!( scene, position[1] + 5*stretchDirection1[1], position[2]+5*stretchDirection1[2], position[3]+5*stretchDirection1[3]; color= :black)
   # @show scaleMatrix
   # @show points[1]
    points = Point3f.(  Ref(transform) .* points ) .+  Ref(position)
   # @show points[1]

    #scatter!(scene, points)
    
    nPhi = length(phiRange)
    nTheta = length(thetaRange)

    indices = Vector{Tuple{UInt32,UInt32,UInt32}}() # triangles over the points

    for y in 1:(nPhi-1)
        for x in 1:nTheta

            #@show "???"
            p11 = x + nTheta*(y - 1)
            p21 = x < nTheta ? (x+1) + nTheta*(y -1) : 1+ nTheta*(y - 1) 
            p31 = x + nTheta*(y)

            p12 = x < nTheta ? (x+1) + nTheta*(y -1) : 1+ nTheta*(y - 1) 
            p22 =  x < nTheta ? (x+1) + nTheta*y : 1+ nTheta*y # index 
            p32 = x + nTheta*(y) 

            push!(indices, (p11, p31, p21))
            push!(indices, (p32, p22, p12 ))

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

function link_cameras_lscene(f; step = 0.01)
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

#Runs the logic
# function activate(app)

#     print( "Using $(Threads.nthreads()) threads\n")

#     screen = Gtk4Makie.GTKScreen(resolution=(800, 800),title="TransVis",app=app)


#     plotWindow = Figure()
#     molWindow = Figure()


#     lsceneLeft = LScene(molWindow[1:3, 1:3], show_axis=false, scenekw = (backgroundcolor = :whitesmoke, clear = true))
#     lsceneRight = LScene(molWindow[1:3, 4:6], show_axis=false, scenekw = (backgroundcolor = :whitesmoke, clear = true))
#     #lscenec = LScene(fig[1:2, 7:9], show_axis=false, scenekw = (backgroundcolor = :whitesmoke, clear = true))
#     #lscene2 = LScene(fig[3:4, 1:3], show_axis=false, scenekw = (backgroundcolor = :whitesmoke, clear = true))

#     axI1 = Axis(plotWindow[1:2, 1:4], xlabel = "Atom Number", ylabel = "Invariant 1")
#     axI2 = Axis(plotWindow[3:4, 1:4], xlabel = "Atom Number", ylabel = "Invariant 2")
#     axI3 = Axis(plotWindow[5:6, 1:4], xlabel = "Atom Number", ylabel = "Invariant 3")

#     axDR = Axis(plotWindow[1:6, 5:7],  title = "t-SNE")
#     #axDR = LScene(fig[5:10, 5:6], show_axis=false, scenekw = (backgroundcolor = :white, clear = true))

#     # axI1 = PolarAxis(fig[4:5, 1:6], title = "Transition Invariants")
#     # axI2 = PolarAxis(fig[6:7, 1:6], title = "Transition Invariants")
#     # axI3 = PolarAxis(fig[8:9, 1:6], title = "Transition Invariants")




#     display(screen, lines(rand(10)))
#     ax=current_axis()
#     f=current_figure()

#     g=grid(screen)

#     #g[1,2]=GtkButton("Generate new random plot")

#     #function gen_cb(b)
#     #    empty!(ax)
#     #    lines!(ax,rand(10))
#     #end

#     signal_connect(gen_cb,g[1,2],"clicked")
# end



function go()

    GLMakie.closeall() #close all windows for rerun!

    plotWindow = Figure( size=(600,400))
    molWindow = Figure( size=(600,400))


    lscenePre = LScene(
        molWindow[1:3, 1:3],
        show_axis = false,
        scenekw = (backgroundcolor = :white, clear = true),
    )
    lscenePost = LScene(
        molWindow[4:6, 1:3],
        show_axis = false,
        scenekw = (backgroundcolor = :white, clear = true),
    )
    
    lsceneRight = LScene(
        molWindow[1:5, 4:6],
        show_axis = false,
        scenekw = (backgroundcolor = :white, clear = true),
    )



    axI1 = Axis(plotWindow[1:2, 1:4], xlabel = "Atom Number", ylabel = "Invariant 1")
    axI2 = Axis(plotWindow[3:4, 1:4], xlabel = "Atom Number", ylabel = "Invariant 2")
    axI3 = Axis(plotWindow[5:6, 1:4], xlabel = "Atom Number", ylabel = "Invariant 3")
    axDR = Axis(plotWindow[1:6, 5:7], title = "t-SNE")




    stateDataPath = "/Users/Bote/Documents/ASU/state_data copy/"
    sequencePath = "/Users/Bote/Documents/ASU/state_data copy/seq.txt"
    transitionPath = "/Users/Bote/Documents/ASU/nano_pt_labels.pickle"

    combinedData = getDataSets(stateDataPath, sequencePath, transitionPath)

    transitionInvariants1 = combinedData["transitionInvariants1"]
    transitionInvariants2 = combinedData["transitionInvariants2"]
    transitionInvariants3 = combinedData["transitionInvariants3"]

    transitionRefPositions = combinedData["transitionRefPositions"]
    transitionLabels = combinedData["transitionLabels"]

    stretchedPrincipalAxes = combinedData["stretchedPrincipalAxes"]


    atoms = [1:1:length(values(transitionInvariants1) |> first);]

    #   balltree = BallTree(data, Minkowski(3.5); reorder = false)

    # lines!.(axI1, Ref(atoms), values(transitionInvariants1), alpha = 0.01)
    # lines!.(axI2, Ref(atoms), values(transitionInvariants2), alpha = 0.01)
    # lines!.(axI3, Ref(atoms), values(transitionInvariants3), alpha = 0.01)


    featureVectorMatrix = zeros(
        length(values(transitionInvariants1)),
        length(values(transitionInvariants1) |> first) * 3,
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

        col = length(values(transitionInvariants1) |> first) + 1
        for invariant in transitionInvariants2[transition]
            featureVectorMatrix[row, col] = invariant |> abs
            col = col + 1
        end

        col = length(values(transitionInvariants1) |> first) * 2 + 1
        for invariant in transitionInvariants3[transition]
            featureVectorMatrix[row, col] = invariant |> abs
            col = col + 1
        end


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




    #rescale(A; dims=1) = (A .- mean(A, dims=dims)) ./ max.(std(A, dims=dims), eps())
    #featureVectorMatrix = featureVectorMatrix |> rescale

    #tsne(X, ndim, reduce_dims, max_iter, perplexit; [keyword arguments])
    @time Y = tsne(featureVectorMatrix[:, :], 2, 20, 20, 40.0;)
    tsnePlot = scatter!(axDR, Y, color = labels, colormap = (:viridis, 1.0))

    on(events(plotWindow).mousebutton, priority = 2) do event
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

    for sequenceStep = 1:(length(sequence)-1)
        currentState = sequence[sequenceStep]
        nextState = sequence[sequenceStep+1]
        transitionName = currentState * ">" * nextState
        push!(transitionSequence, transitionName)

    end

    sliderTransition = SliderGrid(
        molWindow[7, 1:6],
        (
            label = "Transition",
            range = [1:length(transitionSequence);],
            startvalue = 1,
            format = x -> string(transitionSequence[x]),
        ),
    )

    transitionGlyphSizeSlider = Slider(molWindow[1:4, 7], range = 0.1:0.01:2, horizontal = false, startvalue = 1)



    currentTransition = lift(sliderTransition.sliders[1].value) do val
        return transitionSequence[val]
    end

    currentStatePair = lift(sliderTransition.sliders[1].value) do val
        transString = transitionSequence[val]
        return split( transString, ">" )
    end

    


    transitionGlyphSize = lift(transitionGlyphSizeSlider.value) do val
        return val
    end


    atomPositions = combinedData["atomPositions"]


    scatter!(
        lscenePre,
        lift(x -> atomPositions[x[1]], currentStatePair);
        markersize = 10,
        color = :gray,
        colormap = :bwr,
        colorrange = (-0.4, 0.4),
    )
    scatter!(
        lscenePost,
        lift(x ->  atomPositions[x[2]], currentStatePair);
        markersize = 10,
        color = :gray,
        colormap = :bwr,
        colorrange = (-0.4, 0.4),
    )

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
    

    glyps = mesh!( 
        lsceneRight, 
        lift((x,y) -> superquadric.(y, Ref(lsceneRight), transitionRefPositions[x] ,stretchedPrincipalAxes[x],  transitionInvariants2[x], -1.0 ,3.0)[:], currentTransition, transitionGlyphSize),
         transparency=false, 
         color = lift(x -> transitionInvariants1[x], currentTransition),
         colormap = :bwr,
         colorrange = (-0.4, 0.4),
         fxaa = true,
          )
    Colorbar(molWindow[6,4:6], glyps, vertical = false)



    
    #@show typeof(testCase)
   # @show keys(stretchedPrincipalAxes)

    #meshscatter!( lsceneRight, lift(x -> transitionRefPositions[x], currentTransition);markersize = lift(x -> x, transitionGlyphSize) , marker=superquadric( testCase[1],  0.99, 1.0 ,0.3) )


    stem!(axI1, atoms, lift(x -> transitionInvariants1[x], currentTransition))
    stem!(axI1, atoms, lift(x -> transitionInvariants1[x], currentTransition))

    stem!(axI2, atoms, lift(x -> transitionInvariants2[x], currentTransition))
    stem!(axI2, atoms, lift(x -> transitionInvariants2[x], currentTransition))

    stem!(axI3, atoms, lift(x -> transitionInvariants3[x], currentTransition))
    stem!(axI3, atoms, lift(x -> transitionInvariants3[x], currentTransition))


    link_cameras_lscene(molWindow)


    screen1 = GLMakie.Screen()
    screen2 = GLMakie.Screen()

    display(screen1, molWindow)
    display(screen2, plotWindow)


        #animation stuff
        # fps = 30
       # nframes = 120
    
        # keep function alive while program is running
        # while true || !nothing(screen2)
        #     sleep(1 / (fps * abs(speedFactor[])))
        # end
    

end


end # module TransVis
