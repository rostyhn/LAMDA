module TransVis

#Data handling
using Pickle
using JLD2
using CodecZlib

#GUI
using Gtk4
using Gtk4Makie

#Vis
using GLMakie
using Makie

#Processing and Helpers
using NearestNeighbors
#using Distances
using LinearAlgebra
using TSne
#using GeometryBasics
using Base.Threads

include("io.jl")
include("processing.jl")


export go


function link_cameras_lscene(f; step = 0.01)
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


    plotWindow = Figure()
    molWindow = Figure()


    lsceneLeft = LScene(
        molWindow[1:3, 1:3],
        show_axis = false,
        scenekw = (backgroundcolor = :whitesmoke, clear = true),
    )
    lsceneRight = LScene(
        molWindow[1:3, 4:6],
        show_axis = false,
        scenekw = (backgroundcolor = :whitesmoke, clear = true),
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



    atoms = [1:1:length(values(transitionInvariants1) |> first);]

    #   balltree = BallTree(data, Minkowski(3.5); reorder = false)

    lines!.(axI1, Ref(atoms), values(transitionInvariants1), alpha = 0.01)
    lines!.(axI2, Ref(atoms), values(transitionInvariants2), alpha = 0.01)
    lines!.(axI3, Ref(atoms), values(transitionInvariants3), alpha = 0.01)


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




    selectedTransition = Observable{String}("1>3")

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
                selectedTransition[] = mapIdxToName[i]
                notify(selectedTransition)
                @show plt
                @show i
                @show selectedTransition[]
                @show mapNameToIdx[mapIdxToName[i]]

                @show Y[mapNameToIdx[mapIdxToName[i]], :]
                #scatter!(axDR , Y[mapNameToIdx[mapIdxToName[i]],:], color=:black)
                return Consume(true)
            end
            return Consume(false)

        end
        # return Consume(false)
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
        molWindow[4, 1:6],
        (
            label = "Transition",
            range = [1:length(transitionSequence);],
            startvalue = 1,
            format = x -> string(transitionSequence[x]),
        ),
    )
    currentTransition = lift(sliderTransition.sliders[1].value) do val
        return transitionSequence[val]
    end


    scatter!(
        lsceneLeft,
        lift(x -> transitionRefPositions[x], currentTransition);
        markersize = lift(x -> abs.(transitionInvariants1[x]) * 70, currentTransition),
        color = lift(x -> transitionInvariants1[x], currentTransition),
        colormap = :bwr,
        colorrange = (-0.4, 0.4),
    )

    scatter!(
        lsceneRight,
        lift(x -> transitionRefPositions[x], selectedTransition);
        markersize = lift(x -> abs.(transitionInvariants1[x]) * 70, selectedTransition),
        color = lift(x -> transitionInvariants1[x], selectedTransition),
        colormap = :bwr,
        colorrange = (-0.4, 0.4),
    )

    stem!(axI1, atoms, lift(x -> transitionInvariants1[x], selectedTransition))
    stem!(axI1, atoms, lift(x -> transitionInvariants1[x], currentTransition))

    stem!(axI2, atoms, lift(x -> transitionInvariants2[x], selectedTransition))
    stem!(axI2, atoms, lift(x -> transitionInvariants2[x], currentTransition))

    stem!(axI3, atoms, lift(x -> transitionInvariants3[x], selectedTransition))
    stem!(axI3, atoms, lift(x -> transitionInvariants3[x], currentTransition))


    link_cameras_lscene(molWindow)


    screen1 = GLMakie.Screen()
    screen2 = GLMakie.Screen()

    display(screen1, molWindow)
    display(screen2, plotWindow)
end


end # module TransVis
