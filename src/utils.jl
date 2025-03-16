using GLMakie: ScreenConfig
function pair_array(v)
    pairs = Vector{Pair{Any,Any}}()
    for i in 1:length(v)-1
        e1 = v[i]
        e2 = v[i+1]
        push!(pairs, Pair(e1, e2))
    end
    return pairs
end

function link_cameras_lscene(f; step=0.01)
    scenes = vcat(map(y -> filter(x -> x isa LScene, y.content), f)...)
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

function link_cameras_lscenes(scenes; step=0.01)
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
end

# normalizes an array of matrices
function normalize_matrices(data)
    d = collect(values(data))

    # simple min-max norm
    max_val = maximum(map((x) -> maximum(x), d))
    min_val = minimum(map((x) -> minimum(x), d))

    norm = Dict{Tuple{Int,Int},Matrix}()

    for (transition, val) in data
        norm[transition] = (val .- min_val) / (max_val - min_val)
    end

    return norm, min_val, max_val
end

splitobs(o::Observable{Tuple{}}) = ()
splitobs(o::Observable{<:Tuple}) = (lift(first, o), splitobs(lift(Base.tail, o))...)

function str_limit(s; len=40)
    return "$(string(s)[1:min(end, len)])$(length(string(s)) > len ? "..." : "")"
end

function set_color_alpha(c, a)
    return RGBAf(c.r, c.g, c.b, a)
end

function cycle_colormap(i, cmap)
    return cmap[mod1(i, length(cmap))]
end

# leaving this an example
# to render a view to an image, you need to create a new screen for it, render to it
# and then save its information to a buffer, which is what show() does.

function snapshot_view(render_fn)
    fig = Figure()
    campixel!(fig.scene)
    ax = LScene(fig[1, 1], show_axis=false, scenekw=(clear=true, backgroundcolor=:black))
    cam3d!(ax.scene)

    fn = function (t)
        # even though the screen isn't visible, it will still render to a colorbuffer which show() dumps internally into the iobuffer
        config = Makie.merge_screen_config(ScreenConfig, Dict{Symbol,Any}(:visible => false))
        buf = IOBuffer()
        s = Screen(fig.scene, config, buf, MIME"image/png"())

        plot = render_fn[](ax, t)
        # make sure that you are updating the camera to the new limits inside the render fn
        center!(ax.scene)

        show(buf, MIME"image/png"(), ax.scene, update=false)
        img = deepcopy(FileIO.load(Stream{FileIO.format"PNG"}(buf)))
        close(buf)
        close(s)
        # in case render_fn returns multiple plots
        foreach(x -> delete!(ax, plot), collect(plot))
        return img
    end

    return fn
end
