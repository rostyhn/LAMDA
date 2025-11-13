function default_cache()::String
    if Sys.iswindows()
        error("Not implemented for Window yet")
    else
        hdir = abspath(homedir())
        p = joinpath([hdir, ".cache", "LAMDA"])
        mkpath(p)
        return p
    end
end

function clear_listener_list(xs)
    for x in xs
        off(x)
        x = nothing
    end
    empty!(xs)
end

function link_cameras_lscene(f; step::AbstractFloat=0.01)
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

function link_cameras_lscenes(scenes; step::AbstractFloat=0.01)
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

function screenshot_3d!(scene::Makie.Scene, render_fn::Function, fname::String, width=1920)
    sw, sh = widths(scene.viewport[])
    aspect = sh / sw
    height = width * aspect

    f = Figure(size=(width, height))
    ax = LScene(f.scene,
        show_axis=false,
        scenekw=(backgroundcolor=EMBEDDED_SCENE_BACKGROUND, camera=cam3d!, clear=true),
    )
    f[1, 1] = ax

    render_fn(ax)
    update_cam!(ax, cameracontrols(scene))
    GLMakie.save(fname, f, update=false)
    @info "Saved screenshot to $(relative_path)/f"

    empty!(f)
    Makie.free(ax.scene)
    Makie.free(f)
end

function screenshot_3d!(render_fn, fname, args...; width=1000, height=1000)
    f = Figure(size=(width, height))
    ax = LScene(f.scene,
        show_axis=false,
        scenekw=(backgroundcolor=EMBEDDED_SCENE_BACKGROUND, camera=cam3d!, clear=true),
    )
    f[1, 1] = ax

    render_fn(ax.scene, args...)

    update_cam!(ax.scene)
    center!(ax.scene)
    GLMakie.save(fname, f, update=true)

    empty!(f)
    Makie.free(ax.scene)
    Makie.free(f.scene)
end


splitobs(o::Observable{Tuple{}}) = ()
splitobs(o::Observable{<:Tuple}) = (lift(first, o), splitobs(lift(Base.tail, o))...)

function str_limit(s; len::Integer=25, ending="...")
    return "$(string(s)[1:min(end, len)])$(length(string(s)) > len ? ending : "")"
end

function set_color_alpha(c::RGBAf, a::AbstractFloat)
    return RGBAf(c.r, c.g, c.b, a)
end

function cycle_colormap(i::Integer, cmap; alpha::AbstractFloat=1.0)
    return set_color_alpha(cmap[mod1(i, length(cmap))], alpha)
end

function disable_interactions(ax)
    for x in keys(interactions(ax))
        deactivate_interaction!(ax, x)
    end
end

function enable_interactions(ax)
    for x in keys(interactions(ax))
        activate_interaction!(ax, x)
    end
end

function filesafestr(s::String)
    # https://stackoverflow.com/questions/42210199/remove-illegal-characters-from-a-file-name-but-leave-spaces
    re = r"[\\\\/:*?\"<>|\[\]\(\) ]"
    cre = r"[\,\.]"
    return str_limit(hash(s), len=25) #str_limit(replace(s, re => "", cre => "_"), len=125, end
end

function relative_path(s::String)
    rootPath = dirname(dirname(@__FILE__))
    p = joinpath(rootPath, s)
    return p
end
