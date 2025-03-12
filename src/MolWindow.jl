using Makie: clear_temporary_plots!, Orthographic, GridLayout, clear!, GridLayoutBase

function build_mol_window(transition, t_idx, render_views, widgets; fig_size=(400, 400))
    molWindow = Figure(size=fig_size)
    menu_bar = top_bar(molWindow, "$(transition)", 2)

    setup_state_view!(molWindow, (2, 1), "Volume", transition, t_idx, render_views, widgets)
    setup_state_view!(molWindow, (2, 2), "Volume", transition, t_idx, render_views, widgets)

    screen = GLMakie.Screen(title="TransVis - $transition")
    display(screen, molWindow)

    #on(events(molWindow).entered_window) do is_hovered
    # on_window_hover(transition, is_hovered)
    #end
end

function setup_state_view!(fig, loc, startState, transition, t_idx, render_views, widgets)
    rootScene = LScene(
        fig,
        show_axis=false,
        scenekw=(backgroundcolor=:black, clear=true),
    )

    m = Menu(fig, options=SINGLE_TRANSITION_RENDER_OPTIONS,
        default=startState)

    i, j = loc
    g = vgrid!(m, rootScene)
    fig[i, j] = g

    inspector = DataInspector(rootScene)

    t_obs = Observable(transition)
    t_idx_obs = Observable(t_idx)

    # render_views expect Observables as input
    function choose_scene(selection)
        if selection == "Volume"
            render_views[selection](rootScene, t_idx_obs, t_obs)
            return [], []
        elseif selection == "Superquadric"
            il, is = render_views[selection](rootScene, inspector, t_obs)
            return il, []
        elseif selection == "Atom"
            gg, time, scalar_vals = widgets["Atom"](0.0, fig, g)
            render_views[selection](rootScene, t_obs, scalar_vals, time)
            return [], [gg]
        else
            gg = GridLayout(g[end+1, :])
            time, slider = widgets["Movement"](0.0, fig)
            gg[1, 1:2] = slider
            render_views[selection](rootScene, t_obs, time)
            return [], [gg]
        end
    end

    scene_switcher(rootScene, g, m.selection, choose_scene)
end
