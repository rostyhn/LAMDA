function build_settings_menu(selected_invariant; fig_size=(400, 400))
    window = Figure(size=fig_size)

    invar_menu = Menu(window, options=["t1", "t2", "t3"], tellwidth=false)
    on(invar_menu.selection) do val
        selected_invariant[] = val
    end

    Label(window[1, :], "Settings", fontsize=30, tellwidth=false)
    window[2, :] = hgrid!(Label(window, "Selected invariant"), invar_menu)

    return window

    screen = GLMakie.Screen(title="TransVis - Settings")
    display(screen, window)
end



