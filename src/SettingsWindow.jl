function build_settings_menu(selected_invariant, selected_alignment, alignment_options; fig_size=(400, 400))
    window = Figure(size=fig_size)

    invar_menu = Menu(window, options=["t1", "t2", "t3"], default=selected_invariant[], tellwidth=false)
    on(invar_menu.selection) do val
        selected_invariant[] = val
    end

    alignment_menu = Menu(window, options=alignment_options, default=selected_alignment[], tellwidth=false)
    on(alignment_menu.selection) do val
        selected_alignment[] = val
    end

    Label(window[1, :], "Settings", fontsize=30, tellwidth=false)
    window[2, :] = hgrid!(Label(window, "Selected invariant"), invar_menu)
    window[3, :] = hgrid!(Label(window, "Selected alignment"), alignment_menu)

    return window
end
