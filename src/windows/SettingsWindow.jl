function build_settings_menu(selected_invariant, selected_alignment, alignment_options; fig_size=(640, 480))
    window = Figure(size=fig_size)
    menu_bar = top_bar(window, "Settings", 1)

    invar_menu = Menu(window, options=["t1", "t2", "t3"], default=selected_invariant[], tellwidth=false)
    on(invar_menu.selection) do val
        selected_invariant[] = val
    end

    alignment_menu = Menu(window, options=alignment_options, default=selected_alignment[], tellwidth=false)
    on(alignment_menu.selection) do val
        selected_alignment[] = val
    end

    window[2, 1] = hgrid!(Label(window, "Selected invariant"), invar_menu)
    window[3, 1] = hgrid!(Label(window, "Selected alignment"), alignment_menu)

    return window
end
