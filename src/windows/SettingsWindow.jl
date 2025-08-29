function build_settings_menu(selected_alignment, alignment_options, num_atoms; fig_size=(640, 480))
    window = Figure(size=fig_size)
    menu_bar = top_bar(window, "Settings", 1)

    alignment_menu = Menu(window, options=alignment_options, default=selected_alignment[], tellwidth=false)
    on(alignment_menu.selection) do val
        selected_alignment[] = val
    end

    window[2, 1] = hgrid!(Label(window, "Selected alignment"), alignment_menu)

    return window
end
