function NoteWindow(title::Observable{String}, input::Observable{String}, on_submit; fig_size=(640, 480))
    window = Figure(size=fig_size)
    menu_bar = top_bar(window, "Notes", 2)

    gg = GridLayout()
    window[2, 1:end] = gg

    gg[1, 1] = Label(window, "Title", halign=:left)

    txtbox_title = Textbox(window, halign=:left, tellwidth=false)
    set_text(txtbox_title, title[])
    gg[1, 2] = txtbox_title

    cg = GridLayout()
    window[3, 1:end] = cg

    Label(cg[1, 1:2], "Notes", font=:bold)
    txtbox_notes = Textbox(cg[2, 1:2])
    set_text(txtbox_notes, input[])

    txtbox_notes.width[] = Relative(1.0)
    txtbox_notes.height[] = Relative(1.0)
    rowsize!(cg, 2, Relative(0.95))

    on(txtbox_title.stored_string) do s
        on_submit("titles", s)
    end

    on(txtbox_notes.stored_string) do s
        on_submit("notes", s)
    end

    on(window.scene.viewport) do vp
        resize_to_layout!(window)
    end

    #TODO: add newlines, need to dig around in makie source to do this
    # https://github.com/MakieOrg/Makie.jl/blob/master/src/makielayout/blocks/textbox.jl
    # might make sense to just copy this code and add newline functionality
    #=on(events(window.scene).keyboardbutton; priority=100) do event
        if txtbox_notes.focused[]
            if event.action != Keyboard.release
                key = event.key
                if key == Keyboard.enter || key == Keyboard.kp_enter
                    s = txtbox_notes.stored_string[]

                    txtbox_notes.stored_string[] = s * "\n"
                    txtbox_notes.cursorindex[] = min(length(txtbox_notes.displayed_string[]), txtbox_notes.cursorindex[] + 2)
                    #txtbox_notes.cursor_forward()
                    return Consume(true)
                end
            end
        end
        return Consume(false)
    end=#

    return window
end
