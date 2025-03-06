using Makie

function build_cluster_window(ts, rel_t_idx, vals, aap, volData, sampleRanges; fig_size=(400, 400))
    window = Figure(size=fig_size)

    hm_ax, hm = heatmap(window[1, 1], vals)
    hidedecorations!(hm_ax)
    deregister_interaction!(hm_ax, :rectanglezoom)

    hl = Observable(1)
    hr = Observable(2)

    on(events(hm_ax).mouseposition) do mp
        plot, _ = pick(hm_ax)
        if plot == hm
            xy = mouseposition(hm_ax)
            i, j = Int.(round.(xy))
            hl[] = i
            hr[] = j
            notify(hl)
            notify(hr)
        end
        return Consume(false)
    end




    return window
end
