using Makie: Observable, lift

function pair(v)
    pairs = Vector{Pair{Any,Any}}()
    for i in 1:length(v)-1
        e1 = v[i]
        e2 = v[i+1]
        push!(pairs, Pair(e1, e2))
    end
    return pairs
end

splitobs(o::Observable{Tuple{}}) = ()
splitobs(o::Observable{<:Tuple}) = (lift(first, o), splitobs(lift(Base.tail, o))...)
