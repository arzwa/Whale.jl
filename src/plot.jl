function TreeLayout(wm::WhaleModel{I,T}) where {I,T}
    coords = Dict{I,Tuple}()
    paths  = Tuple[]
    yleaf  = -1.
    function walk(n)
        if Whale.isleaf(n)
            yleaf += 1
            x = rootdist(n, wm)
            coords[n.id] = (x, yleaf)
            return yleaf
        else
            u = []
            for c in n.children
                push!(u, walk(wm[c]))
                push!(paths, (n.id, c))
            end
            y = sum(u)/length(u)
            x = rootdist(n, wm)
            coords[n.id] = (x, y)
            return y
        end
    end
    walk(wm[1])
    BelugaPlots.TreeLayout(coords=coords, paths=paths, labels=wm.leaves)
end

function rootdist(n, wm)
    dist = 0.
    while n.id != wm[1].id
        dist += n.event.t
        n = wm[n.parent]
    end
    dist
end
