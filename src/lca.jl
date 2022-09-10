# LCA reconciliation
speciesname(x) = string(split(x, "_")[1])

"""
    lca_reconciliation(S, G)

Implements the Zmasek & Eddy algorithm, but also records the speciation+loss
nodes along the way. Produces a RecNode-based tree as Whale does.

!!! note:
    Assumes that the species tree S has preordered `id` field and that gene
    names are prefixed by species names with an `_` separator.
"""
function lca_reconciliation(S, G)
    sindex = Dict(name(n) => n for n in prewalk(S))
    leaf2species = Dict(id(n)=>sindex[speciesname(name(n))] for n in getleaves(G))
    so, ro = lcawalk(G, leaf2species)
    for (i,n) in enumerate(prewalk(ro))
        n.id = i
    end
    return ro
end

function lcawalk(gn, d)
    if isleaf(gn) 
        sn = d[id(gn)]
        rd = RecData(γ=id(gn), e=id(sn), t=NaN, name=name(gn))
        rn = Node(id(gn), rd)
        return sn, rn
    end
    s1, r1 = lcawalk(gn[1], d)
    s2, r2 = lcawalk(gn[2], d)
    p1 = [s1]
    p2 = [s2]
    while s1 != s2
        if id(s1) > id(s2)
            s1 = parent(s1)
            push!(p1, s1)
        else
            s2 = parent(s2)
            push!(p2, s2)
        end
    end
    ln = (id(s1) == r1.data.e || id(s1) == r2.data.e) ? 
        "duplication" : "speciation"
    p1 = ln == "duplication" ? p1[2:end] : p1[2:end-1]
    p2 = ln == "duplication" ? p2[2:end] : p2[2:end-1]
    rn = Node(id(gn), RecData(γ=id(gn), e=id(s1), t=NaN, label=ln))
    # add the sploss nodes
    for p in p1
        ln = Node(id(gn), RecData(γ=id(gn), e=id(p), t=NaN, label="sploss"))
        loss = Node(id(gn), RecData(γ=id(gn), e=id(p), t=NaN, label="loss"), ln)
        push!(ln, r1)
        r1 = ln
    end
    for p in p2
        ln = Node(id(gn), RecData(γ=id(gn), e=id(p), t=NaN, label="sploss"))
        loss = Node(id(gn), RecData(γ=id(gn), e=id(p), t=NaN, label="loss"), ln)
        push!(ln, r2)
        r2 = ln
    end
    push!(rn, r1, r2)
    return s1, rn
end
