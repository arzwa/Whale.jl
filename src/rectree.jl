"""
    RecTree{I}

Holds a reconciled tree, typically a summary from a posterior sample.
"""
struct RecTree{I}
    root  ::RecNode{I}
    annot ::Dict{}  # flexible
end

Base.show(io::IO, rtree::RecTree) = show(io, hash(rtree.root))

"""
    sumtrees(trees, ccd, wm)

Summarize backtracked reconciled trees.
"""
sumtrees(trees::AbstractMatrix, ccd::AbstractVector, wm::WhaleModel) =
    map((i)->sumtrees(trees[:,i], ccd[i], wm), 1:length(ccd))

function sumtrees(trees::AbstractVector, ccd::CCD, wm::WhaleModel)
    N = length(trees)
    hashes = hash.(trees)
    counts = countmap(hashes)
    clades = cladecounts(trees)
    summary = NamedTuple[]
    for (h, count) in sort(collect(counts), by=x->x[2], rev=true)
        tree = trees[findfirst(x->x==h, hashes)]
        rtree = RecTree(tree, clades, N, ccd.leaves, wm)
        push!(summary, (freq=count/N, tree=rtree))
    end
    summary
end

cladecounts(trees) = countmap(vcat(map((t)->cladehash.(postwalk(t)), trees)...))

function RecTree(tree::RecNode, clades, N, leafnames, wm)
    d = Dict{typeof(id(tree)),NamedTuple}()
    for n in postwalk(tree)
        cred = clades[id(n)]/N
        label = getlabel(n, wm)
        name = isleaf(n) && label != "loss" ? leafnames[n.γ] : ""
        d[id(n)] = (cred=cred, label=label, name=name)
    end
    RecTree(tree, d)
end

# don't like this, very ad hoc
function getlabel(n::RecNode, wm::WhaleModel)
    childrec = [c.rec for c in children(n)]
    dup = all(n.rec .== childrec) && length(childrec) == 2
    wgd = iswgd(wm[n.rec])
    return if n.γ == 0
        "loss"
    elseif dup && wgd
        "wgd"
    elseif !dup && wgd
        "wgdloss"
    elseif dup
        "duplication"
    elseif length(childrec) == 1
        "sploss"
    else
        "speciation"
    end
end
