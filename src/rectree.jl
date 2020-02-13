"""
    RecTree{I}

Holds a reconciled tree, typically a summary from a posterior sample.
"""
struct RecTree{I}
    root  ::RecNode{I}
    annot ::Dict{}  # flexible
end

Base.show(io::IO, rtree::RecTree) = write(io, "RecTree($(rtree.root))")

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
    d = Dict{typeof(cladehash(tree)),NamedTuple}()
    for n in postwalk(tree)
        cred = clades[cladehash(n)]/N
        label = getlabel(n, wm)
        name = isleaf(n) && label != "loss" ? leafnames[n.γ] : ""
        d[cladehash(n)] = (cred=cred, label=label, name=name)
    end
    RecTree(tree, d)
end

# don't like this, very ad hoc
function getlabel(n::RecNode, wm::WhaleModel)
    childrec = [c.e for c in children(n)]
    dup = all(n.e .== childrec) && length(childrec) == 2
    wgd = iswgd(wm[n.e])
    return if n.γ == 0
        "loss"
    elseif dup && wgd
        "wgd"
    elseif !dup && wgd
        "wgdloss"
    elseif dup
        "duplication"
    elseif any(x->x==0, [c.γ for c in children(n)])
        "sploss"
    else
        "speciation"
    end
end

function pruneloss!(tree::RecTree)
    @unpack root, annot = tree
    nodes = postwalk(root)
    ids = cladehash.(nodes)
    for (h, n) in zip(ids, nodes)
        if annot[h].label == "loss"  # deletion happens at sploss node
            delete!(tree.annot, h)
        elseif annot[h].label == "sploss"
            child = children(n)[1]
            newchild = RecNode(child.γ, child.e, child.t,
                child.children, n.parent)
            delete!(n.parent.children, n)
            delete!(tree.annot, h)
            push!(n.parent, newchild)
        else
            ann = annot[h]
            delete!(tree.annot, h)
            tree.annot[cladehash(n)] = ann
        end
    end
end
