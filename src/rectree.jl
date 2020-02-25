"""
    RecTree{I}

Holds a reconciled tree, typically a summary from a posterior sample.
"""
struct RecTree{I}
    root  ::RecNode{I}
    annot ::Dict{}  # flexible
end

Base.show(io::IO, rtree::RecTree) = write(io, "RecTree($(rtree.root))")

struct RecSummary
    trees ::Vector{NamedTuple}
    events::DataFrame
end

Base.show(io::IO, rsum::RecSummary) = 
    write(io, "RecSummary(# unique trees = $(length(rsum.trees)))")

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
    events = nothing
    for (h, count) in sort(collect(counts), by=x->x[2], rev=true)
        tree = trees[findfirst(x->x==h, hashes)]
        freq = count/N
        @unpack rtree, df = RecTree(tree, clades, N, ccd.leaves, wm)
        push!(summary, (freq=freq, tree=rtree))
        events = isnothing(events) ? df .* freq : events .+ (df .* freq)
    end
    RecSummary(summary, events)
end

cladecounts(trees) = countmap(vcat(map((t)->cladehash.(postwalk(t)), trees)...))

function RecTree(tree::RecNode, clades, N, leafnames, wm)
    d = Dict{typeof(cladehash(tree)),NamedTuple}()
    e = Dict{String,Vector{Int}}(l=>zeros(Int, length(wm)) for l in Labels)
    for n in postwalk(tree)
        cred = clades[cladehash(n)]/N
        label = getlabel(n, wm)
        name = isleaf(n) && label != "loss" ? leafnames[n.γ] : ""
        e[label][n.e]+= 1
        d[cladehash(n)] = (cred=cred, label=label, name=name)
    end
    (rtree=RecTree(tree, d), df=DataFrame(e))
end

# don't like this, very ad hoc
const Labels = ["loss", "wgd", "wgdloss", "duplication", "sploss", "speciation"]

function getlabel(n::RecNode, wm::WhaleModel)
    childrec = [c.e for c in children(n)]
    dup = all(n.e .== childrec) && length(childrec) == 2
    wgd = iswgd(wm[n.e])
    return if n.γ == 0
        Labels[1]
    elseif dup && wgd
        Labels[2]
    elseif !dup && wgd
        Labels[3]
    elseif dup
        Labels[4]
    elseif any(x->x==0, [c.γ for c in children(n)])
        Labels[5]
    else
        Labels[6]
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
