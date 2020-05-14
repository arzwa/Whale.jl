"""
    RecSummary
"""
struct RecSummary
    trees ::Vector{NamedTuple}
    events::DataFrame
end

Base.show(io::IO, rsum::RecSummary) =
    write(io, "RecSummary(# unique trees = $(length(rsum.trees)))")

function summarize(xs::Vector{RecSummary}) # no joke
    dfs = []
    for (i, rs) in enumerate(xs)
        rs.events[!,:family] .= i
        push!(dfs, rs.events)
    end
    vcat(dfs...)
end

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
        rtree, df = label_and_summarize!(tree, clades, N, ccd.leaves, wm)
        push!(summary, (freq=freq, tree=rtree))
        events = isnothing(events) ? df .* freq : events .+ (df .* freq)
    end
    events[!,:node] = [getnodelabel(wm[i]) for i=1:length(wm)]
    RecSummary(summary, events)
end

sumevents(r::AbstractVector{RecSummary}) =
    reduce((x,y)->x .+ y.events, r[2:end], init=r[1].events)

cladecounts(trees) = countmap(vcat(map((t)->cladehash.(postwalk(t)), trees)...))

function label_and_summarize!(tree::RecNode, clades, N, leafnames, wm)
    I = typeof(id(tree))
    d = Dict{typeof(cladehash(tree)),NamedTuple}()
    e = Dict{String,Vector{Int}}(l=>zeros(Int, length(wm)) for l in Labels)
    for (i,n) in enumerate(postwalk(tree))
        n.id = I(i)
        label = getlabel(n, wm)
        e[label][gete(n)]+= 1
        n.data.label = label
        n.data.cred = clades[cladehash(n)]/N
        startswith(label, "wgd") ?
            n.data.name = name(wm[gete(n)]) : nothing
    end
    (rtree=tree, df=DataFrame(e))
end

# don't like this, very ad hoc,we could add a 'kind' field in `SliceState` instead
# that would annotate RecNodes directly?
const Labels = ["loss", "wgd", "wgdloss", "duplication", "sploss", "speciation"]

function getlabel(n::RecNode, wm::WhaleModel)
    childrec = [gete(c) for c in children(n)]
    dup = all(gete(n) .== childrec) && length(childrec) == 2
    wgd = iswgd(wm[gete(n)])
    loss = any(x->x==0, [getγ(c) for c in children(n)])
    return if getγ(n) == 0
        Labels[1]
    elseif !dup && !loss && wgd
        Labels[2]
    elseif loss && wgd
        Labels[3]
    elseif dup
        Labels[4]
    elseif loss
        Labels[5]
    else
        Labels[6]
    end
end

# get a human readable node label
function getnodelabel(n)
    name(n) != "" && return name(n)
    !isleaf(n) && return join([name(getleaves(c)[1]) for c in children(n)], ",")
end

# WGD-reated summarization
iswgddup(n::RecNode) = n.data.label=="wgd"
getwgds(tree::RecNode) = filter(iswgddup, postwalk(tree))
getwgds(tree::RecNode, wgds) = filter(
    x->iswgddup(x) && name(x) ∈ wgds, postwalk(tree))

"""
    getwgdtables(recs::Vector{RecSummary}, ccd, model::WhaleModel)
    getwgdtables(recs::Vector{RecSummary}, ccd, wgds::Vector{String})
"""
getwgdtables(recs::Vector{RecSummary}, ccd, wm::WhaleModel) =
    getwgdtables(recs, ccd, name.(getwgds(wm)))
function getwgdtables(recs::Vector{RecSummary}, ccd, wgds::Vector)
    data = [getwgddups(r, wgds) for r in recs]
    [wgd=>wgdtable(data, ccd, wgd) for wgd in wgds]
end

function getwgddups(recs::RecSummary, wgds)
    d = Dict(wgd=>Dict() for wgd in wgds)
    for (f, tree) in recs.trees, n in getwgds(tree, wgds)
        triple = (getγ(n), getγ(n[1]), getγ(n[2]))
        haskey(d[name(n)], triple) ?
            d[name(n)][triple] += f : d[name(n)][triple] = f
    end
    d
end

function wgdtable(data, ccd, wgd)
    X = []
    for (i, (d, c)) in enumerate(zip(data, ccd)), (t, f) in d[wgd]
        push!(X, (family=i, frequency=round(f, digits=4),
            clade=t[1], left=t[2], right=t[3],
            lleaves=join(getleaves(c, t[2]), ";"),
            rleaves=join(getleaves(c, t[3]), ";")))
    end
    DataFrame(X)
end

# function pruneloss!(tree::RecTree)
#     @unpack root, annot = tree
#     nodes = postwalk(root)
#     ids = cladehash.(nodes)
#     for (h, n) in zip(ids, nodes)
#         if annot[h].label == "loss"  # deletion happens at sploss node
#             delete!(tree.annot, h)
#         elseif annot[h].label == "sploss"
#             child = children(n)[1]
#             newchild = RecNode(child.γ, child.e, child.t,
#                 child.children, n.parent)
#             delete!(n.parent.children, n)
#             delete!(tree.annot, h)
#             push!(n.parent, newchild)
#         else
#             ann = annot[h]
#             delete!(tree.annot, h)
#             tree.annot[cladehash(n)] = ann
#         end
#     end
# end
