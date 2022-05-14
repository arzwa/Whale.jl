"""
    RecSummary
"""
struct RecSummary
    trees ::Vector{NamedTuple}
    events::DataFrame
    fname ::String   # original file name of the associated CCD
end

Base.getindex(rs::RecSummary, i) = rs.trees[i]

Base.show(io::IO, rsum::RecSummary) =
    write(io, "RecSummary(# unique trees = $(length(rsum.trees)))")

writetrees(s::String, rsum::Vector{NamedTuple}) = open(s, "w") do io
    writetrees(io, rsum)
end

function writetrees(io::IO, rsum::Vector{NamedTuple}, sep="\n")
    for (f,t) in rsum
        write(io, "# pp $f$sep$(nwstr(t))\n")
    end
end

function summarize(xs::AbstractVector{RecSummary}) # no joke
    dfs = []
    for (i, rs) in enumerate(xs)
        rs.events[!,:family] .= i
        push!(dfs, rs.events)
    end
    events = vcat(dfs...)
    gdf = groupby(events, [:node, :clade])
    sm = combine(gdf, 
                 eventexpectation(mean)..., 
                 eventexpectation(std)..., 
                 eventquantile(0.025, "q1")...,
                 eventquantile(0.975, "q2")...)
    (full=events, sum=sm)
end

function eventexpectation(fun, name=fun)
    funs = [AsTable([lab, :freq]) => 
            (x->fun(getindex(x, lab), weights(x.freq))) =>
            "$(lab)_$(name)" for lab in Symbol.(Labels)]
end

function eventquantile(q, name)
    funs = [AsTable([lab, :freq]) => 
            (x->quantile(getindex(x, lab), weights(x.freq), q)) =>
            "$(lab)_$name" for lab in Symbol.(Labels)]
end

function getpairs(rsum::AbstractVector{RecSummary}, model)
    labels = String[]
    for n in model.order
        e = id(n)
        push!(labels, "$(e)_duplication")
        iswgd(n) ?
            push!(labels, "$(e)_wgd", "$(e)_wgdloss") :
            push!(labels, "$(e)_speciation")
    end
    mapreduce(x->getpairs(x, labels), vcat, rsum)
end

# TODO: quite ugly, clean up
function getpairs(rsum::RecSummary, labels)
    # goal: for each pair of genes in a family the approximate posterior
    # reconciliation distribution.
    # first get all pair IDs
    leafset = getleaves(rsum.trees[1].tree)
    pairs = String[]
    for i=1:length(leafset), j=1:i-1
        l1 = leafset[i]; l2 = leafset[j]
        (l1.data.label == "loss" || l2.data.label == "loss") && continue
        pairid = join(sort([name(l1), name(l2)]), "__")
        push!(pairs, pairid)
    end
    d = Dict(l=>zeros(length(pairs)) for l in labels)
    idx = Dict(p=>i for (i,p) in enumerate(pairs))
    for (f,t) in rsum.trees
        _getpairs!(d, idx, f, t)
    end
    roundvals!(d)
    DataFrame(d..., "family"=>rsum.fname, "pair"=>pairs)
end

function roundvals!(d, digits=3)
    for k in keys(d)
        d[k] = round.(d[k], digits=digits)
    end
end

function _getpairs!(d, idx, f, tree::Node)
    for n in postwalk(tree)
        (isleaf(n) || degree(n) == 1) && continue
        for l1 in getleaves(n[1]), l2 in getleaves(n[2])
            (l1.data.label == "loss" || l2.data.label == "loss") && continue
            pairid = join(sort([name(l1), name(l2)]), "__")
            label = "$(n.data.e)_$(n.data.label)"
            d[label][idx[pairid]] += f
        end
    end
end

"""
    sumtrees(trees, ccd, wm)

Summarize backtracked reconciled trees.
"""
sumtrees(trees::AbstractMatrix, ccd::AbstractVector, wm::WhaleModel) =
    map((i)->sumtrees(trees[:,i], ccd[i], wm), 1:length(ccd))

function sumtrees(trees::AbstractVector, ccd::CCD, wm::WhaleModel)
    N = length(trees)
    hashes = nodehash.(trees)
    counts = countmap(hashes)
    clades = cladecounts(trees)
    summary = NamedTuple[]
    events = nothing
    cladelab = [getcladelabel(wm[i]) for i=1:length(wm)]
    for (h, count) in sort(collect(counts), by=x->x[2], rev=true)
        tree = trees[findfirst(x->x==h, hashes)]
        freq = count/N
        rtree, df = label_and_summarize!(tree, clades, N, wm)
        push!(summary, (freq=freq, tree=rtree))
        df[!,:tree] .= h
        df[!,:freq] .= freq
        df[!,:clade] = cladelab
        df[!,:node] = [id(wm[i]) for i=1:length(wm)]
        events = isnothing(events) ? df : vcat(events, df)
    end
    RecSummary(summary, events, ccd.fname)
end

sumevents(r::AbstractVector{RecSummary}) =
    reduce((x,y)->x .+ y.events, r[2:end], init=r[1].events)

cladecounts(trees) = countmap(vcat(map((t)->cladehash.(postwalk(t)), trees)...))

function label_and_summarize!(tree::RecNode, clades, N, wm)
    I = typeof(id(tree))
    e = Dict{String,Vector{Int}}(l=>zeros(Int, length(wm)) for l in Labels)
    for (i,n) in enumerate(postwalk(tree))
        e[n.data.label][gete(n)] += 1
        n.data.cred = clades[cladehash(n)]/N
        startswith(n.data.label, "wgd") ?
            n.data.name = name(wm[gete(n)]) : nothing
    end
    (rtree=tree, df=DataFrame(e))
end

# get a human readable node label
function getcladelabel(n)
    name(n) != "" && return name(n)
    !isleaf(n) && return join([name(getleaves(c)[1]) for c in children(n)], ",")
end

"""
    gettables(trees::Vector{RecSummary}, nodes=[])

Get for every node in the species tree a data structure with all
gene tree nodes reconciled to that nodes for each relevant event type.
Output is suggested to be saved as JSON string.
"""
function gettables(trees, nodes=[]; leaves=true)
    table = Dict()
    for (i,s) in enumerate(trees)
        seen = Set()
        for (_, t) in s.trees
            for n in prewalk(t)
                e = n.data.e
                l = n.data.label
                (isleaf(n) && !leaves) && continue
                (isleaf(n) && (l == "loss" || l == "sploss")) && continue
                (!isempty(nodes) && e ∉ nodes) && continue
                x = isleaf(n) ? (e, name(n)) : (Whale.cladehash(n), l)
                x ∈ seen && continue
                push!(seen, x)
                !haskey(table, e) ? table[e] = Dict() : nothing
                !haskey(table[e], l) ? table[e][l] = [] : nothing
                push!(table[e][l], (i = i, 
                    f = n.data.cred, fname = s.fname, 
                    left  = isleaf(n) ? name(n) : name.(getleaves(n[1])),
                    right = isleaf(n) ? "" : name.(getleaves(n[2]))))
            end
        end
    end
    table
end

"""
    subgenome_assignments(tables, nodes)

```
tables = Whale.gettables(trees)
df = Whale.subgenome_assignments(tables, [1,3])
```
"""
function subgenome_assignments(tables, nodes)
    df = DataFrame(:gene=>String[])
    for n in nodes
        tab = tables[n]["speciation"]
        genes = String[]
        ps = Float64[]
        for entry in tab
            gene = entry.left
            push!(genes, gene)
            push!(ps, entry.f)
        end
        df = outerjoin(df, DataFrame(:gene=>genes, Symbol("subgenome_$n")=>ps), on=:gene)
    end
    for col in names(df)[2:end]
        df[:,col] = replace!(df[:,col], missing=>0.)
    end
    return df
end

# Per Hengchi's request
function recstr(n::RecNode)
    function walk(n)
        isleaf(n) && return "$(name(n))"
        s = join([walk(c) for c in children(n)], ",")
        label = n.data.label
        cred  = n.data.cred
        internal = "$(label)_$(cred)" 
        return "($s)$internal"
    end
    s = walk(n)
    s*";"
end

function count_events(trees::Vector)
    counts = DefaultDict{String,Int}(0)
    for tree in trees
        count_events!(counts, tree)
    end
    return counts
end

function count_events(tree::RecNode)
    counts = DefaultDict{String,Int}(0)
    count_events!(counts, tree)
    return counts
end

function count_events!(dict, tree)
    for n in prewalk(tree)
        event = "$(n.data.e)_$(n.data.label)" 
        dict[event] += 1
    end
end
