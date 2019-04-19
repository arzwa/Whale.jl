#= Posterior reconciled tree summary
1. Consensus trees I believe this would be the most interesting strategy: -
Sample reconciled trees from MAP parameter values and compute consensus -
Compute for every event (dup/loss/wgd/speciation) in the MAP tree the posterior
probability based on a sample of (consensus) reconciled trees from the posterior

Alternatively, the consensus tree could be computed from the full posterior?
The best way to compute a consensus tree is probably by means of MRP

TODO: The fitch parsimony is working but haven't implemented good tree
rearrangement schemes yet... It's orders of magnitude slower than pars from
phylip...

NOTE: the MRP seems to work when not including loss nodes (tested with
`pars` from phylip), so we have everything to get a consensus topology, but
note really a consensus reconciliation...

2. ALE-like summary of number of dups and losses/branch
3. WGD summary
4. Consensus reconciliation
    - get consensus topology
    - get majority vote reconciliation for each node in consensus topology
=#
# ALE-like summary =============================================================
"""
    alelike_summary(rectrees::Dict,)
"""
function alelike_summary(rt::Dict{Any,Array{RecTree}}, S::SpeciesTree)
    dfs = []
    for (k, v) in rt
        push!(dfs, alelike_summary(v, S, fname=k))
    end
    return vcat(dfs...)
end

# this is implemented for rectrees of one family
function alelike_summary(rt::Array{RecTree}, S::SpeciesTree; fname="")
    df = DataFrame(family=String[], branch=Int64[], speciation=Float64[], leaf=Float64[],
        wgd=Float64[], duplication=Float64[], loss=Float64[], retained=Float64[],
        species=String[], total=Int64[])
    for n in keys(S.tree.nodes)
        push!(df, [fname; alesum(rt, n); getspecies(S, n); length(rt)])
    end
    return df
end

function alesum(rt::Array{RecTree}, node::Int64;
        order=["speciation", "leaf", "wgd", "duplication", "loss", "retained"])
    data = Dict(l=>0 for l in order)
    for t in rt
        nodes = [k for (k,v) in t.σ if v == node]
        for n in nodes
            data[t.labels[n]] += 1
            if t.labels[n] == "wgd" && length(childnodes(t.tree, n)) == 2
                data["retained"] += 1
            end
        end
    end
    out = Number[node]
    for l in order; push!(out, data[l]/length(rt)); end
    return out
end

function getspecies(S::SpeciesTree, n::Int64; sep=",")
    return join([haskey(S.ambiguous, x) ? S.ambiguous[x] :
        S.leaves[x] for x in S.clades[n]], sep)
end

# WGD specific summary =========================================================
# this is implemented for rectrees over all families
"""
    summarize_wgds(nrtrees::Dict, S::SpeciesTree)
Summarize retained WGD events for every gene family.
"""
function summarize_wgds(nrtrees::Dict{Any,Array{RecTree}}, S::SpeciesTree)
    data = DataFrame(:gf => Any[], :wgd_id => Int64[], :wgd_node => Int64[],
        :rectree_node => Int64[], :gleft => String[], :gright => String[],
        :sleft => String[], :sright => String[], :count => Int64[])
    for (gf, rtrees) in nrtrees
        for (wgd, x) in summarize_wgds(rtrees, S)
            push!(data, [gf ; x])
        end
    end
    return data
end

# function for summarizing WGDs in samples of backtracked trees, uses hashes to count
function summarize_wgds(rtrees::Array{RecTree}, S::SpeciesTree)
    d = Dict{UInt64,Array}()
    for rt in rtrees
        for (n, l) in rt.labels
            children = childnodes(rt.tree, n)
            if l == "wgd" && length(children) == 2
                left   = sort(Whale.subtree_leaves(rt, children[1]))
                right  = sort(Whale.subtree_leaves(rt, children[2]))
                h = hash(sort([left, right]))
                if !haskey(d, h)
                    sleft  = sort(Whale.subtree_leaves(S,  rt.σ[children[1]]))
                    sright = sort(Whale.subtree_leaves(S,  rt.σ[children[2]]))
                    wgd_node = rt.σ[n]
                    wgd_id = S.wgd_index[wgd_node]
                    d[h] = [wgd_id, wgd_node, n, join(left, ";"), join(right, ";"),
                            join(sleft, ";"), join(sright, ";"), 0]
                end
                d[h][end] += 1
            end
        end
    end
    return d
end

# Subgenome assignment =========================================================
function sumambiguous(rt::Array{RecTree}, S::SpeciesTree, ccd::CCD)
    length(S.ambiguous) == 0 ? (return) : nothing
    ambgenes = [ccd.leaves[k] for (k, v) in ccd.m3 if haskey(S.ambiguous, v)]
    data = Dict{String,Dict}(g=>Dict() for g in ambgenes)
    N = length(rt)
    for t in rt
        leaf2node = Dict(v=>k for (k,v) in t.leaves)
        ambnodes = [leaf2node[g] for g in ambgenes]
        for (g, n) in zip(ambgenes, ambnodes)
            sp = S.leaves[t.σ[n]]
            haskey(data[g], sp) ? data[g][sp] += 1/N : data[g][sp] = 1/N
        end
    end
    return data
end

# Reconciliation of the consensus tree =========================================
# Idea: if we aer able to read and write the `reconciliation_distribution` to
# and from file, then we don't need the consensus trees in the library, yet it
# isn't a nice idea, since we do want the consensus trees in here.
"""
    consensus_tree_reconciliation(contree::Arboreal, rectrees::Array{RecTree})
Given a consensus tree topology Tᶜ, determine for every node (not clade?) in Tᶜ
the reconciliation 'distribution' (e.g. `Tᶜ node 2: 80%S@S₃, 20%D@S₄`) and do a
majority vote.
"""
function consensus_tree_reconciliation(contree::Arboreal,
        rectrees::Array{RecTree})
    recdist = reconciliation_distribution(rectrees)
    T = ConRecTree(contree.tree, contree.leaves)
    function walk(n)
        hsh = hashnode(n, T)
        if haskey(recdist, hsh)  # HACK, should test by outdegree?
            cnt, rec = findmax(recdist[hsh])
            T.labels[n] = rec[1]
            T.σ[n] = rec[2]
            T.rsupport[n] = cnt / length(rectrees)
            T.tsupport[n] = contree.support[n]
            T.recdist[n] = collect(recdist[hsh])
        end
        isleaf(T.tree, n) ? (return) : [walk(c) for c in childnodes(T.tree, n)]
    end
    walk(findroots(T.tree)[1])
    return T
end

"""
    reconciliation_distribution(rectrees::Array{RecTree})
Get the reconciliation distribution for each node of a gene tree from a bunch
of reconciled trees.
"""
function reconciliation_distribution(rectrees::Array{RecTree})
    dist = Dict{UInt64,Dict{Tuple,Int64}}()
    for rt in rectrees
        rt_ = Whale.prune_loss_nodes(rt)
        addrectodist!(dist, rt_)
    end
    return dist
end

function addrectodist!(dist, tree::RecTree)
    function walk(n)
        hsh = hashnode(n, tree)
        haskey(dist, hsh) ? nothing : dist[hsh] = Dict{Tuple,Int64}()
        tup = (tree.labels[n], tree.σ[n])
        haskey(dist[hsh], tup) ? dist[hsh][tup] += 1 : dist[hsh][tup] = 1
        if isleaf(tree.tree, n)
            return
        else
            for c in childnodes(tree.tree, n)
                walk(c)
            end
        end
    end
    walk(findroots(tree.tree)[1])
end

leafset(node::Int64, tree::Arboreal) = [tree.leaves[n] for n in
    [node ; descendantnodes(tree.tree, node)] if haskey(tree.leaves, n)]

hashnode(node::Int64, tree::Arboreal) = hash(Set(leafset(node, tree)))

function Base.write(io::IO, contree::ConRecTree, S::SpeciesTree; format="table")
    format == "table" ?
        contreetable(io, contree, S) : (@error "Not implemented")
end

function contreetable(io::IO, contree::ConRecTree, S::SpeciesTree)
    write(io,"gnode,hash,leaves,mvrec,mvlabel,mvperc,mvspeciesset\n")
    for (n, node) in contree.tree.nodes
        haskey(contree.labels, n) ? nothing : continue
        h = hashnode(n, contree)
        s = contree.σ[n]
        l = contree.labels[n]
        p = contree.rsupport[n]
        ls = join(sort(leafset(n, contree)), ";")
        ss = getspecies(S, s, sep=";")
        write(io, "$n,$h,$ls,$s,$l,$p,$ss\n")
    end
end

function write_speciestable(io::IO, S::SpeciesTree)
    write(io, "node,label,clades\n")
    for (n, node) in S.tree.nodes
        clades = join(sort(leafset(n, S)), ";")
        l = haskey(S.wgd_index, n) ? "wgd" : "speciation"
        write(io, "$n,$l,$clades\n")
    end
end

function write_consensus_reconciliations(rtrees, S, dirname, thresh=0.0)
    open(joinpath(dirname, "species.csv"), "w") do f
        write_speciestable(f, S)
    end
    p = Progress(length(rtrees), 0.1, "| Computing consensus reconciliations")
    for (gf, rts) in rtrees
        @debug gf
        gfname = basename(gf)
        rt = [Whale.prune_loss_nodes(t) for t in rts]
        ct = majority_consensus(rt, thresh=thresh)
        open(joinpath(dirname, "$gfname.ct"), "w") do f
            write(f, ct.tree, ct.leaves)
        end
        crt = consensus_tree_reconciliation(ct, rt)
        open(joinpath(dirname, "$gfname.crt"), "w") do f
            write(f, crt)
        end
        open(joinpath(dirname, "$gfname.csv"), "w") do f
            write(f, crt, S)
        end
        next!(p)
    end
end
