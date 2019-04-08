#= Posterior reconciled tree summary
1. Consensus trees
    I believe this would be the most interesting strategy:
    - Sample reconciled trees from MAP parameter values and compute consensus
    - Compute for every event (dup/loss/wgd/speciation) in the MAP tree the posterior
      probability based on a sample of (consensus) reconciled trees from the posterior

    Alternatively, the consensus tree could be computed from the full poterior?
    The best way to compute a consensus tree is probably by means of MRP

    TODO: The fitch parsimony is working but haven't implemented good tree rearrangement
    schemes yet... It's orders of magnitude slower than pars from phylip...

    NOTE: the MRP seems to work when not including loss nodes (tested with `pars`
    from phylip), so we have everything to get a consensus topology, but note really
    a consensus reconciliation...

2. ALE-like summary of number of dups and losses/branch
3. WGD summary
=#
# MRP consensus ====================================================================================
"""
    MatRep()
Matrix representation of multiple trees.
"""
mutable struct MatRep
    leafmap::Dict{String,Int64}
    matrix::Array{Int64,2}

    MatRep() = new(Dict{String,Int64}(), zeros(0,0))
end

"""
    mrpencode(trees::Array{RecTree}; loss=false)
Encode a bunch of trees as a binary matrix.
"""
function mrpencode(trees::Array{RecTree}; loss=false)
    mr = MatRep()
    for t in trees; mrpencode!(mr, t, loss=loss) ; end
    return mr
end

function mrpencode!(mr::MatRep, tree::RecTree; loss=false)
    leafmap = getleafmap(tree; loss=loss)
    leaves = keys(leafmap)
    intern = setdiff(collect(keys(tree.tree.nodes)), collect(leaves))
    updateleafmap!(mr, leafmap)
    matrix = zeros(Int64, size(mr.matrix)[1], length(intern))
    for i in leaves
        for (j, n) in enumerate(intern)
            i in descendantnodes(tree.tree, n) ?
                matrix[mr.leafmap[leafmap[i]], j] = 1 : nothing
        end
    end
    mr.matrix = [mr.matrix matrix]
end

function updateleafmap!(mr::MatRep, leafmap::Dict{Int64,String})
    for (k,v) in leafmap
        if !haskey(mr.leafmap, v)
            mr.leafmap[v] = length(mr.leafmap) + 1
            mr.matrix = [mr.matrix ; zeros(1, size(mr.matrix)[2])]
        end
    end
end

function getleafmap(tree::RecTree; loss=false)
    leafmap = Dict{Int64,String}()
    for n in findleaves(tree.tree)
        if tree.labels[n] == "loss" && loss
            leafmap[n] = "loss-$(tree.σ[n])"
        elseif tree.labels[n] != "loss"
            leafmap[n] = tree.leaves[n]
        end
    end
    return leafmap
end

function treehash(tree::RecTree)
    internmap = Dict{Int64,UInt64}()
    leafmap   = Dict{Int64,UInt64}()
    function walk(n)
        if isleaf(tree.tree, n)
            id = tree.labels[n] == "loss" ? hash(("loss",tree.σ[n])) : hash(tree.leaves[n])
            leafmap[n] = id
            return id
        else
            children = UInt64[]
            for c in childnodes(tree.tree, n); push!(children, walk(c)) ; end
            id = hash(Set(children))
            internmap[n] = id
            return id
        end
    end
    walk(1)
    return internmap, leafmap
end

"""
    write(io::IO, mr::MatRep)
Write a matrix representation file in phylip format (e.g. for use with pars)
"""
function Base.write(io::IO, mr::MatRep)
    write(io, join(size(mr.matrix), " ") * "\n")
    for (k, v) in mr.leafmap
        write(io, (@sprintf "%-10s" split(k, "_")[1] * string(v)) * join(mr.matrix[v, :]) * "\n")
    end
end

# Fitch parsimony ==================================================================================
function parsfitch(mr::MatRep; ctol=1e3)
    tree, lmap = parsfitch(mr.matrix, ctol=ctol)
    invmap = Dict(v=>k for (k,v) in mr.leafmap)
    leaves = Dict(n=>invmap[lmap[n]] for n in findleaves(tree))
    return tree, leaves
end

function parsfitch(aln::Matrix; ctol=10)
    tree = inittree(size(aln)[1])
    lmap = initleafmap(tree)
    best = parsfitch(tree, aln, lmap)
    n = 0
    while n < ctol
        for node in Whale.preorder_t(tree)[2:end]
            tree_ = randspr(tree, node)
            score = parsfitch(tree_, aln, lmap)
            if score < best
                @printf "%4d - " score
                n = 0
                best = score
                tree = tree_
                break
            end
        end
        n += 1
    end
    return tree, lmap
end

"""
    parsfitch(tree::Tree, aln::Matrix, leafmap::Dict{Int64,Int64})
Fitch parsimony for a matrix site.
"""
function parsfitch(tree::Tree, aln::Matrix, leafmap::Dict{Int64,Int64})
    return sum([parsfitch(tree, aln[:, i], leafmap) for i=1:size(aln)[2]])
end

"""
    parsfitch(tree::Tree, site::Vector, leafmap::Dict{Int64,Int64})
Fitch parsimony for a single site.
"""
function parsfitch(tree::Tree, site::Vector, leafmap::Dict)
    treelen = 0
    function walk(n)
        if isleaf(tree, n)
            return Set(site[leafmap[n]])
        else
            child_states = []
            for c in childnodes(tree, n)
                push!(child_states, walk(c))
            end
            common = intersect(child_states...)
            if length(common) > 0
                return common
            else
                treelen += 1
                return union(child_states...)
            end
        end
    end
    walk(findroots(tree)[1])
    return treelen
end

# slower
function parsfitch2(tree::Tree, aln::Matrix, leafmap::Dict{Int64,Int64})
    treelen = 0
    L = size(aln)[2]
    function fitchfun(states)
        isect = intersect(states...)
        if length(isect) > 0
            return isect
        else
            treelen += 1
            return union(states...)
        end
    end
    function walk(n)
        if isleaf(tree, n)
            return Set.(aln[leafmap[n], :])
        else
            child_states = Array{Set{Int64}}[]
            for c in childnodes(tree, n); push!(child_states, walk(c)); end
            return [fitchfun([child_states[j][i] for j in 1:length(child_states)])
                for i in 1:L]
        end
    end
    walk(findroots(tree)[1])
    return treelen
end

# initialize a tree of nleaves leaves
function inittree(nleaves::Int64)
    tree = Tree()
    nodes = collect(1:nleaves)
    for n in nodes; addnode!(tree); end
    while length(nodes) > 1
        a, b = nodes[1:2]
        addnode!(tree)
        c = length(tree.nodes)
        push!(nodes, c)
        addbranch!(tree, c, a, 1.)
        addbranch!(tree, c, b, 1.)
        nodes = nodes[3:end]
    end
    return tree
end

# initialize the leafmap
initleafmap(tree::Tree) = Dict(x=>i for (i,x) in enumerate(findleaves(tree)))
lastnode(tree::Tree) = maximum(keys(tree.nodes))

# return the two trees defined by the NNI neighborhood
function nni(tree, leafmap, node)
    a, b = childnodes(tree, node)
    c, d = childnodes(tree, a)
    e, f = childnodes(tree, b)
    # now get ((c, e),(d, f)) and ((c, f),(d, e))
    # delete the branch and its source and target, if the source is the root, delete the other
    # branch stemming from the root as well
    trees = [do_nni(deepcopy(tree), node, a, b, c, d, e, f)]
    trees = [trees ; do_nni(deepcopy(tree), node, a, b, c, f, e, d)]
end

# i don't think this is a correct NNI
function do_nni(tree, node, a, b, c, d, e, f)
    deletebranch!(tree, tree.nodes[c].in[1])
    deletebranch!(tree, tree.nodes[d].in[1])
    # delete branches b - e, b - f
    deletebranch!(tree, tree.nodes[e].in[1])
    deletebranch!(tree, tree.nodes[f].in[1])
    # create branches a - c, a - e
    addbranch!(tree, a, c, 1.)
    addbranch!(tree, a, e, 1.)
    # create branches b - d, b - f
    addbranch!(tree, b, d, 1.)
    addbranch!(tree, b, f, 1.)
    return tree
end

function spr(tree, node1, node2)
    t = deepcopy(tree)
    node3 = parentnode(t, node2)
    node5 = parentnode(t, node1)
    deletebranch!(t, t.nodes[node1].in[1])
    deletebranch!(t, t.nodes[node2].in[1])
    addnode!(t); node4 = lastnode(t)
    addbranch!(t, node3, node4, 1.)
    addbranch!(t, node4, node1, 1.)
    addbranch!(t, node4, node2, 1.)
    if !isroot(t, node5) && outdegree(t, node5) == 1 # if we moved a leaf
        node6 = parentnode(t, node5)
        child = childnodes(t, node5)[1]
        deletebranch!(t, t.nodes[node5].out[1])
        deletebranch!(t, t.nodes[node5].in[1])
        deletenode!(t, node5)
        addbranch!(t, node6, child, 1.)
    end
    cleanspr!(t)
    return t
end

function cleanspr!(tree)
    root = findroots(tree)[1]
    if outdegree(tree, root) != 2
        deletebranch!(tree, tree.nodes[root].out[1])
        deletenode!(tree, root)
    end
end

function randspr(tree)
    node1 = rand(setdiff(keys(tree.nodes), findroots(tree)[1]))
    node2 = rand(setdiff(keys(tree.nodes), descendantnodes(tree, node1),
        findroots(tree)[1], node1))
    return spr(tree, node1, node2)
end

function randspr(tree, node1)
    node2 = rand(setdiff(keys(tree.nodes), descendantnodes(tree, node1),
        findroots(tree)[1], node1))
    return spr(tree, node1, node2)
end

# ALE-like summary =================================================================================
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

function getspecies(S::SpeciesTree, n::Int64)
    return join([haskey(S.ambiguous, x) ? S.ambiguous[x] :
        S.species[x] for x in S.clades[n]], ",")
end

# WGD specific summary =============================================================================
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
#=
"""
    wgdsum(nrtrees::Array{RecTree}, S::SpeciesTree)
Summarize retained WGD events for every gene family.
"""
function wgdsum(rt::Array{RecTree}, S::SpeciesTree; fname="")
    columns = ["family", "wgd", "wgd_id", "gnode", "gleft", "gright",
        "sleft", "sright", "retained", "total"]
    types = [String[], Int64[], String[], String[], String[], String[]
        String[], String[], Float64[], Int64[]]
    df = DataFrame(columns[i]=>types[i] for i = 1:length(columns))
    for n in keys(S.wgd_index)
        wgdsum(t, S)
            push!(data, [gf ; x])
    end
    return data
end

function wgdsum(rt::Array{RecTree}, wgdnode::Int64)
    d = Any[]
    for t in rt
        nodes = [k for (k,v) in t.σ if (v == wgdnode && length(childnodes(t.tree, k)) == 2)]
        for n in nodes
        end
    end
end

# function for summarizing WGDs in samples of backtracked trees, uses hashes to count
function summarize_wgds(rt::RecTree, S::SpeciesTree)
    d = Dict{UInt64,Array}()
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
    return d
end
=#

# Subgenome assignment =============================================================================
function sumambiguous(rt::Array{RecTree}, S::SpeciesTree, ccd::CCD)
    length(S.ambiguous) == 0 ? (return) : nothing
    ambgenes = [ccd.leaves[k] for (k, v) in ccd.m3 if haskey(S.ambiguous, v)]
    data = Dict{String,Dict}(g=>Dict() for g in ambgenes)
    N = length(rt)
    for t in rt
        leaf2node = Dict(v=>k for (k,v) in t.leaves)
        ambnodes = [leaf2node[g] for g in ambgenes]
        for (g, n) in zip(ambgenes, ambnodes)
            sp = S.species[t.σ[n]]
            haskey(data[g], sp) ? data[g][sp] += 1/N : data[g][sp] = 1/N
        end
    end
    return data
end
