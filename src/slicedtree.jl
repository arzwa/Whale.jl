
const Index = Dict{Int64,Int64}

"""
    SlicedTree(tree::Arboreal, wgdconf=Dict(), Δt=0.05, minn=5, maxn=Inf)

Sliced species tree with optional WGD nodes. WGDs are specified using a configuration dict such as for example:

```julia
wgdconf = Dict(
    "PPAT" => ("PPAT", 0.655),
    "CPAP" => ("CPAP", 0.275),
    "BETA" => ("ATHA", 0.55),
    "ANGI" => ("ATRI,ATHA", 3.08),
    "SEED" => ("GBIL,ATHA", 3.9),
    "MONO" => ("OSAT", 0.91),
    "ALPH" => ("ATHA", 0.501))
```

where the keys are WGD IDs (names), and values are given as a tuple or array with as first entry a single taxon or pair of taxa (comma-separated) that specifies the last common ancestor node that is preceded by the WGD of interest and the second entry the estimate time (before present) of the WGD event. `Δt` is the desired slice length (time interval), `minn` the minimum number of slices a branch should have, and `maxn` the maximum number of slices a branch should have.
"""
struct SlicedTree <: Arboreal
    tree::Tree
    qindex::Index
    rindex::Index
    leaves::Dict{Int64,String}
    clades::Dict{Int64,Set{Int64}}
    slices::Dict{Int64,Array{Float64,1}}
    border::Array{Int64,1}  # postorder of species tree branches
    windex::Dict{String,Int64}
end

function SlicedTree(tree::Arboreal, wgdconf=Dict(), Δt=0.05, minn=5, maxn=Inf)
    tree = deepcopy(tree)
    qidx, widx = add_wgds!(tree, wgdconf)
    ridx = getrindex(tree.tree, qidx)
    ordr = postorder(tree.tree)
    clds = getclades(tree.tree)
    slcs = getslices(tree.tree, Δt, minn, maxn)
    SlicedTree(tree.tree, qidx, ridx, tree.leaves, clds, slcs, ordr, widx)
end

function SlicedTree(treefile::String, wgdconf=Dict(), Δt=0.05, minn=5, maxn=Inf)
    SlicedTree(readtree(treefile), wgdconf, Δt, minn, maxn)
end

Base.show(io::IO, s::SlicedTree) = write(io,
    "SlicedTree($(ntaxa(s)), $(nrates(s)), $(nwgd(s)))")
Base.getindex(s::SlicedTree, e::Int64, i::Int64) = s.slices[e][i]
Base.getindex(s::SlicedTree, e::Int64) = s.slices[e]
Base.getindex(s::SlicedTree, e::Int64, q::Symbol) =
    q == :q ? s.qindex[e] : s.rindex[e]

Base.lastindex(s::SlicedTree, e::Int64) = length(s[e])

Base.setindex!(s::SlicedTree, x::Float64, e::Int64, i::Int64) =
    s.slices[e][i] = x

nslices(s::SlicedTree, e::Int64) = length(s[e])
nrates(s::SlicedTree) = length(Set(values(s.rindex)))
nwgd(s::SlicedTree) = length(s.qindex)
ntaxa(s::SlicedTree) = length(s.leaves)

function getslices(T::Tree, Δt::Real, minn::Int64=5, maxn::Number=Inf)
    slices = Dict{Int64,Array{Float64}}()
    function walk(node)
        if !isroot(T, node)
            l = parentdist(T, node)
            n = Int64(min(max(ceil(Int64, l / Δt), minn), maxn))
            slices[node] = [[0] ; repeat([l/n], n)]
            @assert isapprox(sum(slices[node]), l, atol=0.00001)
        end
        isleaf(T, node) ? (return) : [walk(c) for c in childnodes(T, node)]
    end
    walk(findroots(T)[1])
    slices[1] = [0.]
    return slices
end

getslices(T, d::Dict) = getslices(T, d["length"], d["min"], d["max"])

function add_wgds!(T::Arboreal, conf::Dict)
    qindex = Index()
    nindex = Dict{String,Int64}()
    for (i, (wgd, tup)) in enumerate(conf)
        n = insert_wgd!(T, [string(x) for x in split(tup[1], ",")], tup[2])
        qindex[n] = i
        nindex[wgd] = n
    end
    return qindex, nindex
end

wgds(S::SlicedTree) = wgds(stdout, S)

function wgds(io::IO, S::SlicedTree)
    for (k, v) in S.windex
        write(io, "$k → node $v → q$(S.qindex[v])\n")
    end
end

function insert_wgd!(T::Arboreal, lca::Array{String}, t::Number)
    n = lca_node(T, lca)
    while leafdist(T, parentnode(T, n)) < t
        n = parentnode(T, n)
    end
    tn = leafdist(T, n)
    tbefore = t - tn
    wgd_node = insert_node!(T.tree, n, tbefore)
    return wgd_node
end

# Get the default rate index
function getrindex(tree::Tree, qindex::Index)
    rindex = Dict{Int64,Int64}()
    i = 1
    for n in sort(collect(keys(tree.nodes)))
        if haskey(qindex, n)
            rindex[n] = rindex[non_wgd_child(tree, n)]
        else
            rindex[n] = i
            i += 1
        end
    end
    return rindex
end

function non_wgd_child(tree, n)
    while outdegree(tree, n) == 1
        n = childnodes(tree, n)[1]
    end
    return n
end

function non_wgd_children(s::SlicedTree, node::Int64)
    children = []
    for c in childnodes(s.tree, node)
        haskey(s.qindex, c) ?
            push!(children, non_wgd_child(s.tree, c)) : push!(children, c)
    end
    return children
end

function non_wgd_parent(s::SlicedTree, n::Int64)
    n == findroot(s) ? (return findroot(s)) : nothing
    x = parentnode(s.tree, n)
    while haskey(s.qindex, x)
        x = parentnode(s.tree, x)
    end
    return x
end

function wgd_branchchildren(s::SlicedTree, n::Int64)
    if !haskey(s.qindex, n) ; return Int64[] ; end
    nodes = Int64[n]
    r = s.rindex[n]
    while length(childnodes(s, n)) == 1
        n = childnodes(s, n)[1]
        push!(nodes, n)
    end
    return nodes
end

function wgdbranches(s::SlicedTree)
    x = Tuple{Int64,Int64}[]
    for i in keys(s.qindex)
        push!(x, (i, non_wgd_child(s.tree, i)))
    end
    return x
end

function get_parentbranches(s::SlicedTree, node::Int64)
    branches = Int64[]
    n = node
    root = findroot(s)
    while n != root
        push!(branches, n)
        n = parentnode(s.tree, n)
    end
    return [branches; [root]]
end

function branches_to_recompute(s::SlicedTree, node::Int64)
    # order matters!
    branches = get_parentbranches(s, node)
    exclude = wgd_branchchildren(s, node)
    for (k, v) in s.rindex
        if k in branches || k in exclude
            continue
        elseif v == s.rindex[node]
            branches = [branches ; get_parentbranches(s, k)]
        end
    end
    branches
end

function set_constantrates!(s::SlicedTree)
    for (k, v) in s.rindex
        s.rindex[k] = 1
    end
end

function set_equalrootrates!(s::SlicedTree)
    root = findroot(s)
    a, b = childnodes(s, root)
    c = s.rindex[b]
    for (k, v) in s.rindex
        if v == c
            s.rindex[k] = s.rindex[a]
        elseif v > c && v != s.rindex[a]
            s.rindex[k] -= 1
        end
    end
end

function example_tree()
    tree = LabeledTree(read_nw("((MPOL:4.752,PPAT:4.752):0.292,(SMOE:4.457,(((OSAT:1.555,(ATHA:0.5548,CPAP:0.5548):1.0002):0.738,ATRI:2.293):1.225,(GBIL:3.178,PABI:3.178):0.34):0.939):0.587);")[1:2]...)
    wgdconf = Dict(
        "PPAT" => ("PPAT", 0.655, -1.0),
        "CPAP" => ("CPAP", 0.275, -1.0),
        "BETA" => ("ATHA", 0.55, -1.0),
        "ANGI" => ("ATRI,ATHA", 3.08, -1.0),
        "SEED" => ("GBIL,ATHA", 3.9, -1.0),
        "MONO" => ("OSAT", 0.91, -1.0),
        "ALPH" => ("ATHA", 0.501, -1.0))
    SlicedTree(tree, wgdconf)
end
