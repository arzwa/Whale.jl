
const Index = Dict{Int64,Int64}

"""
    SlicedTree

Sliced species tree with WGD nodes.
"""
struct SlicedTree <: Arboreal
    tree::Tree
    qindex::Index
    rindex::Index
    leaves::Dict{Int64,String}
    clades::Dict{Int64,Set{Int64}}
    slices::Dict{Int64,Array{Float64,1}}
    border::Array{Int64,1}  # postorder of species tree branches
end

"""
    SlicedTree(tree::Arboreal, wgdconf=Dict(), Δt=0.05, minn=5, maxn=Inf)

Get a SlicedTree from a given tree and a configuration file (read into a dict).
"""
function SlicedTree(tree::Arboreal, wgdconf=Dict(), Δt=0.05, minn=5, maxn=Inf)
    tree = deepcopy(tree)
    qindex, nindex = add_wgds!(tree, wgdconf)
    rindex = getrindex(tree.tree, qindex)
    border = postorder(tree.tree)
    clades = getclades(tree.tree)
    slices = getslices(tree.tree, Δt, minn, maxn)
    SlicedTree(tree.tree, qindex, rindex, tree.leaves, clades, slices, border)
end

function SlicedTree(treefile::String, wgdconf=Dict(), Δt=0.05, minn=5, maxn=Inf)
    SlicedTree(readtree(treefile))
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
    nindex = Dict{Int64,String}()
    for (i, (wgd, tup)) in enumerate(conf)
        n = insert_wgd!(T, [string(x) for x in split(tup[1], ",")], tup[2])
        qindex[n] = i
        nindex[n] = wgd
    end
    return qindex, nindex
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

function get_parentbranches(s::SlicedTree, node::Int64)
    branches = Int64[]
    n = node
    while n != 1
        push!(branches, n)
        n = parentnode(s.tree, n)
    end
    return [branches; [1]]
end

function set_constantrates!(s::SlicedTree)
    for (k, v) in s.rindex
        s.rindex[k] = 1
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
