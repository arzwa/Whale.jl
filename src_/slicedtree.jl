
const Index = Dict{Int64,Int64}

"""
    $(TYPEDEF)

Sliced species tree with WGD nodes.
"""
struct SlicedTree <: Arboreal
    tree::Tree
    qindex::Index
    rindex::Index
    leaves::Dict{Int64,String}
    clades::Dict{Int64,Set{Int64}}
    slices::Dict{Int64,Array{Float64,1}}
    border::Array{Int64}  # postorder of species tree branches
end

"""
    $(TYPEDSIGNATURES)

Get a SlicedTree from a given tree and a configuration file (read into a dict).
"""
function SlicedTree(tree::Arboreal, conf::Dict)
    tree = deepcopy(tree)
    qindex, nindex = add_wgds!(tree, conf["wgd"])
    rindex = getrindex(tree.tree, qindex)
    border = postorder(tree.tree)
    clades = getclades(tree.tree)
    slices = getslices(tree.tree, conf["slices"])
    SlicedTree(tree.tree, qindex, rindex, tree.leaves, clades, slices, border)
end

nslices(s::SlicedTree, e::Int64) = length(s.slices[e])
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

function add_wgds!(T::Arboreal, conf::Dict{String,Any})
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
