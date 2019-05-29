
"""
    $(TYPEDEF)

Sliced species tree with WGD nodes.
"""
struct SlicedTree <: Arboreal
    tree::Tree
    qindex::Dict{Int64,Int64}
    rindex::Dict{Int64,Int64}
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
    qindex = add_wgds!(tree, conf["wgd"])
    rindex = getrindex(tree.tree, qindex)
    border = postorder(tree.tree)
    clades = getclades(tree.tree)
    slices = getslices(tree.tree, conf["slices"])
    SlicedTree(tree.tree, qindex, rindex, tree.leaves, clades, slices, border)
end

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
    qindex = Dict{Int64,Int64}()
    for (i, (wgd, tup)) in enumerate(conf)
        n = insert_wgd!(T, [string(x) for x in split(tup[1], ",")], tup[2])
        qindex[n] = i
    end
    return qindex
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
function getrindex(tree::Tree, qindex)
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
