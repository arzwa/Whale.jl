#= Posterior reconciled tree summary
I believe this would be the most interesting strategy:
- Sample reconciled trees from MAP parameter values and compute consensus
- Compute for every event (dup/loss/wgd/speciation) in the MAP tree the posterior
  probability based on a sample of (consensus) reconciled trees from the posterior

Alternatively, the consensus tree could be computed from the full poterior?

The best way to compute a consensus tree is probably by means of MRP

TODO: I think I have the matrix representation, but should I really be implementing
parimony tree search here?
=#

mutable struct MatRep
    leafmap::Dict{String,Int64}
    matrix::Array{Int64,2}

    MatRep() = new(Dict{String,Int64}(), zeros(0,0))
end

function updateleafmap!(mr::MatRep, leafmap::Dict{Int64,String})
    for (k,v) in leafmap
        if !haskey(mr.leafmap, v)
            mr.leafmap[v] = length(mr.leafmap) + 1
            mr.matrix = [mr.matrix ; zeros(1, size(mr.matrix)[2])]
        end
    end
end

function mrpencode(trees::Array{RecTree})
    mr = MatRep()
    for t in trees; mrpencode!(mr, t) ; end
    return mr
end

function mrpencode!(mr::MatRep, tree::RecTree)
    leafmap = getleafmap(tree)
    leaves = keys(leafmap)
    intern = setdiff(collect(keys(tree.tree.nodes)), collect(leaves))
    updateleafmap!(mr, leafmap)
    matrix = zeros(Int64, size(mr.matrix)[1], length(intern))
    for i in leaves
        for (j, n) in enumerate(intern)
            i in descendantnodes(tree.tree, n) ? matrix[mr.leafmap[leafmap[i]], j] = 1 : nothing
        end
    end
    mr.matrix = [mr.matrix matrix]
end

function getleafmap(tree::RecTree)
    leafmap = Dict{Int64,String}()
    for n in findleaves(tree.tree)
        id = tree.labels[n] == "loss" ? "loss-$(tree.σ[n])" : tree.leaves[n]
        leafmap[n] = id
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

function Base.write(io::IO, mr::MatRep)
    write(io, join(size(mr.matrix), " ") * "\n")
    for (k, v) in mr.leafmap
        write(io, (@sprintf "%-10d" v) * join(mr.matrix[v, :]) * "\n")
    end
end
