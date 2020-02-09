"""
    sumtrees(trees)

Summarize backtracked reconciled trees.
"""
sumtrees(trees::AbstractMatrix) = map(sumtrees, eachcol(trees))

function sumtrees(trees::AbstractVector{RecNode{I}}) where I
    N = length(trees)
    hashes = hash.(trees)
    counts = countmap(hashes)
    summary = NamedTuple[]
    for (h, count) in sort(collect(counts), by=x->x[2], rev=true)
        tree = trees[findfirst(x->x==h, hashes)]
        push!(summary, (freq=count/N, tree=tree))
    end
    summary
end

function cladecredibility(tree, trees)
    N = length(trees)
    clades = getclades(trees)
    counts = countmap(clades)
    Dict{UInt64,Float64}(cladehash(n)=>counts[cladehash(n)]
        for n in postwalk(tree) if !isleaf(n))
end

getclades(trees) = vcat(map((t)->cladehash.(postwalk(t)), trees)...)
