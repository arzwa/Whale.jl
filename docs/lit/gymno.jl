using Pkg; Pkg.activate(@__DIR__)
using Whale, NewickTree, Parameters, Turing, Distributions
using Test, Random, Distributed
using Plots, Measures

bdir = "/home/arzwa/research/gymno-seed/"
data = joinpath("$bdir/genetrees/orthogroups-100-cds-aln-trimal-ale")
tree = readnw(readline("$bdir/Orthogroups/tree.nw"))

n = length(postwalk(tree))
r = ConstantDLWGD(λ=0.1, μ=0.2, η=0.6)
w = WhaleModel(r, tree, 0.01)
ccd = read_ale(data, w)

l = logpdf!(w, ccd[1])

G = Whale.backtrack(w, ccd[1])

plot(w, G, sscale=8.)
