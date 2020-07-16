# # Reconciled tree inference example
using Whale, DynamicHMC, DynamicHMC.Diagnostics, Random, NewickTree, DataFrames
Random.seed!(624)

# This is a use case I haven't been exploring before, namely large gene
# families. Here we consider a family of about 100 leaves.

base  = joinpath(@__DIR__, "../../example/example-4")
tree  = readnw(readline(joinpath(base, "tree.nw")))
model = WhaleModel(RatesModel(ConstantDLWGD(λ=0.1, μ=0.2, η=0.9)), tree, .1)
data  = read_ale(joinpath(base, "cytp450.ale"), model, true)

# Reading in the single CCD is already a very heavy operation. The CCD has about
# 5000 unique clades.

# We will use the DynamicHMC interface, using a constant-rates model (i.e. a single
# duplication and loss rate for the entire tree).
prior = Whale.CRPrior()
problem = WhaleProblem(data, model, prior)

# Now run the actual HMC sampler
results = mcmc_with_warmup(Random.GLOBAL_RNG, problem, 500,
    warmup_stages=DynamicHMC.default_warmup_stages(doubling_stages=2))
posterior = Whale.transform(problem, results.chain)
@info summarize_tree_statistics(results.tree_statistics)

# We can get a data frame
df = Whale.unpack(posterior)
describe(df)

# Get reconciled trees from the posterior
# rectrees = track(problem, posterior[1:10])
trees = track(problem, posterior)
trees[1].trees

# Note that there are many trees with similar posterior probability!

# Now we will plot the MAP (maximum a posteriori) reconciled tree, showing duplication and loss events
using PalmTree, Luxor
import Luxor: RGB

rectree = trees[1].trees[1].tree
outdir  = mkpath(joinpath(@__DIR__, "../assets/"))
outpath = joinpath(outdir, "cytp450-map.svg")

tl = TreeLayout(rectree, cladogram=true, dims=(400,800))
@svg begin
    Luxor.origin(Point(0,20))
    Luxor.setline(1)
    setfont("Noto sans italic", 7)
    colfun = n->n.data.label != "loss" ? RGB() : RGB(0.99,0.99,0.99)
    drawtree(tl, color=colfun)
    nodemap(tl, prewalk(rectree),
        (n, p) -> !isleaf(n) ?
            settext("  $(n.data.cred)", p, valign="center") :
            settext("  $(n.data.name)", p, valign="center"))
    nodemap(tl, prewalk(rectree),
        (n, p) -> n.data.label == "duplication" && box(p, 4, 4, :fill))
end 500 850 outpath

# Squares show duplication events and internal node labels show the posterior probability of observing the relevant node in the reconciled tree.
