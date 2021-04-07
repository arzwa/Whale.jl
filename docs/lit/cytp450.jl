# # Reconciled tree inference example
using Whale, DynamicHMC, DynamicHMC.Diagnostics, Random, NewickTree, DataFrames
Random.seed!(624);

# In this case study, we will perform Bayesian gene tree reconciliation for a
# single (large) gene family. The data can be found in the `example4` directory
# in the `Whale` git repository. We first load the data:
base  = joinpath(@__DIR__, "../../example/example-4")
tree  = readnw(readline(joinpath(base, "tree.nw")))
model = WhaleModel(RatesModel(ConstantDLWGD(λ=0.1, μ=0.2, η=0.9)), tree, .1)
data  = read_ale(joinpath(base, "cytp450.ale"), model, true)

# Reading in the single CCD (in the `read_ale` step is already a rather heavy
# operation. The CCD has about 5000 unique clades.

# For Bayesian inference we will use the DynamicHMC interface, using a
# constant-rates model (i.e. assuming a single duplication and loss rate for
# the entire tree). The default prior should of course not be chosen lightly,
# although for our current purposes it is reasonable:
prior = Whale.CRPrior()
problem = WhaleProblem(data, model, prior)

# Now we run the actual HMC sampler. Note that we perform a very short run here
# to reduce build times of the documentation, in reality you'd rather use
# something like a 1000 iterations.
results = mcmc_with_warmup(Random.GLOBAL_RNG, problem, 100,
    warmup_stages=DynamicHMC.default_warmup_stages(doubling_stages=2))
posterior = Whale.transform(problem, results.chain)
@info summarize_tree_statistics(results.tree_statistics)

# A data frame may be easier to work with (and save to disk)
df = Whale.unpack(posterior)
describe(df, :mean, :q025=>x->quantile(x, 0.025), :q975=>x->quantile(x, 0.975))

# Now we will obtain reconciled trees from the posterior
trees = track(problem, posterior)
trees[1].trees

# Note that there are many trees with similar posterior probability, so in
# other words the maximum a posteriori (MAP) tree is not that meaningful in
# itself. We can however plot the MAP tree with posterior node probabilities to
# get an idea of the reconciled tree and the nodes with considerable posterior
# uncertainty. I will use `Luxor.jl` together with my small helper library for
# plotting trees:

# ```julia
# using PalmTree, Luxor
# import Luxor: RGB
# 
# rectree = trees[1].trees[1].tree
# outdir  = mkpath(joinpath(@__DIR__, "../assets/"))
# outpath = joinpath(outdir, "cytp450-map.svg")
# 
# tl = TreeLayout(rectree, cladogram=true, dims=(400,800))
# @svg begin
#     Luxor.origin(Point(0,20))
#     Luxor.setline(1)
#     setfont("Noto sans italic", 7)
#     colfun = n->n.data.label != "loss" ? RGB() : RGB(0.99,0.99,0.99)
#     drawtree(tl, color=colfun)
#     nodemap(tl, prewalk(rectree),
#         (n, p) -> !isleaf(n) ?
#             settext("  $(n.data.cred)", p, valign="center") :
#             settext("  $(n.data.name)", p, valign="center"))
#     nodemap(tl, prewalk(rectree),
#         (n, p) -> n.data.label == "duplication" && box(p, 4, 4, :fill))
# end 500 850 outpath
# ```

# Squares show duplication events and internal node labels show the posterior
# probability of observing the relevant node in the reconciled tree.
