# # Bayesian inference using NUTS with `DynamicHMC.jl`

# In this example case, the basic workflow for assessing WGD hypotheses using
# Whale will be illustrated. We will use the
# [`DynamicHMC`](https://github.com/tpapp/DynamicHMC.jl) library for Bayesian
# inference.

# !!! note
#     Inference in Whale with DynamicHMC.jl supports distributed computing. To
#     use distributed parallelism, start up julia with `-p <ncores>` (or do
#     `using Distributed; addprocs(ncores)`. Instead of loading Whale with
#     `using Whale`, use `@everywhere using Whale` and normally all
#     log-likelihood computations should now run in parallel.

using Whale, DynamicHMC, Random, NewickTree, Distributions, DataFrames
using DynamicHMC.Diagnostics
Random.seed!(562);

# Set up the model and the data, here I will use a model with constant
# duplication and loss rates across the species tree. Note that the tree
# contains two WGD events (internal nodes labeled with a name starting with
# `wgd`).
tree  = readnw("((MPOL:4.752,(PPAT:2.752)wgd_1:2.0):0.292,(SMOE:4.457,((((OSAT:1.555,(ATHA:0.5548,CPAP:0.5548):1.0002):0.738,ATRI:2.293):1.0)wgd_2:0.225,(GBIL:3.178,PABI:3.178):0.34):0.939):0.587);")
n = length(postwalk(tree))
ntaxa = (n+1)÷2
rates = RatesModel(ConstantDLWGD(λ=0.1, μ=0.1, q=[0.2, 0.3], η=0.9))
model = WhaleModel(rates, tree, 0.1)
data  = read_ale(joinpath(@__DIR__, "../../example/example-1/ale"), model, true)

# !!! note
#     To use the `DynamicHMC` interface, the third argument of `read_ale`
#     should be set to `true`.

# And next we set up the Bayesian inference 'problem', using the default priors:
prior = Whale.CRPrior()
problem = WhaleProblem(data, model, prior)

# Now we run NUTS (of course this is a ridicuously short run, and in reality
# you want to use something like a 1000 iterations. Also, it's better to keep
# `doubling_stages` >= 3).
results = mcmc_with_warmup(Random.GLOBAL_RNG, problem, 200,
    warmup_stages=DynamicHMC.default_warmup_stages(doubling_stages=2))
summarize_tree_statistics(results.tree_statistics)

# Now we obtain the posterior distribution in the form of a data frame
posterior = Whale.transform(problem, results.chain)
df = Whale.unpack(posterior)
describe(df, :mean, :q025=>x->quantile(x, 0.025), :q975=>x->quantile(x, 0.975))

# And we can visualize the marginal posterior distributions:
using Plots
kwargs = (bins=20, color=:white, grid=false, legend=false)
ps1 = [histogram(df[!,x], xlabel=x; kwargs...) for x in names(df)]
kwargs = (color=:black, grid=false, legend=false)
ps2 = [plot(df[!,x], xlabel=x; kwargs...) for x in names(df)]
plot(ps1..., ps2..., layout=(2,5), size=(900,300), guidefont=font(8))

# From these results (NB: which are based on a mere 12 gene families), we find
# little support for the second genome duplication (in the angiosperm branch),
# i.e. the retention rate `q_2` is not markedly different from 0. The WGD on
# the *P. patens* tip branch however seems to gain some support, with a
# posterior mean retention rate (`q_1`) of about 0.4, which is quite high.
# However, this is definitely too small a data set to make substantial
# conclusions!

# We can obtain reconciled trees sampled from the posterior
trees = track(problem, posterior)

# Consider the first gene family
family1 = trees[1].trees

# Note that the `freq` field gives the approximate posterior probability of
# this tree (estimated by the sample frequency). We can get the MAP tree as a
# newick string
nwstr(family1[1].tree)

# Now we'll plot the MAP tree
# ```julia
# using PalmTree, Luxor
# import Luxor: RGB
# 
# outdir  = mkpath(joinpath(@__DIR__, "../assets/"))
# outpath = joinpath(outdir, "dhmc-fam1-map.svg")
# 
# rectree = family1[1].tree
# tl = TreeLayout(rectree, cladogram=true, dims=(300,300))
# @svg begin
#     Luxor.origin(Point(0,20))
#     Luxor.setline(2)
#     setfont("Noto sans italic", 12)
#     colfun = n->n.data.label != "loss" ? RGB() : RGB(0.99,0.99,0.99)
#     drawtree(tl, color=colfun)
#     nodemap(tl, prewalk(rectree),
#         (n, p) -> !isleaf(n) ?
#             settext("  $(n.data.cred)", p, valign="center") :
#             settext("  $(split(n.data.name, "_")[1])", p, valign="center"))
#     nodemap(tl, prewalk(rectree),
#         (n, p) -> n.data.label == "duplication" && box(p, 8, 8, :fill))
#     nodemap(tl, prewalk(rectree),
#         (n, p) -> startswith(n.data.label, "wgd") && star(p,3,5,3,0.5,:fill))
# end 500 400 outpath
# ```

# The support values are posterior probabilities for the associated reconciled
# split. Note that the tree does not contain branch lengths. Duplication events
# are marked by squares, whereas the WGDs are marked by stars.

# The events field for each gene family contains a summary of the expected
# number of events for each branch (where each branch is identified by the node
# to which the branch leads, as shown in the `node` column)
trees[1].events

# We can get for every gene pair the posterior reconciliation probability. The
# following data frame can therefore be used to probabilistically assess
# whether two homologous genes are orthologs, WGD-derived paralogs or non-WGD
# derived paralogs.
pair_pps = Whale.getpairs(trees, model)
first(pair_pps, 5)

# Every row of this data frame is a probability distribution over
# reconciliation events, so each row sums to one, as we can verify:
map(sum, eachrow(pair_pps[!,1:end-2]))

# Take for instance the following gene pair (second row)
x = pair_pps[2,1:end-2]
for (n, v) in zip(names(x), Array(x))
    v > 0 && println(n, ": ", v)
end

# The posterior probability (under the DL model) that this gene pair traces
# back to the speciation corresponding to node 17 (i.e. the root) is
# approximately 0.86, whereas the posterior probability that this gene pair
# traces back to an ancestral duplication event is 0.14.

# We can get a WGD-centric view as well. The following retrieves a table for
# each WGD with all gene tree nodes that have a non-zero posterior probability
# of being reconciled to that particular WGD node
tables = Whale.getwgdtables(trees, data, model)
tables
